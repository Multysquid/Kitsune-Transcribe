"""The canonical ingest sequence and the extent record: the one definition the label box and the A100 rebuild share.

Utterance ids depend on the order the sources are ingested in (scripts/01_prepare_data.py): reazon_large skips the
rows already in reazon_small, eval_emilia leaves out the videos already in emilia_yodas, and emilia_yodas those already
in eval_emilia. The laptop ran Emilia as 300 h, then eval_emilia, then the rest, so its labelled emilia_yodas stems and
its eval_emilia hold-out come out again only if that chain is replayed: a single `--emilia-hours inf` call keeps clips
the laptop dropped and picks another hold-out. CANONICAL is that order, and `01 --extent-config CFG` runs it
(plan_steps), so the label box and a later A100 rebuild produce the same ids AND the same shard stems: stems are
numbered from the manifest, every input is flushed on its own, and a capped source reads a sorted prefix of its upstream
files, so a prefix of the inputs gives a prefix of the full run's stems.

A run config's extent block names the label set and optionally caps sources:
    "extent": {"name": "full", "root": "labels/full", "inputs": {"reazon_large": 216, "galgame": 16}}
`inputs` takes the first N of a CAPPABLE source's sorted upstream files. emilia_yodas takes "300h" (the 300 h step only)
or an int N >= EMILIA_300H_INPUTS (the 300 h step, then the rest over the first N tars; N = 9 continues the tar the
300 h step cut). `root` (labels/<name>) is the common parent of the config's teacher_root, second_root, parakeet_root
and selection. A subset config is the same extent with smaller caps (within): its label files are a subset of the full
run's.

Files:
    <data>/extent_progress.json  01 writes it after every step: {canonical_version, config, extent, steps: [[key, cap]],
                                 completed: [key], repos: {source: repo}, revisions: {repo: sha}, tools}
    <data>/pruned.jsonl          the label box appends {"path", "bytes"} for a shard BEFORE it deletes the shard's
                                 audio (its id sidecar stays), so the record still knows the shard's size
    <root>/extent.json           build_record, at the label box's finalize: every upstream input of every ingested
                                 source with the stems it produced (split, step, rows, hours, ids_sha256, shard_bytes)
The A100 reads extent.json to pull only a subset's label files (pull_plan), to size its disk and rebuild timeout
(sizing) and to check that its rebuilt stems are the labelled ones (the sidecars' ids_sha256 against the record's).

Pure Python (stdlib + kitsune.store); no torch.
"""
import json
import math
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kitsune.prereg import study_files
from kitsune.store import fsync_path, load_progress, read_manifest, sidecar_meta
from kitsune.store import ids_sha256  # noqa: F401  re-exported: the record's per-stem join key

CANONICAL_VERSION = 1  # bump when CANONICAL changes: a record from another sequence has other stems
RECORD_SCHEMA = 1
GATE_SETS = ("eval_jsut", "eval_cv8", "eval_reazon")
# sorted upstream files at the pinned revisions (01's REVISIONS); emilia_yodas without the eval_emilia tar JA-B000029
UPSTREAM_INPUTS = {"reazon_small": 15, "reazon_large": 705, "galgame": 115, "emilia_yodas": 29, "emilia_nc": 66}
# reazon_small stays whole (reazon_large's dedup set) and so do the eval sets (fixed hold-outs)
CAPPABLE = ("reazon_large", "galgame", "emilia_yodas", "emilia_nc")
EMILIA_300H_INPUTS = 9  # the 300 h step reads JA-B000000..B000008 and cuts inside tar 8
PROGRESS_FILE = "extent_progress.json"  # in the data root
PRUNED_FILE = "pruned.jsonl"  # in the data root
RECORD_FILE = "extent.json"  # in the extent root


@dataclass(frozen=True)
class Step:
    key: str  # step id recorded in the sidecars and extent.json
    source: str  # 01 source name
    emilia_hours: float | None = None  # the step's Emilia budget


CANONICAL = (
    Step("reazon_small", "reazon_small"),  # dedup anchor of reazon_large (01 DEDUP_AGAINST)
    Step("eval_jsut", "eval_jsut"), Step("eval_cv8", "eval_cv8"), Step("eval_reazon", "eval_reazon"),
    Step("emilia_yodas@300h", "emilia_yodas", 300.0),  # reproduces laptop train-00000..00061 (the 109-row cut)
    Step("eval_emilia", "eval_emilia"),  # excludes the videos of the 300 h step only
    Step("galgame", "galgame"),  # tar 0 carries the 1000-row eval hold-out
    Step("emilia_yodas", "emilia_yodas", float("inf")),  # the rest; skips eval_emilia's videos
    Step("emilia_nc", "emilia_nc"),
    Step("reazon_large", "reazon_large"),
)
# step key -> the steps whose ids it reads (the 300 h step and eval_emilia read nothing the rest step writes)
DEPENDS = {"reazon_large": ("reazon_small",), "eval_emilia": ("emilia_yodas@300h",),
           "emilia_yodas": ("emilia_yodas@300h", "eval_emilia")}
REFUSED = {
    "reazon_medium": "no canonical step: reazon_large already holds its rows, and ingested after reazon_large it would "
                     "dedup against reazon_small only and duplicate ~1,500 h",
    "cv": "a manual download (not on the Hub), so no box can rebuild it",
}
EXTENT_KEYS = ("name", "root", "inputs")
# A100 sizing (vast/launch.py): everything on its disk but audio and labels (image, caches, train states, weights and
# 01's 30 GB free-space floor), the label files per audio-hour (teacher npz + jsonl + second opinion, laptop-measured)
# and a conservative upstream download rate for the rebuild
DISK_BASE_GB = 127
DISK_MARGIN, DISK_STEP_GB = 1.1, 50
LABEL_GB_PER_HOUR = 0.0016
REBUILD_BYTES_PER_S, REBUILD_SLACK, REBUILD_BASE_MIN = 40e6, 1.5, 30


def extent_block(cfg: dict) -> dict | None:
    """The config's extent block, or None (the viability config has none)."""
    return cfg.get("extent") or None


def names(cfg: dict) -> list[str]:
    """The config's train sources and eval sets, in order, each once (galgame is both)."""
    return list(dict.fromkeys(list(cfg.get("sources") or []) + list(cfg.get("eval_sets") or [])))


def closure(names) -> set[str]:
    """The canonical step keys `names` (source names or step keys) need, dependencies included: reazon_small for
    reazon_large, the 300 h step for eval_emilia, and both for the rest of emilia_yodas."""
    out, todo = set(), list(names)
    while todo:
        key = todo.pop()
        if key not in out:
            out.add(key)
            todo += DEPENDS.get(key, ())
    return out


def _inputs(cfg: dict) -> dict:
    return (extent_block(cfg) or {}).get("inputs") or {}


def _plan(names_: list[str], inputs: dict) -> list[tuple[Step, int | None]]:
    # emilia_yodas "300h" is the 300 h step alone: no rest step, and so no eval_emilia unless it is named itself
    wanted = ["emilia_yodas@300h" if n == "emilia_yodas" and inputs.get(n) == "300h" else n for n in names_]
    keys = closure(wanted)
    return [(s, None if s.key == "emilia_yodas@300h" else inputs.get(s.source)) for s in CANONICAL if s.key in keys]


def plan_steps(cfg: dict) -> list[tuple[Step, int | None]]:
    """The canonical steps a valid extent config needs, in canonical order, each with its input cap (None: every
    upstream file; the 300 h step is bounded by its budget)."""
    return _plan(names(cfg), _inputs(cfg))


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _str_list(v) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def _rel(p) -> PurePosixPath | None:
    """A normalized relative repo path, or None for anything else (absolute, backslashes, '..', '.', '//', a trailing
    '/', empty): consumers join these into Hub paths, so 'labels/full/' would give 'labels/full//extent.json'."""
    if not isinstance(p, str) or not p.strip("/") or p.startswith("/") or "\\" in p or ".." in p.split("/"):
        return None
    return PurePosixPath(p) if str(PurePosixPath(p)) == p else None


def _cap_problem(src: str, n) -> str | None:
    top = UPSTREAM_INPUTS[src]
    if src == "emilia_yodas":
        if n == "300h" or _is_int(n) and EMILIA_300H_INPUTS <= n <= top:
            return None
        return (f"extent.inputs.emilia_yodas is \"300h\" or an int {EMILIA_300H_INPUTS}..{top}, not {n!r}: the 300 h "
                f"step already reads tars 0-{EMILIA_300H_INPUTS - 1}, so a smaller N would not mean fewer tars")
    if _is_int(n) and 1 <= n <= top:
        return None
    why = " (tar 0 holds the galgame eval hold-out)" if src == "galgame" else ""
    return f"extent.inputs.{src} is an int 1..{top} (its upstream files){why}, not {n!r}"


def validate(cfg: dict) -> list[str]:
    """Problems with the config's extent block; [] if it is a valid extent config.

    sources and eval_sets are spelled out (the trainer's defaults are the viability run's, which the extent would not
    rebuild or pull); every name must be a canonical source (reazon_medium and cv are refused); `inputs` caps only
    CAPPABLE sources the config names, each with 1..UPSTREAM_INPUTS files (emilia_yodas: "300h" or
    >= EMILIA_300H_INPUTS); selection_recipe judges every source in full (partial_second_opinion is []); `root` is
    labels/<name> and the common parent of teacher_root, second_root, parakeet_root (if set) and selection, which all
    lie under it. All of them normalized paths."""
    ext = extent_block(cfg)
    if not isinstance(ext, dict):
        return ["the config has no extent block ({\"name\", \"root\", \"inputs\"})"]
    problems = [f"extent.{k}: unknown key (the keys are {', '.join(EXTENT_KEYS)})" for k in ext if k not in EXTENT_KEYS]
    if not isinstance(ext.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", ext["name"]):
        problems.append(f"extent.name {ext.get('name')!r} is not a plain name ([A-Za-z0-9_.-])")
    inputs = ext.get("inputs", {})
    if not isinstance(inputs, dict):
        problems.append(f"extent.inputs is a {{source: N}} object, not {inputs!r}")
        inputs = {}
    bad_lists = [k for k in ("sources", "eval_sets") if not _str_list(cfg.get(k))]
    for k in bad_lists:
        problems.append(f"{k}: missing; an extent config spells it out (the trainer's default is the viability run's)"
                        if k not in cfg else f"{k} is a list of names, not {cfg[k]!r}")
    recipe = cfg.get("selection_recipe")
    if not isinstance(recipe, dict):
        problems.append(f"selection_recipe is an object (make_selection.py --config builds the selection from it), "
                        f"not {recipe!r}")
    else:
        if not _str_list(recipe.get("filter_eval_sets")):
            problems.append(f"selection_recipe.filter_eval_sets is a list of names, not "
                            f"{recipe.get('filter_eval_sets')!r}")
        if recipe.get("partial_second_opinion"):
            problems.append(f"selection_recipe.partial_second_opinion is {recipe['partial_second_opinion']!r}, not []: "
                            f"an extent is judged in full, and pull_plan needs a second opinion for every train stem")
    if bad_lists:
        return problems  # the names below come from them
    ns = names(cfg)
    if not ns:
        problems.append("the config names no sources or eval_sets")
    known = {s.source for s in CANONICAL}
    for n in ns:
        if n in REFUSED:
            problems.append(f"{n}: refused, {REFUSED[n]}")
        elif n not in known:
            problems.append(f"{n}: not a canonical source ({', '.join(sorted(known))})")
    for src, n in inputs.items():
        if src not in CAPPABLE:
            problems.append(f"extent.inputs.{src}: not cappable (only {', '.join(CAPPABLE)}): reazon_small is "
                            f"reazon_large's dedup set and the eval sets are fixed hold-outs")
        elif src not in ns:
            problems.append(f"extent.inputs.{src}: the config names no {src} (sources / eval_sets)")
        elif why := _cap_problem(src, n):
            problems.append(why)
    root = _rel(ext.get("root"))
    if root is None:
        problems.append(f"extent.root {ext.get('root')!r} is not a normalized relative repo path like 'labels/full'")
        return problems
    if len(root.parts) != 2 or root.parts[0] != "labels":
        # the label box's upload guard is write-once under labels/ only (label_runs/ holds its overwritable logs)
        problems.append(f"extent.root {root} is not labels/<name>")
        return problems
    paths = {k: cfg.get(k, d) for k, d in (("teacher_root", "teacher_out"), ("second_root", "second_out"),
                                            ("selection", None))}
    if cfg.get("parakeet_root") is not None:
        paths["parakeet_root"] = cfg["parakeet_root"]
    under = {}
    for k, v in paths.items():
        p = _rel(v)
        if p is None:
            problems.append(f"{k} {v!r} is not a normalized relative repo path")
        elif root not in p.parents:
            problems.append(f"{k} {v!r} is not under extent.root {root}")
        else:
            under[k] = p
    if len(under) == len(paths):
        common = PurePosixPath(posixpath.commonpath([str(p) for p in under.values()]))
        if common != root:
            problems.append(f"extent.root {root} is not the common parent of {', '.join(paths)} ({common} is)")
    return problems


def _reach(names_: list[str], inputs: dict) -> dict[str, float]:
    """source -> how far into its sorted inputs the canonical plan reads it: inf for all, N for the first N, and just
    under EMILIA_300H_INPUTS for the 300 h step alone."""
    out: dict[str, float] = {}
    for step, cap in _plan(names_, inputs):
        r = EMILIA_300H_INPUTS - 0.5 if step.key == "emilia_yodas@300h" else math.inf if cap is None else cap
        out[step.source] = max(out.get(step.source, 0), r)
    return out


def _describe(r: float) -> str:
    return "every input" if r == math.inf else "the 300 h step" if r % 1 else f"the first {int(r)} inputs"


def _within(outer_names: list[str], outer: dict[str, float], inner_names: list[str], inner: dict[str, float],
            what: str) -> list[str]:
    problems = []
    for src, r in inner.items():
        if src not in outer:
            problems.append(f"{src} is not in {what}")
        elif r > outer[src]:
            problems.append(f"{src}: needs {_describe(r)}, {what} has {_describe(outer[src])}")
    # the label box labels only the extent's names: a dependency (reazon_small of reazon_large, the 300 h step of
    # eval_emilia) is rebuilt there but has no label files
    return problems + [f"{n}: {what} rebuilds it only as a dependency and has no labels for it"
                       for n in inner_names if n in outer and n not in outer_names]


def within(outer: dict, inner: dict) -> list[str]:
    """Problems if config `inner`'s extent is not inside config `outer`'s: the same root, every name inner takes
    labelled by outer (not only rebuilt as a dependency), and every source inner ingests (dependencies included)
    ingested by outer at least as far. Then inner's stems are a subset of outer's labelled stems."""
    o, i = extent_block(outer) or {}, extent_block(inner) or {}
    problems = [] if o.get("root") == i.get("root") else [f"extent.root {i.get('root')!r} is not {o.get('root')!r}"]
    return problems + _within(names(outer), _reach(names(outer), _inputs(outer)), names(inner),
                              _reach(names(inner), _inputs(inner)), f"extent {o.get('name')!r}")


def record_problems(record: dict, cfg: dict) -> list[str]:
    """Problems if the extent record cannot serve config `cfg`: another schema or canonical sequence, another root, a
    name the record's extent did not label, or an extent that does not reach as far as the config's."""
    ext = extent_block(cfg) or {}
    problems = []
    if record.get("schema") != RECORD_SCHEMA:
        problems.append(f"extent record schema {record.get('schema')!r}, expected {RECORD_SCHEMA}")
    if record.get("canonical_version") != CANONICAL_VERSION:
        problems.append(f"extent record from canonical sequence v{record.get('canonical_version')}, this code runs "
                        f"v{CANONICAL_VERSION}: a rebuild would give other stems")
    if record.get("root") != ext.get("root"):
        problems.append(f"extent record is for root {record.get('root')!r}, the config's is {ext.get('root')!r}")
    outer = {s: r for s, r in _reach(record.get("names") or [], record.get("inputs") or {}).items()
             if s in (record.get("sources") or {})}
    return problems + _within(record.get("names") or [], outer, names(cfg), _reach(names(cfg), _inputs(cfg)),
                              f"extent record {record.get('name')!r}")


# ---------------------------------------------------------------------------------------------- extent_progress.json


def read_progress(data_root: Path) -> dict | None:
    p = Path(data_root) / PROGRESS_FILE
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def write_progress(data_root: Path, progress: dict):
    _write_json(Path(data_root) / PROGRESS_FILE, progress, indent=1)


def _write_json(path: Path, obj, indent=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=indent, ensure_ascii=False), encoding="utf-8")
    fsync_path(tmp)
    tmp.replace(path)


def tools() -> dict:
    """The versions that decide what an ingest keeps: libsndfile reads the durations (the 0.3-30 s filter, the Emilia
    budget cut), pyarrow writes the shards."""
    import pyarrow
    import soundfile

    return {"soundfile": soundfile.__version__, "libsndfile": soundfile.__libsndfile_version__,
            "pyarrow": pyarrow.__version__}


# ------------------------------------------------------------------------------------------------------ extent.json


def _pruned_bytes(data_root: Path) -> dict[str, int]:
    out = {}
    p = Path(data_root) / PRUNED_FILE
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                row = json.loads(line)
                out[row["path"]] = int(row["bytes"])
            except (ValueError, KeyError, TypeError):
                continue  # a torn line from a killed append; that shard's size is then unknown
    return out


def build_record(data_root: Path, cfg: dict, *, run_ids: list[str], kitsune_sha: str) -> dict:
    """extent.json for the extent `cfg` ingested into `data_root`: per source its upstream inputs (sorted by name, so
    `ordinal` is the position in the sorted upstream listing: the box reads a prefix, all of it for an uncapped source)
    with the stems each produced, grouped by the inputs the id sidecars name. A budget-cut tar and inputs that kept no
    rows are included. A shard's size comes from the file, or from pruned.jsonl once its audio is deleted (null if
    neither has it). Raises ValueError unless 01 --extent-config completed every step of this extent's plan and every
    planned source still holds as many inputs as the plan reads."""
    data_root = Path(data_root)
    if problems := validate(cfg):
        raise ValueError(f"not a valid extent config: {'; '.join(problems)}")
    plan = plan_steps(cfg)
    steps = [[s.key, cap] for s, cap in plan]
    prog = read_progress(data_root)
    if prog is None or prog.get("canonical_version") != CANONICAL_VERSION or prog.get("steps") != steps:
        raise ValueError(f"{data_root / PROGRESS_FILE}: not an ingest of this extent's plan {steps}")
    if missing := [k for k, _ in steps if k not in prog.get("completed", [])]:
        raise ValueError(f"{data_root / PROGRESS_FILE}: the ingest has not completed {missing}")
    pruned = _pruned_bytes(data_root)
    manifest = read_manifest(data_root)
    sources = {}
    for src in dict.fromkeys(s.source for s, _ in plan):
        progress = load_progress(data_root, src)
        by_input: dict[str, list[dict]] = {}
        for sh in manifest:
            if sh.source != src:
                continue
            meta = sidecar_meta(data_root, sh)
            if meta is None or not meta.get("input") or meta.get("rows") != sh.rows:
                raise ValueError(f"{sh.path}: no id sidecar naming its input and its {sh.rows} rows ({meta})")
            shard = data_root / sh.path
            by_input.setdefault(meta["input"], []).append(dict(
                stem=Path(sh.path).stem, split=sh.split, step=meta["step"], rows=sh.rows, hours=sh.hours,
                ids_sha256=meta["ids_sha256"],
                shard_bytes=shard.stat().st_size if shard.is_file() else pruned.get(sh.path)))
        sizes = progress.get("input_bytes", {})
        inputs = [dict(input=name, ordinal=i, bytes=sizes.get(name), stems=by_input.get(name, []))
                  for i, name in enumerate(sorted(set(progress["finished_inputs"]) | set(by_input)))]
        if (n := progress.get("n_listed")) is not None:  # eval_emilia reads one fixed tar and lists nothing
            want = max(EMILIA_300H_INPUTS if s.key == "emilia_yodas@300h" else n if cap is None else min(cap, n)
                       for s, cap in plan if s.source == src)
            if len(inputs) < want:  # e.g. a later `--sources galgame --force` on this data root
                raise ValueError(f"{src}: the ingest recorded {len(inputs)} of the {want} inputs this extent reads; "
                                 f"was it re-ingested outside --extent-config after its steps completed?")
        stems = [st for inp in inputs for st in inp["stems"]]
        sources[src] = dict(repo=prog.get("repos", {}).get(src), n_listed=progress.get("n_listed"),
                            rows=sum(st["rows"] for st in stems), hours=sum(st["hours"] for st in stems),
                            bytes=sum(inp["bytes"] or 0 for inp in inputs), inputs=inputs)
    ext = extent_block(cfg)
    return {"schema": RECORD_SCHEMA, "name": ext["name"], "root": ext["root"], "canonical_version": CANONICAL_VERSION,
            "kitsune_sha": kitsune_sha, "run_ids": list(run_ids), "revisions": prog.get("revisions", {}),
            "tools": prog.get("tools", {}), "names": names(cfg), "inputs": dict(ext.get("inputs") or {}),
            "steps": [k for k, _ in steps], "min_inputs": {"emilia_yodas": EMILIA_300H_INPUTS}, "sources": sources}


def write_record(path: Path, record: dict):
    """Atomic and compact (~1 MB at the full extent); the same record gives the same bytes."""
    _write_json(Path(path), record)


def load_record(path: Path) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema") != RECORD_SCHEMA:
        raise ValueError(f"{path}: extent record schema {record.get('schema')!r}, expected {RECORD_SCHEMA}")
    return record


def _planned(record: dict, plan: list[tuple[Step, int | None]]):
    """(source, input, stem) of every recorded stem a plan's steps produce: a step's own stems, of its first `cap`
    inputs. The 300 h step's stems all count; the rest step's (tar 8 onward) up to its cap."""
    for step, cap in plan:
        for inp in record["sources"].get(step.source, {}).get("inputs", []):
            if cap is None or inp["ordinal"] < cap:
                for st in inp["stems"]:
                    if st["step"] == step.key:
                        yield step.source, inp, st


def subset_stems(record: dict, cfg: dict) -> dict[str, set[str]]:
    """source -> the stems ("train-00012", "eval-00000", ...) the config's extent takes, for every source and eval set
    it names: a capped source the stems of its first N inputs, emilia_yodas "300h" the 300 h step's, an int K those
    plus the rest step's of the first K tars, an uncapped name all of them. Its dependencies (reazon_small for
    reazon_large) are rebuilt but not taken unless named."""
    out = {n: set() for n in names(cfg)}
    for src, _, st in _planned(record, plan_steps(cfg)):
        if src in out:
            out[src].add(st["stem"])
    return out


def pull_plan(cfg: dict, record: dict, files: list[str]) -> dict:
    """What a consumer of the extent pulls from the data repo listing `files`: {dir_patterns, explicit, required,
    problems}. An uncapped name is pulled by directory globs, a capped one by its explicit <stem>.npz/.jsonl files (a
    prefix of a big source); second_out only for the train sources and selection_recipe.filter_eval_sets (the gate sets
    have none). parakeet_out (<stem>.npz/.jsonl of every name, and its meta.json) only for a CTC student
    (cfg family "ctc") or with cfg pull_parakeet true (the study box pulls both teachers for every run); a CTC run
    without pull_parakeet needs teacher_out only for its eval sets' eval stems (the Cohere baselines), not for its train
    stems. A study selection (selection_recipe.study) also brings its sidecar and manifest (kitsune.prereg.study_files:
    <selection>.json and study_manifest.json next to it; the box checks the manifest hash and every scorer reads it).
    Without these keys (every config before the study) the plan is what it always was. `required` is every file the
    run needs; missing ones are problems. The student's files are the caller's to check (vast/launch.py
    STUDENT_FILES)."""
    problems = validate(cfg)
    if problems:
        return dict(dir_patterns=[], explicit=[], required=[], problems=problems)
    problems = record_problems(record, cfg)
    capped = set(_inputs(cfg))
    teacher_root, second_root = cfg.get("teacher_root", "teacher_out"), cfg.get("second_root", "second_out")
    judged = set(cfg.get("sources") or []) | set((cfg.get("selection_recipe") or {}).get("filter_eval_sets") or [])
    ctc = cfg.get("family", "aed") == "ctc"
    parakeet_root = cfg.get("parakeet_root") if ctc or cfg.get("pull_parakeet") else None
    if (ctc or cfg.get("pull_parakeet")) and not parakeet_root:
        problems.append(f"the config {'trains a CTC student' if ctc else 'sets pull_parakeet'} but has no "
                        f"parakeet_root")
    teacher_all = not ctc or bool(cfg.get("pull_parakeet"))  # else: eval stems of the eval sets only
    eval_sets = set(cfg.get("eval_sets") or [])
    required = [f"{teacher_root}/meta.json", f"{second_root}/meta.json", cfg["selection"]]
    if (cfg.get("selection_recipe") or {}).get("study") is not None:
        required += list(study_files(cfg["selection"]))
    required += [f"{parakeet_root}/meta.json"] if parakeet_root else []
    dir_patterns = list(required) + ([f"{cfg['student'].rstrip('/')}/*"] if cfg.get("student") else [])
    explicit = []
    for name, stems in subset_stems(record, cfg).items():
        if not stems:
            problems.append(f"{name}: the extent record has no stems for it")
        roots = [(teacher_root, (".npz", ".jsonl"), stems if teacher_all else
                  {st for st in stems if name in eval_sets and st.startswith("eval-")})]
        roots += [(second_root, (".jsonl",), stems)] if name in judged else []
        roots += [(parakeet_root, (".npz", ".jsonl"), stems)] if parakeet_root else []
        for r, exts, some in roots:
            wanted = [f"{r}/{name}/{stem}{ext}" for stem in sorted(some) for ext in exts]
            required += wanted
            if name in capped or some != stems:  # a prefix of a big source, or part of a directory (or nothing)
                explicit += wanted
            else:
                dir_patterns.append(f"{r}/{name}/*")
    have = set(files)
    if missing := [f for f in required if f not in have]:
        problems.append(f"{len(missing)} of the {len(required)} files the extent needs are not in the data repo, "
                        f"e.g. {missing[:3]}")
    return dict(dir_patterns=dir_patterns, explicit=explicit, required=required, problems=problems)


def _source_stems(record: dict, src: str) -> list[dict]:
    return [st for inp in record["sources"].get(src, {}).get("inputs", []) for st in inp["stems"]]


def _stem_bytes(record: dict, src: str, st: dict) -> float:
    """A stem's shard size; unknown (pruned without a size): its hours at the source's upstream bytes per hour (a
    source's shards keep the upstream audio bytes, so the two are within ~15 %)."""
    if st["shard_bytes"] is not None:
        return st["shard_bytes"]
    s = record["sources"][src]
    return st["hours"] * s["bytes"] / s["hours"] if s["hours"] else 0.0


def sizing(record: dict, cfg: dict, kept_hours: dict[str, float | dict] | None = None) -> dict:
    """The A100's rebuild of the extent `cfg` from the record, in GB (unrounded) and minutes:
    - down_gb: the upstream bytes every planned step downloads (tar 8 twice: the 300 h step cuts it, the rest reads it
      again);
    - shard_gb: the rebuilt stems of every planned step, dependencies included;
    - sel_gb: the audio the trainer copies into its cache: `kept_hours` (by source, or make_selection's `kept` as it
      is stored, {"<source>/<split>": {"utts", "hours"}}) at each source's shard GB per hour; None: every subset stem;
    - hours: the audio of the subset stems; labels_gb: their label files, LABEL_GB_PER_HOUR each;
    - disk_gb = DISK_MARGIN x (DISK_BASE_GB + shard + sel + labels), up to a multiple of DISK_STEP_GB;
    - rebuild_timeout_min = REBUILD_BASE_MIN + REBUILD_SLACK x the download time at REBUILD_BYTES_PER_S."""
    plan = plan_steps(cfg)
    down = 0
    for step, cap in plan:
        lo = EMILIA_300H_INPUTS - 1 if step.key == "emilia_yodas" else 0  # the rest step starts at the cut tar
        hi = EMILIA_300H_INPUTS if step.key == "emilia_yodas@300h" else math.inf if cap is None else cap
        down += sum(inp["bytes"] or 0 for inp in record["sources"].get(step.source, {}).get("inputs", [])
                    if lo <= inp["ordinal"] < hi)
    shard = sum(_stem_bytes(record, src, st) for src, _, st in _planned(record, plan))
    subset = subset_stems(record, cfg)
    taken = [(src, st) for src, _, st in _planned(record, plan) if st["stem"] in subset.get(src, ())]
    hours = sum(st["hours"] for _, st in taken)
    if kept_hours is None:
        sel = sum(_stem_bytes(record, src, st) for src, st in taken)
    else:
        sel = 0.0
        for key, h in kept_hours.items():
            h = h["hours"] if isinstance(h, dict) else h
            src = key.split("/", 1)[0]
            stems = _source_stems(record, src)
            src_hours = sum(st["hours"] for st in stems)
            if src_hours:
                sel += h * sum(_stem_bytes(record, src, st) for st in stems) / src_hours
    labels_gb = hours * LABEL_GB_PER_HOUR
    need = DISK_MARGIN * (DISK_BASE_GB + shard / 1e9 + sel / 1e9 + labels_gb)
    return dict(down_gb=down / 1e9, shard_gb=shard / 1e9, sel_gb=sel / 1e9, labels_gb=labels_gb, hours=hours,
                disk_gb=int(math.ceil(need / DISK_STEP_GB) * DISK_STEP_GB),
                rebuild_timeout_min=int(math.ceil(REBUILD_BASE_MIN + REBUILD_SLACK * down / REBUILD_BYTES_PER_S / 60)))
