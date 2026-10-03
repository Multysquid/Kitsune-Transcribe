"""vast/bootstrap.sh in extent mode: the A100 pulls exactly a config's extent of the label box's labels, rebuilds its
audio with `01 --extent-config` and checks the join exactly (plan §12.2, §3.5). The viability path is unchanged.

CPU only, no network: huggingface_hub is replaced by a stub that serves a fake data repo from a local directory (the
real filter_repo_objects matcher is loaded from the installed package, as snapshot_download filters with it); the
helper heredoc is extracted and run with the real kitsune.extent; the rebuild dispatch runs under Git Bash with fake
phase/retry (skipped without bash)."""
import functools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "vast" / "bootstrap.sh"
sys.path.insert(0, str(ROOT))
from kitsune.store import ShardWriter, ids_sha256  # noqa: E402

STUDENT = "students/s"
STUDENT_FILES = ("config.json", "model.safetensors", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                 "student_meta.json", "README.md")  # vast/launch.py STUDENT_FILES
LR = "labels/full"


def find_bash() -> str | None:
    for cand in (shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        # as test_infra.find_bash: System32\bash.exe and WindowsApps\bash.exe are WSL, not Git Bash, and cannot read the
        # Windows paths the tests pass (exit 127 from a PowerShell-started pytest)
        if cand and Path(cand).exists() and not any(w in cand.lower() for w in ("system32", "windowsapps")):
            return cand
    return None


def helper_source() -> str:
    return re.search(r"<<'PYEOF'\n(.*?\n)PYEOF\n", BOOTSTRAP.read_text(encoding="utf-8"), re.S).group(1)


def make_cfg(inputs: dict | None = None, extent: bool = True) -> dict:
    """A small extent config shaped like configs/full_sub3k.json: galgame (train and hold-out, capped) and a gate
    set."""
    cfg = {"run_name": "sub", "student": STUDENT, "data_root": "data",
           "teacher_root": f"{LR}/teacher_out", "second_root": f"{LR}/second_out",
           "parakeet_root": f"{LR}/parakeet_out", "selection": f"{LR}/selections/sub.parquet",
           "extent": {"name": "sub", "root": LR, "inputs": dict({"galgame": 1} if inputs is None else inputs)},
           "sources": ["galgame"], "eval_sets": ["eval_jsut", "galgame"],
           "selection_recipe": {"agree_max": 0.5, "filter_eval_sets": ["galgame"], "partial_second_opinion": []}}
    if not extent:
        cfg = {k: v for k, v in cfg.items() if k not in ("extent", "parakeet_root")}
        cfg.update(teacher_root="teacher_out", second_root="second_out", selection="selection/v.parquet")
    return cfg


# the rows of every stem the toy label box rebuilt: galgame tar 0 -> train-00000 + the hold-out eval-00000, tar 1 ->
# train-00001; eval_jsut's one input -> eval-00000
ROWS = {("galgame", "train-00000"): [f"galgame/t0_{i}" for i in range(3)],
        ("galgame", "eval-00000"): ["galgame/e0_0", "galgame/e0_1"],
        ("galgame", "train-00001"): [f"galgame/t1_{i}" for i in range(2)],
        ("eval_jsut", "eval-00000"): [f"eval_jsut/BASIC5000_{i:04d}" for i in range(2)]}


def make_record(rows: dict | None = None) -> dict:
    rows = rows or ROWS

    def stem(src, name, step):
        ids = rows[(src, name)]
        return {"stem": name, "split": name.split("-")[0], "step": step, "rows": len(ids), "hours": 0.01,
                "ids_sha256": ids_sha256(ids), "shard_bytes": 1000}

    def inp(n, name, stems):
        return {"input": name, "ordinal": n, "bytes": 5000, "stems": stems}

    return {"schema": 1, "name": "full", "root": LR, "canonical_version": 1, "kitsune_sha": "0" * 40, "run_ids": ["r"],
            "names": ["galgame", "eval_jsut"], "inputs": {}, "steps": ["eval_jsut", "galgame"],
            "sources": {
                "galgame": {"repo": "g", "n_listed": 2, "rows": 7, "hours": 0.03, "bytes": 10000, "inputs": [
                    inp(0, "tar0", [stem("galgame", "train-00000", "galgame"),
                                    stem("galgame", "eval-00000", "galgame")]),
                    inp(1, "tar1", [stem("galgame", "train-00001", "galgame")])]},
                "eval_jsut": {"repo": "j", "n_listed": None, "rows": 2, "hours": 0.01, "bytes": 5000, "inputs": [
                    inp(0, "jsut.zip", [stem("eval_jsut", "eval-00000", "eval_jsut")])]}}}


def label_files() -> list[str]:
    """Everything the label box uploaded for the toy extent (plus laptop roots a consumer never pulls)."""
    files = [f"{LR}/teacher_out/meta.json", f"{LR}/second_out/meta.json", f"{LR}/parakeet_out/meta.json",
             f"{LR}/selections/full.parquet", f"{LR}/selections/sub.parquet", f"{LR}/extent.json",
             f"{LR}/COMPLETE.json", *(f"{STUDENT}/{n}" for n in STUDENT_FILES),
             "teacher_out/galgame/train-00000.npz", "selection/v.parquet", "README.md"]
    for src, stem in ROWS:
        files += [f"{LR}/teacher_out/{src}/{stem}.npz", f"{LR}/teacher_out/{src}/{stem}.jsonl",
                  f"{LR}/parakeet_out/{src}/{stem}.npz", f"{LR}/parakeet_out/{src}/{stem}.jsonl"]
        files += [f"{LR}/second_out/{src}/{stem}.jsonl"] if src == "galgame" else []
    return files


STUB = '''\
import json, os, shutil, threading
from pathlib import Path
REMOTE = Path(os.environ["FAKE_REMOTE"])
LOG = Path(os.environ["FAKE_LOG"])
_LOG_LOCK = threading.Lock()

def _log(*a):
    # one call at a time, each line one write on an O_APPEND fd: the pull calls hf_hub_download from 16 threads, and
    # Windows' C runtime emulates O_APPEND (seek to the end, then write), so two threads could write at one offset and
    # lose a line or leave a torn one (Box.calls(): a missing call, or JSONDecodeError, 2026-10-03)
    with _LOG_LOCK, open(LOG, "ab", buffering=0) as f:
        f.write((json.dumps(a) + "\\n").encode("utf-8"))

def _files():
    return sorted(p.relative_to(REMOTE).as_posix() for p in REMOTE.rglob("*") if p.is_file())

def _get(f, local_dir):
    dst = Path(local_dir) / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REMOTE / f, dst)
    return str(dst)

class HfApi:
    def whoami(self):
        return {"name": "u"}
    def auth_check(self, *a, **k):
        pass
    def list_repo_files(self, *a, **k):
        return _files()

def hf_hub_download(repo_id, filename, *, repo_type=None, revision=None, local_dir=None):
    _log("hf_hub_download", filename, repo_type, revision)
    return _get(filename, local_dir)

def snapshot_download(repo_id, *, repo_type=None, revision=None, local_dir=None, allow_patterns=None, max_workers=8):
    from huggingface_hub.utils import filter_repo_objects
    got = list(filter_repo_objects(_files(), allow_patterns=allow_patterns))
    _log("snapshot_download", len(got), revision)
    for f in got:
        _get(f, local_dir)
    return str(local_dir)
'''


@pytest.fixture
def box(tmp_path):
    import huggingface_hub.utils._paths as hf_paths

    stub = tmp_path / "stub" / "huggingface_hub"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(STUB, encoding="utf-8")
    (stub / "utils.py").write_text(  # the real matcher (pure Python), which snapshot_download filters the tree with
        "import importlib.util\n"
        f"_s = importlib.util.spec_from_file_location('_hf_paths', {hf_paths.__file__!r})\n"
        "_m = importlib.util.module_from_spec(_s)\n_s.loader.exec_module(_m)\n"
        "filter_repo_objects = _m.filter_repo_objects\n", encoding="utf-8")
    remote, kdir, state = tmp_path / "remote", tmp_path / "box", tmp_path / "state"
    for f in label_files():
        (remote / f).parent.mkdir(parents=True, exist_ok=True)
        (remote / f).write_bytes(b"x")
    (remote / LR / "extent.json").write_text(json.dumps(make_record()), encoding="utf-8")
    (kdir / "configs").mkdir(parents=True)
    state.mkdir()
    (tmp_path / "helper.py").write_text(helper_source(), encoding="utf-8")
    log = tmp_path / "hub.log"
    env = dict(os.environ, KITSUNE_DIR=str(kdir), STATE=str(state), CONFIG="configs/c.json",
               KITSUNE_DATA_REPO="u/data", KITSUNE_OUT_REPO="u/runs", KITSUNE_DATA_REVISION="0123abc",
               FAKE_REMOTE=str(remote), FAKE_LOG=str(log), CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([str(tmp_path / "stub"), str(ROOT)]))

    class Box:
        root, remote_dir, state_dir = kdir, remote, state

        @staticmethod
        def config(cfg: dict):
            (kdir / "configs" / "c.json").write_text(json.dumps(cfg), encoding="utf-8")

        @staticmethod
        def helper(cmd: str, **extra):
            return subprocess.run([sys.executable, str(tmp_path / "helper.py"), cmd], capture_output=True, text=True,
                                  env=dict(env, **extra), timeout=120)

        @staticmethod
        def calls() -> list:
            if not log.is_file():
                return []
            out = [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]
            log.unlink()
            return out

        @staticmethod
        def on_disk() -> set[str]:
            skip = ("configs/", "data/", ".cache/")
            return {p.relative_to(kdir).as_posix() for p in kdir.rglob("*") if p.is_file()
                    and not p.relative_to(kdir).as_posix().startswith(skip)}

    return Box


def test_extent_plan_pulls_exactly_the_subset_files(box):
    """A capped source's label files are pulled one by one (its first N inputs' stems only), an uncapped one by
    directory, second opinions only where the selection reads them, never parakeet_out and never a laptop root; the
    plan records extent mode, the explicit files and every name to rebuild. A re-run of the pull downloads nothing it
    already has."""
    box.config(make_cfg())
    r = box.helper("plan")
    assert r.returncode == 0, r.stdout + r.stderr
    plan = json.loads((box.state_dir / "bootstrap_plan.json").read_text(encoding="utf-8"))
    assert plan["extent"] is True and plan["parked"] == [] and plan["rebuild"] == ["galgame", "eval_jsut"]
    assert plan["revision"] == "0123abc" and Path(plan["record"]).is_file()
    calls = box.calls()
    assert calls == [["hf_hub_download", f"{LR}/extent.json", "dataset", "0123abc"]], "the record at the pinned rev"
    tar0 = [f"{LR}/{r}/galgame/{s}.{e}" for s in ("eval-00000", "train-00000")
            for r, e in (("teacher_out", "npz"), ("teacher_out", "jsonl"), ("second_out", "jsonl"))]
    assert sorted(plan["explicit"]) == sorted(tar0)
    want = {f"{LR}/teacher_out/meta.json", f"{LR}/second_out/meta.json", f"{LR}/selections/sub.parquet",
            *(f"{STUDENT}/{n}" for n in STUDENT_FILES), *tar0,
            f"{LR}/teacher_out/eval_jsut/eval-00000.npz", f"{LR}/teacher_out/eval_jsut/eval-00000.jsonl"}
    assert set(plan["files"]) == want
    assert not any("parakeet_out" in p for p in plan["patterns"] + plan["explicit"])

    r = box.helper("pull")
    assert r.returncode == 0, r.stdout + r.stderr
    assert box.on_disk() == want, "exactly the subset: no tar 1 stem, no parakeet_out, no laptop root"
    assert sorted(c[1] for c in box.calls() if c[0] == "hf_hub_download") == sorted(tar0)
    r = box.helper("pull")
    assert r.returncode == 0, r.stdout + r.stderr
    assert [c for c in box.calls() if c[0] == "hf_hub_download"] == [], "explicit files on disk are not fetched again"
    assert "pulled 0 of 6 explicit files (6 on disk)" in r.stdout

    # the uncapped extent pulls galgame by directory: tar 1's stem comes too
    box.config(make_cfg(inputs={}))
    assert box.helper("plan").returncode == 0
    plan = json.loads((box.state_dir / "bootstrap_plan.json").read_text(encoding="utf-8"))
    assert plan["explicit"] == [] and f"{LR}/teacher_out/galgame/*" in plan["patterns"]
    assert f"{LR}/teacher_out/galgame/train-00001.npz" in plan["files"]


def test_extent_plan_refuses_an_unsealed_root_or_missing_files(box):
    """Only a sealed label root is consumed: without COMPLETE.json (or extent.json) the plan refuses with exit 3,
    which bootstrap's retry() does not repeat, before downloading anything. A subset stem, a second opinion or a
    student file the repo lacks is refused the same way."""
    box.config(make_cfg())
    for gone in ("COMPLETE.json", "extent.json"):
        (box.remote_dir / LR / gone).rename(box.remote_dir / f"{gone}.away")
        r = box.helper("plan")
        assert r.returncode == 3 and "(not retried)" in r.stderr, r.stdout + r.stderr
        assert f"lacks ['{LR}/{gone}']" in r.stderr and "has not sealed" in r.stderr
        assert not (box.state_dir / "bootstrap_plan.json").exists() and box.calls() == []
        (box.remote_dir / f"{gone}.away").rename(box.remote_dir / LR / gone)
    for gone in (f"{LR}/second_out/galgame/train-00000.jsonl", f"{STUDENT}/tokenizer.json"):
        (box.remote_dir / gone).unlink()
        r = box.helper("plan")
        assert r.returncode == 3 and "cannot serve extent 'sub'" in r.stderr, r.stdout + r.stderr
        assert gone.rsplit("/", 1)[1] in r.stderr and not (box.state_dir / "bootstrap_plan.json").exists()
        (box.remote_dir / gone).write_bytes(b"x")
    # a config the record cannot serve: the label box's extent read galgame's first input only, the config wants two
    (box.remote_dir / LR / "extent.json").write_text(json.dumps(dict(make_record(), inputs={"galgame": 1})),
                                                     encoding="utf-8")
    box.config(make_cfg(inputs={"galgame": 2}))
    r = box.helper("plan")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "galgame: needs the first 2 inputs, extent record 'full' has the first 1 inputs" in r.stderr
    # an invalid extent config
    box.config(make_cfg(inputs={"galgame": 0}))
    r = box.helper("plan")
    assert r.returncode == 3 and "extent.inputs.galgame" in r.stderr, r.stdout + r.stderr


def test_the_viability_plan_is_unchanged(box):
    """A config without an extent takes the legacy path: no record download, no extent keys in the plan, and the pull
    uses snapshot_download alone."""
    box.config(make_cfg(extent=False))
    for f in ["teacher_out/meta.json", "second_out/meta.json", "teacher_out/galgame/train-00000.npz",
              "teacher_out/galgame/eval-00000.npz", "teacher_out/eval_jsut/eval-00000.npz",
              "second_out/galgame/train-00000.jsonl", "second_out/galgame/eval-00000.jsonl"]:
        (box.remote_dir / f).parent.mkdir(parents=True, exist_ok=True)
        (box.remote_dir / f).write_bytes(b"x")
    r = box.helper("plan")
    assert r.returncode == 0, r.stdout + r.stderr
    plan = json.loads((box.state_dir / "bootstrap_plan.json").read_text(encoding="utf-8"))
    assert not {"extent", "explicit", "record"} & set(plan) and plan["rebuild"] == ["galgame", "eval_jsut"]
    assert box.calls() == []
    assert box.helper("pull").returncode == 0
    assert [c[0] for c in box.calls()] == ["snapshot_download"]
    assert not any(f.startswith("labels/") for f in box.on_disk())


def write_rebuilt(kdir: Path, rows: dict):
    """What `01 --extent-config` leaves: shards, id sidecars and manifest lines (one input per stem, as 01 flushes)."""
    for (src, stem), ids in rows.items():
        split = stem.split("-")[0]
        w = ShardWriter(kdir / "data", src, split, meta={"input": stem, "step": src})
        for i in ids:
            w.add(i, b"", "x", 1.0, 16000)
        w.close()


def write_teacher(kdir: Path, stems: dict):
    for (src, stem), ids in stems.items():
        d = kdir / LR / "teacher_out" / src
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{stem}.npz", ids=np.array(ids))


def test_extent_coverage_is_exact(box):
    """In extent mode every pulled stem's rebuilt ids must hash to the record's ids_sha256 and hold its teacher ids,
    and every split must join at 1.0 whatever KITSUNE_MIN_COVERAGE says: the rebuild replays the label box's ingest,
    so anything less is another extent (an upstream that changed, another canonical sequence) and would train on
    rows the trainer silently drops."""
    box.config(make_cfg())
    assert box.helper("plan").returncode == 0
    subset = {k: v for k, v in ROWS.items() if k != ("galgame", "train-00001")}  # the capped subset: tar 0 only
    write_rebuilt(box.root, ROWS)  # 01 --extent-config rebuilt tar 0 of galgame (and more) plus eval_jsut
    write_teacher(box.root, {k: v[:-1] if k[1] == "train-00000" else v for k, v in subset.items()})  # a skipped row
    r = box.helper("coverage", KITSUNE_MIN_COVERAGE="0.5")
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads((box.state_dir / "bootstrap_coverage.json").read_text(encoding="utf-8"))
    assert report["extent_stems"] == {"checked": 3, "failed": 0, "failures": []}
    assert "3 of 3 pulled stems rebuilt with the labelled ids" in r.stdout

    # a rebuilt stem whose ids differ from the labelled ones (same count, other order): refused by the hash
    rec = make_record({**ROWS, ("galgame", "eval-00000"): ["galgame/e0_1", "galgame/e0_0"]})
    plan = json.loads((box.state_dir / "bootstrap_plan.json").read_text(encoding="utf-8"))
    Path(plan["record"]).write_text(json.dumps(rec), encoding="utf-8")
    r = box.helper("coverage", KITSUNE_MIN_COVERAGE="0.5")
    assert r.returncode != 0 and "galgame/eval-00000: rebuilt ids_sha256" in r.stdout, r.stdout + r.stderr
    assert "1 extent stem(s)" in r.stderr and "coverage below 1.0" in r.stderr
    Path(plan["record"]).write_text(json.dumps(make_record()), encoding="utf-8")

    # a teacher id the rebuilt stem lacks: refused per stem, and the split's join is below the exact floor
    write_teacher(box.root, {("eval_jsut", "eval-00000"): ROWS[("eval_jsut", "eval-00000")] + ["eval_jsut/X"]})
    r = box.helper("coverage", KITSUNE_MIN_COVERAGE="0.5")
    assert r.returncode != 0 and "eval_jsut/eval-00000: 1 teacher ids not in the rebuilt stem" in r.stdout
    assert "eval_jsut/eval" in r.stderr, "0.667 joined passes the env floor but not the extent's 1.0"
    write_teacher(box.root, {("eval_jsut", "eval-00000"): ROWS[("eval_jsut", "eval-00000")]})

    # a pulled stem 01 never rebuilt
    lines = (box.root / "data" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    (box.root / "data" / "manifest.jsonl").write_text(
        "".join(ln + "\n" for ln in lines if "galgame/train-00000" not in ln), encoding="utf-8")
    r = box.helper("coverage", KITSUNE_MIN_COVERAGE="0.5")
    assert r.returncode != 0 and "galgame/train-00000: not rebuilt" in r.stdout, r.stdout + r.stderr


def test_the_legacy_rebuild_line_stays_first_and_the_helper_is_the_first_heredoc():
    """The regexes of tests/test_infra.py find the first `phase rebuild_audio` line and the first heredoc: the legacy
    rebuild stays first and literally unchanged, the extent line comes after it, and the helper stays the first
    heredoc."""
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert helper_source().startswith('"""bootstrap helper: plan / pull / coverage.')
    lines = [ln.strip() for ln in text.splitlines() if ln.lstrip().startswith("phase rebuild_audio")]
    assert len(lines) == 2
    assert lines[0] == ('phase rebuild_audio retry 3 timeout -k 60 60m "$PY" scripts/01_prepare_data.py --data '
                        '"$KITSUNE_DIR/$DATA_ROOT" \\')
    assert re.search(r"phase rebuild_audio retry \d+ timeout -k \d+ \d+m \"\$PY\" scripts/01_prepare_data\.py",
                     lines[0])
    assert lines[1] == 'phase rebuild_audio retry 3 timeout -k 60 "${KITSUNE_REBUILD_TIMEOUT_MIN:-60}m" "$PY" \\'
    assert '--sources "${REBUILD[@]}" "${PREP_ARGS[@]}"' in text
    assert 'scripts/01_prepare_data.py --data "$KITSUNE_DIR/$DATA_ROOT" --extent-config "$CONFIG"' in text
    assert "KITSUNE_REBUILD_TIMEOUT_MIN" in text.split("set -euo pipefail", 1)[0], "the header documents it"
    assert b"\r\n" not in BOOTSTRAP.read_bytes()


@pytest.mark.parametrize("extent", [False, True])
def test_the_rebuild_dispatch(tmp_path, extent):
    """The shell picks the rebuild from the plan: the legacy `--sources` call with KITSUNE_PREP_ARGS, or in extent mode
    `01 --extent-config $CONFIG` under KITSUNE_REBUILD_TIMEOUT_MIN, with KITSUNE_PREP_ARGS ignored (and logged)."""
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    text = BOOTSTRAP.read_text(encoding="utf-8")
    block = re.search(r"^EXTENT=.*?^fi\n", text, re.M | re.S).group(0)
    plan = dict(rebuild=["galgame", "eval_jsut"], data_root="data", **({"extent": True} if extent else {}))
    (tmp_path / "bootstrap_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    script = tmp_path / "dispatch.sh"
    script.write_text("\n".join([
        "set -euo pipefail", "log() { printf 'LOG %s\\n' \"$*\"; }", "phase() { printf 'PHASE %s\\n' \"$*\"; }",
        f'PY="{Path(sys.executable).as_posix()}"', f'STATE="{tmp_path.as_posix()}"', 'KITSUNE_DIR=/k', "DATA_ROOT=data",
        "CONFIG=configs/sub.json", "REBUILD=(galgame eval_jsut)", block, ""]), encoding="utf-8", newline="\n")
    env = dict(os.environ, KITSUNE_PREP_ARGS="--galgame-shards 8", KITSUNE_REBUILD_TIMEOUT_MIN="95")
    r = subprocess.run([bash, str(script)], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    (phase,) = [ln for ln in r.stdout.splitlines() if ln.startswith("PHASE")]
    if extent:
        assert phase == ("PHASE rebuild_audio retry 3 timeout -k 60 95m " + Path(sys.executable).as_posix()
                         + " scripts/01_prepare_data.py --data /k/data --extent-config configs/sub.json")
        assert "LOG KITSUNE_PREP_ARGS ignored: the config's extent defines the rebuild" in r.stdout
    else:
        assert phase.endswith("scripts/01_prepare_data.py --data /k/data --sources galgame eval_jsut "
                              "--galgame-shards 8") and " 60m " in phase
        assert "ignored" not in r.stdout


# ======================================================================================= full-data boxes (WP3)
# fix 9 (the CTC-only box: parakeet ids for its train stems), the job-full plan (the registry's students and extra
# files, the scratch repo's token check, the labels pulled apart from the rest), and the job-full phase order run end to
# end under Git Bash with a fake interpreter that records every call


def test_extent_canonical_sequence_is_unchanged():
    """kitsune/extent.py CANONICAL and CANONICAL_VERSION (1) are frozen: another sequence gives other ids and stems
    than the label box's, and every sealed record says v1 (contract 0.5)."""
    from kitsune import extent

    assert extent.CANONICAL_VERSION == 1
    assert [(s.key, s.source, s.emilia_hours) for s in extent.CANONICAL] == [
        ("reazon_small", "reazon_small", None), ("eval_jsut", "eval_jsut", None), ("eval_cv8", "eval_cv8", None),
        ("eval_reazon", "eval_reazon", None), ("emilia_yodas@300h", "emilia_yodas", 300.0),
        ("eval_emilia", "eval_emilia", None), ("galgame", "galgame", None),
        ("emilia_yodas", "emilia_yodas", float("inf")), ("emilia_nc", "emilia_nc", None),
        ("reazon_large", "reazon_large", None)]


def ctc_cfg(**kw) -> dict:
    """make_cfg as box p01's data: a CTC run without pull_parakeet."""
    return dict(make_cfg(), family="ctc", **kw)


def test_label_root_for_is_the_pull_plans_rule():
    """The one rule: parakeet for every stem of a CTC run without pull_parakeet, except its eval sets' eval stems
    (whose teacher_out it pulls for the Cohere baselines); teacher otherwise. The teacher files pull_plan asks for are
    exactly the stems label_root_for gives to the teacher."""
    from kitsune import extent

    rec, files = make_record(), label_files()
    for cfg in (ctc_cfg(), ctc_cfg(pull_parakeet=True), make_cfg()):
        plan = extent.pull_plan(cfg, rec, files)
        assert plan["problems"] == [], plan["problems"]
        teacher = {f.rsplit("/", 2)[1] + "/" + f.rsplit("/", 1)[1][:-4] for f in plan["required"]
                   if "/teacher_out/" in f and f.endswith(".npz")}
        for (src, stem) in [("galgame", "train-00000"), ("galgame", "eval-00000"), ("eval_jsut", "eval-00000")]:
            which = extent.label_root_for(cfg, src, stem)
            assert which == ("teacher" if f"{src}/{stem}" in teacher else "parakeet"), (cfg.get("family"), src, stem)
    assert extent.label_root_for(ctc_cfg(), "galgame", "train-00000") == "parakeet"
    assert extent.label_root_for(ctc_cfg(), "galgame", "eval-00000") == "teacher"  # galgame is an eval set too
    assert extent.label_root_for(ctc_cfg(eval_sets=["eval_jsut"]), "galgame", "eval-00000") == "parakeet"
    assert extent.label_root_for(ctc_cfg(pull_parakeet=True), "galgame", "train-00000") == "teacher"
    assert extent.label_root_for(make_cfg(), "galgame", "train-00000") == "teacher"


def write_parakeet(kdir: Path, stems: dict):
    for (src, stem), ids in stems.items():
        d = kdir / LR / "parakeet_out" / src
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / f"{stem}.npz", ids=np.array(ids))


def test_ctc_only_extent_plan_and_coverage_read_the_parakeet_ids(box):
    """Box 1 (fix 9): a CTC config without pull_parakeet pulls parakeet_out for every subset stem and teacher_out only
    for its eval sets' eval stems. The extent check and the coverage then read each train stem's ids from parakeet_out:
    at fb77ee1 the extent check crashed on the missing teacher npz of a train stem, and coverage counted the train
    splits at 0 after the paid rebuild."""
    box.config(ctc_cfg())
    r = box.helper("plan")
    assert r.returncode == 0, r.stdout + r.stderr
    plan = json.loads((box.state_dir / "bootstrap_plan.json").read_text(encoding="utf-8"))
    teacher = sorted(f for f in plan["files"] if "/teacher_out/" in f and not f.endswith("meta.json"))
    assert teacher == [f"{LR}/teacher_out/{s}/eval-00000.{e}" for s in ("eval_jsut", "galgame") for e in ("jsonl", "npz")]
    parakeet = {f for f in plan["files"] if "/parakeet_out/" in f and not f.endswith("meta.json")}
    assert parakeet == {f"{LR}/parakeet_out/{s}/{st}.{e}" for s, st in [("galgame", "train-00000"),
                                                                          ("galgame", "eval-00000"),
                                                                          ("eval_jsut", "eval-00000")]
                        for e in ("npz", "jsonl")}
    subset = {k: v for k, v in ROWS.items() if k != ("galgame", "train-00001")}
    write_rebuilt(box.root, ROWS)
    write_teacher(box.root, {k: v for k, v in subset.items() if k[1].startswith("eval-")})  # no train npz at all
    write_parakeet(box.root, subset)
    r = box.helper("coverage", KITSUNE_MIN_COVERAGE="0.99")
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads((box.state_dir / "bootstrap_coverage.json").read_text(encoding="utf-8"))
    assert report["extent_stems"] == {"checked": 3, "failed": 0, "failures": []}
    assert report["galgame/train"]["coverage"] == 1.0 and report["galgame/train"]["root"] == "parakeet"
    assert report["galgame/train"]["teacher_ids"] == 3  # the label ids (here Parakeet's)
    assert "root" not in report["galgame/eval"] and "root" not in report["eval_jsut/eval"]
    assert re.search(r"galgame/train +parakeet ", r.stdout), r.stdout
    # a Parakeet id the rebuilt stem lacks is refused per stem, like a teacher id
    write_parakeet(box.root, {("galgame", "train-00000"): ROWS[("galgame", "train-00000")] + ["galgame/X"]})
    r = box.helper("coverage", KITSUNE_MIN_COVERAGE="0.5")
    assert r.returncode != 0 and "galgame/train-00000: 1 parakeet ids not in the rebuilt stem" in r.stdout
    # a stem whose labels are not on disk is a failure, not a crash
    (box.root / LR / "parakeet_out" / "galgame" / "train-00000.npz").unlink()
    r = box.helper("coverage", KITSUNE_MIN_COVERAGE="0.5")
    assert r.returncode != 0 and "galgame/train-00000: no parakeet labels" in r.stdout, r.stdout + r.stderr
    assert "Traceback" not in r.stderr


@pytest.mark.parametrize("cfg", [dict(family="ctc", pull_parakeet=True), {}], ids=["both-roots", "aed"])
def test_both_roots_and_aed_coverage_read_teacher_ids_as_before(box, cfg):
    """A box with both label roots (pull_parakeet) or an AED box joins the teacher ids, as at fb77ee1: Parakeet npz on
    disk (with ids the rebuilt stem lacks) are not read, and the report has no root field."""
    box.config(dict(make_cfg(), **cfg))
    assert box.helper("plan").returncode == 0
    subset = {k: v for k, v in ROWS.items() if k != ("galgame", "train-00001")}
    write_rebuilt(box.root, ROWS)
    write_teacher(box.root, subset)
    write_parakeet(box.root, {k: v + ["not/rebuilt"] for k, v in subset.items()})
    r = box.helper("coverage", KITSUNE_MIN_COVERAGE="0.99")
    assert r.returncode == 0, r.stdout + r.stderr
    report = json.loads((box.state_dir / "bootstrap_coverage.json").read_text(encoding="utf-8"))
    assert all("root" not in v for k, v in report.items() if k != "extent_stems")
    assert "parakeet" not in r.stdout and re.search(r"galgame/train +teacher ", r.stdout), r.stdout


# ------------------------------------------------------------------------------------------- the job-full plan


P01_STUDENT = "students/study/p01"
SIDECAR = f"{LR}/selections/sub.json"
MANIFEST = f"{LR}/selections/study_manifest.json"
PARAKEET_MODEL = "models/parakeet-tdt_ctc-0.6b-ja-hf"

FULL_STUB = STUB.replace('''    def auth_check(self, *a, **k):
        pass
''', '''    def auth_check(self, repo_id, *, repo_type=None, write=False):
        _log("auth_check", repo_id, repo_type, write)
        deny = os.environ.get("DENY_" + repo_id.replace("/", "_").upper())
        if deny and (write or deny.startswith("r")):
            import types
            e = Exception(f"{deny.lstrip('r')} from the Hub")
            e.response = types.SimpleNamespace(status_code=int(deny.lstrip("r")))
            raise e
''')


def full_box(box, *, timed=True, extra_files=(SIDECAR, MANIFEST), extra_dirs=(PARAKEET_MODEL,)) -> dict:
    """Box p01 of a registry written into the box's checkout (configs/full/, as WP2c's), on the toy extent: the data
    config is ctc_cfg() without a student, the train item's config the same plus run name and student; the remote
    gains the student (with its CTC card), the extra files and the extra dir."""
    from kitsune import fullrun

    assert "auth_check(self, repo_id" in FULL_STUB
    stub = box.root.parent / "stub" / "huggingface_hub" / "__init__.py"
    stub.write_text(FULL_STUB, encoding="utf-8")
    data = {k: v for k, v in ctc_cfg().items() if k not in ("student", "run_name")}
    folder = box.root / "configs" / "full"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data-p01.json").write_text(json.dumps(data), encoding="utf-8")
    (folder / "full-p01.json").write_text(json.dumps(dict(data, run_name="full-p01", student=P01_STUDENT)),
                                          encoding="utf-8")
    reg = {"version": 1, "boxes": {"p01": {
        "gpus": 1, "data_config": "configs/full/data-p01.json", "est_hours": 19.5, "max_hours": 22, "max_dph": 1.0,
        "deadline_reserve_min": 45, "watchdog": {"orphan_s": 3600, "action": "stop"}, "timed_states": timed,
        "extra_files": list(extra_files), "extra_dirs": list(extra_dirs),
        "items": [{"name": "stores-ctc", "kind": "stores", "config": "configs/full/full-p01.json"},
                  {"name": "full-p01", "kind": "train", "config": "configs/full/full-p01.json",
                   "study_run": "study-p01", "family": "ctc", "max_hours": 13.56, "needs": ["stores-ctc"]}]}}}
    (folder / "boxes.json").write_text(json.dumps(reg), encoding="utf-8")
    fullrun.load_registry(reg, root=box.root)  # a valid registry
    for f in [*(f"{P01_STUDENT}/{n}" for n in (*STUDENT_FILES, "MODEL_CARD.md")), SIDECAR, MANIFEST,
              f"{PARAKEET_MODEL}/config.json"]:
        (box.remote_dir / f).parent.mkdir(parents=True, exist_ok=True)
        (box.remote_dir / f).write_bytes(b"x")
    return dict(KITSUNE_JOB="full", KITSUNE_BOX="p01", CONFIG="configs/full/data-p01.json",
                KITSUNE_SCRATCH_REPO="u/scratch", KITSUNE_FULL_REGISTRY="")


def test_full_plan_pulls_the_registrys_students_and_extra_files_and_the_labels_apart(box):
    """KITSUNE_JOB=full: the data config names no student; the registry's train student (with its CC-BY-4.0 card), its
    extra files and dirs are pulled with the derived data, and every label file goes to the plan's labels part, which
    pull_labels fetches while 01 rebuilds the audio. Between them the two pulls leave exactly the files of the plan."""
    env = full_box(box)
    r = box.helper("plan", **env)
    assert r.returncode == 0, r.stdout + r.stderr
    plan = json.loads((box.state_dir / "bootstrap_plan.json").read_text(encoding="utf-8"))
    derived, labels = set(plan["files"]), set(plan["labels"]["files"])
    assert derived == {f"{LR}/teacher_out/meta.json", f"{LR}/second_out/meta.json", f"{LR}/parakeet_out/meta.json",
                       f"{LR}/selections/sub.parquet", SIDECAR, MANIFEST, f"{PARAKEET_MODEL}/config.json",
                       *(f"{P01_STUDENT}/{n}" for n in (*STUDENT_FILES, "MODEL_CARD.md"))}
    assert labels and all(re.match(rf"{LR}/(teacher|second|parakeet)_out/(galgame|eval_jsut)/", f) for f in labels)
    assert f"{LR}/parakeet_out/galgame/train-00000.npz" in labels and not derived & labels
    assert not any(f"{LR}/teacher_out/galgame/train" in f for f in labels), "box 1 pulls no teacher train labels"
    assert "pull_labels" in r.stdout
    r = box.helper("pull", **env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert box.on_disk() == derived, "the derived pull brings no label file"
    r = box.helper("pull_labels", **env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert box.on_disk() == derived | labels
    assert "pulled the labels" in r.stdout
    # a missing extra file refuses the plan (exit 3), as a missing student file does
    (box.remote_dir / SIDECAR).unlink()
    r = box.helper("plan", **env)
    assert r.returncode == 3 and f"no {SIDECAR}" in r.stderr, r.stdout + r.stderr


def test_full_plan_checks_the_scratch_repo_with_the_box_token(box):
    """A box with timed states refuses (exit 3, not retried) without KITSUNE_SCRATCH_REPO, and the box token must read
    and write the scratch repo (auth_check, as for the output repo) before anything is pulled; a box without timed
    states needs none."""
    env = full_box(box)
    r = box.helper("plan", **env)
    assert r.returncode == 0, r.stdout + r.stderr
    checks = [c[1:] for c in box.calls() if c[0] == "auth_check"]
    assert checks == [["u/runs", "model", False], ["u/runs", "model", True], ["u/scratch", "model", False],
                      ["u/scratch", "model", True]]
    r = box.helper("plan", **dict(env, DENY_U_SCRATCH="403"))
    assert r.returncode == 3 and "HF_TOKEN cannot write u/scratch" in r.stderr and "(not retried)" in r.stderr
    assert "vast/README.md, full-data runs" in r.stderr
    r = box.helper("plan", **dict(env, DENY_U_SCRATCH="503"))
    assert r.returncode not in (0, 3) and "Hub error checking u/scratch" in r.stderr, "a Hub error is retried"
    r = box.helper("plan", **dict(env, KITSUNE_SCRATCH_REPO=""))
    assert r.returncode == 3 and "KITSUNE_SCRATCH_REPO is not set" in r.stderr
    box.calls()
    env = full_box(box, timed=False)
    r = box.helper("plan", **dict(env, KITSUNE_SCRATCH_REPO=""))
    assert r.returncode == 0, r.stdout + r.stderr
    assert [c[1] for c in box.calls() if c[0] == "auth_check"] == ["u/runs", "u/runs"]


def test_full_plan_refuses_a_registry_it_cannot_load(box):
    env = full_box(box)
    (box.root / "configs" / "full" / "full-p01.json").unlink()
    r = box.helper("plan", **env)
    assert r.returncode == 3 and "box p01" in r.stderr and "(not retried)" in r.stderr, r.stdout + r.stderr


# ------------------------------------------------------------------------------------- the job-full phases


def test_the_full_phases_are_new_lines_and_the_shared_lines_stay_byte_identical():
    """Job full reuses the plan, pull_derived, both rebuild and the coverage lines exactly as they were; its own phases
    are separate lines in `if [ "${KITSUNE_JOB:-}" = "full" ]` blocks, in the contract's order, and the full
    check_students comes after the study one."""
    text = BOOTSTRAP.read_text(encoding="utf-8")
    shared = ['phase plan retry 3 timeout -k 30 10m "$PY" "$HELPER" plan',
              'phase pull_derived retry 3 timeout -k 30 30m "$PY" "$HELPER" pull',
              'phase rebuild_audio retry 3 timeout -k 60 60m "$PY" scripts/01_prepare_data.py --data '
              '"$KITSUNE_DIR/$DATA_ROOT" \\',
              'phase rebuild_audio retry 3 timeout -k 60 "${KITSUNE_REBUILD_TIMEOUT_MIN:-60}m" "$PY" \\',
              'phase coverage "$PY" "$HELPER" coverage']
    lines = [ln.strip() for ln in text.splitlines()]
    for s in shared:
        assert lines.count(s) == 1, s
    for prefix in ("phase plan", "phase pull_derived", "phase rebuild_audio", "phase coverage"):
        assert len([ln for ln in lines if ln.startswith(prefix)]) == (2 if prefix == "phase rebuild_audio" else 1)
    order = ["phase download_gate", "phase plan ", "phase pull_derived", "phase check_students \"$PY\" -m "
             "kitsune.study_queue", "phase resume_pull", "phase check_students \"$PY\" -m kitsune.fullrun",
             "phase pull_labels retry", "phase rebuild_audio", "phase pull_labels_wait", "phase coverage"]
    at = [text.index(o) for o in order]
    assert at == sorted(at), [o for o, _ in sorted(zip(order, at), key=lambda x: x[1])]
    body = text.split("PYEOF\n")[-1]
    for name in ("download_gate", "resume_pull", "check_students \"$PY\" -m kitsune.fullrun", "pull_labels retry",
                 "pull_labels_wait"):
        i = body.index(f"phase {name}")
        assert 'if [ "${KITSUNE_JOB:-}" = "full" ]' in body[max(0, body.rfind("\nif ", 0, i)):i], name
    assert 'retry 3 timeout -k 30 "${KITSUNE_PULL_TIMEOUT_MIN:-30}m" "$PY" "$HELPER" pull_labels &' in text
    assert 'retry 3 timeout -k 30 60m "$PY" -m kitsune.full_queue resume-pull --box "$KITSUNE_BOX"' in text
    assert 'retry 2 timeout -k 30 30m "$PY" -m kitsune.netgate --out "$STATE/download_gate.json"' in text


FAKE_PY = r'''#!/bin/bash
# a stand-in interpreter: records every call (the helper path as HELPER) and plays each step
log="$FAKE_CALLS"
case "$1" in
    -c) exec "$REAL_PY" "$@" ;;
esac
line="$*"
case "$1" in scripts/*) ;; *.py) line="HELPER ${*:2}" ;; esac
[ "$1" = scripts/01_prepare_data.py ] && line="$line rebuild_timeout=${KITSUNE_REBUILD_TIMEOUT_MIN:-unset}"
[ "${2:-}" = pull_labels ] && line="$line pull_timeout=${KITSUNE_PULL_TIMEOUT_MIN:-unset}"
[ "${2:-}" = coverage ] && line="$line labels_done=$([ -e "$STATE/labels_done" ] && echo 1 || echo 0)"
echo "$line" >> "$log"
case "$*" in
    "-m kitsune.netgate --out"*) printf '{"verdict": "pass"}\n' > "$3"; exit "${FAKE_GATE_RC:-0}" ;;
    "-m kitsune.netgate --timeouts"*) echo "${FAKE_TIMEOUTS:-41 400}"; exit 0 ;;
    "-m kitsune.full_queue resume-pull"*) exit "${FAKE_RESUME_RC:-0}" ;;
    "-m kitsune.fullrun check-students"*) exit 0 ;;
    scripts/01_prepare_data.py*) command sleep "${FAKE_REBUILD_S:-0}"
                                 [ -z "${FAKE_RM_HELPER:-}" ] || rm -f "$STATE/bootstrap_helper.py"
                                 exit "${FAKE_REBUILD_RC:-0}" ;;
esac
case "${2:-}" in
    plan) printf '{"data_root": "data", "rebuild": ["galgame"], "extent": %s}\n' "${FAKE_EXTENT:-true}" \
              > "$STATE/bootstrap_plan.json"; exit 0 ;;
    pull) exit 0 ;;
    pull_labels) echo $$ > "$STATE/labels_pid"; command sleep "${FAKE_LABELS_S:-1}"
                 [ "${FAKE_LABELS_RC:-0}" = 0 ] && touch "$STATE/labels_done"; exit "${FAKE_LABELS_RC:-0}" ;;
    coverage) exit 0 ;;
esac
echo "unexpected call: $*" >&2
exit 97
'''


def run_bootstrap(tmp_path, **env_extra):
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    kdir, state = tmp_path / "box", tmp_path / "state"
    (kdir / "configs").mkdir(parents=True, exist_ok=True)
    (kdir / "configs" / "c.json").write_text("{}", encoding="utf-8")
    state.mkdir(exist_ok=True)
    (tmp_path / "tmp").mkdir(exist_ok=True)
    py = tmp_path / "fakepy"
    py.write_text(FAKE_PY, encoding="utf-8", newline="\n")
    calls = tmp_path / "calls.txt"
    calls.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("KITSUNE_")}
    env.update(KITSUNE_DIR=kdir.as_posix(), KITSUNE_STATE=state.as_posix(), KITSUNE_CONFIG="configs/c.json",
               KITSUNE_PY=py.as_posix(), KITSUNE_DATA_REPO="u/data", KITSUNE_OUT_REPO="u/runs", HF_TOKEN="hf_fake",
               FAKE_CALLS=calls.as_posix(), REAL_PY=Path(sys.executable).as_posix(), TMPDIR=(tmp_path / "tmp").as_posix(),
               KITSUNE_JOB="full", KITSUNE_BOX="p01", KITSUNE_GATE_BYTES="571200000000")
    env["BASH_FUNC_sleep%%"] = "() { command sleep 0.05; }"  # retry's minutes and the toucher's 60 s
    env.update({k: v for k, v in env_extra.items() if v is not None})
    for k, v in env_extra.items():
        if v is None:
            env.pop(k, None)
    r = subprocess.run([bash, str(BOOTSTRAP)], capture_output=True, text=True, env=env, timeout=180)
    got = calls.read_text(encoding="utf-8").splitlines() if calls.exists() else []
    return r, got, state


def test_a_full_bootstrap_runs_its_phases_in_order(tmp_path):
    """The whole job-full bootstrap under Git Bash with a fake interpreter: the gate first, its timeouts exported (the
    label pull's and the rebuild's), the plan, the derived pull, resume_pull only with KITSUNE_RESUME=1, the registry's
    student check, the label pull in the background during the rebuild, and the coverage check only once the labels
    are down. Every phase is timed, and train_hb is fresh."""
    r, calls, state = run_bootstrap(tmp_path, KITSUNE_RESUME="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "bootstrap complete" in r.stdout
    seq = [c.split(" rebuild_timeout")[0].split(" pull_timeout")[0].split(" labels_done")[0] for c in calls]
    labels = next(c for c in calls if c.startswith("HELPER pull_labels"))
    seq.remove(labels.split(" pull_timeout")[0])
    s = state.as_posix()
    assert seq == [f"-m kitsune.netgate --out {s}/download_gate.json --dir {s}/netgate",
                   f"-m kitsune.netgate --timeouts {s}/download_gate.json",
                   "HELPER plan", "HELPER pull",
                   f"-m kitsune.full_queue resume-pull --box p01 --root {(tmp_path / 'box').as_posix()}",
                   f"-m kitsune.fullrun check-students --box p01 --root {(tmp_path / 'box').as_posix()}",
                   f"scripts/01_prepare_data.py --data {(tmp_path / 'box').as_posix()}/data --extent-config "
                   "configs/c.json", "HELPER coverage"], calls
    assert labels.endswith("pull_timeout=41")
    assert next(c for c in calls if c.startswith("scripts/01")).endswith("rebuild_timeout=400")
    assert calls[-1] == "HELPER coverage labels_done=1", "coverage runs after the labels are down"
    timed = [json.loads(ln)["phase"] for ln in (state / "bootstrap_timings.jsonl").read_text().splitlines()]
    assert set(timed) == {"download_gate", "plan", "pull_derived", "resume_pull", "check_students", "pull_labels",
                          "rebuild_audio", "pull_labels_wait", "coverage"}
    assert (state / "train_hb").exists()
    r, calls, _ = run_bootstrap(tmp_path / "fresh")  # a fresh box: no resume_pull
    assert r.returncode == 0 and not any("resume-pull" in c for c in calls), calls


def test_a_helper_lost_during_the_rebuild_is_written_again(tmp_path):
    """2026-10-01: a box lost its /tmp helper between the label pull and the coverage check and destroyed itself. The
    helper now lives in the box state, and every phase writes it again (and says so) when it is gone: here the rebuild
    deletes it, and the coverage check still runs it."""
    r, calls, state = run_bootstrap(tmp_path, FAKE_RM_HELPER="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "bootstrap_helper.py was missing; writing it again" in r.stdout
    assert calls[-1].startswith("HELPER coverage"), calls
    assert not list((tmp_path / "tmp").iterdir()), "nothing of the bootstrap's goes to TMPDIR"
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'HELPER="$STATE/bootstrap_helper.py"' in text and "mktemp" not in text.split("PYEOF")[0]
    onstart = (BOOTSTRAP.parent / "onstart.sh").read_text(encoding="utf-8")
    assert 'export TMPDIR="${KITSUNE_TMPDIR:-/workspace/tmp}"' in onstart and 'mkdir -p "$TMPDIR"' in onstart


@pytest.mark.parametrize("what", ["gate", "resume"])
def test_a_refusal_in_a_full_phase_is_not_retried(tmp_path, what):
    """Exit 3 of the gate (a slow host) or of resume_pull (a refusal) returns at once and fails the bootstrap before
    anything is pulled or rebuilt; a slow gate never reaches the plan."""
    env = dict(FAKE_GATE_RC="3") if what == "gate" else dict(FAKE_RESUME_RC="3", KITSUNE_RESUME="1")
    r, calls, _ = run_bootstrap(tmp_path, **env)
    assert r.returncode == 3, r.stdout + r.stderr
    key = "kitsune.netgate --out" if what == "gate" else "resume-pull"
    assert len([c for c in calls if key in c]) == 1 and "not retrying" in r.stdout, calls
    assert not any(c.startswith("scripts/01") for c in calls) and not any("pull_labels" in c for c in calls)
    if what == "gate":
        assert not any(c.startswith("HELPER") for c in calls)


def test_a_failed_label_pull_fails_the_bootstrap_after_the_rebuild(tmp_path):
    r, calls, _ = run_bootstrap(tmp_path, FAKE_LABELS_RC="1")
    assert r.returncode != 0, r.stdout + r.stderr
    assert len([c for c in calls if c.startswith("HELPER pull_labels")]) == 3, calls  # retried like the pull
    assert any(c.startswith("scripts/01") for c in calls) and not any("coverage" in c for c in calls)
    assert "phase pull_labels_wait ..." in r.stdout


def budget_s(tries: int, minutes: int) -> int:
    """bootstrap's phase_budget_s: the longest `retry <tries> timeout -k <=60 <minutes>m` runs (each attempt + 60 s of
    kill grace, retry's pauses of 60, 120, ... s) plus 10 min."""
    return tries * (minutes * 60 + 60) + sum(60 * i for i in range(1, tries)) + 600


def hb_bounds(stdout: str) -> dict:
    return {m.group(1): int(m.group(2))
            for m in re.finditer(r"phase (\S+) keeps train_hb fresh for at most (\d+) s", stdout)}


def test_each_phase_keeps_train_hb_fresh_for_its_own_worst_case(tmp_path):
    """The rebuild may legitimately run three attempts of the gate's KITSUNE_REBUILD_TIMEOUT_MIN: at the gate's floor
    rate the full extent's 481 min per attempt make ~24 h, past the 12 h default. Its train_hb toucher lasts exactly its
    worst case (so is the label pull's, and pull_labels_wait's), so the watchdog stops a box only once the rebuild's
    own timeouts have run out; the other phases keep KITSUNE_PHASE_HB_MAX_S (default 12 h)."""
    r, _, _ = run_bootstrap(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    got = hb_bounds(r.stdout)
    assert got == {"download_gate": 43200, "plan": 43200, "pull_derived": 43200, "check_students": 43200,
                   "pull_labels": budget_s(3, 41), "rebuild_audio": budget_s(3, 400),
                   "pull_labels_wait": budget_s(3, 41), "coverage": 43200}, got
    r, _, _ = run_bootstrap(tmp_path / "floor", FAKE_TIMEOUTS="77 481", KITSUNE_PHASE_HB_MAX_S="600")
    assert r.returncode == 0, r.stdout + r.stderr
    got = hb_bounds(r.stdout)
    assert got["rebuild_audio"] == budget_s(3, 481) == 87540 > 43200 and got["pull_labels"] == budget_s(3, 77)
    assert got["plan"] == got["coverage"] == 600  # the default is the operator's to set
    r, _, _ = run_bootstrap(tmp_path / "short", FAKE_TIMEOUTS="30 45")
    assert hb_bounds(r.stdout)["rebuild_audio"] == budget_s(3, 60), "never below the non-extent line's 60 min"


def test_a_failed_bootstrap_stops_the_background_label_pull_and_its_toucher(tmp_path):
    """The rebuild fails while the labels are still coming down: bootstrap's exit stops the pull's whole tree, its
    subshell, the timeout and the pull under it, and the phase's toucher, so nothing keeps train_hb fresh or holds
    onstart's supervise.lock after bootstrap (a subshell killed alone leaves its children running)."""
    bash = find_bash()
    r, calls, state = run_bootstrap(tmp_path, FAKE_REBUILD_RC="1", FAKE_LABELS_S="60")
    assert r.returncode == 1, r.stdout + r.stderr
    assert len([c for c in calls if c.startswith("scripts/01")]) == 3 and "HELPER coverage" not in calls, calls
    assert any(c.startswith("HELPER pull_labels") for c in calls) and "phase pull_labels_wait" not in r.stdout
    hb = state / "train_hb"
    before = hb.stat().st_mtime_ns
    pid = (state / "labels_pid").read_text().strip()
    alive = None
    for _ in range(50):  # SIGTERM reaches the pull through timeout within moments
        alive = subprocess.run([bash, "-c", f"kill -0 {pid}"], capture_output=True).returncode == 0
        if not alive:
            break
        time.sleep(0.1)
    assert not alive, "the label pull outlived bootstrap"
    time.sleep(1.0)  # 20 of the fake toucher's 0.05 s beats
    assert hb.stat().st_mtime_ns == before, "a train_hb toucher outlived bootstrap"
    assert not (state / "labels_done").exists()


def test_a_toucher_stops_at_its_bound(tmp_path):
    """beat_train_hb <max_s> touches train_hb until max_s has passed, then returns by itself."""
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    text = BOOTSTRAP.read_text(encoding="utf-8")
    func = re.search(r"^beat_train_hb\(\) \{.*?^\}\n", text, re.M | re.S).group(0)
    script = tmp_path / "hb.sh"
    script.write_text("\n".join(["set -euo pipefail", f'STATE="{tmp_path.as_posix()}"',
                                 'sleep() { command sleep 0.2; }', func, "beat_train_hb 2", 'echo "returned"', ""]),
                      encoding="utf-8", newline="\n")
    t0 = time.monotonic()
    r = subprocess.run([bash, str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "returned" in r.stdout, r.stdout + r.stderr
    assert 1.0 < time.monotonic() - t0 < 30 and (tmp_path / "train_hb").exists()  # date +%s: whole seconds


def test_other_jobs_run_no_full_phase(tmp_path):
    """Without KITSUNE_JOB=full nothing of the above runs, whatever gate env is set: no gate, no train_hb, no label
    pull; the plan, pull, rebuild and coverage as before."""
    r, calls, state = run_bootstrap(tmp_path, KITSUNE_JOB=None, KITSUNE_RESUME="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert [c.split(" ")[0] + " " + c.split(" ")[1] for c in calls] == [
        "HELPER plan", "HELPER pull", "scripts/01_prepare_data.py --data", "HELPER coverage"], calls
    assert not (state / "train_hb").exists()


# ------------------------------------------------------------------------------------- a chain box (addendum E)

P03_STUDENT = "students/study/p03"
SB_SELECTION = f"{LR}/selections/sb.parquet"  # smoke-b's own (study) selection, next to stage 1's rebuild


def chain_box(box) -> dict:
    """A chain p01-chain on the toy extent, written into the box's checkout: stage 1 = full-smoke (a CTC train item,
    student p03; its data config the capped extent, both label roots: the rebuild) + smoke-b (its own study selection
    on the same extent: its selection files join stage 1's pull); stage 2 = p01 (student p01) on the uncapped extent.
    The remote gains every student, extra file and dir."""
    from kitsune import fullrun

    stub = box.root.parent / "stub" / "huggingface_hub" / "__init__.py"
    stub.write_text(FULL_STUB, encoding="utf-8")
    folder = box.root / "configs" / "full"
    folder.mkdir(parents=True, exist_ok=True)
    base = {k: v for k, v in make_cfg().items() if k not in ("student", "run_name")}
    s1 = dict(base, pull_parakeet=True)
    sb = dict(s1, selection=SB_SELECTION, selection_recipe=dict(base["selection_recipe"], study={"draw_audio_s": 1}))
    s2 = dict({k: v for k, v in make_cfg(inputs={}).items() if k not in ("student", "run_name")}, family="ctc",
              selection=f"{LR}/selections/full.parquet")
    configs = {"data-s1": s1, "data-sb": sb, "data-s2": s2,
               "smoke-p03": dict(s1, run_name="smoke-p03", family="ctc", student=P03_STUDENT),
               "sb-eval": dict(sb, run_name="sb-eval", student=P03_STUDENT),
               "full-p01": dict(s2, run_name="full-p01", student=P01_STUDENT)}
    for name, cfg in configs.items():
        (folder / f"{name}.json").write_text(json.dumps(cfg), encoding="utf-8")

    def box_of(dc, items, **kw):
        return dict({"gpus": 1, "data_config": f"configs/full/{dc}.json", "est_hours": 1.0, "max_hours": 9,
                     "max_dph": 1.0, "deadline_reserve_min": 20, "watchdog": {"orphan_s": 3600, "action": "stop"},
                     "timed_states": False, "items": items}, **kw)

    reg = {"version": 1, "boxes": {
        "full-smoke": box_of("data-s1", [{"name": "smoke-p03", "kind": "train", "config": "configs/full/smoke-p03.json",
                                          "study_run": "study-p03", "family": "ctc", "max_hours": 0.75}],
                             smoke=True, watchdog={"orphan_s": 600, "action": "alert"},
                             extra_files=[SIDECAR, MANIFEST]),
        "smoke-b": box_of("data-sb", [{"name": "stores-eval", "kind": "stores", "config": "configs/full/sb-eval.json",
                                       "eval_only": True}], smoke=True, gate=False, max_hours=3,
                          extra_dirs=[PARAKEET_MODEL]),
        "p01": box_of("data-s2", [{"name": "full-p01", "kind": "train", "config": "configs/full/full-p01.json",
                                   "study_run": "study-p01", "family": "ctc", "max_hours": 13.56}],
                      est_hours=19.5, max_hours=22, extra_files=[MANIFEST]),
        "p01-chain": {"est_hours": 25.2, "max_hours": 35, "max_dph": 1.0, "extra_gb": 120, "chain": [
            {"parts": ["full-smoke", "smoke-b"], "gate_box": "full-smoke", "max_hours": 10.5},
            {"parts": ["p01"]}]}}}
    (folder / "boxes.json").write_text(json.dumps(reg), encoding="utf-8")
    fullrun.load_registry(reg, root=box.root)  # a valid chain (the file rules included)
    for f in [*(f"{s}/{n}" for s in (P01_STUDENT, P03_STUDENT) for n in (*STUDENT_FILES, "MODEL_CARD.md")), SIDECAR,
              MANIFEST, SB_SELECTION, f"{LR}/selections/sb.json", f"{PARAKEET_MODEL}/config.json"]:
        (box.remote_dir / f).parent.mkdir(parents=True, exist_ok=True)
        (box.remote_dir / f).write_bytes(b"x")
    return dict(KITSUNE_JOB="full", KITSUNE_BOX="p01-chain", KITSUNE_FULL_REGISTRY="", KITSUNE_SCRATCH_REPO="")


def test_a_chains_plan_serves_its_stage(box):
    """KITSUNE_BOX=p01-chain: stage 1 pulls both selections (its rebuild's and smoke-b's study selection with its
    sidecar and manifest) and stage 1's students only; stage 2 box 1's student and the uncapped extent's labels; a
    KITSUNE_CONFIG that is not the stage's rebuild is refused (exit 3, not retried)."""
    env = chain_box(box)
    r = box.helper("plan", **env, KITSUNE_CHAIN_STAGE="1", CONFIG="configs/full/data-s1.json")
    assert r.returncode == 0, r.stdout + r.stderr
    plan = json.loads((box.state_dir / "bootstrap_plan.json").read_text(encoding="utf-8"))
    files = set(plan["files"])
    assert {f"{LR}/selections/sub.parquet", SB_SELECTION, f"{LR}/selections/sb.json", SIDECAR, MANIFEST} <= files
    assert {f"{P03_STUDENT}/{n}" for n in (*STUDENT_FILES, "MODEL_CARD.md")} <= files
    assert not any(f.startswith(P01_STUDENT) for f in files), "box 1's student is stage 2's"
    assert f"{PARAKEET_MODEL}/config.json" in files and plan["rebuild"] == ["galgame", "eval_jsut"]
    stage1_labels = set(plan["labels"]["files"])
    assert f"{LR}/parakeet_out/galgame/train-00001.npz" not in stage1_labels, "stage 1: the capped extent"
    r = box.helper("plan", **env, KITSUNE_CHAIN_STAGE="2", CONFIG="configs/full/data-s2.json")
    assert r.returncode == 0, r.stdout + r.stderr
    plan = json.loads((box.state_dir / "bootstrap_plan.json").read_text(encoding="utf-8"))
    files = set(plan["files"])
    assert {f"{P01_STUDENT}/{n}" for n in (*STUDENT_FILES, "MODEL_CARD.md")} <= files
    assert not any(f.startswith(P03_STUDENT) for f in files) and SB_SELECTION not in files
    assert f"{LR}/parakeet_out/galgame/train-00001.npz" in plan["labels"]["files"], "stage 2: the uncapped extent"
    for stage, config in (("2", "configs/full/data-s1.json"), ("1", "configs/full/data-s2.json")):
        r = box.helper("plan", **env, KITSUNE_CHAIN_STAGE=stage, CONFIG=config)
        assert r.returncode == 3 and f"chain stage {stage} rebuilds" in r.stderr and "(not retried)" in r.stderr, (
            stage, r.stdout + r.stderr)


def test_a_chains_phase_timings_carry_its_stage(tmp_path):
    """bootstrap_timings.jsonl: every phase of a chain stage carries "stage" (the controller's stage-2 bootstrap
    appends to the boot's file); no other box's records change."""
    r, calls, state = run_bootstrap(tmp_path, KITSUNE_CHAIN_STAGE="2")
    assert r.returncode == 0, r.stdout + r.stderr
    recs = [json.loads(ln) for ln in (state / "bootstrap_timings.jsonl").read_text().splitlines()]
    assert recs and all(x["stage"] == 2 and set(x) == {"phase", "seconds", "end", "stage"} for x in recs), recs
    r, calls, state = run_bootstrap(tmp_path / "plain")
    recs = [json.loads(ln) for ln in (state / "bootstrap_timings.jsonl").read_text().splitlines()]
    assert recs and all(set(x) == {"phase", "seconds", "end"} for x in recs), recs


# ---------------------------------------- the EXIT trap and onstart's supervise.lock (box p01-chain, 2026-10-01)

def bootstrap_funcs(*names: str) -> str:
    """The named top-level functions of vast/bootstrap.sh, verbatim."""
    text = BOOTSTRAP.read_text(encoding="utf-8")
    return "".join(re.search(rf"^{n}\(\) \{{.*?^\}}\n", text, re.M | re.S).group(0) for n in names)


def bash_script(path: Path, *lines: str) -> Path:
    path.write_text("\n".join([*lines, ""]), encoding="utf-8", newline="\n")
    return path


def test_the_exit_trap_acts_in_bootstraps_own_process_only(tmp_path):
    """A background subshell that SIGTERM reaches before bash has reset its traps runs the parent's EXIT trap (Linux
    bash 5.2; the Linux-only test below shows it): phase pull_labels_wait's toucher, killed right after its fork,
    deleted the helper before the coverage check on two boxes. on_exit acts only in bootstrap's own process: a child
    running it keeps the helper and the label pull, and bootstrap's own exit still removes both."""
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    assert re.search(r"^trap on_exit EXIT$", BOOTSTRAP.read_text(encoding="utf-8"), re.M)
    helper = tmp_path / "bootstrap_helper.py"
    helper.write_text("x", encoding="utf-8")
    script = bash_script(
        tmp_path / "trap.sh", "set -euo pipefail", f'HELPER="{helper.as_posix()}"',
        bootstrap_funcs("stop_label_pull", "on_exit"), "trap on_exit EXIT",
        "command sleep 30 &  # stands in for the label pull", "LABELS_PID=$!",
        f'echo "$LABELS_PID" > "{tmp_path.as_posix()}/pull"',
        "( on_exit ) &  # what bash's fatal-signal handler runs in a child killed before its traps were reset",
        'wait "$!"',
        '[ -e "$HELPER" ] && echo "child kept the helper"',
        'kill -0 "$LABELS_PID" && echo "child kept the pull"')
    r = subprocess.run([bash, str(script)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "child kept the helper" in r.stdout and "child kept the pull" in r.stdout, r.stdout + r.stderr
    assert not helper.exists(), "bootstrap's own exit removes the helper"
    pid = (tmp_path / "pull").read_text().strip()
    alive = True
    for _ in range(50):
        alive = subprocess.run([bash, "-c", f"kill -0 {pid}"], capture_output=True).returncode == 0
        if not alive:
            break
        time.sleep(0.1)
    assert not alive, "bootstrap's own exit stops the label pull"


def toucher_script(tmp_path: Path, func: str) -> Path:
    """beat_train_hb started and killed as phase does, with a `sleep` on PATH that notes its pid when the toucher calls
    it (`sleep 60`) and then sleeps 60 s in place (exec): the script says whether that sleep outlived its toucher. The
    killed sleep gets 20 s to go (a busy Git Bash took over 3 s), well short of the old toucher's 60 s sleep."""
    fake = tmp_path / "bin"
    fake.mkdir()
    bash_script(fake / "sleep", "#!/bin/bash", '[ "$1" != 60 ] || { echo $$ >> "$SLEEP_PIDS"; set -- 60; }',
                'exec "$REAL_SLEEP" "$@"')
    pids = tmp_path / "sleep_pids"
    return bash_script(
        tmp_path / "toucher.sh", "set -euo pipefail", 'export REAL_SLEEP="$(command -v sleep)"',
        f'fake="$(cd "{fake.as_posix()}" && pwd)"  # /d/... under Git Bash: PATH splits at the colon of D:/',
        f'export SLEEP_PIDS="{pids.as_posix()}" PATH="$fake:$PATH"', 'chmod +x "$fake/sleep"',
        f'STATE="{tmp_path.as_posix()}"', func,
        "beat_train_hb 100 &", "hb=$!",
        'for i in $(seq 200); do [ -s "$SLEEP_PIDS" ] && break; "$REAL_SLEEP" 0.05; done',
        'kill "$hb"; wait "$hb" || true', 'p=$(head -1 "$SLEEP_PIDS")',
        'for i in $(seq 400); do kill -0 "$p" 2>/dev/null || { echo "sleep gone"; exit 0; }; "$REAL_SLEEP" 0.05; done',
        'kill "$p"; echo "sleep outlived its toucher"')


def test_a_toucher_takes_its_sleep_with_it(tmp_path):
    """phase kills its train_hb toucher, not the toucher's child: a `sleep 60` left behind kept onstart's supervise.lock
    (fd 7) up to a minute after bootstrap, and supervise.py, started right then, refused to run. The toucher now sleeps
    in the background and kills that sleep when it is killed (the old toucher fails this under Git Bash and Linux)."""
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    script = toucher_script(tmp_path, bootstrap_funcs("beat_train_hb"))
    r = subprocess.run([bash, str(script)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "sleep gone" in r.stdout, r.stdout + r.stderr


def test_nothing_of_bootstraps_holds_onstarts_supervise_lock():
    """onstart's detached subshell holds supervise.lock (fd 7) while bootstrap runs and closes it right before it execs
    supervise.py: bootstrap runs with fd 7 closed, and bootstrap closes it itself before its first command (for a
    bootstrap started some other way), so no phase, toucher or label pull ever has it. supervise.py's bounded wait for
    the lock: test_infra."""
    text = BOOTSTRAP.read_text(encoding="utf-8")
    code = [ln for ln in text.split("<<'PYEOF'")[0].splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert code[:2] == ["set -euo pipefail", "exec 7>&-"], code[:3]
    onstart = (BOOTSTRAP.parent / "onstart.sh").read_text(encoding="utf-8")
    body = re.search(r"^\(\n(.*?)^\) < /dev/null &$", onstart, re.M | re.S).group(1)
    assert body.index('exec 7>"$KITSUNE_STATE/supervise.lock"') < body.index(
        'bash "$KITSUNE_DIR/vast/bootstrap.sh" 7>&-\n') < body.index("exec 7>&-  # handed over")


# Linux-only behaviour: the EXIT trap race needs Linux bash (Git Bash's slower fork does not show it). Run on Linux, or
# under WSL from Windows (KITSUNE_LINUX_BASH, e.g. "wsl.exe -e bash", or wsl.exe when it answers); skipped otherwise.

@functools.lru_cache(maxsize=None)
def linux_bash() -> tuple[str, ...] | None:
    """The command of a Linux bash with flock, timeout and python3: bash itself on Linux, else KITSUNE_LINUX_BASH (a
    WSL-style command: Windows drive X: is its /mnt/x), else WSL's when it answers. None: the Linux tests skip."""
    if sys.platform.startswith("linux"):
        cmd = ("bash",)
    elif os.environ.get("KITSUNE_LINUX_BASH"):
        cmd = tuple(shlex.split(os.environ["KITSUNE_LINUX_BASH"]))
    elif os.name == "nt" and shutil.which("wsl.exe"):
        cmd = ("wsl.exe", "-e", "bash")
    else:
        return None
    try:
        r = subprocess.run([*cmd, "-c", "uname -s; command -v flock timeout python3 >/dev/null && echo ok"],
                           capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return cmd if r.returncode == 0 and r.stdout.decode("utf-8", "replace").split() == ["Linux", "ok"] else None


def linux_path(p: Path) -> str:
    """p as the Linux bash sees it: a Windows drive path under /mnt/<drive>."""
    s = Path(p).resolve().as_posix()
    return f"/mnt/{s[0].lower()}{s[2:]}" if os.name == "nt" and re.match(r"^[A-Za-z]:/", s) else s


def run_linux(script: Path, *args: str, timeout: float = 300) -> subprocess.CompletedProcess:
    cmd = linux_bash()
    if cmd is None:
        pytest.skip("no Linux bash with flock, timeout and python3 (Linux, KITSUNE_LINUX_BASH or WSL)")
    return subprocess.run([*cmd, linux_path(script), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def test_linux_a_child_killed_at_once_never_runs_bootstraps_exit_trap(tmp_path):
    """The mechanism of the lost helper, under Linux bash: background subshells killed right after their fork (as
    pull_labels_wait killed its toucher when the label pull was long done). With the old EXIT trap (rm -f "$HELPER")
    some of them run it and delete the helper; with on_exit none does. Skipped when even the old trap shows no race
    on this machine (on_exit is checked first)."""
    helper = tmp_path / "bootstrap_helper.py"
    script = bash_script(
        tmp_path / "race.sh", "set -euo pipefail", f'HELPER="{linux_path(helper)}"',
        bootstrap_funcs("stop_label_pull", "on_exit"),
        """if [ "$1" = old ]; then trap 'rm -f "$HELPER"' EXIT; else trap on_exit EXIT; fi""",
        "f() { while :; do command sleep 2; done; }", "n=0", 'echo x > "$HELPER"',
        "for i in $(seq 200); do",
        '    f &', '    hb=$!', '    kill "$hb" 2>/dev/null || true', '    wait "$hb" 2>/dev/null || true',
        '    [ -e "$HELPER" ] || { n=$(( n + 1 )); echo x > "$HELPER"; }',
        "done", 'echo "children that ran the EXIT trap: $n"')
    got = {}
    for which in ("new", "old"):
        r = run_linux(script, which)
        m = re.search(r"children that ran the EXIT trap: (\d+)", r.stdout)
        assert r.returncode == 0 and m, r.stdout + r.stderr
        got[which] = int(m.group(1))
    assert got["new"] == 0, got
    if got["old"] == 0:
        pytest.skip(f"the old EXIT trap showed no race on this machine ({got}); on_exit was checked")


def test_linux_onstarts_handover_finds_the_lock_free_and_the_helper_intact(tmp_path):
    """The whole job-full bootstrap (the fake interpreter above, the labels down before the rebuild ends, real 60 s
    toucher sleeps) under Linux bash in an onstart-like wrapper that holds supervise.lock on fd 7 and passes it to
    bootstrap (onstart itself no longer does): right after bootstrap the lock is free, and only bootstrap's own process
    removed the helper, never before the coverage check. Before the fix the touchers' sleeps held the lock on every
    run, and about half the runs lost the helper. Then supervise.wait_for_lock gets a lock that a leftover holds for
    8 s, with real flock (8 s: python3 under WSL may take seconds to start and import supervise from /mnt/d, and the
    probe must find the lock still held)."""
    rounds = 3
    py = tmp_path / "fakepy"
    py.write_text(FAKE_PY, encoding="utf-8", newline="\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    bash_script(bin_dir / "rm", "#!/bin/bash", 'echo "$PPID $*" >> "$RM_LOG"', 'exec /bin/rm "$@"')
    env = dict(KITSUNE_CONFIG="configs/c.json", KITSUNE_PY=linux_path(py), KITSUNE_DATA_REPO="u/data",
               KITSUNE_OUT_REPO="u/runs", HF_TOKEN="hf_fake", REAL_PY="python3", KITSUNE_JOB="full", KITSUNE_BOX="p01",
               KITSUNE_GATE_BYTES="571200000000", FAKE_LABELS_S="1", FAKE_REBUILD_S="2")
    lock = tmp_path / "held.lock"
    probe = ("import sys, time, supervise; from pathlib import Path; supervise.LOCK_POLL_S = 0.25; t0 = time.time(); "
             "lock = supervise.wait_for_lock(Path(sys.argv[1])); "
             "print('wait_for_lock:', 'got' if lock else 'none', f'after {time.time() - t0:.1f} s')")
    script = bash_script(
        tmp_path / "handover.sh", "set -uo pipefail", f'chmod +x "{linux_path(py)}" "{linux_path(bin_dir)}/rm"',
        f'export PATH="{linux_path(bin_dir)}:$PATH"', *(f"export {k}={shlex.quote(v)}" for k, v in env.items()),
        f"for i in $(seq {rounds}); do",
        f'    D="{linux_path(tmp_path)}/round$i"', '    mkdir -p "$D/box/configs" "$D/state" "$D/tmp"',
        """    echo '{}' > "$D/box/configs/c.json\"""",
        '    export KITSUNE_DIR="$D/box" KITSUNE_STATE="$D/state" TMPDIR="$D/tmp" FAKE_CALLS="$D/calls.txt" '
        'RM_LOG="$D/rm.log"',
        "    (",
        '        exec 7>"$KITSUNE_STATE/supervise.lock"',
        '        flock -n 7 || { echo "round $i: lock busy"; exit 1; }',
        f'        bash "{linux_path(BOOTSTRAP)}" > "$D/boot.log" 2>&1 &',
        '        b=$!', '        echo "$b" > "$D/boot.pid"', '        wait "$b"', '        rc=$?',
        '        exec 7>&-',
        '        if flock -n "$KITSUNE_STATE/supervise.lock" true; then l=free; else l=held; fi',
        '        echo "round $i: bootstrap exit $rc, lock $l"',
        "    )",
        "done",
        f'flock "{linux_path(lock)}" sleep 8 &', "command sleep 0.5",
        f'cd "{linux_path(BOOTSTRAP.parent)}"', f'python3 -c {shlex.quote(probe)} "{linux_path(lock)}"')
    r = run_linux(script, timeout=600)
    out = r.stdout + r.stderr
    for i in range(1, rounds + 1):
        d = tmp_path / f"round{i}"
        assert f"round {i}: bootstrap exit 0, lock free" in r.stdout, out
        boot = (d / "boot.log").read_text(encoding="utf-8")
        assert "bootstrap complete" in boot and "was missing" not in boot, boot
        pid = (d / "boot.pid").read_text().strip()
        rms = (d / "rm.log").read_text().splitlines()
        assert [ln.split()[0] for ln in rms if "bootstrap_helper.py" in ln] == [pid], (rms, pid)
    m = re.search(r"wait_for_lock: got after ([\d.]+) s", r.stdout)
    assert m and 1.5 < float(m.group(1)) < 60, out
    assert "is held by another process; waiting up to 180 s" in r.stdout, out


def test_linux_onstarts_exit_trap_unlocks_in_onstart_only(tmp_path):
    """onstart.sh's EXIT trap unlocks onstart.lock (fd 8, shared with every child it starts): a child killed right after
    its fork that ran it would unlock the running onstart's lock. Under Linux bash its trap line, verbatim: no such
    child releases the lock, and onstart's own exit (normal, or by SIGTERM) still does."""
    onstart = (BOOTSTRAP.parent / "onstart.sh").read_text(encoding="utf-8")
    trap = re.search(r"^trap '.*flock -u 8.*' EXIT$", onstart, re.M).group(0)
    lock = linux_path(tmp_path / "onstart.lock")
    script = bash_script(
        tmp_path / "onstart_trap.sh", "set -euo pipefail", "set -o errtrace", f'exec 8>"{lock}"', "flock -n 8", trap,
        'case "$1" in',
        "    race) n=0",
        "          for i in $(seq 200); do",
        "              ( command sleep 3 ) &", "              k=$!", '              kill "$k" 2>/dev/null || true',
        '              wait "$k" 2>/dev/null || true',
        f'              if flock -n "{lock}" true; then n=$(( n + 1 )); fi',
        "          done",
        '          echo "children that released onstart.lock: $n" ;;',
        "    term) command sleep 30 & wait $! ;;",
        "esac")
    r = run_linux(script, "race")
    assert "children that released onstart.lock: 0" in r.stdout, r.stdout + r.stderr
    driver = bash_script(
        tmp_path / "driver.sh", f'S="{linux_path(script)}"', 'command sleep 3.5  # the race\'s leftover sleeps hold fd 8',
        'bash "$S" normal', f'flock -n "{lock}" true && echo "normal exit: unlocked"',
        'bash "$S" term & m=$!', "command sleep 1", 'kill -TERM "$m"', 'wait "$m" || true',
        f'flock -n "{lock}" true && echo "SIGTERM exit: unlocked"')
    r = run_linux(driver)
    assert "normal exit: unlocked" in r.stdout and "SIGTERM exit: unlocked" in r.stdout, r.stdout + r.stderr


def toucher_term_script(path: Path, trap: str) -> Path:
    """The toucher of the test below, for a given TERM trap line, under Linux bash: a background subshell under set -e
    starts `sleep 300 &`, records its pid only in the test's own file (never in `s`), and TERMs itself at once - the
    race of beat_train_hb, a TERM between `sleep 60 &` and `s=$!` -, so only the trap can take that sleep along
    (with-sleep); or it first kills and reaps the sleep itself, so the trap meets no job and must still exit 0 (none).
    The checks give the killed sleep 400 polls (20 s) to go, well short of its 300 s."""
    d = linux_path(path.parent)
    return bash_script(
        path, "set -euo pipefail",
        "command sleep 300 & pull=$!  # bootstrap's own job (the label pull): not the toucher's",
        "toucher() {", f"    {trap}", '    command sleep 300 & echo "$!" > "$1"',
        '    [ "$2" = with-sleep ] || { kill "$(cat "$1")"; wait || true; }', "    kill -TERM $BASHPID",
        "    command sleep 300  # never reached", "}",
        f'for m in with-sleep none; do f="{d}/$m"',
        '    toucher "$f" "$m" & t=$!; wait "$t" && echo "$m: toucher exit 0" || echo "$m: toucher exit $?"',
        '    for i in $(seq 400); do kill -0 "$(cat "$f")" 2>/dev/null || break; command sleep 0.05; done',
        '    if kill -0 "$(cat "$f")" 2>/dev/null; then echo "$m: sleep left"; kill "$(cat "$f")"',
        '    else echo "$m: sleep gone"; fi',
        "done", 'kill -0 "$pull" && echo "pull alive"; kill "$pull"')


def test_linux_a_toucher_killed_before_it_noted_its_sleep_still_kills_it(tmp_path):
    """Bash may run the toucher's TERM trap after `sleep 60 &` forked and before `s=$!` is set: the trap kills the
    toucher's own jobs, not a remembered pid, so that sleep goes too; with no job left it still exits 0 (set -e would
    end it at the failing kill). beat_train_hb's trap line, verbatim (toucher_term_script); PR #34's earlier trap,
    which killed the remembered "$s", leaves the sleep behind. Under Linux bash only - the boxes' bash: Git Bash's
    emulated signals lost the trap's kill (or the TERM's interruption of a `wait`) in up to 33 of 36 runs at 12 copies
    at once on a busy laptop (2026-10-02, whichever way the test sent the TERM), so there it tested MSYS's signal
    emulation, not the trap."""
    trap = re.search(r"^ *(trap '.*' TERM)$", bootstrap_funcs("beat_train_hb"), re.M).group(1)
    r = run_linux(toucher_term_script(tmp_path / "trap_term.sh", trap))
    for line in ("with-sleep: toucher exit 0", "with-sleep: sleep gone", "none: toucher exit 0", "none: sleep gone",
                 "pull alive"):
        assert line in r.stdout, r.stdout + r.stderr
    (tmp_path / "old").mkdir()
    old = "trap '[ -z \"${s:-}\" ] || kill \"$s\" 2>/dev/null; exit 0' TERM"  # 7f87a7f's: $s is not set yet
    r = run_linux(toucher_term_script(tmp_path / "old" / "trap_term.sh", old))
    assert "with-sleep: sleep left" in r.stdout, r.stdout + r.stderr
