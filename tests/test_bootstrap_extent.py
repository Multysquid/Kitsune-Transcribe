"""vast/bootstrap.sh in extent mode: the A100 pulls exactly a config's extent of the label box's labels, rebuilds its
audio with `01 --extent-config` and checks the join exactly (plan §12.2, §3.5). The viability path is unchanged.

CPU only, no network: huggingface_hub is replaced by a stub that serves a fake data repo from a local directory (the
real filter_repo_objects matcher is loaded from the installed package, as snapshot_download filters with it); the
helper heredoc is extracted and run with the real kitsune.extent; the rebuild dispatch runs under Git Bash with fake
phase/retry (skipped without bash)."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = ROOT / "vast" / "bootstrap.sh"
sys.path.insert(0, str(ROOT))
from kitsune.store import ShardWriter, ids_sha256  # noqa: E402

STUDENT = "students/s"
STUDENT_FILES = ("config.json", "model.safetensors", "processor_config.json", "tokenizer.json", "tokenizer_config.json")
LR = "labels/full"


def find_bash() -> str | None:
    for cand in (shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if cand and Path(cand).exists() and "system32" not in cand.lower():  # System32\bash.exe is WSL, not Git Bash
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
import json, os, shutil
from pathlib import Path
REMOTE = Path(os.environ["FAKE_REMOTE"])
LOG = Path(os.environ["FAKE_LOG"])

def _log(*a):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(a) + "\\n")

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
