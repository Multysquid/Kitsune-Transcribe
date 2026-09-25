"""Infra checks: the image recipe, the CI workflow, the vast scripts and the upstream revision pins.

Everything runs on CPU without network or credentials: vastai, the vast REST API and the HF Hub are faked, shell
scripts are only syntax-checked (plus the watchdog's --dry-run), and the Docker image is not built.
"""
import hashlib
import importlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
VAST = ROOT / "vast"
SHELL_SCRIPTS = [VAST / "onstart_stub.sh", VAST / "onstart.sh", VAST / "bootstrap.sh", VAST / "watchdog.sh"]
SHA = "0123456789abcdef0123456789abcdef01234567"
DIGEST_IMAGE = "ghcr.io/multysquid/kitsune-train@sha256:" + "ab" * 32

sys.path.insert(0, str(VAST))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
finish = importlib.import_module("finish")
supervise = importlib.import_module("supervise")
launch = importlib.import_module("launch")
from fixtures import REAL, no_real_data  # noqa: E402


def load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_bash() -> str | None:
    for cand in (shutil.which("bash"), r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if cand and Path(cand).exists() and "system32" not in cand.lower():  # System32\bash.exe is WSL, not Git Bash
            return cand
    return None


# ------------------------------------------------------------------------------------------ requirements / image / CI

def test_requirements_pins():
    lines = [ln.split("#", 1)[0].strip() for ln in (ROOT / "requirements-train.txt").read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    pins = dict(ln.split("==", 1) for ln in lines)
    assert all("==" in ln for ln in lines), "every requirement is pinned exactly"
    for name, ver in {"transformers": "5.13.1", "huggingface_hub": "1.23.0", "hf_xet": "1.5.1", "tensorboard": "2.20.0",
                      "pyarrow": "25.0.1", "numpy": "2.5.3", "soundfile": "0.14.0", "librosa": "1.0.0",
                      "soxr": "1.1.0", "jiwer": "4.0.0", "pandas": "2.3.2"}.items():
        assert pins.get(name) == ver, name
    for name in ("tqdm", "nvidia-ml-py", "pytest", "vastai", "tokenizers", "sentencepiece", "safetensors"):
        assert name in pins, name
    assert "torch" not in pins, "torch comes from the base image"


def test_smoke_import_covers_every_pin():
    smoke = load_path("smoke_import", ROOT / "docker" / "smoke_import.py")
    pins = smoke.parse_pins(ROOT / "requirements-train.txt")
    covered = {smoke.normalise(d) for d in smoke.MODULES}
    assert set(pins) <= covered, f"not imported by smoke_import.py: {set(pins) - covered}"


def test_smoke_audio_fails_without_a_codec_the_box_decodes(monkeypatch):
    # a libsndfile without MP3 (soundfile's none-any wheel falls back to whatever system library it finds) must fail the
    # CI smoke, not pass on FLAC and turn every Emilia row into bad_audio on the box
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("soxr")
    pytest.importorskip("librosa")
    smoke = load_path("smoke_import", ROOT / "docker" / "smoke_import.py")
    assert smoke.audio_roundtrip() == sf.__libsndfile_version__

    class NoMp3(sf.SoundFile):
        def __init__(self, file, mode="r", samplerate=None, channels=None, subtype=None, endian=None, format=None,
                     *args, **kwargs):
            if str(format).upper() == "MP3":
                raise RuntimeError("Error opening <_io.BytesIO>: Format not recognised.")
            super().__init__(file, mode, samplerate, channels, subtype, endian, format, *args, **kwargs)

    monkeypatch.setattr(sf, "SoundFile", NoMp3)
    with pytest.raises(RuntimeError, match="Format not recognised"):
        smoke.audio_roundtrip()


def test_dockerfile():
    text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    instr = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    froms = [ln for ln in instr if ln.startswith("FROM ")]
    assert len(froms) == 1
    assert re.fullmatch(r"FROM vastai/pytorch:2\.14\.0-cu130-cuda-13\.2-mini-py312-2026-09-08@sha256:[0-9a-f]{64}",
                        froms[0]), froms[0]
    for kv in ("PYTHONUNBUFFERED=1", "TZ=UTC", "HF_XET_HIGH_PERFORMANCE=1"):
        assert re.search(rf"^ENV[^\n]*\b{kv}\b", text, re.M), kv
    assert "org.opencontainers.image.source=\"https://github.com/Multysquid/Kitsune-Transcribe\"" in text
    copies = [ln for ln in instr if ln.startswith(("COPY", "ADD"))]
    assert copies == ["COPY requirements-train.txt docker/smoke_import.py /opt/kitsune/"], "no code, data or weights"
    assert "/venv/main" in text and "requirements-train.txt" in text
    assert "HF_TOKEN" not in text


def test_workflow():
    yaml = pytest.importorskip("yaml")
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "image.yml").read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))  # YAML 1.1 reads the bare key `on` as True
    push = on["push"]
    assert push.get("branches") in (None, ["**"])
    assert set(push["paths"]) == {"docker/**", "requirements-train.txt", ".github/workflows/image.yml"}
    assert wf["permissions"]["packages"] == "write"
    assert wf["env"]["IMAGE"] == "ghcr.io/multysquid/kitsune-train"
    steps = wf["jobs"]["build"]["steps"]
    uses = {s.get("uses", "").split("@")[0]: s for s in steps}
    assert "jlumbroso/free-disk-space" in uses
    assert all(re.search(r"@[0-9a-f]{40}$", s["uses"]) for s in steps if "uses" in s), "actions pinned by commit"
    meta = uses["docker/metadata-action"]["with"]
    assert "type=sha,prefix=sha-" in meta["tags"] and "type=ref,event=branch" in meta["tags"]
    assert "org.opencontainers.image.source=https://github.com/Multysquid/Kitsune-Transcribe" in meta["labels"]
    build = uses["docker/build-push-action"]["with"]
    assert build["file"] == "docker/Dockerfile" and build["platforms"] == "linux/amd64"
    assert build["cache-from"].startswith("type=gha") and build["cache-to"].startswith("type=gha")
    assert "push=true" in build["outputs"]
    assert uses["docker/login-action"]["with"]["password"] == "${{ secrets.GITHUB_TOKEN }}"
    runs = [s.get("run", "") for s in steps]
    smoke = next(i for i, r in enumerate(runs) if "smoke_import.py" in r)
    tag = next(i for i, r in enumerate(runs) if "imagetools create" in r)
    build_i = next(i for i, s in enumerate(steps) if s.get("uses", "").startswith("docker/build-push-action"))
    assert build_i < smoke < tag, "push, then smoke, then tag"


# ------------------------------------------------------------------------------------------------------ shell scripts

@pytest.mark.parametrize("script", SHELL_SCRIPTS, ids=lambda p: p.name)
def test_shell_script_syntax_and_hygiene(script: Path):
    text = script.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/bash\n")
    assert "\r" not in text, "LF line endings (vast runs these with bash on Linux)"
    assert "set -euo pipefail" in text
    assert not re.search(r"^\s*set -[a-z]*x|set -o xtrace", text, re.M), "xtrace would print secrets"
    for line in text.splitlines():
        if re.search(r"\b(echo|printf|log)\b", line):
            assert not re.search(r"\$\{?HF_TOKEN\b(?!:-\})", line), f"prints HF_TOKEN: {line.strip()}"
            assert "CONTAINER_API_KEY" not in line or "header = " in line, f"prints the vast key: {line.strip()}"
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    r = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_onstart_stub_fits_vast_limits():
    raw = (VAST / "onstart_stub.sh").read_bytes()
    assert len(raw) <= launch.ONSTART_MAX_BYTES < 4048, "the API's onstart field may be capped near 4 KB"
    raw.decode("ascii")  # vastai reads the file with the platform default encoding on Windows
    assert b"\r\n" not in raw
    text = raw.decode()
    for needle in ("KITSUNE_SHA", "https://github.com/Multysquid/Kitsune-Transcribe", "fetch -q --depth 1",
                   'exec bash "$D/vast/onstart.sh"', "console.vast.ai/api/v0/instances", "KITSUNE_NO_SELF_STOP",
                   "--config -", "/workspace/kitsune.log"):
        assert needle in text, needle
    assert launch.ONSTART == VAST / "onstart_stub.sh"


def test_onstart_stub_stops_the_box_when_the_clone_fails(tmp_path):
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    log = tmp_path / "kitsune.log"
    env = dict(os.environ, KITSUNE_DIR=(tmp_path / "repo").as_posix(), KITSUNE_LOG=log.as_posix(), KITSUNE_SHA="not-a-sha",
               KITSUNE_NO_SELF_STOP="1")
    r = subprocess.run([bash, str(VAST / "onstart_stub.sh")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 1
    text = log.read_text(encoding="utf-8")
    assert "full 40-hex" in text and "KITSUNE_NO_SELF_STOP=1" in text


# vast's stop reply as curl -f leaves it: the body of a 2xx on stdout, nothing and exit 22 on an HTTP error. An exported
# bash function stands in for curl (no PATH games on Windows); it swallows the key config curl reads from stdin.
FAKE_CURL = '() { cat >/dev/null; [ "${FAKE_CURL_RC:-0}" = 0 ] || return "$FAKE_CURL_RC"; printf %s "$FAKE_CURL_REPLY"; }'
STOP_REPLIES = [('{"success": false, "msg": "nope"}', "0", False), ('{"success":false,"msg":"nope"}', "0", False),
                ('{"success": true}', "0", True), ("", "22", False)]


def curl_env(reply: str, rc: str, **kw) -> dict:
    env = dict(os.environ, CONTAINER_API_KEY="secret-key-123", CONTAINER_ID="7", FAKE_CURL_REPLY=reply,
               FAKE_CURL_RC=rc, **kw)
    env["BASH_FUNC_curl%%"] = FAKE_CURL
    env.pop("KITSUNE_NO_SELF_STOP", None)
    return env


@pytest.mark.parametrize("reply,rc,ok", STOP_REPLIES, ids=["refused", "refused-compact", "ok", "http-error"])
def test_onstart_stub_counts_a_refused_stop_as_failed(tmp_path, reply, rc, ok):
    """vast can answer the stop with a 2xx whose body says {"success": false} (finish.py vast_rest reads it so); the
    stub, with no watchdog yet, must not log that the box was stopped then."""
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    log = tmp_path / "kitsune.log"
    env = curl_env(reply, rc, KITSUNE_DIR=(tmp_path / "repo").as_posix(), KITSUNE_LOG=log.as_posix(),
                   KITSUNE_SHA="not-a-sha")
    r = subprocess.run([bash, str(VAST / "onstart_stub.sh")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 1
    text = log.read_text(encoding="utf-8")
    assert "full 40-hex" in text and "secret-key-123" not in text
    if ok:
        assert "stop requested via REST" in text and "FAILED" not in text, text
    else:
        assert "stop requested via REST" not in text and "stop request FAILED" in text, text
        assert ("nope" if reply else "(no reply)") in text, text


@pytest.mark.parametrize("reply,rc,ok", STOP_REPLIES, ids=["refused", "refused-compact", "ok", "http-error"])
def test_onstart_and_watchdog_curl_stops_read_the_reply(tmp_path, reply, rc, ok):
    """The same for the curl fallbacks behind finish.py --stop: onstart.sh stop_instance and watchdog.sh stop_now return
    0 only for a stop vast accepted, and log vast's reply otherwise."""
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    for script, func, call in (("onstart.sh", "stop_instance", 'stop_instance "why"'), ("watchdog.sh", "stop_now",
                                                                                          "stop_now")):
        text = (VAST / script).read_text(encoding="utf-8")
        body = re.search(rf"^{func}\(\) \{{[^\n]*\n.*?^\}}\n", text, re.M | re.S).group(0)
        test = tmp_path / f"{func}.sh"
        test.write_text("\n".join([
            "set -euo pipefail", "log() { printf '%s\\n' \"$*\"; }",
            f'KITSUNE_DIR="{tmp_path.as_posix()}"', "PY=false", "MAX_HOURS=5.5",  # finish.py is missing or fails
            body, f'{call} && echo "rc=0" || echo "rc=$?"', ""]), encoding="utf-8", newline="\n")
        r = subprocess.run([bash, str(test)], capture_output=True, text=True, env=curl_env(reply, rc), timeout=60)
        out = r.stdout
        assert r.returncode == 0 and ("rc=0" in out) == ok and "secret-key-123" not in out, (script, out, r.stderr)
        if not ok:
            assert "rc=1" in out and "vast REST stop failed: " + (reply or "no reply") in out, (script, out)
            assert "stop requested via REST" not in out, (script, out)


def test_bootstrap_retries_the_audio_rebuild(tmp_path):
    """One transient Hub error in 01's listing calls (sent once, with no HTTP timeout) must not stop the box after the
    paid boot: the rebuild runs through retry() with a timeout. retry() returns 0 once an attempt passes, and after
    the last one the last exit code, so set -e still stops bootstrap with the real cause."""
    text = (VAST / "bootstrap.sh").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.lstrip().startswith("phase rebuild_audio"))
    assert re.search(r"phase rebuild_audio retry \d+ timeout -k \d+ \d+m \"\$PY\" scripts/01_prepare_data\.py", line)
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    func = re.search(r"^retry\(\) \{\n.*?^\}\n", text, re.M | re.S).group(0)
    # an external command that fails until its n-th run
    (tmp_path / "flaky.sh").write_text('n=$(( $(cat "$1" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$1"; '
                                       '[ "$n" -ge "$2" ]\n', encoding="utf-8", newline="\n")

    def run(counter: str, passes_at: int):
        script = tmp_path / f"{counter}.sh"
        script.write_text("\n".join([
            "set -euo pipefail", f'cd "{tmp_path.as_posix()}"', "log() { printf '%s\\n' \"$*\"; }",
            "sleep() { :; }",  # the real waits are minutes
            func, f'retry 3 "$BASH" flaky.sh {counter} {passes_at}', f'echo "after $(cat {counter})"', ""]),
            encoding="utf-8", newline="\n")
        return subprocess.run([bash, str(script)], capture_output=True, text=True, timeout=60)

    r = run("once", 2)
    assert r.returncode == 0 and "attempt 1/3" in r.stdout and "after 2" in r.stdout, r.stdout + r.stderr
    r = run("always", 99)
    assert r.returncode == 1 and "attempt 3/3" in r.stdout and "after" not in r.stdout, r.stdout + r.stderr
    assert (tmp_path / "always").read_text().strip() == "3"


def test_bootstrap_lists_parked_shards_in_the_manifest(tmp_path, prep):
    """A parked source's shards come from the data repo without data/manifest.jsonl, but 01 reads other sources' rows
    from the manifest: with emilia_yodas parked, eval_emilia's rebuild stopped with "ingest emilia_yodas first" and the
    box was stopped. The helper's pull lists the parked shards as a local ingest would; a second pull adds nothing."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from kitsune.store import SCHEMA, read_manifest

    text = (VAST / "bootstrap.sh").read_text(encoding="utf-8")
    (tmp_path / "helper.py").write_text(re.search(r"<<'PYEOF'\n(.*?\n)PYEOF\n", text, re.S).group(1), encoding="utf-8")
    stub = tmp_path / "stub" / "huggingface_hub"  # the parked shards below are what snapshot_download pulled
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text("def snapshot_download(*args, **kwargs):\n    pass\n", encoding="utf-8")
    box, state = tmp_path / "box", tmp_path / "state"
    (box / "configs").mkdir(parents=True)
    state.mkdir()
    (box / "configs" / "c.json").write_text(json.dumps(dict(
        sources=["emilia_yodas"], eval_sets=["eval_emilia"], data_root="data", selection="selection/v.parquet",
        student="students/s")), encoding="utf-8")
    (state / "bootstrap_plan.json").write_text(json.dumps(dict(
        patterns=[], parked=["emilia_yodas"], rebuild=["eval_emilia"],
        files=["data/shards/emilia_yodas/train-00000.parquet"])), encoding="utf-8")
    shards = box / "data" / "shards" / "emilia_yodas"
    shards.mkdir(parents=True)
    rows = [dict(id=f"emilia_yodas/JA_{v}_W{i:06d}", source="emilia_yodas", split="train", audio=b"", text="x",
                 duration=1.5, sr=16000) for v, i in (("vid_a", 1), ("vid_a", 2), ("vid_b", 1))]
    pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), shards / "train-00000.parquet")
    env = dict(os.environ, KITSUNE_DIR=str(box), STATE=str(state), CONFIG="configs/c.json", KITSUNE_DATA_REPO="u/data",
               KITSUNE_DATA_REVISION="main", PYTHONPATH=os.pathsep.join([str(tmp_path / "stub"), str(ROOT)]))
    for _ in range(2):
        r = subprocess.run([sys.executable, str(tmp_path / "helper.py"), "pull"], capture_output=True, text=True,
                           env=env, timeout=120)
        assert r.returncode == 0, r.stdout + r.stderr
    (sh,) = read_manifest(box / "data")
    assert (sh.path, sh.source, sh.split, sh.rows) == ("shards/emilia_yodas/train-00000.parquet", "emilia_yodas",
                                                       "train", 3)
    assert sh.hours == pytest.approx(4.5 / 3600)

    class Reached(Exception):
        pass

    def download(repo, filename):
        raise Reached(filename)

    ing = prep.Ingest(box / "data", tmp_path / "raw", "eval_emilia", None)
    ing.download = download
    with pytest.raises(Reached):  # past the "ingest emilia_yodas first" check, on to the hold-out's tar
        prep.ingest_emilia_eval(ing, prep.EMILIA_EVAL_TAR, prep.EMILIA_EVAL_ROWS)


def test_bootstrap_pull_fails_before_the_rebuild_when_nothing_arrived(tmp_path):
    """When snapshot_download's one repo_info request fails (a Hub 5xx/429, a dropped connection; HF-X2) it returns the
    checkout it was given as local_dir, never empty, with only a warning and downloads nothing: the 10-20 min audio
    rebuild then ran and only the coverage check (no teacher ids) stopped the box. The plan now records the files the
    pull must bring, by the Hub client's own matcher; the pull exits non-zero while one is missing, under retry()."""
    import huggingface_hub.utils._paths as hf_paths

    text = (VAST / "bootstrap.sh").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.lstrip().startswith("phase pull_derived"))
    assert re.search(r'phase pull_derived retry \d+ timeout -k \d+ \d+m "\$PY" "\$HELPER" pull', line)
    (tmp_path / "helper.py").write_text(re.search(r"<<'PYEOF'\n(.*?\n)PYEOF\n", text, re.S).group(1), encoding="utf-8")
    repo_files = ["README.md", "teacher_out/meta.json", "second_out/meta.json", "selection/v.parquet",
                  *(f"students/s/{n}" for n in launch.STUDENT_FILES), "teacher_out/src_a/s0.npz",
                  "second_out/src_a/s0.jsonl", "teacher_out/eval_x/s0.npz", "teacher_out/other/s0.npz"]
    stub = tmp_path / "stub" / "huggingface_hub"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(
        "class HfApi:\n"
        "    def whoami(self):\n        return {'name': 'u'}\n"
        "    def auth_check(self, *a, **k):\n        pass\n"
        f"    def list_repo_files(self, *a, **k):\n        return {repo_files!r}\n"
        "def snapshot_download(*args, **kwargs):\n    pass  # the silent no-op: nothing arrives\n", encoding="utf-8")
    (stub / "utils.py").write_text(  # the real matcher (pure Python), which snapshot_download filters the tree with
        "import importlib.util\n"
        f"_s = importlib.util.spec_from_file_location('_hf_paths', {hf_paths.__file__!r})\n"
        "_m = importlib.util.module_from_spec(_s)\n_s.loader.exec_module(_m)\n"
        "filter_repo_objects = _m.filter_repo_objects\n", encoding="utf-8")
    box, state = tmp_path / "box", tmp_path / "state"
    (box / "configs").mkdir(parents=True)
    (box / "README.md").write_text("git checkout\n", encoding="utf-8")
    state.mkdir()
    (box / "configs" / "c.json").write_text(json.dumps(dict(
        sources=["src_a"], eval_sets=["eval_x"], data_root="data", selection="selection/v.parquet",
        student="students/s")), encoding="utf-8")
    env = dict(os.environ, KITSUNE_DIR=str(box), STATE=str(state), CONFIG="configs/c.json", KITSUNE_DATA_REPO="u/data",
               KITSUNE_OUT_REPO="u/runs", KITSUNE_DATA_REVISION="main",
               PYTHONPATH=os.pathsep.join([str(tmp_path / "stub"), str(ROOT)]))

    def helper(cmd):
        return subprocess.run([sys.executable, str(tmp_path / "helper.py"), cmd], capture_output=True, text=True,
                              env=env, timeout=120)

    r = helper("plan")
    assert r.returncode == 0, r.stdout + r.stderr
    want = json.loads((state / "bootstrap_plan.json").read_text(encoding="utf-8"))["files"]
    assert want == [f for f in repo_files if f not in ("README.md", "teacher_out/other/s0.npz")]
    r = helper("pull")
    assert r.returncode != 0 and f"pull incomplete: {len(want)} of {len(want)} planned files missing" in r.stderr
    for f in want:  # what a pull that reached the Hub leaves
        (box / f).parent.mkdir(parents=True, exist_ok=True)
        (box / f).write_bytes(b"x")
    r = helper("pull")
    assert r.returncode == 0, r.stdout + r.stderr


def test_bootstrap_coverage_is_checked_per_split(tmp_path):
    """The trainer joins a split's teacher ids only against that split's audio shards (<split>-*.parquet). Pooled over
    a source's splits, galgame's 1,000-row hold-out is 0.5 % of its ids: all of it missing (or moved into a train
    shard by a rebuild that split the rows differently) passed the 0.99 floor, and the eval store then dropped the
    hold-out with one printed line. The coverage check now holds every split of a source to the floor."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    text = (VAST / "bootstrap.sh").read_text(encoding="utf-8")
    (tmp_path / "helper.py").write_text(re.search(r"<<'PYEOF'\n(.*?\n)PYEOF\n", text, re.S).group(1), encoding="utf-8")
    box, state = tmp_path / "box", tmp_path / "state"
    (box / "configs").mkdir(parents=True)
    state.mkdir()
    (box / "configs" / "c.json").write_text(json.dumps(dict(
        sources=["gal"], eval_sets=["gal"], data_root="data", selection="selection/v.parquet", student="students/s")),
        encoding="utf-8")
    train, held = [f"gal/t{i:03d}" for i in range(300)], ["gal/e0", "gal/e1"]
    (box / "teacher_out" / "gal").mkdir(parents=True)
    np.savez(box / "teacher_out" / "gal" / "train-00000.npz", ids=np.array(train))
    np.savez(box / "teacher_out" / "gal" / "eval-00000.npz", ids=np.array(held))
    shards = box / "data" / "shards" / "gal"
    shards.mkdir(parents=True)
    pq.write_table(pa.table({"id": train}), shards / "train-00000.parquet")
    pq.write_table(pa.table({"id": held}), shards / "train-00001.parquet")  # the hold-out's rows landed in train
    env = dict(os.environ, KITSUNE_DIR=str(box), STATE=str(state), CONFIG="configs/c.json", KITSUNE_DATA_REPO="u/data",
               KITSUNE_DATA_REVISION="main", KITSUNE_MIN_COVERAGE="0.99")

    def coverage():
        r = subprocess.run([sys.executable, str(tmp_path / "helper.py"), "coverage"], capture_output=True, text=True,
                           env=env, timeout=120)
        return r, json.loads((state / "bootstrap_coverage.json").read_text(encoding="utf-8"))

    r, report = coverage()  # pooled: 302 of 302 ids joined
    assert r.returncode != 0 and "coverage below 0.99 for ['gal/eval']" in r.stderr, r.stdout + r.stderr
    assert report["gal/train"]["coverage"] == 1.0 and report["gal/eval"]["joined"] == 0
    (shards / "train-00001.parquet").unlink()
    pq.write_table(pa.table({"id": held}), shards / "eval-00000.parquet")
    r, report = coverage()
    assert r.returncode == 0, r.stdout + r.stderr
    assert set(report) == {"gal/train", "gal/eval"} and report["gal/eval"]["coverage"] == 1.0


def test_bootstrap_plan_checks_the_box_token_on_the_output_repo(tmp_path):
    """launch.py checks the HF repos with the laptop's own (broader) login; the box HF_TOKEN's first use of the output
    repo was the trainer's hf_roundtrip, after the pull, the 10-20 min audio rebuild and the model load. plan() now
    asks the Hub whether the token can read and write it (auth_check: GETs, no commit) before anything is pulled, so a
    wrongly scoped token stops the box in its first minutes. Only a refusal (401/403/404) is blamed on the token: a
    Hub 5xx, 429 or dropped connection is reported as a Hub error, and the plan phase runs under retry() with a
    timeout (its GETs have none), as the pull does."""
    text = (VAST / "bootstrap.sh").read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.lstrip().startswith("phase plan"))
    assert re.search(r'phase plan retry \d+ timeout -k \d+ \d+m "\$PY" "\$HELPER" plan', line)
    assert re.search(r'if \[ -z "\$\{KITSUNE_OUT_REPO:-\}" \]; then\n.*\n\s+exit 2\n', text)
    (tmp_path / "helper.py").write_text(re.search(r"<<'PYEOF'\n(.*?\n)PYEOF\n", text, re.S).group(1), encoding="utf-8")
    stub = tmp_path / "stub" / "huggingface_hub"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(
        "import os, types\n"
        "class HTTPError(Exception):\n"
        "    def __init__(self, status):\n"
        "        super().__init__(f'{status} from the Hub')\n"
        "        self.response = types.SimpleNamespace(status_code=status)\n"
        "class HfApi:\n"
        "    def whoami(self):\n        return {'name': 'u'}\n"
        "    def auth_check(self, repo_id, *, repo_type=None, write=False):\n"
        "        print(f'auth_check {repo_id} {repo_type} write={write}')\n"
        "        deny = os.environ.get('DENY_WRITE')\n"
        "        if write and deny == 'drop':\n"
        "            raise ConnectionError('Server disconnected without sending a response.')\n"
        "        if write and deny:\n"
        "            raise HTTPError(int(deny))\n"
        "    def list_repo_files(self, *a, **k):\n        return []\n", encoding="utf-8")
    (stub / "utils.py").write_text("def filter_repo_objects(items, **kw):\n    return list(items)\n", encoding="utf-8")
    box, state = tmp_path / "box", tmp_path / "state"
    (box / "configs").mkdir(parents=True)
    state.mkdir()
    (box / "configs" / "c.json").write_text(json.dumps(dict(
        sources=["src_a"], eval_sets=[], selection="selection/v.parquet", student="students/s")), encoding="utf-8")
    env = dict(os.environ, KITSUNE_DIR=str(box), STATE=str(state), CONFIG="configs/c.json", KITSUNE_DATA_REPO="u/data",
               KITSUNE_OUT_REPO="u/runs", KITSUNE_DATA_REVISION="main", DENY_WRITE="403",
               PYTHONPATH=os.pathsep.join([str(tmp_path / "stub"), str(ROOT)]))
    run = [sys.executable, str(tmp_path / "helper.py"), "plan"]
    r = subprocess.run(run, capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode != 0 and "auth_check u/runs model write=False" in r.stdout
    assert "HF_TOKEN cannot write u/runs (HTTPError: 403 from the Hub)" in r.stderr, r.stdout + r.stderr
    assert not (state / "bootstrap_plan.json").exists()
    r = subprocess.run(run, capture_output=True, text=True, env=dict(env, DENY_WRITE="404"), timeout=120)
    assert "HF_TOKEN cannot write u/runs (HTTPError: 404 from the Hub)" in r.stderr, "RepoNotFound hides the repo"
    for deny, what in (("503", "HTTPError: 503 from the Hub"), ("429", "HTTPError: 429 from the Hub"),
                       ("drop", "ConnectionError: Server disconnected")):
        r = subprocess.run(run, capture_output=True, text=True, env=dict(env, DENY_WRITE=deny), timeout=120)
        assert r.returncode != 0 and "HF_TOKEN cannot" not in r.stderr, r.stdout + r.stderr
        assert f"Hub error checking u/runs ({what}" in r.stderr and "retries the plan" in r.stderr, r.stderr
        assert not (state / "bootstrap_plan.json").exists()
    del env["DENY_WRITE"]
    r = subprocess.run(run, capture_output=True, text=True, env=env, timeout=120)
    assert "auth_check u/runs model write=True" in r.stdout and "HF_TOKEN cannot" not in r.stderr
    assert "data repo u/data@main lacks" in r.stderr, "past the token check, on to the data listing"


def test_onstart_fits_vast_limits():
    raw = (VAST / "onstart.sh").read_bytes()
    assert len(raw) < 16 * 1024, "vast's on-start field is limited to 16 KB (it is run from the clone by the stub)"
    raw.decode("ascii")
    text = raw.decode()
    for needle in ("/etc/environment", "ulimit -Sn", "/dev/shm", "KITSUNE_SHARING=file_system",
                   "entrypoint.sh", "https://github.com/Multysquid/Kitsune-Transcribe", "vast/watchdog.sh",
                   "vast/bootstrap.sh", "vast/supervise.py", "/workspace/kitsune.log", "halt", "--rearm",
                   "supervise.lock", "pids.max", "OPENBLAS_NUM_THREADS", "TOKIO_WORKER_THREADS"):
        assert needle in text, needle
    # onstart runs with errtrace: a cgroup read that fails inside $(...) would fire the ERR trap and stop the box
    for line in text.splitlines():
        if "/sys/fs/cgroup" in line and not line.lstrip().startswith("#"):
            assert "2>/dev/null ||" in line, line


def test_onstart_skips_bootstrap_on_a_restart_with_a_supervisor_history(tmp_path):
    """A container restart re-ran all of bootstrap (whoami, auth_check, the repo listing, 01's listings: Hub calls
    with only 3 tries) before the supervisor, so a Hub outage at restart time halted and stopped a run the supervisor
    would have resumed, or turned a recorded destroy into a stop. The supervisor starts only after a bootstrap that
    passed its coverage check and writes supervise.json before its first attempt (--rearm moves it aside): with that
    history the detached subshell hands straight over to it, still after the supervise.lock check."""
    text = (VAST / "onstart.sh").read_text(encoding="utf-8")
    body = re.search(r"^\(\n(.*?)^\) < /dev/null &$", text, re.M | re.S).group(1)
    assert body.index("supervise.lock") < body.index('-s "$KITSUNE_STATE/supervise.json"') < body.index(
        'bash "$KITSUNE_DIR/vast/bootstrap.sh"')
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    state, repo = tmp_path / "state", tmp_path / "repo"
    (repo / "vast").mkdir(parents=True)
    state.mkdir()
    for name in ("bootstrap.sh", "supervise.py"):  # run by bash (PY=bash below): each notes that it ran
        (repo / "vast" / name).write_text(f'echo {name} >> "$KITSUNE_STATE/ran"\n', encoding="utf-8", newline="\n")
    script = tmp_path / "subshell.sh"
    script.write_text("\n".join(["set -euo pipefail", "log() { printf '%s\\n' \"$*\"; }", "PY=bash", body]),
                      encoding="utf-8", newline="\n")
    env = dict(os.environ, KITSUNE_STATE=state.as_posix(), KITSUNE_DIR=repo.as_posix())

    def boot() -> tuple[list[str], str]:
        (state / "ran").unlink(missing_ok=True)
        r = subprocess.run([bash, str(script)], capture_output=True, text=True, env=env, timeout=60)
        assert r.returncode == 0, r.stdout + r.stderr
        return (state / "ran").read_text(encoding="utf-8").split(), r.stdout

    ran, out = boot()  # first boot
    assert ran == ["bootstrap.sh", "supervise.py"] and "bootstrap start" in out
    (state / "supervise.json").write_text('{"attempts": [{"t0": 1.0, "resume": null}]}', encoding="utf-8")
    ran, out = boot()  # a restart mid-attempt (or with a recorded final decision)
    assert ran == ["supervise.py"] and "supervisor history present" in out and "bootstrap start" not in out


def test_watchdog_dry_run(tmp_path):
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    env = dict(os.environ, KITSUNE_STATE=str(tmp_path / "state"), KITSUNE_MAX_HOURS="0.25")
    r = subprocess.run([bash, str(VAST / "watchdog.sh"), "--dry-run"], capture_output=True, text=True, env=env,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    assert "cap 0.25 h" in r.stdout and "finish.py --stop" in r.stdout
    assert not (tmp_path / "state" / "deadline").exists(), "a dry run must not fix the deadline"


# --------------------------------------------------------------------------------------------------------- launch.py

OFFERS_40 = [
    {"id": 222, "gpu_name": "A100 SXM4", "gpu_ram": 40960, "dph_total": 0.926, "reliability": 0.99,
     "cuda_max_good": 13.2, "cpu_cores_effective": 16, "cpu_ram": 96000, "inet_down": 900, "disk_bw": 2000,
     "geolocation": "US"},
    {"id": 111, "gpu_name": "A100 SXM4", "gpu_ram": 40960, "dph_total": 0.672, "reliability": 0.995,
     "cuda_max_good": 13.0, "cpu_cores_effective": 12, "cpu_ram": 64000, "inet_down": 600, "disk_bw": 800,
     "inet_down_cost": 0.01, "inet_up_cost": 0.02, "geolocation": "SE"},
]
OFFERS_80 = [{"id": 333, "gpu_name": "A100 SXM4", "gpu_ram": 81920, "dph_total": 1.604, "reliability": 0.99}]


class FakeVastai:
    """Stands in for subprocess.run of the vastai CLI; answers searches from a queue of offer lists (a
    CompletedProcess in the queue is returned as it is), and a create with `create` if given."""

    def __init__(self, searches, create=None):
        self.searches, self.calls, self.create = list(searches), [], create

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if argv[1:3] == ["search", "offers"]:
            if isinstance(self.searches[0], subprocess.CompletedProcess):
                return self.searches.pop(0)
            out = json.dumps(self.searches.pop(0))
        elif argv[1:3] == ["create", "instance"] and self.create is not None:
            return self.create(argv)
        elif argv[1:3] == ["create", "instance"]:
            out = "WARNING: something\n" + json.dumps({"success": True, "new_contract": 98765})
        else:
            raise AssertionError(f"unexpected vastai call {argv}")
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")


def launch_args(*extra):
    return ["--data-repo", "Multy123/kitsune-data", "--out-repo", "Multy123/kitsune-runs", "--sha", SHA,
            "--image", DIGEST_IMAGE, "--no-hf-check", "--skip-git-checks", *extra]


@pytest.fixture
def fake_vastai(monkeypatch):
    def install(searches, create=None):
        fake = FakeVastai(searches, create)
        monkeypatch.setattr(launch.shutil, "which", lambda name: "/fake/vastai" if name == "vastai" else None)
        monkeypatch.setattr(launch.subprocess, "run", fake)
        return fake
    return install


def test_launch_dry_run_builds_query_and_command(fake_vastai, capsys):
    fake = fake_vastai([OFFERS_40])
    assert launch.main(launch_args("--dry-run", "--yes")) == 0
    assert len(fake.calls) == 1, "dry run searches but never creates"
    search = fake.calls[0]
    assert search[:3] == ["/fake/vastai", "search", "offers"]
    query = search[3]
    for term in ("gpu_name in [A100_SXM4]", "num_gpus=1", "verified=true", "reliability>=0.98", "cuda_vers>=13.0",
                 "cpu_cores_effective>=12", "cpu_ram>=64", "disk_bw>=500", "inet_down>=500", "inet_up>=500",
                 "direct_port_count>=1", "gpu_ram<=48"):
        assert term in query.split(" ") or term in query, term
    assert search[4:] == ["--type", "on-demand", "-o", "dph", "--storage", "150", "--raw"]
    out = capsys.readouterr().out
    create_line = next(ln for ln in out.splitlines() if ln.strip().startswith("vastai create instance"))
    # cheapest offer wins even if the API returned it second
    assert create_line.strip().startswith("vastai create instance 111 ")
    for piece in (f"--image {DIGEST_IMAGE}", "--disk 150", "--ssh", "--direct", "--cancel-unavail",
                  f"-e KITSUNE_SHA={SHA}", "-e KITSUNE_CONFIG=configs/viability.json",
                  "-e KITSUNE_DATA_REPO=Multy123/kitsune-data", "-e KITSUNE_OUT_REPO=Multy123/kitsune-runs",
                  "-e TZ=UTC", "--onstart"):
        assert piece in create_line, piece
    assert "onstart_stub.sh" in create_line
    assert "HF_TOKEN" not in create_line
    assert "not creating anything (--dry-run)" in out
    # bandwidth priced from the host's $/GB (OFFERS_40[1]: 0.01 down, 0.02 up)
    assert "~$0.85 (~25 GB down, ~30 GB up" in out and "$/GBup" in out


def test_launch_no_self_stop_goes_inside_the_env_value(fake_vastai, capsys):
    """vastai's create has no -e option and a second --env replaces the first: the flag must join the one value."""
    fake_vastai([OFFERS_40])
    assert launch.main(launch_args("--dry-run")) == 0
    assert "KITSUNE_NO_SELF_STOP" not in capsys.readouterr().out
    fake = fake_vastai([OFFERS_40])
    assert launch.main(launch_args("--yes", "--no-self-stop")) == 0
    create = next(c for c in fake.calls if c[1:3] == ["create", "instance"])
    assert create.count("--env") == 1 and not any(a == "-e" for a in create)
    env = create[create.index("--env") + 1]
    assert "-e KITSUNE_NO_SELF_STOP=1" in env and f"-e KITSUNE_SHA={SHA}" in env


def test_onstart_stub_stops_the_box_when_kitsune_sha_is_unset(tmp_path):
    bash = find_bash()
    if bash is None:
        pytest.skip("bash not available")
    log = tmp_path / "kitsune.log"
    env = dict(os.environ, KITSUNE_DIR=(tmp_path / "repo").as_posix(), KITSUNE_LOG=log.as_posix(),
               KITSUNE_NO_SELF_STOP="1")
    env.pop("KITSUNE_SHA", None)
    r = subprocess.run([bash, str(VAST / "onstart_stub.sh")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 1
    text = log.read_text(encoding="utf-8")
    assert "onstart.sh missing" in text and "unbound variable" not in text, text


def test_launch_needs_yes_to_create(fake_vastai, capsys):
    fake = fake_vastai([OFFERS_40])
    assert launch.main(launch_args()) == 0
    assert all(c[1:3] != ["create", "instance"] for c in fake.calls)
    assert "re-run with --yes" in capsys.readouterr().out


def test_launch_falls_back_to_80gb_and_creates_with_yes(fake_vastai, capsys):
    fake = fake_vastai([[], OFFERS_80])
    assert launch.main(launch_args("--yes")) == 0
    searches = [c for c in fake.calls if c[1:3] == ["search", "offers"]]
    assert "gpu_ram<=48" in searches[0][3] and "gpu_ram>=70" in searches[1][3]
    create = next(c for c in fake.calls if c[1:3] == ["create", "instance"])
    assert create[3] == "333"
    assert create[create.index("--image") + 1] == DIGEST_IMAGE
    assert create[create.index("--disk") + 1] == "150"
    assert Path(create[create.index("--onstart") + 1]) == VAST / "onstart_stub.sh"
    env = create[create.index("--env") + 1]
    assert f"-e KITSUNE_SHA={SHA}" in env and "-e TZ=UTC" in env and "HF_TOKEN" not in env
    assert not any("HF_TOKEN" in a for a in create)
    assert "created instance 98765" in capsys.readouterr().out


def test_launch_refuses_expensive_offer(fake_vastai):
    fake = fake_vastai([[], OFFERS_80])
    assert launch.main(launch_args("--yes", "--max-dph", "1.0")) == 1
    assert all(c[1:3] != ["create", "instance"] for c in fake.calls)


def test_launch_reports_vastais_api_error_despite_exit_0(fake_vastai):
    """vastai 1.8.0 exits 0 on an API error, with nothing on stdout and the JSON error on stderr (VAST-X1)."""
    def refused(msg):
        return lambda argv: subprocess.CompletedProcess(argv, 0, stdout="", stderr=json.dumps(
            {"error": True, "status_code": 410, "msg": msg}))

    fake_vastai([OFFERS_40], create=refused("error 410/3907: no_such_ask Instance type 111 is no longer available."))
    with pytest.raises(launch.LaunchError, match="no_such_ask"):
        launch.main(launch_args("--yes"))
    bad_key = subprocess.CompletedProcess([], 0, stdout="", stderr='{"error": true, "status_code": 401, '
                                                                   '"msg": "Invalid user key"}')
    fake_vastai([bad_key])
    with pytest.raises(launch.LaunchError, match="search offers failed .*Invalid user key"):
        launch.main(launch_args("--yes"))


def test_launch_create_timeout_says_an_instance_may_exist(fake_vastai):
    def hangs(argv):
        raise subprocess.TimeoutExpired(argv, 180)

    fake_vastai([OFFERS_40], create=hangs)
    with pytest.raises(launch.LaunchError, match=r"vastai show instances.*kitsune-viability-0123456"):
        launch.main(launch_args("--yes"))


def test_launch_without_vastai_prints_install_help(monkeypatch, capsys):
    monkeypatch.setattr(launch.shutil, "which", lambda name: None)
    assert launch.main(launch_args()) == 2
    out = capsys.readouterr().out
    assert "pip install vastai==" in out and "checks passed" in out


def test_launch_runs_the_checks_without_vastai(monkeypatch, capsys):
    """The git/image/HF checks report before the CLI is needed (here: an image not pinned by digest)."""
    monkeypatch.setattr(launch.shutil, "which", lambda name: None)
    args = [a if a != DIGEST_IMAGE else "ghcr.io/multysquid/kitsune-train:main" for a in launch_args()]
    assert launch.main(args) == 2
    out = capsys.readouterr().out
    assert "must be pinned by digest" in out and "pip install vastai==" in out


class FakeRegistry:
    """urllib.request.urlopen for ghcr.io: an anonymous token, an image index (an attestation entry first, then the
    linux/amd64 image), that image's manifest and its config blob with the given labels."""

    def __init__(self, labels):
        self.paths = []
        self.docs = {
            "manifests/sha256:idx": {"manifests": [
                {"digest": "sha256:att", "platform": {"architecture": "unknown", "os": "unknown"}},
                {"digest": "sha256:amd", "platform": {"architecture": "amd64", "os": "linux"}}]},
            "manifests/sha256:amd": {"config": {"digest": "sha256:cfg"}},
            "blobs/sha256:cfg": {"architecture": "amd64", "config": {"Labels": labels}},
        }

    def __call__(self, req, timeout=None):
        import io

        url = req if isinstance(req, str) else req.full_url
        if url.startswith("https://ghcr.io/token?"):
            return io.BytesIO(json.dumps({"token": "t"}).encode())
        assert req.get_header("Authorization") == "Bearer t"
        path = url.removeprefix("https://ghcr.io/v2/multysquid/kitsune-train/")
        self.paths.append(path)
        return io.BytesIO(json.dumps(self.docs[path]).encode())


def test_image_revision_reads_the_build_commit_label(monkeypatch):
    rev = "3ab6eb6f62493b195b480bebab8d6ce1be477c78"
    reg = FakeRegistry({"org.opencontainers.image.revision": rev, "org.opencontainers.image.version": "main"})
    monkeypatch.setattr(launch.urllib.request, "urlopen", reg)
    assert launch.image_revision(f"{launch.IMAGE_REPO}@sha256:idx") == rev
    assert reg.paths == ["manifests/sha256:idx", "manifests/sha256:amd", "blobs/sha256:cfg"]
    monkeypatch.setattr(launch.urllib.request, "urlopen", FakeRegistry({}))
    with pytest.raises(launch.LaunchError, match="no org.opencontainers.image.revision"):
        launch.image_revision(f"{launch.IMAGE_REPO}@sha256:idx")


def test_image_problems_refuse_an_image_built_from_other_dependencies(monkeypatch):
    """CI moves a branch tag only after the build and its smoke test pass: while a pin bump is still building, or after
    its smoke failed, the tag names the previous image and the box would run the new code on the old pins. The image's
    build commit must have the same requirements-train.txt and docker/Dockerfile as the commit to run."""
    rev, image = "f" * 40, f"{launch.IMAGE_REPO}@sha256:" + "cd" * 32
    monkeypatch.setattr(launch, "image_revision", lambda img: rev)

    def fake_git(changed="", known=True):
        def git(*args):
            if args[0] == "cat-file":
                assert args[1:] == ("-e", f"{rev}^{{commit}}")
                if not known:
                    raise subprocess.CalledProcessError(128, args)
                return ""
            assert args == ("diff", "--name-only", rev, SHA, "--", *launch.IMAGE_INPUTS)
            return changed
        return git

    monkeypatch.setattr(launch, "git", fake_git())
    assert launch.image_problems(image, SHA) == []
    monkeypatch.setattr(launch, "git", fake_git("requirements-train.txt\n"))
    (p,) = launch.image_problems(image, SHA)
    assert f"was built from {rev[:12]} but {SHA[:12]} changes ['requirements-train.txt']: wait for the image build" in p
    monkeypatch.setattr(launch, "git", fake_git(known=False))
    (p,) = launch.image_problems(image, SHA)
    assert f"built from {rev[:12]}, which this clone lacks: git fetch" in p

    def unreadable(img):
        raise launch.LaunchError(f"{img} carries no org.opencontainers.image.revision label")

    monkeypatch.setattr(launch, "image_revision", unreadable)
    (p,) = launch.image_problems(image, SHA)
    assert p.startswith(f"cannot read the commit {image} was built from (LaunchError")


def test_launch_checks_the_image_revision_and_explains_a_missing_branch_tag(monkeypatch, capsys):
    import urllib.error

    monkeypatch.setattr(launch.shutil, "which", lambda name: None)
    monkeypatch.setattr(launch, "git_checks", lambda sha, config: [])
    calls = []
    monkeypatch.setattr(launch, "image_problems", lambda image, sha: calls.append((image, sha)) or ["image is stale"])
    no_skip = [a for a in launch_args() if a != "--skip-git-checks"]
    assert launch.main(no_skip) == 2
    assert calls == [(DIGEST_IMAGE, SHA)] and "image is stale" in capsys.readouterr().out
    assert launch.main(launch_args()) == 2 and len(calls) == 1, "--skip-git-checks skips it"

    def missing(tag):
        raise urllib.error.HTTPError(f"https://ghcr.io/v2/x/manifests/{tag}", 404, "Not Found", None, None)

    monkeypatch.setattr(launch, "resolve_image_digest", missing)
    args = [a for a in no_skip if a not in ("--image", DIGEST_IMAGE)] + ["--image-tag", "audit-fixes"]
    assert launch.main(args) == 2
    out = capsys.readouterr().out
    assert "cannot resolve ghcr.io/multysquid/kitsune-train:audit-fixes to a digest (HTTP Error 404" in out
    assert "a code-only branch or a detached HEAD has none: pass --image-tag main" in out
    assert len(calls) == 1, "an unresolved image is not checked"


VIAB_CFG ={"sources": ["reazon_small", "galgame"], "eval_sets": ["eval_jsut", "galgame"],
            "selection": "selection/viability.parquet", "student": "students/b20x2560-d4"}
DATA_FILES = ["teacher_out/meta.json", "second_out/meta.json", "selection/viability.parquet",
              "students/b20x2560-d4/config.json", "students/b20x2560-d4/model.safetensors",
              "students/b20x2560-d4/processor_config.json", "students/b20x2560-d4/tokenizer.json",
              "students/b20x2560-d4/tokenizer_config.json", "teacher_out/reazon_small/train-00000.npz", "teacher_out/reazon_small/train-00000.jsonl",
              "second_out/reazon_small/train-00000.jsonl",
              "teacher_out/galgame/train-00000.npz", "teacher_out/galgame/train-00001.npz",
              "teacher_out/galgame/eval-00000.npz", "second_out/galgame/train-00000.jsonl",
              "second_out/galgame/train-00001.jsonl", "second_out/galgame/eval-00000.jsonl",
              "teacher_out/eval_jsut/eval-00000.npz", "teacher_out/galgame/train-00000.jsonl",
              "teacher_out/galgame/train-00001.jsonl", "teacher_out/galgame/eval-00000.jsonl",
              "teacher_out/eval_jsut/eval-00000.jsonl"]


def test_data_problems_complete_repo_passes():
    assert launch.data_problems(DATA_FILES, VIAB_CFG) == []


def test_data_problems_catch_a_partial_second_opinion_pass_and_missing_files():
    files = [f for f in DATA_FILES if f not in ("second_out/galgame/train-00001.jsonl", "second_out/meta.json",
                                                 "teacher_out/eval_jsut/eval-00000.npz")]
    problems = launch.data_problems(files, VIAB_CFG)
    assert any("second_out/galgame: 1 of 3 teacher shards have no second opinion (e.g. train-00001)" in p
               for p in problems), problems
    assert "no second_out/meta.json" in problems and "no teacher_out/eval_jsut/*.npz" in problems
    # an eval-only set needs no second opinion
    assert not any("eval_jsut: " in p and "second opinion" in p for p in problems)


def test_data_problems_need_the_students_processor_and_tokenizer():
    """The trainer loads the processor and tokenizer from the student dir and has no fallback (the old one read the
    gated teacher repo, a 403 with the box token): a student uploaded without them is caught on the laptop, before any
    money is spent, and again by bootstrap's plan, before the paid audio rebuild, with the same list."""
    files = [f for f in DATA_FILES if f != "students/b20x2560-d4/tokenizer.json"]
    assert launch.data_problems(files, VIAB_CFG) == ["no students/b20x2560-d4/tokenizer.json"]
    text = (VAST / "bootstrap.sh").read_text(encoding="utf-8")
    box = re.search(r"^STUDENT_FILES = (\(.*\))$", text, re.M)
    assert box and eval(box.group(1)) == launch.STUDENT_FILES
    assert 'required = [f"{selection}", *(f"{student}/{n}" for n in STUDENT_FILES)]' in text


SEL_CFG = dict(VIAB_CFG, selection_recipe={"agree_max": 0.5, "agree_max_source": ["galgame=0.4"],
                                           "filter_eval_sets": ["galgame"]})
SEL_ARGS = {"sources": ["reazon_small", "galgame"], "eval_sets": ["eval_jsut", "galgame"], "agree_max": 0.5,
            "agree_max_source": ["galgame=0.4"], "filter_eval_sets": ["galgame"], "out": "C:/laptop/sel.parquet",
            "config": "configs/viability.json"}
SEL_ROWS = [("reazon_small", "train", True, "kept"), ("galgame", "train", True, "kept"),
            ("galgame", "train", False, "agree>0.4"), ("eval_jsut", "eval", True, "kept"),
            ("galgame", "eval", True, "kept")]


def write_selection(path: Path, rows, args=None) -> Path:
    """A selection parquet as make_selection.py writes it: its arguments in the metadata key b"kitsune_selection";
    each row's teacher_file is <source>/<split>-00000."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.DataFrame(rows, columns=["source", "split", "keep", "reason"])
    df["teacher_file"] = df["source"] + "/" + df["split"] + "-00000"
    t = pa.Table.from_pandas(df, preserve_index=False)
    if args is not None:
        t = t.replace_schema_metadata({b"kitsune_selection": json.dumps(dict(args=args, created="x")).encode()})
    pq.write_table(t, path)
    return path


def test_selection_problems_flag_no_agree_rows(tmp_path):
    name = "selection/viability.parquet"
    path = write_selection(tmp_path / "sel.parquet", SEL_ROWS + [("galgame", "train", False, "no_agree")], SEL_ARGS)
    problems = launch.selection_problems(path, name, SEL_CFG)
    assert len(problems) == 1 and "1 rows as no_agree ({'galgame': 1})" in problems[0]
    assert launch.selection_problems(write_selection(path, SEL_ROWS, SEL_ARGS), name, SEL_CFG) == []


def test_selection_problems_flag_kept_rows_without_teacher_output_in_the_repo(tmp_path):
    """data_problems only asks for some teacher npz per source (a galgame train npz also satisfies its hold-out), so a
    kept row whose own teacher_file is not uploaded passed the preflight and build_stores raised FileNotFoundError on
    the box, after the paid bootstrap. The selection's kept teacher_file references are checked against the repo."""
    name = "selection/viability.parquet"
    path = write_selection(tmp_path / "sel.parquet", SEL_ROWS, SEL_ARGS)
    assert launch.selection_problems(path, name, SEL_CFG, set(DATA_FILES)) == []
    files = [f for f in DATA_FILES if f != "teacher_out/galgame/eval-00000.npz"]
    assert launch.data_problems(files, VIAB_CFG) == []  # the train npz still satisfy "teacher shards for galgame"
    problems = launch.selection_problems(path, name, SEL_CFG, set(files))
    assert problems == [f"{name} keeps rows whose teacher output is not in the data repo: 1 of 8 files missing (e.g. "
                        f"teacher_out/galgame/eval-00000.npz): upload the teacher_out the selection was built from"]
    # only kept rows count: a dropped row's teacher file may be absent
    rows = SEL_ROWS + [("reazon_small", "eval", False, "agree>0.5")]
    assert launch.selection_problems(write_selection(path, rows, SEL_ARGS), name, SEL_CFG, set(DATA_FILES)) == []


def test_selection_problems_check_the_recipe_and_the_kept_rows(tmp_path):
    """A selection rebuilt with a flag forgotten (a threshold, a hold-out filter, an eval set, a source) or keeping
    nothing of a configured source or eval set is refused before renting; the recorded paths do not matter, the order
    of the lists and how a threshold is spelled neither."""
    name, path = "selection/viability.parquet", tmp_path / "sel.parquet"
    same = dict(SEL_ARGS, sources=["galgame", "reazon_small"], agree_max_source=["galgame=0.40"],
                out="/elsewhere/x.parquet", config=None)
    assert launch.selection_problems(write_selection(path, SEL_ROWS, same), name, SEL_CFG) == []
    for key, value in (("agree_max_source", []), ("agree_max_source", ["galgame=0.5"]), ("agree_max", 0.3),
                       ("filter_eval_sets", []), ("eval_sets", ["eval_jsut"]), ("sources", ["reazon_small"])):
        problems = launch.selection_problems(write_selection(path, SEL_ROWS, dict(SEL_ARGS, **{key: value})), name,
                                             SEL_CFG)
        assert len(problems) == 1 and f"was built with {key} " in problems[0], (key, problems)
    problems = launch.selection_problems(write_selection(path, SEL_ROWS), name, SEL_CFG)  # not from make_selection
    assert len(problems) == 1 and "no record of how it was built" in problems[0]
    rows = [r for r in SEL_ROWS if r[:2] != ("galgame", "eval")] + [("galgame", "eval", False, "agree>0.5")]
    problems = launch.selection_problems(write_selection(path, rows, SEL_ARGS), name, SEL_CFG)
    assert problems == [f"{name} keeps no eval rows of ['galgame'] (in the config's eval_sets): rebuild it with "
                        f"scripts/make_selection.py --config <the run config> and upload it"]
    rows = [r for r in SEL_ROWS if r[0] != "reazon_small"]
    problems = launch.selection_problems(write_selection(path, rows, SEL_ARGS), name, SEL_CFG)
    assert len(problems) == 1 and "keeps no train rows of ['reazon_small']" in problems[0]
    problems = launch.selection_problems(write_selection(path, SEL_ROWS, SEL_ARGS), name, VIAB_CFG)
    assert problems == ["the run config has no selection_recipe to check the selection against"]
    # the real run config carries the recipe, spelled as make_selection.py records it
    via = json.loads((ROOT / "configs" / "viability.json").read_text(encoding="utf-8"))
    assert via["selection_recipe"] == {"agree_max": 0.5, "agree_max_source": ["emilia_yodas=0.2", "eval_emilia=0.2"],
                                       "filter_eval_sets": ["eval_emilia", "galgame"]}


class FakeHfApi:
    """dataset_info / model_info report the visibility in `private`; the data repo lists no selection, so the preflight
    downloads nothing."""
    private: dict = {}

    def dataset_info(self, repo):
        return SimpleNamespace(sha="a" * 40, private=self.private[repo])

    def model_info(self, repo):
        return SimpleNamespace(sha="b" * 40, private=self.private[repo])

    def list_repo_files(self, repo, **kw):
        return []


def test_hf_preflight_refuses_repos_that_are_not_private(monkeypatch):
    """Both repos carry dataset reference transcripts, and the trainer's hf.private only applies to a repo it creates:
    a public (or unknown-visibility) data or output repo is refused before renting."""
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeHfApi)  # hf_preflight imports it at call time
    monkeypatch.setattr(launch, "data_problems", lambda files, cfg: [])
    data, out = "u/kitsune-data", "u/kitsune-runs"
    for vis, flagged in (((True, True), []), ((False, True), [data]), ((True, None), [out]),
                         ((False, False), [data, out])):
        monkeypatch.setattr(FakeHfApi, "private", dict(zip((data, out), vis)))
        rev, problems = launch.hf_preflight(data, out, VIAB_CFG)
        assert rev == "a" * 40
        assert [p.split(" ", 1)[0] for p in problems] == flagged, problems
        assert all("is not private" in p and "hf repos settings" in p and "--private" in p for p in problems)


def test_hf_preflight_blames_a_missing_output_repo_only_on_a_4xx(monkeypatch):
    """A 5xx, 429 or dropped connection reading the output repo is the Hub's (model_info is not retried): the operator is
    told to re-run, not to create a repo that exists. A 401/404 (missing, or private and unseen by this login) is."""
    import httpx
    import huggingface_hub
    from huggingface_hub.utils import hf_raise_for_status

    def hub_error(status, **headers):
        url = "https://huggingface.co/api/models/u/kitsune-runs"
        try:
            hf_raise_for_status(httpx.Response(status, headers=headers, request=httpx.Request("GET", url)))
        except Exception as e:  # the exception the installed huggingface_hub raises for this reply
            return e
        raise AssertionError(status)

    class FailingModelInfo(FakeHfApi):
        error: Exception = None

        def model_info(self, repo):
            raise self.error

    monkeypatch.setattr(huggingface_hub, "HfApi", FailingModelInfo)
    monkeypatch.setattr(launch, "data_problems", lambda files, cfg: [])
    monkeypatch.setattr(FakeHfApi, "private", {"u/kitsune-data": True})
    for error, missing in ((hub_error(404, **{"X-Error-Code": "RepoNotFound"}), True), (hub_error(401), True),
                           (hub_error(503), False), (hub_error(429), False), (httpx.ConnectError("timed out"), False)):
        monkeypatch.setattr(FailingModelInfo, "error", error)
        _, problems = launch.hf_preflight("u/kitsune-data", "u/kitsune-runs", VIAB_CFG)
        assert len(problems) == 1 and str(error).splitlines()[0] in problems[0], (error, problems)
        assert ("hf repos create u/kitsune-runs --private" in problems[0]) == missing, problems
        assert ("re-run launch" in problems[0]) != missing, problems


def test_launch_help_needs_nothing():
    r = subprocess.run([sys.executable, str(VAST / "launch.py"), "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "--yes" in r.stdout
    assert "token" not in " ".join(re.findall(r"--[a-z-]+", r.stdout)), "no option takes the HF token"


def test_launch_env_string_refuses_secrets():
    with pytest.raises(launch.LaunchError):
        launch.env_string({"HF_TOKEN": "hf_x"})
    with pytest.raises(launch.LaunchError):
        launch.env_string({"KITSUNE_CONFIG": "has space"})


# ------------------------------------------------------------------------------------------------------ supervise.py

@pytest.mark.parametrize("rc,step,n_fail,full,action", [
    (0, 5000, 0, False, "destroy"),
    (3, 150, 1, True, "stop"),
    (1, 150, 1, True, "resume"),
    (None, 150, 1, True, "resume"),
    (1, 99, 1, True, "stop"),
    (1, 150, 1, False, "stop"),
    (1, 3000, 2, True, "stop"),
])
def test_decide(rc, step, n_fail, full, action):
    assert supervise.decide(rc, step, n_fail, Path("full_step_100") if full else None)[0] == action


@pytest.mark.parametrize("rc,n_fail,summary,action", [
    (-6, 1, "complete", "destroy"),  # an abort in the interpreter's teardown after the final eval and verdict
    (None, 1, "complete", "destroy"),
    (-6, 2, "complete", "destroy"),
    (-6, 1, "failed", "resume"),
    (3, 1, "complete", "stop"),
])
def test_decide_a_crash_after_the_complete_summary_destroys(rc, n_fail, summary, action):
    act, reason = supervise.decide(rc, 5000, n_fail, Path("full_step_5000"), summary)
    assert act == action
    if action == "destroy":
        assert "summary complete" in reason


class FakeTrainer:
    """Plays a script of attempts: (exit code, last step, write a full state?[, summary.json status]) into one run
    dir."""

    def __init__(self, runs_root: Path, script):
        self.run_dir, self.script, self.argvs = runs_root / "viability-b20x2560-test", list(script), []

    def __call__(self, argv, env):
        self.argvs.append(list(argv))
        rc, step, full, *summary = self.script.pop(0)
        (self.run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text("{}", encoding="utf-8")
        with open(self.run_dir / "metrics" / "scalars.jsonl", "a", encoding="utf-8") as f:
            for s in range(0, step + 1, 10):
                f.write(json.dumps({"step": s, "wall": 0, "elapsed_s": 0, "tag": "loss/total", "value": 1.0}) + "\n")
            f.write('{"step": 99999, "tag": "torn')  # a crash can leave half a line
        if full:
            (self.run_dir / "checkpoints" / f"full_step_{step}").mkdir(parents=True)
        if summary:
            (self.run_dir / "summary.json").write_text(json.dumps({"status": summary[0]}), encoding="utf-8")
        return rc


def run_supervise(tmp_path, monkeypatch, script, state=None, finish=None):
    runs = tmp_path / "runs"
    trainer = FakeTrainer(runs, script)
    finishes = []
    monkeypatch.setattr(supervise, "run_trainer", trainer)
    monkeypatch.setattr(supervise, "call_finish",
                        finish or (lambda args, timeout=None: finishes.append(list(args)) or 0))
    state_path = tmp_path / "state" / "supervise.json"
    if state is not None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state), encoding="utf-8")
    rc = supervise.supervise("configs/viability.json", "Multy123/kitsune-runs", runs, ["python", "04.py"], state_path)
    return rc, trainer, finishes, json.loads(state_path.read_text(encoding="utf-8"))


def test_supervise_success_destroys(tmp_path, monkeypatch):
    rc, trainer, finishes, state = run_supervise(tmp_path, monkeypatch, [(0, 500, True)])
    assert rc == 0 and len(trainer.argvs) == 1
    assert trainer.argvs[0] == ["python", "04.py", "--config", "configs/viability.json",
                                "--set", "hf.output_repo=Multy123/kitsune-runs"]
    assert [f[0] for f in finishes] == ["--destroy"]
    assert state["final"]["action"] == "destroy" and state["attempts"][0]["step"] == 500


def test_supervise_throughput_stops(tmp_path, monkeypatch):
    rc, trainer, finishes, state = run_supervise(tmp_path, monkeypatch, [(3, 120, True)])
    assert rc == 3 and len(trainer.argvs) == 1
    assert [f[0] for f in finishes] == ["--sync-only", "--stop"]


def test_supervise_late_failure_resumes_once_then_stops(tmp_path, monkeypatch):
    rc, trainer, finishes, state = run_supervise(tmp_path, monkeypatch, [(1, 150, True), (1, 300, True)])
    assert rc == 1 and len(trainer.argvs) == 2
    resume = trainer.argvs[1][trainer.argvs[1].index("--resume") + 1]
    assert Path(resume).name == "full_step_150"
    assert "--resume" not in trainer.argvs[0]
    assert [f[0] for f in finishes] == ["--sync-only", "--sync-only", "--stop"]
    assert state["final"]["action"] == "stop" and "second failure" in state["final"]["reason"]


def test_supervise_resume_then_success_destroys(tmp_path, monkeypatch):
    rc, trainer, finishes, _ = run_supervise(tmp_path, monkeypatch, [(1, 150, True), (0, 900, True)])
    assert rc == 0 and len(trainer.argvs) == 2
    assert [f[0] for f in finishes] == ["--sync-only", "--destroy"]


def test_supervise_early_failure_stops_without_resume(tmp_path, monkeypatch):
    rc, trainer, finishes, state = run_supervise(tmp_path, monkeypatch, [(1, 50, True)])
    assert len(trainer.argvs) == 1 and [f[0] for f in finishes] == ["--sync-only", "--stop"]
    assert "before step 100" in state["final"]["reason"]


def test_supervise_never_credits_a_rearmed_attempt_with_the_old_run(tmp_path, monkeypatch):
    """After --rearm the old run dir (step 5000, a full state) stays under runs/ while supervise.json starts afresh; a
    new trainer that dies before writing its config.json failed before step 100, it is not the old run to resume."""
    old = tmp_path / "runs" / "viability-b20x2560-20260923T000000Z"
    (old / "metrics").mkdir(parents=True)
    (old / "config.json").write_text("{}", encoding="utf-8")
    (old / "metrics" / "scalars.jsonl").write_text(json.dumps({"step": 5000, "tag": "x", "value": 0}) + "\n")
    (old / "checkpoints" / "full_step_5000").mkdir(parents=True)
    day_ago = time.time() - 86400
    for p in (old / "config.json", old):
        os.utime(p, (day_ago, day_ago))
    calls, finishes = [], []

    def dies_at_import(argv, env):
        calls.append(list(argv))
        (tmp_path / "runs" / "viability-b20x2560-20260925T000000Z").mkdir()  # build() mkdirs before RunLogger
        return 1

    monkeypatch.setattr(supervise, "run_trainer", dies_at_import)
    monkeypatch.setattr(supervise, "call_finish", lambda args, timeout=None: finishes.append(list(args)) or 0)
    state_path = tmp_path / "state" / "supervise.json"
    supervise.supervise("configs/viability.json", None, tmp_path / "runs", ["python", "04.py"], state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert len(calls) == 1 and [f[0] for f in finishes] == ["--sync-only", "--stop"]
    assert state["attempts"][0]["run_dir"] is None and state["attempts"][0]["step"] == 0
    assert "before step 100" in state["final"]["reason"]


def test_supervise_late_failure_without_full_state_stops(tmp_path, monkeypatch):
    rc, trainer, finishes, _ = run_supervise(tmp_path, monkeypatch, [(1, 400, False)])
    assert len(trainer.argvs) == 1 and finishes[-1][0] == "--stop"


def test_supervise_recovers_after_container_restart(tmp_path, monkeypatch):
    # the container died during attempt 1 after step 200 and a full state; the record is exactly what supervise
    # saved before starting the trainer ({t0, resume}: the run dir is only added when an attempt ends)
    run_dir = tmp_path / "runs" / "viability-b20x2560-test"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "metrics" / "scalars.jsonl").write_text(json.dumps({"step": 200, "tag": "x", "value": 0}) + "\n")
    (run_dir / "checkpoints" / "full_step_200").mkdir(parents=True)
    state = {"attempts": [{"t0": time.time() - 60, "resume": None}], "final": None}
    rc, trainer, finishes, state = run_supervise(tmp_path, monkeypatch, [(0, 800, False)], state=state)
    first = state["attempts"][0]
    assert first["interrupted"] and first["step"] == 200 and Path(first["run_dir"]) == run_dir
    assert len(trainer.argvs) == 1
    assert Path(trainer.argvs[0][trainer.argvs[0].index("--resume") + 1]).name == "full_step_200"
    assert finishes[-1][0] == "--destroy"


def test_supervise_restart_during_the_resumed_attempt_stops(tmp_path, monkeypatch):
    # attempt 1 failed at step 150 and was resumed; the container died during attempt 2: that is the second failure,
    # found through attempt 2's --resume path
    run_dir = tmp_path / "runs" / "viability-b20x2560-test"
    (run_dir / "metrics").mkdir(parents=True)
    (run_dir / "metrics" / "scalars.jsonl").write_text(json.dumps({"step": 420, "tag": "x", "value": 0}) + "\n")
    (run_dir / "checkpoints" / "full_step_400").mkdir(parents=True)
    state = {"attempts": [{"t0": 0.0, "resume": None, "rc": 1, "step": 150, "run_dir": str(run_dir)},
                          {"t0": 1.0, "resume": str(run_dir / "checkpoints" / "full_step_150")}], "final": None}
    rc, trainer, finishes, state = run_supervise(tmp_path, monkeypatch, [], state=state)
    assert trainer.argvs == [] and [f[0] for f in finishes] == ["--stop"]
    assert state["attempts"][1]["step"] == 420 and "second failure" in state["final"]["reason"]


def test_supervise_steps_aside_when_another_supervisor_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(supervise, "acquire_lock", lambda path: None)
    state = {"attempts": [{"t0": 0.0, "resume": None}], "final": None}  # the live attempt of the other supervisor
    rc, trainer, finishes, after = run_supervise(tmp_path, monkeypatch, [(0, 10, False)], state=state)
    assert rc == 0 and trainer.argvs == [] and finishes == [] and after == state  # untouched


def test_supervise_destroys_a_run_that_died_after_its_complete_summary(tmp_path, monkeypatch, capsys):
    """An xet upload still running when the trainer returns 0 can abort the interpreter's teardown (rc -6, SIGABRT):
    the run had finished (final eval, verdict, summary.json complete), and a resume would only run the end phase and
    its final eval again. It goes to finish --destroy, which uploads what is missing and verifies first."""
    rc, trainer, finishes, state = run_supervise(tmp_path, monkeypatch, [(-6, 900, True, "complete")])
    assert len(trainer.argvs) == 1 and [f[0] for f in finishes] == ["--destroy"]
    assert state["attempts"][0]["rc"] == -6 and state["attempts"][0]["summary"] == "complete"
    assert state["final"]["action"] == "destroy" and "summary complete, exit -6" in state["final"]["reason"]
    assert "exited -6 at step 900" in capsys.readouterr().out
    # a failed summary (a Python exception rewrites it) is a crash like any other: resume once
    rc, trainer, finishes, state = run_supervise(tmp_path / "b", monkeypatch, [(1, 150, True, "failed"), (0, 900, True)])
    assert len(trainer.argvs) == 2 and [f[0] for f in finishes] == ["--sync-only", "--destroy"]
    assert "summary" not in state["attempts"][0]


def test_supervise_restart_after_a_complete_summary_destroys(tmp_path, monkeypatch):
    """The container restarted while a finished trainer's end-state upload drained: the interrupted attempt's
    summary.json is complete, so the replayed decision is destroy, not a resume."""
    run = tmp_path / "runs" / "viability-b20x2560-test"
    (run / "checkpoints" / "full_step_900").mkdir(parents=True)
    (run / "config.json").write_text("{}", encoding="utf-8")
    (run / "summary.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    state = {"attempts": [{"t0": 0.0, "resume": None, "run_dir": str(run)}], "final": None}
    rc, trainer, finishes, state = run_supervise(tmp_path, monkeypatch, [], state=state)
    assert trainer.argvs == [] and [f[0] for f in finishes] == ["--destroy"]
    assert state["attempts"][0]["summary"] == "complete" and state["final"]["action"] == "destroy"


def test_supervise_post_crash_sync_leaves_the_full_state_to_the_stop_path(tmp_path, monkeypatch):
    rc, trainer, finishes, _ = run_supervise(tmp_path, monkeypatch, [(1, 150, True), (0, 900, True)])
    assert finishes[0] == ["--sync-only", "--no-full"] and finishes[-1][0] == "--destroy"


def test_supervise_records_the_containers_oom_kills(tmp_path, monkeypatch, capsys):
    """An OOM kill that takes a DataLoader worker ends the trainer with a plain error (rc 1, not -9), and the logs'
    RAM numbers are the host's: the cgroup's counter names it, in supervise.json and in the attempt's exit line."""
    cg = tmp_path / "cgroup"
    cg.mkdir()
    (cg / "memory.events").write_text("low 0\nhigh 0\nmax 4\noom 1\noom_kill 1\noom_group_kill 0\n")
    monkeypatch.setattr(supervise, "CGROUP", cg)
    trainer = FakeTrainer(tmp_path / "runs", [(1, 50, False)])

    def killed(argv, env):
        (cg / "memory.events").write_text("low 0\nhigh 0\nmax 9\noom 2\noom_kill 2\noom_group_kill 0\n")
        return trainer(argv, env)

    monkeypatch.setattr(supervise, "run_trainer", killed)
    monkeypatch.setattr(supervise, "call_finish", lambda args, timeout=None: 0)
    state_path = tmp_path / "state" / "supervise.json"
    supervise.supervise("configs/viability.json", None, tmp_path / "runs", ["python", "04.py"], state_path)
    assert json.loads(state_path.read_text(encoding="utf-8"))["attempts"][0]["oom_kills"] == 1
    assert "attempt 1 exited 1 (oom_kill +1) at step 50" in capsys.readouterr().out
    assert supervise.oom_kills(tmp_path / "nowhere") is None  # no cgroup (Windows): unknown, not 0
    (cg / "memory").mkdir()
    (cg / "memory" / "memory.oom_control").write_text("oom_kill_disable 0\nunder_oom 0\noom_kill 3\n")
    (cg / "memory.events").unlink()
    assert supervise.oom_kills(cg) == 3  # cgroup v1


def test_supervise_never_reruns_after_final(tmp_path, monkeypatch):
    state = {"attempts": [{"t0": 0.0, "rc": 0, "step": 10}], "final": {"action": "destroy"}}
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "halt").write_text("{}", encoding="utf-8")  # finish.py got as far as acting
    rc, trainer, finishes, _ = run_supervise(tmp_path, monkeypatch, [], state=state)
    assert rc == 0 and trainer.argvs == [] and finishes == []


@pytest.mark.parametrize("raw", [b"", b"\0" * 300, b'\xff\xfe{"att', b"null", b'{"final": null}'],
                         ids=["empty", "nul", "not-utf8", "null", "no-attempts"])
def test_supervise_unreadable_state_stops_instead_of_starting_a_fresh_run(tmp_path, monkeypatch, raw):
    """A supervise.json that is not an attempt history (empty or NUL-filled after an unclean host crash, torn bytes,
    other JSON) may have held a final decision or a used-up resume. It used to start a fresh run from step 0 (or, for
    bytes that are not UTF-8, crash the supervisor); now it is moved aside, a stop is recorded before finish runs, and
    a restart replays that stop without a trainer."""
    state_path = tmp_path / "state" / "supervise.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_bytes(raw)
    rc, trainer, finishes, state = run_supervise(tmp_path, monkeypatch, [(0, 500, True)])
    assert rc == 1 and trainer.argvs == [] and [f[0] for f in finishes] == ["--stop"]
    (aside,) = state_path.parent.glob("supervise.json.corrupt-*")
    assert aside.read_bytes() == raw
    assert state["attempts"] == [] and state["final"]["action"] == "stop" and "corrupt" not in state
    assert state["final"]["corrupt_state"] == str(aside) and "unreadable" in state["final"]["reason"]
    # a restart before finish wrote its halt marker runs that stop again, still no trainer
    rc, trainer, finishes, _ = run_supervise(tmp_path, monkeypatch, [(0, 500, True)])
    assert rc == 0 and trainer.argvs == [] and [f[0] for f in finishes] == ["--stop"]


def test_supervise_state_is_fsynced(tmp_path, monkeypatch):
    """save_state fsyncs the file before the rename and (POSIX) the directory after it; Windows skips the directory."""
    real, calls = os.fsync, []
    monkeypatch.setattr(os, "fsync", lambda fd: calls.append(fd) or real(fd))
    path = tmp_path / "state" / "supervise.json"
    supervise.save_state(path, {"attempts": [], "final": None})
    assert supervise.load_state(path) == {"attempts": [], "final": None}
    assert len(calls) == (1 if os.name == "nt" else 2) and not path.with_suffix(".json.tmp").exists()


def test_supervise_reruns_an_interrupted_final_finish(tmp_path, monkeypatch):
    """A host reboot during finish --destroy's upload leaves the recorded decision but no halt marker (finish.py writes
    it only once it acts): the rebooted supervisor runs that finish again, bounded, and never the trainer; without it
    the box idled until the watchdog's deadline, which only stops it."""
    state = {"attempts": [{"t0": 0.0, "rc": 0, "step": 900}],
             "final": {"action": "destroy", "reason": "trainer finished (exit 0)", "wall": 0.0}}
    calls = []
    rc, trainer, _, after = run_supervise(tmp_path, monkeypatch, [], state=state,
                                          finish=lambda args, timeout=None: calls.append((list(args), timeout)) or 0)
    assert rc == 0 and trainer.argvs == [] and after["final"] == state["final"]
    assert calls == [(["--destroy", "--reason", "trainer finished (exit 0)"], supervise.FINISH_TIMEOUT_S["destroy"])]


@pytest.mark.parametrize("script,final,finish_rc,fallback", [
    ([(3, 120, True)], "--stop", 1, True),        # the stop request failed
    ([(0, 500, True)], "--destroy", 124, True),   # timed out (a stalled Hub call in its sync or verification)
    ([(0, 500, True)], "--destroy", 2, False),    # verification failed and the instance was stopped
    ([(0, 500, True)], "--destroy", 0, False),
])
def test_supervise_bounds_the_final_finish_and_falls_back_to_a_plain_stop(tmp_path, monkeypatch, script, final,
                                                                           finish_rc, fallback):
    calls = []

    def finish(args, timeout=None):
        calls.append((list(args), timeout))
        return finish_rc if args[0] == final else 0

    rc, _, _, _ = run_supervise(tmp_path, monkeypatch, script, finish=finish)
    assert rc == script[0][0]
    finals = [(a, t) for a, t in calls if a[0] != "--sync-only"]
    assert finals[0][0][0] == final and finals[0][1] == supervise.FINISH_TIMEOUT_S[final[2:]]
    if fallback:
        assert len(finals) == 2 and finals[1][0][:2] == ["--stop", "--no-sync"]
        assert finals[1][1] == supervise.FALLBACK_TIMEOUT_S and final in finals[1][0][3]
    else:
        assert len(finals) == 1


def test_supervise_kills_a_hung_final_finish_and_stops_without_sync(tmp_path, monkeypatch):
    # finish.py --stop hangs in its sync (a Hub request with no reply): the supervisor must not wait for the watchdog
    calls = tmp_path / "calls.txt"
    fake = tmp_path / "root" / "vast" / "finish.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("import sys, time\n"
                    f"with open({str(calls)!r}, 'a') as f:\n"
                    "    f.write(' '.join(sys.argv[1:3]) + '\\n')\n"
                    "if sys.argv[1] in ('--stop', '--destroy') and '--no-sync' not in sys.argv:\n"
                    "    time.sleep(40)\n", encoding="utf-8")
    monkeypatch.setattr(supervise, "ROOT", tmp_path / "root")
    monkeypatch.setattr(supervise, "FINISH_TIMEOUT_S", {"stop": 4, "destroy": 4})
    monkeypatch.setattr(supervise, "run_trainer", FakeTrainer(tmp_path / "runs", [(3, 120, False)]))
    t0 = time.time()
    rc = supervise.supervise("configs/viability.json", None, tmp_path / "runs", ["python", "04.py"],
                             tmp_path / "state" / "supervise.json")
    assert rc == 3 and time.time() - t0 < 30
    assert calls.read_text(encoding="utf-8").splitlines() == ["--sync-only --no-full", "--stop --reason",
                                                              "--stop --no-sync"]


# --------------------------------------------------------------------------------------------------------- finish.py

def make_run(root: Path) -> Path:
    run = root / "runs" / "viability-b20x2560-20260924"
    files = {
        "config.json": b"{}",
        "summary.json": b'{"verdict": "GO"}',
        "metrics/scalars.jsonl": b'{"step": 1}\n' * 50,
        "tb/events.out.tfevents.1.box": os.urandom(3000),
        "checkpoints/step_100/model.safetensors": os.urandom(2000),
        "checkpoints/step_200/model.safetensors": os.urandom(2100),
        "checkpoints/step_200/config.json": b"{}",
        "checkpoints/full_step_100/state.pt": os.urandom(1500),
        "checkpoints/full_step_200/state.pt": os.urandom(1600),
        "checkpoints/full_step_300.tmp/state.pt": os.urandom(10),
    }
    for rel, data in files.items():
        p = run / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return run


def remote_from_local(expected: dict) -> dict:
    """What the Hub would list: LFS info (sha256) for binaries, git blob ids for small text files."""
    out = {}
    for path, local in expected.items():
        data = local.read_bytes()
        lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest(), size=len(data)) if len(data) > 1000 else None
        out[path] = SimpleNamespace(path=path, size=len(data), blob_id=finish.git_blob_id(local), lfs=lfs)
    return out


class FakeHub:
    def __init__(self, files: dict):
        self.files, self.uploads, self.commits = files, [], []

    def list_repo_tree(self, repo, path_in_repo=None, recursive=False, repo_type=None, **kw):
        folder = SimpleNamespace(path=f"{path_in_repo}/metrics")  # folders have no size
        return [folder] + [f for p, f in self.files.items() if p.startswith(path_in_repo + "/")]

    def upload_folder(self, **kw):
        folder = Path(kw["folder_path"])  # what is there during the call (a staging dir is gone afterwards)
        kw["files"] = {f.relative_to(folder).as_posix(): f.read_bytes() for f in folder.rglob("*") if f.is_file()}
        self.uploads.append(kw)

    def create_commit(self, **kw):
        self.commits.append(kw)

    def committed(self, name: str) -> bytes:
        """Bytes of the newest committed operation whose path ends in `name`."""
        return next(op.path_or_fileobj for c in reversed(self.commits) for op in c["operations"]
                    if op.path_in_repo.endswith("/" + name))


def test_expected_files_picks_every_weights_dir_and_the_newest_full_state(tmp_path):
    run = make_run(tmp_path)
    exp = finish.expected_files(run)
    prefix = f"runs/{run.name}"
    assert f"{prefix}/checkpoints/step_200/model.safetensors" in exp
    assert f"{prefix}/checkpoints/step_100/model.safetensors" in exp, "an older weights upload may have failed"
    assert f"{prefix}/checkpoints/full_step_200/state.pt" in exp
    assert not any("full_step_100" in p or ".tmp" in p for p in exp)
    assert f"{prefix}/tb/events.out.tfevents.1.box" in exp and f"{prefix}/config.json" in exp
    assert not any("full_step" in p for p in finish.expected_files(run, expect_full=False))


def test_expected_files_takes_a_full_state_whose_trainer_upload_failed(tmp_path):
    """The trainer marks a full state meant for the Hub (the pre_cooldown one) until its upload succeeds and keeps it
    from rotation meanwhile: finish uploads and verifies a marked one too (not the marker), and --no-full none."""
    run = make_run(tmp_path)
    (run / "checkpoints" / "full_step_100" / finish.UPLOAD_MARK).write_bytes(b"")
    exp = finish.expected_files(run)
    prefix = f"runs/{run.name}"
    assert {f"{prefix}/checkpoints/full_step_{s}/state.pt" for s in (100, 200)} <= set(exp)
    assert not any(p.endswith(finish.UPLOAD_MARK) for p in exp)
    assert not any("full_step" in p for p in finish.expected_files(run, expect_full=False))


def test_git_blob_id_matches_git(tmp_path):
    p = tmp_path / "x.json"
    p.write_bytes(b'{"a": 1}\n')
    if shutil.which("git") is None:
        pytest.skip("git not available")
    want = subprocess.run(["git", "hash-object", str(p)], capture_output=True, text=True).stdout.strip()
    assert finish.git_blob_id(p) == want


def test_verify_detects_missing_size_and_hash(tmp_path):
    run = make_run(tmp_path)
    exp = finish.expected_files(run)
    good = remote_from_local(exp)
    assert finish.verify(FakeHub(good), "o/r", "model", exp) == []
    prefix = f"runs/{run.name}"
    bad = dict(good)
    del bad[f"{prefix}/summary.json"]
    weights = f"{prefix}/checkpoints/step_200/model.safetensors"
    bad[weights] = SimpleNamespace(path=weights, size=1, blob_id="0", lfs=None)
    full = f"{prefix}/checkpoints/full_step_200/state.pt"
    bad[full] = SimpleNamespace(path=full, size=good[full].size, blob_id="0", lfs=SimpleNamespace(sha256="0" * 64))
    problems = finish.verify(FakeHub(bad), "o/r", "model", exp)
    assert any("summary.json: missing" in p for p in problems)
    assert any("model.safetensors: size" in p for p in problems)
    assert any("state.pt: sha256 differs" in p for p in problems)
    assert finish.verify(FakeHub({}), "o/r", "model", {}) != [], "nothing to verify must not count as verified"


@pytest.fixture
def finish_env(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    actions = []
    monkeypatch.setattr(finish, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(finish, "vast_rest", lambda action, timeout=30: actions.append(action) or True)
    monkeypatch.setattr(finish, "vast_cli", lambda action: pytest.fail("CLI fallback must not be needed"))
    for var in ("KITSUNE_OUT_REPO", "CONTAINER_ID", "CONTAINER_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    def go(hub, *extra):
        monkeypatch.setattr(finish, "hf_api", lambda: hub)
        rc = finish.main(["--repo", "Multy123/kitsune-runs", "--runs-root", str(tmp_path / "runs"), *extra])
        return rc, actions
    return run, go


def test_finish_destroys_only_when_verified(finish_env, tmp_path):
    run, go = finish_env
    hub = FakeHub(remote_from_local(finish.expected_files(run)))
    rc, actions = go(hub, "--destroy")
    assert rc == 0 and actions == ["destroy"]
    assert json.loads((tmp_path / "state" / "halt").read_text())["action"] == "destroy"
    live, ckpt = hub.uploads  # the logs first (sync)
    assert ckpt["path_in_repo"] == live["path_in_repo"] == f"runs/{run.name}"
    # finished checkpoints go up in place; the logs from a snapshot copy that no later append can change
    assert Path(ckpt["folder_path"]) == run and "checkpoints/full_step_200/state.pt" in ckpt["allow_patterns"]
    assert Path(live["folder_path"]) != run and not any(p.startswith("checkpoints/") for p in live["allow_patterns"])
    assert live["files"]["metrics/scalars.jsonl"] == (run / "metrics" / "scalars.jsonl").read_bytes()
    kinds = [json.loads(ln)["kind"] for ln in (tmp_path / "state" / "events.jsonl").read_text().splitlines()]
    assert kinds == ["verify", "destroy"]
    # the last infra upload ran after the verify and destroy records and the halt marker were written
    assert [json.loads(ln)["kind"] for ln in hub.committed("events.jsonl").decode().splitlines()] == kinds
    assert json.loads(hub.committed("halt"))["action"] == "destroy"


def test_finish_retries_a_transient_hub_error_on_the_listing_and_the_infra_commit(finish_env, monkeypatch):
    """huggingface_hub retries neither the first page of list_repo_tree nor the create_commit POST: one 503 on the
    verify listing stopped a fully uploaded run (its disk billed until a human looked), one 502 on the infra commit lost
    the box logs with the destroyed disk. finish retries both; a listing that keeps failing still stops."""
    run, go = finish_env
    monkeypatch.setattr(finish, "HUB_RETRY_WAITS", (0, 0))

    class Flaky(FakeHub):
        def __init__(self, files, always=False):
            super().__init__(files)
            self.failed, self.always = [], always

        def fail(self, what):
            if self.always or what not in self.failed:
                self.failed.append(what)
                raise RuntimeError(f"{what}: 503 Service Unavailable")

        def list_repo_tree(self, *a, **kw):  # a generator like the real one: its error comes while it is iterated
            self.fail("tree")
            yield from super().list_repo_tree(*a, **kw)

        def create_commit(self, **kw):
            self.fail("commit")
            super().create_commit(**kw)

    hub = Flaky(remote_from_local(finish.expected_files(run)))
    rc, actions = go(hub, "--destroy")
    assert rc == 0 and actions == ["destroy"] and sorted(hub.failed) == ["commit", "tree"]
    assert len(hub.commits) == 1 and json.loads(hub.committed("halt"))["action"] == "destroy"
    hub = Flaky(remote_from_local(finish.expected_files(run)), always=True)
    rc, actions = go(hub, "--destroy", "--no-sync")
    assert rc == 2 and actions == ["destroy", "stop"] and hub.failed.count("tree") == 3


def test_finish_stop_without_sync_still_uploads_the_infra_logs(finish_env, tmp_path):
    run, go = finish_env
    hub = FakeHub({})
    rc, actions = go(hub, "--stop", "--no-sync", "--reason", "onstart failed at line 7")
    assert rc == 0 and actions == ["stop"] and hub.uploads == []
    assert hub.commits[0]["operations"][0].path_in_repo.startswith(f"runs/{run.name}/infra/")
    assert b"onstart failed at line 7" in hub.committed("events.jsonl")


def test_finish_infra_upload_redacts_the_portal_secrets(tmp_path, monkeypatch):
    """portal.log is the base image's boot output: once a PORTAL_CONFIG reaches the box, caddy_config_manager prints
    the web password, the open-button token and a Bearer header there (a generated uuid when no WEB_PASSWORD is set).
    The infra upload redacts them, and every secret-looking env value in any infra or state file, and keeps the rest
    (the CUDA selection is recorded nowhere else)."""
    pw, obt, hf, uuid = "MyReusedPw_9876", "OBT_SECRET_abc123", "hf_FAKEtoken1234", "Zx8pQ2mK7vR4tY9w"
    monkeypatch.setenv("WEB_PASSWORD", pw)
    monkeypatch.setenv("OPEN_BUTTON_TOKEN", obt)
    monkeypatch.setenv("HF_TOKEN", hf)
    portal = tmp_path / "portal.log"
    portal.write_text("CUDA 13.0 selected (GPU: A100, CC 8.0, Driver 580, Max CUDA 13.0, Forward Compat: no)\n"
                      f"* Your web credentials are: vastai / {pw}\n"
                      f"* Open button token is also valid: {obt}\n"
                      f"* To make API requests, pass an Authorization header (Authorization: Bearer {pw})\n"
                      f"* Your web credentials are: vastai / {uuid}\n"
                      f"(Authorization: Bearer {uuid})\n"
                      f"syncthing --gui-apikey={obt} --no-browser\n", encoding="utf-8")
    kitsune = tmp_path / "kitsune.log"
    kitsune.write_text(f"[bootstrap] pulling data\nHF_TOKEN={hf}\n[train] step 1 loss 2.5\n", encoding="utf-8")
    state = tmp_path / "state"
    state.mkdir()
    (state / "events.jsonl").write_text(json.dumps({"kind": "stop", "reason": f"token {obt}"}) + "\n")
    monkeypatch.setattr(finish, "INFRA_LOGS", [str(portal), str(kitsune)])
    monkeypatch.setattr(finish, "STATE_DIR", state)
    hub = FakeHub({})
    finish.upload_infra(hub, "Multy123/kitsune-runs", "model", "runs/r/infra", dry_run=False)
    sent = {op.path_in_repo: op.path_or_fileobj for c in hub.commits for op in c["operations"]}
    assert set(sent) == {"runs/r/infra/portal.log", "runs/r/infra/kitsune.log", "runs/r/infra/events.jsonl"}
    for data in sent.values():
        assert not any(s.encode() in data for s in (pw, obt, hf, uuid)), data
    log = sent["runs/r/infra/portal.log"].decode()
    assert "CUDA 13.0 selected (GPU: A100, CC 8.0, Driver 580, Max CUDA 13.0, Forward Compat: no)" in log
    assert "credentials are: vastai / <redacted>" in log and "(Authorization: Bearer <redacted>)" in log
    assert b"[train] step 1 loss 2.5" in sent["runs/r/infra/kitsune.log"]


def test_finish_sync_uploads_the_logs_before_the_checkpoints(finish_env, tmp_path):
    """The watchdog's --sync-only gets 10 minutes before the stop: the ~9 GB of a full state not yet on the hub came
    first and could take all of them, and a checkpoint commit that raised skipped the run's logs altogether (rc 0). The
    logs now go first, so they reach the hub either way."""
    run, go = finish_env

    class CheckpointsFail(FakeHub):
        def upload_folder(self, **kw):
            if any(p.startswith("checkpoints/") for p in kw["allow_patterns"]):
                raise RuntimeError("502 Bad Gateway")
            super().upload_folder(**kw)

    hub = CheckpointsFail({})
    rc, actions = go(hub, "--sync-only")
    assert rc == 0 and actions == []
    (live,) = hub.uploads
    assert {"config.json", "summary.json", "metrics/scalars.jsonl"} <= set(live["files"])
    assert live["files"]["metrics/scalars.jsonl"] == (run / "metrics" / "scalars.jsonl").read_bytes()
    kinds = [json.loads(ln)["kind"] for ln in (tmp_path / "state" / "events.jsonl").read_text().splitlines()]
    assert kinds == ["sync_failed"]


def test_finish_sync_uploads_the_checkpoints_when_the_log_upload_fails(finish_env, tmp_path):
    """The reverse holds too: a log upload that raises (the unretried repo_info GET of a snapshot the trainer's close()
    already pushed, say) skipped the checkpoint upload, and --destroy then found the end state missing and stopped the
    box, which kept billing storage. Each part now has its own try; sync still fails as a whole, once."""
    run, go = finish_env
    expected = finish.expected_files(run)
    full = f"runs/{run.name}/checkpoints/full_step_200/state.pt"

    class LogsFail(FakeHub):
        def upload_folder(self, **kw):
            if not any(p.startswith("checkpoints/") for p in kw["allow_patterns"]):
                raise RuntimeError("502 Bad Gateway (repo_info)")
            super().upload_folder(**kw)
            self.files.update(remote_from_local({full: expected[full]}))  # now on the hub

    hub = LogsFail(remote_from_local({p: f for p, f in expected.items() if p != full}))
    rc, actions = go(hub, "--destroy")
    (ckpt,) = hub.uploads
    assert "checkpoints/full_step_200/state.pt" in ckpt["allow_patterns"]
    assert rc == 0 and actions == ["destroy"]
    kinds = [json.loads(ln)["kind"] for ln in (tmp_path / "state" / "events.jsonl").read_text().splitlines()]
    assert kinds[0] == "sync_failed" and kinds.count("sync_failed") == 1


def test_snapshot_copies_a_growing_file_as_it_was(tmp_path):
    src = tmp_path / "run"
    (src / "metrics").mkdir(parents=True)
    (src / "metrics" / "scalars.jsonl").write_bytes(b'{"step": 1}\n' * 3)
    finish.snapshot(src, ["metrics/scalars.jsonl", "gone.tmp"], tmp_path / "stage")
    with open(src / "metrics" / "scalars.jsonl", "ab") as f:
        f.write(b'{"step": 2}\n')
    assert (tmp_path / "stage" / "metrics" / "scalars.jsonl").read_bytes() == b'{"step": 1}\n' * 3
    assert not (tmp_path / "stage" / "gone.tmp").exists()


def test_vastai_cli_timeout_still_falls_back_to_stop(tmp_path, monkeypatch):
    """A hung vastai CLI must not crash finish.py before its destroy -> stop fallback."""
    calls = []

    def hung(argv, **kw):
        calls.append(argv[1])
        raise subprocess.TimeoutExpired(argv, kw.get("timeout"))

    monkeypatch.setattr(finish, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(finish, "vast_rest", lambda action, timeout=30: False)
    monkeypatch.setattr(finish.shutil, "which", lambda name: "/fake/vastai")
    monkeypatch.setattr(finish.subprocess, "run", hung)
    monkeypatch.setenv("CONTAINER_API_KEY", "k")
    monkeypatch.setenv("CONTAINER_ID", "123")
    assert finish.instance_action("destroy", "verified", dry_run=False) is False
    assert calls == ["destroy", "stop"]
    assert json.loads((tmp_path / "state" / "halt").read_text())["action"] == "stop"


def _cli_env(tmp_path, monkeypatch, reply):
    """REST down, the vastai CLI faked: reply(verb) -> (exit code, stdout, stderr)."""
    calls = []

    def run(argv, **kw):
        calls.append(argv[1])
        return subprocess.CompletedProcess(argv, *reply(argv[1]))

    monkeypatch.setattr(finish, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(finish, "vast_rest", lambda action, timeout=30: False)
    monkeypatch.setattr(finish.shutil, "which", lambda name: "/fake/vastai")
    monkeypatch.setattr(finish.subprocess, "run", run)
    monkeypatch.setenv("CONTAINER_API_KEY", "k")
    monkeypatch.setenv("CONTAINER_ID", "123")
    return calls


def test_vastai_cli_refusal_with_exit_0_still_falls_back_to_stop(tmp_path, monkeypatch):
    """vastai 1.8.0 exits 0 when the API refuses or errors (it prints the error and returns): only its own success line
    counts, so a refused destroy still falls back to stop, and a refused stop is reported as failed."""
    prompt = "Are you sure you want to destroy instance 123? This is irreversible and will delete all data. [y/N] "
    calls = _cli_env(tmp_path, monkeypatch, lambda verb: (0, prompt if verb == "destroy" else "nope\n",
                                                           "Failed with error 500: boom"))
    assert finish.instance_action("destroy", "verified", dry_run=False) is False
    assert calls == ["destroy", "stop"]
    assert json.loads((tmp_path / "state" / "halt").read_text())["action"] == "stop"


def test_vastai_cli_success_line_counts(tmp_path, monkeypatch):
    prompt = "Are you sure you want to destroy instance 123? This is irreversible and will delete all data. [y/N] "
    calls = _cli_env(tmp_path, monkeypatch, lambda verb: (0, prompt + "destroying instance 123.\n", ""))
    assert finish.instance_action("destroy", "verified", dry_run=False) is True
    assert calls == ["destroy"]
    assert json.loads((tmp_path / "state" / "halt").read_text())["action"] == "destroy"
    # a refused destroy, then a stop the API accepts: stopped, and the halt marker says so
    calls = _cli_env(tmp_path, monkeypatch, lambda verb: (0, prompt + "nope\n", "") if verb == "destroy"
                     else (0, "stopping instance 123.\n", ""))
    assert finish.instance_action("destroy", "verified", dry_run=False) is True
    assert calls == ["destroy", "stop"]
    assert json.loads((tmp_path / "state" / "halt").read_text())["action"] == "stop"


def test_vast_rest_retries_only_what_a_retry_can_fix(monkeypatch, capsys):
    """Like curl --retry: a 408, 429 or 5xx is tried 3 times (no sleep after the last); any other 4xx (a revoked key,
    no such instance) does not change on a retry, so it goes to the CLI fallback at once, with vast's msg in the log
    (it was retried for 30 s and logged as a bare "HTTP Error 403")."""
    monkeypatch.setenv("CONTAINER_API_KEY", "k")
    monkeypatch.setenv("CONTAINER_ID", "123")
    calls, sleeps = [], []
    monkeypatch.setattr(finish.time, "sleep", sleeps.append)

    def answer(code):
        def urlopen(req, timeout=None):
            calls.append(req.get_method())
            if code == 200:
                return io.BytesIO(b'{"success": true}')
            raise urllib.error.HTTPError(req.full_url, code, "x", {}, io.BytesIO(b'{"success": false, "msg": "nope"}'))
        return urlopen

    for code, ok, tries in ((200, True, 1), (403, False, 1), (404, False, 1), (429, False, 3), (502, False, 3)):
        calls.clear()
        sleeps.clear()
        monkeypatch.setattr(finish.urllib.request, "urlopen", answer(code))
        assert finish.vast_rest("destroy") is ok, code
        assert calls == ["DELETE"] * tries and sleeps == [5, 10][:tries - 1], code
    assert 'HTTP 403: {"success": false, "msg": "nope"}' in capsys.readouterr().out


def test_finish_stops_when_verification_fails(finish_env):
    run, go = finish_env
    remote = remote_from_local(finish.expected_files(run))
    remote.pop(next(p for p in remote if p.endswith("full_step_200/state.pt")))
    rc, actions = go(FakeHub(remote), "--destroy", "--no-sync")
    assert rc == 2 and actions == ["stop"]


def test_finish_dry_run_touches_nothing(finish_env, tmp_path):
    run, go = finish_env
    hub = FakeHub(remote_from_local(finish.expected_files(run)))
    rc, actions = go(hub, "--destroy", "--dry-run")
    assert rc == 0 and actions == [] and hub.uploads == [] and hub.commits == []
    assert not (tmp_path / "state" / "halt").exists()


def test_finish_without_repo_never_destroys(finish_env, monkeypatch, tmp_path):
    run, _ = finish_env
    actions = []
    monkeypatch.setattr(finish, "vast_rest", lambda action, timeout=30: actions.append(action) or True)
    assert finish.main(["--destroy", "--runs-root", str(tmp_path / "runs")]) == 2
    assert actions == ["stop"]


def test_finish_hub_client_has_a_timeout():
    """huggingface_hub's shared client has no timeout by default: a sync's commit POST the Hub accepted and never
    answered would keep the box up until the watchdog. finish.hf_api() bounds it and keeps the hub's request hook."""
    import huggingface_hub
    from huggingface_hub.utils import _http

    try:
        finish.hf_api()
        c = _http.get_session()
        assert (c.timeout.connect, c.timeout.read) == (60, 300)
        assert _http.hf_request_event_hook in c.event_hooks["request"] and c.follow_redirects
    finally:
        huggingface_hub.set_client_factory(_http.default_client_factory)


# -------------------------------------------------------------------------------------------- 01_prepare_data pins

@pytest.fixture(scope="module")
def prep():
    return load_path("prepare_data_01", ROOT / "scripts" / "01_prepare_data.py")


def test_every_upstream_repo_is_pinned(prep):
    repos = {r for r, _ in prep.HF_PARQUET_SOURCES.values()} | {prep.GALGAME_REPO, prep.EMILIA_REPO, prep.EMOLIA_REPO}
    assert repos == set(prep.REVISIONS)
    assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in prep.REVISIONS.values())
    assert prep.REVISIONS[prep.GALGAME_REPO].startswith("3fb86654")
    assert prep.REVISIONS["japanese-asr/whisper_transcriptions.reazonspeech.small"].startswith("c74b52fc")


def test_pins_match_the_local_download_cache(prep):
    """The pins must be the commits data/ was built from (hf_hub_download records them in refs/main)."""
    raw = REAL / "data" / "raw"
    checked = 0
    for repo, sha in prep.REVISIONS.items():
        ref = raw / f"datasets--{repo.replace('/', '--')}" / "refs" / "main"
        if ref.exists():
            assert ref.read_text(encoding="utf-8").strip() == sha, repo
            checked += 1
    if not checked:
        no_real_data(f"no local download cache under {raw}")


def test_downloads_use_the_pinned_revision(prep, tmp_path, monkeypatch):
    seen = []

    class FakeApi:
        def list_repo_files(self, repo, repo_type=None, revision=None):
            seen.append(("list", repo, revision))
            return []

    def fake_download(repo, filename, repo_type=None, revision=None, cache_dir=None):
        seen.append(("get", repo, revision))
        return str(tmp_path / filename)

    monkeypatch.setattr(prep, "HfApi", FakeApi)
    monkeypatch.setattr(prep, "hf_hub_download", fake_download)
    repo = prep.HF_PARQUET_SOURCES["eval_cv8"][0]
    ing = prep.Ingest(tmp_path / "data", tmp_path / "data" / "raw", "eval_cv8", None)
    prep.ingest_hf_parquet(ing, repo, "eval")
    ing.download(repo, "data/eval-00000.parquet")
    prep.ingest_galgame(prep.Ingest(tmp_path / "data", tmp_path / "data" / "raw", "galgame", None), 1)
    assert ("list", repo, prep.REVISIONS[repo]) in seen and ("get", repo, prep.REVISIONS[repo]) in seen
    assert ("list", prep.GALGAME_REPO, prep.REVISIONS[prep.GALGAME_REPO]) in seen


def test_second_opinion_mirror_pins_match_01(prep):
    """02b keeps its own copy of the ReazonSpeech mirror pins; it must read the commits the audio was built from."""
    second = load_path("second_opinion_02b", ROOT / "scripts" / "02b_second_opinion.py")
    pins = dict(second.JOIN_SOURCES.values())
    assert pins == {repo: prep.REVISIONS[repo] for repo in pins}
    assert all(re.fullmatch(r"[0-9a-f]{40}", sha) for sha in (second.MODEL2_REVISION, second.WHISPER_TOK_REVISION))


def test_student_build_reads_the_teacher_commit_the_targets_and_the_card_name():
    """03 keeps its own copy of the teacher pin: the student's init weights, tokenizer and processor must come from the
    commit 02 computed the targets from, and the model card copied into every checkpoint names that commit."""
    import ast

    tree = ast.parse((ROOT / "scripts" / "02_teacher_pass.py").read_text(encoding="utf-8"))  # no torch import
    pin02 = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "MODEL_REVISION" for t in n.targets))
    build = load_path("build_student_03", ROOT / "scripts" / "03_build_student.py")
    card = re.findall(r"\b[0-9a-f]{40}\b", (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8"))
    assert re.fullmatch(r"[0-9a-f]{40}", pin02)
    assert build.TEACHER_REVISION == pin02 and set(card) == {pin02}, (build.TEACHER_REVISION, pin02, card)
    assert build.parse_args([]).teacher_revision == pin02


def test_label_passes_read_the_hub_at_a_pin():
    """Every hub read in 02, 02b and 03 names a revision; without one it silently follows `main`."""
    import ast

    for name in ("02_teacher_pass.py", "02b_second_opinion.py", "03_build_student.py"):
        tree = ast.parse((ROOT / "scripts" / name).read_text(encoding="utf-8"))
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        reads = [c for c in calls if c.func.attr in ("from_pretrained", "list_repo_files")]
        assert reads and all(any(k.arg == "revision" for k in c.keywords) for c in reads), name
        for c in calls:  # HfFileSystem paths carry the revision as datasets/<repo>@<rev>/<file>
            if c.func.attr == "open" and c.args and isinstance(c.args[0], ast.JoinedStr):
                parts = [v.value for v in c.args[0].values if isinstance(v, ast.Constant)]
                assert "@" in parts, name


def test_second_opinion_launcher_watches_its_one_source():
    """The supervisor's blocking mode runs --limit-shards <files under --watch-dir>+1, which is the stuck shard only
    when the watch dir holds exactly the outputs of the --sources being run."""
    line = next(ln for ln in (ROOT / "scripts" / "start_second_opinion.cmd").read_text(encoding="utf-8").splitlines()
                if "run_teacher_pass.py" in ln)
    argv = line.split(">>")[0].split()
    sources = argv[argv.index("--sources") + 1:]
    sources = sources[:next((i for i, a in enumerate(sources) if a.startswith("--")), len(sources))]
    assert len(sources) == 1 and argv[argv.index("--watch-dir") + 1] == f"second_out\\{sources[0]}"


def test_teacher_meta_from_before_the_pin_resumes(tmp_path):
    """A teacher_out/meta.json written before the teacher was pinned has no model_revision: it counts as the pin, so the
    pass resumes and the key is back-filled; a meta.json from another teacher commit still stops it."""
    teacher_pass = load_path("teacher_pass_02", ROOT / "scripts" / "02_teacher_pass.py")
    settings = dict(model=teacher_pass.MODEL_ID, model_revision=teacher_pass.MODEL_REVISION, language="ja",
                    punctuation=True, k=16, save_encoder=False)
    legacy = {key: val for key, val in settings.items() if key != "model_revision"} | dict(eos_token_id=3)
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps(legacy), encoding="utf-8")
    teacher_pass.check_meta(meta, settings)
    assert json.loads(meta.read_text(encoding="utf-8")) == legacy | {"model_revision": teacher_pass.MODEL_REVISION}
    teacher_pass.check_meta(meta, settings)
    meta.write_text(json.dumps(legacy | {"model_revision": "f" * 40}), encoding="utf-8")
    with pytest.raises(SystemExit, match="model_revision"):
        teacher_pass.check_meta(meta, settings)


# ------------------------------------------------------------------------------------------------------ smoke model

def test_smoke_tiny_forward_backward():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    smoke = load_path("smoke_import", ROOT / "docker" / "smoke_import.py")
    out = smoke.tiny_forward_backward()
    assert out["params_with_grad"] > 0 and out["loss"] > 0
