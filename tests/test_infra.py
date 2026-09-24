"""Infra checks: the image recipe, the CI workflow, the vast scripts and the upstream revision pins.

Everything runs on CPU without network or credentials: vastai, the vast REST API and the HF Hub are faked, shell
scripts are only syntax-checked (plus the watchdog's --dry-run), and the Docker image is not built.
"""
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
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
finish = importlib.import_module("finish")
supervise = importlib.import_module("supervise")
launch = importlib.import_module("launch")


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


def test_onstart_fits_vast_limits():
    raw = (VAST / "onstart.sh").read_bytes()
    assert len(raw) < 16 * 1024, "vast's on-start field is limited to 16 KB (it is run from the clone by the stub)"
    raw.decode("ascii")
    text = raw.decode()
    for needle in ("/etc/environment", "ulimit -Sn", "/dev/shm", "KITSUNE_SHARING=file_system",
                   "entrypoint.sh", "https://github.com/Multysquid/Kitsune-Transcribe", "vast/watchdog.sh",
                   "vast/bootstrap.sh", "vast/supervise.py", "/workspace/kitsune.log", "halt", "--rearm",
                   "supervise.lock"):
        assert needle in text, needle


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
    """Stands in for subprocess.run of the vastai CLI; answers searches from a queue of offer lists."""

    def __init__(self, searches):
        self.searches, self.calls = list(searches), []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if argv[1:3] == ["search", "offers"]:
            out = json.dumps(self.searches.pop(0))
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
    def install(searches):
        fake = FakeVastai(searches)
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
                 "cpu_cores_effective>=12", "cpu_ram>=64", "disk_bw>=500", "inet_down>=500",
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


VIAB_CFG = {"sources": ["reazon_small", "galgame"], "eval_sets": ["eval_jsut", "galgame"],
            "selection": "selection/viability.parquet", "student": "students/b20x2560-d4"}
DATA_FILES = ["teacher_out/meta.json", "second_out/meta.json", "selection/viability.parquet",
              "students/b20x2560-d4/config.json", "students/b20x2560-d4/model.safetensors",
              "teacher_out/reazon_small/train-00000.npz", "teacher_out/reazon_small/train-00000.jsonl",
              "second_out/reazon_small/train-00000.jsonl",
              "teacher_out/galgame/train-00000.npz", "teacher_out/galgame/train-00001.npz",
              "teacher_out/galgame/eval-00000.npz", "second_out/galgame/train-00000.jsonl",
              "second_out/galgame/train-00001.jsonl", "second_out/galgame/eval-00000.jsonl",
              "teacher_out/eval_jsut/eval-00000.npz"]


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


SEL_CFG = dict(VIAB_CFG, selection_recipe={"agree_max": 0.5, "agree_max_source": ["galgame=0.4"],
                                           "filter_eval_sets": ["galgame"]})
SEL_ARGS = {"sources": ["reazon_small", "galgame"], "eval_sets": ["eval_jsut", "galgame"], "agree_max": 0.5,
            "agree_max_source": ["galgame=0.4"], "filter_eval_sets": ["galgame"], "out": "C:/laptop/sel.parquet",
            "config": "configs/viability.json"}
SEL_ROWS = [("reazon_small", "train", True, "kept"), ("galgame", "train", True, "kept"),
            ("galgame", "train", False, "agree>0.4"), ("eval_jsut", "eval", True, "kept"),
            ("galgame", "eval", True, "kept")]


def write_selection(path: Path, rows, args=None) -> Path:
    """A selection parquet as make_selection.py writes it: its arguments in the metadata key b"kitsune_selection"."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pa.Table.from_pandas(pd.DataFrame(rows, columns=["source", "split", "keep", "reason"]), preserve_index=False)
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


class FakeTrainer:
    """Plays a script of attempts: (exit code, last step, write a full state?) into one run dir."""

    def __init__(self, runs_root: Path, script):
        self.run_dir, self.script, self.argvs = runs_root / "viability-b20x2560-test", list(script), []

    def __call__(self, argv, env):
        self.argvs.append(list(argv))
        rc, step, full = self.script.pop(0)
        (self.run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text("{}", encoding="utf-8")
        with open(self.run_dir / "metrics" / "scalars.jsonl", "a", encoding="utf-8") as f:
            for s in range(0, step + 1, 10):
                f.write(json.dumps({"step": s, "wall": 0, "elapsed_s": 0, "tag": "loss/total", "value": 1.0}) + "\n")
            f.write('{"step": 99999, "tag": "torn')  # a crash can leave half a line
        if full:
            (self.run_dir / "checkpoints" / f"full_step_{step}").mkdir(parents=True)
        return rc


def run_supervise(tmp_path, monkeypatch, script, state=None):
    runs = tmp_path / "runs"
    trainer = FakeTrainer(runs, script)
    finishes = []
    monkeypatch.setattr(supervise, "run_trainer", trainer)
    monkeypatch.setattr(supervise, "call_finish", lambda args, timeout=None: finishes.append(list(args)) or 0)
    state_path = tmp_path / "state" / "supervise.json"
    if state is not None:
        state_path.parent.mkdir(parents=True)
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


def test_supervise_post_crash_sync_leaves_the_full_state_to_the_stop_path(tmp_path, monkeypatch):
    rc, trainer, finishes, _ = run_supervise(tmp_path, monkeypatch, [(1, 150, True), (0, 900, True)])
    assert finishes[0] == ["--sync-only", "--no-full"] and finishes[-1][0] == "--destroy"


def test_supervise_never_reruns_after_final(tmp_path, monkeypatch):
    state = {"attempts": [{"t0": 0.0, "rc": 0, "step": 10}], "final": {"action": "destroy"}}
    rc, trainer, finishes, _ = run_supervise(tmp_path, monkeypatch, [], state=state)
    assert rc == 0 and trainer.argvs == [] and finishes == []


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
    ckpt, live = hub.uploads
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


def test_finish_stop_without_sync_still_uploads_the_infra_logs(finish_env, tmp_path):
    run, go = finish_env
    hub = FakeHub({})
    rc, actions = go(hub, "--stop", "--no-sync", "--reason", "onstart failed at line 7")
    assert rc == 0 and actions == ["stop"] and hub.uploads == []
    assert hub.commits[0]["operations"][0].path_in_repo.startswith(f"runs/{run.name}/infra/")
    assert b"onstart failed at line 7" in hub.committed("events.jsonl")


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
    raw = ROOT / "data" / "raw"
    checked = 0
    for repo, sha in prep.REVISIONS.items():
        ref = raw / f"datasets--{repo.replace('/', '--')}" / "refs" / "main"
        if ref.exists():
            assert ref.read_text(encoding="utf-8").strip() == sha, repo
            checked += 1
    if not checked:
        pytest.skip("no local data/raw download cache")


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


# ------------------------------------------------------------------------------------------------------ smoke model

def test_smoke_tiny_forward_backward():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    smoke = load_path("smoke_import", ROOT / "docker" / "smoke_import.py")
    out = smoke.tiny_forward_backward()
    assert out["params_with_grad"] > 0 and out["loss"] > 0
