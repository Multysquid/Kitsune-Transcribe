"""Tests for vast/launch.py's label job (--job label: RTX 5090 tiers, 250 GB disk, ranking by the estimated total,
the avoid list, the label env and preflight) and its extent path for the A100 (extent_problems, extent_preflight and
the sizing it feeds into --disk/--storage/--max-hours and KITSUNE_REBUILD_TIMEOUT_MIN). The train job's own tests stay
in tests/test_infra.py, untouched. No network, no vastai CLI, no GPU.
"""
import importlib
import json
import sys
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
VAST = ROOT / "vast"
SHA = "0123456789abcdef0123456789abcdef01234567"
DIGEST_IMAGE = "ghcr.io/multysquid/kitsune-train@sha256:" + "ab" * 32

sys.path.insert(0, str(VAST))
sys.path.insert(0, str(ROOT))
launch = importlib.import_module("launch")
import kitsune  # noqa: E402
from kitsune import extent  # noqa: E402

FULL_SOURCES = ["reazon_small", "reazon_large", "emilia_yodas", "emilia_nc", "galgame"]
EVAL_SETS = ["eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia", "galgame"]


def make_cfg(name: str = "full", inputs: dict | None = None, sources=FULL_SOURCES, eval_sets=EVAL_SETS,
             root: str = "labels/full") -> dict:
    """A run config shaped like configs/full.json (as tests/test_extent.py builds it)."""
    return {"run_name": f"{name}-b20x2560", "student": "students/b20x2560-d4", "data_root": "data",
            "teacher_root": f"{root}/teacher_out", "second_root": f"{root}/second_out",
            "parakeet_root": f"{root}/parakeet_out", "selection": f"{root}/selections/{name}.parquet",
            "extent": {"name": name, "root": root, "inputs": dict(inputs or {})},
            "sources": list(sources), "eval_sets": list(eval_sets),
            "selection_recipe": {"agree_max": 0.5, "agree_max_source": ["emilia_yodas=0.2", "eval_emilia=0.2"],
                                 "filter_eval_sets": [s for s in ("eval_emilia", "galgame") if s in eval_sets],
                                 "partial_second_opinion": []}}


# ------------------------------------------------------------------------------------------------------ fake vastai

OFFERS_5090 = [
    # cheapest $/h, but its traffic is dear: est 0.59 x 16 + 0.05 x 590 + 0.05 x 45 = 41.19
    {"id": 501, "machine_id": 9001, "gpu_name": "RTX 5090", "gpu_ram": 32607, "dph_total": 0.590,
     "inet_down_cost": 0.05, "inet_up_cost": 0.05, "storage_cost": 0.15},
    # est 0.64 x 16 + 0.001 x 590 + 0.002 x 45 = 10.92: ranked first
    {"id": 502, "machine_id": 9002, "gpu_name": "RTX 5090", "gpu_ram": 32607, "dph_total": 0.640,
     "inet_down_cost": 0.001, "inet_up_cost": 0.002, "storage_cost": 0.10},
    # no $/GB: counted at 0.01 each: 0.62 x 16 + 5.9 + 0.45 = 16.27
    {"id": 503, "machine_id": 9003, "gpu_name": "RTX 5090", "gpu_ram": 32607, "dph_total": 0.620},
]


class FakeVastai:
    def __init__(self, searches):
        self.searches, self.calls = list(searches), []

    def __call__(self, argv, **kw):
        import subprocess
        self.calls.append(list(argv))
        if argv[1:3] == ["search", "offers"]:
            out = json.dumps(self.searches.pop(0))
        elif argv[1:3] == ["create", "instance"]:
            out = json.dumps({"success": True, "new_contract": 4242})
        else:
            raise AssertionError(f"unexpected vastai call {argv}")
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")


@pytest.fixture
def fake_vastai(monkeypatch):
    def install(searches):
        fake = FakeVastai(searches)
        monkeypatch.setattr(launch.shutil, "which", lambda name: "/fake/vastai" if name == "vastai" else None)
        monkeypatch.setattr(launch.subprocess, "run", fake)
        return fake
    return install


def label_args(*extra):
    return ["--job", "label", "--data-repo", "Multy123/kitsune-data", "--sha", SHA, "--image", DIGEST_IMAGE,
            "--skip-git-checks", *extra]


@pytest.fixture(autouse=True)
def passing_label_preflight(monkeypatch, request):
    """--job label refuses --no-hf-check (it pins KITSUNE_DATA_REVISION), so the HF side is faked as passing; the
    tests of the preflight and the avoid list themselves keep the real functions."""
    if "preflight" in request.node.name or "avoided" in request.node.name:
        return
    monkeypatch.setattr(launch, "config_at", lambda sha, c: make_cfg(Path(c).stem))
    monkeypatch.setattr(launch, "label_preflight", lambda *a, **kw: ("d" * 40, [], []))
    monkeypatch.setattr(launch, "avoided_machines", lambda *a, **kw: (set(), []))


def test_label_job_refuses_no_hf_check(fake_vastai, capsys):
    fake_vastai([OFFERS_5090])
    assert launch.main(label_args("--yes", "--no-hf-check")) == 1
    assert "needs the HF preflight" in capsys.readouterr().out


def test_label_ranking_drops_offers_over_max_dph_before_the_total():
    """The est_total winner at $1.05/h must not block a run when an offer under --max-dph exists."""
    job = launch.JOBS["label"]
    pricey = {"id": 1, "machine_id": 1, "dph_total": 1.05, "inet_down_cost": 0.0, "inet_up_cost": 0.0}
    cheap = {"id": 2, "machine_id": 2, "dph_total": 0.95, "inet_down_cost": 5.0, "inet_up_cost": 5.0}
    assert launch.rank_offers([pricey, cheap], job)[0]["id"] == 1  # without the cap the total decides
    assert [o["id"] for o in launch.rank_offers([pricey, cheap, {"id": 3}], job, max_dph=1.0)] == [2]


def env_of(create: list[str]) -> dict[str, str]:
    assert create.count("--env") == 1 and not any(a == "-e" for a in create)
    value = create[create.index("--env") + 1]
    pairs = value.split(" ")
    assert pairs[::2] == ["-e"] * (len(pairs) // 2)
    return dict(p.split("=", 1) for p in pairs[1::2])


def test_label_job_query_disk_env_and_label(fake_vastai, capsys):
    fake = fake_vastai([OFFERS_5090])
    assert launch.main(label_args("--yes")) == 0
    search = next(c for c in fake.calls if c[1:3] == ["search", "offers"])
    terms = search[3].split(" ")
    for t in ("gpu_name=RTX_5090", "gpu_ram>=30", "reliability>=0.97", "disk_bw>=300", "inet_up>=100",
              "num_gpus=1", "verified=true", "rentable=true", "cuda_vers>=13.0", "cpu_cores_effective>=16",
              "cpu_ram>=64", "inet_down>=500", "direct_port_count>=1", "disk_space>=250"):
        assert t in terms, t
    assert search[4:] == ["--type", "on-demand", "-o", "dph", "--storage", "250", "--raw"]
    create = next(c for c in fake.calls if c[1:3] == ["create", "instance"])
    assert create[3] == "502", "ranked by est_total, not $/h"
    assert create[create.index("--disk") + 1] == "250"
    assert create[create.index("--label") + 1] == "kitsune-label-full-0123456"
    assert not any("HF_TOKEN" in a for a in create)
    env = env_of(create)
    assert env["KITSUNE_JOB"] == "label" and env["KITSUNE_SHA"] == SHA
    assert env["KITSUNE_CONFIG"] == "configs/full.json"
    assert env["KITSUNE_LABEL_CONFIGS"] == "configs/full.json,configs/full_sub3k.json"
    assert env["KITSUNE_DATA_REPO"] == "Multy123/kitsune-data" and env["KITSUNE_MAX_HOURS"] == "30"
    assert env["KITSUNE_WATCHDOG_SYNC_LEAD_S"] == "1800" and env["KITSUNE_WATCHDOG_ORPHAN_S"] == "900"
    assert env["KITSUNE_IMAGE"] == DIGEST_IMAGE and env["KITSUNE_MACHINE_ID"] == "9002"
    assert env["KITSUNE_DPH"] == "0.6400" and env["TZ"] == "UTC"
    assert "KITSUNE_OUT_REPO" not in env and "KITSUNE_STEAL_LEASE" not in env
    out = capsys.readouterr().out
    assert "est$" in out and "$/GBmo" in out
    # the cost line: $/h x 16 h + 590 GB down + 45 GB up at the host's $/GB, and the cap at 30 h
    assert "x ~16 h (12-24; cap 30 h)" in out and "~590 GB down x $0.001/GB" in out and "= ~$10.92" in out
    assert "(cap = ~$19.88)" in out  # 0.64 x 30 + 0.59 + 0.09


def test_label_optional_env_and_steal_lease(fake_vastai):
    fake = fake_vastai([OFFERS_5090])
    assert launch.main(label_args("--yes", "--steal-lease", "--cohere-procs", "1", "--no-self-stop",
                                  "--label-configs", "configs/full.json")) == 0
    env = env_of(next(c for c in fake.calls if c[1:3] == ["create", "instance"]))
    assert env["KITSUNE_STEAL_LEASE"] == "1" and env["KITSUNE_COHERE_PROCS"] == "1"
    assert env["KITSUNE_NO_SELF_STOP"] == "1" and env["KITSUNE_LABEL_CONFIGS"] == "configs/full.json"


def test_label_strict_tier_falls_back_and_hints_without_direct_port(fake_vastai, capsys):
    fake = fake_vastai([[], OFFERS_5090[:1]])
    assert launch.main(label_args("--dry-run")) == 0
    searches = [c for c in fake.calls if c[1:3] == ["search", "offers"]]
    assert "reliability>=0.97" in searches[0][3] and "reliability>=0.97" not in searches[1][3]
    assert "gpu_name=RTX_5090" in searches[1][3]
    fake = fake_vastai([[], []])
    assert launch.main(label_args("--dry-run")) == 1
    out = capsys.readouterr().out
    hint = next(ln for ln in out.splitlines() if ln.startswith("hint:")).split("vastai ", 1)[1]
    assert "direct_port_count" not in hint and "gpu_name=RTX_5090" in hint and "--storage 250" in hint
    assert "no RTX 5090 offer" in out


def test_label_avoids_machines(fake_vastai, capsys):
    fake = fake_vastai([OFFERS_5090])
    assert launch.main(label_args("--yes", "--avoid-machine", "9002")) == 0
    create = next(c for c in fake.calls if c[1:3] == ["create", "instance"])
    assert create[3] == "503"
    # every offer of the strict tier avoided: the next tier is searched
    fake = fake_vastai([OFFERS_5090[:1], OFFERS_5090])
    assert launch.main(label_args("--dry-run", "--avoid-machine", "9001", "--avoid-machine", "9002")) == 0
    assert "all 1 on avoided machines" in capsys.readouterr().out


def test_label_needs_no_out_repo_but_train_does(fake_vastai, capsys):
    fake_vastai([OFFERS_5090])
    assert launch.main(label_args("--dry-run")) == 0
    with pytest.raises(SystemExit) as e:
        launch.main(["--data-repo", "Multy123/kitsune-data", "--sha", SHA, "--image", DIGEST_IMAGE, "--no-hf-check"])
    assert e.value.code == 2
    assert "the following arguments are required: --out-repo" in capsys.readouterr().err


def test_label_disk_gb_refused_below_the_minimum(fake_vastai, capsys):
    fake_vastai([OFFERS_5090])
    assert launch.main(label_args("--dry-run", "--disk-gb", "150")) == 1
    assert f"below the {launch.LABEL_DISK_MIN_GB} GB" in capsys.readouterr().out
    fake = fake_vastai([OFFERS_5090])
    assert launch.main(label_args("--yes", "--disk-gb", "300")) == 0
    create = next(c for c in fake.calls if c[1:3] == ["create", "instance"])
    assert create[create.index("--disk") + 1] == "300"
    assert "disk_space>=300" in next(c for c in fake.calls if c[1:3] == ["search", "offers"])[3]


def test_train_job_spec_is_todays_constants():
    train = launch.JOBS["train"]
    assert (train.tiers, train.base_filter, train.disk_gb) == (launch.TIERS, launch.HOST_FILTER, launch.DISK_GB)
    assert launch.job_query(train, launch.TIERS[0][1], launch.DISK_GB) == launch.build_query(launch.TIERS[0][1])
    assert launch.search_args("q") == launch.search_args("q", launch.DISK_GB)
    assert launch.host_filter(["a", "disk_space>=150"], 400) == ["a", "disk_space>=400"]


# ------------------------------------------------------------------------------------------------------ fake hub


class FakeHub:
    """HfApi + hf_hub_download over an in-memory repo {path: bytes}."""

    def __init__(self, files: dict[str, bytes], private=True, lfs: dict[str, str] | None = None):
        self.files, self.private, self.lfs, self.calls = files, private, lfs or {}, []

    def dataset_info(self, repo, files_metadata=False):
        sib = [types.SimpleNamespace(rfilename=f, size=len(b)) for f, b in self.files.items()]
        return types.SimpleNamespace(sha="d" * 40, private=self.private, siblings=sib)

    def list_repo_files(self, repo, repo_type=None, revision=None):
        return list(self.files)

    def get_paths_info(self, repo, paths, repo_type=None, revision=None):
        return [types.SimpleNamespace(path=p, lfs={"sha256": self.lfs[p]} if p in self.lfs else None) for p in paths]

    def download(self, repo, path, repo_type=None, revision=None, local_dir=None):
        self.calls.append(path)
        out = Path(local_dir) / path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.files[path])
        return str(out)


@pytest.fixture
def hub(monkeypatch):
    def install(files, **kw):
        fake = FakeHub(files, **kw)
        monkeypatch.setattr(launch, "_hub", lambda: (fake, fake.download))
        return fake
    return install


PK_PINS = {"model.safetensors": "a" * 64, "config.json": "b" * 64}


@pytest.fixture
def parakeet_pins(monkeypatch):
    fake = types.SimpleNamespace(PARAKEET_PATH="models/parakeet-tdt_ctc-0.6b-ja-hf", PARAKEET_FILES=dict(PK_PINS))
    monkeypatch.setitem(sys.modules, "kitsune.parakeet", fake)
    monkeypatch.setattr(kitsune, "parakeet", fake, raising=False)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "auth_check", lambda *a, **k: None)
    return fake


def label_repo(**extra) -> dict[str, bytes]:
    files = {"teacher_out/meta.json": b"{}",
             **{f"teacher_out/{g}/eval-00000{ext}": b"x" for g in extent.GATE_SETS for ext in (".npz", ".jsonl")},
             **{f"models/parakeet-tdt_ctc-0.6b-ja-hf/{n}": b"m" for n in PK_PINS}}
    files.update(extra)
    return files


def label_lfs():
    return {f"models/parakeet-tdt_ctc-0.6b-ja-hf/{n}": h for n, h in PK_PINS.items()}


def test_label_preflight_clean_and_resume_plan(hub, parakeet_pins):
    full, sub = make_cfg("full"), make_cfg("full_sub3k", {"reazon_large": 2, "galgame": 3})
    hub(label_repo(**{"labels/full/teacher_out/galgame/train-00000.npz": b"x",
                      "labels/full/teacher_out/galgame/train-00001.npz": b"x",
                      "labels/full/parakeet_out/galgame/train-00000.npz": b"x"}), lfs=label_lfs())
    rev, problems, notes = launch.label_preflight("r", SHA, full, {"configs/full.json": full,
                                                                    "configs/full_sub3k.json": sub})
    assert rev == "d" * 40 and problems == [], problems
    resume = next(n for n in notes if n.startswith("resume plan"))
    assert "parakeet_out/galgame 1 stems" in resume and "teacher_out/galgame 2 stems" in resume
    assert any("GB" in n and "after the label run" in n for n in notes)


def test_label_preflight_refusals(hub, parakeet_pins, monkeypatch):
    full = make_cfg("full")
    now = 1_800_000_000.0
    lease = json.dumps({"run_id": "r1", "container_id": "C1", "machine_id": 7, "heartbeat": now - 600,
                        "released": False}).encode()
    bad_lfs = dict(label_lfs(), **{"models/parakeet-tdt_ctc-0.6b-ja-hf/model.safetensors": "f" * 64})
    files = label_repo(**{"labels/full/COMPLETE.json": b"{}", "labels/full/LEASE.json": lease})
    del files["teacher_out/eval_cv8/eval-00000.jsonl"]
    del files["models/parakeet-tdt_ctc-0.6b-ja-hf/config.json"]
    hub(files, private=False, lfs=bad_lfs)
    outside = make_cfg("full_sub3k", sources=["reazon_large"], eval_sets=[], root="labels/other")
    called = []
    monkeypatch.setattr(launch, "data_problems", lambda *a: called.append(a) or [])
    monkeypatch.setattr(launch, "selection_problems", lambda *a, **k: called.append(a) or [])
    _, problems, _ = launch.label_preflight("r", SHA, full, {"configs/full.json": full, "sub": outside}, now=now)
    text = "\n".join(problems)
    for piece in ("not private", "COMPLETE.json exists", "held by a live box", "teacher_out/eval_cv8/",
                  "Parakeet model files missing", "model.safetensors has sha256 ffffffffffff", "sub: extent.root"):
        assert piece in text, (piece, text)
    assert called == [], "the label preflight never checks the trainer's data"
    # --steal-lease turns the live lease into a note; a stale or released lease is no problem
    _, problems, notes = launch.label_preflight("r", SHA, full, {"configs/full.json": full}, steal_lease=True, now=now)
    assert not any("live box" in p for p in problems) and any("--steal-lease" in n for n in notes)
    assert not launch.lease_live(json.loads(lease), now + launch.LEASE_MAX_AGE_S)
    assert not launch.lease_live(dict(json.loads(lease), released=True), now)
    assert launch.lease_live(dict(json.loads(lease), heartbeat="2027-01-15T08:00:00Z"),
                             launch._heartbeat_s("2027-01-15T08:10:00Z"))


def test_label_preflight_without_parakeet_pins(hub, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "auth_check", lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "kitsune.parakeet", types.SimpleNamespace())  # no PARAKEET_FILES yet
    monkeypatch.setattr(kitsune, "parakeet", sys.modules["kitsune.parakeet"], raising=False)
    full = make_cfg("full")
    hub(label_repo())
    _, problems, _ = launch.label_preflight("r", SHA, full, {"configs/full.json": full})
    assert any("no Parakeet pins" in p for p in problems)


def test_avoided_machines_from_label_end(hub):
    ends = {"label_runs/a/label_end.json": {"class": "host_failure", "machine_id": 9002},
            "label_runs/b/label_end.json": {"class": "slow_host", "machine_id": "9003"},
            "label_runs/c/label_end.json": {"class": "budget", "machine_id": 9001},
            "label_runs/d/label_end.json": {"class": "integrity", "machine_id": 9004}}
    hub({k: json.dumps(v).encode() for k, v in ends.items()})
    avoid, notes = launch.avoided_machines("r", "d" * 40)
    assert avoid == {"9002", "9003"} and len(notes) == 2


def test_label_main_runs_the_preflight_and_avoid_list(fake_vastai, hub, parakeet_pins, monkeypatch, capsys):
    full, sub = make_cfg("full"), make_cfg("full_sub3k", {"reazon_large": 2})
    cfgs = {"configs/full.json": full, "configs/full_sub3k.json": sub}
    monkeypatch.setattr(launch, "config_at", lambda sha, c: cfgs[c])
    called = []
    monkeypatch.setattr(launch, "hf_preflight", lambda *a: called.append(a) or (None, []))
    hub(label_repo(**{"label_runs/a/label_end.json": json.dumps({"class": "host_failure",
                                                                 "machine_id": 9002}).encode()}), lfs=label_lfs())
    fake = fake_vastai([OFFERS_5090])
    args = label_args("--yes")
    assert launch.main(args) == 0, capsys.readouterr().out
    assert called == []
    create = next(c for c in fake.calls if c[1:3] == ["create", "instance"])
    assert create[3] == "503", "9002 ended as host_failure"
    env = env_of(create)
    assert env["KITSUNE_DATA_REVISION"] == "d" * 40 and env["KITSUNE_MACHINE_ID"] == "9003"
    assert "resume plan" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------- A100 extent


def stem(name, step, hours, size, split="train"):
    return {"stem": name, "split": split, "step": step, "rows": 1, "hours": hours, "ids_sha256": "",
            "shard_bytes": size}


def source(inputs):
    stems = [st for i in inputs for st in i["stems"]]
    return {"repo": "r", "n_listed": len(inputs), "rows": len(stems), "hours": sum(st["hours"] for st in stems),
            "bytes": sum(i["bytes"] for i in inputs),
            "inputs": [dict(i, input=f"f{n}", ordinal=n) for n, i in enumerate(inputs)]}


def toy_record() -> dict:
    return {"schema": extent.RECORD_SCHEMA, "name": "full", "root": "labels/full",
            "canonical_version": extent.CANONICAL_VERSION, "names": ["reazon_small", "reazon_large"], "inputs": {},
            "sources": {
                "reazon_small": source([{"bytes": 2e9, "stems": [stem("train-00000", "reazon_small", 20.0, 1.9e9)]}]),
                "reazon_large": source([
                    {"bytes": 20e9, "stems": [stem("train-00000", "reazon_large", 200.0, 19e9)]},
                    {"bytes": 20e9, "stems": []},
                    {"bytes": 20e9, "stems": [stem("train-00001", "reazon_large", 400.0, 38e9)]}])}}


def sub_cfg():
    return make_cfg("sub", {"reazon_large": 2}, sources=["reazon_large"], eval_sets=[])


def sub_files(cfg) -> list[str]:
    return ["labels/full/COMPLETE.json", "labels/full/extent.json", "labels/full/teacher_out/meta.json",
            "labels/full/second_out/meta.json", cfg["selection"],
            "labels/full/teacher_out/reazon_large/train-00000.npz",
            "labels/full/teacher_out/reazon_large/train-00000.jsonl",
            "labels/full/second_out/reazon_large/train-00000.jsonl",
            *(f"students/b20x2560-d4/{n}" for n in launch.STUDENT_FILES)]


def test_extent_problems():
    cfg, rec = sub_cfg(), toy_record()
    files = sub_files(cfg)
    assert launch.extent_problems(files, cfg, rec) == []
    missing = lambda f: launch.extent_problems([x for x in files if x != f], cfg, rec)  # noqa: E731
    assert any("COMPLETE.json" in p for p in missing("labels/full/COMPLETE.json"))
    assert any("train-00000.npz" in p for p in missing("labels/full/teacher_out/reazon_large/train-00000.npz"))
    assert any("second_out/reazon_large/train-00000.jsonl" in p
               for p in missing("labels/full/second_out/reazon_large/train-00000.jsonl"))
    assert any("config.json" in p for p in missing("students/b20x2560-d4/config.json"))
    assert any("selections/sub.parquet" in p for p in missing(cfg["selection"]))
    # the subset's own stems only: the third input (train-00001) is outside reazon_large=2
    assert launch.extent_problems(files, cfg, rec) == []
    bad = dict(cfg, extent=dict(cfg["extent"], inputs={"reazon_large": 9999}))
    assert launch.extent_problems(files, bad, rec)


def selection_parquet(path: Path, cfg: dict, extent_arg, kept=None):
    recipe = cfg["selection_recipe"]
    args = dict(sources=cfg["sources"], eval_sets=cfg["eval_sets"], agree_max=recipe["agree_max"],
                agree_max_source=recipe["agree_max_source"], filter_eval_sets=recipe["filter_eval_sets"],
                partial_second_opinion=[])
    if extent_arg is not None:
        args["extent"] = extent_arg
    meta = dict(args=args, kept=kept or {"reazon_large/train": {"utts": 1, "hours": 100.0}})
    table = pa.table(dict(id=["a"], source=["reazon_large"], split=["train"], keep=[True], reason=["kept"],
                          teacher_file=["reazon_large/train-00000"]))
    pq.write_table(table.replace_schema_metadata({b"kitsune_selection": json.dumps(meta).encode()}), path)
    return path


def test_selection_problems_compares_the_extent(tmp_path):
    cfg = sub_cfg()
    want = {"name": "sub", "inputs": {"reazon_large": 2}}
    ok = selection_parquet(tmp_path / "ok.parquet", cfg, want)
    assert launch.selection_problems(ok, "sel", cfg) == []
    other = selection_parquet(tmp_path / "other.parquet", cfg, {"name": "sub", "inputs": {"reazon_large": 3}})
    assert any("built with extent" in p for p in launch.selection_problems(other, "sel", cfg))
    none = selection_parquet(tmp_path / "none.parquet", cfg, None)
    assert any("built with extent None" in p for p in launch.selection_problems(none, "sel", cfg))
    # the viability path: no extent on either side
    viab = {k: v for k, v in cfg.items() if k != "extent"}
    assert launch.selection_problems(none, "sel", viab) == []


def extent_hub_files(tmp_path, cfg) -> dict[str, bytes]:
    sel = selection_parquet(tmp_path / "sel.parquet", cfg, {"name": "sub", "inputs": {"reazon_large": 2}},
                            kept={"reazon_large/train": {"utts": 9, "hours": 500.0},
                                  "reazon_large/eval": {"utts": 2, "hours": 100.0}})
    files = {f: b"x" for f in sub_files(cfg)}
    files["labels/full/extent.json"] = json.dumps(toy_record()).encode()
    files[cfg["selection"]] = sel.read_bytes()
    return files


def test_extent_preflight_sizing(tmp_path, hub):
    cfg = sub_cfg()
    hub(extent_hub_files(tmp_path, cfg))
    problems, size = launch.extent_preflight("r", "d" * 40, cfg)
    assert problems == []
    # reazon_small (a dependency) + reazon_large's first 2 inputs; kept hours at 57 GB / 600 h
    assert size["down_gb"] == pytest.approx(42) and size["sel_gb"] == pytest.approx(57)
    assert size["disk_gb"] == 250 and size["rebuild_timeout_min"] == 57
    files = extent_hub_files(tmp_path, cfg)
    del files["labels/full/COMPLETE.json"]
    hub(files)
    assert any("COMPLETE.json" in p for p in launch.extent_preflight("r", "d" * 40, cfg)[0])
    files = extent_hub_files(tmp_path, cfg)
    del files["labels/full/extent.json"]
    hub(files)
    assert launch.extent_preflight("r", "d" * 40, cfg) == (
        ["r: no labels/full/extent.json: the label box writes it when it finishes the root"], None)


OFFERS_A100 = [{"id": 111, "machine_id": 1, "gpu_name": "A100 SXM4", "gpu_ram": 40960, "dph_total": 0.672,
                "inet_down_cost": 0.01, "inet_up_cost": 0.02}]


def test_train_with_an_extent_config_sizes_the_box(tmp_path, fake_vastai, hub, monkeypatch, capsys):
    cfg = sub_cfg()
    monkeypatch.setattr(launch, "config_at", lambda sha, c: cfg)
    seen = []
    monkeypatch.setattr(launch, "data_problems", lambda *a: seen.append(a) or ["never"])
    monkeypatch.setattr(launch, "hf_preflight", lambda data, out, c: ("d" * 40, []))
    hub(extent_hub_files(tmp_path, cfg))
    fake = fake_vastai([OFFERS_A100])
    args = ["--data-repo", "Multy123/kitsune-data", "--out-repo", "Multy123/kitsune-runs", "--sha", SHA,
            "--image", DIGEST_IMAGE, "--skip-git-checks", "--config", "configs/full_sub3k.json", "--yes"]
    assert launch.main(args) == 0, capsys.readouterr().out
    search = next(c for c in fake.calls if c[1:3] == ["search", "offers"])
    assert "disk_space>=250" in search[3] and search[search.index("--storage") + 1] == "250"
    assert "gpu_name in [A100_SXM4]" in search[3] and "reliability>=0.98" in search[3]
    create = next(c for c in fake.calls if c[1:3] == ["create", "instance"])
    assert create[create.index("--disk") + 1] == "250"
    env = env_of(create)
    assert env["KITSUNE_REBUILD_TIMEOUT_MIN"] == "57" and env["KITSUNE_MAX_HOURS"] == "6.45"  # 5.5 + 57 / 60
    assert env["KITSUNE_OUT_REPO"] == "Multy123/kitsune-runs" and "KITSUNE_JOB" not in env
    assert "~42 GB down" in capsys.readouterr().out
    # an explicit --disk-gb below the sizing is refused; above it wins
    fake_vastai([OFFERS_A100])
    assert launch.main([*args[:-1], "--disk-gb", "200", "--dry-run"]) == 1
    fake = fake_vastai([OFFERS_A100])
    assert launch.main([*args[:-1], "--disk-gb", "400", "--max-hours", "7", "--yes"]) == 0
    create = next(c for c in fake.calls if c[1:3] == ["create", "instance"])
    assert create[create.index("--disk") + 1] == "400" and env_of(create)["KITSUNE_MAX_HOURS"] == "7"


def test_train_extent_config_under_no_hf_check_needs_explicit_sizing(fake_vastai, capsys):
    """Without the HF preflight there is no extent sizing: the 150 GB default disk cannot hold a full rebuild."""
    fake_vastai([OFFERS_5090])
    args = ["--data-repo", "Multy123/kitsune-data", "--out-repo", "Multy123/out", "--sha", SHA, "--image",
            DIGEST_IMAGE, "--skip-git-checks", "--no-hf-check", "--config", "configs/full_sub3k.json", "--yes"]
    assert launch.main(args) == 1
    assert "pass --disk-gb and --max-hours" in capsys.readouterr().out
