"""The full-data boxes on vast (WP3): vast/launch.py --job full, vast/finish.py for a full box and --abort,
vast/watchdog.sh on train_hb, vast/blocklist.json.

launch: the box registry (kitsune/fullrun.py, tests/fixtures_full.tiny_registry) read at the commit the box runs (a
fake git serves a temp checkout), the query, env table and label of box p01 and the 2-GPU box full, the client filter
(verification, max rental, RAM per GPU, avoided machines), the ranking by the estimated total, --machine, --tier a100,
the flag refusals, full_preflight's refusals (configs, tools and speed_probe kinds or args flags missing at the sha,
students, extra files, the scratch repo, the selection sidecar via a stub kitsune.devslice, --resume's Hub summary and
run ids), the blocklist and the gate refusals. finish: --job full is lean with its infra under full/box-<box>/, never uploads or expects SCRATCH_MARK,
and --abort destroys a box without a run dir, stops one with a run dir, and is --stop --no-sync for any other job.
watchdog: train_hb stale -> sync and stop with the box reason; alert mode writes watchdog_alerts.jsonl and re-arms.
No network, no vastai CLI, no GPU; the shell tests need Git Bash.
"""
import copy
import importlib
import json
import os
import re
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vast"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
launch = importlib.import_module("launch")
finish = importlib.import_module("finish")
import kitsune  # noqa: E402
from fixtures_full import STUDY_WEIGHTS, tiny_registry  # noqa: E402
from kitsune import fullrun, prereg  # noqa: E402
from test_label_onstart import VAST, calls, need_bash, watchdog_env  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"
DIGEST_IMAGE = "ghcr.io/multysquid/kitsune-train@sha256:" + "ab" * 32
DATA, RUNS, SCRATCH = "Multy123/kitsune-data", "Multy123/kitsune-runs", "Multy123/kitsune-scratch"
SIZING = dict(down_gb=571.3, shard_gb=1100.0, sel_gb=90.0, stores=1, labels_gb=23.4, hours=13000.0, extra_gb=25.0,
              disk_gb=1400, rebuild_timeout_min=387)
TOOLS = {"kitsune/full_queue.py": "", "scripts/04_distill.py": "", "scripts/05_evaluate.py": "",
         "kitsune/quant.py": "", "tools/whisper_eval.py": "",
         "tools/speed_probe.py": 'KINDS = ("aed", "cohere", "ctc", "parakeet-ctc", "parakeet-tdt")\n'}
DAY = 86400


def offer(i, machine, dph, **kw):
    """A 1x RTX 5090 offer that passes the full client filter unless kw says otherwise."""
    return dict({"id": i, "machine_id": machine, "gpu_name": "RTX 5090", "gpu_ram": 32607, "num_gpus": 1,
                 "dph_total": dph, "reliability": 0.99, "verification": "verified", "duration": 30 * DAY,
                 "cpu_ram": 64439, "inet_down_cost": 0.002, "inet_up_cost": 0.002, "storage_cost": 0.1}, **kw)


# ============================================================================================ the fake checkout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A checkout at SHA: the tiny registry and its configs, the tools the items run; launch's git, git_show,
    config_at and working-tree reads serve it."""
    root = tmp_path / "repo"
    reg = tiny_registry(root, write_boxes=True)
    for rel, text in TOOLS.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")

    def at(sha, rel):
        assert sha == SHA, sha
        f = root / rel
        if not f.is_file():
            raise subprocess.CalledProcessError(128, ["git", "show", f"{sha}:{rel}"])
        return f.read_bytes()

    def git(*a):
        if a[:2] == ("cat-file", "-e"):
            at(*a[2].split(":", 1))
            return ""
        if a[:1] == ("rev-parse",):
            return SHA
        return ""

    monkeypatch.setattr(launch, "git_show", at)
    monkeypatch.setattr(launch, "git", git)
    monkeypatch.setattr(launch, "config_at", lambda sha, c: json.loads(at(sha, c)))
    monkeypatch.setattr(launch, "worktree_file", lambda rel: None)
    # the quant go signal's git reads: a verdict's commit is an ancestor of SHA with the same quant code
    monkeypatch.setattr(launch, "git_ancestry", lambda old, new: "ancestor")
    monkeypatch.setattr(launch, "git_blob", lambda sha, p: f"blob:{p}")
    return SimpleNamespace(root=root, reg=reg)


def write_reg(root: Path, reg: dict):
    (root / fullrun.BOXES_FILE).write_text(json.dumps(reg, indent=2), encoding="utf-8")


# ================================================================================================ launch main


class FakeVastai:
    def __init__(self, searches, instances=()):
        self.searches, self.calls, self.instances = list(searches), [], list(instances)

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        if argv[1:3] == ["search", "offers"]:
            out = json.dumps(self.searches.pop(0) if self.searches else [])
        elif argv[1:3] == ["create", "instance"]:
            out = json.dumps({"success": True, "new_contract": 777})
        elif argv[1:3] == ["show", "instances"]:
            out = json.dumps(self.instances)
        else:
            raise AssertionError(f"unexpected vastai call {argv}")
        return subprocess.CompletedProcess(argv, 0, stdout=out, stderr="")


@pytest.fixture
def full_launch(repo, monkeypatch):
    """launch.main for --job full with the vastai CLI faked and the Hub side passing (hf_preflight, extent_preflight
    recording its extra_gb, full_preflight recording its arguments, no label or gate avoids)."""
    seen = {}
    monkeypatch.setattr(launch, "hf_preflight", lambda data, out, c: ("d" * 40, []))

    def extent_preflight(data, rev, cfg, extra_gb=0.0):
        seen["extra_gb"], seen["cfg"] = extra_gb, cfg
        return [], dict(SIZING, extra_gb=extra_gb)

    def full_preflight(*a, **kw):
        seen["preflight"] = (a, kw)
        return list(seen.get("preflight_problems", [])), ["full preflight ok"]

    monkeypatch.setattr(launch, "extent_preflight", extent_preflight)
    monkeypatch.setattr(launch, "full_preflight", full_preflight)
    monkeypatch.setattr(launch, "avoided_machines", lambda data, rev: (set(), []))
    monkeypatch.setattr(launch, "gate_refusals", lambda out: (dict(seen.get("gates", {})), []))

    def go(searches, *args, instances=()):
        fake = FakeVastai(searches, instances)
        monkeypatch.setattr(launch.shutil, "which", lambda name: "/fake/vastai" if name == "vastai" else None)
        monkeypatch.setattr(launch.subprocess, "run", fake)
        rc = launch.main(["--job", "full", "--data-repo", DATA, "--out-repo", RUNS, "--sha", SHA, "--image",
                          DIGEST_IMAGE, "--skip-git-checks", *args])
        return rc, fake
    go.seen = seen
    return go


def env_of(create: list[str]) -> dict[str, str]:
    value = create[create.index("--env") + 1]
    return dict(p.split("=", 1) for p in value.split(" ")[1::2])


def created(fake) -> list[str] | None:
    return next((c for c in fake.calls if c[1:3] == ["create", "instance"]), None)


def search_query(fake, n: int = 0) -> str:
    return [c for c in fake.calls if c[1:3] == ["search", "offers"]][n][3]


def test_box_p01_rents_one_5090_with_the_registrys_numbers(full_launch, capsys):
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p01", "--scratch-repo", SCRATCH, "--yes")
    out = capsys.readouterr().out
    assert rc == 0, out
    search = next(c for c in fake.calls if c[1:3] == ["search", "offers"])
    q = search[3].split(" ")
    for term in ("gpu_name=RTX_5090", "gpu_ram>=30", "num_gpus=1", "verified=any", "rentable=true",
                 "reliability>=0.98", "cuda_vers>=13.0", "cpu_cores_effective>=16", "cpu_ram>=60", "disk_bw>=500",
                 "inet_down>=500", "inet_up>=100", "direct_port_count>=1", "disk_space>=1400"):
        assert term in q, term
    assert "verified=true" not in q and search[search.index("--storage") + 1] == "1400"
    create = created(fake)
    env = env_of(create)
    assert env == {
        "KITSUNE_JOB": "full", "KITSUNE_BOX": "p01", "KITSUNE_SHA": SHA, "KITSUNE_CONFIG": "configs/full/data-p01.json",
        "KITSUNE_DATA_REPO": DATA, "KITSUNE_OUT_REPO": RUNS, "KITSUNE_N_GPUS": "1",
        "KITSUNE_WATCHDOG_HB_FILE": "train_hb", "KITSUNE_WATCHDOG_ORPHAN_S": "3600",
        "KITSUNE_WATCHDOG_ORPHAN_ACTION": "stop", "KITSUNE_SCRATCH_REPO": SCRATCH,
        "KITSUNE_GATE_BYTES": str(int(571.3e9)), "KITSUNE_GATE_MAX_H": "5", "KITSUNE_REBUILD_BYTES": str(int(571.3e9)),
        "KITSUNE_PULL_BYTES": str(int(1e9 * (23.4 + 2))), "KITSUNE_MAX_HOURS": "22", "TZ": "UTC",
        "KITSUNE_DATA_REVISION": "d" * 40, "KITSUNE_REBUILD_TIMEOUT_MIN": "387", "KITSUNE_DPH": "0.8100",
        "KITSUNE_MACHINE_ID": "70001"}
    assert create[create.index("--label") + 1].startswith("kitsune-full-p01-data-p01-")
    assert create[create.index("--disk") + 1] == "1400" and "HF_TOKEN" not in " ".join(create)
    assert full_launch.seen["extra_gb"] == 25  # the registry's extra_gb goes into the sizing
    assert full_launch.seen["cfg"]["family"] == "ctc"  # the data config read at the sha
    a, kw = full_launch.seen["preflight"]
    assert a[3] == SCRATCH and a[5] == "p01" and kw["resume"] is False
    assert "full preflight ok" in out and "(1x RTX 5090 + 1400 GB) x ~19.5 h" in out and "watchdog cap 22 h" in out
    assert "avoiding machine 151760: vast/blocklist.json" in out  # every job avoids the blocklist


def test_min_disk_bw_lowers_only_the_disk_floor(full_launch, capsys):
    rc, fake = full_launch([[offer(1, 139369, 0.70)]], "--box", "p01", "--scratch-repo", SCRATCH, "--min-disk-bw",
                           "250", "--dry-run")
    out = capsys.readouterr().out
    assert rc == 0, out
    q = search_query(fake).split(" ")
    assert "disk_bw>=250" in q and "disk_bw>=500" not in q
    assert {"inet_down>=500", "inet_up>=100", "cpu_cores_effective>=16", "reliability>=0.98"} <= set(q)
    assert "--min-disk-bw 250" in out  # the note says the floor was lowered
    assert "disk_bw>=500" in launch.full_filter(1)  # the default floor is kept
    two = launch.full_filter(2, min_disk_bw=300)
    assert "disk_bw>=300" in two and not any(t.startswith("disk_bw") and t != "disk_bw>=300" for t in two)
    for bad in ("100", "149"):
        with pytest.raises(SystemExit):
            launch.main(["--job", "full", "--box", "p01", "--min-disk-bw", bad, "--dry-run"])
    with pytest.raises(SystemExit):  # a full-only flag
        launch.main(["--job", "train", "--min-disk-bw", "300", "--dry-run"])


def test_cost_guards_drop_dear_traffic_and_totals_over_the_cap():
    """2026-10-01: with the cheap offers gone, the ranking's winner under the $/h cap charged $0.039/GB both ways (~$53
    expected for a ~$26 run). A full job now drops traffic over FULL_MAX_GB_COST and totals over max_total."""
    from dataclasses import replace
    job = replace(launch.full_job("p01", {"gpus": 1}, "5090", 20.0, 22.0, 1.0), est_down_gb=600.0, est_up_gb=50.0)
    assert job.max_gb_cost == launch.FULL_MAX_GB_COST == 0.01
    dear = offer(1, 11, 0.80, inet_down_cost=0.039, inet_up_cost=0.039)
    fair = offer(2, 12, 0.95)
    assert any("inet_down_cost" in p for p in launch.offer_problems(dear, job))
    assert any("inet_up_cost" in p for p in launch.offer_problems(dear, job))
    assert launch.offer_problems(fair, job) == []
    assert [o["id"] for o in launch.rank_offers([dear, fair], job, max_dph=1.0)] == [2]
    loose = replace(job, max_gb_cost=0.05)  # traffic allowed: ranked by the total, the dear one last
    assert [o["id"] for o in launch.rank_offers([dear, fair], loose, max_dph=1.0)] == [2, 1]
    capped = replace(loose, max_total=25.0)  # fair: 0.95 x 20 + 650 x 0.002 = 20.3; dear: 16 + 650 x 0.039 = 41.4
    assert [o["id"] for o in launch.rank_offers([dear, fair], capped, max_dph=1.0)] == [2]
    assert launch.rank_offers([dear, fair], replace(capped, max_total=10.0), max_dph=1.0) == []


def test_cost_guard_flags_are_full_only_and_positive(full_launch, capsys):
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p01", "--scratch-repo", SCRATCH, "--dry-run")
    out = capsys.readouterr().out
    assert rc == 0, out
    m = re.search(r"cost guards: traffic <= \$0\.01/GB each way; expected total <= \$([0-9.]+)", out)
    assert m and 24.0 < float(m.group(1)) < 40.0, out  # 1.25 x $1.00 x 19.5 h + ~620 GB x $0.01
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p01", "--scratch-repo", SCRATCH, "--max-total", "5",
                           "--dry-run")
    out = capsys.readouterr().out
    assert rc != 0 and "cost guards" in out and created(fake) is None
    for args in (["--max-total", "0"], ["--max-gb-cost", "-1"]):
        with pytest.raises(SystemExit):
            launch.main(["--job", "full", "--box", "p01", *args, "--dry-run"])
    with pytest.raises(SystemExit):
        launch.main(["--job", "train", "--max-total", "30", "--dry-run"])


def test_box_full_rents_two_gpus_under_its_cap_and_ranks_by_the_total(full_launch, capsys):
    two = [offer(1, 11, 1.60, num_gpus=2, cpu_ram=130000, inet_down_cost=0.05, inet_up_cost=0.05),  # cheap $/h, dear GB
           offer(2, 12, 1.65, num_gpus=2, cpu_ram=130000),
           offer(3, 13, 1.75, num_gpus=2, cpu_ram=130000)]  # over the registry's 1.70 cap
    rc, fake = full_launch([two], "--box", "full", "--scratch-repo", SCRATCH, "--yes")
    out = capsys.readouterr().out
    assert rc == 0, out
    q = search_query(fake).split(" ")
    assert {"num_gpus=2", "cpu_cores_effective>=32", "cpu_ram>=120", "inet_up>=200"} <= set(q)
    env = env_of(created(fake))
    assert env["KITSUNE_N_GPUS"] == "2" and env["KITSUNE_MAX_HOURS"] == "47" and env["KITSUNE_MACHINE_ID"] == "12"
    assert created(fake)[3] == "2", "the est_total winner, not the cheapest $/h"
    assert "machine" in out and "est$" in out and "verif" in out and "maxd" in out  # the table shows the filter
    # --tier a100: the A100 terms and option C's cap
    a100 = [offer(4, 21, 2.50, gpu_name="A100 SXM4", gpu_ram=40960, num_gpus=2, cpu_ram=130000)]
    rc, fake = full_launch([a100], "--box", "full", "--scratch-repo", SCRATCH, "--tier", "a100", "--dry-run")
    assert rc == 0, capsys.readouterr().out
    q = search_query(fake)
    assert "gpu_name in [A100_SXM4,A100_PCIE]" in q and "gpu_ram<=48" in q and "RTX_5090" not in q
    assert created(fake) is None


def test_the_client_filter_keeps_only_verified_long_rentals_with_the_ram(full_launch, capsys, monkeypatch):
    offers = [offer(1, 1, 0.50, verification="unverified"),  # the cheapest: never an unverified host
              offer(2, 2, 0.52, verification=None),
              offer(3, 3, 0.54, duration=0.4 * DAY),  # m52214: 0.4 d left
              offer(4, 4, 0.56, cpu_ram=63183),  # below 64,000 MB a GPU
              offer(5, 151760, 0.57),  # the blocklist
              offer(6, 6, 0.58, gpu_name="RTX 5090", duration=None, end_date=time.time() + 2 * DAY),
              offer(7, 7, 0.60, verification="deverified"),  # kept
              offer(8, 8, 0.70)]
    full_launch.seen["gates"] = {"8": "its download gate said slow 2 d ago"}
    rc, fake = full_launch([offers], "--box", "p01", "--scratch-repo", SCRATCH, "--dry-run")
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "avoiding machine 8: its download gate said slow" in out
    assert env_of(created_or_printed(out))["KITSUNE_MACHINE_ID"] == "7"
    spec = {"gpus": 1}
    j = launch.full_job("p01", spec, "5090", 19.5, 22, 1.0)
    now = time.time()
    assert launch.offer_problems(offer(1, 1, 0.5, verification="unverified"), j, now)
    assert launch.offer_problems(offer(1, 1, 0.5, duration=None), j, now) == ["max rental unknown < 4 d"]
    assert launch.offer_problems(offer(1, 1, 0.5, duration=None, end_date=now + 5 * DAY), j, now) == []
    assert launch.offer_problems(offer(1, 1, 0.5, cpu_ram=126367), launch.full_job("full", {"gpus": 2}, "5090", 41.9,
                                                                                    47, 1.7), now)
    # the other jobs filter nothing on the client (their JobSpec defaults)
    assert launch.offer_problems(offer(1, 1, 0.5, verification="unverified", duration=0), launch.JOBS["label"]) == []


def test_a_registry_min_ram_and_a_long_cap_tighten_the_filter(full_launch, repo, capsys):
    """min_ram_gb (registry; RAM analysis 2026-10-02: box 1's real peaks were 33 GiB trainer, 37 GiB store build, the
    164 GB peak_rss_gb is page cache counted per DataLoader worker): the query asks floor(0.94 x it) GB, the client
    filter round(0.97 x it x 1000) MB, so vast's 96 GB machines listed at 95,758 MB stay; None keeps 64,000 MB a GPU
    and the query's 60. A cap past MIN_RENTAL_DAYS raises the rental floor to cap / 24 + 0.5 d."""
    now = time.time()
    j = launch.full_job("full-t", {"gpus": 1, "min_ram_gb": 96}, "5090", 26.0, 37, 1.1)
    assert "cpu_ram>=90" in j.base_filter and j.ram_mb_min == 93120 and j.min_rental_days == 4
    assert launch.offer_problems(offer(1, 1, 0.6, cpu_ram=95758), j, now) == []
    assert launch.offer_problems(offer(1, 1, 0.6, cpu_ram=64439), j, now) == ["cpu_ram 64439 MB < 93120 MB"]
    j0 = launch.full_job("p01", {"gpus": 1, "min_ram_gb": None}, "5090", 19.5, 22, 1.0)
    assert "cpu_ram>=60" in j0.base_filter and j0.ram_mb_min == 0
    assert launch.offer_problems(offer(1, 1, 0.6, cpu_ram=64439), j0, now) == []
    assert launch.offer_problems(offer(1, 1, 0.6, cpu_ram=63183), j0, now) == ["cpu_ram 63183 MB < 64000 MB"]
    assert "cpu_ram>=120" in launch.full_filter(2, min_ram_gb=96)  # never below 60 a GPU
    # a hypothetical long box (104 h, as a 10-epoch box T would have been): the floor rises past 4 d
    long = launch.full_job("full-t", {"gpus": 1, "min_ram_gb": 96}, "5090", 78.9, 104, 1.1)
    assert long.min_rental_days == launch.min_rental_days(104) == 4.83
    assert launch.offer_problems(offer(1, 1, 0.6, cpu_ram=96000, duration=4.2 * DAY), long, now) == [
        "max rental 4.2 d < 4.83 d"]
    assert launch.offer_problems(offer(1, 1, 0.6, cpu_ram=96000, duration=4.9 * DAY), long, now) == []
    assert launch.min_rental_days(37) == launch.min_rental_days(22) == launch.MIN_RENTAL_DAYS == 4
    # the launch: the registry's min_ram_gb reaches the query and the look-only says so
    reg = copy.deepcopy(dict(repo.reg))
    reg["boxes"]["p01"]["min_ram_gb"] = 96
    write_reg(repo.root, reg)
    rc, fake = full_launch([[offer(1, 2, 0.50, cpu_ram=64439), offer(2, 3, 0.60, cpu_ram=95758)]], "--box", "p01",
                           "--scratch-repo", SCRATCH, "--dry-run")
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "cpu_ram>=90" in search_query(fake).split(" ") and "cpu_ram>=60" not in search_query(fake).split(" ")
    assert "RAM >= 96 GB (registry min_ram_gb): the query asks cpu_ram >= 90 GB, the client filter >= 93120 MB" in out
    assert "host max rental >= 4 d" in out
    assert env_of(created_or_printed(out))["KITSUNE_MACHINE_ID"] == "3"


def created_or_printed(out: str) -> list[str]:
    """The create command as printed (look-only runs print it with the offer they would rent)."""
    import shlex

    line = out.split("create command:\n  vastai ", 1)[1].splitlines()[0]
    return ["vastai", *shlex.split(line)]


def test_machine_picks_that_machine_or_refuses(full_launch, capsys):
    offers = [offer(1, 1, 0.50), offer(2, 70001, 0.81)]
    rc, fake = full_launch([offers], "--box", "p01", "--scratch-repo", SCRATCH, "--machine", "70001", "--yes")
    assert rc == 0, capsys.readouterr().out
    assert env_of(created(fake))["KITSUNE_MACHINE_ID"] == "70001"
    rc, fake = full_launch([offers], "--box", "p01", "--scratch-repo", SCRATCH, "--machine", "999", "--yes")
    out = capsys.readouterr().out
    assert rc == 1 and "machine 999 has no offer passing the filter now" in out and created(fake) is None
    for dead in ("151760", "54650"):  # the blocklist: refused even with an offer listed for it
        rc, fake = full_launch([offers + [offer(3, int(dead), 0.60)]], "--box", "p01", "--scratch-repo", SCRATCH,
                               "--machine", dead, "--yes")
        out = capsys.readouterr().out
        assert rc == 1 and f"--machine {dead} is avoided (vast/blocklist.json" in out and created(fake) is None
    rc, fake = full_launch([offers], "--box", "p01", "--scratch-repo", SCRATCH, "--machine", "70001", "--offer-id",
                           "1", "--yes")
    assert rc == 1 and "offer 1 is not in the results" in capsys.readouterr().out


@pytest.mark.parametrize("args,err", [
    (["--box", "p01", "--gpus", "2"], "box p01 is planned for 1 GPU(s)"),
    (["--box", "p01"], "--scratch-repo <the private scratch model repo> is required"),
    (["--box", "p01", "--scratch-repo", RUNS], "is the output or the data repo"),
    (["--box", "p01", "--scratch-repo", SCRATCH, "--disk-gb", "900"], "--disk-gb 900 is below the 1400 GB"),
    (["--box", "p01", "--scratch-repo", SCRATCH, "--config", "configs/full/data-full.json"],
     "data config is configs/full/data-p01.json"),
    (["--box", "p01", "--scratch-repo", SCRATCH, "--no-hf-check"], "--job full needs the HF preflight"),
    (["--box", "p01", "--scratch-repo", SCRATCH, "--max-dph", "0.5"], "no 1x RTX 5090 offer matches the full-box"),
], ids=["gpus", "no-scratch", "scratch-is-out", "disk", "config", "no-hf-check", "max-dph"])
def test_launch_full_refusals(full_launch, capsys, args, err):
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], *args, "--yes")
    out = capsys.readouterr().out
    assert rc == 1 and err in out and created(fake) is None, out


def test_launch_full_argument_errors(full_launch, capsys):
    for args, err in ((["--job", "full"], "--job full needs --box"),
                      (["--job", "full", "--box", "A"], "--box A is not a box of --job full"),
                      (["--job", "study", "--box", "p01"], "--box p01 is not a box of --job study"),
                      (["--job", "train", "--scratch-repo", SCRATCH], "--scratch-repo: for --job full only"),
                      (["--job", "full", "--box", "p01", "--resume-set", "full-p01-20260927T120000Z:optim.lr=1"],
                       "only schedule.epochs, early_stop.patience, augment.enabled, augment.truncate_p, "
                       "augment.concat_p, augment.mix_p, augment.truncate_min_row_s, augment.end_trim_p, "
                       "augment.noise_p, augment.noise_bank, augment.noise_bank_sha256, augment.speech_p, "
                       "augment.reverb_p, augment.rir_bank, augment.rir_bank_sha256, augment.gain_p, augment.codec_p, "
                       "augment.background_min_row_s, schedule.deadline_cooldown may change on a resume"),
                      # DECISIONS H14: only a registry continuation keeps an early-stop cooldown (continuation_preflight
                      # checks the state's record before renting; no env resume set does)
                      (["--job", "full", "--box", "p01", "--resume-set",
                        "full-p01-20260927T120000Z:schedule.resume_reset_keep_cooldown=true"],
                       "augment.background_min_row_s, schedule.deadline_cooldown may change on a resume"),
                      (["--job", "full", "--box", "p01", "--resume-set",
                        "full-p01-20260927T120000Z:augment.noise_bank=../bank"],
                       "augment.noise_bank must be a relative data-repo path"),
                      (["--job", "full", "--box", "p01", "--resume-set",
                        "full-p01-20260927T120000Z:early_stop.patience=0"], "early_stop.patience must be an int >= 1"),
                      (["--job", "full", "--box", "p01", "--resume-set",
                        "full-p01-20260927T120000Z:augment.enabled=yes"], "augment.enabled must be true or false"),
                      (["--job", "full", "--box", "p01", "--resume-set",
                        "full-p01-20260927T120000Z:augment.concat_p=1.5"],
                       "augment.concat_p must be a probability in [0, 1]"),
                      (["--job", "full", "--box", "p01", "--resume-set",
                        "full-p01-20260927T120000Z:augment.seed=7"], "may change on a resume"),
                      (["--job", "full", "--box", "p01", "--resume-set", "full-p01-20260927T120000Z:schedule.epochs=8",
                        "--resume-set", "full-p01-20260927T120000Z:schedule.epochs=9"], "given twice"),
                      (["--job", "train", "--allow-done-trains"], "--allow-done-trains: for --job full only"),
                      (["--job", "train", "--fresh-over-done"], "--fresh-over-done: for --job full only"),
                      (["--job", "full", "--box", "p01", "--resume", "--fresh-over-done"],
                       "--fresh-over-done is for a launch without --resume"),
                      (["--job", "full", "--box", "p01", "--fresh-over-done", "--resume-reset",
                        "full-p01-20261001T184145Z"], "--fresh-over-done is for a launch without --resume"),
                      (["--job", "full", "--box", "p01", "--machine", "m70001"], "machine_id (digits)")):
        with pytest.raises(SystemExit):
            launch.main([*args, "--data-repo", DATA, "--out-repo", RUNS, "--sha", SHA])
        assert err in capsys.readouterr().err, args


def test_a_box_the_registry_lacks_is_refused(full_launch, repo, capsys):
    reg = copy.deepcopy(dict(repo.reg))
    del reg["boxes"]["smoke-b"]
    write_reg(repo.root, reg)
    rc, fake = full_launch([[offer(1, 1, 0.5)]], "--box", "smoke-b", "--yes")
    out = capsys.readouterr().out
    assert rc == 1 and "box 'smoke-b' is not in the registry" in out and fake.calls == []


def test_the_registry_is_read_at_the_sha_and_a_dirty_copy_refuses(full_launch, repo, monkeypatch, capsys):
    """launch reads configs/full/boxes.json at the commit the box runs; a working-tree copy that differs (an edit not
    committed) refuses without --skip-git-checks; a registry the sha lacks refuses."""
    changed = copy.deepcopy(dict(repo.reg))
    changed["boxes"]["p01"]["max_hours"] = 30
    monkeypatch.setattr(launch, "worktree_file", lambda rel: json.dumps(changed).encode())
    reg, reader, problems, _ = launch.full_registry(SHA)
    assert reg["boxes"]["p01"]["max_hours"] == 22, "the sha's, not the working tree's"
    assert problems and "differs from the one at 0123456789ab" in problems[0]
    assert launch.full_registry(SHA, skip_git_checks=True)[2] == []
    assert reader("configs/full/data-p01.json")["family"] == "ctc"
    monkeypatch.setattr(launch, "worktree_file", lambda rel: (repo.root / rel).read_bytes().replace(b"\n", b"\r\n"))
    assert launch.full_registry(SHA)[2] == [], "line endings are not a difference"
    (repo.root / "configs/full/data-p01.json").unlink()  # an item config missing at the sha: the registry refuses
    reg, _, problems, _ = launch.full_registry(SHA)
    assert reg is None and any("data-p01.json" in p for p in problems)
    (repo.root / fullrun.BOXES_FILE).unlink()
    monkeypatch.setattr(launch, "worktree_file", lambda rel: None)
    reg, _, problems, _ = launch.full_registry(SHA)
    assert reg is None and "does not exist at 0123456789ab" in problems[0]


def test_resume_flags_go_to_the_box_env(full_launch, capsys):
    rid, rid2 = "full-p01-20260927T120000Z", "full-p01-20260928T010203Z-2"
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p01", "--scratch-repo", SCRATCH, "--resume-reset",
                           rid, "--resume-set", f"{rid}:schedule.epochs=5", "--resume-set",
                           f"{rid2}:schedule.epochs=05", "--yes",
                           instances=[{"id": 5, "label": "kitsune-full-p01-data-p01-abc", "actual_status": "running"},
                                      {"id": 6, "label": "kitsune-full-full-data-full-abc"}])
    out = capsys.readouterr().out
    assert rc == 0, out
    env = env_of(created(fake))
    assert env["KITSUNE_RESUME"] == "1" and env["KITSUNE_RESUME_RESET"] == rid
    assert env["KITSUNE_RESUME_SETS"] == f"{rid}:schedule.epochs=5,{rid2}:schedule.epochs=5"
    a, kw = full_launch.seen["preflight"]
    assert kw["resume"] is True and kw["resets"] == [rid] and set(kw["sets"]) == {rid, rid2}
    assert "WARNING: a live instance of box p01: kitsune-full-p01-data-p01-abc (instance 5, running)" in out
    assert "kitsune-full-full" not in out
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p01", "--scratch-repo", SCRATCH, "--dry-run")
    assert rc == 0 and "KITSUNE_RESUME" not in " ".join(created_or_printed(capsys.readouterr().out))
    assert not any(c[1:3] == ["show", "instances"] for c in fake.calls)


def test_the_continuation_flags_go_to_the_box_env(full_launch, capsys):
    """DECISIONS G3: P-0.1B's continuation from its pre_cooldown state to 8 epochs with patience 12 - the reset and
    both sets of one run reach the box as one KITSUNE_RESUME_RESET and one KITSUNE_RESUME_SETS word, in the order
    given (fullrun.parse_resume_sets on the box gives the queue the same list)."""
    rid = "full-p01-20261001T184145Z"
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p01", "--scratch-repo", SCRATCH, "--resume-reset",
                           rid, "--resume-set", f"{rid}:schedule.epochs=8", "--resume-set",
                           f"{rid}:early_stop.patience=012", "--yes")
    out = capsys.readouterr().out
    assert rc == 0, out
    env = env_of(created(fake))
    assert env["KITSUNE_RESUME"] == "1" and env["KITSUNE_RESUME_RESET"] == rid
    assert env["KITSUNE_RESUME_SETS"] == f"{rid}:schedule.epochs=8,{rid}:early_stop.patience=12"
    assert re.fullmatch(fullrun._ENV_WORD, env["KITSUNE_RESUME_SETS"])
    assert fullrun.parse_resume_sets(env["KITSUNE_RESUME_SETS"]) == {
        rid: ["schedule.epochs=8", "early_stop.patience=12"]}
    a, kw = full_launch.seen["preflight"]
    assert kw["resets"] == [rid] and kw["sets"] == {rid: ["schedule.epochs=8", "early_stop.patience=12"]}
    assert kw["allow_done_trains"] is False and kw["allow_fresh_over_done"] is False


def test_the_recipe_test_flags_reach_the_box_env_in_one_spelling(full_launch, capsys):
    """DECISIONS H1: the recipe test box re-runs box 1's cooldown with the augmentation on - launch's reset and five
    sets of one run, typed in loose spellings, reach the box as one KITSUNE_RESUME_SETS word in the normalised spelling
    (fullrun.resume_set_value: the trainer's --set parses JSON, so "True" must arrive as true), which parses back to
    the same list; the preflight sees the normalised sets too."""
    rid = "full-p01-20261001T184145Z"
    loose = ["schedule.epochs=04", "augment.enabled=True", "augment.truncate_p=.3", "augment.concat_p=0.50",
             "augment.mix_p=.05"]
    want = ["schedule.epochs=4", "augment.enabled=true", "augment.truncate_p=0.3", "augment.concat_p=0.5",
            "augment.mix_p=0.05"]  # weak mix: DECISIONS H4
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p01", "--scratch-repo", SCRATCH, "--resume-reset",
                           rid, *sum((["--resume-set", f"{rid}:{s}"] for s in loose), []), "--yes")
    out = capsys.readouterr().out
    assert rc == 0, out
    env = env_of(created(fake))
    assert env["KITSUNE_RESUME"] == "1" and env["KITSUNE_RESUME_RESET"] == rid
    assert env["KITSUNE_RESUME_SETS"] == ",".join(f"{rid}:{s}" for s in want)
    assert re.fullmatch(fullrun._ENV_WORD, env["KITSUNE_RESUME_SETS"])
    assert fullrun.parse_resume_sets(env["KITSUNE_RESUME_SETS"]) == {rid: want}
    a, kw = full_launch.seen["preflight"]
    assert kw["resets"] == [rid] and kw["sets"] == {rid: want}


def test_the_p_test_box_flags_reach_the_box_env_in_one_spelling(full_launch, capsys):
    """DECISIONS H8: the P test box re-runs box 1's cooldown with recipe v3 - its seconds, path and sha256 sets, in
    loose spellings, reach the box in the normalised spelling (3 -> 3.0, the sha256 lowercased, the path as given)
    and parse back to the same list."""
    rid, sha = "full-p01-20261001T184145Z", "89abcdef" * 8
    loose = ["schedule.epochs=4", "augment.enabled=TRUE", "augment.truncate_p=.2", "augment.concat_p=0.5",
             "augment.mix_p=0", "augment.truncate_min_row_s=3", "augment.end_trim_p=0.30", "augment.noise_p=.3",
             "augment.noise_bank=aug/musan-bg-v1", f"augment.noise_bank_sha256={sha.upper()}"]
    want = ["schedule.epochs=4", "augment.enabled=true", "augment.truncate_p=0.2", "augment.concat_p=0.5",
            "augment.mix_p=0.0", "augment.truncate_min_row_s=3.0", "augment.end_trim_p=0.3", "augment.noise_p=0.3",
            "augment.noise_bank=aug/musan-bg-v1", f"augment.noise_bank_sha256={sha}"]
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p01", "--scratch-repo", SCRATCH, "--resume-reset",
                           rid, *sum((["--resume-set", f"{rid}:{s}"] for s in loose), []), "--yes")
    out = capsys.readouterr().out
    assert rc == 0, out
    env = env_of(created(fake))
    assert env["KITSUNE_RESUME_SETS"] == ",".join(f"{rid}:{s}" for s in want)
    assert re.fullmatch(fullrun._ENV_WORD, env["KITSUNE_RESUME_SETS"])
    assert fullrun.parse_resume_sets(env["KITSUNE_RESUME_SETS"]) == {rid: want}


def test_augment_sets_are_refused_for_a_box_without_a_ctc_trainer(full_launch, repo, capsys):
    """augment.* resume sets are the CTC family's: on a box whose train items are all AED, launch refuses them before
    renting (an AED student's augmentation is its config's, with the cut table no set can name, and 04_distill's
    validate_augment would refuse a cut without it only on the box, after the paid boot and the store build, on every
    attempt); a CTC box's sets pass (box p01 above), and so do an AED box's int sets."""
    reg = copy.deepcopy(dict(repo.reg))
    reg["boxes"]["full-t"] = dict(copy.deepcopy(reg["boxes"]["p01"]), data_config="configs/full/data-full.json", items=[
        {"name": "stores-aed", "kind": "stores", "config": "configs/full/full-t06.json"},
        {"name": "full-t06", "kind": "train", "config": "configs/full/full-t06.json", "study_run": "study-t06",
         "family": "aed", "max_hours": 22.81, "needs": ["stores-aed"]}])
    write_reg(repo.root, reg)
    rid = "full-t06-20261003T000000Z"
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "full-t", "--scratch-repo", SCRATCH, "--resume-reset",
                           rid, "--resume-set", f"{rid}:schedule.epochs=4", "--resume-set",
                           f"{rid}:augment.enabled=true", "--resume-set", f"{rid}:augment.mix_p=0.2", "--yes")
    out = capsys.readouterr().out
    assert rc == 1 and created(fake) is None, out
    assert ("--resume-set augment.enabled, augment.mix_p: box full-t trains no CTC student (its train items' "
            "families: aed), and augment.* resume sets are the CTC family's") in out
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "full-t", "--scratch-repo", SCRATCH, "--resume-reset",
                           rid, "--resume-set", f"{rid}:schedule.epochs=4", "--dry-run")
    out = capsys.readouterr().out
    assert rc == 0 and "trains no CTC student" not in out, out


def test_the_fresh_over_done_flag_goes_to_the_preflight(full_launch, capsys):
    """--fresh-over-done reaches full_preflight (a fresh launch, resume False); a refusal of the preflight (a done
    train item on the Hub) stops the launch, look-only or not."""
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "full-smoke", "--scratch-repo", SCRATCH,
                           "--fresh-over-done", "--dry-run")
    assert rc == 0, capsys.readouterr().out
    a, kw = full_launch.seen["preflight"]
    assert kw["resume"] is False and kw["allow_fresh_over_done"] is True
    full_launch.seen["preflight_problems"] = ["a fresh launch of box p01 (no --resume): its Hub summary ..."]
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p01", "--scratch-repo", SCRATCH, "--yes")
    out = capsys.readouterr().out
    assert rc == 1 and "a fresh launch of box p01 (no --resume)" in out and created(fake) is None, out
    assert full_launch.seen["preflight"][1]["allow_fresh_over_done"] is False


def test_live_instances_are_the_boxs_own(monkeypatch):
    """Box full's labels start kitsune-full-full-data-full-: the smoke box's (kitsune-full-full-smoke-...) is not one."""
    fake = FakeVastai([], instances=[{"id": 1, "label": "kitsune-full-full-data-full-0123456", "actual_status": "running"},
                                     {"id": 2, "label": "kitsune-full-full-smoke-data-smoke-0123456"}, {"id": 3}])
    monkeypatch.setattr(launch.subprocess, "run", fake)
    assert launch.live_instances("/fake/vastai", "kitsune-full-full-data-full-") == [
        "kitsune-full-full-data-full-0123456 (instance 1, running)"]


def test_the_gate_follows_the_registry_and_the_flag(full_launch, repo, capsys):
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p01", "--scratch-repo", SCRATCH, "--gate-hours",
                           "8", "--dry-run")
    env = env_of(created_or_printed(capsys.readouterr().out))
    assert rc == 0 and env["KITSUNE_GATE_MAX_H"] == "8"
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p01", "--scratch-repo", SCRATCH, "--gate-hours",
                           "0", "--dry-run")
    out = capsys.readouterr().out
    assert rc == 0 and "the download gate is OFF" in out
    assert not any(k.startswith("KITSUNE_GATE") for k in env_of(created_or_printed(out)))
    # smoke-b: gate false, no timed states: neither the gate nor the scratch repo
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "smoke-b", "--dry-run")
    out = capsys.readouterr().out
    env = env_of(created_or_printed(out))
    assert rc == 0, out
    assert "KITSUNE_GATE_BYTES" not in env and "KITSUNE_SCRATCH_REPO" not in env and "no download gate" in out
    assert env["KITSUNE_WATCHDOG_ORPHAN_ACTION"] == "stop" and env["KITSUNE_CONFIG"] == "configs/full/data-smoke-b.json"


def test_the_smoke_box_alerts_and_its_gate_is_judged_on_the_reference(full_launch, capsys):
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "full-smoke", "--scratch-repo", SCRATCH, "--dry-run")
    env = env_of(created_or_printed(capsys.readouterr().out))
    assert rc == 0
    assert env["KITSUNE_WATCHDOG_ORPHAN_ACTION"] == "alert" and env["KITSUNE_WATCHDOG_ORPHAN_S"] == "600"
    assert env["KITSUNE_MAX_HOURS"] == "9" and env["KITSUNE_GATE_BYTES"] == str(int(571.3e9))


# ============================================================================================== full_preflight


def registered_meta(run: str) -> dict:
    spec = prereg.RUNS[run]
    meta = {k: spec[k] for k in ("family", "init_class", "seed", "params_total", "params_non_embedding")}
    meta.update(stage="complete", closed_form_params=spec["params_total"])
    if spec["init_class"] != "scratch":
        key = "importance_ids_sha256" if spec["family"] == "aed" else "ids_sha256"
        meta["calibration"] = {key: spec["calib_ids_sha256"]}
    return meta


class FullHub:
    """The data, runs and scratch repos of full_preflight: file lists, LFS sha256s, contents, privacy."""

    def __init__(self, data: dict, runs: dict | None = None, scratch: dict | None = None, private=True):
        self.data, self.runs, self.scratch, self.private = data, runs or {}, scratch or {}, private

    def list_repo_files(self, repo, repo_type=None, revision=None):
        return sorted(self.data if repo == DATA else self.runs)

    def get_paths_info(self, repo, paths, repo_type=None, revision=None):
        return [SimpleNamespace(path=p, lfs=SimpleNamespace(sha256="5e" * 32) if p.endswith(".parquet") else None)
                for p in paths if p in self.data]

    def file_exists(self, repo, path, repo_type=None):
        return path in self.runs

    def model_info(self, repo):
        if repo not in (SCRATCH, RUNS):
            raise RuntimeError("404 Repository Not Found")
        return SimpleNamespace(private=self.private)

    def list_repo_tree(self, repo, path_in_repo=None, recursive=False, repo_type=None):
        return [SimpleNamespace(path=p) for p in {"/".join(f.split("/")[:path_in_repo.count("/") + 2])
                                                   for f in self.runs if f.startswith(path_in_repo + "/")}]

    def download(self, repo, path, repo_type=None, revision=None, local_dir=None):
        src = {DATA: self.data, RUNS: self.runs, SCRATCH: self.scratch}[repo]
        if path not in src or src[path] is None:
            raise FileNotFoundError(f"404: {repo}/{path}")
        p = Path(local_dir) / path
        p.parent.mkdir(parents=True, exist_ok=True)
        v = src[path]
        p.write_bytes(v if isinstance(v, bytes) else json.dumps(v).encode())
        return str(p)


def box_data(box: str, reg, root: Path) -> dict:
    """Every file the box's full_preflight asks the data repo for, present and right (the configs under root)."""
    spec = fullrun.box_spec(box, reg)
    files = {}
    ctc = set(fullrun.box_ctc_students(box, reg, root))
    runs = {it["config"]: it["study_run"] for it in spec["items"] if it["kind"] == "train"}
    for s in fullrun.box_students(box, reg, root):
        run = next(r for r in runs.values() if prereg.RUNS[r]["student"] == s)
        for n in launch.STUDENT_FILES + ((launch.CTC_CARD,) if s in ctc else ()):
            files[f"{s}/{n}"] = registered_meta(run) if n == "student_meta.json" else b"x"
    for f in spec["extra_files"]:
        files[f] = b"{}"
    for d in spec["extra_dirs"]:
        files[f"{d}/config.json"] = b"{}"
    sel = json.loads((root / spec["data_config"]).read_text(encoding="utf-8"))["selection"]
    files[sel] = b"parquet"
    files[fullrun.FROZEN_MANIFEST] = b'{"manifest": 1}'
    files[sel[:-len(".parquet")] + ".json"] = {"kind": "full_study"}
    return files


@pytest.fixture
def devslice(monkeypatch):
    """kitsune.devslice (WP1) is not on this base: a stub records sidecar_problems' arguments and answers from
    .answer."""
    stub = types.ModuleType("kitsune.devslice")
    stub.calls, stub.answer = [], []

    def sidecar_problems(sidecar, *, selection_sha256=None, manifest_sha256=None):
        stub.calls.append((sidecar, selection_sha256, manifest_sha256))
        return list(stub.answer)

    stub.sidecar_problems = sidecar_problems
    monkeypatch.setitem(sys.modules, "kitsune.devslice", stub)
    monkeypatch.setattr(kitsune, "devslice", stub, raising=False)
    return stub


def preflight(monkeypatch, hub: FullHub, box="p01", reg=None, cfg=None, **kw):
    monkeypatch.setattr(launch, "_hub", lambda: (hub, hub.download))
    reg = reg if reg is not None else launch.full_registry(SHA)[0]
    cfg = cfg if cfg is not None else launch.config_at(SHA, fullrun.box_spec(box, reg)["data_config"])
    return launch.full_preflight(DATA, "d" * 40, RUNS, kw.pop("scratch", SCRATCH), SHA, box, reg, cfg, **kw)


def test_full_preflight_passes_a_ready_box(repo, monkeypatch, devslice):
    reg = launch.full_registry(SHA)[0]
    problems, notes = preflight(monkeypatch, FullHub(box_data("p01", reg, repo.root)))
    assert problems == [], problems
    assert any("registered builds" in n for n in notes) and any("full.json: the sidecar" in n for n in notes)
    ((side, sel_sha, man_sha),) = devslice.calls
    import hashlib
    assert side == {"kind": "full_study"} and sel_sha == "5e" * 32
    assert man_sha == hashlib.sha256(b'{"manifest": 1}').hexdigest(), "a small file in git: hashed"
    for box in ("full", "full-smoke"):  # the other training boxes of the registry pass too (box full: verified quant)
        problems, _ = preflight(monkeypatch, FullHub(box_data(box, reg, repo.root), runs=go_runs()), box=box)
        assert problems == [], (box, problems)


def test_full_preflight_refusals(repo, monkeypatch, devslice):
    reg = launch.full_registry(SHA)[0]
    good = box_data("p01", reg, repo.root)

    def check(hub, want, box="p01", **kw):
        problems, _ = preflight(monkeypatch, hub, box=box, **kw)
        assert any(want in p for p in problems), (want, problems)

    check(FullHub({k: v for k, v in good.items() if k != "students/study/p01/MODEL_CARD.md"}),
          "students/study/p01 lacks ['MODEL_CARD.md']")
    stale = dict(good, **{"students/study/p01/student_meta.json": dict(registered_meta("study-p01"), seed=1)})
    check(FullHub(stale), "students/study/p01: ")
    check(FullHub({k: v for k, v in good.items() if k != fullrun.FROZEN_MANIFEST}), f"no {fullrun.FROZEN_MANIFEST}")
    check(FullHub(good, private=False), f"{SCRATCH} is not private")
    check(FullHub(good), "cannot read the scratch repo nobody/scratch", scratch="nobody/scratch")
    devslice.answer = ["the selection sha256 is not the sidecar's"]
    check(FullHub(good), "full.json: the selection sha256 is not the sidecar's")
    devslice.answer = []
    (repo.root / "configs/full/full-p01.json").rename(repo.root / "moved.json")  # not committed at the sha
    check(FullHub(good), "configs/full/full-p01.json does not exist at 0123456789ab", reg=reg)
    (repo.root / "moved.json").rename(repo.root / "configs/full/full-p01.json")


def test_full_preflight_holds_the_background_bank_and_the_cut_table_to_their_pins(repo, monkeypatch, devslice):
    """DECISIONS H11: a run's background bank (its config's augment block, or a continuation's --resume-set) must be
    one of the box's extra dirs and its index.json the sha256 the run pins, and a cut table its pinned sha256 - checked
    before renting, since a stale pin fails every train attempt on the box after the paid rebuild."""
    import hashlib

    bank, index = "aug/bank-v1", b'{"version": 1, "clips": []}'
    pin = hashlib.sha256(index).hexdigest()
    f = repo.root / "configs/full/full-p01.json"
    plain = f.read_text(encoding="utf-8")
    reg = launch.full_registry(SHA)[0]
    reg["boxes"]["p01"]["extra_dirs"] = [bank]
    good = dict(box_data("p01", reg, repo.root), **{f"{bank}/index.json": index})

    def run(augment=None, sets=None, extra_dirs=(bank,), data=good):
        cfg = json.loads(plain)
        if augment is not None:
            cfg["augment"] = augment
        f.write_text(json.dumps(cfg), encoding="utf-8")
        reg["boxes"]["p01"]["extra_dirs"] = list(extra_dirs)
        return preflight(monkeypatch, FullHub(data), reg=reg, sets=sets)

    on = {"enabled": True, "noise_p": 0.3, "noise_bank": bank, "noise_bank_sha256": pin}
    problems, notes = run(on)
    assert problems == [] and any(f"bank {bank}, index.json sha256 {pin[:12]}" in n for n in notes), (
        problems, notes)
    problems, _ = run(dict(on, noise_bank_sha256="0" * 64))
    assert any(f"full-p01 pins {bank}/index.json at sha256 000000000000..., {DATA} holds {pin[:12]}" in x
               for x in problems), problems
    problems, _ = run(on, extra_dirs=())
    assert any(f"full-p01 reads the bank {bank}, which box p01 does not pull" in x for x in problems)
    problems, _ = run(on, data={k: v for k, v in good.items() if k != f"{bank}/index.json"})
    assert any(f"no {bank}/index.json (full-p01's bank" in x for x in problems), problems
    assert run(dict(on, enabled=False))[0] == [] and run(dict(on, noise_p=0.0))[0] == []  # no background: no pin
    # a continuation's sets: its bank and pin are checked as a config's
    rid = "full-p01-20261001T184145Z"
    problems, _ = run(sets={rid: ["augment.noise_bank=" + bank, "augment.noise_bank_sha256=" + "1" * 64]})
    assert any(f"--resume-set {rid} pins {bank}/index.json at sha256 111111111111" in x for x in problems), problems
    assert run(sets={rid: ["augment.noise_bank=" + bank, "augment.noise_bank_sha256=" + pin]})[0] == []
    # an RIR bank (DECISIONS H12): its pin and the box's pull checked as a background bank's, from a config and from
    # a continuation's sets
    room = {"enabled": True, "reverb_p": 0.2, "rir_bank": bank, "rir_bank_sha256": pin}
    assert run(room)[0] == [] and run(dict(room, reverb_p=0.0, rir_bank_sha256="2" * 64))[0] == []
    problems, _ = run(dict(room, rir_bank_sha256="2" * 64))
    assert any(f"full-p01 pins {bank}/index.json at sha256 222222222222" in x for x in problems), problems
    problems, _ = run(sets={rid: ["augment.rir_bank=" + bank, "augment.rir_bank_sha256=" + "3" * 64]})
    assert any(f"--resume-set {rid} pins {bank}/index.json at sha256 333333333333" in x for x in problems), problems
    # the cut table: its LFS sha256 (FullHub: "5e" x 32 for a .parquet)
    cuts = fullrun.FULL_DIR + "/aed_cuts.parquet"
    data = dict(good, **{cuts: b"parquet"})
    assert run({"enabled": True, "cuts": cuts, "cuts_sha256": "5e" * 32}, data=data)[0] == []
    problems, _ = run({"enabled": True, "cuts": cuts, "cuts_sha256": "ab" * 32}, data=data)
    assert any(f"full-p01 pins {cuts} at sha256 abababababab..., {DATA} holds 5e5e5e5e5e5e" in x for x in problems)
    f.write_text(plain, encoding="utf-8")


def test_full_preflight_refuses_a_box_whose_tools_the_sha_lacks(repo, monkeypatch, devslice):
    """The eval and speed items name the CLI they run; a box that needs a tool of a package not merged yet (WP5's
    kitsune.quant, WP6's whisper kind of speed_probe) is refused before renting."""
    reg = launch.full_registry(SHA)[0]
    data = box_data("full", reg, repo.root)
    assert preflight(monkeypatch, FullHub(data, runs=go_runs()), box="full")[0] == []
    (repo.root / "kitsune/quant.py").unlink()
    problems, _ = preflight(monkeypatch, FullHub(data, runs=go_runs()), box="full")
    assert any("kitsune/quant.py (item quant-int8-w8a8-full-p03) does not exist" in p for p in problems), problems
    (repo.root / "kitsune/full_queue.py").unlink()
    problems, _ = preflight(monkeypatch, FullHub(data), box="p01", reg=reg)
    assert any("kitsune/full_queue.py (the box's queue) does not exist" in p for p in problems), problems
    (repo.root / "kitsune/full_queue.py").write_text("", encoding="utf-8")
    (repo.root / "tools/speed_probe.py").write_text('KINDS = ("aed", "ctc")\n', encoding="utf-8")
    problems, _ = preflight(monkeypatch, FullHub(box_data("full-smoke", reg, repo.root)), box="full-smoke")
    assert any("has no --kind cohere" in p for p in problems) and any("--kind parakeet-tdt" in p for p in problems)
    assert launch.argv_target(["{python}", "-m", "kitsune.quant", "readout"]) == "kitsune/quant.py"
    assert launch.argv_target(["{python}", "tools/whisper_eval.py", "--model", "x"]) == "tools/whisper_eval.py"
    assert launch.argv_target(["{python}", "echo"]) is None


def test_full_preflight_refuses_speed_args_the_sha_lacks(repo, monkeypatch, devslice):
    """A speed item's args add flags to the queue's speed_probe argv (WP5's --quant/--profile-kernels/--compile, WP6's
    --hf-cache); a sha whose speed_probe has the kind but not a flag is refused on the laptop, not by argparse's exit 2
    on the rented box. Values and placeholders are not flags; --flag=value counts as --flag."""
    reg = copy.deepcopy(dict(repo.reg))
    speed = {it["name"]: it for it in reg["boxes"]["full"]["items"] if it["kind"] == "speed"}
    speed["speed-full-t06"]["args"] = ["--quant", "int8-w8a8", "--profile-kernels", "--threads=8"]
    speed["speed-study-t06"]["args"] = ["--hf-cache", "{hf_cache}", "--compile"]
    write_reg(repo.root, reg)
    reg = launch.full_registry(SHA)[0]
    data = box_data("full", reg, repo.root)
    probe = repo.root / "tools/speed_probe.py"
    base = probe.read_text(encoding="utf-8")  # the kinds only: no flag of the args
    problems, _ = preflight(monkeypatch, FullHub(data, runs=go_runs()), box="full", reg=reg)
    want = {"--quant": "speed-full-t06", "--profile-kernels": "speed-full-t06", "--threads": "speed-full-t06",
            "--hf-cache": "speed-study-t06", "--compile": "speed-study-t06"}
    for flag, item in want.items():
        line = f"tools/speed_probe.py at 0123456789ab has no {flag} (in the args of speed item {item} of box full)"
        assert line in problems, (flag, problems)
    assert not any("int8-w8a8" in p or "{hf_cache}" in p or "--kind" in p for p in problems), problems
    # a sha with every flag but --compile (WP5's, say, before its compile commit): only that one is refused
    flags = [f for f in want if f != "--compile"]
    probe.write_text(base + "".join(f'ap.add_argument("{f}")\n' for f in flags), encoding="utf-8")
    problems, _ = preflight(monkeypatch, FullHub(data, runs=go_runs()), box="full", reg=reg)
    assert [p for p in problems if "speed_probe" in p] == [
        "tools/speed_probe.py at 0123456789ab has no --compile (in the args of speed item speed-study-t06 of box full)"]
    probe.write_text(base + "".join(f'ap.add_argument("{f}")\n' for f in want), encoding="utf-8")
    assert preflight(monkeypatch, FullHub(data, runs=go_runs()), box="full", reg=reg)[0] == []


def test_full_preflight_without_devslice_refuses_the_sidecar(repo, monkeypatch):
    monkeypatch.setitem(sys.modules, "kitsune.devslice", None)  # an import of it raises ImportError
    # as when WP1's tests imported the real module earlier in the session: the package attribute is set, which a
    # `from kitsune import devslice` would return despite the sys.modules entry
    monkeypatch.setattr(kitsune, "devslice", types.ModuleType("kitsune.devslice"), raising=False)
    reg = launch.full_registry(SHA)[0]
    problems, _ = preflight(monkeypatch, FullHub(box_data("p01", reg, repo.root)))
    assert any("kitsune.devslice is not in this checkout" in p for p in problems), problems


def test_resume_preflight_needs_the_hub_summary_and_known_run_ids(repo, monkeypatch, devslice):
    rid = "full-p01-20260927T120000Z"
    reg = launch.full_registry(SHA)[0]
    data = box_data("p01", reg, repo.root)
    problems, _ = preflight(monkeypatch, FullHub(data), resume=True)
    assert any("has no full/box-p01/queue_summary.json" in p for p in problems), problems
    summary = {"items": {"full-p01": {"kind": "train", "status": "running", "run_dir": f"runs/{rid}"},
                         "m4-full-p01": {"kind": "readout", "status": "pending"}}}
    runs = {fullrun.box_summary_path("p01"): summary, f"runs/{rid}/checkpoints/full_step_4000/trainer.pt": b"x",
            f"runs/{rid}/checkpoints/step_4000/model.safetensors": b"x"}
    scratch = {fullrun.scratch_pointer(rid): {"step": 5200}}
    problems, notes = preflight(monkeypatch, FullHub(data, runs, scratch), resume=True, resets=[rid], sets={})
    assert problems == [], problems
    (note,) = [n for n in notes if n.startswith("resume:")]
    assert f"resume: full-p01: running, run {rid}, reset" in note
    assert "scratch step 5200" in note and "runs repo full_step_4000" in note
    problems, _ = preflight(monkeypatch, FullHub(data, runs, scratch), resume=True, resets=[],
                            sets={"full-p01-20260101T000000Z": ["schedule.epochs=4"]})
    assert any("full-p01-20260101T000000Z: no train item of box p01 ran it" in p for p in problems), problems


def test_resume_preflight_guards_a_done_box(repo, monkeypatch, devslice):
    """DECISIONS G3's two launch guards. (a) A plain --resume of a box whose train items are all done is refused (box
    p01 would adopt full-p01 as done and score its new items on the 4-epoch weights) unless --allow-done-trains (a box
    lost in its eval pool); a reset of the run, or a continuation lost on its way (the summary shows the run running
    with its continuation record), passes. (b) A set-only id of a done run is refused (resume-pull refuses a set-only
    run past its cooldown)."""
    rid = "full-p01-20261001T184145Z"
    reg = launch.full_registry(SHA)[0]
    data = box_data("p01", reg, repo.root)

    def summary(status, **kw):
        return {fullrun.box_summary_path("p01"): {"items": {
            "full-p01": dict({"kind": "train", "status": status, "run_dir": f"runs/{rid}"}, **kw),
            "m4-full-p01": {"kind": "readout", "status": "done", "verified": True}}}}

    done = summary("done", result={"steps": 107910})
    problems, _ = preflight(monkeypatch, FullHub(data, done), resume=True, resets=[], sets={})
    assert any("a plain --resume of box p01: every train item of its Hub summary is done (full-p01)" in p
               and "--resume-reset <run_id>" in p for p in problems), problems
    problems, notes = preflight(monkeypatch, FullHub(data, done), resume=True, resets=[], sets={},
                                allow_done_trains=True)
    assert problems == [], problems
    assert any("--allow-done-trains: not refused" in n for n in notes), notes
    sets = {rid: ["schedule.epochs=8", "early_stop.patience=12"]}
    problems, notes = preflight(monkeypatch, FullHub(data, done), resume=True, resets=[rid], sets=sets)
    assert problems == [], problems
    assert any(f"resume: full-p01: done, run {rid}, reset sets ['schedule.epochs=8', 'early_stop.patience=12']" in n
               for n in notes), notes
    problems, _ = preflight(monkeypatch, FullHub(data, done), resume=True, resets=[], sets=sets)
    assert any(f"--resume-set {rid} without --resume-reset: full-p01 is done" in p for p in problems), problems
    lost = summary("running", continuation={"run_id": rid, "reset": True, "sets": sets[rid],
                                            "resume_resets_before": 0})
    problems, _ = preflight(monkeypatch, FullHub(data, lost), resume=True, resets=[], sets={})
    assert problems == [], "a continuation lost on its way resumes with a plain --resume"


def test_a_fresh_launch_over_a_done_box_is_refused(repo, monkeypatch, devslice):
    """Box p01 after box 1: its Hub summary has full-p01 done. A launch without --resume would train a new 4-epoch
    P-0.1B in a new run dir and overwrite that summary, after which the continuation (--resume-reset of box 1's run)
    is refused ("no train item of box p01 ran it"). So a fresh launch over a done train item is refused unless
    --fresh-over-done; no summary, or one whose train items are not done (a box that died in its first training: its
    owner relaunches it fresh or resumes it), passes; the continuation itself passes."""
    rid = "full-p01-20261001T184145Z"
    reg = launch.full_registry(SHA)[0]
    data = box_data("p01", reg, repo.root)

    def summary(status):
        return {fullrun.box_summary_path("p01"): {"items": {
            "full-p01": {"kind": "train", "status": status, "run_dir": f"runs/{rid}", "result": {"steps": 107910}},
            "m4-full-p01": {"kind": "readout", "status": "done", "verified": True}}}}

    problems, _ = preflight(monkeypatch, FullHub(data, summary("done")))
    (p,) = [p for p in problems if "a fresh launch" in p]
    assert p.startswith(f"a fresh launch of box p01 (no --resume): its Hub summary full/box-p01/queue_summary.json has "
                        f"train items done: full-p01 (run {rid})") and "--fresh-over-done" in p, p
    problems, notes = preflight(monkeypatch, FullHub(data, summary("done")), allow_fresh_over_done=True)
    assert problems == [], problems
    assert any("--fresh-over-done: not refused" in n for n in notes), notes
    for runs in ({}, summary("running"), summary("failed")):
        problems, notes = preflight(monkeypatch, FullHub(data, runs))
        assert problems == [] and not any("fresh launch" in n for n in notes), (runs, problems, notes)
    sets = {rid: ["schedule.epochs=8", "early_stop.patience=12"]}
    problems, _ = preflight(monkeypatch, FullHub(data, summary("done")), resume=True, resets=[rid], sets=sets)
    assert problems == [], "the continuation is a resume: the fresh guard does not apply"

    class Down(FullHub):
        def file_exists(self, repo, path, repo_type=None):
            raise ConnectionError("Hub down")

    problems, _ = preflight(monkeypatch, Down(data))
    assert any("cannot read full/box-p01/queue_summary.json" in p and "a fresh launch would overwrite it" in p
               for p in problems), problems



# ================================================================================ continuations (DECISIONS H14)


RID03, RID005 = "full-p03-20261003T230143Z", "full-p005-20261008T160232Z"
# item -> (run id, source box, from_step, to_step): box p-cool in miniature continues box full's full-p03 (a planned
# 0.8 T cooldown) and box p005's full-p005 (an early-stop cooldown it keeps)
COOL = {"full-p03": (RID03, "full", 169376, 211720), "full-p005": (RID005, "p005", 67449, 80939)}
GENTLE = {"schedule.epochs": 10, "schedule.deadline_cooldown": False, "schedule.resume_reset_keep_cooldown": True,
          "augment.noise_snr_db": [5.0, 20.0], "augment.codecs": ["mp3", "ulaw8k"],
          "augment.background_min_row_s": 3.0, "augment.noise_bank": "aug/bank-v1",
          "augment.noise_bank_sha256": "AB" * 32}
COOL_SETS = {"full-p03": {"schedule.epochs": 6, "schedule.deadline_cooldown": False}, "full-p005": GENTLE}


def cool_registry(root: Path, reg: dict) -> dict:
    """The tiny registry plus box p005 (P-0.05B alone) and box p-cool, whose train items continue box full's
    full-p03 and box p005's full-p005 (both on data-full, one CTC store), written to root's boxes.json."""
    reg = copy.deepcopy(dict(reg))
    base = copy.deepcopy(reg["boxes"]["p01"])

    def box(items):
        return dict(base, data_config="configs/full/data-full.json", items=items)

    def train(name, run, h, cont=None):
        return {"name": name, "kind": "train", "config": f"configs/full/{name}.json", "study_run": run,
                "family": "ctc", "max_hours": h, "needs": ["stores-ctc"], **({"continues": cont} if cont else {})}

    def readout(name):
        return {"name": f"m4-{name}", "kind": "readout", "of": name, "max_hours": 0.3}

    def cont(name):
        rid, src, fs, ts = COOL[name]
        return {"box": src, "run_id": rid, "from_step": fs, "to_step": ts, "resets_before": 0,
                "sets": copy.deepcopy(COOL_SETS[name])}

    reg["boxes"]["p005"] = box([{"name": "stores-ctc", "kind": "stores", "config": "configs/full/full-p005.json"},
                                train("full-p005", "study-p005", 9.92), readout("full-p005")])
    reg["boxes"]["p-cool"] = box([{"name": "stores-ctc", "kind": "stores", "config": "configs/full/full-p03.json"},
                                  train("full-p03", "study-p03", 6.0, cont("full-p03")), readout("full-p03"),
                                  train("full-p005", "study-p005", 1.0, cont("full-p005")), readout("full-p005")])
    write_reg(root, reg)
    return reg


@pytest.fixture
def cool(repo, monkeypatch):
    """The p-cool registry at SHA, and resume_preflight(box, runs, ...) of it against a FullHub with those runs."""
    cool_registry(repo.root, repo.reg)
    reg = launch.full_registry(SHA)[0]

    def go(runs, box="p-cool", resets=(), sets=None, scratch=None, **kw):
        hub = FullHub({}, runs, scratch)
        monkeypatch.setattr(launch, "_hub", lambda: (hub, hub.download))
        return launch.resume_preflight(RUNS, SCRATCH, box, list(resets), sets or {}, reg=reg, **kw)
    go.reg = reg
    return go


def cool_brief(name: str, **st) -> dict:
    """The trainer.json of the item's pre_cooldown state as the source box left it: full-p03 at its planned 0.8 T
    (no early-stop record, total_steps its end), full-p005 at its early-stop cooldown's start (the record's T its end;
    total_steps the 10-epoch plan)."""
    rid, src, fs, ts = COOL[name]
    es = ({"cooldown": {"t_c": float(fs), "T": float(ts), "clock": "epochs", "at_step": fs},
           "triggered": {"reason": "patience", "step": fs}} if name == "full-p005" else {"cooldown": None})
    return {"format": 1, "step": fs, "reason": "pre_cooldown", "run_id": rid, "cfg": {"schedule": {"clock": "epochs"}},
            "planner": {"n_utts": 4_200_000, "fingerprint": "f" * 16},
            "st": dict({"step": fs, "resume_resets": 0, "total_steps": 269777 if name == "full-p005" else ts,
                        "early_stop": es}, **st)}


def source_summary(name: str, status="done", **it) -> dict:
    rid, src, fs, ts = COOL[name]
    return {"format": 1, "box": src, "status": status, "items": {name: dict(
        {"kind": "train", "status": "done", "verified": True, "run_dir": f"runs/{rid}",
         "result": {"steps": ts, "resume_resets": 0}}, **it)}}


def cool_runs() -> dict:
    """The runs repo before box p-cool's first launch: both source summaries, both pre_cooldown states, both ends."""
    runs = {}
    for name, (rid, src, fs, ts) in COOL.items():
        runs[fullrun.box_summary_path(src)] = source_summary(name)
        for f in fullrun.STATE_FILES_REQUIRED:
            runs[f"runs/{rid}/checkpoints/full_step_{fs}/{f}"] = cool_brief(name) if f == "trainer.json" else b"x"
        runs[f"runs/{rid}/checkpoints/step_{ts}/model.safetensors"] = b"w"
        runs[f"runs/m4-{rid}/summary.json"] = b"{}"  # the source's own readout (r0): never in the way
    return runs


def state_path(name: str, f: str = "trainer.json") -> str:
    rid, _, fs, _ = COOL[name]
    return f"runs/{rid}/checkpoints/full_step_{fs}/{f}"


def own_summary(**items) -> dict:
    """Box p-cool's own Hub summary with the given items."""
    return {fullrun.box_summary_path("p-cool"): {"format": 1, "box": "p-cool", "status": "failed", "items": items}}


def record(name: str, **kw) -> dict:
    """A train item of box p-cool's own summary with the continuation record resume-pull wrote for it."""
    rid = COOL[name][0]
    sets = ["schedule.resume_reset=true", *fullrun.continue_sets({"sets": COOL_SETS[name]})]
    return dict({"kind": "train", "status": "running", "run_dir": f"runs/{rid}", "verified": False,
                 "continuation": {"run_id": rid, "reset": True, "sets": sets, "resume_resets_before": 0}}, **kw)


def test_a_continuation_box_launches_as_a_resume_without_the_flag(full_launch, repo, capsys):
    """DECISIONS H14: box p-cool's train items continue other boxes' runs, so every launch of it is a resume
    (KITSUNE_RESUME=1, no --resume needed) and the registry, not the env, carries each reset and its sets: no
    KITSUNE_RESUME_RESET / KITSUNE_RESUME_SETS; full_preflight checks it as a resume."""
    cool_registry(repo.root, repo.reg)
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p-cool", "--scratch-repo", SCRATCH, "--yes")
    out = capsys.readouterr().out
    assert rc == 0, out
    env = env_of(created(fake))
    assert env["KITSUNE_RESUME"] == "1" and "KITSUNE_RESUME_RESET" not in env and "KITSUNE_RESUME_SETS" not in env
    assert env["KITSUNE_BOX"] == "p-cool"
    a, kw = full_launch.seen["preflight"]
    assert a[5] == "p-cool" and kw["resume"] is True and kw["resets"] == [] and kw["sets"] == {}
    assert kw["allow_continued_runs"] is False
    assert ("box p-cool continues other boxes' runs (full-p03, full-p005; registry continues): launched as a resume "
            "(KITSUNE_RESUME=1") in out
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p-cool", "--scratch-repo", SCRATCH, "--resume",
                           "--dry-run")
    out = capsys.readouterr().out
    assert rc == 0 and "launched as a resume" not in out, out  # --resume says it already
    assert env_of(created_or_printed(out))["KITSUNE_RESUME"] == "1"
    # a box without continues items keeps its fresh launch
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p005", "--scratch-repo", SCRATCH, "--dry-run")
    out = capsys.readouterr().out
    assert rc == 0 and "KITSUNE_RESUME" not in env_of(created_or_printed(out)), out
    assert full_launch.seen["preflight"][1]["resume"] is False


@pytest.mark.parametrize("args, err", [
    (["--fresh-over-done"], "--fresh-over-done is refused for box p-cool: its train items (full-p03, full-p005) "
                            "continue other boxes' runs"),
    (["--resume-reset", RID03], f"--resume-reset {RID03}: item full-p03 of box p-cool continues that run (its "
                                f"registry continues block carries the reset from full_step_169376 and its sets): "
                                f"drop the flag"),
    (["--resume-set", f"{RID005}:schedule.epochs=4"], f"--resume-set {RID005}: item full-p005 of box p-cool "
                                                      f"continues that run"),
], ids=["fresh-over-done", "reset", "set"])
def test_a_continuation_box_refuses_flags_that_would_reset_or_set_its_runs_again(full_launch, repo, capsys, args, err):
    """The registry carries the continuations' resets and sets: --fresh-over-done (it is never fresh) and a
    --resume-reset/--resume-set naming a continued run (a relaunch would restart the continuation from its
    pre_cooldown state and throw its progress away; resume-pull refuses them on the box) are refused before renting."""
    cool_registry(repo.root, repo.reg)
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p-cool", "--scratch-repo", SCRATCH, *args, "--yes")
    out = capsys.readouterr().out
    assert rc == 1 and err in out and created(fake) is None, out
    assert launch.continues_flag_problems("p-cool", {}, [RID03], {RID03: []}) == []  # a box without continuations


@pytest.mark.parametrize("label, row, refused", [
    ("kitsune-full-p-cool-data-full-0123456", {"actual_status": "running"},
     "a live instance of box p-cool: kitsune-full-p-cool-"),
    ("kitsune-full-p005-data-full-0123456", {"actual_status": "loading"},
     "a live instance of source box p005 (its runs continue here)"),
    ("kitsune-full-full-data-full-0123456", {"actual_status": "created"},
     "a live instance of source box full (its runs continue here)"),
    # a just-created second box p-cool: vast still schedules it, actual_status null, the other two say running
    ("kitsune-full-p-cool-data-full-0123456", {"actual_status": None, "intended_status": "running",
                                               "cur_state": "running"}, "a live instance of box p-cool"),
    ("kitsune-full-p-cool-data-full-0123456", {}, "a live instance of box p-cool"),  # no actual_status at all
    ("kitsune-full-p005-data-full-0123456", {"actual_status": "exited", "intended_status": "running"},
     "a live instance of source box p005"),  # vast is starting it again
    ("kitsune-full-p-cool-data-full-0123456", {"actual_status": "exited"}, None),
    ("kitsune-full-p-cool-data-full-0123456", {"actual_status": "offline", "intended_status": "stopped",
                                               "cur_state": "stopped"}, None),
    ("kitsune-full-p01-data-p01-0123456", {"actual_status": "running"}, None),
], ids=["own-running", "source-loading", "source-created", "own-null-scheduling", "own-no-status",
        "source-exited-restarting", "own-exited", "own-offline-stopped", "unrelated"])
def test_live_instances_of_a_continuation_box_or_its_sources_are_refused(full_launch, repo, capsys, label, row,
                                                                         refused):
    """Two boxes must never write one run dir: a second box p-cool (a re-run launch after a create that did rent:
    vast lists it with a null actual_status while it schedules it) or a live source box (its end syncs the same
    runs/<rid>/) is refused while it may hold a host (any of actual_status, intended_status, cur_state running,
    loading or created, or no actual_status yet: launch.holds_host); one clearly exited, stopped or offline is a
    warning (a host-loss relaunch may still list the lost instance); another box's instance is none of its
    business."""
    cool_registry(repo.root, repo.reg)
    rc, fake = full_launch([[offer(1, 70001, 0.81)]], "--box", "p-cool", "--scratch-repo", SCRATCH, "--yes",
                           instances=[dict({"id": 9, "label": label}, **row)])
    out = capsys.readouterr().out
    shown = row.get("actual_status") or row.get("cur_state") or "?"
    if refused:
        assert rc == 1 and created(fake) is None and refused in out, out
        assert f"{label} (instance 9, {shown}): destroy it before renting box p-cool" in out
    else:
        assert rc == 0 and created(fake) is not None, out
        assert ("WARNING: an instance of box p-cool: " in out) == ("p-cool" in label), out
    assert launch.LIVE_STATUSES == ("running", "loading", "created")


def test_a_first_launch_of_a_continuation_box_passes_with_its_sources_checked(cool):
    """No summary of box p-cool yet: each continuation is checked against its source box's summary (done, verified,
    the run and its resets the registry's) and its pre_cooldown state (complete, pre_cooldown, resume_resets 0; the
    kept early-stop cooldown of full-p005 that ends at its to_step; full-p03's end at its to_step); the notes name the
    source box, the state, the resets and the readout dir."""
    problems, notes = cool(cool_runs())
    assert problems == [], problems
    assert f"resume: {RUNS} has no full/box-p-cool/queue_summary.json yet (box p-cool's first launch)" in notes[0]
    assert (f"resume: full-p03: continues box full's run {RID03} (done and verified in "
            f"full/box-full/queue_summary.json): reset from runs/{RID03}/checkpoints/full_step_169376 "
            f"(resume_resets 0), sets ['schedule.epochs=6', 'schedule.deadline_cooldown=false'], readout dir "
            f"runs/m4-{RID03}-r1") in notes
    assert any(n.startswith(f"resume: full-p005: continues box p005's run {RID005}") and f"m4-{RID005}-r1" in n
               and "'schedule.resume_reset_keep_cooldown=true'" in n for n in notes), notes
    assert (f"resume: full-p03: runs/{RID03}/checkpoints/full_step_169376: pre_cooldown, resume_resets 0, the run ends "
            f"at step 211720, planner n_utts 4200000") in notes
    assert (f"resume: full-p005: runs/{RID005}/checkpoints/full_step_67449: pre_cooldown, resume_resets 0, early-stop "
            f"cooldown kept (t_c 67449.0 -> T 80939.0), planner n_utts 4200000") in notes


def _set(path, **kw):
    """A change of the runs repo: path's JSON object updated with kw."""
    def change(runs):
        runs[path] = dict(runs[path], **kw)
    return change


def _st(name, **kw):
    def change(runs):
        b = runs[state_path(name)]
        b["st"] = dict(b["st"], **kw)
    return change


def _cooldown(**kw):
    def change(runs):
        b = runs[state_path("full-p005")]
        b["st"]["early_stop"]["cooldown"] = dict(b["st"]["early_stop"]["cooldown"], **kw)
    return change


def _drop(path):
    return lambda runs: runs.pop(path)


@pytest.mark.parametrize("change, want", [
    (_drop("full/box-p005/queue_summary.json"),
     f"full-p005 (continues box p005's run {RID005}): {RUNS} has no full/box-p005/queue_summary.json"),
    (lambda r: r.update({"full/box-full/queue_summary.json": source_summary("full-p03", status="running")}),
     "full/box-full/queue_summary.json says running: box full may still be writing"),
    (lambda r: r.update({"full/box-full/queue_summary.json": source_summary("full-p03", status="running",
                                                                            verified=False)}), "says running"),
    (lambda r: r["full/box-full/queue_summary.json"]["items"]["full-p03"].update(status="failed"),
     "has its train item full-p03 'failed' (kind 'train', verified True), not done and verified"),
    (lambda r: r["full/box-p005/queue_summary.json"]["items"]["full-p005"].update(status="running"),
     "has its train item full-p005 'running' (kind 'train', verified True), not done and verified"),
    (lambda r: r["full/box-p005/queue_summary.json"]["items"]["full-p005"].update(verified=False),
     "not done and verified"),
    (lambda r: r["full/box-p005/queue_summary.json"]["items"].pop("full-p005"), "has its train item full-p005 None"),
    (lambda r: r["full/box-full/queue_summary.json"]["items"]["full-p03"].update(
        run_dir="runs/full-p03-20261001T000000Z"),
     f"records run full-p03-20261001T000000Z for full-p03, the registry continues {RID03}"),
    (lambda r: r["full/box-full/queue_summary.json"]["items"]["full-p03"]["result"].update(resume_resets=1),
     "records resume_resets 1 for the run, the registry's resets_before is 0"),
    (_drop(state_path("full-p03", "trainer.pt")), f"{RUNS} lacks ['trainer.pt'] of runs/{RID03}/checkpoints/"
                                                  f"full_step_169376/"),
    (_drop(state_path("full-p005", "optimizer.pt")), "lacks ['optimizer.pt']"),
    (lambda r: r[state_path("full-p03")].update(reason="timed"), "its trainer.json reason is 'timed', not "
                                                                 "pre_cooldown"),
    (_st("full-p03", resume_resets=2), "its st.resume_resets is 2, the registry's resets_before 0"),
    (_st("full-p005", early_stop={"cooldown": None}), "it has no early-stop cooldown record (st.early_stop.cooldown) "
                                                     "for schedule.resume_reset_keep_cooldown to keep"),
    (_cooldown(at_step=60000), "its early-stop cooldown record starts at at_step 60000, t_c 67449.0, not at the "
                               "state's step 67449"),
    (_cooldown(t_c=60000.0), "starts at at_step 67449, t_c 60000.0"),
    (_cooldown(clock="steps"), "its early-stop cooldown record is on the 'steps' clock, the state's schedule.clock "
                               "is 'epochs'"),
    (_cooldown(T=269777.0), "its early-stop cooldown ends at T 269777.0, the registry's to_step is 80939"),
    (_st("full-p03", early_stop={"cooldown": {"t_c": 169376.0, "T": 190000.0, "clock": "epochs",
                                              "at_step": 169376}}),
     "without schedule.resume_reset_keep_cooldown the reset drops it"),
    (_st("full-p03", total_steps=215815), "its st.total_steps is 215815, the registry's to_step 211720"),
    (lambda r: r[state_path("full-p005")]["planner"].update(n_utts=4_100_000),
     "the continued states' planner n_utts differ (full-p03 4200000, full-p005 4100000): box p-cool builds one CTC "
     "store"),
    (lambda r: r.update({f"runs/m4-{RID005}-r1/tables/cer.csv": b"x"}),
     f"{RUNS} already holds runs/m4-{RID005}-r1/, which its readout would write into"),
], ids=["no-source-summary", "source-running", "source-running-unverified", "item-failed",
        "item-running", "item-unverified", "item-missing", "rid", "resets", "no-trainer.pt", "no-optimizer.pt",
        "reason", "state-resets", "keep-no-record", "keep-at-step", "keep-t_c", "keep-clock", "keep-T",
        "plain-has-record", "plain-total-steps", "n_utts", "readout-dir"])
def test_a_continuation_box_is_refused_before_renting(cool, change, want):
    """Every check resume-pull (or the trainer's keep_cooldown refusal, or check-resume) would make after the paid
    boot is made here first."""
    runs = cool_runs()
    change(runs)
    problems, _ = cool(runs)
    assert any(want in p for p in problems), (want, problems)


def test_a_continuation_that_already_ran_from_its_state_resumes(cool):
    """The continuation re-saves its pre_cooldown state at the same step with its reset (resume_resets + 1): a box
    lost before its summary went up finds it there, and resume-pull resumes it as a lost continuation."""
    runs = cool_runs()
    _st("full-p03", resume_resets=1)(runs)
    problems, notes = cool(runs)
    assert problems == [], problems
    want = (f"resume: full-p03: runs/{RID03}/checkpoints/full_step_169376: pre_cooldown, resume_resets 1 (the "
            f"continuation already ran from it: the box resumes it)")
    assert any(n.startswith(want) for n in notes), notes


def test_a_relaunch_reads_the_boxs_own_summary_first(cool):
    """Once box p-cool's summary is up, it decides: a done and verified item is adopted (its state no longer
    matters), a recorded continuation goes on (its sets must still be the registry's), and the source summaries are
    not read for them. A run id or a record that does not fit the registry is refused."""
    runs = cool_runs()
    del runs["full/box-full/queue_summary.json"], runs["full/box-p005/queue_summary.json"]  # unread now
    for f in fullrun.STATE_FILES_REQUIRED:
        del runs[state_path("full-p03", f)]  # the done item's state: unread
    runs[f"runs/m4-{RID03}-r1/summary.json"] = b"{}"  # its readout, done
    runs[f"runs/m4-{RID005}-r1/summary.json"] = b"{}"  # full-p005's readout, begun on the lost rental
    own = dict(record("full-p03", status="done", verified=True),
               result={"steps": 211720, "resume_resets": 1})
    runs.update(own_summary(**{"full-p03": own, "m4-full-p03": {"kind": "readout", "status": "done", "verified": True,
                                                               "out": f"runs/m4-{RID03}-r1"},
                               "full-p005": record("full-p005"),
                               "m4-full-p005": {"kind": "readout", "status": "failed", "verified": False,
                                                "out": f"runs/m4-{RID005}-r1"}}))
    problems, notes = cool(runs)
    assert problems == [], problems
    assert f"resume: full-p03: done and verified on box p-cool (its own summary), run {RID03}: adopted, never " \
           f"continued again" in notes
    assert any(n.startswith(f"resume: full-p005: running on box p-cool, run {RID005}, its continuation recorded "
                            f"(resume_resets before 0): goes on from the newest state holding its reset, else again "
                            f"from runs/{RID005}/checkpoints/full_step_67449; newest state: ") for n in notes), notes
    # (all train items done: a plain relaunch of the box is guard (a)'s, as for any box)
    done = copy.deepcopy(runs)
    done[fullrun.box_summary_path("p-cool")]["items"]["full-p005"].update(status="done", verified=True)
    problems, _ = cool(done)
    assert any("a plain --resume of box p-cool: every train item of its Hub summary is done" in p for p in problems)
    assert cool(done, allow_done_trains=True)[0] == []

    def refused(item: dict, want: str, name="full-p005"):
        bad = copy.deepcopy(runs)
        bad[fullrun.box_summary_path("p-cool")]["items"][name] = item
        problems, _ = cool(bad)
        assert any(want in p for p in problems), (want, problems)

    refused(record("full-p005", run_dir="runs/full-p005-20261009T000000Z"),
            f"box p-cool's Hub summary records run full-p005-20261009T000000Z for it, its registry continues block "
            f"run {RID005}")
    refused(dict(record("full-p005"), continuation=None), "box p-cool's Hub summary records the run without its "
                                                          "continuation record")
    rec = record("full-p005")
    rec["continuation"]["sets"] = rec["continuation"]["sets"][:-1]
    refused(rec, f"box p-cool's queue summary records the continuation of {RID005} with sets ")
    refused(rec, "the registry changed under a started continuation")
    # the record's resume_resets_before is the registry's resets_before too (resume-pull's continue_item refuses
    # either change on the box, after the paid boot); a missing one counts as 0, as there
    rec = record("full-p005")
    rec["continuation"]["resume_resets_before"] = 1
    refused(rec, f"with sets {rec['continuation']['sets']} after 1 resets; the registry has "
                 f"{rec['continuation']['sets']} after 0: the registry changed under a started continuation")
    rec["continuation"].pop("resume_resets_before")
    ok = copy.deepcopy(runs)
    ok[fullrun.box_summary_path("p-cool")]["items"]["full-p005"] = rec
    assert cool(ok)[0] == []
    # an item this box adopted done and verified owns its readout dir, even without its continuation record (resume-
    # pull keeps the record on a done continuation since the H14 review; an older summary may lack it): the readout's
    # re-run writes the same -r<N> dir
    other = copy.deepcopy(runs)
    items = other[fullrun.box_summary_path("p-cool")]["items"]
    items["full-p005"] = dict(record("full-p005", status="done", verified=True), continuation=None)
    items["m4-full-p005"].update(out=None)
    problems, _ = cool(other)  # (every train item done now: the all-done plain-resume guard may speak, not this one)
    assert not any("already holds" in p for p in problems), problems
    # (a readout dir no record ties to this box - a first launch - stays refused: the case near line 1239)


@pytest.mark.parametrize("status", ["pending", "fresh", "failed", "running"])
def test_a_readout_dir_of_the_boxs_own_continuation_is_its_own(cool, status):
    """Rental 1 uploaded runs/m4-<rid>-r1 and was lost before the summary put that verifies it; rental 2's resume-pull
    marked that readout fresh, adopt() set it pending with no out, the summary went up, and that host was lost too
    before the readout ran again. Rental 3 is no other box's readout: the own summary records the train item as this
    box's continuation of the run, and the readout's re-run writes the same -r<N> dir, whatever its status."""
    runs = cool_runs()
    runs[f"runs/m4-{RID005}-r1/summary.json"] = b"{}"
    runs.update(own_summary(**{"full-p005": record("full-p005"),
                               "m4-full-p005": {"kind": "readout", "status": status, "verified": False, "out": None}}))
    problems, _ = cool(runs)
    assert problems == [], problems
    # the train item's continuation record decides it: a summary without the readout item passes the same way
    runs[fullrun.box_summary_path("p-cool")]["items"].pop("m4-full-p005")
    assert cool(runs)[0] == []


def test_a_resume_of_a_box_whose_runs_are_continued_elsewhere_needs_the_flag(cool):
    """After box p-cool, box p005's run dir holds p-cool's continuation (its final export at the same step): a
    --resume (or --resume-reset/--resume-set) of box p005 would adopt, score or sync over those weights, so it is
    refused unless --allow-continued-runs; a box whose runs nobody continues is not concerned."""
    runs = cool_runs()
    runs["full/box-p005/queue_summary.json"]["items"]["m4-full-p005"] = {"kind": "readout", "status": "running"}
    for kw in ({}, {"resets": [RID005]}, {"sets": {RID005: ["schedule.epochs=12"]}, "resets": [RID005]}):
        problems, _ = cool(runs, box="p005", **kw)
        assert any(f"a resume of box p005: its run(s) {RID005} (full-p005) are continued by box p-cool" in p
                   and "--allow-continued-runs" in p for p in problems), (kw, problems)
    problems, notes = cool(runs, box="p005", resets=[RID005], allow_continued_runs=True)
    assert problems == [] and any("(--allow-continued-runs: not refused)" in n for n in notes), (problems, notes)
    runs[fullrun.box_summary_path("p01")] = {"items": {"full-p01": {"kind": "train", "status": "running",
                                                                    "run_dir": "runs/full-p01-20261001T184145Z"}}}
    assert cool(runs, box="p01")[0] == []


def test_the_allow_continued_runs_flag_reaches_the_preflight_and_is_full_only(full_launch, capsys):
    rc, _ = full_launch([[offer(1, 70001, 0.81)]], "--box", "p01", "--scratch-repo", SCRATCH, "--resume",
                        "--allow-continued-runs", "--dry-run")
    assert rc == 0, capsys.readouterr().out
    assert full_launch.seen["preflight"][1]["allow_continued_runs"] is True
    with pytest.raises(SystemExit):
        launch.main(["--job", "train", "--data-repo", DATA, "--out-repo", RUNS, "--allow-continued-runs"])
    assert "--allow-continued-runs: for --job full only" in capsys.readouterr().err


def test_full_preflight_checks_a_continuation_box_as_a_resume_and_its_banks(cool, repo, monkeypatch, devslice):
    """full_preflight of box p-cool runs the resume checks even without resume=True (it is never fresh), and the
    background banks its continuations' registry sets name (augment_pins) are held to the box's pull and their pin
    as a config's are."""
    reg = cool.reg
    reg["boxes"]["p-cool"]["extra_dirs"] = ["aug/bank-v1"]
    data = dict(box_data("p-cool", reg, repo.root), **{"aug/bank-v1/index.json": b"{}"})
    problems, notes = preflight(monkeypatch, FullHub(data, {}), box="p-cool", reg=reg)
    assert any("has no full/box-full/queue_summary.json" in p for p in problems), problems
    assert not any("a fresh launch" in p for p in problems)
    assert any(f"full-p005 (continues {RID005}) pins aug/bank-v1/index.json at sha256 {'ab' * 6}..." in p
               for p in problems), problems  # the registry's pin, lowercased, is not the bank's
    problems, notes = preflight(monkeypatch, FullHub(data, cool_runs()), box="p-cool", reg=reg)
    assert [p for p in problems if "bank" not in p] == [], problems
    spec = fullrun.box_spec("p-cool", reg)
    banks, cuts = launch.augment_pins(spec, lambda rel: json.loads((repo.root / rel).read_text(encoding="utf-8")),
                                      None)
    assert banks == [(f"full-p005 (continues {RID005})", "aug/bank-v1", "ab" * 32)] and cuts == []
    rir = copy.deepcopy(spec)
    rir["items"][3]["continues"]["sets"].update({"augment.rir_bank": "aug/rirs-v1"})
    banks, _ = launch.augment_pins(rir, lambda rel: {}, None)
    assert (f"full-p005 (continues {RID005})", "aug/rirs-v1", None) in banks



# ========================================================================================= blocklist and gates


# ======================================================================================== the quant go signal (F2)


GO_SHA = "fedcba9876543210fedcba9876543210fedcba98"


def go_verdict(**kw) -> dict:
    """A passing smoke-b verdict (checks 12-16; smoke B has no built-in checks) at GO_SHA."""
    checks = {n: {"pass": True, "evidence": []} for n in launch.QUANT_GO_CHECKS}
    return dict({"format": 1, "box": "smoke-b", "sha": GO_SHA, "machine_id": "149252",
                 "time_utc": "2026-10-01T23:10:00+00:00", "overall": "pass", "checks": checks}, **kw)


def go_runs(verdict=None) -> dict:
    """The runs repo's files of a passing smoke-b (its verdict at full/box-smoke-b/smoke_verdict.json)."""
    return {fullrun.box_verdict_path("smoke-b"): go_verdict() if verdict is None else verdict}


@pytest.fixture
def quant_go(repo, monkeypatch):
    """quant_go_problems of box full at SHA against a hub with the given verdict (None: none on the Hub); the verdict's
    commit an ancestor with the same quant code unless .ancestry / .blobs say otherwise."""
    st = SimpleNamespace(ancestry="ancestor", blobs={}, asked=[])
    monkeypatch.setattr(launch, "git_ancestry", lambda old, new: st.asked.append((old, new)) or st.ancestry)
    monkeypatch.setattr(launch, "git_blob", lambda sha, p: st.blobs.get((sha, p), f"blob:{p}"))

    def go(verdict="pass", box="full", allow=False, spec=None):
        runs = {} if verdict is None else go_runs(None if verdict == "pass" else verdict)
        hub = FullHub({}, runs=runs)
        monkeypatch.setattr(launch, "_hub", lambda: (hub, hub.download))
        spec = spec if spec is not None else fullrun.box_spec(box, repo.reg)
        return launch.quant_go_problems(RUNS, SHA, box, spec, allow_unverified_quant=allow)
    go.st = st
    return go


def test_the_quant_go_signal_passes_a_verified_box(quant_go):
    problems, notes = quant_go()
    assert problems == [], problems
    assert notes[0] == ("smoke-b verdict: sha fedcba987654, machine 149252, 2026-10-01T23:10:00+00:00, overall pass, "
                        "check 12 True, check 13 True, check 14 True, check 15 True, check 16 True")
    assert notes[1].startswith("quant go signal: the smoke-b verdict at fedcba987654 passed checks 12-16, and "
                               "kitsune/quant.py, tools/speed_probe.py")
    assert quant_go.st.asked == [(GO_SHA, SHA)]


@pytest.mark.parametrize("case, want", [
    ("missing", "no readable full/box-smoke-b/smoke_verdict.json in Multy123/kitsune-runs (FileNotFoundError"),
    ("overall", "its overall is 'fail'"),
    ("check16", "check(s) 16 False (each of 12-16 must pass)"),
    ("absent", "check(s) 14 absent (each of 12-16 must pass)"),
    ("null", "check(s) 13 None (each of 12-16 must pass)"),
    ("nosha", "it names no commit (sha None)"),
    ("unknown", "its commit fedcba987654 is not in this clone (git fetch origin)"),
    ("not_ancestor", "its commit fedcba987654 is not an ancestor of 0123456789ab"),
    ("quant", "kitsune/quant.py differs between fedcba987654 (verified) and 0123456789ab"),
    ("pins", "requirements-train.txt, docker/Dockerfile differ between fedcba987654"),
])
def test_the_quant_go_signal_refusals(quant_go, case, want):
    """Box full is refused without a verdict, with a failed one or a check of 12-16 not passed, or when the verified
    commit is unknown, not an ancestor of the one the box runs, or ran other quant code (QUANT_CODE: the quant path,
    its CLIs and the image's pins)."""
    v = go_verdict()
    if case == "overall":
        v["overall"] = "fail"
    elif case == "check16":
        v["checks"]["16"]["pass"] = False
    elif case == "absent":
        del v["checks"]["14"]
    elif case == "null":
        v["checks"]["13"]["pass"] = None
    elif case == "nosha":
        del v["sha"]
    elif case in ("unknown", "not_ancestor"):
        quant_go.st.ancestry = case
    elif case == "quant":
        quant_go.st.blobs[(SHA, "kitsune/quant.py")] = "changed"
    elif case == "pins":
        quant_go.st.blobs.update({(GO_SHA, "requirements-train.txt"): "old", (GO_SHA, "docker/Dockerfile"): "old"})
    problems, notes = quant_go(None if case == "missing" else v)
    assert len(problems) == 1 and want in problems[0], problems
    assert problems[0].startswith("box full's 2 quantised item(s) (e.g. quant-int8-w8a8-full-p03) need a passing "
                                  "smoke-b verdict (checks 12-16) at this quant code (DECISIONS F2): ")
    assert problems[0].endswith("rent the standalone smoke-B first (launch --job full --box smoke-b) or pass "
                                "--allow-unverified-quant")
    # --allow-unverified-quant: the same text as a warning, nothing refused
    problems, notes = quant_go(None if case == "missing" else v, allow=True)
    assert problems == [] and any(n.startswith("WARNING: box full's 2 quantised item(s)") and want in n
                                  and n.endswith("(--allow-unverified-quant: not refused)") for n in notes), notes


def test_the_quant_go_signal_only_concerns_boxes_with_quantised_items(quant_go, repo, monkeypatch):
    """smoke-b (the verifier) and the boxes without a quantised item never read the verdict; a box whose only
    quantised item is a speed probe with --quant needs it."""
    class Unreadable(FullHub):
        def download(self, *a, **kw):
            raise AssertionError("the verdict must not be read")

    hub = Unreadable({})
    for box in ("p01", "full-smoke", "smoke-b"):
        monkeypatch.setattr(launch, "_hub", lambda: (hub, hub.download))
        assert launch.quant_go_problems(RUNS, SHA, box, fullrun.box_spec(box, repo.reg)) == ([], []), box
    assert launch.quant_items(fullrun.box_spec("full", repo.reg)) == ["quant-int8-w8a8-full-p03",
                                                                     "quant-int8-w8a8-full-p01"]
    speed_only = {"items": [{"name": "speed-x", "kind": "speed", "args": ["--quant", "int8-w8a8"]},
                            {"name": "speed-y", "kind": "speed", "args": ["--compile"]},
                            {"name": "fp16-x", "kind": "eval", "argv": ["{python}", "scripts/05_evaluate.py", "--quant",
                                                                        "fp16"]},
                            {"name": "whisper", "kind": "eval", "argv": ["{python}", "tools/whisper_eval.py"]}]}
    assert launch.quant_items(speed_only) == ["speed-x", "fp16-x"]
    problems, _ = quant_go(None, box="p01", spec=speed_only)
    assert problems and "box p01's 2 quantised item(s) (e.g. speed-x)" in problems[0]


def test_full_preflight_carries_the_quant_go_signal_and_the_hours_warning(repo, monkeypatch, devslice):
    """full_preflight ends with the quant go signal (allow_unverified_quant passed through) and, for a box whose hours
    come from the speed record (SPEED_RECORD_BOXES: full-t, p005, p-cool; the tiny registry's 2-GPU box full stands
    in for them here), a warning while that record has no box-1 part (never a refusal)."""
    monkeypatch.setattr(launch, "SPEED_RECORD_BOXES", ("full",))
    monkeypatch.setattr(launch, "git_ancestry", lambda old, new: "ancestor")
    monkeypatch.setattr(launch, "git_blob", lambda sha, p: f"blob:{p}")
    reg = launch.full_registry(SHA)[0]
    data = box_data("full", reg, repo.root)
    problems, notes = preflight(monkeypatch, FullHub(data), box="full")
    assert len(problems) == 1 and "need a passing smoke-b verdict" in problems[0], problems
    want = f"WARNING: no {launch.SPEED_RECORD} at 0123456789ab: box full's hours are not held to measured speeds"
    assert want in notes
    problems, notes = preflight(monkeypatch, FullHub(data), box="full", allow_unverified_quant=True)
    assert problems == [] and any(n.startswith("WARNING: box full's 2 quantised item(s)") for n in notes)
    rec = repo.root / launch.SPEED_RECORD
    rec.parent.mkdir(parents=True, exist_ok=True)
    rec.write_text(json.dumps({"smoke": {}, "box1": None}), encoding="utf-8")
    problems, notes = preflight(monkeypatch, FullHub(data, runs=go_runs()), box="full")
    assert problems == [] and any(n.startswith("quant go signal: ") for n in notes)
    assert any(n.startswith("WARNING: box full's hours are smoke-only") for n in notes), notes
    rec.write_text(json.dumps({"smoke": {}, "box1": {"sha": "ab" * 20, "sec_per_step": 0.29}}), encoding="utf-8")
    notes = preflight(monkeypatch, FullHub(data, runs=go_runs()), box="full")[1]
    assert not any("smoke-only" in n for n in notes) and any("box 1 at abababababab, 0.29 s/step" in n for n in notes)
    assert launch.speed_record_notes(SHA, "p01") == []  # not in the patched tuple
    monkeypatch.setattr(launch, "SPEED_RECORD_BOXES", ("p01", "full-t", "full-p"))
    assert launch.speed_record_notes(SHA, "full-smoke") == [] and launch.speed_record_notes(SHA, "full-t")


def test_allow_unverified_quant_reaches_the_preflight_and_is_full_only(full_launch, capsys):
    rc, fake = full_launch([[offer(1, 10, 1.40, num_gpus=2, cpu_ram=130000)]], "--box", "full", "--scratch-repo",
                           SCRATCH, "--allow-unverified-quant")
    assert rc == 0, capsys.readouterr().out
    assert full_launch.seen["preflight"][1]["allow_unverified_quant"] is True
    rc, _ = full_launch([[offer(1, 10, 1.40, num_gpus=2, cpu_ram=130000)]], "--box", "full", "--scratch-repo",
                        SCRATCH)
    assert rc == 0 and full_launch.seen["preflight"][1]["allow_unverified_quant"] is False
    with pytest.raises(SystemExit):
        launch.main(["--job", "study", "--box", "A", "--data-repo", DATA, "--out-repo", RUNS,
                     "--allow-unverified-quant"])
    assert "--allow-unverified-quant: for --job full only" in capsys.readouterr().err


def test_offers_in_a_country_without_hub_access_are_dropped():
    job = launch.full_job("smoke-b", {"gpus": 1}, "5090", 1.75, 3, 1.0)
    assert job.avoid_countries == ("CN",)
    for geo, bad in ((", CN", True), ("Beijing, CN", True), ("Tokyo, JP", False), ("Hong Kong, HK", False),
                     (None, False)):
        o = offer(1, 58555, 0.5, cpu_ram=64439, **({"geolocation": geo} if geo is not None else {}))
        problems = launch.offer_problems(o, job)
        assert bool(problems) == bad, (geo, problems)
        if bad:
            assert problems == [f"geolocation {geo!r}: the Hugging Face Hub is not reachable from there"]
    cn = offer(1, 58555, 0.5, cpu_ram=64439, geolocation=", CN")
    assert launch.offer_problems(cn, launch.JOBS["train"]) == []  # the other jobs keep their filters
    assert [o["id"] for o in launch.rank_offers([cn, offer(2, 149252, 0.72, cpu_ram=64439, geolocation="Japan, JP")],
                                                job)] == [2]


def test_the_quant_code_and_the_speed_record_exist_in_this_checkout():
    for rel in launch.QUANT_CODE:
        assert (ROOT / rel).is_file(), rel
    rec = json.loads((ROOT / launch.SPEED_RECORD).read_text(encoding="utf-8"))
    assert set(rec) >= {"smoke", "box1"}
    assert launch.SPEED_RECORD_BOXES == ("full-t", "p005", "p-cool") and launch.QUANT_GO_BOX in fullrun.BOX_NAMES
    assert set(launch.SPEED_RECORD_BOXES) <= set(fullrun.BOX_NAMES)


def test_the_blocklist_holds_151760_and_54650_and_refuses_a_broken_file(tmp_path):
    bl = launch.load_blocklist()
    assert "151760" in bl and "2.9 MB/s" in bl["151760"]
    assert "54650" in bl and "2026-10-01" in bl["54650"] and "never started" in bl["54650"]  # box 2's dead m54650
    raw = json.loads(launch.BLOCKLIST.read_text(encoding="utf-8"))
    assert set(raw) == {"_comment", "machines"}
    for bad in ('{"machines": ["151760"]}', '{"machines": {"m1": "x"}}', '{"machines": {"1": ""}}',
                '{"machines": {}, "extra": 1}', "not json"):
        f = tmp_path / "bl.json"
        f.write_text(bad, encoding="utf-8")
        with pytest.raises(launch.LaunchError):
            launch.load_blocklist(f)
    with pytest.raises(launch.LaunchError):
        launch.load_blocklist(tmp_path / "missing.json")


def test_gate_refusals_avoid_slow_hosts_for_30_days(monkeypatch):
    now = 2_000_000_000.0
    runs = {"full/box-p01/infra/C1/download_gate.json": {"verdict": "slow", "machine_id": "151761", "wall": now - DAY,
                                                          "reason": "2.9 MB/s"},
            "full/box-full/infra/C2/download_gate.json": {"verdict": "slow", "machine_id": "42", "wall": now - 31 * DAY},
            "full/box-p01/infra/C3/download_gate.json": {"verdict": "pass", "machine_id": "43", "wall": now},
            "full/box-p01/infra/C4/queue.json": {"verdict": "slow", "machine_id": "44", "wall": now},
            "runs/x/download_gate.json": {"verdict": "slow", "machine_id": "45", "wall": now}}
    hub = FullHub({}, runs)
    monkeypatch.setattr(launch, "_hub", lambda: (hub, hub.download))
    got, notes = launch.gate_refusals(RUNS, now=now)
    assert set(got) == {"151761"} and "download gate said slow 1.0 d ago" in got["151761"]
    assert any("42" in n and "no longer avoided" in n for n in notes)
    assert launch.GATE_BLOCK_DAYS == 30

    def broken():
        raise RuntimeError("Hub down")

    monkeypatch.setattr(launch, "_hub", broken)
    got, notes = launch.gate_refusals(RUNS, now=now)
    assert got == {} and "could not read the download gates" in notes[0]


def test_every_job_avoids_the_blocklist(monkeypatch, capsys):
    """The train job's ranking skips a blocklisted machine too (vast/blocklist.json is for every job)."""
    import test_infra as ti

    fake = ti.FakeVastai([[dict(ti.OFFERS_40[1], machine_id=151760), ti.OFFERS_40[0]]])
    monkeypatch.setattr(launch.shutil, "which", lambda name: "/fake/vastai" if name == "vastai" else None)
    monkeypatch.setattr(launch.subprocess, "run", fake)
    assert launch.main(ti.launch_args("--dry-run")) == 0
    out = capsys.readouterr().out
    assert "avoiding machine 151760" in out and "create instance 222" in out


# ====================================================================================================== finish


def full_run(root: Path) -> Path:
    """A full box's run dir: logs, two weights dirs, a timed state still marked for the scratch repo, the
    pre_cooldown state marked for the runs repo, and an older timed state whose upload finished."""
    run = root / "runs" / "full-p01-20260927T120000Z"
    cfg = {"schedule": {"clock": "epochs"}, "ckpt": {"upload_full_at": ["pre_cooldown"]}}
    files = {"config.json": json.dumps({"config": cfg}).encode(), "events.jsonl": b"{}\n", "summary.json": b"{}",
             "checkpoints/step_900/model.safetensors": b"w9", "checkpoints/step_1000/model.safetensors": b"wA",
             "checkpoints/full_step_800/model.pt": b"f8", f"checkpoints/full_step_800/{finish.UPLOAD_MARK}": b"",
             "checkpoints/full_step_1000/model.pt": b"fA", f"checkpoints/full_step_1000/{finish.SCRATCH_MARK}": b"",
             "checkpoints/full_step_600/model.pt": b"f6"}
    for rel, data in files.items():
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).write_bytes(data)
    return run


def test_finish_names_match_the_full_core():
    assert finish.SCRATCH_MARK == fullrun.SCRATCH_MARK and finish.GATE_FILE == fullrun.GATE_FILE
    assert finish.UPLOAD_MARK == fullrun._UPLOAD_MARK


def test_lean_full_files_never_hold_a_marker_nor_a_timed_state(tmp_path):
    run = full_run(tmp_path)
    got = {p.split("/", 2)[2] for p in finish.expected_files(run, False, lean=True)}
    assert got == {"config.json", "events.jsonl", "summary.json", "checkpoints/step_900/model.safetensors",
                   "checkpoints/step_1000/model.safetensors", "checkpoints/full_step_800/model.pt"}
    # even a job that expects the newest full state never uploads the scratch marker
    assert not any(p.endswith(finish.SCRATCH_MARK) for p in finish.expected_files(run))


class FinishHub:
    def __init__(self, run):
        self.run, self.uploads, self.commits, self.sent = run, [], [], {}

    def upload_folder(self, **kw):
        self.uploads.append(kw["allow_patterns"])

    def create_commit(self, **kw):
        self.commits.append([op.path_in_repo for op in kw["operations"]])
        self.sent.update({op.path_in_repo: op.path_or_fileobj for op in kw["operations"]})

    def list_repo_tree(self, repo, path_in_repo=None, recursive=False, repo_type=None):
        exp = finish.expected_files(self.run, False, lean=True)
        return [SimpleNamespace(path=p, size=f.stat().st_size, lfs=None, blob_id=finish.git_blob_id(f))
                for p, f in exp.items()]


@pytest.fixture
def full_finish(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    (state / "queue.json").write_text("{}", encoding="utf-8")
    (state / "logs").mkdir()
    (state / "logs" / "full-p01.log").write_text("step 1\n", encoding="utf-8")
    monkeypatch.setattr(finish, "STATE_DIR", state)
    actions = []
    monkeypatch.setattr(finish, "vast_rest", lambda action, timeout=30: actions.append(action) or True)
    monkeypatch.setenv("KITSUNE_JOB", "full")
    monkeypatch.setenv("KITSUNE_BOX", "p01")
    monkeypatch.setenv("CONTAINER_ID", "C77")
    monkeypatch.setenv("KITSUNE_MACHINE_ID", "70001")

    def go(*args, run=True):
        r = full_run(tmp_path) if run else None
        hub = FinishHub(r)
        monkeypatch.setattr(finish, "hf_api", lambda: hub)
        rc = finish.main([*args, "--repo", RUNS, "--runs-root", str(tmp_path / "runs")])
        events = [json.loads(x) for x in (state / "events.jsonl").read_text().splitlines()] \
            if (state / "events.jsonl").exists() else []
        return rc, hub, actions, events
    go.state = state
    return go


def test_finish_full_is_lean_with_its_infra_under_the_box(full_finish):
    rc, hub, actions, _ = full_finish("--destroy")
    assert rc == 0 and actions == ["destroy"]
    ck = [p for pats in hub.uploads for p in pats if p.startswith("checkpoints/")]
    assert "checkpoints/full_step_800/model.pt" in ck
    assert not any("full_step_1000" in p or "full_step_600" in p or p.endswith(finish.SCRATCH_MARK) for p in ck)
    infra = [p for c in hub.commits for p in c]
    assert "full/box-p01/infra/C77/queue.json" in infra and "full/box-p01/infra/C77/logs/full-p01.log" in infra


def test_abort_destroys_a_box_without_a_run_dir_with_the_gates_reason(full_finish):
    gate = {"verdict": "slow", "reason": "2.9 MB/s < 31.7 MB/s: 571.2 GB would take 54.7 h > 5 h",
            "machine_id": "151761"}
    (full_finish.state / "download_gate.json").write_text(json.dumps(gate), encoding="utf-8")
    rc, hub, actions, events = full_finish("--abort", "--reason", "onstart failed at line 298 (exit 3)", run=False)
    assert rc == 0 and actions == ["destroy"], "no run dir: nothing on the disk is unique"
    (ab,) = [e for e in events if e["kind"] == "abort"]
    assert ab["destroyed"] is True and ab["machine_id"] == "70001"
    assert ab["reason"].startswith("download gate: 2.9 MB/s") and "onstart failed at line 298" in ab["reason"]
    assert hub.uploads == [], "no sync"
    assert "full/box-p01/infra/C77/download_gate.json" in [p for c in hub.commits for p in c]
    halt = json.loads((full_finish.state / "halt").read_text(encoding="utf-8"))
    assert halt["action"] == "destroy" and "download gate" in halt["reason"]


def test_abort_stops_a_box_with_a_run_dir(full_finish):
    rc, hub, actions, events = full_finish("--abort", "--reason", "bootstrap failed")
    assert rc == 0 and actions == ["stop"] and hub.uploads == []
    (ab,) = [e for e in events if e["kind"] == "abort"]
    assert ab["destroyed"] is False and ab["reason"] == "bootstrap failed"


@pytest.mark.parametrize("job", ["study", "train", "label"])
def test_abort_is_stop_no_sync_for_every_other_job(full_finish, monkeypatch, job):
    monkeypatch.setenv("KITSUNE_JOB", job)
    if job == "label":
        seen = {}
        monkeypatch.setattr(finish, "label_main", lambda args: seen.update(vars(args)) or 0)
        assert finish.main(["--abort", "--reason", "x"]) == 0
        assert seen["stop"] is True and seen["no_sync"] is True and seen["abort"] is False
        return
    rc, hub, actions, events = full_finish("--abort", "--reason", "onstart failed", run=False)
    assert rc == 0 and actions == ["stop"] and not [e for e in events if e["kind"] == "abort"]


# ==================================================================================================== watchdog


def box_watchdog(tmp_path, orphan_s: str, action: str | None, **kw):
    env, state, record = watchdog_env(tmp_path, orphan_s, KITSUNE_WATCHDOG_HB_FILE="train_hb", **kw)
    env.pop("KITSUNE_WATCHDOG_ORPHAN_ACTION", None)
    if action is not None:
        env["KITSUNE_WATCHDOG_ORPHAN_ACTION"] = action
    return env, state, record


def test_watchdog_stops_a_box_whose_train_hb_went_stale(tmp_path):
    bash = need_bash()
    env, state, record = box_watchdog(tmp_path, "2", None)
    t0 = int(time.time())
    hb = state / "train_hb"
    hb.write_text("", encoding="utf-8")
    os.utime(hb, (t0 + 1, t0 + 1))
    r = subprocess.run([bash, str(VAST / "watchdog.sh")], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    got = calls(record)
    assert [a for _, a in got] == ["--sync-only", "--stop --no-sync --reason watchdog: box controller heartbeat stale"]
    assert "box controller heartbeat stale" in r.stdout and not (state / "label_hb").exists()


def test_watchdog_alert_mode_records_and_rearms_without_stopping(tmp_path):
    """Smoke A's freeze fault: the controller heartbeat goes stale on purpose; the watchdog appends one orphan_alert
    per stale spell, never syncs or stops, and re-arms when the file is fresh again."""
    bash = need_bash()
    env, state, record = box_watchdog(tmp_path, "1", "alert", FAKE_SLEEP_MAX="12")
    hb = state / "train_hb"
    hb.write_text("", encoding="utf-8")
    t0 = int(time.time())
    os.utime(hb, (t0 + 1, t0 + 1))
    # a toucher that revives the file after ~5 s, then lets it go stale again
    toucher = subprocess.Popen([sys.executable, "-c",
                                "import os, sys, time; time.sleep(5.5); os.utime(sys.argv[1], None)", str(hb)])
    try:
        r = subprocess.run([bash, str(VAST / "watchdog.sh")], capture_output=True, text=True, env=env, timeout=90)
    finally:
        toucher.wait(30)
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls(record) == [], "alert mode never syncs or stops"
    alerts = [json.loads(x) for x in (state / "watchdog_alerts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(alerts) == 2, (alerts, r.stdout)
    assert all(a["kind"] == "orphan_alert" and a["hb"] == "train_hb" and a["limit_s"] == 1 and a["age_s"] >= 2
               for a in alerts)
    assert "the alert re-armed" in r.stdout and not (state / "halt").exists()


def test_watchdog_dry_run_names_the_file_and_the_action(tmp_path):
    bash = need_bash()
    env, state, _ = box_watchdog(tmp_path, "600", "alert")
    r = subprocess.run([bash, str(VAST / "watchdog.sh"), "--dry-run"], capture_output=True, text=True, env=env,
                       timeout=60)
    assert r.returncode == 0 and "train_hb is stale by 600 s" in r.stdout and "orphan_alert" in r.stdout, r.stdout
    assert "then stop the instance" not in r.stdout and "no sync, no stop" in r.stdout


# ================================================================================ the chained box (addendum E)

from fixtures_chain import CHAIN_BOX, with_chain  # noqa: E402

SMOKE_SIZING = dict(down_gb=59.2, shard_gb=100.0, sel_gb=12.0, stores=2, labels_gb=5.5, hours=1000.0, extra_gb=0.0,
                    disk_gb=400, rebuild_timeout_min=120)


@pytest.fixture
def chain_launch(repo, monkeypatch):
    """launch.main for --job full --box p01-chain on the tiny registry + the E.1.7 entry, the Hub side recorded: every
    hf_preflight / extent_preflight / full_preflight call (a part's problems from .seen["part_problems"]), and
    chain_preflight's."""
    write_reg(repo.root, with_chain(repo.reg))
    seen = {"hf": [], "extent": [], "preflight": [], "chain": [], "part_problems": {}}

    def hf_preflight(data, out, cfg):
        seen["hf"].append(cfg["selection"])
        return "d" * 40, []

    def extent_preflight(data, rev, cfg, extra_gb=0.0):
        seen["extent"].append((cfg["selection"], extra_gb))
        full = not (cfg.get("extent") or {}).get("inputs")  # box 1's uncapped extent, else stage 1's
        return [], dict(SIZING if full else SMOKE_SIZING, extra_gb=extra_gb)

    def full_preflight(*a, **kw):
        seen["preflight"].append((a, kw))
        return list(seen["part_problems"].get(a[5], [])), [f"{a[5]} preflight ok"]

    monkeypatch.setattr(launch, "hf_preflight", hf_preflight)
    monkeypatch.setattr(launch, "extent_preflight", extent_preflight)
    monkeypatch.setattr(launch, "full_preflight", full_preflight)
    monkeypatch.setattr(launch, "chain_preflight",
                        lambda *a, **kw: seen["chain"].append((a, kw)) or ([], ["chain preflight ok"]))
    monkeypatch.setattr(launch, "avoided_machines", lambda data, rev: (set(), []))
    monkeypatch.setattr(launch, "gate_refusals", lambda out: ({}, []))

    def go(searches, *args, box=CHAIN_BOX, instances=()):
        fake = FakeVastai(searches, instances)
        monkeypatch.setattr(launch.shutil, "which", lambda name: "/fake/vastai" if name == "vastai" else None)
        monkeypatch.setattr(launch.subprocess, "run", fake)
        rc = launch.main(["--job", "full", "--box", box, "--data-repo", DATA, "--out-repo", RUNS, "--sha", SHA,
                          "--image", DIGEST_IMAGE, "--skip-git-checks", *args])
        return rc, fake
    go.seen = seen
    return go


def test_the_chain_rents_one_5090_with_its_derived_env(chain_launch, capsys):
    """E.1.8: KITSUNE_CONFIG is stage 1's rebuild, the watchdog's stage-1 env with its hand-over bound, the disk and
    the gate on box 1's extent (+ the chain's extra_gb), the boot's rebuild bytes and timeout on stage 1's; every part
    preflighted as a box, each distinct data config's selection checked, the chain's own files; 35 h, 25.2 h, $1.00."""
    rc, fake = chain_launch([[offer(1, 70001, 0.81)]], "--scratch-repo", SCRATCH, "--yes")
    out = capsys.readouterr().out
    assert rc == 0, out
    create = created(fake)
    env = env_of(create)
    assert env == {
        "KITSUNE_JOB": "full", "KITSUNE_BOX": CHAIN_BOX, "KITSUNE_SHA": SHA,
        "KITSUNE_CONFIG": "configs/full/data-smoke.json", "KITSUNE_DATA_REPO": DATA, "KITSUNE_OUT_REPO": RUNS,
        "KITSUNE_N_GPUS": "1", "KITSUNE_WATCHDOG_HB_FILE": "train_hb", "KITSUNE_WATCHDOG_ORPHAN_S": "600",
        "KITSUNE_WATCHDOG_ORPHAN_ACTION": "alert", "KITSUNE_CHAIN_STAGE": "1", "KITSUNE_WATCHDOG_HANDOVER_S": "34200",
        "KITSUNE_SCRATCH_REPO": SCRATCH, "KITSUNE_GATE_BYTES": str(int(571.3e9)), "KITSUNE_GATE_MAX_H": "5",
        "KITSUNE_REBUILD_BYTES": str(int(59.2e9)), "KITSUNE_PULL_BYTES": str(int(1e9 * (5.5 + 2))),
        "KITSUNE_MAX_HOURS": "35", "TZ": "UTC", "KITSUNE_DATA_REVISION": "d" * 40,
        "KITSUNE_REBUILD_TIMEOUT_MIN": "120", "KITSUNE_DPH": "0.8100", "KITSUNE_MACHINE_ID": "70001"}
    assert create[create.index("--disk") + 1] == "1400" and create[create.index("--label") + 1].startswith(
        "kitsune-full-p01-chain-data-smoke-")
    s = chain_launch.seen
    assert sorted(s["extent"]) == sorted([(fullrun.FULL_SELECTION, 120.0), (fullrun.SMOKE_SELECTION, 0.0)])
    assert s["hf"] == [fullrun.SMOKE_SELECTION, "labels/full/selections/study_1000h.parquet", fullrun.FULL_SELECTION]
    parts = {a[5]: (a, kw) for a, kw in s["preflight"]}
    assert list(parts) == ["full-smoke", "smoke-b", "p01"]
    assert all(kw["allow_fresh_over_done"] is False for _, kw in parts.values()), "each part: a fresh launch's guard"
    assert parts["smoke-b"][0][3] is None and parts["p01"][0][3] == SCRATCH, "the scratch repo only for timed parts"
    assert parts["smoke-b"][0][7]["selection"] == "labels/full/selections/study_1000h.parquet"
    assert all(not kw.get("resume") for _, kw in parts.values())
    assert len(s["chain"]) == 1 and "chain preflight ok" in out and "part p01: p01 preflight ok" in out
    assert "x ~25.2 h (box p01-chain; watchdog cap 35 h)" in out
    assert "chain p01-chain: the gate part must end by first boot + 9 h" in out and "+ 9.5 h" in out
    assert "free disk was re-checked just now: the offer search keeps only offers with disk_space >= 1400 GB" in out
    assert "disk_space>=1400" in search_query(fake).split(" "), "the offer search itself filters on the chain's disk"
    assert "chain stage 1 (configs/full/data-smoke.json): ~59 GB upstream down" in out


@pytest.mark.parametrize("args, err", [
    (["--config", "configs/full/data-smoke.json"], "--config is refused for chain box p01-chain"),
    (["--gate-hours", "0"], "--gate-hours 0 is refused for chain box p01-chain"),
    (["--max-hours", "29"], "--max-hours 29 is below chain p01-chain's floor 30"),
], ids=["config", "gate-off", "max-hours"])
def test_launch_chain_refusals(chain_launch, capsys, args, err):
    rc, fake = chain_launch([[offer(1, 70001, 0.81)]], "--scratch-repo", SCRATCH, *args, "--yes")
    out = capsys.readouterr().out
    assert rc == 1 and err in out and created(fake) is None, out


def test_launch_chain_warns_below_its_cap_and_a_parts_problem_refuses(chain_launch, capsys):
    rc, fake = chain_launch([[offer(1, 70001, 0.81)]], "--scratch-repo", SCRATCH, "--max-hours", "32", "--dry-run")
    out = capsys.readouterr().out
    assert rc == 0 and "WARNING: --max-hours 32 is below chain p01-chain's 35 h" in out
    assert env_of(created_or_printed(out))["KITSUNE_MAX_HOURS"] == "32"
    chain_launch.seen["part_problems"]["smoke-b"] = ["tools/whisper_eval.py (item whisper-large-v3) does not exist"]
    rc, fake = chain_launch([[offer(1, 70001, 0.81)]], "--scratch-repo", SCRATCH, "--yes")
    out = capsys.readouterr().out
    assert rc == 1 and "part smoke-b: tools/whisper_eval.py (item whisper-large-v3) does not exist" in out
    assert created(fake) is None


@pytest.mark.parametrize("flag", [["--resume"], ["--resume-reset", "full-p01-20260927T120000Z"],
                                  ["--resume-set", "full-p01-20260927T120000Z:schedule.epochs=5"]])
def test_a_chain_is_never_resumed_as_a_chain(chain_launch, capsys, flag):
    """E.8: launch --box p01-chain --resume exits 1 with what to run instead; nothing is searched or rented."""
    rc, fake = chain_launch([[offer(1, 70001, 0.81)]], "--scratch-repo", SCRATCH, *flag, "--yes")
    out = capsys.readouterr().out
    assert rc == 1 and "is not resumed as a chain" in out and "--box p01 --resume" in out and fake.calls == []


def test_a_resume_of_box_1_checks_the_chains_summary_and_its_live_instances(chain_launch, monkeypatch, capsys):
    """--box p01 --resume with p01-chain in the registry: E.8's summary check runs (its refusal refuses), and a live
    p01-chain instance is warned about as a live p01 one is."""
    got = []
    monkeypatch.setattr(launch, "chain_resume_checks", lambda out, box, chain: got.append((box, chain)) or (
        ["the newest p01 summary on the Hub is from another rental (container X)"], []))
    rc, fake = chain_launch([[offer(1, 70001, 0.81)]], "--scratch-repo", SCRATCH, "--resume", "--yes", box="p01",
                            instances=[{"id": 9, "label": "kitsune-full-p01-chain-data-smoke-abc",
                                        "actual_status": "running"}])
    out = capsys.readouterr().out
    assert rc == 1 and got == [("p01", CHAIN_BOX)] and "from another rental (container X)" in out
    assert "WARNING: a live instance of box p01: kitsune-full-p01-chain-data-smoke-abc (instance 9" in out
    monkeypatch.setattr(launch, "chain_resume_checks", lambda out, box, chain: ([], ["continues stage 2 of chain"]))
    rc, fake = chain_launch([[offer(1, 70001, 0.81)]], "--scratch-repo", SCRATCH, "--resume", "--dry-run", box="p01")
    assert rc == 0 and "continues stage 2 of chain" in capsys.readouterr().out


def chain_summaries(container="C1", started=100.0, queue_started=150.0, status="running", p01_container="C1",
                    p01_started=150.0) -> dict:
    return {fullrun.box_summary_path(CHAIN_BOX): {"kind": "chain", "box": CHAIN_BOX, "container_id": container,
                                                  "started": started, "gate": {"result": "pass", "time_utc": "T"},
                                                  "parts": {"p01": {"status": status,
                                                                    "queue_started": queue_started}}},
            fullrun.box_summary_path("p01"): {"kind": "full", "box": "p01", "container_id": p01_container,
                                              "started": p01_started, "items": {}}}


@pytest.mark.parametrize("case, problem, note", [
    ("match", None, "--box p01 --resume continues stage 2 of chain p01-chain (gate passed T); the chain's container "
                    "C1"),
    ("other-rental", "the newest p01 summary on the Hub is from another rental (container C0); chain p01-chain on "
                     "container C1 died before box 1 started: launch --box p01 fresh or the chain", None),
    ("p01-newer", None, None),
    ("no-chain", None, None),
])
def test_chain_resume_checks(monkeypatch, case, problem, note):
    runs = {"match": chain_summaries(),
            "other-rental": chain_summaries(status="pending", queue_started=None, p01_container="C0", p01_started=50.0),
            "p01-newer": chain_summaries(status="pending", p01_container="C9", p01_started=500.0),
            "no-chain": {k: v for k, v in chain_summaries().items() if "chain" not in k}}[case]
    hub = FullHub({}, runs)
    monkeypatch.setattr(launch, "_hub", lambda: (hub, hub.download))
    problems, notes = launch.chain_resume_checks(RUNS, "p01", CHAIN_BOX)
    if problem is None:
        assert problems == [], problems
    else:
        assert len(problems) == 1 and problem in problems[0], problems
    assert notes == ([note] if note else []), notes


def test_chain_preflight_checks_the_chains_configs_and_stage_files(repo, monkeypatch):
    write_reg(repo.root, with_chain(repo.reg))
    reg, reader, _, _ = launch.full_registry(SHA)
    have = {f: b"x" for f in fullrun.box_extra_files(CHAIN_BOX, reg, read_json=reader)}
    have["models/parakeet-tdt_ctc-0.6b-ja-hf/config.json"] = b"x"
    hub = FullHub(have)
    monkeypatch.setattr(launch, "_hub", lambda: (hub, hub.download))
    problems, notes = launch.chain_preflight(DATA, "d" * 40, SHA, CHAIN_BOX, reg)
    assert problems == [], problems
    assert any(n.startswith("chain p01-chain stage 1: parts full-smoke, smoke-b; rebuilds configs/full/data-smoke.json"
                            "; gate part full-smoke gates stage 2 (checks 1-11) by first boot + 9 h") for n in notes)
    del hub.data["labels/full/selections/study_1000h.parquet"]  # smoke-b's frozen selection, stage 1 pulls it
    problems, _ = launch.chain_preflight(DATA, "d" * 40, SHA, CHAIN_BOX, reg)
    assert problems == [f"{DATA}: no labels/full/selections/study_1000h.parquet (chain p01-chain pulls it in a stage)"]
    (repo.root / "configs/full/data-smoke-b.json").rename(repo.root / "moved.json")  # not committed at the sha
    problems, _ = launch.chain_preflight(DATA, "d" * 40, SHA, CHAIN_BOX, reg)
    assert any("configs/full/data-smoke-b.json does not exist at 0123456789ab" in p for p in problems), problems
    assert any("cannot check chain p01-chain's stage files" in p for p in problems), problems
    (repo.root / "moved.json").rename(repo.root / "configs/full/data-smoke-b.json")


def test_a_smoke_b_item_whose_cli_the_sha_lacks_is_refused_by_its_part_preflight(repo, monkeypatch, devslice):
    reg = launch.full_registry(SHA)[0]
    data = box_data("smoke-b", reg, repo.root)
    assert preflight(monkeypatch, FullHub(data), box="smoke-b", scratch=None)[0] == []
    (repo.root / "tools/whisper_eval.py").unlink()
    problems, _ = preflight(monkeypatch, FullHub(data), box="smoke-b", scratch=None)
    assert any("tools/whisper_eval.py (item whisper-large-v3) does not exist" in p for p in problems), problems


# --------------------------------------------------------------------------------------------------- finish


class ChainHub(FinishHub):
    """FinishHub (the run dir's verification passes) plus what a chain's --destroy reads: the infra commits kept by
    path (their listing), files downloaded and put by path. keep_old: a put that never takes (a Hub that keeps the
    old copy); slow_infra_s: every infra commit takes that long."""

    def __init__(self, run, remote=None, keep_old=False, slow_infra_s=0.0):
        super().__init__(run)
        self.remote, self.keep_old, self.slow = dict(remote or {}), keep_old, slow_infra_s
        self.order, self.files_put = [], []

    def create_commit(self, **kw):
        time.sleep(self.slow)
        super().create_commit(**kw)
        self.order.append("infra")
        for op in kw["operations"]:
            self.remote[op.path_in_repo] = op.path_or_fileobj

    def list_repo_tree(self, repo, path_in_repo=None, recursive=False, repo_type=None):
        if path_in_repo and "/infra/" in path_in_repo:
            self.order.append("listing")
            return [SimpleNamespace(path=p, size=len(v)) for p, v in self.remote.items()
                    if p.startswith(path_in_repo + "/")]
        return super().list_repo_tree(repo, path_in_repo, recursive, repo_type)

    def hf_hub_download(self, repo_id, filename, repo_type=None, local_dir=None):
        if filename not in self.remote:
            e = FileNotFoundError(f"404: {filename}")
            e.response = SimpleNamespace(status_code=404)  # huggingface_hub's EntryNotFoundError: not retried
            raise e
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(self.remote[filename])
        return str(p)

    def upload_file(self, path_or_fileobj=None, path_in_repo=None, repo_id=None, repo_type=None, commit_message=None):
        self.files_put.append(path_in_repo)
        self.order.append(f"put {path_in_repo}")
        if not self.keep_old:
            self.remote[path_in_repo] = Path(path_or_fileobj).read_bytes()


def chain_state_dir(state: Path) -> dict:
    """A chain box's state dir after its controller: chain.json, each part's records, the chain summary; returns the
    Hub copies the controller put (repo path -> bytes)."""
    files = {"chain/chain.json": {"box": CHAIN_BOX, "gate_box": "full-smoke", "gate": {"part": "full-smoke"}},
             "chain/full-smoke/queue.json": {"box": "full-smoke"}, "chain/full-smoke/events.jsonl": None,
             "chain/full-smoke/queue_summary.json": {"box": "full-smoke", "status": "complete"},
             "chain/full-smoke/smoke_verdict.json": {"box": "full-smoke", "overall": "pass"},
             "chain/full-smoke/logs/smoke-p03.log": None, "chain/smoke-b/queue_summary.json": {"box": "smoke-b"},
             "chain/smoke-b/smoke_verdict.json": {"box": "smoke-b"}, "chain/p01/queue_summary.json": {"box": "p01"},
             "chain/stage1/bootstrap_plan.json": {"record": "x"}, "queue_summary.json": {"kind": "chain"},
             "download_gate.json": {"verdict": "pass"}, "watchdog_alerts.jsonl": None, "events.jsonl": None}
    hub = {}
    for rel, doc in files.items():
        p = state / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(doc).encode() if doc is not None else b'{"kind": "x"}\n'
        p.write_bytes(data)
        if rel == "queue_summary.json":
            hub[fullrun.box_summary_path(CHAIN_BOX)] = data
        elif rel.startswith("chain/") and rel.endswith(("queue_summary.json", "smoke_verdict.json")):
            hub[f"full/box-{rel.split('/')[1]}/{Path(rel).name}"] = data
    return hub


@pytest.fixture
def chain_finish(full_finish, monkeypatch):
    monkeypatch.setenv("KITSUNE_BOX", CHAIN_BOX)
    monkeypatch.setenv("KITSUNE_CHAIN_STAGE", "2")
    monkeypatch.setattr(finish, "HUB_RETRY_WAITS", (0.01,))
    hub_copies = chain_state_dir(full_finish.state)

    def go(*args, **hub_kw):
        hub = ChainHub(full_run(full_finish.state.parent), dict(hub_copies), **hub_kw)
        monkeypatch.setattr(finish, "hf_api", lambda: hub)
        rc = finish.main([*args, "--repo", RUNS, "--runs-root", str(full_finish.state.parent / "runs")])
        return rc, hub
    go.hub_copies, go.state = hub_copies, full_finish.state
    return go


def test_a_chains_destroy_verifies_its_records_after_a_synchronous_infra_upload(chain_finish, monkeypatch):
    """E.7.2: the deep infra (chain/** included) goes up first, then every summary and verdict is compared with the
    Hub's (a differing one put once and compared again), then the infra listing: destroy."""
    actions = []
    monkeypatch.setattr(finish, "vast_rest", lambda action, timeout=30: actions.append(action) or True)
    chain_finish.hub_copies["full/box-smoke-b/queue_summary.json"] = b'{"box": "smoke-b", "old": true}'
    rc, hub = chain_finish("--destroy")
    assert rc == 0 and actions == ["destroy"]
    assert hub.files_put == ["full/box-smoke-b/queue_summary.json"]
    put = hub.order.index("put full/box-smoke-b/queue_summary.json")
    assert hub.order.index("infra") < put < hub.order.index("listing") and hub.order[-1] == "infra", hub.order
    infra = set(hub.commits[0])
    dest = f"full/box-{CHAIN_BOX}/infra/C77"
    for rel in ("chain/chain.json", "chain/full-smoke/events.jsonl", "chain/full-smoke/queue.json",
                "chain/full-smoke/logs/smoke-p03.log", "chain/stage1/bootstrap_plan.json", "events.jsonl",
                "download_gate.json", "watchdog_alerts.jsonl", "queue_summary.json"):
        assert f"{dest}/{rel}" in infra, rel
    ev = [json.loads(x) for x in (chain_finish.state / "events.jsonl").read_text().splitlines() if x.startswith("{")]
    assert [e["problems"] for e in ev if e.get("kind") == "chain_verify"] == [[]]


@pytest.mark.parametrize("hub_kw, why", [
    (dict(keep_old=True), "the Hub's copy still differs"),
    (dict(slow_infra_s=0.5), "did not finish within"),
], ids=["put-does-not-take", "slow-infra"])
def test_a_chain_whose_records_do_not_verify_is_stopped(chain_finish, monkeypatch, hub_kw, why):
    actions = []
    monkeypatch.setattr(finish, "vast_rest", lambda action, timeout=30: actions.append(action) or True)
    monkeypatch.setattr(finish, "CHAIN_INFRA_TIMEOUT_S", 0.2)
    monkeypatch.setattr(finish, "INFRA_TIMEOUT_S", 0.2)
    chain_finish.hub_copies["full/box-p01/queue_summary.json"] = b'{"box": "p01", "old": true}'
    rc, hub = chain_finish("--destroy", **hub_kw)
    assert rc == 2 and actions == ["stop"]
    halt = json.loads((chain_finish.state / "halt").read_text(encoding="utf-8"))
    assert halt["action"] == "stop" and halt["reason"].startswith("chain verification failed") and why in halt["reason"]


def seed_hub_reads(state: Path):
    """The hub_reads layout box 53693389 left (hf_hub_download(local_dir=...) of check 7's scratch pointer and of
    smoke-b's machine check), plus a re-armed copy; returns the JSON copies the infra must carry (E.7.2 item 1)."""
    rid = "smoke-p01-20261001T131836Z"
    keep = {f"chain/full-smoke/hub_reads/verdict/runs/{rid}/timed_state.json": b'{"step": 50}',
            "chain/smoke-b/hub_reads/full/box-full-smoke/queue_summary.json": b'{"machine_id": "m1"}'}
    cache = [f"chain/full-smoke/hub_reads/verdict/.cache/huggingface/{n}" for n in (
        ".gitignore", "CACHEDIR.TAG", f"download/runs/{rid}/timed_state.json.metadata",
        f"download/runs/{rid}/timed_state.json.lock")]
    cache += [f"chain/smoke-b/hub_reads/.cache/huggingface/{n}" for n in (
        ".gitignore", "CACHEDIR.TAG", "download/full/box-full-smoke/queue_summary.json.metadata")]
    cache += ["rearm-20261001T000000Z/chain/full-smoke/hub_reads/.cache/huggingface/.gitignore", ".cache/x/.gitignore"]
    for rel in [*keep, *cache]:
        (state / rel).parent.mkdir(parents=True, exist_ok=True)
        (state / rel).write_bytes(keep.get(rel, b"*\n"))
    return keep


@pytest.mark.parametrize("mode", ["--destroy", "--sync-only", "--stop", "--abort"])
def test_a_chains_infra_leaves_out_the_hub_download_caches(chain_finish, monkeypatch, mode):
    """Box 53693389 (2026-10-01): chain/<part>/hub_reads/ held huggingface_hub's .cache/huggingface/ bookkeeping, the
    first such path made CommitOperationAdd raise, the whole deep infra commit was lost, and --destroy stopped the box
    (the stop's own infra push failed the same way). The infra now leaves out only the .cache/.git paths: every mode
    commits the rest, the hub_reads JSON copies included, and --destroy verifies and destroys."""
    actions = []
    monkeypatch.setattr(finish, "vast_rest", lambda action, timeout=30: actions.append(action) or True)
    keep = seed_hub_reads(chain_finish.state)
    rc, hub = chain_finish(mode, *(["--reason", "x"] if mode in ("--stop", "--abort") else []))
    want_rc, want_actions = {"--destroy": (0, ["destroy"]), "--sync-only": (0, []), "--stop": (0, ["stop"]),
                             "--abort": (0, ["stop"])}[mode]  # --abort with a run dir on the disk: stop
    assert (rc, actions) == (want_rc, want_actions)
    sent = [p for c in hub.commits for p in c]
    assert sent and all(finish.hub_path_ok(p) for p in sent), [p for p in sent if not finish.hub_path_ok(p)]
    dest = f"full/box-{CHAIN_BOX}/infra/C77"
    for rel, data in keep.items():
        assert hub.sent[f"{dest}/{rel}"] == data, rel
    if mode == "--destroy":
        ev = [json.loads(x) for x in (chain_finish.state / "events.jsonl").read_text().splitlines()
              if x.startswith("{")]
        assert [e["problems"] for e in ev if e.get("kind") == "chain_verify"] == [[]]


def test_a_chain_infra_upload_that_raises_names_its_error(chain_finish, monkeypatch):
    """An infra commit that fails at once (a 4xx no retry fixes) stops the box with that error in the reason, not as a
    timeout ("did not finish within" stays the slow upload's wording)."""
    actions = []
    monkeypatch.setattr(finish, "vast_rest", lambda action, timeout=30: actions.append(action) or True)

    class Refused(ChainHub):
        def create_commit(self, **kw):
            e = RuntimeError("400 Bad Request: refused")
            e.response = SimpleNamespace(status_code=400)
            raise e

    hub = Refused(full_run(chain_finish.state.parent), dict(chain_finish.hub_copies))
    monkeypatch.setattr(finish, "hf_api", lambda: hub)
    rc = finish.main(["--destroy", "--repo", RUNS, "--runs-root", str(chain_finish.state.parent / "runs")])
    assert rc == 2 and actions == ["stop"]
    halt = json.loads((chain_finish.state / "halt").read_text(encoding="utf-8"))
    assert "the infra upload to full/box-p01-chain/infra/C77 failed (RuntimeError: 400 Bad Request" in halt["reason"]
    assert "did not finish within" not in halt["reason"]


def test_a_plain_full_box_and_a_chains_other_modes_are_unchanged(chain_finish, monkeypatch):
    actions = []
    monkeypatch.setattr(finish, "vast_rest", lambda action, timeout=30: actions.append(action) or True)
    rc, hub = chain_finish("--stop", "--reason", "x")
    assert rc == 0 and actions == ["stop"] and hub.files_put == [] and "listing" not in hub.order
    monkeypatch.delenv("KITSUNE_CHAIN_STAGE")
    rc, hub = chain_finish("--destroy")
    assert rc == 0 and actions[-1] == "destroy" and hub.files_put == [] and "listing" not in hub.order


# ------------------------------------------------------------------------------------------------- watchdog


def later(seconds: float, fn):
    import threading

    th = threading.Timer(seconds, fn)
    th.start()
    return th


def test_the_mode_file_switches_the_watchdog_from_alert_to_stop(tmp_path):
    """Stage 1 alerts on a stale heartbeat; once the controller writes "stop 1" the same stale heartbeat syncs and
    stops the box."""
    bash = need_bash()
    env, state, record = box_watchdog(tmp_path, "1", "alert", FAKE_SLEEP_MAX="20")
    hb = state / "train_hb"
    hb.write_text("", encoding="utf-8")
    t0 = int(time.time())
    os.utime(hb, (t0 + 1, t0 + 1))

    def write_mode():
        (state / "watchdog_mode.tmp").write_bytes(b"stop 1\n")  # as the controller writes it: LF
        (state / "watchdog_mode.tmp").replace(state / "watchdog_mode")

    th = later(4.5, write_mode)
    try:
        r = subprocess.run([bash, str(VAST / "watchdog.sh")], capture_output=True, text=True, env=env, timeout=90)
    finally:
        th.cancel()
    assert r.returncode == 0, r.stdout + r.stderr
    alerts = (state / "watchdog_alerts.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(alerts) == 1, "alert mode before the mode file"
    got = calls(record)
    assert [a for _, a in got] == ["--sync-only", "--stop --no-sync --reason watchdog: box controller heartbeat stale"]
    assert "mode file: stop 1 (was alert 1)" in r.stdout and got[0][0] >= t0 + 4


@pytest.mark.parametrize("content", [b"", b"banana x\n", b"stop\n", b"alert -3\n", b"\x00\x01garbage"],
                         ids=["empty", "garbage", "no-limit", "negative", "binary"])
def test_an_empty_or_malformed_mode_file_is_ignored(tmp_path, content):
    bash = need_bash()
    env, state, record = box_watchdog(tmp_path, "1", "alert", FAKE_SLEEP_MAX="5")
    (state / "watchdog_mode").write_bytes(content)
    hb = state / "train_hb"
    hb.write_text("", encoding="utf-8")
    t0 = int(time.time())
    os.utime(hb, (t0 + 1, t0 + 1))
    r = subprocess.run([bash, str(VAST / "watchdog.sh")], capture_output=True, text=True, env=env, timeout=90)
    assert r.returncode == 0, r.stdout + r.stderr
    assert calls(record) == [] and (state / "watchdog_alerts.jsonl").is_file(), "still in its env's alert mode"
    assert r.stdout.count("ignoring malformed") == (1 if content.strip() else 0), r.stdout


def test_a_stage_1_that_does_not_hand_over_is_stopped_whatever_its_heartbeat(tmp_path):
    """KITSUNE_WATCHDOG_HANDOVER_S: no mode file by first boot + that -> sync, then stop, with a fresh heartbeat (a
    hung controller or a stage-1 bootstrap's toucher can no longer hold the box); a mode file disarms it."""
    bash = need_bash()
    for moded in (False, True):
        d = tmp_path / str(moded)
        env, state, record = box_watchdog(d, "600", "alert", KITSUNE_WATCHDOG_HANDOVER_S="2", FAKE_SLEEP_MAX="8")
        (state / "first_boot").write_text(f"{int(time.time())}\n")
        if moded:
            (state / "watchdog_mode").write_bytes(b"stop 600\r\n")  # a CRLF line is read too
        hb = state / "train_hb"
        hb.write_text("", encoding="utf-8")
        toucher = subprocess.Popen([sys.executable, "-c", "import os, sys, time\nfor _ in range(60):\n"
                                    "    os.utime(sys.argv[1], None); time.sleep(0.2)", str(hb)])
        try:
            r = subprocess.run([bash, str(VAST / "watchdog.sh")], capture_output=True, text=True, env=env,
                               timeout=90)
        finally:
            toucher.kill()
        assert r.returncode == 0, r.stdout + r.stderr
        got = [a for _, a in calls(record)]
        if moded:
            assert got == [] and "did not hand over" not in r.stdout
        else:
            assert got == ["--sync-only", "--stop --no-sync --reason watchdog: chain stage 1 over its sub-deadline"]
            assert "chain stage 1 did not hand over by first boot + 2 s" in r.stdout


@pytest.mark.parametrize("case", ["fires", "no-self-stop", "old-halt", "not-full"])
def test_the_halt_retry(tmp_path, case):
    """A halt marker written during this container's life and older than HALT_RETRY_S (1 s here) with the instance
    still up: the stop is requested again (job full only; never with KITSUNE_NO_SELF_STOP=1, nor for a marker older
    than the watchdog)."""
    bash = need_bash()
    extra = dict(KITSUNE_JOB="full" if case != "not-full" else "study", KITSUNE_WATCHDOG_HALT_RETRY_S="1",
                 FAKE_SLEEP_MAX="6")
    env, state, record = box_watchdog(tmp_path, "0", "stop", **extra)
    if case == "no-self-stop":
        env["KITSUNE_NO_SELF_STOP"] = "1"  # after watchdog_env, which clears it
    halt = state / "halt"
    th = None
    if case == "old-halt":
        halt.write_text('{"action": "destroy"}\n')
        os.utime(halt, (time.time() - 3600, time.time() - 3600))
    else:
        th = later(1.0, lambda: halt.write_text('{"action": "destroy"}\n'))
    try:
        r = subprocess.run([bash, str(VAST / "watchdog.sh")], capture_output=True, text=True, env=env, timeout=90)
    finally:
        if th is not None:
            th.cancel()
    assert r.returncode == 0, r.stdout + r.stderr
    got = [a for _, a in calls(record)]
    if case == "fires":
        assert len(got) == 1 and got[0].startswith("--stop --no-sync --reason watchdog: halt marker ") and \
            got[0].endswith(" s old, instance still up"), got
    else:
        assert got == [], got


def test_the_watchdogs_dry_run_prints_the_chain_rules(tmp_path):
    bash = need_bash()
    env, state, _ = box_watchdog(tmp_path, "600", "alert", KITSUNE_WATCHDOG_HANDOVER_S="34200", KITSUNE_JOB="full")
    (state / "first_boot").write_text("1000000000\n")
    r = subprocess.run([bash, str(VAST / "watchdog.sh"), "--dry-run"], capture_output=True, text=True, env=env,
                       timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "chain stage 1: with no mode file by first boot + 34200 s (2001-09-09T11:16:40Z)" in r.stdout
    assert "halt retry: a halt marker written after this start and older than 1200 s" in r.stdout
    assert "mode file" in r.stdout and not (state / "deadline").exists()
