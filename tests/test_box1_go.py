"""tools/box1_go.py: box 1's go / no-go for box 2 (plan v3 section 3, contract 5's M4 < 0.1382) and the box-1
measurement make_full_configs --import-speed takes, on fixtures with the runs repo's layout (--hub-dir). CPU only, no
network: the Hub reader is replaced by a local folder.
"""
import json
import shutil
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import box1_go as G  # noqa: E402
import make_full_configs as M  # noqa: E402

from kitsune import fullrun  # noqa: E402

RUN = "full-p01-20261001T200000Z"
CID = "53711872"
SHA = "7715f3f622096505d4d3cf9872c31f835edfa5d2"
STEPS, SPS = 1000, 0.3
DEV_STEPS = list(range(0, STEPS + 1, 100))  # 11 dev evals: at the start, every 100 steps, the final one
MINI_STEPS = [250, 750]


def step_s(s: int) -> float:
    """time/step_s of step s: slow in the smoke phase, eval steps far slower, the steady ones around SPS."""
    if s in DEV_STEPS or s in MINI_STEPS:
        return 5.0
    if s <= 100:
        return 1.0
    return SPS + (s - 550) * 1e-5  # distinct values: any step wrongly kept or dropped moves the median


def steady_median(last: int) -> float:
    return statistics.median(step_s(s) for s in range(101, last + 1) if s not in DEV_STEPS and s not in MINI_STEPS)


def write(root: Path, rel: str, obj):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, list):  # JSONL
        p.write_text("".join(json.dumps(x) + "\n" for x in obj), encoding="utf-8")
    elif isinstance(obj, (bytes, str)):
        p.write_bytes(obj if isinstance(obj, bytes) else obj.encode("utf-8"))
    else:
        p.write_text(json.dumps(obj), encoding="utf-8")


def go_hub(root: Path, *, m4=0.125, status="complete", verified=True, dev_steps=DEV_STEPS, halt="destroy",
           early_stop=False, smoke_b_14=True, host_mem=None, wall_s=None) -> Path:
    """A runs repo where box 1 ended well (every line passes) unless the arguments say otherwise."""
    ended = status != "running"
    wall_s = wall_s if wall_s is not None else 837 + 4 / 0.1 * 8.5 + STEPS * SPS * 1.1  # o = 0.1 by the model
    res = dict(run_id=RUN, status="complete", steps=STEPS, epochs=4.0, end_reason="early_stop" if early_stop
               else "schedule", stopped_early=early_stop, resume_resets=0)
    items = {
        "stores-ctc": dict(kind="stores", status="done", verified=None, peak_rss_gb=36.8, result=None, run_dir=None),
        "full-p01": dict(kind="train", status="done" if ended else "running", verified=verified if ended else None,
                         peak_rss_gb=25.1, result=res if ended else None, run_dir=f"runs/{RUN}"),
        "m4-full-p01": dict(kind="readout", status="done" if ended else "pending", verified=verified if ended else None,
                            peak_rss_gb=4.2, run_dir=None,
                            result=dict(m4=m4, m4_ratio=1.34, m4_teacher=0.0933, jsut_cer_nostyle=0.071)
                            if ended else None)}
    write(root, fullrun.box_summary_path("p01"), dict(
        box="p01", container_id=CID, sha=SHA, machine_id="32513", status=status, rc=0 if ended else None,
        reason=None, started=1790878130.0, ended=1790950000.0 if ended else None, host_mem_peak_gb=host_mem,
        items=items, format=1, kind="full"))
    rd = f"runs/{RUN}"
    write(root, f"{rd}/config.json", {"config": {"run_name": "full-p01", "smoke": {"steps": 100}}})
    write(root, f"{rd}/summary.json", dict(status="complete" if ended else "running", run_id=RUN, steps=STEPS,
                                           epochs=4.0, elapsed_s_total=wall_s - 60, stopped_early=None))
    last = STEPS if ended else 600
    events = [dict(kind="eval_dev", step=s, at_step=s, epoch=s / 250, final=ended and s == STEPS,
                   dev_ce=2.0 - s / 1000, wall_s=8.5) for s in dev_steps if s <= last]
    events += [dict(kind="eval_mini", step=s, at_step=s) for s in MINI_STEPS if s <= last]
    if early_stop:
        events.append(dict(kind="early_stop", step=800, at_step=800, epoch=3.2, reason="patience", metric="dev_ce",
                           value=1.21, best=1.2, best_step=600, action="cooldown"))
    write(root, f"{rd}/events.jsonl", events)
    rows = [dict(step=s, tag="time/step_s", value=step_s(s)) for s in range(1, last + 1)]
    rows += [dict(step=s, tag="early_stop/value", value=2.0 - s / 1000) for s in dev_steps if 0 < s <= last]
    write(root, f"{rd}/metrics/scalars.jsonl", rows)
    infra = fullrun.box_infra_dir("p01", CID)
    if ended:
        write(root, f"{infra}/queue.json", {"items": {
            "stores-ctc": {"attempts": [{"t0": 1000.0, "t1": 1000.0 + 0.9 * 3600}]},
            "full-p01": {"attempts": [{"t0": 5000.0, "t1": 5000.0 + wall_s}]}}})
        write(root, f"{infra}/bootstrap_timings.jsonl", [
            {"phase": "download_gate", "seconds": 26.0, "end": 100.0 + 60},
            {"phase": "pull_labels", "seconds": 600.0, "end": 100.0 + 900},
            {"phase": "rebuild_audio", "seconds": 4 * 3600.0, "end": 100.0 + 2.9 * 3600},
            {"phase": "coverage", "seconds": 30.0, "end": 100.0 + 3.0 * 3600}])
        write(root, f"{infra}/first_boot", "100\n")
        if halt is not None:
            write(root, f"{infra}/halt", {"action": halt, "reason": "queue complete", "wall": 1790950100.0})
    checks = {n: {"pass": True} for n in ("12", "13", "14", "15", "16")}
    checks["14"]["pass"] = smoke_b_14
    write(root, fullrun.box_verdict_path("smoke-b"), dict(box="smoke-b", sha="ab" * 20, machine_id="149252",
                                                          time_utc="2026-10-01T23:00:00+00:00", checks=checks,
                                                          overall="pass" if smoke_b_14 else "fail"))
    return root


def run(tmp_path, capsys, **kw):
    n = len(list(tmp_path.glob("hub*")))
    hub = go_hub(tmp_path / f"hub{n}", **kw)
    out = tmp_path / "go.json"
    rc = G.main(["--hub-dir", str(hub), "--json", str(out)])
    printed = capsys.readouterr().out
    return rc, printed, json.loads(out.read_text(encoding="utf-8"))


def line(rec: dict, gid: str, status: str | None = None) -> list[str]:
    return [x["text"] for x in rec["lines"] if x["id"] == gid and (status is None or x["status"] == status)]


def test_go(tmp_path, capsys):
    rc, printed, rec = run(tmp_path, capsys)
    assert rc == G.EXIT_GO and rec["go"] is True and rec["exit"] == 0, printed
    assert [x["status"] for x in rec["lines"] if x["status"] != "INFO"] == ["PASS"] * 5, printed
    assert line(rec, "G2", "PASS") == ["M4 0.1250 < 0.1382 (P-0.1B's study value); m4_ratio 1.34, m4_teacher 0.0933, "
                                       "jsut_cer_nostyle 0.071"]
    assert "GO: box 2 may launch" in printed and "vastai show instances" in line(rec, "G1", "PASS")[0]
    assert "11 dev evals, dev_ce first 2.0000 best 1.0000 last 1.0000; 10 early-stop checks" in line(rec, "G3")[0]
    b = rec["box1"]
    smoke = M.load_speed()["smoke"]["sec_per_step"]["p01"]
    assert b["sec_per_step"] == steady_median(STEPS) == pytest.approx(SPS, abs=1e-4)
    assert b["smoke_sec_per_step"] == smoke
    assert (b["sha"], b["machine_id"], b["container_id"], b["run_id"], b["steps"], b["m4"]) == (
        SHA, "32513", CID, RUN, STEPS, 0.125)
    # o = (wall - 837 - 40 x 8.5) / (1000 x s/step) - 1, the wall built for o 0.1 at 0.3 s
    assert b["overhead"] == round(STEPS * SPS * 1.1 / (STEPS * b["sec_per_step"]) - 1, 4)
    assert (b["stores_ctc_h"], b["bootstrap_h"], b["rebuild_h"]) == (0.9, 3.0, 4.0)
    assert b["peak_rss_gb"] == {"stores-ctc": 36.8, "full-p01": 25.1, "m4-full-p01": 4.2}
    assert f"(smoke A {smoke}: r {b['sec_per_step'] / smoke:.3f})" in line(rec, "G4", "PASS")[0]
    # the projection of the HOURS_BOXES - full-t, p005 and p-cool (its three cooldown re-runs, DECISIONS H14) - with
    # box 1's measurement, and what boxes.json would change; boxes full-p and p01 are records, never projected
    r = round(b["sec_per_step"] / smoke, 4)
    for box, run_ in (("full-t", "full-t06"), ("p005", "full-p005"), ("p-cool", "full-p03")):
        assert f"box {box} with box 1's speed (r {r:g}, o {b['overhead']:g}): {run_} " in printed, box
    assert "full-p01 " in printed.split("box p-cool with box 1's speed", 1)[1].splitlines()[0]
    assert "box full-p with" not in printed and "box p01 with" not in printed
    assert "P-0.1B's 4 epochs by the same model " in printed and "vs measured" in printed
    assert "boxes.json would change: boxes.full-t.items.full-t06: {'max_hours': 63.62}" in printed


def test_the_json_round_trips_through_import_speed(tmp_path, capsys):
    rc, _, rec = run(tmp_path, capsys)
    assert rc == 0
    out = tmp_path / "configs" / "full"
    shutil.copytree(ROOT / "configs" / "full", out)
    M.import_speed(None, tmp_path / "go.json", out)
    speed = M.load_speed(out)
    assert speed["box1"] == rec["box1"] and speed["smoke"] == M.load_speed()["smoke"]
    for box in M.HOURS_BOXES:
        h = M.box_hours(M.load_plan(), speed, box)
        assert h["r"] == round(rec["box1"]["sec_per_step"] / speed["smoke"]["sec_per_step"]["p01"], 4)
        assert h["o"] == rec["box1"]["overhead"]
        assert h["store_h"] == (0.9, round(0.9 * M.STORES_PESS, 2)) and h["setup_h"] == (1.6, 3.3)
        assert len(h["continuations"]) == (3 if box == "p-cool" else 0), box  # DECISIONS H14


@pytest.mark.parametrize("m4", [0.1390, 0.1382])
def test_an_m4_not_below_the_study_value_is_no_go(tmp_path, capsys, m4):
    rc, printed, rec = run(tmp_path, capsys, m4=m4)
    assert rc == G.EXIT_NOGO and rec["go"] is False and "NO-GO" in printed
    assert line(rec, "G2", "FAIL") == [f"M4 {m4:.4f} >= 0.1382 (P-0.1B's study value); m4_ratio 1.34, m4_teacher "
                                       f"0.0933, jsut_cer_nostyle 0.071"]


def test_unverified_uploads_or_a_stopped_box_are_no_go(tmp_path, capsys):
    rc, _, rec = run(tmp_path, capsys, verified=False)
    assert rc == G.EXIT_NOGO and line(rec, "G1", "FAIL") == ["full-p01 done verified False; m4-full-p01 done "
                                                              "verified False"]
    assert rec["box1"] is None  # no measurement to import from a box that did not end well
    rc, _, rec = run(tmp_path, capsys, halt="stop")
    assert rc == G.EXIT_NOGO and "finish ended the box with 'stop'" in line(rec, "G1", "FAIL")[0]
    rc, _, rec = run(tmp_path, capsys, halt=None)
    assert rc == G.EXIT_WAIT and "no full/box-p01/infra/53711872/halt yet" in line(rec, "G1", "WAIT")[0]


def test_a_running_box_is_not_decidable_and_projects_its_run(tmp_path, capsys):
    rc, printed, rec = run(tmp_path, capsys, status="running")
    assert rc == G.EXIT_WAIT and rec["go"] is None and rec["box1"] is None and "NOT DECIDABLE YET" in printed
    assert line(rec, "G1", "WAIT") == ["box 1 is running (stores-ctc done, full-p01 running, m4-full-p01 pending)"]
    # the scalars so far: steady s/step and plan v3's run model at it
    (partial,) = [t for t in line(rec, "G4", "INFO") if t.startswith("partial:")]
    total = M.plan_total_steps("p01", M.load_plan())
    proj = (total * steady_median(600) * 1.08 + 837 + 40 * 8.5) / 3600
    # its registry need: box p01's full-p01 now (the record of H9's 8-epoch plan again, DECISIONS H14)
    assert partial.startswith(f"partial: step 600 of {total}; at this speed full-p01 runs ~{proj:.2f} h at o 0.08 "
                              f"(its registry need 11.47 h)"), partial


def test_too_few_dev_evals_are_no_go(tmp_path, capsys):
    rc, _, rec = run(tmp_path, capsys, dev_steps=[0, 400, 800, 1000])
    assert rc == G.EXIT_NOGO and "4 dev evals < 9" in line(rec, "G3", "FAIL")[0]


def test_the_early_stop_is_printed(tmp_path, capsys):
    rc, printed, rec = run(tmp_path, capsys, early_stop=True)
    assert rc == G.EXIT_GO
    assert line(rec, "G3", "INFO") == ["early_stop at step 800, epoch 3.2: patience (dev_ce 1.21, best 1.2 at step "
                                       "600), action cooldown"]
    assert "end_reason early_stop, stopped_early False" in line(rec, "G3", "PASS")[0]


def test_smoke_b_must_pass_checks_12_to_16(tmp_path, capsys):
    rc, _, rec = run(tmp_path, capsys, smoke_b_14=False)
    assert rc == G.EXIT_NOGO
    assert line(rec, "G5", "FAIL") == ["smoke-b at abababababab (machine 149252, 2026-10-01T23:00:00+00:00): overall "
                                       "fail, check 12 True, check 13 True, check 14 False, check 15 True, check 16 "
                                       "True"]


def test_the_steady_rule_is_the_smoke_verdicts(tmp_path):
    """box1_go's steady s/step is kitsune.full_queue.SmokeVerdict.sec_per_step on the same run dir: the steps after
    smoke.steps that ran no eval, median."""
    from kitsune import full_queue

    hub = go_hub(tmp_path / "hub")
    q = SimpleNamespace(state={"items": {"full-p01": {"run_dir": f"runs/{RUN}"}}}, root=hub)
    want = full_queue.SmokeVerdict(q).sec_per_step("full-p01")
    reader = G.DirReader(hub)
    got, n = G.steady_sec_per_step(G.read_jsonl(reader, f"runs/{RUN}/metrics/scalars.jsonl"),
                                   G.read_jsonl(reader, f"runs/{RUN}/events.jsonl"), 100)
    assert got == want and n == STEPS - 100 - 9 - 2  # dev evals after step 100 and the two minis excluded
    # without the exclusions the slow eval steps would move it
    assert got != G.steady_sec_per_step(G.read_jsonl(reader, f"runs/{RUN}/metrics/scalars.jsonl"), [], 0)[0]


def test_the_ram_falls_back_to_the_cgroup_column(tmp_path, capsys):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    hub = go_hub(tmp_path / "hub")
    p = hub / f"runs/{RUN}/metrics/steps.parquet"
    pq.write_table(pa.table({"step": [1, 2, 3], "sys/cgroup/anon_gb": [10.0, 41.5, None]}), p)
    rc = G.main(["--hub-dir", str(hub), "--json", str(tmp_path / "go.json")])
    rec = json.loads((tmp_path / "go.json").read_text(encoding="utf-8"))
    assert rc == 0 and rec["box1"]["host_mem_peak_gb"] == 41.5
    assert any("RAM: host peak 41.5 GB (steps.parquet sys/cgroup/anon_gb; no host peak on vast)" in t
               for t in line(rec, "G4", "INFO"))
    capsys.readouterr()
    hub = go_hub(tmp_path / "hub2", host_mem=88.0)  # a readable host peak wins
    G.main(["--hub-dir", str(hub), "--json", str(tmp_path / "go2.json")])
    assert json.loads((tmp_path / "go2.json").read_text(encoding="utf-8"))["box1"]["host_mem_peak_gb"] == 88.0


def test_an_unreadable_hub_is_not_decidable(tmp_path, capsys, monkeypatch):
    class Broken:
        def __init__(self, *a, **kw):
            pass

        def get(self, path):
            raise G.HubUnreadable(f"{path}: ConnectionError")

    monkeypatch.setattr(G, "HubReader", Broken)
    rc = G.main(["--json", str(tmp_path / "go.json")])
    assert rc == G.EXIT_HUB == 3 and "NOT DECIDABLE (Hub unreachable)" in capsys.readouterr().out
    assert json.loads((tmp_path / "go.json").read_text(encoding="utf-8")) == {"go": None, "exit": 3, "lines": [],
                                                                               "box1": None}
    # nothing of box 1 on the Hub yet: not decidable either
    (tmp_path / "empty").mkdir()
    assert G.main(["--hub-dir", str(tmp_path / "empty")]) == G.EXIT_WAIT


class FakeHfHub:
    """huggingface_hub's HfApi.repo_info and hf_hub_download over a local folder at one head commit: a missing file is
    the Hub's 404 (RemoteEntryNotFoundError); `down` makes every download fail as a dropped network does (no cached
    copy: LocalEntryNotFoundError, an EntryNotFoundError too); `head_down` makes repo_info fail."""

    def __init__(self, root: Path, sha: str = "c0ffee" * 6 + "abcd"):
        self.root, self.sha, self.down, self.head_down, self.calls = root, sha, set(), False, []

    def install(self, monkeypatch):
        import huggingface_hub

        fake = self

        class Api:
            def repo_info(self, repo):
                fake.calls.append(("repo_info", repo))
                if fake.head_down:
                    raise ConnectionError("Hub unreachable")
                return SimpleNamespace(sha=fake.sha)

        monkeypatch.setattr(huggingface_hub, "HfApi", Api)
        monkeypatch.setattr(huggingface_hub, "hf_hub_download", self.download)

    def download(self, repo, path, revision=None, cache_dir=None):
        import httpx
        from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError

        self.calls.append(("download", path, revision))
        if path in self.down or "*" in self.down:
            raise LocalEntryNotFoundError("cannot reach the Hub and the file is not in the cache")
        p = self.root / path
        if not p.is_file():
            raise RemoteEntryNotFoundError(f"404: {path}", response=httpx.Response(
                404, request=httpx.Request("GET", f"https://huggingface.co/{repo}/resolve/{revision}/{path}")))
        return str(p)


def test_the_hub_reader_tells_a_missing_file_from_an_unreachable_hub(tmp_path, capsys, monkeypatch):
    """A file the Hub says is not there is None (G5: run smoke-B, a NO-GO); a download that could not ask the Hub
    (LocalEntryNotFoundError, which huggingface_hub 1.x derives from EntryNotFoundError) or a head commit it could not
    resolve is HubUnreadable: exit EXIT_HUB, never a false NO-GO. Every file is read at the head commit."""
    hub = go_hub(tmp_path / "hub")
    fake = FakeHfHub(hub)
    fake.install(monkeypatch)
    r = G.HubReader()
    assert r.get(fullrun.box_summary_path("p01")) is not None and r.get("full/box-p01/nothing.json") is None
    assert [c for c in fake.calls if c[0] == "repo_info"] == [("repo_info", G.RUNS_REPO)]  # resolved once
    assert {c[2] for c in fake.calls if c[0] == "download"} == {fake.sha}
    assert G.main(["--json", str(tmp_path / "go.json")]) == G.EXIT_GO
    capsys.readouterr()
    fake.down = {fullrun.box_verdict_path("smoke-b")}  # G1-G4 read fine, smoke-B's verdict cannot be fetched
    rc = G.main(["--json", str(tmp_path / "go.json")])
    out = capsys.readouterr().out
    assert rc == G.EXIT_HUB and "NOT DECIDABLE (Hub unreachable)" in out and "NO-GO" not in out.replace(
        "nothing here is a NO-GO", "")
    assert json.loads((tmp_path / "go.json").read_text(encoding="utf-8"))["go"] is None
    fake.down = set()
    (hub / fullrun.box_verdict_path("smoke-b")).unlink()  # the Hub's own answer: no verdict, a NO-GO (G5)
    assert G.main([]) == G.EXIT_NOGO
    capsys.readouterr()
    fake.head_down = True
    assert G.main([]) == G.EXIT_HUB and "Hub unreachable" in capsys.readouterr().out


def test_revision_pins_box_1s_reads_and_never_smoke_bs(tmp_path, capsys, monkeypatch):
    """DECISIONS G4: --revision REV reads G1-G4 at that runs-repo commit (P-0.1B's continuation overwrites box p01's
    records at the head) without resolving the head for them, and records it as box1.revision; G5, smoke-B's verdict,
    still reads the head (a pinned G5 would hide a newer smoke-B)."""
    rev = "c4604304db76e068df7bbe39d00d006b74d6c134"
    hub = go_hub(tmp_path / "hub")
    fake = FakeHfHub(hub)
    fake.install(monkeypatch)
    out = tmp_path / "go.json"
    assert G.main(["--revision", rev, "--json", str(out)]) == G.EXIT_GO, capsys.readouterr().out
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["box1"]["revision"] == rev
    dl = [c for c in fake.calls if c[0] == "download"]
    verdict = fullrun.box_verdict_path("smoke-b")
    assert {c[2] for c in dl if c[1] != verdict} == {rev}
    assert {c[2] for c in dl if c[1] == verdict} == {fake.sha}
    assert fake.calls.count(("repo_info", G.RUNS_REPO)) == 1  # the head, for G5 only
    # without --revision the head is recorded
    fake.calls.clear()
    assert G.main(["--json", str(out)]) == G.EXIT_GO
    assert json.loads(out.read_text(encoding="utf-8"))["box1"]["revision"] == fake.sha
    capsys.readouterr()
    for bad in (["--revision", "c4604304"], ["--revision", rev, "--hub-dir", str(hub)]):
        with pytest.raises(SystemExit):
            G.main(bad)
