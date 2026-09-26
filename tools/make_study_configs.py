"""Generate the size study's trainer configs, configs/study/<name>.json, from one base (STUDY.md 2.2-2.5, 6.1).

Every config is three layers, merged in this order:
  1. the trainer part of configs/next_run_template.json (its data keys and its _comment left out);
  2. the data block, study/data.json verbatim (the selection, both label roots with pull_parakeet, the extent, the
     sources and eval sets, the selection recipe with its study block);
  3. the study's settings: the steps clock with cooldown_frac 0.2, early stop off, L2-SP 0, the complete evals at
     eval.full_at_fracs [0.2, 0.4, 0.6, 0.8] (eval.every_min null, no per-epoch evals), the lean checkpoints (weights
     at 0.4 and at the end, a local full state at 0.4 for the T/2 branch, and at 0.8 uploaded for the runs of at most
     0.1B), log syncs every 20 min, the runs repo as hf.output_repo; and per run what kitsune.prereg.rules()["runs"]
     fixes: family, student, seed, the warm-up, the weight decay, the aux-CTC weight, the planned micro-batch and the
     BatchNorm mode (frozen for the pruned students, train for the ones built from scratch, bridge included).
The LR probe grids, lengths and classes are read from kitsune.prereg.rules()["lr_probes"] and the calibration window
from kitsune.prereg.CALIB_STEPS, never written here: a change to the rules changes the generated files, and --check
says so.

Names (CONTRACT.md section 1):
  <run>.json                 the 9 study runs. schedule.max_steps, optim.lr and eval.mini.every_steps are null: the
                             study box fills them from its PREREG_numbers file at queue time (kitsune/study_queue.py,
                             `--set`), and scripts/04_distill.py refuses the unfilled file (max_steps and lr)
  <run>-half.json            its T/2 branch: the same config with run_name <run>-half and branch.parent a placeholder
                             the queue replaces with the parent's local run dir (unreplaced, the trainer stops at once:
                             it is no run dir)
  probe-<class>-<lr>.json    every LR probe of every class, on the class's probed_on run: lr_probe on, the probe's
                             steps, warm-up and cooldown (cooldown_frac = cooldown / steps), local full states only
  calib-<run>.json           the calibration run of every calibrated run: lr_probe + calibrate on (metrics only; the
                             step-time table over the pre-registered window), the planned micro-batch with the memory
                             probe, no checkpoints; max_steps CALIB_MAX_STEPS is a cap, the box ends a calibration group
                             with the STOP file once every run of it has its window
  shake-*.json               the shakedown box (STUDY.md 6.1, both families: every run of boxes A and B): per run a
                             100-step smoke at the planned micro with the run's warm-up (a calibration over steps 50-100,
                             which the box ends after the smoke; the loss-trend gate on for the pruned runs, a note for
                             the scratch ones), and per family (AED; CTC with the suffix -ctc) a crash-and-resume run,
                             a 50-step toy parent and its T/2 branch, a complete eval on a subset with an in-loop
                             complete eval, a mini eval and a lean upload
  anchor-b20.json            not a training config: scripts/05_evaluate.py reads its data and eval keys to re-score the
                             first run's 0.6B (decision 29) on the study's eval rows; its schedule and LR are inert
CTC configs (the Parakeet students) carry family "ctc", parakeet_root and loss.w_ctc, which scripts/04_distill.py knows
only once the CTC trainer (WP4b) is merged: until then they are written but not loadable.

Usage:
  python tools/make_study_configs.py                 # write configs/study/*.json (and remove stale ones)
  python tools/make_study_configs.py --check         # exit 1 if the committed files differ from the generator's
  python tools/make_study_configs.py --out-repo Multy123/kitsune-runs
"""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kitsune import prereg  # noqa: E402

OUT_DIR = ROOT / "configs" / "study"
TEMPLATE = ROOT / "configs" / "next_run_template.json"
DATA = ROOT / "study" / "data.json"
RUNS_REPO = "Multy123/kitsune-runs"
# the data keys the template carries for the first run's data; the study takes them from study/data.json
TEMPLATE_DATA_KEYS = ("student", "data_root", "teacher_root", "second_root", "selection", "sources", "eval_sets",
                      "selection_recipe")
SMALL_PARAMS = 110_000_000  # "the runs of at most 0.1B": T-0.1B (104.0M), the replicate, T-0.05B, P-0.1B, P-0.05B
EVAL_FRACS = [0.2, 0.4, 0.6, 0.8]  # STUDY.md 4.2: complete evals at 20/40/60/80 % of max_steps (and the final one)
FULL_FRACS, FULL_FRACS_SMALL, WEIGHTS_FRACS = [0.4], [0.4, 0.8], [0.4]
SYNC_EVERY_MIN = 20  # STUDY.md 5.5: nine writers share the runs repo
CALIB_MAX_STEPS = 20000  # a cap: the box stops a calibration group once every run of it has its window
BRANCH_PARENT = "<the parent's local run dir: set by the study box (kitsune/study_queue.py)>"
SHAKE_SMOKE_STEPS = 100  # STUDY.md 6.1: a 100-step smoke per config, its step time over steps 50-100
SHAKE_CTC_SUFFIX = "-ctc"  # the CTC family's shakedown items: shake-resume-ctc, shake-parent-ctc(-half), shake-eval-ctc
ANCHOR = "anchor-b20"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(out.get(k), dict) and isinstance(v, dict) else copy.deepcopy(v)
    return out


def _load(path: Path) -> dict:
    return {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if not k.startswith("_")}


def base_config(out_repo: str = RUNS_REPO) -> dict:
    """Layers 1-2 and the study settings every config shares (layer 3 without the per-run part)."""
    cfg = {k: v for k, v in _load(TEMPLATE).items() if k not in TEMPLATE_DATA_KEYS}
    cfg = _merge(cfg, _load(DATA))
    return _merge(cfg, {
        "subset": {"train_utts": None, "eval_utts_per_set": None, "train_audio_s": None, "eval_audio_s": None},
        "loss": {"l2sp_lambda": 0.0},
        "schedule": {"clock": "steps", "epochs": None, "max_steps": None, "cooldown_frac": 0.2},
        "optim": {"lr": None},
        "eval": {"every_min": None, "every_steps": None, "every_epochs": None, "full_every_epochs": None,
                 "full_at_fracs": list(EVAL_FRACS), "mini": {"every_steps": None}},
        "ckpt": {"weights_every_min": None, "weights_every_steps": None, "full_every_steps": None,
                 "full_at_fracs": list(FULL_FRACS), "weights_at_fracs": list(WEIGHTS_FRACS), "upload_full_at": []},
        "log": {"sync_every_min": SYNC_EVERY_MIN},
        "hf": {"output_repo": out_repo},
        "early_stop": {"enabled": False},
    })


def rules() -> dict:
    return prereg.rules()


def run_config(run: str, r: dict | None = None, out_repo: str = RUNS_REPO) -> dict:
    """The study run `run` (a key of rules()["runs"]) with its placeholders (max_steps, lr, mini cadence) null."""
    r = r or rules()
    spec = r["runs"][run]
    scratch = spec["init_class"] == "scratch"
    small = int(spec["params_total"]) <= SMALL_PARAMS
    over = {
        "run_name": run, "student": spec["student"], "seed": int(spec["seed"]),
        "schedule": {"warmup_steps": int(spec["warmup_steps"])},
        "optim": {"weight_decay": float(spec["weight_decay"])},
        "loss": {"aux_ctc_weight": float(spec["aux_ctc"])},
        "bn": {"mode": "train" if scratch else "frozen"},
        "batch": {"micro_audio_s": spec["micro_audio_s"]},
        "ckpt": {"full_at_fracs": list(FULL_FRACS_SMALL if small else FULL_FRACS),
                 "upload_full_at": ["frac:0.8"] if small else []},
    }
    if spec["family"] == "ctc":  # the CTC trainer's keys (WP4b; CONTRACT.md section 2)
        over.update(family="ctc", loss=dict(over["loss"], w_ctc=float(r["training"]["ctc_loss"]["w_ctc"])))
    return _merge(base_config(out_repo), over)


def branch_config(run: str, r: dict | None = None, out_repo: str = RUNS_REPO) -> dict:
    """<run>-half: the run's config with its own run name and the branch block (resume and end fractions from the
    rules); the parent's run dir is the queue's to set."""
    r = r or rules()
    br = r["branch"]
    return _merge(run_config(run, r, out_repo), {
        "run_name": f"{run}-half",
        "branch": {"parent": BRANCH_PARENT, "resume_frac": float(br["resume_frac"]), "end_frac": float(br["end_frac"])}})


def _metrics_only(cfg: dict) -> dict:
    """What an LR probe and a calibration run leave out (lr_probe: no evals, weights or checkpoint uploads anyway):
    the fraction schedules, the mini evals, the smoke's loss-trend gate (a high LR or a random init may not fall in
    100 steps; these runs measure, they are no health gate) and the smoke profiler."""
    return _merge(cfg, {"lr_probe": {"enabled": True},
                        "eval": {"full_at_fracs": None, "mini": {"every_steps": None}},
                        "ckpt": {"full_at_fracs": None, "weights_at_fracs": None, "upload_full_at": []},
                        "smoke": {"require_loss_decrease": False}, "perf": {"profile_smoke": False}})


def probe_config(cls: str, lr: float, r: dict | None = None, out_repo: str = RUNS_REPO) -> dict:
    """probe-<cls>-<lr>: the class's probed_on run with the probe's LR and schedule (kitsune.prereg rules), metrics
    only, local full states for a resume."""
    r = r or rules()
    p = r["lr_probes"]["classes"][cls]
    steps = int(p["max_steps"])
    if int(p["warmup"]) + int(p["stable"]) + int(p["cooldown"]) != steps:
        raise ValueError(f"probe class {cls}: warm-up + stable + cooldown != max_steps in the rules")
    cfg = run_config(p["probed_on"], r, out_repo)
    return _metrics_only(_merge(cfg, {
        "run_name": prereg.probe_run_name(cls, lr), "optim": {"lr": float(lr)},
        "schedule": {"max_steps": steps, "warmup_steps": int(p["warmup"]),
                     "cooldown_frac": int(p["cooldown"]) / steps}}))


def calib_config(run: str, r: dict | None = None, out_repo: str = RUNS_REPO) -> dict:
    """calib-<run>: the run at its planned micro-batch (the memory probe confirms or halves it) for up to
    CALIB_MAX_STEPS steps, ended by the box; the step-time table over the pre-registered window. The LR is the run's
    class's lowest grid point (the step time does not depend on it; the lowest keeps a random init finite)."""
    r = r or rules()
    spec = r["runs"][run]
    lr = min(float(x) for x in r["lr_probes"]["classes"][spec["lr_from"]]["grid"])
    cfg = run_config(run, r, out_repo)
    return _metrics_only(_merge(cfg, {
        "run_name": f"calib-{run}", "optim": {"lr": lr}, "schedule": {"max_steps": CALIB_MAX_STEPS},
        "calibrate": {"enabled": True, "window": [int(x) for x in prereg.CALIB_STEPS]},
        "ckpt": {"full_local_every_min": None, "full_after_smoke": False}}))


def _toy(cfg: dict, name: str, steps: int, warmup: int, lr: float, **over) -> dict:
    """A shakedown run: `steps` optimizer steps with a short warm-up, cheap evals on 50 rows per eval set, and no full
    state after the smoke (its step can be a fraction's or a periodic one's)."""
    return _merge(cfg, _merge({"run_name": name, "optim": {"lr": float(lr)},
                               "schedule": {"max_steps": steps, "warmup_steps": warmup},
                               "subset": {"eval_utts_per_set": 50}, "perf": {"profile_smoke": False},
                               "ckpt": {"full_after_smoke": False}}, over))


def shake_configs(box_runs: list[str], r: dict | None = None, out_repo: str = RUNS_REPO) -> dict[str, dict]:
    """The shakedown box's configs (STUDY.md 6.1) for the runs of both study boxes (study_queue.shakedown_runs):
      shake-smoke-<run>   the memory probe at the planned micro and a 100-step smoke: the run's student, seed, data
                          order and warm-up, at its class's lowest grid LR (the slowest start any chosen LR gives), as a
                          calibration over steps 50-100 capped by CALIB_MAX_STEPS; the box ends it with the STOP file
                          once its smoke checks are logged (a scratch run's 2,000-step warm-up does not fit a 100-step
                          schedule, and the smoke must see the start the main sees). The smoke gate (finite losses,
                          the throughput floor, the undecodable rows) is the study run's own; its loss-trend check is
                          on for a pruned run (warm-up <= 1,000: a flat start there is a defect, and fails the
                          shakedown) and off for a scratch run, whose flat start 100 steps into a 2,000-step warm-up
                          the queue reports as a note (the study box's smoke_gate_guard handles it)
    and per family (AED: no suffix, CTC: SHAKE_CTC_SUFFIX) of the runs:
      shake-resume        the family's smallest run for 60 steps as an LR probe with a full state every 20 steps; the
                          box crashes it at step 45 (KITSUNE_CRASH_AT_STEP) and resumes it from step 40, and its end
                          phase scores the complete gate sets teacher-forced (lr_probe_eval)
      shake-parent / shake-parent-half   a 50-step toy run with the full state at 0.4 and its T/2 branch (steps 21-25,
                          a final complete eval on the subset), on the family's smallest scratch run (BN train mode,
                          the aux-CTC head), else its smallest
      shake-eval          the family's largest run for 20 steps: a complete eval at step 10 (eval.full_at_fracs) and
                          the final one on the subset, mini evals every 5 steps, weights at 0.5 and at the end; the box
                          uploads and verifies its run dir (lean) like every other"""
    r = r or rules()
    runs = r["runs"]
    out = {}
    lr_of = {run: min(float(x) for x in r["lr_probes"]["classes"][runs[run]["lr_from"]]["grid"]) for run in box_runs}
    for run in box_runs:
        c = calib_config(run, r, out_repo)
        gate = bool(run_config(run, r, out_repo)["smoke"]["require_loss_decrease"]) and             runs[run]["init_class"] != "scratch"
        out[f"shake-smoke-{run}"] = _merge(c, {"run_name": f"shake-smoke-{run}",
                                               "smoke": {"steps": SHAKE_SMOKE_STEPS, "require_loss_decrease": gate},
                                               "calibrate": {"window": [50, SHAKE_SMOKE_STEPS]}})
    for fam in dict.fromkeys(runs[x]["family"] for x in box_runs):
        sfx = SHAKE_CTC_SUFFIX if fam == "ctc" else ""
        by_size = sorted((x for x in box_runs if runs[x]["family"] == fam), key=lambda x: int(runs[x]["params_total"]))
        smallest, largest = by_size[0], by_size[-1]
        scratch = next((x for x in by_size if runs[x]["init_class"] == "scratch"), smallest)
        base = run_config(smallest, r, out_repo)
        out[f"shake-resume{sfx}"] = _metrics_only(_toy(
            base, f"shake-resume{sfx}", 60, 5, lr_of[smallest], smoke={"steps": 20},
            subset={"eval_utts_per_set": None},  # lr_probe scores the complete sets
            ckpt={"full_every_steps": 20, "keep_local": 5}))
        parent = _toy(run_config(scratch, r, out_repo), f"shake-parent{sfx}", 50, 5, lr_of[scratch],
                      smoke={"steps": 10}, eval={"full_at_fracs": None, "mini": {"every_steps": None}},
                      ckpt={"full_at_fracs": [0.4], "weights_at_fracs": None, "upload_full_at": []})
        out[f"shake-parent{sfx}"] = parent
        out[f"shake-parent{sfx}-half"] = _merge(parent, {"run_name": f"shake-parent{sfx}-half",
                                                         "branch": {"parent": BRANCH_PARENT,
                                                                    "resume_frac": float(r["branch"]["resume_frac"]),
                                                                    "end_frac": float(r["branch"]["end_frac"])}})
        out[f"shake-eval{sfx}"] = _toy(run_config(largest, r, out_repo), f"shake-eval{sfx}", 20, 5, lr_of[largest],
                                       smoke={"steps": 10}, eval={"full_at_fracs": [0.5], "mini": {"every_steps": 5}},
                                       ckpt={"full_at_fracs": None, "weights_at_fracs": [0.5], "upload_full_at": []})
    return out


def anchor_config(r: dict | None = None, out_repo: str = RUNS_REPO) -> dict:
    """What scripts/05_evaluate.py reads to re-score the first run's 0.6B on the study's eval rows: the reference run's
    data and eval keys (its student dir is the study's T-0.6B, whose processor the checkpoint carries too). The
    schedule and LR only make the file load: 05_evaluate never trains."""
    r = r or rules()
    return _merge(run_config(prereg.T_REF_RUN, r, out_repo),
                  {"run_name": ANCHOR, "optim": {"lr": 1e-4}, "schedule": {"max_steps": 10000},
                   "hf": {"output_repo": None}})


def all_configs(r: dict | None = None, out_repo: str = RUNS_REPO, box_runs: list[str] | None = None) -> dict:
    """name -> config for every file under configs/study/."""
    r = r or rules()
    out = {}
    for run in r["runs"]:
        out[run] = run_config(run, r, out_repo)
        out[f"{run}-half"] = branch_config(run, r, out_repo)
        if run != prereg.REPLICATE:  # the replicate takes study-t01's numbers: it is never calibrated
            out[f"calib-{run}"] = calib_config(run, r, out_repo)
    for cls, p in r["lr_probes"]["classes"].items():
        for lr in p["grid"]:
            out[prereg.probe_run_name(cls, lr)] = probe_config(cls, lr, r, out_repo)
    from kitsune.study_queue import shakedown_runs

    out.update(shake_configs(list(box_runs or shakedown_runs(r)), r, out_repo))
    out[ANCHOR] = anchor_config(r, out_repo)
    return out


def render(cfg: dict) -> str:
    return json.dumps(cfg, indent=1, ensure_ascii=False) + "\n"


def write_all(out_dir: Path = OUT_DIR, out_repo: str = RUNS_REPO) -> tuple[list[str], list[str]]:
    """Write every config; remove *.json in out_dir the generator no longer makes. Returns (written, removed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfgs = all_configs(out_repo=out_repo)
    for name, cfg in cfgs.items():
        (out_dir / f"{name}.json").write_bytes(render(cfg).encode("utf-8"))
    stale = sorted(p.name for p in out_dir.glob("*.json") if p.stem not in cfgs)
    for name in stale:
        (out_dir / name).unlink()
    return sorted(cfgs), stale


def check(out_dir: Path = OUT_DIR, out_repo: str = RUNS_REPO) -> list[str]:
    """The differences between the files in out_dir and the generator's output (parsed JSON, so a CRLF checkout
    compares equal)."""
    cfgs = all_configs(out_repo=out_repo)
    problems = []
    for name, cfg in cfgs.items():
        p = out_dir / f"{name}.json"
        if not p.is_file():
            problems.append(f"{p.name}: missing")
        elif json.loads(p.read_text(encoding="utf-8")) != json.loads(render(cfg)):
            problems.append(f"{p.name}: differs from the generator's")
    problems += [f"{p.name}: not made by the generator" for p in sorted(out_dir.glob("*.json")) if p.stem not in cfgs]
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare the committed files with the generator's output")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--out-repo", default=RUNS_REPO, help="hf.output_repo of every config (the runs repo)")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    if args.check:
        problems = check(out_dir, args.out_repo)
        print(f"{out_dir}: " + ("up to date" if not problems else f"{len(problems)} problem(s)")
              + "".join(f"\n  {p}" for p in problems))
        return 1 if problems else 0
    written, removed = write_all(out_dir, args.out_repo)
    print(f"wrote {len(written)} configs to {out_dir}" + (f"; removed {removed}" if removed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
