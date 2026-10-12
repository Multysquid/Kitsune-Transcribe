"""The size study's run configs, the base of every full-data config (tools/make_full_configs.py full_config).

A study run's config (run_config) is three layers, merged in this order:
  1. the trainer part of configs/next_run_template.json (its data keys and its _comment left out);
  2. the data block, study/data.json verbatim (the selection, both label roots with pull_parakeet, the extent, the
     sources and eval sets, the selection recipe with its study block);
  3. the study's settings: the steps clock with cooldown_frac 0.2, early stop off, L2-SP 0, the complete evals at
     eval.full_at_fracs [0.2, 0.4, 0.6, 0.8] (eval.every_min null, no per-epoch evals), the lean checkpoints (weights
     at 0.4 and at the end, a local full state at 0.4, and at 0.8 uploaded for the runs of at most 0.1B), log syncs
     every 20 min, the runs repo as hf.output_repo; and per run what kitsune.prereg.RUNS fixes: family, student,
     seed, the warm-up, the weight decay and the planned micro-batch, plus loss.w_ctc (kitsune.prereg.W_CTC) for the
     CTC students. BatchNorm stays frozen and the aux-CTC weight 0, as for every pruned student: the trainer has no
     other mode now (the from-scratch runs, bridge included, trained BatchNorm and an aux-CTC head, both retired), so
     run_config refuses a from-scratch run.
schedule.max_steps, optim.lr and eval.mini.every_steps stay null: full_config puts the full run on the epochs clock
and sets its LR (FULL_RUNS).

The size study's own configs, configs/study/*.json, were generated from this layer by the study's generator (removed
after the study; the last commit with it is 76b28a1). Only configs/study/study-t06.json is left: box smoke-b's
stores-eval item reads it, and tests/test_prereg.py pins its bytes.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kitsune import prereg  # noqa: E402

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


def run_config(run: str, out_repo: str = RUNS_REPO) -> dict:
    """The study run `run` (a key of kitsune.prereg.RUNS, a pruned student's) with its placeholders (max_steps, lr,
    mini cadence) null. ValueError for a from-scratch run: it trained modes the trainer no longer has."""
    spec = prereg.RUNS[run]
    if spec["init_class"] == "scratch":
        raise ValueError(f"{run}: the size study's from-scratch runs trained retired modes (bn.mode train and the "
                         "aux-CTC head; 04_distill BN_MODES, validate_study)")
    small = int(spec["params_total"]) <= SMALL_PARAMS
    over = {
        "run_name": run, "student": spec["student"], "seed": int(spec["seed"]),
        "schedule": {"warmup_steps": int(spec["warmup_steps"])},
        "optim": {"weight_decay": float(spec["weight_decay"])},
        "loss": {"aux_ctc_weight": 0.0},  # spelled, as the byte-pinned configs/full/*.json do
        "bn": {"mode": "frozen"},
        "batch": {"micro_audio_s": spec["micro_audio_s"]},
        "ckpt": {"full_at_fracs": list(FULL_FRACS_SMALL if small else FULL_FRACS),
                 "upload_full_at": ["frac:0.8"] if small else []},
    }
    if spec["family"] == "ctc":  # the CTC trainer's keys (WP4b; CONTRACT.md section 2)
        over.update(family="ctc", loss=dict(over["loss"], w_ctc=float(prereg.W_CTC)))
    return _merge(base_config(out_repo), over)


def render(cfg: dict) -> str:
    return json.dumps(cfg, indent=1, ensure_ascii=False) + "\n"
