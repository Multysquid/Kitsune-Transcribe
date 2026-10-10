"""A tiny, valid box registry of the full-data runs (kitsune/fullrun.py, contract 2.3), for the tests of every package.

    from fixtures_full import tiny_registry
    reg = tiny_registry(tmp_path)                    # writes tmp_path/configs/full/*.json, returns the registry dict
    fullrun.load_registry(reg, root=tmp_path)        # validates (the configs are read under tmp_path)
    fullrun.box_students("p01", reg)                 # the dict remembers tmp_path: no root= needed (a JSON round trip
                                                     # or dict(reg) forgets it, and then root= is needed again)
    tiny_registry(tmp_path, write_boxes=True)        # + tmp_path/configs/full/boxes.json (for KITSUNE_FULL_REGISTRY)

It has the four boxes of section 7 in miniature, and between them every registry feature, so a test takes the box it
needs and changes what it tests:
  p01         1 GPU, the box-1 shape: stores-ctc -> full-p01 (train, ctc) -> m4-full-p01 (readout)
  full        2 GPUs, the shared queue: stores-ctc, stores-aed (needs stores-ctc); full-t06 (aed), full-p03, full-p005
              (droppable); their readouts; an eval of full-p03 (same-box `of`, needs its readout), an eval of box
              p01's full-p01 (`of_box`), a Whisper eval (no model source); the decision-22 re-time pair (speed: `of` and
              `weights`)
  full-smoke  1 GPU, smoke: the four smoke trainers (stall_min 10, plan_total_steps / plan_hours) + smoke-nostart
              (droppable, max_hours 999), two readouts, speed items with stall_min null (weights, cohere, a
              data-repo `model`), and the five faults F1-F5 (watchdog 600 s / alert)
  smoke-b     1 GPU, smoke, no gate, no timed states: an eval-only store build with --set values, evals on `weights`
              ({config:<name>} / {ckpt:<name>}, {out:<item>}), verdict specs of checks 12, 14, 15 and 16, and a speed
              item with only_if_new_machine full-smoke
The configs are minimal: the data keys of fullrun.FULL_DATA / SMOKE_DATA (smoke-b: study/data.json's), the family,
the student and the run name; no trainer can load them (they are registry fixtures, not training configs).

    reg = cool_registry(tmp_path)                    # DECISIONS H14's continuation box p-cool and its source boxes

cool_registry is another registry (tiny_registry's configs, with full-p03 and full-p005 on box p01's data config): box
p01 and the source boxes full-p (full-p03) and p005 (full-p005) in miniature, and box p-cool, which continues their
three runs (COOL_RUNS, cool_continues) in H14's order, each train item followed by its M4 readout; box `full` (whose
data config the rewritten configs no longer match) is left out, the smoke boxes stay.
"""
import copy
import json
import sys
from pathlib import Path

# stdlib only (no `from fixtures import ROOT`: that loads numpy, pyarrow and soundfile), so a test of a stdlib-only
# script (launch, bootstrap's helper) can use it as cheaply as kitsune.fullrun itself
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kitsune import fullrun  # noqa: E402

STUDY_WEIGHTS = {"study-t06": ("study-t06-20260926T174027Z", 9370), "study-p03": ("study-p03-20260926T172336Z", 25120),
                 "study-p01": ("study-p01-20260926T172421Z", 34620)}
PARAKEET_DIR = "models/parakeet-tdt_ctc-0.6b-ja-hf"


def _weights(*names: str) -> list[dict]:
    return [{"name": n, "run_id": STUDY_WEIGHTS[n][0], "step": STUDY_WEIGHTS[n][1]} for n in names]


def _study_data() -> dict:
    """study/data.json's data keys (the frozen study selection; every configs/study/*.json carries them): smoke-b's
    data block."""
    d = json.loads((ROOT / "study" / "data.json").read_text(encoding="utf-8"))
    return {k: d[k] for k in fullrun.DATA_KEYS if k in d}


def _data_configs() -> dict[str, dict]:
    full = copy.deepcopy(fullrun.FULL_DATA)
    smoke = copy.deepcopy(fullrun.SMOKE_DATA)
    return {"data-p01": dict(full, family="ctc"), "data-full": dict(full, pull_parakeet=True),
            "data-smoke": dict(smoke, pull_parakeet=True), "data-smoke-b": _study_data()}


def _item_configs(data: dict[str, dict]) -> dict[str, dict]:
    """<config name> -> (its data config, family, student)."""
    spec = {"full-p01": ("data-p01", "ctc", "p01"), "full-t06": ("data-full", "aed", "t06"),
            "full-p03": ("data-full", "ctc", "p03"), "full-p005": ("data-full", "ctc", "p005"),
            "smoke-t06": ("data-smoke", "aed", "t06"), "smoke-p03": ("data-smoke", "ctc", "p03"),
            "smoke-p01": ("data-smoke", "ctc", "p01"), "smoke-p005": ("data-smoke", "ctc", "p005"),
            "smoke-b-t06": ("data-smoke-b", "aed", "t06")}
    out = {}
    for name, (dc, family, student) in spec.items():
        cfg = {k: copy.deepcopy(v) for k, v in data[dc].items() if k in fullrun.DATA_KEYS}
        out[name] = dict(cfg, run_name=name, family=family, student=f"students/study/{student}")
    return out


def _cfg(name: str) -> str:
    return f"configs/full/{name}.json"


def _box_p01() -> dict:
    return {"gpus": 1, "data_config": _cfg("data-p01"), "est_hours": 19.5, "max_hours": 22, "max_dph": 1.00,
            "extra_gb": 25, "deadline_reserve_min": 45, "watchdog": {"orphan_s": 3600, "action": "stop"},
            "timed_states": True, "gate": True, "smoke": False, "max_attempts": 4,
            "extra_files": [fullrun.FROZEN_MANIFEST, fullrun.FULL_DIR + "/full.json"], "extra_dirs": [],
            "items": [
                {"name": "stores-ctc", "kind": "stores", "config": _cfg("full-p01")},
                {"name": "full-p01", "kind": "train", "config": _cfg("full-p01"), "study_run": "study-p01",
                 "family": "ctc", "max_hours": 13.56, "needs": ["stores-ctc"]},
                {"name": "m4-full-p01", "kind": "readout", "of": "full-p01", "max_hours": 0.3}]}


def _quant_argv(fmt: str) -> list[str]:
    return ["{python}", "-m", "kitsune.quant", "readout", "--config", "{config}", "--ckpt", "{ckpt}", "--fmt", fmt,
            "--out", "{out}", "--cache-dir", "{cache_dir}", "--manifest", "{manifest}", "--max-temp", "0"]


def _whisper_argv(key: str, *extra: str) -> list[str]:
    return ["{python}", "tools/whisper_eval.py", "--model", key, "--store", "{cache_dir}/eval", "--manifest",
            "{manifest}", "--out", "{out}", "--tables", "{out}/tables", "--hf-cache", "{hf_cache}", "--device", "cuda",
            "--max-temp", "0", *extra]


def _box_full() -> dict:
    train = [("full-t06", "study-t06", "aed", 35.19, "stores-aed", False),
             ("full-p03", "study-p03", "ctc", 22.29, "stores-ctc", False),
             ("full-p005", "study-p005", "ctc", 9.92, "stores-ctc", True)]
    items = [{"name": "stores-ctc", "kind": "stores", "config": _cfg("full-p03")},
             {"name": "stores-aed", "kind": "stores", "config": _cfg("full-t06"), "needs": ["stores-ctc"]}]
    items += [{"name": n, "kind": "train", "config": _cfg(n), "study_run": run, "family": fam, "max_hours": h,
               "needs": [st], "droppable": drop} for n, run, fam, h, st, drop in train]
    items += [{"name": f"m4-{n}", "kind": "readout", "of": n, "max_hours": 0.75 if n == "full-t06" else 0.3}
              for n, *_ in train]
    items += [{"name": "quant-int8-w8a8-full-p03", "kind": "eval", "of": "full-p03", "needs": ["m4-full-p03"],
               "argv": _quant_argv("int8-w8a8"), "max_hours": 0.3},
              {"name": "quant-int8-w8a8-full-p01", "kind": "eval", "of_box": "p01", "of": "full-p01",
               "argv": _quant_argv("int8-w8a8"), "max_hours": 0.3},
              {"name": "whisper-small", "kind": "eval", "argv": _whisper_argv("whisper-small"), "max_hours": 0.3},
              {"name": "speed-full-t06", "kind": "speed", "of": "full-t06", "system": "full-t06", "speed_kind": "aed",
               "needs": ["m4-full-t06"], "max_hours": 0.2},
              {"name": "speed-study-t06", "kind": "speed", "weights": _weights("study-t06"), "system": "study-t06",
               "speed_kind": "aed", "max_hours": 0.2}]
    return {"gpus": 2, "data_config": _cfg("data-full"), "est_hours": 41.9, "max_hours": 47, "max_dph": 1.70,
            "extra_gb": 110, "deadline_reserve_min": 60, "watchdog": {"orphan_s": 3600, "action": "stop"},
            "timed_states": True, "extra_files": [fullrun.FROZEN_MANIFEST, fullrun.FULL_DIR + "/full.json"],
            "items": items}


def _box_full_smoke() -> dict:
    plan = {"smoke-t06": (73452, 35.19), "smoke-p03": (109608, 22.29), "smoke-p01": (110520, 13.56),
            "smoke-p005": (110528, 9.92)}
    runs = {"smoke-t06": ("study-t06", "aed", "stores-aed"), "smoke-p03": ("study-p03", "ctc", "stores-ctc"),
            "smoke-p01": ("study-p01", "ctc", "stores-ctc"), "smoke-p005": ("study-p005", "ctc", "stores-ctc")}
    items = [{"name": "stores-ctc", "kind": "stores", "config": _cfg("smoke-p03")},
             {"name": "stores-aed", "kind": "stores", "config": _cfg("smoke-t06"), "needs": ["stores-ctc"]}]
    items += [{"name": n, "kind": "train", "config": _cfg(n), "study_run": run, "family": fam, "max_hours": 0.75,
               "stall_min": 10, "needs": [st], "plan_total_steps": plan[n][0], "plan_hours": plan[n][1]}
              for n, (run, fam, st) in runs.items()]
    items += [{"name": "smoke-nostart", "kind": "train", "config": _cfg("smoke-p005"), "study_run": "study-p005",
               "family": "ctc", "max_hours": 999, "droppable": True, "needs": ["stores-ctc"]},
              {"name": "m4-smoke-t06", "kind": "readout", "of": "smoke-t06", "max_hours": 0.75},
              {"name": "m4-smoke-p03", "kind": "readout", "of": "smoke-p03", "max_hours": 0.3},
              {"name": "speed-study-p01", "kind": "speed", "weights": _weights("study-p01"), "system": "study-p01",
               "speed_kind": "ctc", "stall_min": None, "max_hours": 0.3},
              {"name": "speed-cohere", "kind": "speed", "system": "cohere", "speed_kind": "cohere", "stall_min": None,
               "max_hours": 0.3},
              {"name": "speed-parakeet-tdt", "kind": "speed", "model": PARAKEET_DIR, "system": "parakeet-tdt",
               "speed_kind": "parakeet-tdt", "stall_min": None, "max_hours": 0.3}]
    faults = [{"id": "F1", "action": "sigstop", "item": "smoke-p01", "at_step": 150},
              {"id": "F2", "action": "kill", "item": "smoke-p03", "at_step": 130},
              {"id": "F3", "action": "wipe_run_dir", "item": "smoke-p01", "after_event": "timed_state_upload_ok",
               "min_attempt": 2},
              {"id": "F4", "action": "deadline", "item": "smoke-p005", "seconds": 600},
              {"id": "F5", "action": "freeze_controller_hb", "item": "smoke-p005", "at_step": 50, "seconds": 900}]
    return {"gpus": 1, "data_config": _cfg("data-smoke"), "est_hours": 5.2, "max_hours": 9, "max_dph": 1.00,
            "extra_gb": 90, "deadline_reserve_min": 20, "watchdog": {"orphan_s": 600, "action": "alert"},
            "timed_states": True, "gate": True, "smoke": True,
            "extra_files": [fullrun.FROZEN_MANIFEST, fullrun.FULL_DIR + "/smoke.json"], "extra_dirs": [PARAKEET_DIR],
            "items": items, "faults": faults}


def _box_smoke_b() -> dict:
    fmt = "int8-w8a8"
    items = [{"name": "stores-eval", "kind": "stores", "config": _cfg("smoke-b-t06"), "eval_only": True,
              "sets": ["optim.lr=0.0002", "schedule.max_steps=20000"]},
             {"name": "selftest", "kind": "eval", "weights": _weights("study-p03", "study-t06"), "max_hours": 0.3,
              "argv": ["{python}", "-m", "kitsune.quant", "selftest", "--device", "cuda:0", "--out",
                       "{out}/selftest.json", "--ckpt", "{ckpt:study-p03}", "--ckpt", "{ckpt:study-t06}"],
              "verdict": [{"check": "12", "json": "{out}/selftest.json", "path": "ok", "equals": True}]},
             {"name": f"quant-{fmt}-study-p03", "kind": "eval", "weights": _weights("study-p03"), "max_hours": 0.3,
              "argv": [a.replace("{config}", "{config:study-p03}").replace("{ckpt}", "{ckpt:study-p03}")
                       for a in _quant_argv(fmt)]},
             {"name": f"mem-{fmt}-study-p03", "kind": "eval", "weights": _weights("study-p03"), "max_hours": 0.3,
              "argv": ["{python}", "scripts/05_evaluate.py", "--config", "{config:study-p03}", "--ckpt",
                       "{ckpt:study-p03}", "--quant", fmt, "--out", "{out}", "--tables", "{out}/tables", "--manifest",
                       "{manifest}", "--cache-dir", "{cache_dir}", "--max-temp", "0"]},
             {"name": f"cmp-{fmt}-study-p03", "kind": "eval", "max_hours": 0.3,
              "argv": ["{python}", "-m", "kitsune.quant", "compare", f"{{out:quant-{fmt}-study-p03}}",
                       f"{{out:mem-{fmt}-study-p03}}", "--exact", "--json-out", "{out}/compare.json"],
              "verdict": [{"check": "14", "json": "{out}/compare.json", "path": "same", "equals": True}]},
             {"name": "whisper-large-v3", "kind": "eval", "max_hours": 0.75,
              "argv": _whisper_argv("whisper-large-v3", "--sets", "eval_jsut"),
              "verdict": [{"check": "15", "json": "{out}/whisper.json", "path": "sets.eval_jsut.cer_corpus",
                           "min": 0.051, "max": 0.091}]},
             {"name": "speed-study-p03", "kind": "speed", "weights": _weights("study-p03"), "system": "study-p03",
              "speed_kind": "ctc", "only_if_new_machine": "full-smoke", "max_hours": 0.3,
              "verdict": [{"check": "16"}]}]
    return {"gpus": 1, "data_config": _cfg("data-smoke-b"), "est_hours": 1.75, "max_hours": 3, "max_dph": 1.00,
            "extra_gb": 40, "deadline_reserve_min": 15, "watchdog": {"orphan_s": 3600, "action": "stop"},
            "timed_states": False, "gate": False, "smoke": True, "extra_dirs": [PARAKEET_DIR], "items": items}


def tiny_registry(root, *, write_boxes: bool = False) -> dict:
    """Write minimal data and item configs under root/configs/full/ and return a valid registry dict whose paths are
    repo-relative. The dict remembers root (as a loaded registry does), so fullrun's readers resolve its paths there
    without root=; passing root=root explicitly works too. write_boxes also writes it to root/configs/full/boxes.json,
    so KITSUNE_FULL_REGISTRY (or load_registry(None, root=root)) can find it."""
    folder = Path(root) / "configs" / "full"
    folder.mkdir(parents=True, exist_ok=True)
    data = _data_configs()
    for name, cfg in {**data, **_item_configs(data)}.items():
        (folder / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    reg = {"_comment": "tests/fixtures_full.tiny_registry: the section 7 boxes in miniature", "version": 1,
           "boxes": {"full-smoke": _box_full_smoke(), "p01": _box_p01(), "full": _box_full(),
                     "smoke-b": _box_smoke_b()}}
    if write_boxes:
        (Path(root) / fullrun.BOXES_FILE).write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
    return fullrun._with_root(reg, root)


# DECISIONS H14's box p-cool in miniature: item -> (source box, run id, from_step, to_step, resets_before, epochs,
# keep_cooldown), in its queue order. The fake trainer's 20 steps per epoch end the run at to_step = epochs x 20 and put
# its pre_cooldown state at 0.8 of that (from_step); P-0.1B's run had one reset before (its readout dir comes out -r2)
COOL_RUNS = {"full-p03": ("full-p", "full-p03-20261003T230143Z", 16, 20, 0, 1, False),
             "full-p01": ("p01", "full-p01-20261001T184145Z", 32, 40, 1, 2, False),
             "full-p005": ("p005", "full-p005-20261008T160232Z", 16, 20, 0, 1, True)}
# the two recipes' explicit ranges (tools/make_full_configs.py RECIPE_GENTLE / RECIPE_V4_FULL's continuation sets)
COOL_GENTLE = {"augment.noise_snr_db": [5.0, 20.0], "augment.speech_snr_db": [15.0, 25.0],
               "augment.gain_db": [-10.0, 10.0], "augment.codecs": ["mp3", "ulaw8k"],
               "augment.background_min_row_s": 3.0}
COOL_V4_FULL = {"augment.noise_snr_db": [0.0, 20.0], "augment.speech_snr_db": [10.0, 25.0],
                "augment.gain_db": [-20.0, 10.0], "augment.codecs": ["mp3", "gsm", "ulaw8k"],
                "augment.background_min_row_s": 0.0}


def cool_continues(name: str) -> dict:
    """The continues block of box p-cool's train item `name` (COOL_RUNS): schedule.epochs, schedule.deadline_cooldown
    false, the keep-cooldown flag where H14 sets it, then the recipe's ranges (make_full_configs'
    continuation_set_dict order)."""
    src, rid, from_step, to_step, before, epochs, keep = COOL_RUNS[name]
    sets = {"schedule.epochs": epochs, "schedule.deadline_cooldown": False}
    if keep:
        sets["schedule.resume_reset_keep_cooldown"] = True
    sets.update(copy.deepcopy(COOL_V4_FULL if name == "full-p01" else COOL_GENTLE))
    return {"box": src, "run_id": rid, "from_step": from_step, "to_step": to_step, "resets_before": before,
            "sets": sets}


def _box_source(name: str, study_run: str, hours: float) -> dict:
    """A 1x source box in box p01's shape: stores-ctc -> <name> (train, ctc) -> m4-<name>."""
    box = _box_p01()
    box["items"] = [{"name": "stores-ctc", "kind": "stores", "config": _cfg(name)},
                    {"name": name, "kind": "train", "config": _cfg(name), "study_run": study_run, "family": "ctc",
                     "max_hours": hours, "needs": ["stores-ctc"]},
                    {"name": f"m4-{name}", "kind": "readout", "of": name, "max_hours": 0.3}]
    return box


def _box_cool() -> dict:
    study = {"full-p03": "study-p03", "full-p01": "study-p01", "full-p005": "study-p005"}
    items = [{"name": "stores-ctc", "kind": "stores", "config": _cfg("full-p03")}]
    for name in COOL_RUNS:
        items += [{"name": name, "kind": "train", "config": _cfg(name), "study_run": study[name], "family": "ctc",
                   "max_hours": 6.0, "needs": ["stores-ctc"], "continues": cool_continues(name)},
                  {"name": f"m4-{name}", "kind": "readout", "of": name, "max_hours": 0.3}]
    box = _box_p01()
    box.update(est_hours=19.0, max_hours=22, max_dph=1.1, extra_gb=70, min_ram_gb=96, items=items)
    return box


def cool_registry(root, *, write_boxes: bool = False) -> dict:
    """tiny_registry's configs under root, with full-p03 and full-p005 rewritten on box p01's data config, and a valid
    registry of boxes p01, full-p, p005 (the sources), p-cool (their continuations) and the two smoke boxes (the
    module docstring). write_boxes: also root/configs/full/boxes.json."""
    reg = copy.deepcopy(dict(tiny_registry(root)))
    folder = Path(root) / "configs" / "full"
    data = _data_configs()["data-p01"]
    for name, student in (("full-p03", "p03"), ("full-p005", "p005")):
        cfg = {k: copy.deepcopy(v) for k, v in data.items() if k in fullrun.DATA_KEYS}
        cfg.update(run_name=name, family="ctc", student=f"students/study/{student}")
        (folder / f"{name}.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    boxes = {k: v for k, v in reg["boxes"].items() if k != "full"}
    boxes.update({"full-p": _box_source("full-p03", "study-p03", 22.29),
                  "p005": _box_source("full-p005", "study-p005", 9.92), "p-cool": _box_cool()})
    reg = {"_comment": "tests/fixtures_full.cool_registry: DECISIONS H14's box p-cool and its source boxes in miniature",
           "version": 1, "boxes": boxes}
    if write_boxes:
        (Path(root) / fullrun.BOXES_FILE).write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
    return fullrun._with_root(reg, root)
