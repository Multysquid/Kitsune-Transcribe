"""The trainer's retired modes (scripts/04_distill.py RETIRED_KEYS): the config keys of the size study's T/2 branch,
LR probe and calibration runs, the laptop's host-memory AdamW (optim.offload), eval.gate false and the augmentation's
mix, quiet pads and short-row guard - modes no released model trained with, gone from DEFAULTS.

- a retired key, or a retired block (each key it spells), at the value that left its mode off is dropped wherever a
  config comes in: a file (_merge: load_config, a resume's saved config, scripts/05_evaluate.py's load_cfg), --set
  (apply_set) and a resume's overrides (a repeat, never a change); any other value, a key a retired block never had and
  a block that is not an object are refused, naming the key
- every committed configs/full/full-*.json and smoke-*.json (byte-pinned provenance that spells several of them) loads:
  its keys are DEFAULTS' or retired ones at their off values; and so does each as a saved run config - the full
  DEFAULTS of before the cleanup, the retired keys at their off values, merged with it, as runs/<run_id>/config.json
  and trainer.pt hold it - through build()'s merge, a resume's --set of a retired key and 05's readout loader
- the keys kept for the saved configs say what is left: bn.mode "frozen", bn.momentum null, loss.aux_ctc_weight 0,
  eval.reference null
CPU only, no model."""
import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does not hide the GPU from the Windows CUDA driver; -1 does
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import pytest  # noqa: E402

from fixtures import load_script  # noqa: E402

FULL = ROOT / "configs" / "full"
COMMITTED = sorted(FULL.glob("full-*.json")) + sorted(FULL.glob("smoke-*.json"))
# the table as the cleanup wrote it: every key it retired, with the value DEFAULTS had for it (the one that left its
# mode off) - what every saved run config of the released runs spells
RETIRED = {
    "branch": {"parent": None, "resume_frac": 0.4, "end_frac": 0.5},
    "lr_probe": {"enabled": False},
    "calibrate": {"enabled": False, "window": [50, 250], "barrier": False},
    "optim.offload": "none", "optim.offload_fused": False,
    "eval.gate": True,
    "augment.mix_p": 0.0, "augment.mix_snr_db": [5.0, 20.0], "augment.truncate_pad_p": 0.0,
    "augment.end_pad_p": 0.0, "augment.pad_frames": [1, 5], "augment.background_min_row_s": 0.0,
}


@pytest.fixture(scope="module")
def m():
    return load_script("04_distill")


def leaves(d: dict, where: str = ""):
    for k, v in d.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            yield from leaves(v, f"{where}{k}.")
        else:
            yield f"{where}{k}", v


def nested(key: str, value) -> dict:
    """{"a": {"b": value}} for "a.b"."""
    out = value
    for part in reversed(key.split(".")):
        out = {part: out}
    return out


def old_defaults(m) -> dict:
    """DEFAULTS as they were before the cleanup: today's with every retired key back at its off value."""
    out = copy.deepcopy(m.DEFAULTS)
    for key, off in RETIRED.items():
        node, parts = out, key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = copy.deepcopy(off)
    return out


def test_the_table_and_defaults(m):
    """RETIRED_KEYS is the table above, none of its keys is in DEFAULTS, and the keys kept for the saved configs hold
    the one value they may still take."""
    assert m.RETIRED_KEYS == RETIRED
    flat = dict(leaves(m.DEFAULTS))
    for key in RETIRED:
        assert key not in flat and not any(k.startswith(key + ".") for k in flat), key
    d = m.load_config(None, [])
    assert d["bn"] == {"mode": "frozen", "momentum": None} and m.BN_MODES == ("frozen",)
    assert d["loss"]["aux_ctc_weight"] == 0.0 and d["eval"]["reference"] is None


@pytest.mark.parametrize("key", list(RETIRED))
def test_a_retired_key_at_its_off_value_is_dropped(m, key):
    """In a file, through --set and as a resume's override: the config is DEFAULTS' as if the key were not there."""
    off = RETIRED[key]
    assert m._merge(m.DEFAULTS, nested(key, copy.deepcopy(off))) == m.DEFAULTS
    cfg = copy.deepcopy(m.DEFAULTS)
    assert m.apply_set(cfg, f"{key}={json.dumps(off)}") == (key, off) and cfg == m.DEFAULTS
    changed, same = m.resume_overrides(m.load_config(None, []), [(key, copy.deepcopy(off))])
    assert changed == {} and same == {key: off}
    if isinstance(off, dict):  # a block's keys one at a time, and a block that spells only some of them
        for k, v in off.items():
            assert m.apply_set(cfg, f"{key}.{k}={json.dumps(v)}") == (f"{key}.{k}", v) and cfg == m.DEFAULTS
        first = next(iter(off))
        assert m._merge(m.DEFAULTS, {key: {first: off[first], "_comment": "x"}}) == m.DEFAULTS
    assert m.load_config(None, [f"{key}={json.dumps(off)}"]) == m.load_config(None, [])


@pytest.mark.parametrize("key, value", [
    ("optim.offload", "cpu"), ("optim.offload", None), ("optim.offload_fused", True), ("eval.gate", False),
    ("eval.gate", 1),  # a bool key takes a bool only
    ("augment.mix_p", 0.05), ("augment.mix_p", False), ("augment.mix_snr_db", [10.0, 20.0]),
    ("augment.truncate_pad_p", 0.3), ("augment.end_pad_p", 0.2), ("augment.pad_frames", [1, 6]),
    ("augment.background_min_row_s", 3.0), ("branch", {"parent": "runs/x"}), ("branch.parent", "runs/x"),
    ("branch", "runs/x"), ("lr_probe", {"enabled": True}), ("lr_probe.enabled", True),
    ("calibrate", {"enabled": False, "window": [50, 300], "barrier": False}), ("calibrate.barrier", True),
])
def test_any_other_value_is_refused(m, key, value):
    """The mode is gone: a config, a --set or a resume that would turn it on (or give it other settings) stops,
    naming the key."""
    with pytest.raises(SystemExit, match=key.split(".")[0]):
        m._merge(m.DEFAULTS, nested(key, value))
    with pytest.raises(SystemExit, match=key.split(".")[0]):
        m.apply_set(copy.deepcopy(m.DEFAULTS), f"{key}={json.dumps(value)}")
    with pytest.raises(SystemExit, match=key.split(".")[0]):
        m.resume_overrides(m.load_config(None, []), [(key, value)])


def test_unknown_keys_stay_unknown(m):
    """A key a retired block never had is an unknown key, as any typo; so is any other key that is not in DEFAULTS."""
    for over in ({"branch": {"parent": None, "nope": 1}}, {"lr_probe": {"enabled": False, "lr": 1e-3}},
                 {"optim": {"offload_cpu": "none"}}, {"augment": {"mix": 0.0}}):
        with pytest.raises(SystemExit, match="unknown config key"):
            m._merge(m.DEFAULTS, over)
    for word in ("branch.nope=1", "calibrate.steps=3", "augment.pads=0"):
        with pytest.raises(SystemExit, match="unknown config key"):
            m.apply_set(copy.deepcopy(m.DEFAULTS), word)


@pytest.mark.parametrize("path", COMMITTED, ids=[p.stem for p in COMMITTED])
def test_the_committed_configs_load(m, path):
    """A committed full or smoke config: every key DEFAULTS' or a retired one at its off value, and load_config takes
    it; the same config saved by the trainer before the cleanup (old DEFAULTS merged with it) loads through build()'s
    merge - its retired keys dropped, the rest as the file says -, takes a resume's --set of a retired key at its off
    value and refuses one that would turn its mode on."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    flat = dict(leaves(m.DEFAULTS))
    for key, value in leaves(raw):
        if key in flat or any(key.startswith(f + ".") for f in flat if isinstance(flat[f], type(None))):
            continue
        block = next((r for r in RETIRED if key == r or key.startswith(r + ".")), None)
        assert block is not None, f"{path.name}: {key} is neither a trainer key nor a retired one"
        off = RETIRED[block] if key == block else RETIRED[block][key[len(block) + 1:]]
        assert value == off, f"{path.name}: {key} {value!r}"
    cfg = m.load_config(str(path), [])
    saved = m._merge(old_defaults(m), raw)  # runs/<run_id>/config.json: the full DEFAULTS of its day
    assert any(k in saved for k in ("branch", "lr_probe", "calibrate")) and "offload" in saved["optim"]
    resumed = m._merge(m.DEFAULTS, copy.deepcopy(saved))  # build()'s resume path
    m.validate(resumed)
    assert resumed == cfg
    word = "augment.mix_p=0.0"
    assert m.resume_overrides(copy.deepcopy(resumed), [m.apply_set(resumed, word)]) == ({}, {"augment.mix_p": 0.0})
    with pytest.raises(SystemExit, match="augment.mix_p"):
        m.apply_set(resumed, "augment.mix_p=0.1")


def test_the_readout_loader_takes_a_saved_run_config(m, tmp_path):
    """scripts/05_evaluate.py's load_cfg (every full run's readout) reads a run's config.json - its resolved config
    under "config", the retired keys in it - and validates it as the trainer does."""
    e = load_script("05_evaluate")
    raw = json.loads((FULL / "full-p03.json").read_text(encoding="utf-8"))
    run_cfg = tmp_path / "config.json"
    run_cfg.write_text(json.dumps(dict(run_id="full-p03-20261003T230143Z", argv=[],
                                       config=m._merge(old_defaults(m), raw))), encoding="utf-8")
    cfg, p = e.load_cfg(m, SimpleNamespace(config=str(run_cfg), set=["eval.gate=true"], batch_s=None,
                                           root=str(ROOT)))
    assert p == run_cfg and "gate" not in cfg["eval"] and "mix_p" not in cfg["augment"] and "branch" not in cfg
    assert cfg["augment"]["concat_p"] == raw["augment"]["concat_p"] and cfg["family"] == "ctc"
