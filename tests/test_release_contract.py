"""The release contract: what the released models' cards, every published quantization.json (`load_with`) and the
external integration's evaluation scripts name without a commit pin, so it must keep working on main whatever else
changes.
- kitsune.quant.load_quantized(path, device, *, impl="auto", dtype=torch.float32) -> (model, recipe), and the format
  constants of the published variant folders: a changed RECIPE_VERSION or FP16_FP32_MODULES makes
  recipe_version_problem refuse every released folder. Importing kitsune.quant stays light (torch and the stdlib)
- kitsune.generation.RepetitionStop(prompt_len, window=24, max_period=12): the T-0.6B card cites it ("after 24
  repeating tokens")
- MODEL_CARD.md at the repo root: the license_link of the uploaded checkpoint READMEs and kitsune/student.py's template
- the names the integration's evaluation scripts import from kitsune
Names, signatures and constants only, on CPU with no model: tests/test_quant.py round-trips a real export. The file
name stays outside image.yml's tests/test_quant*.py filter, so a change here does not rebuild the training image."""
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

# module -> the names the integration's evaluation scripts use (they import kitsune from a checkout, not a commit)
INTEGRATION_NAMES = {
    "kitsune.quant": ("load_quantized", "read_recipe", "verify_export", "recipe_version_problem", "unpack_weight",
                      "QuantPack", "QuantLinear", "_weight_only_dequant", "NonFiniteMonitor"),
    "kitsune.ctc_student": ("load_ctc_student", "save_ctc_student", "load_meta", "ctc_features", "ctc_log_probs",
                            "greedy_ctc_ids", "decode_ids", "lengths_to_mask", "load_parakeet_ctc", "_save_state_dict",
                            "PROCESSOR_FILES", "CTC_BLANK"),
    "kitsune.patches": ("patch_relpos_once_per_batch",),
    "kitsune.audio": ("decode_audio",),
    "kitsune.features": ("pad_waves",),
    "kitsune.trainset": ("pack_micro_batches",),
    "kitsune.text": ("normalize_ja", "cer"),
    "kitsune.acoustics": ("codec", "gain", "reverb"),
    "kitsune.noise_bank": ("add_background", "voiced_power"),
}


def test_load_quantized_keeps_its_signature():
    from kitsune import quant

    params = inspect.signature(quant.load_quantized).parameters
    assert list(params) == ["path", "device", "impl", "dtype"]
    assert all(params[n].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for n in ("path", "device"))
    assert all(params[n].default is inspect.Parameter.empty for n in ("path", "device"))
    assert all(params[n].kind is inspect.Parameter.KEYWORD_ONLY for n in ("impl", "dtype"))
    assert params["impl"].default == "auto" and params["dtype"].default is torch.float32
    args = quant.parse_args(["inspect", "variant"])  # python -m kitsune.quant inspect <dir>
    assert (args.cmd, args.dir) == ("inspect", "variant")


def test_the_published_format_constants_are_unchanged():
    from kitsune import quant

    assert quant.SCHEMA == 1 and quant.RECIPE_VERSION == 2
    assert quant.FP16_FP32_MODULES == ("ParakeetEncoderAttention",)
    assert (quant.QUANT_FILE, quant.WEIGHTS_FILE) == ("quantization.json", "model.safetensors")
    assert quant._PACK_PARTS == {"int8": ("qweight", "qscale"), "fp8": ("qweight", "qscale"),
                                 "nvfp4": ("qweight", "qblock_scale", "qtensor_scale"),
                                 "mxfp4": ("qweight", "qblock_scale")}


def test_importing_the_loader_stays_light():
    """The cards' loader imports torch and the stdlib only: no transformers, safetensors or torchao until a load, and
    no other kitsune module."""
    code = ("import sys, kitsune.quant; print(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('transformers', 'safetensors', 'torchao') or (m.startswith('kitsune.') and m != 'kitsune.quant')))")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["[]"], r.stdout


def test_repetition_stop_keeps_its_signature_and_window():
    from kitsune.generation import RepetitionStop

    params = inspect.signature(RepetitionStop).parameters
    assert list(params) == ["prompt_len", "window", "max_period"]
    assert params["prompt_len"].default is inspect.Parameter.empty
    assert (params["window"].default, params["max_period"].default) == (24, 12)
    stop = RepetitionStop(2)
    ids = torch.tensor([[1, 2, *([5, 6] * 12)], [1, 2, *range(10, 34)]])  # 24 generated tokens: periodic, distinct
    assert stop(ids, None).tolist() == [True, False]
    assert stop(ids[:, :-1], None).tolist() == [False, False]  # 23: not a full window yet


def test_model_card_stays_at_the_repo_root():
    from kitsune import student

    assert (ROOT / "MODEL_CARD.md").is_file() and student.MODEL_CARD == ROOT / "MODEL_CARD.md"


def test_the_integrations_names_exist():
    missing = [f"{mod}.{name}" for mod, names in INTEGRATION_NAMES.items()
               for name in names if not hasattr(importlib.import_module(mod), name)]
    assert not missing, missing
