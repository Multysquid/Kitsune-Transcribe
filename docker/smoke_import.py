"""Import every training dependency and run a tiny CPU forward/backward of a random CohereAsr model.

Why: CI runs this inside the freshly built image before the image is tagged (.github/workflows/image.yml), so a
broken wheel, an ABI mismatch (numpy vs numba, torch vs transformers) or a pin that did not stick is caught for free
on a GitHub runner instead of on a rented A100. It needs no GPU, no network and no data, and it exercises the exact
call path the trainer uses: `model.model(...)` with sdpa attention on a padded batch, then the LM head in fp32.

Usage (inside the image; the base image's ENTRYPOINT must be bypassed):
  docker run --rm --entrypoint /venv/main/bin/python <image> /opt/kitsune/smoke_import.py
  python docker/smoke_import.py --skip-missing          # on a dev box that lacks some training-only packages
"""
import argparse
import importlib
import io
import os
import re
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

HERE = Path(__file__).resolve().parent
# distribution name (as pinned in requirements-train.txt) -> module to import
MODULES = {
    "torch": "torch",
    "transformers": "transformers",
    "tokenizers": "tokenizers",
    "sentencepiece": "sentencepiece",
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface_hub",
    "hf_xet": "hf_xet",
    "numpy": "numpy",
    "pyarrow": "pyarrow",
    "pandas": "pandas",
    "soundfile": "soundfile",
    "librosa": "librosa",
    "soxr": "soxr",
    "numba": "numba",
    "llvmlite": "llvmlite",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "jiwer": "jiwer",
    "tensorboard": "tensorboard",
    "tqdm": "tqdm",
    "nvidia-ml-py": "pynvml",
    "psutil": "psutil",
    "pyyaml": "yaml",
    "pytest": "pytest",
    "vastai": "vastai",
}
PROMPT = [13764, 7, 4, 16, 98, 98, 5, 9, 11, 13]  # the teacher's ja decoder prompt (scripts/02_teacher_pass.py)
PAD, EOS = 2, 3


def find_requirements() -> Path | None:
    for p in (HERE / "requirements-train.txt", HERE.parent / "requirements-train.txt"):
        if p.exists():
            return p
    return None


def parse_pins(path: Path) -> dict[str, str]:
    """`name==version` lines of a requirements file -> {normalised name: version}; unpinned lines are ignored."""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)\s*==\s*([^\s;]+)", line)
        if m:
            pins[normalise(m.group(1))] = m.group(2)
    return pins


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def check_imports(pins: dict[str, str], skip_missing: bool = False) -> list[str]:
    """Import every module and compare installed versions with the pins. Returns a list of problems."""
    problems = []
    for dist, mod in MODULES.items():
        try:
            installed = version(dist)
        except PackageNotFoundError:
            if not skip_missing:
                problems.append(f"{dist}: not installed")
            print(f"  {dist:16s} MISSING")
            continue
        try:
            importlib.import_module(mod)
        except Exception as e:  # an installed-but-broken wheel is exactly what this script exists to catch
            problems.append(f"{dist}: import {mod} failed: {type(e).__name__}: {e}")
            continue
        want = pins.get(normalise(dist))
        flag = ""
        if want is not None and installed != want:
            problems.append(f"{dist}: installed {installed}, pinned {want}")
            flag = f"  != pinned {want}"
        print(f"  {dist:16s} {installed}{flag}")
    return problems


def audio_roundtrip() -> str:
    """FLAC encode/decode in memory (libsndfile inside the soundfile wheel) and a soxr resample."""
    import numpy as np
    import soundfile as sf
    import soxr

    sr = 16000
    wave = (0.1 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="FLAC")
    back, sr2 = sf.read(io.BytesIO(buf.getvalue()), dtype="float32")
    assert sr2 == sr and back.shape == wave.shape and np.abs(back - wave).max() < 1e-3, "FLAC round trip failed"
    up = soxr.resample(wave, sr, 48000)
    assert abs(len(up) - 3 * len(wave)) <= 2, "soxr resample length"
    return sf.__libsndfile_version__


def tiny_forward_backward(seed: int = 0) -> dict:
    """Random tiny CohereAsr (real 16384 vocab and prompt ids), padded batch of 2, sdpa, fp32 head, CE backward."""
    import torch
    import torch.nn.functional as F
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

    torch.manual_seed(seed)
    cfg = CohereAsrConfig(
        vocab_size=16384, hidden_size=64, num_hidden_layers=1, num_attention_heads=2, intermediate_size=128,
        max_position_embeddings=256, decoder_start_token_id=PROMPT[0], tie_word_embeddings=True,
        encoder_config=dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, intermediate_size=128,
                            subsampling_conv_channels=16),
        attn_implementation="sdpa",
    )
    model = CohereAsrForConditionalGeneration(cfg)
    assert model.config._attn_implementation == "sdpa"
    model.train()
    for m in model.modules():  # the trainer keeps BatchNorm frozen (kitsune.patches.freeze_batchnorm)
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

    n_frames = [200, 150]  # 2 s and 1.5 s of 100 fps mel; row 1 is padded
    feats = torch.randn(2, max(n_frames), 128)
    feat_mask = torch.zeros(2, max(n_frames), dtype=torch.long)
    for i, n in enumerate(n_frames):
        feat_mask[i, :n] = 1
        feats[i, n:] = 0
    targets = [torch.randint(20, 16000, (7,)).tolist() + [EOS], torch.randint(20, 16000, (4,)).tolist() + [EOS]]
    dec_len = len(PROMPT) - 1 + max(len(t) for t in targets)
    dec = torch.full((2, dec_len), PAD, dtype=torch.long)
    dec_mask = torch.zeros(2, dec_len, dtype=torch.long)
    pos, tgt = [], []
    for i, t in enumerate(targets):
        seq = PROMPT + t[:-1]
        dec[i, :len(seq)] = torch.tensor(seq)
        dec_mask[i, :len(seq)] = 1
        pos += [(i, len(PROMPT) - 1 + j) for j in range(len(t))]
        tgt += t

    out = model.model(input_features=feats, attention_mask=feat_mask, decoder_input_ids=dec,
                      decoder_attention_mask=dec_mask, use_cache=False)
    rows, cols = zip(*pos)
    h = out.last_hidden_state[list(rows), list(cols)]
    head = model.proj_out
    logits = F.linear(h.float(), head.weight.float(), None if head.bias is None else head.bias.float())
    loss = F.cross_entropy(logits, torch.tensor(tgt))
    loss.backward()
    bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    n_grads = sum(p.grad is not None for p in model.parameters())
    assert torch.isfinite(loss), f"loss {loss}"
    assert not bad, f"non-finite grads: {bad[:5]}"
    assert n_grads > 0, "no gradients"
    return {"loss": float(loss.detach()), "params": sum(p.numel() for p in model.parameters()), "params_with_grad": n_grads}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--requirements", default=None, help="pins to compare against (default: next to this file or repo root)")
    ap.add_argument("--skip-missing", action="store_true", help="do not fail on packages that are not installed")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # behave the same on a GPU host as on the runner
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    t0 = time.time()
    req = Path(args.requirements) if args.requirements else find_requirements()
    pins = parse_pins(req) if req else {}
    print(f"python {sys.version.split()[0]}  requirements: {req or 'not found (versions not compared)'}")
    problems = check_imports(pins, args.skip_missing)

    import torch
    print(f"torch {torch.__version__}  cuda built {torch.version.cuda}  arch {torch.cuda.get_arch_list()}  "
          f"cuda available {torch.cuda.is_available()}")
    for name, fn in (("audio", audio_roundtrip), ("model", tiny_forward_backward)):
        try:
            print(f"  {name}: {fn()}")
        except Exception as e:
            problems.append(f"{name}: {type(e).__name__}: {e}")
    print(f"done in {time.time() - t0:.1f} s")
    if problems:
        print("SMOKE FAILED:\n  " + "\n  ".join(problems))
        return 1
    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
