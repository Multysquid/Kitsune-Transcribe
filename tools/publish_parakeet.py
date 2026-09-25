"""Prepare (and once, upload) the Parakeet teacher dir the label box downloads (laptop, CPU).

    python tools/publish_parakeet.py [--hf-dir cache/parakeet-tdt_ctc-0.6b-ja-hf] [--golden] [--upload]
           [--repo Multy123/kitsune-data] [--path-in-repo models/parakeet-tdt_ctc-0.6b-ja-hf]

1. asserts the converted config; 2. extracts the NeMo CTC head from the cached model_weights.ckpt into
ctc_head.safetensors; 3. CPU sanity check (CTC vs TDT greedy CER on 8 eval_jsut clips); 4. writes kitsune_model.json;
5. prints PARAKEET_FILES for kitsune/parakeet.py; 6. --golden writes kitsune/parakeet_golden.json;
7. --upload pushes the 8 files to the data repo (the user runs this once).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kitsune.parakeet import (  # noqa: E402
    BLANK, CTC_HEAD_FILE, DURATIONS, MODEL_JSON, NEMO_REPO, NEMO_REVISION, PARAKEET_FILES, PARAKEET_PATH, VOCAB,
    file_sha256,
)

NEMO_SHA256 = "a2ed2aab"  # prefix recorded by the bake-off; the full hash is taken from the cache when present
NEMO_FILE = "parakeet-tdt_ctc-0.6b-ja.nemo"
CKPT_MEMBER = "model_weights.ckpt"
HEAD_KEYS = ("ctc_decoder.decoder_layers.0.weight", "ctc_decoder.decoder_layers.0.bias")
HF_FILES = ("config.json", "generation_config.json", "model.safetensors", "processor_config.json", "tokenizer.json",
            "tokenizer_config.json")
CONVERTER = "transformers utils/convert_nemo_to_hf.py (main) + <blank>-as-pad patch for a vocab without <pad>"
TRANSFORMERS_VERSION = "5.13.1"
HEAD_METADATA = {"format": "pt"}  # one key only: safetensors orders metadata by a hash map (bytes would vary)
GOLDEN_SHARD = ROOT / "data" / "shards" / "eval_jsut" / "eval-00000.parquet"
GOLDEN_OUT = ROOT / "kitsune" / "parakeet_golden.json"
GOLDEN_ROWS = 32
SANITY_ROWS = 8
SANITY_MAX_CER = 0.2
DEFAULT_REPO = "Multy123/kitsune-data"


def config_problems(cfg: dict, gen: dict) -> list[str]:
    enc = cfg.get("encoder_config", {})
    want = [("model_type", cfg.get("model_type"), "parakeet_tdt"), ("vocab_size", cfg.get("vocab_size"), VOCAB),
            ("blank_token_id", cfg.get("blank_token_id"), BLANK), ("pad_token_id", cfg.get("pad_token_id"), BLANK),
            ("durations", cfg.get("durations"), list(DURATIONS)),
            ("max_symbols_per_step", cfg.get("max_symbols_per_step"), 10),
            ("encoder_config.hidden_size", enc.get("hidden_size"), 1024),
            ("encoder_config.num_hidden_layers", enc.get("num_hidden_layers"), 24),
            ("encoder_config.subsampling_factor", enc.get("subsampling_factor"), 8),
            ("generation_config.decoder_start_token_id", gen.get("decoder_start_token_id"), BLANK)]
    return [f"{k} = {got!r}, expected {exp!r}" for k, got, exp in want if got != exp]


def nemo_snapshot() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE
    return Path(HF_HUB_CACHE) / f"models--{NEMO_REPO.replace('/', '--')}" / "snapshots" / NEMO_REVISION


def load_nemo_state(snapshot: Path) -> dict:
    """The NeMo state dict at NEMO_REVISION: the extracted model_weights.ckpt if present, else the .nemo member."""
    import torch
    ckpt = snapshot / CKPT_MEMBER
    if ckpt.is_file():
        return torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)
    nemo = snapshot / NEMO_FILE
    if not nemo.is_file():
        raise SystemExit(f"neither {ckpt} nor {nemo} exists; download {NEMO_REPO}@{NEMO_REVISION} first")
    import io
    import tarfile
    with tarfile.open(nemo) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(CKPT_MEMBER))
        return torch.load(io.BytesIO(tf.extractfile(member).read()), map_location="cpu", weights_only=True)


def extract_head(state: dict) -> dict:
    """{weight (V,H) fp32, bias (V,) fp32} from the NeMo CTC decoder's k=1 Conv1d."""
    import torch
    w, b = state[HEAD_KEYS[0]], state[HEAD_KEYS[1]]
    if w.ndim != 3 or w.shape[0] != VOCAB or w.shape[2] != 1 or tuple(b.shape) != (VOCAB,):
        raise SystemExit(f"unexpected CTC head shapes {tuple(w.shape)} / {tuple(b.shape)}")
    return {"weight": w[:, :, 0].to(torch.float32).contiguous(), "bias": b.to(torch.float32).contiguous()}


def write_head(head: dict, path: Path) -> None:
    from safetensors.torch import save_file
    save_file({k: head[k] for k in ("weight", "bias")}, str(path), metadata=HEAD_METADATA)


def model_json(hf_dir: Path, nemo_sha: str) -> dict:
    files = {name: file_sha256(hf_dir / name) for name in sorted(HF_FILES + (CTC_HEAD_FILE,))}
    return {"source_repo": NEMO_REPO, "source_revision": NEMO_REVISION, "nemo_file": NEMO_FILE,
            "nemo_sha256": nemo_sha, "converter": CONVERTER, "transformers": TRANSFORMERS_VERSION,
            "ctc_head_source_keys": list(HEAD_KEYS), "ctc_head_file": CTC_HEAD_FILE, "files": files}


def write_json(path: Path, obj: dict) -> None:
    path.write_bytes((json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))


def nemo_sha256(snapshot: Path) -> str:
    nemo = snapshot / NEMO_FILE
    if nemo.is_file():
        full = file_sha256(nemo)
        if not full.startswith(NEMO_SHA256):
            raise SystemExit(f"{nemo} sha256 {full} does not start with {NEMO_SHA256}")
        return full
    return NEMO_SHA256


def read_rows(n: int):
    import pyarrow.parquet as pq
    from kitsune.audio import decode_audio
    t = pq.read_table(GOLDEN_SHARD, columns=["id", "audio", "text"]).slice(0, n)
    ids = t.column("id").to_pylist()
    waves = [decode_audio(a) for a in t.column("audio").to_pylist()]
    return ids, waves, t.column("text").to_pylist()


def decode(teacher, waves, batch: int = 8) -> list[dict]:
    out = []
    for i in range(0, len(waves), batch):
        fe = teacher.features(waves[i:i + batch])
        out += teacher.run_batch(fe["input_features"], fe.get("attention_mask"))
    return out


def char_cer(ref: str, hyp: str) -> float:
    """Plain character edit distance / len(ref)."""
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / max(1, len(ref))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", type=Path, default=ROOT / "cache" / "parakeet-tdt_ctc-0.6b-ja-hf")
    ap.add_argument("--golden", action="store_true")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--path-in-repo", default=PARAKEET_PATH)
    args = ap.parse_args(argv)
    d = args.hf_dir

    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8"))
    problems = config_problems(cfg, gen)
    if problems:
        raise SystemExit("refusing the converted config:\n  " + "\n  ".join(problems))

    snap = nemo_snapshot()
    write_head(extract_head(load_nemo_state(snap)), d / CTC_HEAD_FILE)
    write_json(d / MODEL_JSON, model_json(d, nemo_sha256(snap)))
    files = {name: file_sha256(d / name) for name in sorted(HF_FILES + (CTC_HEAD_FILE, MODEL_JSON))}

    import torch
    from kitsune.parakeet import ParakeetTeacher
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    teacher = ParakeetTeacher(d, device="cpu")
    n = GOLDEN_ROWS if args.golden else SANITY_ROWS
    ids, waves, refs = read_rows(n)
    utts = decode(teacher, waves)
    tdt, ctc = [u["hyp"] for u in utts], [u["ctc_hyp"] for u in utts]
    san = sum(char_cer(a, b) * max(1, len(a)) for a, b in zip(tdt[:SANITY_ROWS], ctc[:SANITY_ROWS]))
    san /= max(1, sum(max(1, len(a)) for a in tdt[:SANITY_ROWS]))
    print(f"sanity: CER(CTC greedy vs TDT greedy) over {SANITY_ROWS} eval_jsut clips = {san:.4f}")
    if san > SANITY_MAX_CER:
        raise SystemExit(f"the CTC head does not match this encoder (CER {san:.3f} > {SANITY_MAX_CER})")

    print("PARAKEET_FILES = {")
    for name, sha in files.items():
        print(f'    "{name}": "{sha}",')
    print("}")

    if args.golden:
        settings = {"device": "cpu", "dtype": "float32", "k_tdt": 8, "k_ctc": 8, "ctc_dense_thr": 0.95,
                    "max_symbols": 10, "rows": GOLDEN_ROWS, "shard": "eval_jsut/eval-00000.parquet"}
        write_json(GOLDEN_OUT, {"ids": ids, "tdt": tdt, "ctc": ctc, "settings": settings, "files_sha256": files})
        print(f"wrote {GOLDEN_OUT}")

    if args.upload:
        from huggingface_hub import HfApi
        HfApi().upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(d),
                              path_in_repo=args.path_in_repo, allow_patterns=sorted(files))
        print(f"uploaded {len(files)} files to {args.repo}:{args.path_in_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
