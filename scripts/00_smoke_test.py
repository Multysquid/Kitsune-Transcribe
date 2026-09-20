"""Smoke test: load Cohere Transcribe teacher, count params per block, transcribe one clip, report VRAM."""
import time
import torch
import numpy as np
from transformers import AutoProcessor, CohereAsrForConditionalGeneration

MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"


def count(m):
    return sum(p.numel() for p in m.parameters())


def main():
    dev = "cuda"
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = CohereAsrForConditionalGeneration.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(dev).eval()
    print(f"loaded in {time.time()-t0:.1f}s")

    cfg = model.config
    print("\n== architecture ==")
    print(f"encoder: {cfg.encoder_config.num_hidden_layers} layers, d={cfg.encoder_config.hidden_size}, "
          f"ffn={cfg.encoder_config.intermediate_size}, subsample={cfg.encoder_config.subsampling_factor}x")
    print(f"decoder: {cfg.num_hidden_layers} layers, d={cfg.hidden_size}, ffn={cfg.intermediate_size}, vocab={cfg.vocab_size}")
    print("\n== param counts ==")
    total = count(model)
    for name, mod in model.named_children():
        print(f"  {name:20s} {count(mod)/1e6:8.1f}M  ({100*count(mod)/total:.1f}%)")
    for name, mod in model.model.named_children():
        print(f"  model.{name:14s} {count(mod)/1e6:8.1f}M")
    print(f"  {'TOTAL':20s} {total/1e6:8.1f}M")
    print(f"weights VRAM: {torch.cuda.memory_allocated()/2**30:.2f} GiB")

    print("\n== tokenizer ==")
    tok = processor.tokenizer
    print(f"vocab size: {len(tok)}, specials: {tok.all_special_tokens}")
    sample = "今日はいい天気ですね。"
    ids = tok(sample).input_ids
    print(f"'{sample}' -> {len(ids)} tokens: {tok.convert_ids_to_tokens(ids)}")

    # synthetic 10 s clip so the test needs no dataset; real audio comes in the download step
    sr = processor.feature_extractor.sampling_rate
    audio = (0.01 * np.random.randn(sr * 10)).astype(np.float32)
    inputs = processor(audio, language="ja", sampling_rate=sr, return_tensors="pt").to(dev)
    inputs = {k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v)
              for k, v in inputs.items() if torch.is_tensor(v)}
    print(f"\nfeature keys: {list(inputs.keys())}, input shape: {inputs[next(iter(inputs))].shape}")

    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64)
    print(f"generate on 10 s noise: {time.time()-t0:.2f}s -> {processor.batch_decode(out, skip_special_tokens=True)}")

    with torch.inference_mode():
        enc = model.model.encoder(input_features=inputs["input_features"], attention_mask=inputs["attention_mask"])
    if enc is not None:
        h = enc.last_hidden_state
        print(f"encoder output: {tuple(h.shape)}  (frames per second ≈ {h.shape[1]/10:.1f})")

    print(f"\npeak VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB of {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")


if __name__ == "__main__":
    main()
