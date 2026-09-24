"""LogMel must reproduce the HF CohereAsr feature extractor (the teacher's input) and SpecAugment must stay inside
the valid frames and be reproducible from its generator. CPU only; the HF processor comes from the local cache."""
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from kitsune.features import LogMel, SpecAugment, hf_reference, load_hf_processor, pad_waves  # noqa: E402

TOL = 1e-4  # target on normalised features; on CPU the ops are identical, so the measured diff is 0.0
REAL_SHARD = ROOT / "data" / "shards" / "reazon_small" / "train-00000.parquet"


@pytest.fixture(scope="module")
def fe():
    try:
        return load_hf_processor().feature_extractor
    except Exception as e:  # noqa: BLE001 - not cached / offline
        pytest.skip(f"teacher processor not available offline: {e}")


def _compare(lm: LogMel, waves: list[np.ndarray], dither: float, wave=None, lengths=None) -> float:
    ref, ref_mask = hf_reference(waves, dither=dither)
    if wave is None:
        wave, lengths = pad_waves(waves)
    lm.dither = dither
    feats, mask = lm(wave, lengths)
    assert feats.dtype == torch.float32 and mask.dtype == torch.bool
    assert feats.shape == ref.shape and mask.shape == ref_mask.shape
    assert torch.equal(mask, ref_mask)
    diff = (feats - ref).abs().max().item()
    print(f"  dither={dither:g} shape={tuple(feats.shape)} max|diff|={diff:.3g}")
    assert diff < TOL
    return diff


def test_filterbank_is_the_hf_one(fe):
    assert torch.equal(LogMel().mel_filters, fe.mel_filters)
    lm = LogMel.from_feature_extractor(fe)
    assert (lm.n_fft, lm.hop_length, lm.win_length, lm.preemphasis, lm.dither) == (512, 160, 400, 0.97, 1e-5)


@pytest.mark.skipif(not REAL_SHARD.exists(), reason="real reazon_small shard not present")
def test_parity_real_clips(fe):
    import pyarrow.parquet as pq

    from kitsune.audio import decode_audio

    rg = pq.ParquetFile(REAL_SHARD).read_row_group(0, columns=["audio", "duration"])  # read-only, one row group
    durs = np.array(rg.column("duration").to_pylist())
    order = np.argsort(durs)
    pick = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]  # shortest, median, longest
    waves = [decode_audio(rg.column("audio")[i].as_py()) for i in pick]
    assert len({len(w) for w in waves}) == 3
    lm = LogMel.from_feature_extractor(fe)
    for dither in (1e-5, 0.0):
        _compare(lm, waves, dither)


def test_parity_synthetic(fe):
    rng = np.random.default_rng(0)
    sr = 16000
    t = np.arange(int(1.37 * sr)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440 * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)
    tone[4000:9000] = 0.0  # digital silence: the frames where dither decides the log-mel values
    waves = [
        tone,
        (0.1 * rng.standard_normal(int(0.3 * sr))).astype(np.float32),  # shortest utterance allowed
        (0.05 * rng.standard_normal(30 * sr)).astype(np.float32),  # exactly the 30 s limit
        (0.2 * rng.standard_normal(160 * 57)).astype(np.float32),  # length a multiple of the hop
        (0.2 * rng.standard_normal(160 * 57 + 159)).astype(np.float32),  # one sample short of the next frame
        np.zeros(sr, dtype=np.float32),  # all silence: normalisation of pure dither
    ]
    lm = LogMel.from_feature_extractor(fe)
    for dither in (1e-5, 0.0):
        _compare(lm, waves, dither)

    # garbage in the padding and a wider-than-needed buffer must not change anything
    wave, lengths = pad_waves(waves[:2])
    wide = torch.full((2, wave.shape[1] + 1234), 7.0)
    wide[:, : wave.shape[1]] = wave
    wide[1, lengths[1]:] = 3.0
    _compare(lm, waves[:2], 1e-5, wave=wide, lengths=lengths)


def test_batch_composition_invariance():
    rng = np.random.default_rng(1)
    waves = [(0.1 * rng.standard_normal(n)).astype(np.float32) for n in (16000, 5555, 23456)]
    lm = LogMel()
    feats, mask = lm(*pad_waves(waves))
    for i, w in enumerate(waves):
        f1, m1 = lm(*pad_waves([w]))
        T = f1.shape[1]
        assert torch.equal(m1[0], mask[i, :T]) and not mask[i, T:].any()
        # the dither is seeded by length, not batch position; only reduction order differs (measured ~1e-6)
        assert (f1[0] - feats[i, :T]).abs().max().item() < 1e-5


def test_fp32_under_autocast():
    rng = np.random.default_rng(2)
    wave, lengths = pad_waves([(0.1 * rng.standard_normal(n)).astype(np.float32) for n in (8000, 12000)])
    lm = LogMel()
    ref, _ = lm(wave, lengths)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out, _ = lm(wave, lengths)
    assert out.dtype == torch.float32 and torch.equal(out, ref)


def test_rejects_over_30s():
    with pytest.raises(ValueError):
        LogMel()(torch.zeros(1, 30 * 16000 + 1), torch.tensor([30 * 16000 + 1]))


def _feats(lengths: list[int], F: int = 128, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    T = max(lengths) + 7
    mask = torch.arange(T)[None, :] < torch.tensor(lengths)[:, None]
    feats = torch.randn(len(lengths), T, F, generator=g) * mask[..., None]
    return feats, mask


def test_specaug_valid_frames_and_reproducible():
    sa = SpecAugment()
    lengths = [30, 400, 1234, 3000]
    feats, mask = _feats(lengths)
    out1, frac1 = sa(feats, mask, torch.Generator().manual_seed(123))
    out2, frac2 = sa(feats, mask, torch.Generator().manual_seed(123))
    out3, _ = sa(feats, mask, torch.Generator().manual_seed(124))
    assert torch.equal(out1, out2) and torch.equal(frac1, frac2)
    assert not torch.equal(out1, out3)

    changed = out1 != feats
    assert not (changed & ~mask[..., None]).any()  # nothing outside the valid frames
    assert (out1[~mask] == 0).all()
    assert (out1[changed] == 0).all()  # mask value 0, everything else untouched

    for i, L in enumerate(lengths):
        z = out1[i, :L] == 0
        time_masked = z.all(dim=1)
        freq_masked = z.all(dim=0)
        assert freq_masked.sum() <= 2 * 27
        max_frames = 5 * max(1, int(0.05 * L))
        assert time_masked.sum() <= max_frames
        # every masked cell is explained by a masked frame or a masked bin (random data has no exact zeros)
        assert torch.equal(z, time_masked[:, None] | freq_masked[None, :])
        assert frac1[i].item() == pytest.approx(time_masked.sum().item() / L)
        assert 0.0 <= frac1[i].item() <= max_frames / L


def test_specaug_fraction_statistics():
    sa = SpecAugment()
    feats, mask = _feats([1000] * 64, F=128, seed=3)
    g = torch.Generator().manual_seed(7)
    fracs, bins = [], []
    for _ in range(8):
        out, frac = sa(feats, mask, g)
        fracs.append(frac)
        bins.append((out == 0).all(dim=1).sum(dim=1).float())
    frac = torch.cat(fracs)
    nbins = torch.cat(bins)
    # E[n]=3.5 masks of E[w]=25 frames of 1000 -> ~8.75% before overlaps; hard cap 5*50/1000
    assert 0.05 < frac.mean().item() < 0.10 and frac.max().item() <= 0.25
    # 2 bands of E[w]=13.5 bins -> ~27 bins before overlaps; hard cap 54
    assert 20 < nbins.mean().item() < 30 and nbins.max().item() <= 54


def test_specaug_off_is_identity():
    sa = SpecAugment(freq_masks=0, time_masks_min=0, time_masks_max=0)
    feats, mask = _feats([50, 80])
    out, frac = sa(feats, mask, torch.Generator().manual_seed(0))
    assert torch.equal(out, feats) and (frac == 0).all()
