"""Log-mel features on the training device, bit-for-bit the teacher's, plus SpecAugment.

The teacher's targets were produced from the HF `CohereAsrFeatureExtractor`. The student must see exactly the same
features, or it is distilled against inputs the teacher never saw. The HF extractor is a CPU Python loop (dither)
followed by torch ops, and it measured ~360 audio-s per core-second - too slow to feed an A100 from a handful of
vCPUs, and its `device=` argument cannot move the waveform-side steps to the GPU. `LogMel` redoes every step with the
same torch ops in the same order, so on CPU it reproduces HF bitwise and on CUDA it differs only by FFT/matmul
rounding:

  1. dither 1e-5 * N(0,1), the generator re-seeded per utterance with its valid sample count (batch-invariant)
  2. pre-emphasis 0.97, padding re-zeroed
  3. |STFT|^2: n_fft 512, hop 160, win 400, Hann(periodic=False), center=True, constant (zero) padding
  4. 128 Slaney mel bins (the HF extractor's own filterbank tensor), log(x + 2^-24)
  5. per-utterance per-bin mean/std over the valid frames (floor(samples/160); the last, partial STFT frame is
     invalid), std with N-1, (x - mean) / (std + 1e-5); padded frames zeroed

The output layout is what `model.model(input_features=..., attention_mask=...)` takes: (B, T, 128) float32 with
T = max_samples // 160 + 1, and a (B, T) bool mask.

Utterances must be <= 30 s: above that the HF extractor splits the audio at energy minima, and the teacher pass
skipped such utterances anyway.

SpecAugment runs after normalisation (mask value 0 = the per-bin mean), on the device, and only inside valid frames.
"""
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from kitsune.audio import TARGET_SR

HF_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"
MAX_SAMPLES = 30 * TARGET_SR  # HF fast path: max_audio_clip_s (35) - overlap_chunk_second (5)
LOG_ZERO_GUARD = 2**-24
NORM_EPS = 1e-5


def dither_noise(n: int, dither: float = 1e-5) -> torch.Tensor:
    """The exact dither HF adds to an utterance of `n` valid samples: a CPU generator seeded with `n`.

    Public so a DataLoader worker can pre-dither on CPU (then run `LogMel` with `dither=0`) if the main process is
    ever the bottleneck; measured cost is ~3.6 ms per 30 s of audio."""
    g = torch.Generator()
    g.manual_seed(n)
    return dither * torch.randn(n, dtype=torch.float32, generator=g)


def device_dither_noise(ns: list[int], dither: float, device) -> torch.Tensor:
    """LogMel's dither with exact_dither=False: the same scheme drawn on `device` (one generator there, re-seeded with
    each utterance's valid sample count), concatenated in row order, utterances of 0 samples skipped. On CUDA the
    numbers differ from HF's; on CPU it is bitwise torch.cat([dither_noise(n, dither) for n in ns]), which is how a
    CPU test covers the branch every training step on the box uses."""
    g = torch.Generator(device=device)
    parts = []
    for n in ns:
        if n > 0:
            g.manual_seed(n)
            parts.append(dither * torch.randn(n, dtype=torch.float32, device=device, generator=g))
    return torch.cat(parts)


def pad_waves(waves: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """list of 1-D float32 arrays -> (wave (B, S) zero-padded, lengths (B,) int64)."""
    lengths = torch.tensor([len(w) for w in waves], dtype=torch.int64)
    out = torch.zeros(len(waves), int(lengths.max()), dtype=torch.float32)
    for i, w in enumerate(waves):
        out[i, : len(w)] = torch.as_tensor(w, dtype=torch.float32)
    return out, lengths


def feature_lengths(lengths: torch.Tensor, n_fft: int = 512, hop_length: int = 160) -> torch.Tensor:
    """Valid frames per utterance, exactly as HF computes them (== samples // hop for n_fft even)."""
    return torch.floor_divide(lengths + n_fft // 2 * 2 - n_fft, hop_length)


class LogMel(nn.Module):
    """GPU/CPU twin of the HF CohereAsr feature extractor. See the module docstring for the steps.

    `exact_dither=True` draws the dither on CPU exactly as HF does (per-utterance generator seeded by the valid
    length) and copies it to the device. `False` draws it on the waveform's device with the same seeding scheme:
    deterministic and batch-invariant, statistically identical, but not the same numbers on CUDA."""

    def __init__(self, mel_filters: torch.Tensor | None = None, *, n_fft: int = 512, hop_length: int = 160,
                 win_length: int = 400, preemphasis: float | None = 0.97, dither: float = 1e-5,
                 sampling_rate: int = TARGET_SR, n_mels: int = 128, exact_dither: bool = True):
        super().__init__()
        if mel_filters is None:
            # same constructor call as the pretrained extractor -> the same librosa filterbank, no network needed
            from transformers import CohereAsrFeatureExtractor

            mel_filters = CohereAsrFeatureExtractor(feature_size=n_mels, sampling_rate=sampling_rate, n_fft=n_fft,
                                                    hop_length=hop_length, win_length=win_length).mel_filters
        assert mel_filters.shape == (n_mels, n_fft // 2 + 1), mel_filters.shape
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length
        self.preemphasis, self.dither, self.exact_dither = preemphasis, dither, exact_dither
        self.sampling_rate = sampling_rate
        self.register_buffer("mel_filters", mel_filters.detach().to(torch.float32).clone(), persistent=False)
        self.register_buffer("window", torch.hann_window(win_length, periodic=False), persistent=False)

    @classmethod
    def from_feature_extractor(cls, fe, **kw) -> "LogMel":
        """Copy every setting (and the filterbank tensor itself) from an HF CohereAsrFeatureExtractor instance."""
        return cls(fe.mel_filters, n_fft=fe.n_fft, hop_length=fe.hop_length, win_length=fe.win_length,
                   preemphasis=fe.preemphasis, dither=getattr(fe, "dither", 0.0), sampling_rate=fe.sampling_rate,
                   n_mels=fe.feature_size, **kw)

    @classmethod
    def from_pretrained(cls, path_or_repo: str = HF_MODEL_ID, **kw) -> "LogMel":
        """From a saved processor (the teacher repo or a student dir written by save_student)."""
        from transformers import AutoFeatureExtractor

        return cls.from_feature_extractor(AutoFeatureExtractor.from_pretrained(path_or_repo), **kw)

    def _dither(self, wave: torch.Tensor, lengths: list[int], valid: torch.Tensor) -> torch.Tensor:
        S = wave.shape[1]
        ns = [min(n, S) for n in lengths]
        if self.exact_dither or wave.device.type == "cpu":
            flat = torch.cat([dither_noise(n, self.dither) for n in ns if n > 0])
            flat = flat.to(wave.device, non_blocking=True)
        else:
            flat = device_dither_noise(ns, self.dither, wave.device)
        # row-major boolean scatter == utterance i's first n_i samples, in order; x + 0 leaves the padding exact
        return wave + torch.zeros_like(wave).masked_scatter_(valid, flat)

    @torch.no_grad()
    def forward(self, wave: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wave (B, S) float, lengths (B,) valid samples -> feats (B, T, n_mels) float32, feat_mask (B, T) bool.

        Samples beyond each length are ignored (re-zeroed), and S is cut to max(lengths) so T matches HF's
        'longest' padding exactly."""
        with torch.autocast(device_type=wave.device.type, enabled=False):
            len_list = [int(n) for n in lengths.tolist()]  # one host sync per batch; the dither loop needs them
            max_len = max(len_list)
            if min(len_list) <= 0:
                raise ValueError(f"empty utterance in batch (lengths {len_list})")
            if max_len > MAX_SAMPLES:
                raise ValueError(f"utterance of {max_len} samples > {MAX_SAMPLES}: HF would chunk it; not supported")
            dev = wave.device
            lengths = lengths.to(device=dev, dtype=torch.int64)
            wave = wave[:, :max_len].to(torch.float32)
            valid = torch.arange(max_len, device=dev)[None, :] < lengths[:, None]
            wave = wave.masked_fill(~valid, 0.0)  # HF pads with 0.0; callers may leave anything in the padding

            if self.dither > 0:
                wave = self._dither(wave, len_list, valid)
            if self.preemphasis is not None:
                wave = torch.cat([wave[:, :1], wave[:, 1:] - self.preemphasis * wave[:, :-1]], dim=1)
                wave = wave.masked_fill(~valid, 0.0)

            stft = torch.stft(wave, self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                              window=self.window, return_complex=True, pad_mode="constant")
            # HF takes |X| then squares it; kept literally for bitwise parity
            mag = torch.view_as_real(stft)
            mag = torch.sqrt(mag.pow(2).sum(-1)).pow(2)
            mel = torch.log(self.mel_filters @ mag + LOG_ZERO_GUARD).permute(0, 2, 1)  # (B, T, n_mels)

            flen = feature_lengths(lengths, self.n_fft, self.hop_length)
            mask = torch.arange(mel.shape[1], device=dev)[None, :] < flen[:, None]
            m = mask.unsqueeze(-1)
            # HF divides by N and N-1 unguarded (NaN below 2 frames = 0.02 s); clamping only changes that case
            n = flen.clamp_min(1).unsqueeze(-1)
            n1 = (flen - 1).clamp_min(1).unsqueeze(-1)
            masked = mel * m
            mean = (masked.sum(dim=1) / n).unsqueeze(1)
            var = ((masked - mean) ** 2 * m).sum(dim=1) / n1
            std = torch.sqrt(var).unsqueeze(1)
            feats = (mel - mean) / (std + NORM_EPS)
            feats = feats * m
        return feats, mask


@lru_cache(maxsize=2)
def load_hf_processor(path_or_repo: str = HF_MODEL_ID):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(path_or_repo)


def hf_reference(waves: list[np.ndarray], dither: float | None = None,
                 path_or_repo: str = HF_MODEL_ID) -> tuple[torch.Tensor, torch.Tensor]:
    """Featurise with the HF processor exactly as scripts/02_teacher_pass.py does (for parity tests).
    `dither` overrides the extractor's dither on a copy (0.0 = off)."""
    import copy

    proc = load_hf_processor(path_or_repo)
    waves = [np.asarray(w, dtype=np.float32) for w in waves]
    if dither is None:
        out = proc(waves, language="ja", punctuation=True, sampling_rate=TARGET_SR, return_tensors="pt")
    else:  # the processor's audio path is just its feature extractor; call a copy with the dither changed
        fe = copy.copy(proc.feature_extractor)
        fe.dither = dither
        out = fe(waves, sampling_rate=TARGET_SR, return_tensors="pt")
    return out["input_features"], out["attention_mask"]


class SpecAugment(nn.Module):
    """Frequency and time masking on normalised log-mels, applied on the device, inside valid frames only.

    Per utterance: `freq_masks` bands of width ~U{0..freq_width}; n ~ U{time_masks_min..time_masks_max} time masks of
    width ~U{0..max(1, floor(time_width * valid_frames))}, each placed wholly inside the valid frames. All randomness
    comes from the `generator` argument, drawn in one call on the generator's device, so a batch's masks are a
    function of the generator's seed and the batch's shape only, independent of the global RNG (the trainer seeds it
    per micro-batch from (specaug.seed, step, micro-batch index): scripts/04_distill.py specaug_seed). Returns the
    masked features and the fraction of each utterance's valid frames covered by a time mask (for logging)."""

    def __init__(self, freq_masks: int = 2, freq_width: int = 27, time_masks_min: int = 2, time_masks_max: int = 5,
                 time_width: float = 0.05):
        super().__init__()
        assert 0 <= time_masks_min <= time_masks_max and freq_masks >= 0 and freq_width >= 0 and time_width >= 0
        self.freq_masks, self.freq_width = freq_masks, freq_width
        self.time_masks_min, self.time_masks_max, self.time_width = time_masks_min, time_masks_max, time_width

    def forward(self, feats: torch.Tensor, feat_mask: torch.Tensor,
                generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, F = feats.shape
        dev = feats.device
        fm, tm = self.freq_masks, self.time_masks_max
        u = torch.rand(B, 2 * fm + 1 + 2 * tm, generator=generator, device=generator.device).to(dev)
        u_fw, u_fs, u_n, u_tw, u_ts = u.split([fm, fm, 1, tm, tm], dim=1)

        def uniform_int(v: torch.Tensor, hi: torch.Tensor | int) -> torch.Tensor:  # U{0..hi}, v ~ U[0,1)
            return torch.minimum(torch.floor(v * (hi + 1)), torch.as_tensor(hi, dtype=v.dtype, device=dev))

        lengths = feat_mask.sum(dim=1)  # (B,)
        fw = uniform_int(u_fw, min(self.freq_width, F))
        fs = uniform_int(u_fs, F - fw)
        f = torch.arange(F, device=dev, dtype=u.dtype)
        freq_mask = ((f >= fs[..., None]) & (f < (fs + fw)[..., None])).any(dim=1)  # (B, F)

        n_t = self.time_masks_min + uniform_int(u_n[:, 0], self.time_masks_max - self.time_masks_min)
        L = lengths.to(u.dtype)[:, None]
        tw_max = torch.floor(self.time_width * L).clamp_min(1)
        tw = torch.minimum(uniform_int(u_tw, tw_max), L)
        ts = uniform_int(u_ts, (L - tw).clamp_min(0))
        active = torch.arange(tm, device=dev) < n_t[:, None]
        t = torch.arange(T, device=dev, dtype=u.dtype)
        time_mask = ((t >= ts[..., None]) & (t < (ts + tw)[..., None]) & active[..., None]).any(dim=1)  # (B, T)
        time_mask &= feat_mask

        cells = (time_mask[:, :, None] | freq_mask[:, None, :]) & feat_mask[:, :, None]
        masked_frac = time_mask.sum(dim=1).to(torch.float32) / lengths.clamp_min(1).to(torch.float32)
        return feats.masked_fill(cells, 0.0), masked_frac
