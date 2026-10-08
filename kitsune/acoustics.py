"""Per-row acoustic effects of the train loader (kitsune.trainset's acoustic steps; DECISIONS H12): room echo, volume
and codecs. Each returns a float32 row of the input's length, time-aligned with it, so a CTC row's frame targets and
an AED row's tokens - the teacher's on the clean audio - stay its targets. Imported by the loader's workers: numpy,
scipy, soundfile and soxr only (all in requirements-train.txt; the codecs are libsndfile's, in memory).

  reverb  the row convolved with a room impulse response whose direct path is its first sample (tools/
          build_rir_bank.py cuts the pre-delay), the tail past the row's end dropped, the row's voiced level kept
  gain    the row scaled by a drawn number of dB and clipped at full scale: a quiet or an overdriven recording
  codec   the row through a real encoder and back: "mp3" (LAME via libsndfile at a compression level drawn from
          MP3_LEVELS: ~100-30 kbit/s), "gsm" (GSM 6.10 at 8 kHz: a 2G phone), "ulaw8k" (G.711 mu-law at 8 kHz: a landline), and the
          costlier "vorbis" and "opus" (not in the default set: 5-17 ms of CPU per audio second against 1-4)
"""
import io

import numpy as np

SR = 16000
CODECS = ("mp3", "gsm", "ulaw8k", "vorbis", "opus")
DEFAULT_CODECS = ("mp3", "gsm", "ulaw8k")
MP3_LEVELS = (0.3, 0.9)  # libsndfile's MP3 compression level: 0.3 ~ 100 kbit/s, 0.9 ~ 30 kbit/s at 16 kHz
CLIP = 1.0  # full scale


def _voiced_power(w: np.ndarray) -> float:
    from kitsune.noise_bank import voiced_power

    return voiced_power(w)


def reverb(wave: np.ndarray, rir: np.ndarray) -> np.ndarray | None:
    """wave convolved with rir (its direct path at index 0), cut to len(wave) and scaled back to wave's voiced power;
    None for a silent row or RIR."""
    from scipy.signal import oaconvolve

    w = np.asarray(wave, dtype=np.float32)
    h = np.asarray(rir, dtype=np.float32)
    p0 = _voiced_power(w)
    if p0 < 1e-10 or not len(h) or not np.any(h):
        return None
    y = oaconvolve(w, h)[:len(w)].astype(np.float32)
    p1 = _voiced_power(y)
    if p1 < 1e-12:
        return None
    return y * np.float32((p0 / p1) ** 0.5)


def gain(wave: np.ndarray, db: float) -> tuple[np.ndarray, bool]:
    """wave scaled by db dB and clipped at +-CLIP; (the row, whether any sample clipped)."""
    y = np.asarray(wave, dtype=np.float32) * np.float32(10.0 ** (db / 20.0))
    clipped = bool(np.abs(y).max(initial=0.0) > CLIP)
    return (np.clip(y, -CLIP, CLIP) if clipped else y), clipped


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    import soxr

    return soxr.resample(x, sr_in, sr_out, quality="HQ").astype(np.float32)


def _roundtrip(x: np.ndarray, sr: int, fmt: str, subtype: str, **kw) -> np.ndarray:
    import soundfile as sf

    b = io.BytesIO()
    sf.write(b, np.clip(x, -CLIP, CLIP), sr, format=fmt, subtype=subtype, **kw)
    b.seek(0)
    y, _ = sf.read(b, dtype="float32", always_2d=True)
    return y[:, 0]


def _fit(y: np.ndarray, n: int) -> np.ndarray:
    """y cut or zero-padded to n samples (an 8 kHz codec's frame padding; libsndfile already trims the encoders'
    delay, measured: lag 0 for every codec here)."""
    if len(y) >= n:
        return y[:n].astype(np.float32, copy=False)
    return np.concatenate([y, np.zeros(n - len(y), np.float32)])


def codec(wave: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    """wave (16 kHz) through codec `name` and back, as long as wave."""
    x = np.asarray(wave, dtype=np.float32)
    n = len(x)
    if name == "mp3":
        level = float(rng.uniform(*MP3_LEVELS))
        y = _roundtrip(x, SR, "MP3", "MPEG_LAYER_III", compression_level=level)
    elif name == "vorbis":
        y = _roundtrip(x, SR, "OGG", "VORBIS", compression_level=float(rng.uniform(0.3, 0.9)))
    elif name == "opus":
        y = _roundtrip(x, SR, "OGG", "OPUS")
    elif name in ("gsm", "ulaw8k"):
        z = _resample(x, SR, 8000)
        z = _roundtrip(z, 8000, "WAV", "GSM610" if name == "gsm" else "ULAW")
        y = _resample(z, 8000, SR)
    else:
        raise ValueError(f"no codec {name!r} (one of {CODECS})")
    return _fit(y, n)


def available_codecs(names) -> tuple[list[str], dict[str, str]]:
    """The codecs of `names` that round-trip on this machine's libsndfile (a 0.5 s tone), and why the others do not."""
    t = np.arange(SR // 2) / SR
    x = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    ok, bad = [], {}
    for name in names:
        try:
            y = codec(x, name, np.random.default_rng(0))
            if len(y) != len(x) or not np.isfinite(y).all():
                raise ValueError(f"returned {len(y)} samples for {len(x)}")
            ok.append(name)
        except Exception as e:  # noqa: BLE001  (a libsndfile built without the encoder)
            bad[name] = f"{type(e).__name__}: {e}"[:200]
    return ok, bad
