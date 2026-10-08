"""The acoustic steps (DECISIONS H12): kitsune/acoustics.py's room echo, volume and codecs, the banks' kinds and RIRs,
and kitsune.trainset's chain on rows - lengths and alignment kept (a CTC row's frame targets stay valid), levels as
drawn, every step deterministic for a stream, the targets never touched. CPU only, synthetic audio."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import make_noise_bank, make_rir_bank  # noqa: E402
from kitsune import acoustics as A  # noqa: E402
from kitsune import noise_bank as NB  # noqa: E402
from kitsune import trainset as T  # noqa: E402

SR = 16000


def speechlike(seconds: float, seed: int, level: float = 0.1) -> np.ndarray:
    """Filtered noise in 0.2-0.5 s bursts with pauses and a gliding tone: structure for cross-correlation."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = np.convolve(rng.standard_normal(n), np.ones(6) / 6, mode="same")
    t = np.arange(n) / SR
    x += 0.5 * np.sin(2 * np.pi * (200 + 300 * t / seconds) * t)
    env = np.zeros(n)
    i = 0
    while i < n:
        on, off = int(rng.uniform(0.2, 0.5) * SR), int(rng.uniform(0.05, 0.25) * SR)
        env[i:i + on] = 1.0
        i += on + off
    return (level * x * env / np.abs(x).max()).astype(np.float32)


def lag(a: np.ndarray, b: np.ndarray) -> int:
    from scipy.signal import correlate

    n = min(len(a), len(b))
    c = correlate(b[:n], a[:n], mode="full", method="fft")
    return int(np.argmax(c)) - (n - 1)


def test_reverb_keeps_length_level_and_alignment():
    """The echo spreads the row over the room's tail but keeps its first sample where it was (lag 0), its length and
    its voiced level; a silent row or RIR gives None."""
    x = speechlike(3.0, 1)
    h = np.zeros(int(0.4 * SR), np.float32)
    h[0] = 1.0
    # a room's tail at a direct-to-reverberant ratio of about +2 dB (real rooms: about -5 to +10 dB)
    h[1:] = 0.05 * np.random.default_rng(2).standard_normal(len(h) - 1) * np.exp(-np.arange(1, len(h)) / (0.08 * SR))
    y = A.reverb(x, h)
    assert y.dtype == np.float32 and len(y) == len(x) and not np.allclose(y, x)
    assert lag(x, y) == 0
    assert NB.voiced_power(y) == pytest.approx(NB.voiced_power(x), rel=1e-3)
    assert A.reverb(np.zeros(SR, np.float32), h) is None and A.reverb(x, np.zeros(10, np.float32)) is None


def test_gain_scales_and_clips_at_full_scale():
    x = speechlike(1.0, 3, level=0.5)
    y, clipped = A.gain(x, -20.0)
    assert not clipped and np.allclose(y, x * 0.1, atol=1e-7)
    y, clipped = A.gain(x, 12.0)
    assert clipped and np.abs(y).max() == pytest.approx(1.0) and len(y) == len(x)


@pytest.mark.parametrize("name", A.DEFAULT_CODECS)
def test_a_codec_keeps_length_and_alignment(name):
    """Each default codec round-trips in memory to the row's length, time-aligned (lag 0: libsndfile trims the
    encoders' delay) and close to it - a CTC row's frames still hold their sounds."""
    x = speechlike(3.0, 4)
    y = A.codec(x, name, np.random.default_rng(5))
    assert y.dtype == np.float32 and len(y) == len(x) and np.isfinite(y).all() and not np.allclose(y, x)
    assert lag(x, y) == 0
    assert np.corrcoef(x, y)[0, 1] > 0.5


def test_codecs_available_here_and_unknown_ones():
    ok, bad = A.available_codecs(A.DEFAULT_CODECS)
    assert ok == list(A.DEFAULT_CODECS) and bad == {}
    ok, bad = A.available_codecs(["mp3", "aac"])
    assert ok == ["mp3"] and "aac" in bad
    with pytest.raises(ValueError, match="no codec 'aac'"):
        A.codec(speechlike(0.5, 1), "aac", np.random.default_rng(0))


def test_bank_kinds_and_rir_clips(tmp_path):
    """segment draws from the kinds asked for only; an RIR bank is float32 and draw_clip gives each room whole."""
    bank = NB.NoiseBank.load(make_noise_bank(tmp_path / "bg", seconds=4.0, kinds=("music", "speech", "song", "noise")))
    assert bank.has(("speech",)) and bank.has(T.BACKGROUND_KINDS) and not bank.has(("rir",))
    assert bank.kind_hours("speech") == pytest.approx(1.0 / 3600)
    audio = np.load(Path(bank.path) / NB.AUDIO).astype(np.float32) / 32768.0
    lo, n = int(bank.offsets[1]), int(bank.lengths[1])  # the speech clip
    for seed in range(5):
        seg = bank.segment(SR // 4, np.random.default_rng(seed), ("speech",))
        assert any(np.array_equal(seg[:20], audio[i:i + 20]) for i in range(lo, lo + n - 20))
    rirs = NB.NoiseBank.load(make_rir_bank(tmp_path / "rir", n=3))
    assert rirs.scale == 1.0 and rirs.has(("rir",)) and len(rirs.lengths) == 3
    firsts = {float(rirs.draw_clip(np.random.default_rng(s))[0]) for s in range(20)}
    assert firsts == {1.0}
    drawn = {int(np.searchsorted(rirs.cum, 0)) for _ in range(1)}  # (the clip index is internal; the tails differ:)
    tails = {round(float(np.abs(rirs.draw_clip(np.random.default_rng(s))[100:]).sum()), 3) for s in range(30)}
    assert len(tails) == 3 and drawn == {0}


def rows_of(waves):
    return [SimpleNamespace(wave=w.copy()) for w in waves]


def test_speech_goes_under_the_row_at_its_snr(tmp_path):
    """speech_p 1: every row gets voices - the micro-batch's other rows, or the bank's speech when it has none - at
    10-25 dB of voiced power below it, its length kept; a single row without a speech bank gets none."""
    waves = [speechlike(2.0 + i, 10 + i) for i in range(4)]
    a = T.Augment(seed=1, speech_p=1.0, speech_batch_p=1.0)
    rows = rows_of(waves)
    assert T.mix_speech(rows, waves, None, a, np.random.default_rng(0)) == 4
    for r, w in zip(rows, waves):
        added = r.wave - w
        assert len(r.wave) == len(w)
        assert 10.0 - 0.01 <= 10 * np.log10(NB.voiced_power(w) / NB.voiced_power(added)) <= 25.0 + 0.01
    bank = NB.NoiseBank.load(make_noise_bank(tmp_path / "bg", kinds=("speech", "noise")))
    one = rows_of(waves[:1])
    assert T.mix_speech(one, waves[:1], bank, T.Augment(seed=1, speech_p=1.0, speech_batch_p=0.0),
                        np.random.default_rng(1)) == 1
    assert not np.array_equal(one[0].wave, waves[0])
    alone = rows_of(waves[:1])
    assert T.mix_speech(alone, waves[:1], None, a, np.random.default_rng(1)) == 0  # nobody to talk behind it


def test_the_chain_counts_each_step_and_is_deterministic(tmp_path):
    """acoustic_chain: speech, reverb, background, gain, codec in that order, each counted; the same stream gives the
    same rows, another stream others; lengths kept; nothing on when the probabilities are 0."""
    bank = NB.NoiseBank.load(make_noise_bank(tmp_path / "bg", kinds=("music", "speech", "song", "noise")))
    rirs = NB.NoiseBank.load(make_rir_bank(tmp_path / "rir"))
    waves = [speechlike(1.5 + 0.5 * i, 20 + i) for i in range(5)]
    a = T.Augment(seed=1, speech_p=1.0, reverb_p=1.0, noise_p=1.0, gain_p=1.0, codec_p=1.0)
    rows = rows_of(waves)
    got = T.acoustic_chain(rows, waves, bank, rirs, a, np.random.default_rng(3))
    assert got["speech_mixed"] == got["reverbed"] == got["noised"] == got["gained"] == got["coded"] == 5
    assert set(got) == {"speech_mixed", "reverbed", "noised", "gained", "clipped", "coded"}
    assert all(len(r.wave) == len(w) and np.isfinite(r.wave).all() for r, w in zip(rows, waves))
    again = rows_of(waves)
    T.acoustic_chain(again, waves, bank, rirs, a, np.random.default_rng(3))
    assert all(np.array_equal(x.wave, y.wave) for x, y in zip(rows, again))
    other = rows_of(waves)
    T.acoustic_chain(other, waves, bank, rirs, a, np.random.default_rng(4))
    assert any(not np.array_equal(x.wave, y.wave) for x, y in zip(rows, other))
    off = rows_of(waves)
    assert T.acoustic_chain(off, waves, bank, rirs, T.Augment(seed=1), np.random.default_rng(3)) == dict(
        speech_mixed=0, reverbed=0, noised=0, gained=0, clipped=0, coded=0)
    assert all(np.array_equal(r.wave, w) for r, w in zip(off, waves))


def test_the_background_step_never_draws_speech(tmp_path):
    """noise_p draws music, noise and songs only: a bank holding nothing but speech gives no background."""
    bank = NB.NoiseBank.load(make_noise_bank(tmp_path / "bg", kinds=("speech", "speech")))
    rows = rows_of([speechlike(1.0, 1)])
    assert T.mix_background(rows, bank, T.Augment(seed=1, noise_p=1.0), np.random.default_rng(0)) == 0


def test_an_augmentation_needs_its_banks(tmp_path):
    with pytest.raises(ValueError, match="needs an RIR bank"):
        T._rir_of(T.Augment(seed=1, reverb_p=0.5), None)
    with pytest.raises(ValueError, match="noise bank with speech clips"):
        T._noise_of(T.Augment(seed=1, speech_p=0.5, speech_batch_p=0.5),
                    NB.NoiseBank.load(make_noise_bank(tmp_path / "bg")))
    assert T._noise_of(T.Augment(seed=1, speech_p=0.5, speech_batch_p=1.0), None) is None  # the micro-batch's own
    for bad in (dict(speech_talkers=(0, 2)), dict(speech_talkers=(3, 9)), dict(gain_db=(5, -5)),
                dict(speech_snr_db=(25, 10)), dict(codecs=("mp3", "aac")), dict(codec_p=0.5, codecs=()),
                dict(reverb_p=1.5)):
        with pytest.raises(ValueError, match="not an augmentation"):
            T.Augment(seed=1, **bad)
