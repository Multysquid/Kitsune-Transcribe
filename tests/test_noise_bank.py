"""The background bank (kitsune/noise_bank.py) and its builder (tools/build_noise_bank.py): a MUSAN-shaped corpus in
the real layout (music folders with ANNOTATIONS, noise folders, speech left out), the bank's index and audio, the
sha256 and size checks, segments drawn across clips, and the mix at the drawn SNR against the row's voiced power.
CPU only, synthetic audio."""
import importlib.util
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import soundfile as sf  # noqa: E402

from kitsune import noise_bank as NB  # noqa: E402

SR = NB.SR


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"kitsune_tool_{name}", ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tone(s: float, f: float, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(s * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


@pytest.fixture(scope="module")
def musan(tmp_path_factory):
    """music/fma (2 files: one with vocals), music/rfm (1, without an ANNOTATIONS line: left out), noise/free-sound
    (2), speech/librivox (1: never taken)."""
    root = tmp_path_factory.mktemp("musan") / "musan"
    files = {"music/fma/music-fma-0000.wav": tone(1.0, 220), "music/fma/music-fma-0001.wav": tone(0.5, 330),
             "music/rfm/music-rfm-0000.wav": tone(0.7, 440), "noise/free-sound/noise-free-sound-0000.wav": tone(0.3, 880),
             "noise/free-sound/noise-free-sound-0001.wav": tone(0.2, 990),
             "speech/librivox/speech-librivox-0000.wav": tone(0.4, 150)}
    for rel, x in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), x, SR, subtype="PCM_16")
    (root / "music/fma/ANNOTATIONS").write_text("music-fma-0000 Classical N Somebody\nmusic-fma-0001 Pop Y Someone\n",
                                                encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def bank_dir(musan, tmp_path_factory):
    out = tmp_path_factory.mktemp("bank") / "bank"
    assert load_tool("build_noise_bank").main(["--musan", str(musan), "--out", str(out)]) == 0
    return out


def test_the_builder_takes_instrumental_music_and_noise_only(bank_dir):
    info = json.loads((bank_dir / NB.INDEX).read_text(encoding="utf-8"))
    assert [c["id"] for c in info["clips"]] == ["music/fma/music-fma-0000", "noise/free-sound/noise-free-sound-0000",
                                                "noise/free-sound/noise-free-sound-0001"]
    assert [c["length"] for c in info["clips"]] == [SR, int(0.3 * SR), int(0.2 * SR)]
    assert [c["offset"] for c in info["clips"]] == [0, SR, SR + int(0.3 * SR)]
    a = np.load(bank_dir / NB.AUDIO)
    assert a.dtype == np.int16 and len(a) == info["samples"] == SR + int(0.5 * SR)
    assert info["audio_sha256"] == NB.sha256_file(bank_dir / NB.AUDIO) and info["licence"] == "CC BY 4.0"
    assert np.allclose(a[:SR] / 32768.0, tone(1.0, 220), atol=1e-4)


def test_the_music_cap_takes_whole_clips_in_a_fixed_order(musan, tmp_path):
    """--music-hours: whole music clips in the sha256 order of their ids up to the cap, every noise clip."""
    out = tmp_path / "capped"
    assert load_tool("build_noise_bank").main(["--musan", str(musan), "--out", str(out),
                                               "--music-hours", str(0.5 / 3600)]) == 0
    info = json.loads((out / NB.INDEX).read_text(encoding="utf-8"))
    assert [c["kind"] for c in info["clips"]] == ["noise", "noise"] and info["music_hours_cap"] == 0.5 / 3600


def test_load_checks_the_pin_and_the_size(bank_dir, tmp_path):
    sha = NB.sha256_file(bank_dir / NB.INDEX)
    bank = NB.NoiseBank.load(bank_dir, expect_sha256=sha)
    assert bank.info["index_sha256"] == sha and bank.hours == pytest.approx(1.5 / 3600)
    with pytest.raises(ValueError, match="another noise bank"):
        NB.NoiseBank.load(bank_dir, expect_sha256="0" * 64)
    short = tmp_path / "short"
    short.mkdir()
    (short / NB.INDEX).write_bytes((bank_dir / NB.INDEX).read_bytes())
    np.save(short / NB.AUDIO, np.zeros(10, np.int16))
    with pytest.raises(ValueError, match="bytes, the index says"):
        NB.NoiseBank.load(short)


def test_segments_cover_any_length_from_the_clips(bank_dir):
    """A segment is clip audio from drawn positions, clip after clip until filled; deterministic for a seed; the bank
    pickles without its memory map."""
    bank = NB.NoiseBank.load(bank_dir)
    audio = np.load(bank_dir / NB.AUDIO).astype(np.float32) / 32768.0
    for n in (100, SR // 2, 3 * SR):
        seg = bank.segment(n, np.random.default_rng(n))
        assert seg.dtype == np.float32 and len(seg) == n
        assert np.array_equal(seg, bank.segment(n, np.random.default_rng(n)))
        # every 10 samples of it are bank audio somewhere (the clips are distinct tones)
        assert any(np.array_equal(seg[:10], audio[i:i + 10]) for i in range(len(audio) - 10))
    again = pickle.loads(pickle.dumps(bank))
    assert again._mm is None and len(again.segment(50, np.random.default_rng(0))) == 50


def test_the_background_hits_its_snr_on_the_voiced_frames():
    """The gain puts the row's voiced power (pauses left out) the drawn SNR above the background's; the row itself is
    never written; a silent row or background is left alone."""
    speech = np.concatenate([tone(1.0, 300), np.zeros(SR, np.float32), tone(1.0, 300)])
    bg = 0.05 * np.random.default_rng(1).standard_normal(len(speech)).astype(np.float32)
    before = speech.copy()
    out, info = NB.add_background(speech, bg, np.random.default_rng(2), (5.0, 5.0))
    assert np.array_equal(speech, before) and info["snr_db"] == 5.0
    added = out - speech
    p_bg = float(np.mean(added ** 2))
    assert 10 * np.log10(NB.voiced_power(speech) / p_bg) == pytest.approx(5.0, abs=0.01)
    assert NB.voiced_power(speech) == pytest.approx(0.045, rel=0.01)  # the tone's power, not halved by the pause
    assert NB.add_background(np.zeros(SR, np.float32), bg[:SR], np.random.default_rng(0), (0, 10)) is None
    assert NB.add_background(speech, np.zeros(len(speech), np.float32), np.random.default_rng(0), (0, 10)) is None


def test_load_holds_the_audio_to_the_sha256_its_index_records(bank_dir, tmp_path):
    """The pin covers the audio through the index: an audio.npy of the right size but other samples is refused (a stale
    or broken pull would otherwise be mixed under the rows); check_audio=False skips the hash."""
    other = tmp_path / "other"
    other.mkdir()
    (other / NB.INDEX).write_bytes((bank_dir / NB.INDEX).read_bytes())
    a = np.load(bank_dir / NB.AUDIO)
    np.save(other / NB.AUDIO, a[::-1].copy())  # the same size, other samples
    with pytest.raises(ValueError, match="not the audio its index records"):
        NB.NoiseBank.load(other)
    assert NB.NoiseBank.load(other, check_audio=False).hours == pytest.approx(1.5 / 3600)


def test_segment_of_nothing_and_near_silent_stretches(bank_dir, tmp_path):
    """segment(0) is an empty array; background_segment draws again past a stretch quieter than BG_MIN_POWER (-60
    dBFS), and gives None for a bank with nothing louder."""
    bank = NB.NoiseBank.load(bank_dir)
    assert bank.segment(0, np.random.default_rng(0)).shape == (0,)
    seg = NB.background_segment(bank, SR // 4, np.random.default_rng(1))
    assert seg is not None and float(np.mean(seg ** 2)) >= NB.BG_MIN_POWER
    quiet = tmp_path / "quiet"
    quiet.mkdir()
    np.save(quiet / NB.AUDIO, np.ones(SR, np.int16))  # one LSB: about -90 dBFS
    info = dict(version=NB.VERSION, sr=SR, audio_bytes=(quiet / NB.AUDIO).stat().st_size, samples=SR,
                clips=[dict(id="noise/q/a", kind="noise", offset=0, length=SR)])
    (quiet / NB.INDEX).write_text(json.dumps(info), encoding="utf-8")
    assert NB.background_segment(NB.NoiseBank.load(quiet), SR // 2, np.random.default_rng(0)) is None
