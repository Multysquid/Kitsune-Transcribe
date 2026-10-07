"""A bank of background audio - music and noise without speech - that the train loader mixes under the speech
(kitsune.trainset's "background" step; scripts/04_distill.py augment.noise_*).

Why (the P-0.3B review's issue A, DECISIONS H8): on 30 s windows of stream audio - talk over game sound and music - the
Parakeet teacher outputs nothing and Cohere one short phrase, while each transcribes the same talk cut out alone. A
student that only ever hears clean speech inherits that. Mixed under a row (a joined row too: one stretch of
background under all its utterances and the gaps between them), with the targets the teacher made on the clean audio,
the background teaches the student to keep transcribing through it.

Layout (tools/build_noise_bank.py writes it; the box pulls the dir as a boxes.json extra dir):
  audio.npy    (N,) int16   every clip's 16 kHz mono samples, back to back
  index.json   {"version", "sr", "audio_bytes", "audio_sha256", "clips": [{"id", "kind", "offset", "length",
               "license"}], "source", ...} - the sha256 of this file is what augment.noise_bank_sha256 pins
Only numpy at import: the loader's workers import this module.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SR = 16000
INDEX, AUDIO = "index.json", "audio.npy"
VERSION = 1


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class NoiseBank:
    """An opened bank: its index (clips, offsets) in memory, its audio memory-mapped lazily in each process (picklable
    without the map, as the stores are)."""

    def __init__(self, path: str, info: dict):
        self.path, self.info, self._mm = str(path), dict(info), None
        clips = self.info["clips"]
        self.offsets = np.asarray([c["offset"] for c in clips], dtype=np.int64)
        self.lengths = np.asarray([c["length"] for c in clips], dtype=np.int64)
        self.cum = np.cumsum(self.lengths)

    @classmethod
    def load(cls, path, expect_sha256: str | None = None, check_audio: bool = True) -> "NoiseBank":
        """Open the bank at path: its index.json's sha256 checked against expect_sha256 (when given), its audio.npy's
        size against the index's audio_bytes (a short download fails here, not inside a worker) and, with check_audio,
        its sha256 against the index's audio_sha256 (the pin covers the audio through the index; ~5 s for 2.3 GiB,
        once per trainer start)."""
        path = Path(path)
        sha = sha256_file(path / INDEX)
        if expect_sha256 and sha != expect_sha256:
            raise ValueError(f"{path / INDEX}: sha256 {sha}, the config pins {expect_sha256}: another noise bank")
        info = json.loads((path / INDEX).read_text(encoding="utf-8"))
        if info.get("version") != VERSION or int(info.get("sr", 0)) != SR:
            raise ValueError(f"{path / INDEX}: version {info.get('version')} / sr {info.get('sr')}, expected "
                             f"{VERSION} / {SR}")
        size = (path / AUDIO).stat().st_size
        if size != int(info["audio_bytes"]):
            raise ValueError(f"{path / AUDIO}: {size} bytes, the index says {info['audio_bytes']}")
        if check_audio and info.get("audio_sha256") and sha256_file(path / AUDIO) != info["audio_sha256"]:
            raise ValueError(f"{path / AUDIO}: not the audio its index records (sha256 {info['audio_sha256'][:12]}...)")
        bank = cls(str(path), dict(info, index_sha256=sha))
        if not len(bank.lengths) or int(bank.cum[-1]) <= 0:
            raise ValueError(f"{path}: no audio in the bank")
        return bank

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None
        return state

    def _audio(self) -> np.ndarray:
        if self._mm is None:
            self._mm = np.load(Path(self.path) / AUDIO, mmap_mode="r")
        return self._mm

    @property
    def hours(self) -> float:
        return float(self.cum[-1]) / SR / 3600

    def segment(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """n float32 samples of background: from a position drawn uniformly over the whole bank (a clip in proportion
        to its length, a start inside it) to its clip's end, then again from another drawn position, until n samples
        are filled - a long row gets one or more clips back to back."""
        if n <= 0:
            return np.zeros(0, dtype=np.float32)
        audio, out, filled = self._audio(), [], 0
        while filled < n:
            pos = int(rng.integers(int(self.cum[-1])))
            c = int(np.searchsorted(self.cum, pos, side="right"))
            start = pos - (int(self.cum[c]) - int(self.lengths[c]))
            take = min(n - filled, int(self.lengths[c]) - start)
            o = int(self.offsets[c]) + start
            out.append(np.asarray(audio[o:o + take], dtype=np.float32) / 32768.0)
            filled += take
        return np.concatenate(out) if len(out) > 1 else out[0]


def voiced_power(wave: np.ndarray, hop: int = 160, db_range: float = 35.0) -> float:
    """The mean power of a row's voiced hop-sample (10 ms) frames - within db_range dB of its loudest -, the speech
    level a background is scaled against (silences and pauses left out, so a long gap does not make the speech look
    quiet). 0.0 for a row under one frame."""
    n = len(wave) // hop
    if n < 1:
        return 0.0
    fr = np.asarray(wave[:n * hop], dtype=np.float32).reshape(n, hop)
    p = np.einsum("ij,ij->i", fr, fr) / hop
    db = 10.0 * np.log10(p + 1e-12)
    return float(p[db > db.max() - db_range].mean())


BG_MIN_POWER = 1e-6  # -60 dBFS mean power: a quieter stretch (a fade, a pause inside a clip) is drawn again
BG_TRIES = 4  # draws before a row keeps its clean audio (0.2 % of the bank's stretches are this quiet)


def background_segment(bank: "NoiseBank", n: int, rng: np.random.Generator) -> np.ndarray | None:
    """A stretch of n samples of the bank loud enough to mix (mean power >= BG_MIN_POWER), within BG_TRIES draws;
    None otherwise. Scaling a near-silent stretch up to the drawn SNR would turn its noise floor into hiss."""
    for _ in range(BG_TRIES):
        seg = bank.segment(n, rng)
        if len(seg) and float(np.dot(seg, seg)) / len(seg) >= BG_MIN_POWER:
            return seg
    return None


def add_background(wave: np.ndarray, seg: np.ndarray, rng: np.random.Generator, snr_db, min_power: float = 1e-8):
    """wave with seg (its length) added, scaled so the row's voiced power over the background's mean power is an SNR
    drawn uniformly from snr_db (dB). Returns (the new array - wave is never written -, {snr_db, gain}) or None when
    either is silent (below min_power: no level to scale to)."""
    p_speech = voiced_power(wave)
    seg = np.asarray(seg, dtype=np.float32)
    p_bg = float(np.dot(seg, seg)) / max(len(seg), 1)
    if p_speech < min_power or p_bg < min_power:
        return None
    snr = float(rng.uniform(float(snr_db[0]), float(snr_db[1])))
    gain = (p_speech / (p_bg * 10.0 ** (snr / 10.0))) ** 0.5
    out = np.array(wave, dtype=np.float32, copy=True)
    out += np.float32(gain) * seg
    return out, dict(snr_db=snr, gain=gain)
