"""Audio decoding helpers. All model-facing audio is 16 kHz mono float32."""
import io

import numpy as np
import soundfile as sf

TARGET_SR = 16000


def audio_info(data: bytes) -> tuple[float, int]:
    """(duration_seconds, sample_rate) read from the container header only."""
    info = sf.info(io.BytesIO(data))
    return float(info.duration), int(info.samplerate)


def decode_audio(data: bytes, target_sr: int = TARGET_SR) -> np.ndarray:
    """Decode FLAC/OGG/WAV/MP3 bytes to mono float32 at `target_sr`."""
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != target_sr:
        import librosa  # lazy: only the non-16 kHz sources need it (Emilia 24 kHz, JSUT and CommonVoice 48 kHz)

        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return np.ascontiguousarray(wav, dtype=np.float32)
