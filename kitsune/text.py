"""Japanese text normalization and CER, used for teacher-label filtering and evaluation."""
import re
import unicodedata

import jiwer

# Everything that is not a letter/number/kana/kanji is dropped before scoring: punctuation,
# spaces, brackets, "…", full-width symbols, etc. Transcripts across our sources disagree
# wildly on punctuation, so CER must not be sensitive to it.
_DROP = re.compile(r"[^\w]", re.UNICODE)


def normalize_ja(text: str) -> str:
    """NFKC-normalize, lowercase latin, and strip all punctuation/whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = _DROP.sub("", text)
    return text


def cer(hyp: str, ref: str, normalize: bool = True) -> float:
    """Character error rate of `hyp` against `ref`. Empty reference -> 1.0 unless hyp also empty."""
    if normalize:
        hyp, ref = normalize_ja(hyp), normalize_ja(ref)
    if not ref:
        return 0.0 if not hyp else 1.0
    if not hyp:
        return 1.0
    return float(jiwer.cer(ref, hyp))
