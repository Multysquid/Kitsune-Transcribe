"""Planted per-utterance tables on a synthetic eval manifest (tests/test_study_stats.py, tests/test_full_report.py).

    from fixtures_planted import planted
    tables, manifest = planted({"study-t06": 1.0, "cohere": 0.9})   # kitsune.study_stats.build_corpus's inputs

Every system's per-utterance edits are the SAME base edits times a multiplier m, so every metric's ratio between two
systems is exactly the ratio of their multipliers, in the full data and in every bootstrap replicate (v_boot = 0).
"""
import numpy as np
import pandas as pd

from kitsune.store import ids_sha256

SIZES = {"eval_jsut": 120, "eval_cv8": 100, "eval_reazon": 110, "eval_emilia": 40, "galgame": 90}
N_NEUTRAL = 70


def planted(mult: dict, seed: int = 0, sizes: dict = SIZES):
    """(tables, manifest): per set, random reference lengths and base edits; system s's edits = mult[s] x base. The
    last Galgame row has an empty reference (it must be left out of every sum); the neutral view is the first
    N_NEUTRAL Galgame rows."""
    rng = np.random.default_rng(seed)
    ids = {s: [f"{s}/u{i:04d}" for i in range(n)] for s, n in sizes.items()}
    ref_len = {s: rng.integers(4, 30, n) for s, n in sizes.items()}
    base = {s: rng.poisson(0.12 * ref_len[s]) + 1 for s in sizes}
    ref_len["galgame"][-1] = 0
    tables = {}
    for name, m in mult.items():
        tables[name] = pd.concat([pd.DataFrame(dict(id=ids[s], set=s, edits=m * base[s], ref_len=ref_len[s],
                                                    edits_nostyle=0.8 * m * base[s])) for s in sizes],
                                 ignore_index=True)
    manifest = {"sets": {s: {"ids": v, "ids_sha256": ids_sha256(v)} for s, v in ids.items()},
                "galgame_views": {"neutral": ids["galgame"][:N_NEUTRAL]}}
    return tables, manifest
