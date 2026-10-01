"""The chained box's registry entry for the tests (contract addendum E.1.7), built on top of
tests/fixtures_full.tiny_registry - never the committed configs/full/boxes.json, whose p01-chain entry a finalize step
adds once WP2c's configs are merged.

    from fixtures_chain import CHAIN_BOX, with_chain
    reg = with_chain(tiny_registry(tmp_path))       # the tiny registry + p01-chain (a copy; it remembers its root)

Stdlib only, like fixtures_full.
"""
import copy

CHAIN_BOX = "p01-chain"
CHAIN = {"_comment": "addendum E.1.7 (the entry the finalize step adds to configs/full/boxes.json)",
         "est_hours": 25.2, "max_hours": 35, "max_dph": 1.00, "extra_gb": 120, "gate": True,
         "chain": [{"parts": ["full-smoke", "smoke-b"], "gate_box": "full-smoke",
                    "rebuild": "configs/full/data-smoke.json", "gate_by_hours": 9, "max_hours": 10.5},
                   {"parts": ["p01"], "rebuild": "configs/full/data-p01.json"}]}


def with_chain(reg: dict, **changes) -> dict:
    """reg (a tiny registry, which remembers its root: a deepcopy keeps it) plus the p01-chain entry (its fields
    changed by `changes`)."""
    reg = copy.deepcopy(reg)
    reg["boxes"][CHAIN_BOX] = dict(copy.deepcopy(CHAIN), **changes)
    return reg
