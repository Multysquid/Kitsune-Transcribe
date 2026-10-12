"""Tests for scripts/02b_second_opinion.py on the label box (--judge parakeet, --whisper-dir, --require-npz,
--strict-existing).

The box runs 02b every sync cycle next to the passes that are still writing, into a write-once root. So it must take
only finished shards (both npz), keep an output only when its ids are the teacher's, never redo one under
--strict-existing, judge galgame from Parakeet's jsonl, join the whisper parts 01 captured without touching the Hub,
refuse a source whose input is not given, and stop (65) on anything that would silently become no_agree. CPU only, no
network: the whisper tokenizer and the Hub are faked.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import load_script  # noqa: E402

from kitsune import extent  # noqa: E402
from kitsune.store import ShardWriter  # noqa: E402

sb = load_script("02b_second_opinion")

LARGE_INPUT = "large/train-00000-of-00002.parquet"


def text_of(tokens: list[int]) -> str:
    return "".join(chr(0x3042 + t % 40) for t in tokens)


class FakeTokenizer:
    calls: list = []

    @classmethod
    def from_pretrained(cls, name, revision=None, **kw):
        assert (name, revision) == (sb.WHISPER_TOK, sb.WHISPER_TOK_REVISION)
        return cls()

    def batch_decode(self, lists, skip_special_tokens=True):
        assert all(x is not None for x in lists), "batch_decode cannot take a null transcript"
        FakeTokenizer.calls.append(len(lists))
        return [f" {text_of(x)} " for x in lists]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No Hub: the box's 02b runs under HF_HUB_OFFLINE=1, and only the pinned tokenizer (faked) is loaded."""
    def no_hub(*a, **kw):
        raise AssertionError("the Hub was read")

    hub = types.SimpleNamespace(HfApi=no_hub, HfFileSystem=no_hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=FakeTokenizer))
    FakeTokenizer.calls = []


class World:
    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.data, self.teacher, self.parakeet = tmp_path / "data", tmp_path / "teacher_out", tmp_path / "parakeet_out"
        self.out, self.whisper = tmp_path / "second_out", tmp_path / "whisper"

    def shard(self, source: str, ids: list[str], split="train", meta=None, teacher=True, parakeet=True,
              teacher_npz=True, parakeet_npz=True, parakeet_ids=None) -> str:
        w = ShardWriter(self.data, source, split, rows_per_shard=len(ids), meta=meta)
        for i in ids:
            w.add(i, b"", "こんにちは", 1.5, 16000)
        (info,) = w.close()
        stem = Path(info.path).stem
        if teacher:
            self._pair(self.teacher / source / stem, [dict(id=i, hyp="こんにちは", ref="こんにちは", cer=0.1 * n,
                                                           duration=1.5) for n, i in enumerate(ids)], teacher_npz)
        if parakeet:
            pids = ids if parakeet_ids is None else parakeet_ids
            self._pair(self.parakeet / source / stem, [dict(id=i, hyp=f"パラ{i[-1]}") for i in pids], parakeet_npz)
        return stem

    @staticmethod
    def _pair(base: Path, rows: list[dict], npz: bool):
        base.parent.mkdir(parents=True, exist_ok=True)
        base.with_suffix(".jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                              encoding="utf-8")
        if npz:
            np.savez(base.with_suffix(".npz"), ids=np.array([r["id"] for r in rows]))

    def part(self, source: str, input_name: str, names: list[str], lists: list):
        import pyarrow as pa
        import pyarrow.parquet as pq

        p = self.whisper / source / Path(input_name).name
        p.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"name": pa.array(names), "whisper_transcript": pa.array(lists, pa.list_(pa.int64()))}),
                       p)
        return p

    def run(self, *extra: str, strict=True, sources=None, parakeet=True, whisper=True) -> int:
        """The label box's call (vast/label.py second_argv); parakeet / whisper False leave out that input."""
        argv = ["02b", "--data", str(self.data), "--teacher-out", str(self.teacher), "--out", str(self.out),
                "--require-npz"]
        if parakeet:
            argv += ["--judge", "parakeet", "--parakeet-out", str(self.parakeet)]
        if whisper:
            argv += ["--whisper-dir", str(self.whisper)]
        if strict:
            argv.append("--strict-existing")
        if sources:
            argv += ["--sources", *sources]
        argv += list(extra)
        old = sys.argv
        sys.argv = argv
        try:
            sb.main()
            return 0
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else 1
        finally:
            sys.argv = old

    def rows(self, source: str, stem: str) -> list[dict]:
        return sb.read_teacher_rows(self.out / source / f"{stem}.jsonl")


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def test_parakeet_judge_reads_the_jsonl_by_id(world):
    """galgame's hyp2 is Parakeet's hyp of the same id (the jsonl order may differ from the teacher's), model2 names
    the guarded decode, and meta.json records the judge, with the keys of the sealed root's meta.json in its order."""
    stem = world.shard("galgame", ["galgame/a", "galgame/b"], parakeet_ids=["galgame/b", "galgame/a"])
    assert world.run() == 0
    rows = world.rows("galgame", stem)
    assert [(r["id"], r["hyp2"], r["model2"]) for r in rows] == [
        ("galgame/a", "パラa", sb.PARAKEET_MODEL2), ("galgame/b", "パラb", sb.PARAKEET_MODEL2)]
    assert all(r["agree"] is not None for r in rows)
    meta = json.loads((world.out / "meta.json").read_text(encoding="utf-8"))
    assert meta["judges"] == {"galgame": sb.PARAKEET_MODEL2} and meta["fallback"] == sb.FALLBACK_MODEL2
    assert list(meta) == list(sb.META_KEYS)


def test_eligibility_needs_both_npz(world):
    """02 and 02p write the jsonl first: a jsonl without its npz may be a shard still being written."""
    no_teacher_npz = world.shard("galgame", ["galgame/a"], teacher_npz=False)
    no_parakeet_npz = world.shard("galgame", ["galgame/b"], parakeet_npz=False)
    ready = world.shard("galgame", ["galgame/c"])
    assert world.run() == 0
    assert not (world.out / "galgame" / f"{no_teacher_npz}.jsonl").exists()
    assert not (world.out / "galgame" / f"{no_parakeet_npz}.jsonl").exists()
    assert (world.out / "galgame" / f"{ready}.jsonl").exists()


def test_done_by_ids_the_verified_cache_and_strict_mode(world, capsys):
    """An output is kept only when its ids are the teacher's, in order; checked stems go to _cache/verified.txt so the
    next cycle does not re-read them. A stale output is redone, except under --strict-existing (65)."""
    stem = world.shard("galgame", ["galgame/a", "galgame/b"])
    assert world.run() == 0
    cache = world.out / "_cache" / sb.VERIFIED_CACHE
    assert cache.read_text(encoding="utf-8").split() == [f"galgame/{stem}"]
    out = world.out / "galgame" / f"{stem}.jsonl"
    first = out.read_bytes()
    assert world.run() == 0 and out.read_bytes() == first
    assert "0/1 eligible shards to do" in capsys.readouterr().out

    # a stale output (other ids), not yet verified: redone without the flag, 65 with it
    cache.unlink()
    out.write_text(json.dumps(dict(id="galgame/zz", hyp2=None, model2=None, agree=None, cer2=None)) + "\n",
                   encoding="utf-8")
    stale = out.read_bytes()
    assert world.run() == sb.INTEGRITY_EXIT and out.read_bytes() == stale
    assert world.run(strict=False) == 0 and out.read_bytes() == first
    assert cache.read_text(encoding="utf-8").split() == [f"galgame/{stem}"]


def test_whisper_parts_join_without_the_hub_and_nulls_fall_back_to_parakeet(world):
    """The part named by the stem's id sidecar is decoded with the pinned tokenizer (nulls never reach batch_decode);
    a null or missing transcript takes the Parakeet hyp with the fallback model2. The Hub fakes raise if touched."""
    meta = dict(input=LARGE_INPUT, step="reazon_large")
    ids = [f"reazon_large/n{i}" for i in range(10)]
    stem = world.shard("reazon_large", ids, meta=meta)
    # n0..n7 transcribed, n8 and n9 null in the mirror: both fall back to Parakeet
    world.part("reazon_large", LARGE_INPUT, [f"n{i}" for i in range(10)], [[i, i + 1] for i in range(8)] + [None, None])
    assert world.run() == 0
    rows = {r["id"]: r for r in world.rows("reazon_large", stem)}
    assert rows["reazon_large/n3"]["hyp2"] == text_of([3, 4]) and rows["reazon_large/n3"]["model2"] == "whisper-large-v3"
    for i in (8, 9):
        r = rows[f"reazon_large/n{i}"]
        assert (r["hyp2"], r["model2"]) == (f"パラ{i}", sb.FALLBACK_MODEL2) and r["agree"] is not None
    assert FakeTokenizer.calls == [8]


def test_more_than_a_fifth_falling_back_is_a_capture_bug(world):
    meta = dict(input=LARGE_INPUT, step="reazon_large")
    n = 30
    stem = world.shard("reazon_large", [f"reazon_large/n{i}" for i in range(n)], meta=meta)
    world.part("reazon_large", LARGE_INPUT, [f"n{i}" for i in range(n)], [[1]] * 8 + [None] * (n - 8))
    assert world.run() == sb.INTEGRITY_EXIT
    assert not (world.out / "reazon_large" / f"{stem}.jsonl").exists()


def test_a_small_tail_shard_with_one_mirror_null_is_fine(world):
    """1 null of 3 rows is 33 %, but a tail shard that small says nothing about capture: it falls back."""
    meta = dict(input=LARGE_INPUT, step="reazon_large")
    stem = world.shard("reazon_large", [f"reazon_large/n{i}" for i in range(3)], meta=meta)
    world.part("reazon_large", LARGE_INPUT, ["n0", "n1", "n2"], [[1], None, [2]])
    assert world.run() == 0
    assert [r["model2"] for r in world.rows("reazon_large", stem)] == [
        "whisper-large-v3", sb.FALLBACK_MODEL2, "whisper-large-v3"]


def test_a_name_missing_from_its_part_is_a_capture_bug_on_any_count(world):
    meta = dict(input=LARGE_INPUT, step="reazon_large")
    stem = world.shard("reazon_large", [f"reazon_large/n{i}" for i in range(10)], meta=meta)
    world.part("reazon_large", LARGE_INPUT, [f"n{i}" for i in range(9)], [[1]] * 9)
    assert world.run() == sb.INTEGRITY_EXIT
    assert not (world.out / "reazon_large" / f"{stem}.jsonl").exists()


def test_a_missing_part_waits_for_the_ingest_then_is_an_integrity_error(world, capsys):
    """01 writes a file's part before it marks the file finished, so while the step runs a missing part only means
    'not yet'; once extent_progress.json lists the step as complete it can never appear (65)."""
    stem = world.shard("reazon_large", ["reazon_large/n0"], meta=dict(input=LARGE_INPUT, step="reazon_large"))
    extent.write_progress(world.data, dict(canonical_version=extent.CANONICAL_VERSION, completed=["reazon_small"]))
    assert world.run() == 0
    assert "not captured yet" in capsys.readouterr().out
    assert not (world.out / "reazon_large" / f"{stem}.jsonl").exists()
    extent.write_progress(world.data, dict(canonical_version=extent.CANONICAL_VERSION,
                                           completed=["reazon_small", "reazon_large"]))
    assert world.run() == sb.INTEGRITY_EXIT


def test_a_teacher_row_without_a_parakeet_row_exits_in_strict_mode(world):
    """It would become no_agree, which launch refuses; without the flag it is a null second opinion as before."""
    stem = world.shard("galgame", ["galgame/a", "galgame/b"], parakeet_ids=["galgame/a"])
    assert world.run() == sb.INTEGRITY_EXIT
    assert not (world.out / "galgame" / f"{stem}.jsonl").exists()
    assert world.run(strict=False) == 0
    assert [r["hyp2"] for r in world.rows("galgame", stem)] == ["パラa", None]


def test_meta_is_write_once_in_strict_mode(world):
    world.shard("galgame", ["galgame/a"])
    (world.out).mkdir(parents=True)
    other = dict(join_model="whisper-large-v3 (precomputed)", judges={"galgame": "kotoba-tech/kotoba-whisper-v2.0"})
    (world.out / "meta.json").write_text(json.dumps(other), encoding="utf-8")
    assert world.run() == sb.INTEGRITY_EXIT
    assert json.loads((world.out / "meta.json").read_text(encoding="utf-8")) == other
    (world.out / "meta.json").unlink()
    assert world.run() == 0
    written = (world.out / "meta.json").read_bytes()
    world.shard("galgame", ["galgame/b"])
    assert world.run() == 0 and (world.out / "meta.json").read_bytes() == written


def test_a_source_without_its_input_is_refused_and_self_text_needs_none(world, capsys):
    """Nothing streams a mirror or loads a model: a join source without --whisper-dir and a judged one without
    --parakeet-out are refused (2) before anything is written; a self-text source needs neither, and its meta.json then
    names no judge and no fallback."""
    emilia = world.shard("emilia_yodas", ["emilia_yodas/a"], parakeet=False)
    world.shard("reazon_large", ["reazon_large/n0"], meta=dict(input=LARGE_INPUT, step="reazon_large"))
    world.shard("galgame", ["galgame/a"])
    assert world.run(whisper=False, sources=["reazon_large", "emilia_yodas"]) == 2
    assert "reazon_large: a join source needs --whisper-dir" in capsys.readouterr().err
    assert world.run(parakeet=False, sources=["galgame", "emilia_yodas"]) == 2
    assert "galgame: a judged source needs --parakeet-out" in capsys.readouterr().err
    assert not world.out.exists()
    assert world.run(parakeet=False, whisper=False, sources=["emilia_yodas"]) == 0
    (row,) = world.rows("emilia_yodas", emilia)
    assert (row["hyp2"], row["model2"]) == ("こんにちは", sb.SELF_TEXT_SOURCES["emilia_yodas"])  # the reference
    meta = json.loads((world.out / "meta.json").read_text(encoding="utf-8"))
    assert "judges" not in meta and "fallback" not in meta
