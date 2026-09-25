"""Tests for scripts/02b_second_opinion.py on the label box (--judge parakeet, --whisper-dir, --require-npz,
--strict-existing) and tools/judge_report.py.

The box runs 02b every sync cycle next to the passes that are still writing, into a write-once root. So it must take
only finished shards (both npz), keep an output only when its ids are the teacher's, never redo one under
--strict-existing, judge galgame from Parakeet's jsonl without ever loading kotoba (no GPU), join the whisper parts 01
captured without touching the Hub, and stop (65) on anything that would silently become no_agree. CPU only, no network:
the whisper tokenizer, the Hub and kotoba are faked.
"""
import importlib.util
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
ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("kitsune_tool_judge_report", ROOT / "tools" / "judge_report.py")
judge_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(judge_report)

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
    """No kotoba, no Hub: the box's 02b must not need either."""
    class NoKotoba:
        def __init__(self, *a, **kw):
            raise AssertionError("kotoba was loaded")

    def no_hub(*a, **kw):
        raise AssertionError("the Hub was read")

    monkeypatch.setattr(sb, "Kotoba", NoKotoba)
    monkeypatch.setattr(sb, "load_whisper_map", no_hub)
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

    def run(self, *extra: str, judge="parakeet", strict=True, sources=None) -> int:
        argv = ["02b", "--data", str(self.data), "--teacher-out", str(self.teacher), "--out", str(self.out),
                "--require-npz"]
        if judge == "parakeet":
            argv += ["--judge", "parakeet", "--parakeet-out", str(self.parakeet), "--whisper-dir", str(self.whisper)]
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


def test_parakeet_judge_reads_the_jsonl_by_id_and_never_loads_kotoba(world):
    """galgame's hyp2 is Parakeet's hyp of the same id (the jsonl order may differ from the teacher's), model2 names
    the guarded decode, and meta.json records the judge; kotoba (the autouse fake raises) is never constructed."""
    stem = world.shard("galgame", ["galgame/a", "galgame/b"], parakeet_ids=["galgame/b", "galgame/a"])
    assert world.run() == 0
    rows = world.rows("galgame", stem)
    assert [(r["id"], r["hyp2"], r["model2"]) for r in rows] == [
        ("galgame/a", "パラa", sb.PARAKEET_MODEL2), ("galgame/b", "パラb", sb.PARAKEET_MODEL2)]
    assert all(r["agree"] is not None for r in rows)
    meta = json.loads((world.out / "meta.json").read_text(encoding="utf-8"))
    assert meta["judges"] == {"galgame": sb.PARAKEET_MODEL2} and meta["fallback"] == sb.FALLBACK_MODEL2
    assert "gpu_model" not in meta


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
    other = dict(join_model="whisper-large-v3 (precomputed)", judges={"galgame": sb.MODEL2})
    (world.out / "meta.json").write_text(json.dumps(other), encoding="utf-8")
    assert world.run() == sb.INTEGRITY_EXIT
    assert json.loads((world.out / "meta.json").read_text(encoding="utf-8")) == other
    (world.out / "meta.json").unlink()
    assert world.run() == 0
    written = (world.out / "meta.json").read_bytes()
    world.shard("galgame", ["galgame/b"])
    assert world.run() == 0 and (world.out / "meta.json").read_bytes() == written


def test_the_laptop_path_is_unchanged_without_the_new_flags(world):
    """Self-text sources need no model: the laptop command (no new flags) still writes them, and meta.json has the
    kotoba keys and none of the box's."""
    stem = world.shard("emilia_yodas", ["emilia_yodas/a"], parakeet=False)
    assert world.run(judge="kotoba", strict=False) == 0
    (row,) = world.rows("emilia_yodas", stem)
    assert row["model2"] == sb.SELF_TEXT_SOURCES["emilia_yodas"]
    meta = json.loads((world.out / "meta.json").read_text(encoding="utf-8"))
    assert meta["gpu_model"] == sb.MODEL2 and "judges" not in meta and "fallback" not in meta


def test_judge_report_matches_the_kotoba_threshold(tmp_path):
    """Synthetic judges: Parakeet's agree is kotoba's halved, so the Parakeet threshold that drops kotoba@0.5's hours
    is the one just below 0.25's worth; the 2x2 table, Spearman (1.0 for a monotone map) and the teacher CER of kept
    vs dropped rows follow from the same rows."""
    kdir, pdir, tdir = tmp_path / "k", tmp_path / "p", tmp_path / "t"
    rng = np.random.default_rng(0)
    for s in range(3):
        stem = f"train-{s:05d}"
        ids = [f"galgame/{s}_{i}" for i in range(20)]
        ka = rng.uniform(0, 1.2, len(ids)).round(4)
        for d, rows in ((kdir, [dict(id=i, hyp2="x", model2="kotoba", agree=float(a), cer2=0.1) for i, a in zip(ids, ka)]),
                        (pdir, [dict(id=i, hyp2="y", model2="parakeet", agree=float(a) / 2, cer2=0.1)
                                for i, a in zip(ids, ka)]),
                        (tdir, [dict(id=i, hyp="h", ref="r", cer=float(a), duration=3600.0) for i, a in zip(ids, ka)])):
            d.mkdir(exist_ok=True)
            (d / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (kdir / "train-00099.jsonl").write_text("", encoding="utf-8")  # a stem only one judge has: ignored

    out = tmp_path / "reports" / "galgame_judge.json"
    assert judge_report.main(["--kotoba", str(kdir), "--parakeet", str(pdir), "--teacher", str(tdir),
                              "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert (rep["stems"], rep["rows"], rep["hours"]) == (3, 60, 60.0)
    assert rep["matched"]["dropped_hours"] == rep["reference"]["dropped_hours"] == rep["dropped_hours"]["kotoba"]["0.5"]
    assert rep["matched"]["threshold"] <= 0.25 and rep["dropped_hours"]["parakeet"]["0.5"] < rep["reference"]["dropped_hours"]
    assert rep["spearman"] == 1.0
    t = rep["table"]
    assert t["kotoba_dropped/parakeet_kept"]["rows"] > 0 and t["kotoba_kept/parakeet_dropped"]["rows"] == 0
    assert sum(v["rows"] for v in t.values()) == 60
    for j in ("kotoba", "parakeet"):
        assert rep["teacher_cer"][j]["kept"] < rep["teacher_cer"][j]["dropped"]
    assert set(rep["dropped_hours"]["kotoba"]) == {"0.2", "0.3", "0.4", "0.5", "0.7"}
