"""The AED students' train-data augmentation (kitsune.trainset's "AED rows", kitsune/aed_cuts.py,
tools/aed_cut_table.py, scripts/04_distill.py augment.* on an AED student):

- the cut table's rows: token runs of a CTC argmax, a Parakeet boundary carried over to the Cohere tokens only where the
  two texts match around it and a Cohere token starts there, never before a mark, the pause gap, and agreement with the
  CTC truncate rule; the table tool on hand-written label files; the join to a dataset by id
- on a real token store: off is the plain dataset; a cut keeps the tokens the table names for its frame and ends in EOS
  with a one-hot top-k row, its audio still the cut's frames; a join concatenates the pieces' tokens (each piece's EOS
  dropped but the last's) at frame-aligned audio offsets, and the decoder cap refuses a join whose rows would pass it;
  short rows and rows whose decoded frames differ from the table's are never cut; the end trim shortens the trailing
  silence of rows that were not cut and keeps their tokens; the same index list gives the same batch in any process
- the trainer: augment.* on an AED student validated (cuts required to cut, no pads, no table on a CTC student), and a
  tiny AED run with all three augmentations through its smoke phase logs the aug/* shares and the table's coverage
CPU only, tiny models, synthetic data in the real on-disk formats (tests/fixtures.py)."""
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does not hide the GPU from the Windows CUDA driver; -1 does
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from fixtures import load_script, make_fake_corpus, make_fake_selection, make_noise_bank  # noqa: E402
from kitsune import aed_cuts as A  # noqa: E402
from kitsune import trainset as T  # noqa: E402
from kitsune.ctc_targets import FrameTargets  # noqa: E402
from kitsune.parakeet_targets import ctc_greedy  # noqa: E402

EVAL = ["eval_jsut", "eval_cv8", "eval_reazon"]
B = A.PK_BLANK
FS = T.FRAME_SAMPLES
EOS = 3


@pytest.fixture(autouse=True)
def grad_enabled():
    prev = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    yield
    torch.set_grad_enabled(prev)


# ------------------------------------------------------------------------------------------------- the table rows

# a hand-made row: the Parakeet tokens 今日 / は / いい / 天気 / 。 on its CTC frames, the Cohere tokens 今日は / 、 / いい /
# 天気 / です / ね / 。 (one piece per id below)
PK_TEXTS = {10: "今日", 11: "は", 12: "いい", 13: "天気", 1: "。"}
CO_TEXTS = {20: "今日は", 21: "、", 22: "いい", 23: "天気", 24: "です", 25: "ね", 26: "。"}
COL0 = [B, 10, 10, B, 11, B, B, B, B, B, 12, B, 13, 13, B, B, 1, B]
CO_TOKENS = [20, 21, 22, 23, 24, 25, 26]


def texts(d: dict, n: int) -> list[str]:
    return [d.get(i, "") for i in range(n)]


def test_token_runs_pair_each_start_with_its_end():
    st, en, cls = A.token_runs(COL0)
    assert st.tolist() == [1, 4, 10, 12, 16] and en.tolist() == [2, 4, 10, 13, 16]
    assert cls.tolist() == [10, 11, 12, 13, 1]
    st, en, cls = A.token_runs([5, 6, 6, B, 6])  # a token right after another, and the same class again after a blank
    assert st.tolist() == [0, 1, 4] and en.tolist() == [0, 2, 4] and cls.tolist() == [5, 6, 6]
    assert [len(x) for x in A.token_runs([])] == [0, 0, 0]


def test_a_boundary_maps_only_where_both_texts_agree_and_a_cohere_token_starts():
    """Before いい: Parakeet keeps 今日は, which is Cohere's 今日は、 (the comma is not compared): m = 2. Before 天気: m = 3.
    Before は: no Cohere token starts there (今日は is one token). Before 。: a mark, never removed alone. Token u's
    entry spans the frames after token u-1 starts up to u's start, its gap the blank frames before u."""
    lo, b, hi, m = A.row_entries(COL0, texts(PK_TEXTS, 3072), (1,), CO_TOKENS, texts(CO_TEXTS, 30))
    assert lo.tolist() == [5, 11] and b.tolist() == [5, 11] and hi.tolist() == [10, 12] and m.tolist() == [2, 3]
    assert lo.dtype == np.int16
    fr, kept, pause = A.row_candidates([((lo.astype(np.int64), b.astype(np.int64), hi.astype(np.int64),
                                          m.astype(np.int64)), 0, 0)], 0, T.TRUNCATE_PAUSE_FRAMES)
    assert fr.tolist() == [5, 6, 7, 8, 9, 10, 11, 12] and kept.tolist() == [2] * 6 + [3] * 2
    assert pause.tolist() == [True] * 5 + [False] * 3  # the 5-frame gap before いい; 天気's 1-frame gap is no pause


def test_entries_are_ctc_truncate_frames_and_their_pauses():
    """Every entry frame is a frame trainset.truncate_frames allows on the same argmax (no bound), and its pause flag is
    trainset.pause_frames there."""
    col0 = np.asarray(COL0)
    dense = np.flatnonzero(col0 != B).astype(np.int32)
    idx = np.zeros((len(dense), 8), np.int16)
    idx[:, 0] = col0[dense]
    idx[:, 1:] = np.arange(100, 107)
    ft = FrameTargets(len(col0), np.where(col0 == B, -0.01, -4.0).astype(np.float16), dense, idx,
                      np.full((len(dense), 8), -2.0, np.float16), np.asarray(ctc_greedy(col0), dtype=np.int32))
    ent = [x.astype(np.int64) for x in A.row_entries(COL0, texts(PK_TEXTS, 3072), (1,), CO_TOKENS,
                                                       texts(CO_TEXTS, 30))]
    fr, _, pause = A.row_candidates([(tuple(ent), 0, 0)], 0, T.TRUNCATE_PAUSE_FRAMES)
    assert set(fr.tolist()) <= set(T.truncate_frames(ft, 0.0, 0.0, (1,)).tolist())
    assert pause.tolist() == T.pause_frames(ft.col0())[fr].tolist()


def test_disagreeing_texts_and_short_context_give_no_entry():
    """A boundary needs CTX matching characters on each side: Cohere hearing another word right after it, or the text
    ending within CTX characters, leaves the token without an entry."""
    co = {20: "今日は", 22: "悪い", 23: "天気"}  # Cohere: 今日は悪い天気 (Parakeet: 今日はいい天気)
    lo, *_, m = A.row_entries(COL0, texts(PK_TEXTS, 3072), (1,), [20, 22, 23], texts(co, 30))
    assert m.tolist() == []  # いい vs 悪い breaks the run on both sides of the two boundaries
    pk = {10: "今日", 11: "は", 12: "い", 13: "く", 1: "。"}  # the last word one character long: no right context
    lo, *_, m = A.row_entries(COL0, texts(pk, 3072), (1,), [20, 22, 23], texts({20: "今日は", 22: "い", 23: "く"}, 30))
    assert m.tolist() == [1]  # before い (は|い, two characters on each side: いく); not before く


def test_piece_text():
    assert A.piece_text("▁今日") == "今日" and A.piece_text("<0xE3>") == "�"
    assert A.piece_text("<|endoftext|>") == "" and A.piece_text("<unk>") == ""


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(f"kitsune_tool_{name}", ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_table(path: Path, rows: list[tuple]) -> Path:
    """rows: (id, n_frames, lo, b, hi, m)."""
    schema = pa.schema([("id", pa.string()), ("n_frames", pa.int16())]
                       + [(k, pa.list_(pa.int16())) for k in A.ENTRY_ARRAYS])
    cols = list(zip(*rows)) if rows else [[]] * 6
    pq.write_table(pa.table({k: list(v) for k, v in zip(schema.names, cols)}, schema=schema), path)
    return path


def test_the_index_joins_the_table_by_id(tmp_path):
    t = write_table(tmp_path / "cuts.parquet", [("a", 30, [5, 12], [7, 12], [10, 14], [2, 3]),
                                                 ("b", 20, [], [], [], []), ("c", 40, [3], [3], [9], [1])])
    msgs = []
    idx = A.build_cut_index(t, ["c", "x", "a"], tmp_path / "out", log=msgs.append)
    assert [idx.n_frames(i) for i in range(3)] == [40, -1, 30]
    assert [x.tolist() for x in idx.entries(0)] == [[3], [3], [9], [1]]
    assert [x.tolist() for x in idx.entries(1)] == [[], [], [], []]
    assert [x.tolist() for x in idx.entries(2)] == [[5, 12], [7, 12], [10, 14], [2, 3]]
    assert idx.info["rows_in_table"] == 2 and idx.info["rows_with_cuts"] == 2 and idx.info["entries"] == 3
    again = pickle.loads(pickle.dumps(idx))  # the loader's workers get it pickled, without its memmaps
    assert again._mm is None and again.entries(2)[3].tolist() == [2, 3]
    A.build_cut_index(t, ["c", "x", "a"], tmp_path / "out", log=msgs.append)
    assert "reusing" in msgs[-1] and "built" in msgs[0]
    with pytest.raises(ValueError, match="a different cut table"):
        A.build_cut_index(t, ["a"], tmp_path / "out", expect_sha256="0" * 64)
    write_table(tmp_path / "dup.parquet", [("a", 3, [], [], [], []), ("a", 3, [], [], [], [])])
    with pytest.raises(ValueError, match="duplicate ids"):
        A.build_cut_index(tmp_path / "dup.parquet", ["a"], tmp_path / "out2")


def test_the_table_tool_on_hand_written_labels(tmp_path):
    """tools/aed_cut_table.py on a label root holding one stem of three rows (the hand-made row, a row whose Cohere
    transcript lacks its EOS, a row the Parakeet pass never labelled): the hand-made row's entries, the others in the
    table without cuts, and the report's counts and sha256."""
    lab = tmp_path / "labels"
    (lab / "parakeet_out" / "src").mkdir(parents=True)
    (lab / "teacher_out" / "src").mkdir(parents=True)
    col0 = np.asarray(COL0, dtype=np.int64)
    dense = np.flatnonzero(col0 != B)
    top = np.zeros((2 * len(dense), 8), np.int16)
    top[:, 0] = np.concatenate([col0[dense], col0[dense]])
    T_ = len(col0)
    np.savez(lab / "parakeet_out" / "src" / "train-00000.npz", ids=np.asarray(["src/r1", "src/r2"]),
             n_frames=np.asarray([T_, T_], np.int32), frame_offsets=np.asarray([0, T_, 2 * T_], np.int64),
             dense_offsets=np.asarray([0, len(dense), 2 * len(dense)], np.int64),
             ctc_dense_frame=np.concatenate([dense, dense]).astype(np.int16), ctc_topk_idx=top)
    co = CO_TOKENS + [EOS]
    np.savez(lab / "teacher_out" / "src" / "train-00000.npz", ids=np.asarray(["src/r1", "src/r2", "src/r3"]),
             tok_offsets=np.asarray([0, len(co), 2 * len(co) - 1, 3 * len(co) - 1], np.int64),
             tokens=np.asarray(co + co[:-1] + co, np.int16))
    pk_vocab = tmp_path / "pk.vocab"
    pk_vocab.write_text("".join(f"{PK_TEXTS.get(i, '<unk>' if i == 0 else f'x{i}')}\t0\n" for i in range(3072)),
                        encoding="utf-8")
    co_vocab = tmp_path / "co.vocab"
    co_vocab.write_text("".join(f"{CO_TEXTS.get(i, '<|endoftext|>' if i == EOS else f'y{i}')}\t0\n"
                                for i in range(30)), encoding="utf-8")
    sel = tmp_path / "sel.parquet"
    pd.DataFrame(dict(id=["src/r1", "src/r2", "src/r3", "src/dev"], source="src", split=["train"] * 3 + ["dev"],
                      teacher_file="src/train-00000", keep=True)).to_parquet(sel)
    tool = load_tool("aed_cut_table")
    out = tmp_path / "cuts.parquet"
    assert tool.main(["--selection", str(sel), "--labels", str(lab), "--cohere-tokenizer", str(co_vocab),
                      "--parakeet-vocab", str(pk_vocab), "--out", str(out)]) == 0
    t = pq.read_table(out).to_pydict()
    assert t["id"] == ["src/r1", "src/r2", "src/r3"] and t["n_frames"] == [T_, T_, -1]
    assert (t["lo"][0], t["hi"][0], t["m"][0]) == ([5, 11], [10, 12], [2, 3])
    assert t["lo"][1] == [] and t["lo"][2] == []  # r2: no EOS in its Cohere transcript; r3: no Parakeet label
    rep = json.loads(Path(str(out) + ".json").read_text(encoding="utf-8"))
    assert rep["table_sha256"] == A.sha256_file(out) and rep["no_eos"] == 1 and rep["missing"] == 1
    assert rep["parakeet_punct_ids"] == {"1": "。"} and rep["per_source"]["src"]["rows_with_entries"] == 1
    assert rep["per_source"]["src"]["entries"] == 2 and rep["per_source"]["src"]["rows_with_pause_entry"] == 1


# --------------------------------------------------------------------------------------------- on a token store


def store_table(store, path: Path) -> Path:
    """A cut table for every row of a token store: one entry per kept-token count m in 1..body-1, the row's frames split
    evenly between them (gaps in each entry's second half), so the dataset tests know what every frame keeps."""
    rows = []
    for i, u in enumerate(store.utts):
        F = T.ctc_frames(len(store.wave(i)))
        body = u.n_tok - 1
        lo, b, hi, m = [], [], [], []
        edges = np.linspace(0, F - 1, max(body, 1) + 1).astype(int)
        prev = int(edges[0])
        for k in range(1, body):
            h = int(edges[k])
            if h <= prev:
                continue
            lo.append(prev + 1), b.append(prev + 1 + (h - prev - 1) // 2), hi.append(h), m.append(k)
            prev = h
        rows.append((u.id, F, lo, b, hi, m))
    return write_table(path, rows)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("aedaug")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (40, "train"), "src_b": (24, "train"),
                                                    **{s: (6, "eval") for s in EVAL}},
                          dur_range=(0.6, 3.0), token_range=(256, 296), no_second=tuple(EVAL), seed=17)
    sel = make_fake_selection(fc, greedy_n=4, probe_n=5)
    store = T.build_stores(sel, fc.data, fc.teacher_out, root / "cache" / "tok_train", ["src_a", "src_b"], ["train"],
                           log=lambda s: None)
    table = store_table(store, root / "cuts.parquet")
    cuts = A.build_cut_index(table, [u.id for u in store.utts], root / "cache" / "aed_cuts", log=lambda s: None)
    return dict(root=root, fc=fc, sel=sel, store=store, table=table, cuts=cuts)


def spec(**kw) -> T.Augment:
    return T.Augment(seed=kw.pop("seed", 5), max_tokens=kw.pop("max_tokens", 191), **kw)


def micro_batches(store, micro_s: float = 6.0) -> list[list[int]]:
    p = T.StepPlanner(store.utts, step_audio_s=12, micro_audio_s=micro_s, max_dec_len=200, pool_micro=4, seed=0)
    return [mb for step in p.epoch_plan(0) for mb in step]


def test_off_is_the_plain_dataset(env):
    """Without an augmentation a token store's micro-batches are the stored rows (the collate rebuilt here from the
    store's own targets) and carry no `aug`; with every probability 0 they are the same tensors plus zero counts."""
    st = env["store"]
    ds = T.dataset_for(st)
    for idx in micro_batches(st)[:6]:
        got = ds[idx]
        assert "aug" not in got and got["ids"] == [st.utts[i].id for i in idx]
        P = len(ds.prompt)
        for b, i in enumerate(idx):
            tok, ti, tl = st.targets(i)
            t = len(tok)
            assert got["decoder_input_ids"][b, P:P + t - 1].tolist() == tok[:-1].tolist()
            assert got["dec_mask"][b].sum() == P + t - 1 and int(got["n_tok"][b]) == t
        assert torch.equal(got["top_idx"], torch.from_numpy(np.concatenate([st.targets(i)[1] for i in idx]).astype(
            np.int64)))
        zero = ds.with_augment(spec(), env["cuts"])[idx]
        assert zero["aug"] == dict(utts=len(idx), rows=len(idx), concat_groups=0, concat_utts=0, truncated=0, mixed=0,
                                   cut_padded=0, end_padded=0, concat_capped=0, cut_mismatch=0, end_trimmed=0,
                                   noised=0)
        for k, v in got.items():
            assert (torch.equal(v, zero[k]) if isinstance(v, torch.Tensor) else v == zero[k]), k


def test_where_an_aed_augmentation_applies(env):
    st = env["store"]
    with pytest.raises(ValueError, match="AED one"):
        T.Augment(seed=1, max_tokens=191, punct_ids=(1,))
    with pytest.raises(ValueError, match="AED one"):
        T.Augment(seed=1, max_tokens=191, truncate_pad_p=0.5)
    with pytest.raises(ValueError, match="end_trim_p needs punct_ids"):
        T.Augment(seed=1, end_trim_p=0.3)  # a CTC one: it finds the final mark by its ids
    with pytest.raises(ValueError, match="needs the cut table"):
        T.dataset_for(st, augment=spec(truncate_p=0.5))
    with pytest.raises(ValueError, match="CTC augmentation"):
        T.dataset_for(st, augment=T.Augment(seed=1, concat_p=0.5))
    with pytest.raises(ValueError, match="without an augmentation"):
        T.dataset_for(st, cuts=env["cuts"])
    assert T.dataset_for(st, augment=spec(concat_p=0.5)).augment.concat_p == 0.5  # joins need no table


def cut_batches(env, **kw):
    """(micro-batch index list, its augmented batch) of every micro-batch with truncate on (truncate_p 1)."""
    ds = T.dataset_for(env["store"], augment=spec(truncate_p=1.0, truncate_min_s=0.0, truncate_min_frac=0.0, **kw),
                       cuts=env["cuts"])
    return [(idx, ds[idx]) for idx in micro_batches(env["store"])]


def test_a_cut_keeps_the_tokens_its_frame_names(env):
    """Every cut row: its audio ends where it is still the cut frame's frames (never always on a boundary), its tokens
    are the stored ones up to what the table names for that frame, then EOS - whose top-k row is one-hot -, and its
    decoder input is the prompt and the kept tokens."""
    st, cuts = env["store"], env["cuts"]
    n_cut = ends = 0
    for idx, got in cut_batches(env):
        n_cut += got["aug"]["truncated"]
        P, n = len(T.PROMPT), got["n_tok"].tolist()
        off = np.concatenate([[0], np.cumsum(n)])
        for b, i in enumerate(idx):
            L = int(got["lengths"][b])
            tok = got["top_idx"][off[b]:off[b + 1], 0].tolist()
            stored = st.targets(i)[0].tolist()
            if tok == stored:
                continue  # not cut (no frame of the table at or after the bound)
            c = T.ctc_frames(L)
            lo, bb, hi, m = cuts.entries(i)
            j = int(np.flatnonzero((lo <= c) & (c <= hi))[0])
            assert tok == stored[:int(m[j])] + [EOS] and n[b] == int(m[j]) + 1
            assert got["top_lp"][off[b + 1] - 1].tolist() == [0.0] + [T.TOKEN_CUT_FLOOR_LP] * 15
            assert got["top_idx"][off[b + 1] - 1, 0] == EOS and EOS not in got["top_idx"][off[b + 1] - 1, 1:].tolist()
            assert got["decoder_input_ids"][b, P:P + n[b] - 1].tolist() == tok[:-1]
            assert got["durations"][b] == pytest.approx(L / T.TARGET_SR)
            ends += L % FS != 0
    assert n_cut > 10 and ends > n_cut // 2  # most ends mid-frame, as a natural utterance's


def test_the_pause_share_is_honoured(env):
    """truncate_pause_p 1 puts every cut that can be in a pause there (the gap half of an entry)."""
    cuts = env["cuts"]
    for idx, got in cut_batches(env, truncate_pause_p=1.0):
        for b, i in enumerate(idx):
            if got["ids"][b] != env["store"].utts[i].id or got["n_tok"][b] == env["store"].utts[i].n_tok:
                continue
            c = T.ctc_frames(int(got["lengths"][b]))
            lo, bb, hi, m = cuts.entries(i)
            if np.any((hi - bb) >= T.TRUNCATE_PAUSE_FRAMES):
                j = int(np.flatnonzero((lo <= c) & (c <= hi))[0])
                assert hi[j] - bb[j] >= T.TRUNCATE_PAUSE_FRAMES and bb[j] <= c < hi[j]


def test_short_rows_and_mismatched_rows_are_never_cut(env, tmp_path):
    st = env["store"]
    for idx, got in cut_batches(env, truncate_min_row_s=100.0):
        assert got["aug"]["truncated"] == 0
    rows = pq.read_table(env["table"]).to_pydict()
    rows["n_frames"] = [f + 1 for f in rows["n_frames"]]  # every row's audio now "decodes" to other frames
    bad = write_table(tmp_path / "bad.parquet", list(zip(*[rows[k] for k in ("id", "n_frames", "lo", "b", "hi", "m")])))
    cuts = A.build_cut_index(bad, [u.id for u in st.utts], tmp_path / "idx", log=lambda s: None)
    ds = T.dataset_for(st, augment=spec(truncate_p=1.0, truncate_min_s=0.0, truncate_min_frac=0.0), cuts=cuts)
    got = [ds[idx]["aug"] for idx in micro_batches(st)]
    assert sum(a["truncated"] for a in got) == 0 and sum(a["cut_mismatch"] for a in got) == sum(a["rows"] for a in got)


def test_a_join_concatenates_the_tokens_at_frame_aligned_offsets(env):
    """concat_p 1: every joined row's tokens are its pieces' in order, each piece's EOS dropped but the last's, their
    top-k rows with them; piece j's audio starts at FRAME_SAMPLES x the frames before it; the rows x longest decoder
    rectangle never exceeds the planned micro-batch's."""
    st = env["store"]
    ds = T.dataset_for(st, augment=spec(concat_p=1.0))
    groups = 0
    for idx in micro_batches(st):
        got = ds[idx]
        groups += got["aug"]["concat_groups"]
        n = got["n_tok"].tolist()
        off = np.concatenate([[0], np.cumsum(n)])
        planned = max(st.utts[i].n_tok for i in idx) * len(idx)
        assert max(n) * len(n) <= planned
        for b, rid in enumerate(got["ids"]):
            pieces = [next(i for i in idx if st.utts[i].id == x) for x in rid.split("+")]
            want = sum((st.targets(i)[0].tolist()[:-1] for i in pieces[:-1]), []) + st.targets(pieces[-1])[0].tolist()
            assert got["top_idx"][off[b]:off[b + 1], 0].tolist() == want
            w = got["wave"][b].numpy()
            start = 0
            for i in pieces:
                own = st.wave(i)
                assert np.array_equal(w[start:start + min(len(own), FS * T.ctc_frames(len(own)))],
                                      own[:FS * T.ctc_frames(len(own))])
                start += FS * T.ctc_frames(len(own))
    assert groups > 0


def test_the_decoder_cap_refuses_a_join(env):
    st = env["store"]
    ds = T.dataset_for(st, augment=spec(concat_p=1.0, max_tokens=4))
    got = [ds[idx] for idx in micro_batches(st)]
    assert all(g["aug"]["concat_groups"] == 0 for g in got) and sum(g["aug"]["concat_capped"] for g in got) > 0
    assert all("+" not in x for g in got for x in g["ids"])


def test_voiced_end():
    """The sample after the last 10 ms frame within END_TRIM_DB of the loudest: a tone's end before 0.3 s of near
    silence (to the frame), the whole row when nothing is quieter (a tone to its end, digital silence) or the row is
    under 5 frames."""
    sr = T.TARGET_SR
    t = np.arange(int(0.5 * sr)) / sr
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    quiet = (1e-5 * np.random.default_rng(0).standard_normal(int(0.3 * sr))).astype(np.float32)
    assert T.voiced_end(np.concatenate([tone, quiet])) == len(tone)
    assert T.voiced_end(tone) == len(tone)
    assert T.voiced_end(np.zeros(sr, np.float32)) == sr
    assert T.voiced_end(tone[:4 * T.HOP]) == 4 * T.HOP


def quiet_tail(monkeypatch, s: float = 0.4):
    """Every decoded row gets s seconds of near silence after it (the fake corpus's rows are a tone to their end)."""
    real = T.decode_audio

    def decode(b):
        w = real(b)
        q = 1e-5 * np.random.default_rng(len(w)).standard_normal(int(s * T.TARGET_SR))
        return np.concatenate([w, q.astype(np.float32)])

    monkeypatch.setattr(T, "decode_audio", decode)


def test_end_trim_shortens_whole_rows_and_keeps_their_tokens(env, monkeypatch, tmp_path):
    """end_trim_p 1 with quiet tails: every row that was not cut ends END_TRIM_TAIL_S after its last voiced frame, with
    its stored tokens (mark and EOS) and its audio's duration; a cut row (truncate_p 0.5, its table built on the same
    audio) is never trimmed: its audio still ends where its cut frame does."""
    quiet_tail(monkeypatch)
    st = env["store"]
    lo, hi = (int(round(x * T.TARGET_SR)) for x in T.END_TRIM_TAIL_S)
    cuts = A.build_cut_index(store_table(st, tmp_path / "cuts.parquet"), [u.id for u in st.utts], tmp_path / "idx",
                             log=lambda s: None)
    for truncate in (0.0, 0.5):
        ds = T.dataset_for(st, augment=spec(end_trim_p=1.0, truncate_p=truncate, truncate_min_s=0.0,
                                            truncate_min_frac=0.0), cuts=cuts if truncate else None)
        trimmed = cut = 0
        for idx in micro_batches(st)[:10]:
            got = ds[idx]
            n = got["n_tok"].tolist()
            off = np.concatenate([[0], np.cumsum(n)])
            for b, i in enumerate(idx):
                L, full = int(got["lengths"][b]), st.wave(i)
                tok = got["top_idx"][off[b]:off[b + 1], 0].tolist()
                if tok != st.targets(i)[0].tolist():  # cut: its audio ends where its cut frame does, untrimmed
                    cut += 1
                    assert T.ctc_frames(L) < T.ctc_frames(len(full))
                    continue
                end = T.voiced_end(full)
                assert end + lo <= L <= end + hi and L < len(full), (L, end, len(full))
                assert got["durations"][b] == pytest.approx(L / T.TARGET_SR)
                trimmed += 1
            assert got["aug"]["end_trimmed"] == len(idx) - got["aug"]["truncated"]
        assert trimmed > 0 and (cut > 0) == bool(truncate)


def test_background_on_aed_rows(env, tmp_path):
    """noise_p 1 on a token store: the background under every row, its tokens and length unchanged."""
    from kitsune.noise_bank import NoiseBank

    bank = NoiseBank.load(make_noise_bank(tmp_path / "bank"))
    st = env["store"]
    ds = T.dataset_for(st, augment=spec(noise_p=1.0), noise=bank)
    plain = T.dataset_for(st)
    for idx in micro_batches(st)[:5]:
        got, ref = ds[idx], plain[idx]
        assert got["aug"]["noised"] == len(idx) and not torch.equal(got["wave"], ref["wave"])
        assert torch.equal(got["lengths"], ref["lengths"]) and torch.equal(got["top_idx"], ref["top_idx"])


def test_same_index_list_same_batch(env):
    st = env["store"]
    a = spec(truncate_p=0.6, truncate_min_s=0.2, concat_p=0.7, mix_p=0.5, end_trim_p=0.5)
    ds = T.dataset_for(st, augment=a, cuts=env["cuts"])
    other = pickle.loads(pickle.dumps(ds))  # what a spawn worker gets
    for idx in micro_batches(st)[:8]:
        x, y = ds[idx], other[idx]
        for k, v in x.items():
            assert (torch.equal(v, y[k]) if isinstance(v, torch.Tensor) else v == y[k]), k
    z = T.dataset_for(st, augment=spec(seed=6, truncate_p=0.6, truncate_min_s=0.2, concat_p=0.7, mix_p=0.5,
                                       end_trim_p=0.5),
                      cuts=env["cuts"])
    assert any(not torch.equal(ds[idx]["wave"], z[idx]["wave"]) or ds[idx]["ids"] != z[idx]["ids"]
               for idx in micro_batches(st)[:8])


# ---------------------------------------------------------------------------------------------------- the trainer


def test_validate_the_aed_augment_block():
    m = load_script("04_distill")
    ok = m.load_config(None, ["augment.enabled=true", "augment.concat_p=0.5"])  # family aed: joins need no table
    assert m.augment_on(ok)
    ok = m.load_config(None, ["augment.enabled=true", "augment.truncate_p=0.2", "augment.cuts=labels/cuts.parquet",
                              "augment.cuts_sha256=" + "a" * 64])
    assert ok["augment"]["cuts"] == "labels/cuts.parquet"
    for bad, match in ((["augment.enabled=true", "augment.truncate_p=0.2"], "needs augment.cuts"),
                       (["augment.enabled=true", "augment.truncate_pad_p=0.2"], "must be 0 on an AED student"),
                       (["augment.enabled=true", "augment.end_pad_p=0.2"], "must be 0 on an AED student"),
                       (["augment.cuts_sha256=abc"], "cuts_sha256"),
                       (["family=ctc", "parakeet_root=po", "augment.cuts=x.parquet"], "AED student's"),
                       (["augment.end_trim_p=1.5"], "augment.end_trim_p")):
        with pytest.raises(SystemExit, match=match):
            m.load_config(None, bad)


def tiny_aed_student(root: Path):
    """A tiny Cohere-shaped student saved as 03 saves one (tests/test_e2e_tiny.py's), or skip without the processor."""
    try:
        from transformers import AutoProcessor, CohereAsrConfig, CohereAsrForConditionalGeneration

        from kitsune import student as S

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:
        pytest.skip(f"teacher processor not in the local HF cache: {e}")
    torch.manual_seed(0)
    cfg = CohereAsrConfig().to_dict()
    enc = dict(cfg["encoder_config"], num_hidden_layers=2, hidden_size=64, intermediate_size=128, num_attention_heads=2,
               num_key_value_heads=2, subsampling_conv_channels=8)
    cfg.update(encoder_config=enc, num_hidden_layers=1, hidden_size=64, intermediate_size=128, num_attention_heads=2,
               num_key_value_heads=2, head_dim=32)
    tcfg = CohereAsrConfig.from_dict(cfg)
    tcfg._attn_implementation = tcfg.encoder_config._attn_implementation = "sdpa"
    teacher = CohereAsrForConditionalGeneration(tcfg).eval()
    student = S.build_student(teacher, S.StudentSpec(enc_layers=[0, 1], ffn_dim=128, dec_layers=[0]), None)
    sdir = root / "student"
    S.save_student(student, sdir, proc, dict(format=1, stage="complete", spec=dict(enc_layers=[0, 1], ffn_dim=128,
                                                                                    dec_layers=[0], tie_head=True)))
    return sdir


def test_an_aed_run_with_every_augmentation(env):
    """A tiny AED student trained with joins, cuts at the table's frames and mixing: the smoke phase passes on the
    plain dataset, the `augment` event records the family, max_tokens and the table's coverage, every step row has the
    aug/* shares in [0, 1] (each above 0 somewhere) and the AED counts, and the train records carry joined rows."""
    root, fc = env["root"], env["fc"]
    sdir = tiny_aed_student(root)
    sha = A.sha256_file(env["table"])
    config = {
        "run_name": "aed-aug", "student": str(sdir), "data_root": str(fc.data), "teacher_root": str(fc.teacher_out),
        "second_root": str(fc.second_out), "selection": str(env["sel"]), "cache_dir": str(root / "run_cache"),
        "runs_root": str(root / "runs"), "sources": ["src_a", "src_b"], "eval_sets": EVAL,
        "device": "cpu", "autocast": "none", "optim": {"lr": 3e-3},
        "schedule": {"warmup_steps": 2, "cooldown_frac": 0.3, "clock": "steps", "max_steps": 6},
        "batch": {"step_audio_s": 6, "micro_audio_s": 3, "pool_micro": 4},
        "perf": {"num_workers": 0, "prefetch": 2, "tf32": False},
        "eval": {"every_min": None, "every_steps": None, "greedy_subset": 3, "batch_s": 20, "check_baselines": False},
        "ckpt": {"weights_every_min": None, "full_local_every_min": None, "keep_local": 2, "full_after_smoke": False,
                 "upload_full_at": []},
        "log": {"layer_stats_every": 1000, "hist_every": 1000, "train_utts_flush": 5, "sync_every_min": 0.005,
                "samples_per_eval": 2, "capture_env": False},
        "hf": {"output_repo": None},
        "smoke": {"enabled": True, "steps": 3, "min_audio_s_per_s": 0, "require_loss_decrease": False, "pad_utts": 4,
                  "decode_per_set": 2},
        # concat_p 0.5: a joined micro-batch of these 3 s micro-batches is mostly one row, and mix needs two
        "augment": {"enabled": True, "truncate_p": 0.7, "truncate_min_s": 0.3, "concat_p": 0.5, "mix_p": 1.0,
                    "end_trim_p": 0.3, "cuts": str(env["table"]), "cuts_sha256": sha, "noise_p": 0.5,
                    "noise_bank": str(make_noise_bank(root / "noise_bank"))},
    }
    path = root / "aed-aug.json"
    path.write_text(json.dumps(config, indent=1), encoding="utf-8")
    m = load_script("04_distill")
    assert m.main(["--config", str(path)]) == 0
    run = next((root / "runs").glob("aed-aug-2*"))
    evs = [json.loads(x) for x in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    kinds = [e["kind"] for e in evs]
    assert kinds[-1] == "logger_close" and "tb_tag_unmapped" not in kinds and "nonfinite_grad_skipped" not in kinds
    aug = next(e for e in evs if e["kind"] == "augment")
    assert aug["family"] == "aed" and aug["max_tokens"] == 200 - (len(T.PROMPT) - 1) and aug["cuts_sha256"] == sha
    assert aug["cuts_rows_with_cuts"] > 0 and aug["cuts_rows"] == aug["cuts_rows_in_table"]
    smoke = {e["kind"]: e for e in evs if e["kind"].startswith("smoke_")}
    assert smoke["smoke_steps"]["steps"] == 3 and smoke["smoke_steps"]["dropped"] == 0
    st = pd.read_parquet(run / "metrics" / "steps.parquet")
    assert st["step"].tolist() == [1, 2, 3, 4, 5, 6] and np.isfinite(st["loss/objective"]).all()
    for tag in ("aug/concat_frac", "aug/truncated_frac", "aug/mixed_frac"):
        assert ((st[tag] >= 0) & (st[tag] <= 1)).all() and st[tag].max() > 0, tag
    assert (st["aug/cut_mismatch"] == 0).all() and (st["aug/cut_padded_frac"] == 0).all()
    # the fake rows are a tone to their end: nothing to trim, but the share is logged
    assert ((st["aug/end_trimmed_frac"] >= 0) & (st["aug/end_trimmed_frac"] <= 1)).all() and aug["end_trim_p"] == 0.3
    assert st["aug/noised_frac"].max() > 0 and aug["noise_p"] == 0.5 and aug["noise_clips"] == 2
    utts = pd.concat([pd.read_parquet(p) for p in sorted((run / "metrics" / "train_utts").glob("part-*.parquet"))])
    ids = {u.id for u in env["store"].utts}
    assert all(x in ids for r in utts["id"] for x in r.split("+")) and (utts["n_tok"] > 0).all()
    assert json.loads((run / "summary.json").read_text(encoding="utf-8"))["status"] == "complete"
