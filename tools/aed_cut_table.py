"""The AED cut table: for every kept train row of a selection, the frames where recipe v2's truncate may cut it and the
Cohere tokens each cut keeps (kitsune/aed_cuts.py: why, the rule, the table's columns). The T students' augment.cuts.

Inputs: the selection (its kept split-"train" rows, grouped by teacher_file), the label root holding teacher_out
(Cohere's tokens) and parakeet_out (the Parakeet CTC targets) for those stems, Cohere's tokenizer.json and the Parakeet
vocabulary (the .nemo's SentencePiece tokenizer.vocab, one piece per line in id order, or a tokenizer.json). The
Parakeet punctuation ids - the tokens a cut never removes alone - are the vocabulary's pieces whose text is a sentence
mark or comma (PUNCT_TEXT, scripts/04_distill.py punct_token_ids' list): 。 、 ? ! in the ja vocabulary, ids 1, 8, 25, 27.

Writes the table (parquet, one row per train row in selection order; aed_cuts.TABLE_COLUMNS) and <out>.json: the
inputs' sha256, the punctuation ids, per source the rows, the rows >= 3 s, the rows with any entry and with a pause
entry, the entries, the word tokens and how many mapped, and the table's sha256 (what augment.cuts_sha256 pins).

Usage (CPU; the full selection's 7.0M rows take about an hour per worker process):
  python tools/aed_cut_table.py --selection full.parquet --labels D:/kitsune-labels/full \
      --cohere-tokenizer <snapshot>/tokenizer.json --parakeet-vocab <snapshot>/<hash>_tokenizer.vocab \
      --out aed_cuts.parquet --workers 2
"""
import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kitsune import aed_cuts  # noqa: E402

PUNCT_TEXT = ("。", "?", "!", "？", "！", ".", "．", "、", ",", "，")  # == scripts/04_distill.py PUNCT_TEXT
COHERE_EOS = 3
TRUNCATE_PAUSE_FRAMES = 4  # == kitsune.trainset.TRUNCATE_PAUSE_FRAMES (not imported: it pulls in torch)
FRAME_S = 0.08
ROWS_PER_GROUP = 200_000  # parquet row group


def read_vocab(path) -> list[str]:
    """Piece strings in id order: a SentencePiece .vocab (piece<TAB>score per line) or a tokenizers tokenizer.json."""
    path = Path(path)
    if path.suffix == ".json":
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(path))
        return [tok.id_to_token(i) or "" for i in range(tok.get_vocab_size())]
    return [ln.rstrip("\n").split("\t")[0] for ln in path.open(encoding="utf-8")]


def punct_ids(pieces: list[str]) -> list[int]:
    return [i for i, p in enumerate(pieces) if p.replace("\u2581", "") in PUNCT_TEXT]


def read_train_rows(selection) -> "pd.DataFrame":
    import pandas as pd

    sel = pd.read_parquet(selection, columns=["id", "source", "split", "teacher_file", "keep"])
    return sel[sel["keep"].astype(bool) & (sel["split"] == "train")].reset_index(drop=True)


_W = {}


def _init(labels, pk_texts, pk_punct, co_texts):
    _W.update(labels=Path(labels), pk_texts=pk_texts, pk_punct=pk_punct, co_texts=co_texts)


def stem_rows(job: tuple[str, list[str]]) -> dict:
    """One teacher_file's rows (ids in selection order): their entries and counts."""
    from kitsune.parakeet_targets import ctc_col0

    stem, ids = job
    labels = _W["labels"]
    pz = np.load(labels / "parakeet_out" / f"{stem}.npz")
    tz = np.load(labels / "teacher_out" / f"{stem}.npz")
    p_row = {x: j for j, x in enumerate(pz["ids"].tolist())}
    t_row = {x: j for j, x in enumerate(tz["ids"].tolist())}
    n_frames, f_off, d_off = pz["n_frames"], pz["frame_offsets"], pz["dense_offsets"]
    dense, top1 = pz["ctc_dense_frame"], pz["ctc_topk_idx"][:, 0]
    t_off, t_tok = tz["tok_offsets"], tz["tokens"]
    out = dict(stem=stem, ids=[], n_frames=[], lo=[], b=[], hi=[], m=[], missing=[], no_eos=0, word_tokens=0)
    for x in ids:
        i, j = p_row.get(x), t_row.get(x)
        if i is None or j is None:  # no label of one teacher: in the table without cuts (n_frames -1)
            out["missing"].append(x)
            out["ids"].append(x)
            out["n_frames"].append(-1)
            for k in aed_cuts.ENTRY_ARRAYS:
                out[k].append(np.zeros(0, np.int16))
            continue
        T = int(n_frames[i])
        if int(f_off[i + 1]) - int(f_off[i]) != T:
            raise ValueError(f"{stem}.npz: {x} has {int(f_off[i + 1]) - int(f_off[i])} frames stored, n_frames {T}")
        col0 = ctc_col0(T, dense[d_off[i]:d_off[i + 1]], top1[d_off[i]:d_off[i + 1]][:, None])
        tok = t_tok[int(t_off[j]):int(t_off[j + 1])]
        if not len(tok) or int(tok[-1]) != COHERE_EOS:
            out["no_eos"] += 1  # a teacher-truncated transcript: never cut (the selection keeps none)
            ent = (np.zeros(0, np.int16),) * 4
        else:
            ent = aed_cuts.row_entries(col0, _W["pk_texts"], _W["pk_punct"], tok[:-1], _W["co_texts"])
        starts = aed_cuts.token_runs(col0)[0]
        out["word_tokens"] += int(np.sum(~np.isin(col0[starts[1:]], _W["pk_punct"]))) if len(starts) > 1 else 0
        out["ids"].append(x)
        out["n_frames"].append(T)
        for k, v in zip(aed_cuts.ENTRY_ARRAYS, ent):
            out[k].append(v)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selection", required=True)
    ap.add_argument("--labels", required=True, help="the label root: teacher_out/ and parakeet_out/ below it")
    ap.add_argument("--cohere-tokenizer", required=True)
    ap.add_argument("--parakeet-vocab", required=True)
    ap.add_argument("--out", required=True, help="the table (.parquet); its report goes to <out>.json")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit-stems", type=int, default=None, help="only the first N teacher_files (a quick check)")
    a = ap.parse_args(argv)
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.time()
    co_pieces, pk_pieces = read_vocab(a.cohere_tokenizer), read_vocab(a.parakeet_vocab)
    if len(pk_pieces) != aed_cuts.PK_BLANK:
        raise SystemExit(f"{a.parakeet_vocab}: {len(pk_pieces)} pieces, the Parakeet CTC head has {aed_cuts.PK_BLANK} "
                         "non-blank classes")
    pk_punct = punct_ids(pk_pieces)
    co_texts = [aed_cuts.piece_text(p) for p in co_pieces]
    pk_texts = [aed_cuts.piece_text(p) for p in pk_pieces]
    rows = read_train_rows(a.selection)
    stems = list(dict.fromkeys(rows["teacher_file"].tolist()))  # selection order
    if a.limit_stems:
        stems = stems[:a.limit_stems]
    by_stem = rows.groupby("teacher_file", sort=False)["id"].apply(list).to_dict()
    source_of = dict(zip(rows["id"], rows["source"]))
    jobs = [(s, by_stem[s]) for s in stems]
    print(f"aed cut table: {sum(len(j[1]) for j in jobs)} train rows in {len(jobs)} stems, Parakeet punctuation "
          f"{ {i: pk_pieces[i] for i in pk_punct} }, {a.workers} worker(s)", flush=True)

    schema = pa.schema([("id", pa.string()), ("n_frames", pa.int16())]
                       + [(k, pa.list_(pa.int16())) for k in aed_cuts.ENTRY_ARRAYS])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    stats = defaultdict(lambda: defaultdict(int))
    missing, no_eos = [], 0
    buf = defaultdict(list)

    def flush(writer):
        if buf["id"]:
            writer.write_table(pa.table({k: buf[k] for k in schema.names}, schema=schema))
            buf.clear()

    if a.workers > 1:
        from multiprocessing import get_context

        pool = get_context("spawn").Pool(a.workers, initializer=_init,
                                          initargs=(a.labels, pk_texts, pk_punct, co_texts))
        results = pool.imap(stem_rows, jobs, chunksize=1)
    else:
        _init(a.labels, pk_texts, pk_punct, co_texts)
        pool, results = None, map(stem_rows, jobs)
    done = 0
    with pq.ParquetWriter(tmp, schema) as writer:
        for r in results:
            missing += r["missing"]
            no_eos += r["no_eos"]
            for q, x in enumerate(r["ids"]):
                st = stats[source_of[x]]
                lo, b, hi, m = (r[k][q] for k in aed_cuts.ENTRY_ARRAYS)
                st["rows"] += 1
                st["missing"] += r["n_frames"][q] < 0
                st["rows_3s"] += r["n_frames"][q] * FRAME_S >= 3.0 - 1e-9
                st["rows_with_entries"] += len(lo) > 0
                st["rows_3s_with_entries"] += len(lo) > 0 and r["n_frames"][q] * FRAME_S >= 3.0 - 1e-9
                st["rows_with_pause_entry"] += bool(np.any((hi.astype(np.int64) - b) >= TRUNCATE_PAUSE_FRAMES))
                st["entries"] += len(lo)
                st["cut_frames"] += int(np.sum(hi.astype(np.int64) - lo + 1))
                buf["id"].append(x)
                buf["n_frames"].append(int(r["n_frames"][q]))
                for k in aed_cuts.ENTRY_ARRAYS:
                    buf[k].append(r[k][q])
            stats[source_of[r["ids"][0]] if r["ids"] else "?"]["word_tokens"] += r["word_tokens"]
            if len(buf["id"]) >= ROWS_PER_GROUP:
                flush(writer)
            done += 1
            if done % 100 == 0 or done == len(jobs):
                n = sum(s["rows"] for s in stats.values())
                print(f"  {done}/{len(jobs)} stems, {n} rows, {time.time() - t0:.0f} s", flush=True)
        flush(writer)
    if pool is not None:
        pool.close()
        pool.join()
    tmp.replace(out)
    sha = aed_cuts.sha256_file(out)
    report = dict(
        table=str(out), table_sha256=sha, rows=sum(s["rows"] for s in stats.values()), stems=len(jobs),
        selection=str(a.selection), selection_sha256=aed_cuts.sha256_file(a.selection),
        cohere_tokenizer_sha256=aed_cuts.sha256_file(a.cohere_tokenizer),
        parakeet_vocab_sha256=aed_cuts.sha256_file(a.parakeet_vocab),
        parakeet_punct_ids={str(i): pk_pieces[i] for i in pk_punct}, ctx=aed_cuts.CTX, no_eos=no_eos,
        missing=len(missing), missing_ids=missing[:50],
        per_source={src: dict(s) for src, s in sorted(stats.items())}, seconds=round(time.time() - t0, 1))
    Path(str(out) + ".json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for src, s in sorted(stats.items()):
        r3 = max(s["rows_3s"], 1)
        print(f"{src}: {s['rows']} rows, >= 3 s {s['rows_3s']}, with entries "
              f"{s['rows_with_entries'] / max(s['rows'], 1):.3f}"
              f" ({s['rows_3s_with_entries'] / r3:.3f} of those >= 3 s), with a pause entry "
              f"{s['rows_with_pause_entry'] / max(s['rows'], 1):.3f}, mapped word tokens "
              f"{s['entries'] / max(s['word_tokens'], 1):.3f}, missing {s['missing']}")
    print(f"wrote {out} ({out.stat().st_size / 2**20:.1f} MiB, sha256 {sha}) in {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
