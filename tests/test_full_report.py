"""tools/full_report.py: the full-data runs' report (CONTRACT.md 10), on synthetic inputs shaped exactly like what the
full box writes: 05_evaluate --out readouts (study.json with WP5's quant / weights blocks, tables/<system>/<set>.parquet
or only greedy_<set>.parquet, summary.json's tf.sets KL), whisper_eval dirs (whisper.json, greedy parquets without
teacher_hyp), speed_probe files from several runs/speed-*/ dirs and machines (the wave-1 records with only
versions.host, as smoke A writes them, and the full boxes' queue summaries that give their machine), and the
trainer's summary.json.

Two worlds. The TEXT world scores real ref / hyp strings (kitsune.evaluate.utterance_table), so the metrics, the
hallucination and kana counts can be checked against sums taken here by hand. The PLANTED world reuses
test_study_stats.planted: every system's edits are the same base edits times a multiplier, so every ratio is exact and
v_boot is 0, and a CI's half-width is 1.96 sqrt(k) sigma_run: that pins the noise term k of each comparison."""
import hashlib
import importlib.util
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fixtures import ROOT, no_real_data
from kitsune import evaluate as ev
from kitsune import study_stats as ss
from kitsune.store import ids_sha256
from test_study_stats import planted


def _load():
    spec = importlib.util.spec_from_file_location("full_report", ROOT / "tools" / "full_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FR = _load()

SETS = {"eval_jsut": 24, "eval_cv8": 16, "eval_reazon": 16, "eval_emilia": 6, "galgame": 14}
N_NEUTRAL = 10
KANA = "あいうえおかきくけこさしすせそたちつてとなにぬねの"
KANJI = "日本語漢字東京大学先生時間"
STAMP = "20261003T000000Z"
T0 = "2026-10-03T00:00:00+00:00"
PRIMARY = "m54650 / NVIDIA GeForce RTX 5090"


# ------------------------------------------------------------------------------------------------ builders


def text_corpus(seed: int = 0):
    """(ids, refs, manifest): random kana/kanji references (every system scores the same ones); the last Galgame row
    has an empty reference, the neutral view is the first N_NEUTRAL Galgame rows."""
    rng = np.random.default_rng(seed)
    ids = {s: [f"{s}/u{i:03d}" for i in range(n)] for s, n in SETS.items()}
    alphabet = list(KANA + KANJI)
    refs = {s: ["".join(rng.choice(alphabet, int(rng.integers(8, 20)))) for _ in ids[s]] for s in SETS}
    refs["galgame"][-1] = ""
    refs["eval_cv8"][1] = "今日の字幕を読みます"  # holds a SILENCE_PHRASES entry: said, not hallucinated
    refs["eval_jsut"][3] = "東京大学の先生の時間です"  # 8 kanji: the planted kana-mode row
    manifest = {"sets": {s: {"ids": v, "ids_sha256": ids_sha256(v)} for s, v in ids.items()},
                "galgame_views": {"neutral": ids["galgame"][:N_NEUTRAL]}}
    return ids, refs, manifest


def noisy(refs: dict, rate: float, seed: int) -> dict:
    """Each reference character replaced by a random kana with probability rate."""
    rng = np.random.default_rng(seed)
    return {s: ["".join(c if rng.random() >= rate else KANA[int(rng.integers(len(KANA)))] for c in r) for r in rs]
            for s, rs in refs.items()}


def text_table(ids: dict, refs: dict, hyps: dict, truncated: dict | None = None) -> pd.DataFrame:
    return pd.concat([ev.utterance_table(ids[s], s, refs[s], hyps[s], (truncated or {}).get(s)) for s in ids],
                     ignore_index=True)


def write_tables(root: Path, tables: dict) -> Path:
    """The study layout: <root>/<system>/<set>.parquet."""
    for system, df in tables.items():
        d = root / system
        d.mkdir(parents=True, exist_ok=True)
        for s, g in df.groupby("set"):
            g.reset_index(drop=True).to_parquet(d / f"{s}.parquet", index=False)
    return root


def write_readout(root: Path, name: str, system: str, table: pd.DataFrame | None = None, *, greedy=None,
                  family=None, time_utc=T0, summary=None, whisper=None, **study_extra) -> Path:
    """A 05_evaluate --out dir: study.json, tables/<system>/<set>.parquet (when table), greedy_<set>.parquet (when
    greedy: set -> frame), summary.json, whisper.json."""
    d = root / name
    d.mkdir(parents=True)
    if table is not None:
        write_tables(d / "tables", {system: table})
    for s, g in (greedy or {}).items():
        g.to_parquet(d / f"greedy_{s}.parquet", index=False)
    # a manifest sha256 other than the manifest file's: the readout_manifests check fails in the text world
    rec = dict(system=system, family=family, teacher=None, manifest=dict(sha256="x" * 64), time_utc=time_utc,
               **study_extra)
    (d / "study.json").write_text(json.dumps(rec), encoding="utf-8")
    if summary is not None:
        (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if whisper is not None:
        (d / "whisper.json").write_text(json.dumps(whisper), encoding="utf-8")
    return d


def greedy_frames(ids, refs, hyps, *, duration=5.0, n_tok=10, durations=None, truncated=None) -> dict:
    """whisper_eval / 05 greedy_<set>.parquet frames (no teacher_hyp)."""
    out = {}
    for s in ids:
        n = len(ids[s])
        out[s] = pd.DataFrame(dict(id=ids[s], source=s, duration=[(durations or {}).get(i, duration) for i in ids[s]],
                                   ref=refs[s], hyp=hyps[s],
                                   truncated=(truncated or {}).get(s, [False] * n), n_tok=[n_tok] * n))
    return out


def srec(rtf, *, machine="m54650", host="c0ffee", gpu="NVIDIA GeForce RTX 5090", p50_s=None, tokens=None, t=T0,
         kind="ctc", **kw) -> dict:
    """A tools/speed_probe.py record. machine=None: the wave-1 speed_probe's versions (the container host only; WP6
    adds machine_id), as smoke A writes them."""
    p50 = rtf * 40 if p50_s is None else p50_s
    versions = dict(host=host) if machine is None else dict(host=host, machine_id=machine)
    return dict(kind=kind, rtf=rtf, p50_s=p50, p95_s=1.2 * p50, vram_peak_reserved_bytes=3_000_000_000,
                vram_peak_reserved_bytes_1=800_000_000, vram_gb=3.0, weights_bytes=600_000_000,
                params_total=300_000_000, tokens_per_utt=tokens, gpu=gpu, versions=versions, time_utc=t, **kw)


def queue_summary(path: Path, box: str, machine_id: str, started_utc: str, speed_dir: str | None = None) -> Path:
    """A full box's full/box-<box>/queue_summary.json (CONTRACT.md 5): machine_id, started (unix seconds) and the
    speed item's out dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    items = {"speed-study-p03": dict(kind="speed", status="done", out=speed_dir,
                                     result=dict(system="study-p03", out=speed_dir))} if speed_dir else {}
    started = datetime.fromisoformat(started_utc).timestamp()
    path.write_text(json.dumps(dict(format=1, kind="full", box=box, status="complete", machine_id=machine_id,
                                    started=started, items=items)), encoding="utf-8")
    return path


def write_speed(path: Path, ids: list[str], systems: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(schema=1, ids=ids, ids_sha256=ids_sha256(ids), systems=systems)),
                    encoding="utf-8")
    return path


def write_summary(root: Path, run_name: str, **fields) -> None:
    """The trainer's runs/<run_id>/summary.json."""
    d = root / f"{run_name}-{STAMP}"
    d.mkdir(parents=True)
    s = dict(status="complete", run_id=d.name, steps=1000, epochs=3.0, stopped_early=None, early_stop_trigger=None,
             config=dict(run_name=run_name))
    s.update(fields)
    (d / "summary.json").write_text(json.dumps(s), encoding="utf-8")


def tf_summary(kls: dict) -> dict:
    """A 05 summary.json with tf.sets.<set>.{kl, n_tok}."""
    return dict(tf=dict(sets={s: dict(kl=k, n_tok=n) for s, (k, n) in kls.items()}))


def run(args: list, out: Path) -> tuple[int, dict | None]:
    rc = FR.main([*map(str, args), "--out", str(out)])
    rep = json.loads((out / "report.json").read_text(encoding="utf-8")) if rc == 0 else None
    return rc, rep


def hand_cer(df: pd.DataFrame, s: str, ids=None, col="edits") -> float:
    t = df[df["set"] == s]
    if ids is not None:
        t = t[t["id"].isin(set(ids))]
    t = t[t["ref_len"] > 0]
    return t[col].sum() / t["ref_len"].sum()


# ------------------------------------------------------------------------------------------------ the text world


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """Every input kind at once (module docstring), and one report over it."""
    d = tmp_path_factory.mktemp("full_report_world")
    ids, refs, manifest = text_corpus()
    man = d / "study_manifest.json"
    man.write_text(json.dumps(manifest), encoding="utf-8")
    T = {}
    for i, (name, rate) in enumerate((("cohere", 0.05), ("parakeet-ctc", 0.06), ("parakeet-tdt", 0.065),
                                      ("study-t06", 0.10), ("study-p03", 0.09), ("study-p01", 0.12),
                                      ("study-p03-half", 0.095))):
        T[name] = text_table(ids, refs, noisy(refs, rate, i))
    tables = write_tables(d / "tables", T)

    pull = d / "pull" / "runs"
    R = {"full-p03": text_table(ids, refs, noisy(refs, 0.08, 20)),
         "full-p03@int8-w8a8": text_table(ids, refs, noisy(refs, 0.085, 21)),
         "full-p03@nvfp4-w4a4": text_table(ids, refs, noisy(refs, 0.09, 22)),
         "full-p03@mxfp4-w4a4": text_table(ids, refs, noisy(refs, 0.2, 23)),
         "whisper-small": text_table(ids, refs, noisy(refs, 0.15, 24)),
         "kotoba-whisper-v2.0": text_table(ids, refs, noisy(refs, 0.07, 25))}
    kl = {"eval_jsut": (0.1, 100), "eval_cv8": (0.2, 300), "eval_reazon": (0.3, 100)}
    write_readout(pull, f"m4-full-p03-{STAMP}", "full-p03", R["full-p03"], family="ctc", summary=tf_summary(kl),
                  weights=dict(path="x", file_bytes=617_000_000))
    q = dict(impl="torchao", simulated=False, nonfinite=dict(rows=0), share_quantized=0.95)
    write_readout(pull, f"quant-int8-w8a8-full-p03-{STAMP}", "full-p03@int8-w8a8", R["full-p03@int8-w8a8"],
                  family="ctc", summary=tf_summary({k: (v + 0.05, n) for k, (v, n) in kl.items()}),
                  quant=dict(q, format="int8-w8a8", base_system="full-p03", file_bytes=312_000_000, fp32_fallbacks=2))
    write_readout(pull, f"quant-nvfp4-w4a4-full-p03-{STAMP}", "full-p03@nvfp4-w4a4", R["full-p03@nvfp4-w4a4"],
                  family="ctc", quant=dict(q, format="nvfp4-w4a4", base_system="full-p03", file_bytes=180_000_000,
                                           fp32_fallbacks=0))
    write_readout(pull, f"quant-mxfp4-w4a4-full-p03-{STAMP}", "full-p03@mxfp4-w4a4", R["full-p03@mxfp4-w4a4"],
                  family="ctc", quant=dict(q, format="mxfp4-w4a4", impl="emulate", simulated=True,
                                           base_system="full-p03", file_bytes=170_000_000))
    write_readout(pull, f"whisper-small-{STAMP}", "whisper-small", R["whisper-small"], family="whisper",
                  whisper=dict(model=dict(params_total=241_734_912, weights_file_bytes=483_000_000, dtype="bf16")))
    write_readout(pull, f"kotoba-whisper-v2.0-{STAMP}", "kotoba-whisper-v2.0", R["kotoba-whisper-v2.0"],
                  family="whisper")

    # full-t06: only greedy parquets (the greedy fallback), 12 tokens a row where the timed study weights did 10
    t06_hyps = noisy(refs, 0.095, 26)
    write_readout(pull, f"m4-full-t06-{STAMP}", "full-t06", greedy=greedy_frames(ids, refs, t06_hyps, n_tok=12),
                  family="aed")
    R["full-t06"] = text_table(ids, refs, t06_hyps)

    # whisper-large-v3: greedy only, with the hallucinations planted (test_hallucination_...)
    wh = {s: list(v) for s, v in refs.items()}
    trunc = {s: [False] * len(v) for s, v in ids.items()}
    wh["eval_jsut"][0] = ""  # empty output, a short clip
    wh["eval_jsut"][1] = refs["eval_jsut"][1] * 3  # runaway (3x the reference), a short clip
    trunc["eval_jsut"][2] = True
    wh["eval_cv8"][0] = refs["eval_cv8"][0] + "ご視聴ありがとうございました"  # silence phrase
    # eval_cv8 row 1 says 字幕 in its reference (text_corpus) and the output repeats it: speech, not a hallucination
    wh["eval_reazon"][0] = "（笑）" + refs["eval_reazon"][0]  # bracketed
    wh["eval_jsut"][3] = "とうきょうだいがくのせんせいのじかんです"  # kana mode: kanji in the reference, none here
    wh["galgame"][12] = ""  # empty output on a Galgame-all row outside the neutral view
    wh["galgame"][-1] = "あ"  # output on the empty reference
    durations = {ids["eval_jsut"][0]: 1.0, ids["eval_jsut"][1]: 1.5, ids["galgame"][12]: 1.2}
    whisper_json = dict(system="whisper-large-v3", family="whisper",
                        model=dict(repo="openai/whisper-large-v3", dtype="bf16", params_total=1_543_490_560,
                                   weights_file_bytes=3_087_000_000),
                        sets={"eval_jsut": dict(n=24, cer_corpus=0.1, n_truncated=1, n_repetition=1)})
    write_readout(pull, f"whisper-large-v3-{STAMP}", "whisper-large-v3", family="whisper", whisper=whisper_json,
                  greedy=greedy_frames(ids, refs, wh, durations=durations, truncated=trunc))
    R["whisper-large-v3"] = text_table(ids, refs, wh, trunc)

    # speed: smoke A and smoke B on one machine (the chart group), box full's re-time pair on another
    sp_ids = [ids[s][i] for s in ("eval_jsut", "eval_cv8", "eval_reazon") for i in range(3)]
    speed = [
        write_speed(pull / "speed-full-smoke-20261001T000000Z" / "speed.json", sp_ids, {
            "study-p03": srec(0.0004), "study-p01": srec(0.0002), "parakeet-ctc": srec(0.0005),
            "parakeet-tdt": srec(0.0006), "study-t06": srec(0.001, tokens=10.0, kind="aed")}),
        write_speed(pull / "speed-smoke-b-20261002T000000Z" / "speed.json", sp_ids, {
            "study-p03@int8-w8a8": srec(0.00045, quant="int8-w8a8"),
            "study-p03@nvfp4-w4a4": srec(0.0005, quant="nvfp4-w4a4"),
            "study-p03@int8-w8a8+compile": srec(0.0003, quant="int8-w8a8", compile=True),
            "whisper-large-v3": srec(0.004, kind="whisper"), "kotoba-whisper-v2.0": srec(0.002, kind="whisper")}),
        write_speed(pull / "speed-full-20261005T000000Z" / "speed.json", sp_ids, {
            "full-t06": srec(0.003, machine="m99999", p50_s=0.09, kind="aed"),
            "study-t06": srec(0.002, machine="m99999", p50_s=0.06, kind="aed"),
            "whisper-small": srec(0.001, machine="m99999", kind="whisper")})]
    write_summary(pull, "full-p03", epochs=2.5, stopped_early=dict(reason="patience", at_step=900),
                  early_stop_trigger=dict(reason="patience", at_step=900), resume_resets=0)
    write_summary(pull, "full-t06", epochs=3.0, resume_resets=1)
    params = d / "params.json"
    params.write_text(json.dumps({"study-t06": 616_963_328, "study-p03": {"params_total": 308_524_033}}),
                      encoding="utf-8")
    args = ["--manifest", man, "--prereg", "none", "--tables", tables, "--readouts", pull,
            "--speed", *speed, "--run-summaries", pull, "--params", params, "--boot-b", 300]
    rc, rep = run(args, d / "out")
    assert rc == 0
    return dict(dir=d, out=d / "out", rep=rep, args=args, T={**T, **R}, ids=ids, refs=refs, manifest=manifest,
                man=man, pull=pull, tables=tables, sp_ids=sp_ids)


def test_metrics_match_hand_sums(world):
    """Every local metric equals the sum / sum per stratum and macro mean taken here, for systems read from the
    study tables, a readout's tables and a readout's greedy parquets alone (scored as 05 scores them)."""
    rep, T = world["rep"], world["T"]
    neutral = world["manifest"]["galgame_views"]["neutral"]
    for s in ("cohere", "study-p03", "full-p03", "full-t06", "whisper-large-v3", "full-p03@int8-w8a8"):
        df = T[s]
        c = {k: hand_cer(df, k) for k in ("eval_jsut", "eval_cv8", "eval_reazon")}
        cn = {k: hand_cer(df, k, col="edits_nostyle") for k in c}
        gn, ga = hand_cer(df, "galgame", neutral), hand_cer(df, "galgame")
        gnn, gan = hand_cer(df, "galgame", neutral, "edits_nostyle"), hand_cer(df, "galgame", col="edits_nostyle")
        m = rep["systems"][s]["metrics"]
        three = [c["eval_jsut"], c["eval_cv8"], c["eval_reazon"]]
        assert m["m4"] == pytest.approx(np.mean(three + [gn]), rel=1e-12)
        assert m["m4_all"] == pytest.approx(np.mean(three + [ga]), rel=1e-12)
        assert m["m3"] == pytest.approx(np.mean(three), rel=1e-12)
        assert m["m4_nostyle"] == pytest.approx(np.mean(list(cn.values()) + [gnn]), rel=1e-12)
        assert m["m4_all_nostyle"] == pytest.approx(np.mean(list(cn.values()) + [gan]), rel=1e-12)
        assert m["m3_nostyle"] == pytest.approx(np.mean(list(cn.values())), rel=1e-12)
        assert m["jg"] == pytest.approx((c["eval_jsut"] + gn) / 2, rel=1e-12)
        assert m["gap_neutral_all"] == pytest.approx(ga - gn, rel=1e-12)
        g = df[df["set"].isin(list(c)) & (df["ref_len"] > 0)]
        assert m["gate_pooled"] == pytest.approx(g["edits"].sum() / g["ref_len"].sum(), rel=1e-12)
        assert rep["systems"][s]["sets"]["galgame_all"]["cer"] == pytest.approx(ga, rel=1e-12)
    # the greedy-only readouts were scored here, the others read as written
    assert rep["systems"]["full-t06"]["table_from"] == "greedy"
    assert rep["systems"]["full-p03"]["table_from"] == "tables"
    # the local metrics never leak into the study's metric table
    assert "m4_all" not in ss.METRICS and "m3" not in ss.METRICS


def test_roles_offer_blocks_and_columns(world):
    rep = world["rep"]
    sysd = rep["systems"]
    assert {s: sysd[s]["role"] for s in ("cohere", "full-p03", "study-p03", "full-p03@int8-w8a8", "whisper-small",
                                          "kotoba-whisper-v2.0")} == {
        "cohere": "teacher", "full-p03": "full", "study-p03": "study", "full-p03@int8-w8a8": "quant",
        "whisper-small": "whisper", "kotoba-whisper-v2.0": "whisper"}
    assert sysd["full-p03@int8-w8a8"]["base"] == "full-p03" and sysd["full-p03@int8-w8a8"]["format"] == "int8-w8a8"
    assert "study-p03-half" not in sysd and rep["left_out"]["half"] == ["study-p03-half"]
    rows = rep["offer"]["rows"]
    blocks = [r["block"] for r in rows]
    assert blocks == sorted(blocks, key=["parakeet", "cohere", "whisper", "study"].index)
    para = [r["system"] for r in rows if r["block"] == "parakeet"]
    assert para == ["parakeet-ctc", "parakeet-tdt", "full-p03", "full-p03@int8-w8a8", "full-p03@nvfp4-w4a4",
                    "full-p03@mxfp4-w4a4"]
    assert [r["system"] for r in rows if r["block"] == "whisper"] == ["whisper-large-v3", "kotoba-whisper-v2.0",
                                                                       "whisper-small"]
    assert rep["offer"]["columns"][0] == "model" and rep["offer"]["columns"][-1] == "flags"
    r = next(r for r in rows if r["system"] == "full-p03")
    # file size from study.json weights, 'x smaller' against the own teacher's (in-memory) bytes
    assert r["file_bytes"] == 617_000_000 and r["file_bytes_from"] == "weights.file_bytes"
    assert r["smaller_than_teacher"] == pytest.approx(600_000_000 / 617_000_000)
    assert r["params"] == 308_524_033  # the study counterpart's, from --params' {"params_total": ...} form
    q = next(r for r in rows if r["system"] == "full-p03@int8-w8a8")
    assert q["file_bytes"] == 312_000_000 and q["params"] == 308_524_033
    w = next(r for r in rows if r["system"] == "whisper-small")
    assert w["params"] == 241_734_912 and w["file_bytes"] == 483_000_000 and w["vs_teacher"] is None
    csv_head = (world["out"] / "offer.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert csv_head[:3] == ["system", "block", "display"] and "vs_study_lo" in csv_head
    assert len((world["out"] / "offer.csv").read_text(encoding="utf-8").splitlines()) == len(rows) + 1


def test_hallucination_silence_bracket_kana_counts(world):
    """The planted Whisper rows are counted exactly; every system's counts equal the definitions applied here."""
    rep, T, ids = world["rep"], world["T"], world["ids"]
    h = rep["systems"]["whisper-large-v3"]["halluc"]
    assert h["m4"]["n_empty"] == 1 and h["m4_all"]["n_empty"] == 2  # the Galgame-all-only row counts in M4-all
    assert h["m4"]["n_truncated"] == 1 and h["m4"]["n_silence"] == 1 and h["m4"]["n_bracketed"] == 1
    assert h["empty_ref"] == dict(n=1, nonempty_on_empty_ref=1)
    assert h["kana_mode"]["n"] == 1
    assert h["m4_short"] == dict(n=2, n_empty=1, n_runaway=1, n_truncated=0, n_silence=0, n_bracketed=0,
                                 n_duration_known=len(ids["eval_jsut"]) + len(ids["eval_cv8"])
                                 + len(ids["eval_reazon"]) + N_NEUTRAL)
    assert h["m4_all_short"]["n"] == 3 and h["m4_all_short"]["n_empty"] == 2
    neutral = set(world["manifest"]["galgame_views"]["neutral"])
    for s in ("whisper-large-v3", "cohere", "full-t06"):
        df = T[s]
        m4 = df[df["set"].isin(["eval_jsut", "eval_cv8", "eval_reazon"])
                | ((df["set"] == "galgame") & df["id"].isin(neutral))]
        m4 = m4[m4["ref_len"] > 0]
        got = rep["systems"][s]["halluc"]["m4"]
        assert got["n"] == len(m4)
        assert got["n_empty"] == int((m4["hyp_len"] == 0).sum())
        assert got["n_runaway"] == int(((m4["edits"] > m4["ref_len"]) | (m4["hyp_len"] > 2 * m4["ref_len"])).sum())
        eligible = sum(sum(ch in KANJI for ch in r) >= 3 for r in df[df["set"] == "eval_jsut"]["ref"])
        assert rep["systems"][s]["halluc"]["kana_mode"]["eligible"] == eligible
    # the clip durations come from any readout's greedy parquets: a system read from tables gets them too
    assert rep["systems"]["cohere"]["halluc"]["m4_short"]["n"] == 2


def test_ci_noise_terms(tmp_path):
    """k per comparison: 1 for a trained system against its teacher, 2 for full vs study, 0 for a variant against
    its own bf16 weights; Whisper has no own teacher. With v_boot 0 the CI is ln r +- 1.96 sqrt(k) sigma_run."""
    mult = {"parakeet-ctc": 0.85, "cohere": 0.9, "study-p03": 1.0, "study-t06": 1.1}
    tables, manifest = planted(mult)
    readouts = {"full-p03": 0.95, "full-p03@int8-w8a8": 0.95 * 1.02, "whisper-large-v3": 0.8}
    rtab, _ = planted(readouts)
    man = tmp_path / "m.json"
    man.write_text(json.dumps(manifest), encoding="utf-8")
    write_tables(tmp_path / "tables", tables)
    pull = tmp_path / "runs"
    write_readout(pull, "m4-full-p03", "full-p03", rtab["full-p03"], family="ctc")
    write_readout(pull, "quant-int8", "full-p03@int8-w8a8", rtab["full-p03@int8-w8a8"], family="ctc",
                  quant=dict(format="int8-w8a8", base_system="full-p03", impl="torchao", file_bytes=1))
    write_readout(pull, "whisper", "whisper-large-v3", rtab["whisper-large-v3"], family="whisper")
    rc, rep = run(["--manifest", man, "--prereg", "none", "--tables", tmp_path / "tables", "--readouts", pull,
                   "--boot-b", 200], tmp_path / "out")
    assert rc == 0
    s, z = 0.016, 1.96
    comp = rep["comparisons"]

    def check(c, ratio, k):
        assert c["ratio"] == pytest.approx(ratio, rel=1e-12) and c["n_noisy"] == k
        assert c["v_boot"] == pytest.approx(0.0, abs=1e-20)
        assert c["se"] == pytest.approx(math.sqrt(k) * s)
        lo, hi = c["ci_ratio"]
        assert lo == pytest.approx(ratio * math.exp(-z * math.sqrt(k) * s))
        assert hi == pytest.approx(ratio * math.exp(z * math.sqrt(k) * s))

    for m in ("m4", "m4_all"):
        check(comp["vs_teacher"]["full-p03"][m], 0.95 / 0.85, 1)
        check(comp["vs_teacher"]["study-t06"][m], 1.1 / 0.9, 1)
        check(comp["vs_study"]["full-p03"][m], 0.95, 2)
        check(comp["vs_bf16"]["full-p03@int8-w8a8"][m], 1.02, 0)
    assert comp["vs_study"]["full-p03"]["m4"]["verdict"] == "better"
    assert comp["vs_bf16"]["full-p03@int8-w8a8"]["m4"]["verdict"] == "worse"
    assert "whisper-large-v3" not in comp["vs_teacher"] and "cohere" not in comp["vs_teacher"]
    assert rep["settings"]["sigma_run"]["sigma_run"] == s


def test_speed_mapping_and_flags(world):
    """Own record in the chart group; a full-data row falls back to its study weights (S1); a token drift over 5 %
    re-times T-0.6B from box full's pair (S2); another machine (S3); MXFP4 simulated (S4); none (S5). Three speed-*
    files are read, two of them one machine."""
    rep = world["rep"]
    sg = rep["speed_groups"]
    assert sg["primary"] == PRIMARY and set(sg["groups"]) == {PRIMARY, "m99999 / NVIDIA GeForce RTX 5090"}
    assert len(sg["groups"][PRIMARY]["files"]) == 2 and sg["groups"][PRIMARY]["compiled"] == [
        "study-p03@int8-w8a8+compile"]
    sp = {s: v["speed"] for s, v in rep["systems"].items()}
    assert sp["study-p03"]["flags"] == [] and sp["study-p03"]["rtfx"] == pytest.approx(2500)
    assert sp["full-p03"]["flags"] == ["S1"] and sp["full-p03"]["source"] == "study-p03"
    assert sp["full-p03"]["rtfx"] == pytest.approx(2500) and sp["full-p03"]["p50_ms"] == pytest.approx(16)
    assert sp["full-p03@nvfp4-w4a4"]["source"] == "study-p03@nvfp4-w4a4" and sp["full-p03@nvfp4-w4a4"]["flags"] == [
        "S1"]
    assert sp["full-p03@mxfp4-w4a4"]["flags"] == ["S4"] and sp["full-p03@mxfp4-w4a4"]["text"] == "n/a (simulated)"
    assert sp["whisper-small"]["flags"] == ["S3"] and not sp["whisper-small"]["in_chart_group"]
    assert sp["cohere"]["flags"] == ["S5"]
    # T-0.6B: the readout emits 12 tokens a row, the timed study weights 10: S2, and the chart group's study-t06
    # record times the full/study ratio box full timed on its own machine
    t = sp["full-t06"]
    assert t["flags"] == ["S1", "S2"] and t["drift"]["drift"] == pytest.approx(0.2)
    assert t["drift"]["n_ids"] == len(world["sp_ids"])
    assert t["rtf"] == pytest.approx(0.001 * 0.003 / 0.002) and t["rtfx"] == pytest.approx(1 / 0.0015)
    assert t["p50_ms"] == pytest.approx(1000 * 0.04 * 0.09 / 0.06)
    assert t["retime"]["group"] == "m99999 / NVIDIA GeForce RTX 5090" and t["in_chart_group"]
    flags = {s: v["flags"] for s, v in rep["systems"].items()}
    assert flags["full-p03@int8-w8a8"] == ["R1", "S1", "Q1", "E1"]  # fp32 fallbacks > 0; its base stopped early
    assert "Q1" not in flags["full-p03@nvfp4-w4a4"] and "E1" in flags["full-p03"] and "E1" not in flags["full-t06"]
    assert flags["kotoba-whisper-v2.0"][:6] == ["W1", "W2", "W3", "W4", "W5", "W6"]
    assert flags["whisper-small"][:5] == ["W1", "W3", "W4", "W5", "W6"] and "R1" in flags["parakeet-ctc"]
    chk = next(c for c in rep["checks"] if c["rule"] == "speed_groups")
    assert chk["status"] == "fail" and "['whisper-small']" in chk["detail"]  # the re-time group alone would pass
    md = (world["out"] / "report.md").read_text(encoding="utf-8")
    assert "n/a (simulated)" in md
    for f in FR.FLAGS:
        assert f"- **{f}**:" in md


def test_speed_id_list_mismatch_refused(world, tmp_path, capsys):
    other = write_speed(tmp_path / "speed-x" / "speed.json", world["sp_ids"][:-1], {"cohere": srec(0.003)})
    args = [a for a in world["args"]]
    i = args.index("--speed")
    args.insert(i + 1, other)
    rc, _ = run(args, tmp_path / "out")
    assert rc == 2 and "another id list" in capsys.readouterr().err
    assert not (tmp_path / "out" / "report.json").exists()


def test_manifest_mismatch_exits_2(world, tmp_path, capsys):
    """A table that does not hold exactly the manifest's ids refuses (exit 2), and so does a manifest the repo's
    PREREG.json did not freeze; with --prereg none the report is written but does not stack on the study."""
    ids = world["ids"]
    bad = {"study-x": world["T"]["cohere"][world["T"]["cohere"]["id"] != ids["eval_cv8"][0]]}
    write_tables(tmp_path / "bad", bad)
    rc, _ = run(["--manifest", world["man"], "--prereg", "none", "--tables", tmp_path / "bad"], tmp_path / "o1")
    assert rc == 2 and "ids differ from the manifest" in capsys.readouterr().err
    rc, _ = run(["--manifest", world["man"], "--tables", world["tables"]], tmp_path / "o2")  # the repo's PREREG
    assert rc == 2 and "not the one PREREG.json froze" in capsys.readouterr().err
    m = world["rep"]["manifest"]
    assert m["stacks_on_study"] is False and m["prereg_check"]["status"] == "not_checked"
    assert next(c for c in world["rep"]["checks"] if c["rule"] == "stacks_on_study")["status"] == "fail"
    assert next(c for c in world["rep"]["checks"] if c["rule"] == "readout_manifests")["status"] == "fail"


def test_chart_svg_points(world, tmp_path):
    """One marker per drawable point (the chart group only; MXFP4, S3 and speedless rows listed under the plot, the
    study row with a full-data twin left to it); y = M4-all; deterministic bytes."""
    rep, out = world["rep"], world["out"]
    pts = {p["system"]: p for p in rep["chart"]["points"]}
    drawn = {s for s, p in pts.items() if p["drawn"]}
    assert drawn == {"parakeet-ctc", "parakeet-tdt", "full-p03", "full-p03@int8-w8a8", "full-p03@nvfp4-w4a4",
                     "full-t06", "whisper-large-v3", "kotoba-whisper-v2.0", "study-p01"}
    assert pts["study-p03"]["not_drawn"] == "shown as its full-data row full-p03"
    assert "S4" in pts["full-p03@mxfp4-w4a4"]["not_drawn"] and "S3" in pts["whisper-small"]["not_drawn"]
    assert "S5" in pts["cohere"]["not_drawn"]
    for s in drawn:
        assert pts[s]["m4_all_pct"] == pytest.approx(100 * rep["systems"][s]["metrics"]["m4_all"])
    for name in ("chart_error_vs_speed.svg", "chart_error_vs_latency.svg"):
        svg = (out / name).read_text(encoding="utf-8")
        titles = re.findall(r"<g><title>([^<]*)</title>", svg)
        assert len(titles) == len(drawn)
        assert not any("MXFP4" in t for t in titles) and "P-0.3B full MXFP4 W4A4: simulated" in svg
        assert svg.count('class="h s-parakeet"') == 2  # the two timed variants, hollow
        assert svg.count("<polygon class=\"m f-") == 3  # two Parakeet stars, study-p01's diamond
        assert "@media (prefers-color-scheme: dark)" in svg and svg.startswith("<svg ")
        # no direct label runs into the legend (x >= 750 from y 62 to 244; chart_svg's geometry, 0.58 em a glyph);
        # the fastest student sits top right, next to it (here P-0.1B study)
        labels = re.findall(r'<text class="(lab2?)" x="([^"]*)" y="([^"]*)">([^<]*)</text>', svg)
        placed = [(float(x), float(y), 9.5 if c == "lab2" else 11.0, t) for c, x, y, t in labels if float(x) < 750]
        assert any(t == "P-0.1B study" for *_, t in placed)
        for x, y, size, t in placed:
            assert x + 0.58 * size * len(t) <= 750 or y - size >= 244 or y + 2 <= 62, (name, t, x, y)
    rc, _ = run(world["args"], tmp_path / "again")
    assert rc == 0
    for name in ("chart_error_vs_speed.svg", "chart_error_vs_latency.svg", "chart_points.csv"):
        assert (tmp_path / "again" / name).read_bytes() == (out / name).read_bytes()
    pj = json.loads((out / "chart_points.json").read_text(encoding="utf-8"))
    assert pj["group"] == PRIMARY and pj["gpu"] == "NVIDIA GeForce RTX 5090" and len(pj["points"]) == len(pts)


def test_report_md_order_and_whisper_caveats(world, tmp_path):
    md = (world["out"] / "report.md").read_text(encoding="utf-8")
    heads = re.findall(r"^## (.+)$", md, re.M)
    assert heads[:6] == ["Offer table", "Study -> full", "Quantisation", "Whisper", "Charts", "Hallucinations"]
    assert heads[-3:] == ["Flags", "Checks", "Inputs"]
    assert md.index("### Parakeet") < md.index("### Cohere") < md.index("### Whisper") < md.index("### The study")
    for c in FR.WHISPER_CAVEATS:
        assert c in md
    assert len(FR.WHISPER_CAVEATS) == 7 and "not given (--kotoba-jsonl" in md
    assert "![Error vs speed](chart_error_vs_speed.svg)" in md
    assert "yes (patience, step 900)" in md  # the study -> full early-stop cell
    # no Whisper system: no Whisper section, no caveats
    t = world["T"]
    write_tables(tmp_path / "tables", {"cohere": t["cohere"], "study-p03": t["study-p03"]})
    rc, _ = run(["--manifest", world["man"], "--prereg", "none", "--tables", tmp_path / "tables"], tmp_path / "out")
    md2 = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert rc == 0 and "## Whisper" not in md2 and FR.WHISPER_CAVEATS[0] not in md2
    assert re.findall(r"^## (.+)$", md2, re.M)[0] == "Offer table"


def test_kl_study_to_full_and_quant_sections(world):
    rep = world["rep"]
    kl = rep["systems"]["full-p03"]["kl"]
    assert kl["gate"] == pytest.approx((0.1 * 100 + 0.2 * 300 + 0.3 * 100) / 500) and kl["complete"]
    assert rep["comparisons"]["vs_bf16"]["full-p03@int8-w8a8"]["kl_delta"] == pytest.approx(0.05)
    assert rep["comparisons"]["vs_bf16"]["full-p03@nvfp4-w4a4"]["kl_delta"] is None
    assert rep["systems"]["whisper-large-v3"]["kl"] is None
    run_ = rep["systems"]["full-t06"]["run"]
    assert run_["resume_resets"] == 1 and run_["epochs"] == 3.0
    r = next(r for r in rep["offer"]["rows"] if r["system"] == "full-p03")
    assert r["epochs"] == 2.5 and r["stopped_early"]
    assert r["vs_study"]["ratio"] == pytest.approx(rep["systems"]["full-p03"]["metrics"]["m4"]
                                                   / rep["systems"]["study-p03"]["metrics"]["m4"])
    md = (world["out"] / "report.md").read_text(encoding="utf-8")
    assert "torch.compile records" in md and "study-p03@int8-w8a8+compile" in md


def wp5_bytes(deployable: int) -> dict:
    """study.json quant.weights_bytes as WP5 writes it (the recipe's bytes; kitsune.quant.weight_bytes minus
    resident)."""
    return dict(deployable=deployable, quantized=deployable // 4, kept=deployable - deployable // 4)


def test_duplicate_readouts_and_tables_precedence(world, tmp_path):
    """One system found twice: complete beats partial, real beats simulated (each on its own), a readout of the
    exported file beats a newer one scored from memory (smoke B's quant-* then mem-* of study-p03@<fmt>), then the
    newest; a --tables system beats a readout of its name. A variant scored only from memory takes WP5's
    quant.weights_bytes.deployable, never its bf16 checkpoint's weights.file_bytes."""
    t, ids, refs = world["T"], world["ids"], world["refs"]
    pull = tmp_path / "runs"
    q = dict(format="int8-w8a8", base_system="study-p03", impl="torchao", simulated=False,
             weights_bytes=wp5_bytes(318_000_000))
    older = text_table(ids, refs, noisy(refs, 0.1, 40))
    newer = text_table(ids, refs, noisy(refs, 0.11, 41))
    write_readout(pull, "quant-a", "study-p03@int8-w8a8", older, time_utc="2026-10-01T00:00:00+00:00",
                  quant=dict(q, source="file", file_bytes=320_000_000))
    # scored from memory (05 --quant on the bf16 checkpoint), later: no file of its own, weights are the bf16 export's
    mem = dict(q, source="memory", file_bytes=None, weights_bytes=wp5_bytes(312_000_000))
    write_readout(pull, "mem-b", "study-p03@int8-w8a8", newer, time_utc="2026-10-02T00:00:00+00:00", quant=mem,
                  weights=dict(path="ckpt", file_bytes=617_000_000))
    partial = newer[newer["set"].isin(["eval_jsut", "eval_cv8"])]
    write_readout(pull, "emu", "study-p03@int8-w8a8", partial, time_utc="2026-10-09T00:00:00+00:00",
                  quant=dict(q, source="file", impl="emulate", simulated=True))  # partial AND simulated
    write_readout(pull, "emu-all", "study-p03@int8-w8a8", newer, time_utc="2026-10-09T00:00:00+00:00",
                  quant=dict(q, source="file", impl="emulate", simulated=True))  # complete: simulated alone loses
    write_readout(pull, "mem-fp16", "study-p03@fp16", newer, time_utc="2026-10-02T00:00:00+00:00",
                  quant=dict(mem, format="fp16"), weights=dict(path="ckpt", file_bytes=617_000_000))
    write_readout(pull, "shadow", "study-p03", t["cohere"], time_utc="2026-10-09T00:00:00+00:00")
    write_readout(pull, "empty", "orphan")  # no tables and no greedy parquets: ignored, listed
    write_tables(tmp_path / "tables", {"study-p03": t["study-p03"], "parakeet-ctc": t["parakeet-ctc"]})
    rc, rep = run(["--manifest", world["man"], "--prereg", "none", "--tables", tmp_path / "tables",
                   "--readouts", pull, "--boot-b", 200], tmp_path / "out")
    assert rc == 0
    inp = rep["inputs"]
    assert Path(inp["readouts_read"]["study-p03@int8-w8a8"]).name == "quant-a"
    sup = {Path(s["dir"]).name: s for s in inp["readouts_superseded"]}
    assert set(sup) == {"mem-b", "emu", "emu-all", "shadow"} and "study-p03" not in inp["readouts_read"]
    assert sup["emu-all"]["complete"] and sup["emu-all"]["simulated"] and not sup["mem-b"]["file_scored"]
    assert [Path(i["path"]).parent.name for i in inp["readouts_ignored"]] == ["empty"]
    assert rep["systems"]["study-p03"]["metrics"]["m4"] == pytest.approx(
        np.mean([hand_cer(t["study-p03"], s) for s in ("eval_jsut", "eval_cv8", "eval_reazon")]
                + [hand_cer(t["study-p03"], "galgame", world["manifest"]["galgame_views"]["neutral"])]))
    v = rep["systems"]["study-p03@int8-w8a8"]
    assert v["display"] == "P-0.3B study INT8 W8A8" and v["metrics"]["m4"] == pytest.approx(
        np.mean([hand_cer(older, s) for s in ("eval_jsut", "eval_cv8", "eval_reazon")]
                + [hand_cer(older, "galgame", world["manifest"]["galgame_views"]["neutral"])]))
    assert v["file_bytes"] == 320_000_000 and v["file_bytes_from"] == "quant.file_bytes"
    f = rep["systems"]["study-p03@fp16"]
    assert f["file_bytes"] == 312_000_000 and f["file_bytes_from"] == "in-memory (quant.weights_bytes)"
    assert next(r for r in rep["offer"]["rows"] if r["system"] == "study-p03@fp16")["file_bytes"] == 312_000_000


SMOKE_A = "speed-full-smoke-20261001T000000Z"
SMOKE_B = "speed-smoke-b-20261002T000000Z"


def test_speed_groups_from_queue_summaries(world, tmp_path, capsys):
    """Smoke A times the bf16 rows with the wave-1 speed_probe (versions.host only: the container's name) and smoke B
    the variants and Whisper with machine_id, on ONE machine (smoke B's bf16 re-time is then not_needed). The box's
    queue summary gives smoke A's machine: one group, every bf16 row and teacher drawn. Without a summary that
    covers smoke A (the box launched again later, elsewhere) its host is its own group (S3, the check says why);
    --machine-of, or an explicit --queue-summaries naming the dir, resolves it."""
    t, ids, refs = world["T"], world["ids"], world["refs"]
    root = tmp_path / "pull"
    runs = root / "runs"
    write_tables(tmp_path / "tables", {s: t[s] for s in ("cohere", "parakeet-ctc", "parakeet-tdt", "study-p03",
                                                         "study-t06")})
    write_readout(runs, f"m4-full-p03-{STAMP}", "full-p03", t["full-p03"], family="ctc")
    sp_ids = world["sp_ids"]
    a = dict(machine=None, host="c-7f3a9e1b2", t="2026-10-01T06:00:00+00:00")
    b = dict(host="c-0b5e", t="2026-10-02T03:00:00+00:00")
    speed = [
        write_speed(runs / SMOKE_A / "speed.json", sp_ids, {
            "study-p03": srec(0.0004, **a), "study-t06": srec(0.001, kind="aed", tokens=10.0, **a),
            "parakeet-ctc": srec(0.0005, **a), "parakeet-tdt": srec(0.0006, **a),
            "cohere": srec(0.002, kind="cohere", **a)}),
        write_speed(runs / SMOKE_B / "speed.json", sp_ids, {
            "study-p03@int8-w8a8": srec(0.00045, **b), "study-p03@nvfp4-w4a4": srec(0.0005, **b),
            "study-t06@int8-w8a8": srec(0.0009, **b), "study-t06@nvfp4-w4a4": srec(0.0009, **b),
            "whisper-large-v3": srec(0.004, kind="whisper", **b), "kotoba-whisper-v2.0": srec(0.002, **b)})]
    hub = queue_summary(root / "full" / "box-full-smoke" / "queue_summary.json", "full-smoke", "m54650",
                        "2026-10-01T00:00:00+00:00", f"runs/{SMOKE_A}")
    base = ["--manifest", world["man"], "--prereg", "none", "--tables", tmp_path / "tables", "--readouts", runs,
            "--speed", *speed, "--boot-b", 200]

    rc, rep = run(base, tmp_path / "o1")  # the summary at the runs repo's layout is read without the flag
    assert rc == 0
    sg = rep["speed_groups"]
    assert list(sg["groups"]) == [PRIMARY] and sg["primary"] == PRIMARY
    assert sg["groups"][PRIMARY]["machine_from"] == sorted([f"queue summary {hub}", "versions.machine_id"])
    assert rep["inputs"]["queue_summaries"] == [str(hub)]
    sysd = rep["systems"]
    assert sysd["study-p03"]["speed"]["flags"] == [] and sysd["full-p03"]["speed"]["flags"] == ["S1"]
    assert sysd["cohere"]["speed"]["machine_from"] == f"queue summary {hub}"
    drawn = {p["system"] for p in rep["chart"]["points"] if p["drawn"]}
    assert drawn == {"cohere", "parakeet-ctc", "parakeet-tdt", "full-p03", "study-t06"}  # study-p03: as full-p03
    assert next(c for c in rep["checks"] if c["rule"] == "speed_groups")["status"] == "pass"

    # box full-smoke launched again on another machine later: its summary now says m77777, started after smoke A's
    # records were timed, so it does not cover them
    queue_summary(hub, "full-smoke", "m77777", "2026-10-01T12:00:00+00:00", "runs/speed-full-smoke-20261001T120000Z")
    rc, rep = run(base, tmp_path / "o2")
    host = "c-7f3a9e1b2 / NVIDIA GeForce RTX 5090"
    assert rc == 0 and set(rep["speed_groups"]["groups"]) == {PRIMARY, host}
    assert rep["speed_groups"]["groups"][host]["machine_from"] == ["versions.host"]
    assert rep["systems"]["study-p03"]["speed"]["flags"] == ["S3"]
    chk = next(c for c in rep["checks"] if c["rule"] == "speed_groups")
    assert chk["status"] == "fail" and "keyed by a container host" in chk["detail"]

    for extra, where in ((["--machine-of", "c-7f3a9e1b2=m54650"], "--machine-of c-7f3a9e1b2"),
                         (["--machine-of", f"{SMOKE_A}=m54650"], f"--machine-of {SMOKE_A}")):
        rc, rep = run(base + extra, tmp_path / "o3")
        assert rc == 0 and list(rep["speed_groups"]["groups"]) == [PRIMARY]
        assert rep["systems"]["study-p03"]["speed"]["machine_from"] == where
    # the first launch's summary kept elsewhere (the infra dir, a copy): it names smoke A's dir and started before it
    first = queue_summary(tmp_path / "kept" / "box-full-smoke" / "queue_summary.json", "full-smoke", "m54650",
                          "2026-10-01T00:00:00+00:00", f"runs/{SMOKE_A}")
    rc, rep = run(base + ["--queue-summaries", tmp_path / "kept"], tmp_path / "o4")
    assert rc == 0 and list(rep["speed_groups"]["groups"]) == [PRIMARY]
    assert rep["systems"]["study-p03"]["speed"]["machine_from"] == f"queue summary {first}"
    rc, _ = run(base + ["--machine-of", "c-7f3a9e1b2"], tmp_path / "o5")
    assert rc == 2 and "expected NAME=MACHINE_ID" in capsys.readouterr().err


def test_s2_retime_keeps_variants_on_the_bf16_basis_and_chart_twins(world, tmp_path):
    """S2: the bf16 full-t06 row AND its variant take box full's bf16 re-time ratio (the variant decodes the same
    tokens), so "x bf16 speed" and the chart's join compare one basis; each keeps its record's own numbers. A study
    row hides behind its full-data twin only when that twin is drawn: an emulated full variant leaves its timed study
    variant on the chart."""
    t, ids, refs = world["T"], world["ids"], world["refs"]
    runs = tmp_path / "runs"
    write_tables(tmp_path / "tables", {s: t[s] for s in ("cohere", "parakeet-ctc", "study-p03", "study-t06")})
    h = noisy(refs, 0.095, 26)
    write_readout(runs, "m4-full-t06", "full-t06", greedy=greedy_frames(ids, refs, h, n_tok=15), family="aed")
    write_readout(runs, "quant-int8-w8a8-full-t06", "full-t06@int8-w8a8", text_table(ids, refs, h), family="aed",
                  quant=dict(format="int8-w8a8", impl="torchao", simulated=False, source="file",
                             base_system="full-t06", file_bytes=700_000_000))
    write_readout(runs, "m4-full-p03", "full-p03", t["full-p03"], family="ctc")
    write_readout(runs, "quant-nvfp4-w4a4-full-p03", "full-p03@nvfp4-w4a4", text_table(ids, refs, noisy(refs, .09, 6)),
                  family="ctc", quant=dict(format="nvfp4-w4a4", impl="emulate", simulated=True, source="file",
                                           base_system="full-p03", file_bytes=180_000_000))
    write_readout(runs, "quant-nvfp4-w4a4-study-p03", "study-p03@nvfp4-w4a4",
                  text_table(ids, refs, noisy(refs, .095, 7)), family="ctc",
                  quant=dict(format="nvfp4-w4a4", impl="torchao", simulated=False, source="file",
                             base_system="study-p03", file_bytes=180_000_000))
    sp_ids = world["sp_ids"]
    speed = [
        write_speed(runs / SMOKE_B / "speed.json", sp_ids, {
            "study-t06": srec(0.001, tokens=10.0, kind="aed"), "study-t06@int8-w8a8": srec(0.0008, kind="aed"),
            "study-p03": srec(0.0004), "study-p03@nvfp4-w4a4": srec(0.0003), "parakeet-ctc": srec(0.0005),
            "cohere": srec(0.002, kind="cohere")}),
        write_speed(runs / "speed-full-20261005T000000Z" / "speed.json", sp_ids, {
            "full-t06": srec(0.003, machine="m99999", p50_s=0.09, kind="aed"),
            "study-t06": srec(0.002, machine="m99999", p50_s=0.06, kind="aed")})]
    rc, rep = run(["--manifest", world["man"], "--prereg", "none", "--tables", tmp_path / "tables", "--readouts",
                   runs, "--speed", *speed, "--boot-b", 200], tmp_path / "out")
    assert rc == 0
    sp = {s: v["speed"] for s, v in rep["systems"].items()}
    for s, rtf, p50 in (("full-t06", 0.001, 0.04), ("full-t06@int8-w8a8", 0.0008, 0.032)):
        assert sp[s]["flags"] == ["S1", "S2"] and sp[s]["drift"]["drift"] == pytest.approx(0.5)
        assert sp[s]["rtf"] == pytest.approx(rtf * 1.5) and sp[s]["p50_ms"] == pytest.approx(1000 * p50 * 1.5)
        assert sp[s]["unscaled"]["rtf"] == pytest.approx(rtf) and sp[s]["retime"]["full"] == "full-t06"
    assert "applied to the variant" in sp["full-t06@int8-w8a8"]["retime"]["note"]
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    quant = md[md.index("## Quantisation"):md.index("## Charts")].splitlines()
    head = next(ln for ln in quant if ln.startswith("| variant |"))
    row = next(ln for ln in quant if ln.startswith("| T-0.6B full INT8 W8A8 |"))
    col = [c.strip() for c in head.split("|")].index("x bf16 speed")
    assert row.split("|")[col].strip() == "1.25"  # 0.001 / 0.0008: the study weights' own ratio
    svg = (tmp_path / "out" / "chart_error_vs_speed.svg").read_text(encoding="utf-8")
    assert svg.count('class="j s-cohere"') == 1  # the variant joined to its re-timed bf16 point
    pts = {p["system"]: p for p in rep["chart"]["points"]}
    assert pts["study-p03@nvfp4-w4a4"]["drawn"] and "S4" in pts["full-p03@nvfp4-w4a4"]["not_drawn"]
    assert pts["study-p03"]["not_drawn"] == "shown as its full-data row full-p03" and pts["full-p03"]["drawn"]
    # the study variant has no bf16 point to join (study-p03 is shown as full-p03): it carries its whole name, never
    # just "NVFP4 W4A4" next to P-0.3B full; the joined full-t06 variant keeps its format label
    labels = re.findall(r'<text class="(lab2?)" x="[^"]*" y="[^"]*">([^<]*)</text>', svg)
    assert ("lab", "P-0.3B study NVFP4 W4A4") in labels and ("lab2", "NVFP4 W4A4") not in labels
    assert ("lab2", "INT8 W8A8") in labels
    nd = next(ln for ln in md.splitlines() if ln.startswith("Not drawn:"))
    assert "P-0.3B full NVFP4 W4A4" in nd and "P-0.3B study NVFP4 W4A4" not in nd


def test_s2_retime_pair_on_the_chart_machine(world, tmp_path):
    """Box full rented the chart group's machine: full-t06 has its own record there (no S1, no S2) and box full's
    speed-study-t06 re-time, newer and in the same file, stands for study-t06. A full-t06 variant still falls back to
    its study variant (S1) with S2, so it takes that same-file pair's ratio from the chart group: its "x bf16 speed"
    and its join stay on one basis. A pair from two files (two launches) is not a re-time pair."""
    t, ids, refs = world["T"], world["ids"], world["refs"]
    runs = tmp_path / "runs"
    write_tables(tmp_path / "tables", {s: t[s] for s in ("cohere", "study-t06")})
    h = noisy(refs, 0.095, 26)
    write_readout(runs, "m4-full-t06", "full-t06", greedy=greedy_frames(ids, refs, h, n_tok=15), family="aed")
    write_readout(runs, "quant-int8-w8a8-full-t06", "full-t06@int8-w8a8", text_table(ids, refs, h), family="aed",
                  quant=dict(format="int8-w8a8", impl="torchao", simulated=False, source="file",
                             base_system="full-t06", file_bytes=700_000_000))
    sp_ids = world["sp_ids"]
    a = write_speed(runs / SMOKE_A / "speed.json", sp_ids, {
        "study-t06": srec(0.002, tokens=10.0, kind="aed", p50_s=0.06, t="2026-10-01T06:00:00+00:00"),
        "cohere": srec(0.002, kind="cohere", t="2026-10-01T06:10:00+00:00")})
    b = write_speed(runs / SMOKE_B / "speed.json", sp_ids, {
        "study-t06@int8-w8a8": srec(0.0008, kind="aed", p50_s=0.032, t="2026-10-02T03:00:00+00:00")})
    full = write_speed(runs / "speed-full-20261005T000000Z" / "speed.json", sp_ids, {
        "full-t06": srec(0.003, p50_s=0.09, kind="aed", t="2026-10-05T01:00:00+00:00"),
        "study-t06": srec(0.002, p50_s=0.06, tokens=10.0, kind="aed", t="2026-10-05T01:10:00+00:00")})
    base = ["--manifest", world["man"], "--prereg", "none", "--tables", tmp_path / "tables", "--readouts", runs,
            "--boot-b", 200]
    rc, rep = run(base + ["--speed", a, b, full], tmp_path / "out")
    assert rc == 0 and list(rep["speed_groups"]["groups"]) == [PRIMARY]
    sp = {s: v["speed"] for s, v in rep["systems"].items()}
    assert sp["full-t06"]["flags"] == [] and sp["full-t06"]["rtf"] == pytest.approx(0.003)
    assert sp["study-t06"]["file"] == str(full)  # box full's re-time replaces smoke A's record (newer, listed)
    v = sp["full-t06@int8-w8a8"]
    assert v["flags"] == ["S1", "S2"] and v["retime"]["group"] == PRIMARY
    assert v["rtf"] == pytest.approx(0.0008 * 1.5) and v["unscaled"]["rtf"] == pytest.approx(0.0008)
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    quant = md[md.index("## Quantisation"):md.index("## Charts")].splitlines()
    head = next(ln for ln in quant if ln.startswith("| variant |"))
    row = next(ln for ln in quant if ln.startswith("| T-0.6B full INT8 W8A8 |"))
    col = [c.strip() for c in head.split("|")].index("x bf16 speed")
    assert row.split("|")[col].strip() == "2.50"  # 0.002 / 0.0008: the study weights' own ratio
    # full-t06 timed by one launch and study-t06 only by another: no re-time pair, the study speed is shown as it is
    lone = write_speed(runs / "speed-full-20261006T000000Z" / "speed.json", sp_ids, {
        "full-t06": srec(0.003, p50_s=0.09, kind="aed", t="2026-10-06T01:00:00+00:00")})
    rc, rep = run(base + ["--speed", a, b, lone], tmp_path / "out2")
    v = rep["systems"]["full-t06@int8-w8a8"]["speed"]
    assert rc == 0 and v["flags"] == ["S1", "S2"] and v["retime"]["group"] is None
    assert v["rtf"] == pytest.approx(0.0008) and "unscaled" not in v


def test_greedy_set_refused_like_05(world, tmp_path):
    """A greedy-only readout with one set whose rows are not its manifest ids: that set is refused for that system
    (05's tables_from_frames rule), listed, and the report is written with the other sets. A null n_tok or duration
    is unknown for that row, never a crash."""
    t, ids, refs = world["T"], world["ids"], world["refs"]
    runs = tmp_path / "runs"
    h = noisy(refs, 0.08, 40)
    fr = greedy_frames(ids, refs, h)
    fr["eval_cv8"] = fr["eval_cv8"].iloc[1:]
    j = fr["eval_jsut"]
    fr["eval_jsut"] = j.assign(n_tok=[None, *j["n_tok"].tolist()[1:]], duration=[None, *j["duration"].tolist()[1:]])
    write_readout(runs, "m4-full-p03", "full-p03", greedy=fr, family="ctc")
    write_tables(tmp_path / "tables", {"parakeet-ctc": t["parakeet-ctc"]})
    rc, rep = run(["--manifest", world["man"], "--prereg", "none", "--tables", tmp_path / "tables", "--readouts",
                   runs, "--boot-b", 200], tmp_path / "out")
    assert rc == 0
    v = rep["systems"]["full-p03"]
    assert "eval_cv8" not in v["sets"] and v["metrics"]["m4"] is None and v["metrics"]["m4_all"] is None
    assert v["sets"]["eval_jsut"]["cer"] == pytest.approx(hand_cer(text_table(ids, refs, h), "eval_jsut"))
    ref = rep["inputs"]["readouts_refused_sets"]
    assert [(r["system"], r["set"]) for r in ref] == [("full-p03", "eval_cv8")] and "1 manifest ids missing" in \
        ref[0]["why"]
    assert "greedy sets refused" in (tmp_path / "out" / "report.md").read_text(encoding="utf-8")


def test_unreadable_inputs_refuse_with_exit_2(world, tmp_path, capsys):
    """CONTRACT.md 1.5: an unreadable or malformed input refuses (exit 2, the file named), never a traceback (1)."""
    write_tables(tmp_path / "tables", {"cohere": world["T"]["cohere"]})
    base = ["--manifest", world["man"], "--prereg", "none", "--tables", tmp_path / "tables", "--boot-b", 200]
    bad = tmp_path / "bad.json"
    bad.write_text('{"study-t06": ', encoding="utf-8")
    shape = tmp_path / "list.json"
    shape.write_text("[1, 2]", encoding="utf-8")
    runs = tmp_path / "runs"
    (runs / f"full-p03-{STAMP}").mkdir(parents=True)
    (runs / f"full-p03-{STAMP}" / "summary.json").write_text('{"status": ', encoding="utf-8")
    ro = tmp_path / "ro"
    (ro / "m4-x" / "tables" / "full-p03").mkdir(parents=True)
    (ro / "m4-x" / "study.json").write_text(json.dumps(dict(system="full-p03")), encoding="utf-8")
    (ro / "m4-x" / "tables" / "full-p03" / "eval_jsut.parquet").write_bytes(b"not a parquet file")
    for extra, needle in ((["--params", bad], "bad.json"), (["--params", shape], "--params"),
                          (["--run-summaries", runs], "summary.json"),
                          (["--kotoba-jsonl", tmp_path / "missing.jsonl"], "--kotoba-jsonl"),
                          (["--readouts", ro], "eval_jsut.parquet"), (["--queue-summaries", shape], "list.json"),
                          (["--queue-summaries", tmp_path / "nowhere"], "--queue-summaries")):
        rc, _ = run(base + extra, tmp_path / "out")
        err = capsys.readouterr().err
        assert rc == 2 and "REFUSED:" in err and needle in err, (extra, err)
        assert "Traceback" not in err


def test_kotoba_bias_from_the_judge_file(world, tmp_path, monkeypatch, capsys):
    """Decision 29: the stored judge file scored on the neutral view vs every Galgame row with a reference; any
    other file (sha256) refuses."""
    ids, refs = world["ids"], world["refs"]
    hyp2 = noisy(refs, 0.3, 50)["galgame"]
    lines = [json.dumps(dict(id=i, hyp2=h, model2="kotoba-whisper-v2.0", agree=0.1, cer2=0.1), ensure_ascii=False)
             for i, h in zip(ids["galgame"], hyp2)]
    f = tmp_path / "eval-00000.jsonl"
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args = [a for a in world["args"]] + ["--kotoba-jsonl", f]
    rc, _ = run(args, tmp_path / "o1")
    assert rc == 2 and "not the judge file's" in capsys.readouterr().err
    monkeypatch.setattr(FR, "KOTOBA_JSONL_SHA256", hashlib.sha256(f.read_bytes()).hexdigest())
    man = ss.parse_manifest(world["manifest"])
    kb = FR.kotoba_bias(f, {"cohere": world["T"]["cohere"]}, man)
    sc = ss.score_utterances(refs["galgame"], hyp2).assign(id=ids["galgame"])
    sc = sc[sc["ref_len"] > 0]
    neutral = set(world["manifest"]["galgame_views"]["neutral"])
    n = sc[sc["id"].isin(neutral)]
    assert kb["cer_neutral"] == pytest.approx(n["edits"].sum() / n["ref_len"].sum())
    assert kb["cer_all"] == pytest.approx(sc["edits"].sum() / sc["ref_len"].sum())
    assert kb["n_neutral"] == N_NEUTRAL and kb["n_all"] == SETS["galgame"] - 1 and kb["model2"] == {
        "kotoba-whisper-v2.0": SETS["galgame"]}
    ko = world["rep"]["whisper"]["kotoba_redecode_over_half"]
    assert ko["of"] == N_NEUTRAL and 0 <= ko["n"] <= N_NEUTRAL


def test_include_half_and_argument_rules(world, tmp_path):
    rc, rep = run(["--manifest", world["man"], "--prereg", "none", "--tables", world["tables"], "--include-half",
                   "--boot-b", 200], tmp_path / "out")
    assert rc == 0 and rep["systems"]["study-p03-half"]["role"] == "half" and rep["left_out"]["half"] == []
    assert all(r["system"] != "study-p03-half" for r in rep["offer"]["rows"])
    with pytest.raises(SystemExit):
        FR.parse_args(["--manifest", "m.json", "--out", "o"])  # neither --tables nor --readouts


def test_vocabulary_matches_the_quant_and_whisper_modules():
    """The report's local copies of WP5's formats and WP6's model keys, compared once those modules are on the
    branch (the report never needs them at run time)."""
    try:
        from kitsune import quant
    except ImportError:
        quant = None
    try:
        from kitsune import whisper
    except ImportError:
        whisper = None
    if quant is None and whisper is None:
        pytest.skip("kitsune.quant and kitsune.whisper are not on this branch yet (WP5, WP6)")
    if quant is not None:
        assert tuple(quant.QUANT_FORMATS) == FR.QUANT_FORMATS
        assert set(FR.FORMAT_LABEL) == set(quant.QUANT_FORMATS)
    if whisper is not None:
        assert set(whisper.WHISPER_MODELS) == set(FR.WHISPER_SYSTEMS)
    assert set(FR.FULL_RUNS) == {"full-t06", "full-p03", "full-p01", "full-p005"}


def test_real_study_results(tmp_path):
    """The real study results (KITSUNE_STUDY_RESULTS = D:/kitsune-study/results: tables/, speed.json, params.json,
    report/report.json; the manifest beside it in ../selections/): every study row's M4 and set CERs equal the study
    report's, the manifest stacks on the study, and the A100 speed file is one group."""
    res = os.environ.get("KITSUNE_STUDY_RESULTS")
    if not res:
        no_real_data("KITSUNE_STUDY_RESULTS is not set")
    res = Path(res)
    man = res.parent / "selections" / "study_manifest.json"
    rc, rep = run(["--manifest", man, "--tables", res / "tables", "--speed", res / "speed.json",
                   "--params", res / "params.json", "--run-summaries", res / "runs" / "runs", "--boot-b", 1000],
                  tmp_path / "out")
    assert rc == 0 and rep["manifest"]["stacks_on_study"]
    study = json.loads((res / "report" / "report.json").read_text(encoding="utf-8"))
    for s, v in study["systems"].items():
        if s.endswith("-half"):
            continue
        assert rep["systems"][s]["metrics"]["m4"] == pytest.approx(v["metrics"]["m4"], rel=1e-12)
        for k in ss.M4_SETS:
            assert rep["systems"][s]["sets"][k]["cer"] == pytest.approx(v["sets"][k]["cer"], rel=1e-12)
    assert len(rep["speed_groups"]["groups"]) == 1
    assert next(c for c in rep["checks"] if c["rule"] == "speed_ids")["status"] == "pass"
