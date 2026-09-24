"""TensorBoard buckets: every tag the trainer logs goes to TensorBoard under 1_operational/, 2_loss_accuracy/ or 3_misc/
(kitsune.runlog.TB_BUCKET_RULES) while the open-format files keep the logged tag, and tools/regroup_tb.py rebuilds the
event files of an existing run in that layout from its open files. The tags of a real laptop run are frozen in
tests/tb_tags_overfit_1s.json, with the tags the eval-cadence change (mini evals, headline summary/*) adds to that run,
and must each land where the owner's table below puts them. CPU only."""
import contextlib
import importlib.util
import json
import os
import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # "" does not hide the GPU from the Windows CUDA driver; -1 does

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from kitsune.runlog import TB_BUCKETS, RunLogger, TagMapper, load_tag_map, tb_tag  # noqa: E402

CFG = {"run_name": "tiny", "student": "students/none", "hf": {"output_repo": None, "private": True}, "seed": 1}
FIXTURE = json.loads((ROOT / "tests" / "tb_tags_overfit_1s.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------ the owner's table
# Written out from the request without the regexes of runlog.py: 1 operational (time, memory, ...), 2 loss (train,
# val) and accuracy (characters gotten wrong), 3 misc (layers and the like); with the review's placements: token counts
# and CER denominators (ref_chars) are operational, each confidence bucket's token share is misc, top-1 per source and
# per confidence bucket is train accuracy, the early-stop metric and its best are loss, its bookkeeping operational.

SPLITS = ("", "_utt_mean", "_p1_gt_0.99", "_p1_lt_0.9")
LOSS = {f"{m}{s}" for m in ("kl", "ce") for s in SPLITS}
TOP1 = {f"top1{s}" for s in SPLITS}
CER = {"cer_ref_corpus", "cer_ref_mean", "cer_teacher_corpus", "cer_teacher_mean", "ratio_vs_teacher", "trunc_rate",
       "ref_edits", "cer_teacher_edits", "n_truncated", "n_empty_hyp", "teacher_cer_ref_corpus",
       "teacher_cer_ref_mean", "teacher_trunc_rate"}
CER_DEN = {"ref_chars", "cer_teacher_chars"}  # CER denominators: counts
TOK_DIAG = {"student_entropy_coarse", "student_tail", "teacher_entropy_coarse", "teacher_tail", "teacher_p1",
            "frac_p1_gt_0.99", "frac_p1_lt_0.9"}
EVAL_COST = {"wall_s", "rtf", "tok_per_s", "audio_s", "n_bad_audio", "n_utts", "n_tok", "n"}
SET_COST = {"audio_s", "tok_per_s", "n", "n_tok", "n_utts", "n_missing_teacher", "n_empty_ref"}
SECTION = {("tf", "loss"): "val_loss", ("tf", "top1"): "val_accuracy", ("greedy", "cer"): "val_accuracy",
           ("greedy_full", "cer"): "val_accuracy_full", ("probe", "loss"): "train_probe_loss",
           ("probe", "top1"): "train_probe_accuracy", ("probe_greedy", "cer"): "train_probe_accuracy"}


def expected_tb(tag: str, plugin: str = "scalars") -> str:
    p = tag.split("/")
    rest = "/".join(p[1:])
    if plugin == "histograms":
        return f"3_misc/{tag}"
    if plugin == "scalars" and p[0] == "summary":  # every eval's headline numbers, first in the loss/accuracy bucket
        assert len(p) == 3 and p[1] in ("full", "mini"), tag
        return f"2_loss_accuracy/00_summary/{rest}"
    if plugin == "scalars" and p[:2] == ["eval", "mini"]:  # a mini eval: where its full counterpart goes, marked mini
        full = expected_tb("/".join(["eval", *p[2:]]))
        bucket, section, tail = full.split("/", 2)
        if bucket == "2_loss_accuracy":
            return f"{bucket}/{section}_mini/{tail}"
        assert (bucket, section) in (("1_operational", "eval"), ("3_misc", "eval")), (tag, full)
        return f"{bucket}/eval/mini/{tail}"
    if plugin == "text":
        if tag == "config" or p[0] == "events":
            return f"1_operational/{tag}"
        assert tag == "samples", tag
        return "2_loss_accuracy/samples"
    if p[0] == "loss":  # the total (with the L2-SP value, which climbs all run) and l2sp apart from the objective terms
        return f"2_loss_accuracy/train_loss{'_incl_l2sp' if rest in ('total', 'l2sp') else ''}/{rest}"
    if p[0] in ("src", "bucket"):
        assert len(p) == 3, tag
        if (p[0], p[2]) == ("src", "tokens"):
            return f"1_operational/{tag}"
        if (p[0], p[2]) == ("bucket", "frac"):
            return f"3_misc/{tag}"
        split = "by_source" if p[0] == "src" else "by_teacher_confidence"
        if p[2] == "top1":
            return f"2_loss_accuracy/train_accuracy/{split}/{rest}"
        assert p[2] in ("kl", "ce"), tag
        return f"2_loss_accuracy/train_loss/{split}/{rest}"
    if p[0] == "early_stop":
        return f"2_loss_accuracy/{tag}" if tag in ("early_stop/value", "early_stop/best") else f"1_operational/{tag}"
    if tag == "tok/top1":
        return "2_loss_accuracy/train_accuracy/top1_agreement"
    if p[0] in ("time", "perf", "mem", "sys", "data", "sched") or tag == "opt/lr":
        return f"1_operational/{tag}"
    if p[0] in ("layers", "l2sp", "aug", "tok", "bn") or tag in ("opt/grad_norm", "opt/clip_coef"):
        return f"3_misc/{tag}"
    assert p[0] == "eval", tag
    if len(p) == 2:
        if tag == "eval/epoch":
            return "1_operational/progress/epoch"
        if tag == "eval/wall_s":
            return "1_operational/eval/wall_s"
        assert p[1] in ("kl_gap_heldout_minus_probe", "cer_teacher_gap_heldout_minus_probe"), tag
        return f"2_loss_accuracy/overfit_gap/{p[1]}"
    if len(p) == 3:
        assert p[2] in EVAL_COST, tag
        return f"1_operational/{tag}"
    kind, eset, m = p[1], p[2], p[3]
    if m in SET_COST or (m in CER_DEN and kind in ("greedy", "greedy_full", "probe_greedy")):
        return f"1_operational/{tag}"
    if kind in ("tf", "probe") and m in TOK_DIAG:
        return f"3_misc/{tag}"
    what = "loss" if m in LOSS else "top1" if m in TOP1 else "cer" if m in CER else None
    return f"2_loss_accuracy/{SECTION[(kind, what)]}/{eset}/{m}"


def test_real_run_tags_land_in_their_buckets():
    """Every tag of the real overfit-1s run (frozen copy) goes where the owner's table says, no rule is missing (no
    tag falls through to 3_misc unmatched) and no two tags share a TensorBoard tag."""
    assert len(FIXTURE["scalars"]) == 352 and len(FIXTURE["text"]) == 26
    for plugin in ("scalars", "text", "histograms"):
        tm = TagMapper()
        for tag in FIXTURE[plugin]:
            tb, warn = tm.resolve(tag, plugin)
            assert tb == expected_tb(tag, plugin), (tag, tb)
            assert tb_tag(tag, plugin) == (tb, tb.split("/", 1)[0])
            assert not warn, tag
        tbs = [e["tb_tag"] for e in tm.map.values()]
        assert len(set(tbs)) == len(tbs) == len(FIXTURE[plugin])
    by_bucket = Counter(tb_tag(t)[1] for t in FIXTURE["scalars"])
    assert by_bucket == {"1_operational": 81, "2_loss_accuracy": 113, "3_misc": 158}
    assert Counter(tb_tag(t, "text")[1] for t in FIXTURE["text"]) == {"1_operational": 25, "2_loss_accuracy": 1}
    # what the owner asked for, in words
    buckets = {t: tb_tag(t)[1] for t in FIXTURE["scalars"]}
    assert all(buckets[t] == "1_operational" for t in buckets if t.split("/")[0] in ("time", "mem", "perf", "sys"))
    assert all(buckets[t] == "2_loss_accuracy" for t in buckets if t.startswith("loss/") or "/cer_" in t)
    assert all(buckets[t] == "3_misc" for t in buckets if t.startswith(("layers/", "l2sp/")))
    ref_chars = [t for t in buckets if t.endswith("/ref_chars")]  # a CER denominator: a count, not an accuracy
    assert len(ref_chars) == 4 and all(buckets[t] == "1_operational" for t in ref_chars)
    assert {t: tb_tag(t)[0] for t in buckets if t.startswith(("src/", "bucket/"))} == {
        "src/galgame/kl": "2_loss_accuracy/train_loss/by_source/galgame/kl",
        "src/galgame/ce": "2_loss_accuracy/train_loss/by_source/galgame/ce",
        "src/galgame/top1": "2_loss_accuracy/train_accuracy/by_source/galgame/top1",
        "src/galgame/tokens": "1_operational/src/galgame/tokens",
        **{f"bucket/{b}/{m}": f"2_loss_accuracy/train_loss/by_teacher_confidence/{b}/{m}"
           for b in ("p1_gt_0.99", "p1_lt_0.9") for m in ("kl", "ce")},
        **{f"bucket/{b}/top1": f"2_loss_accuracy/train_accuracy/by_teacher_confidence/{b}/top1"
           for b in ("p1_gt_0.99", "p1_lt_0.9")},
        **{f"bucket/{b}/frac": f"3_misc/bucket/{b}/frac" for b in ("p1_gt_0.99", "p1_lt_0.9")}}


def test_eval_cadence_tags_land_in_their_buckets():
    """The tags the eval-cadence change adds to the same run (frozen next to it: the headline summary/*, the mini evals'
    eval/mini/*, the per-set vs-teacher CER numerator / denominator) go where the owner's table puts them, mapped
    together with the run's own tags: no rule missing, no TensorBoard tag shared."""
    added = FIXTURE["added_scalars"]
    assert len(added) == 215 and not set(added) & set(FIXTURE["scalars"])
    tm = TagMapper()
    for tag in [*FIXTURE["scalars"], *added]:
        tb, warn = tm.resolve(tag)
        assert tb == expected_tb(tag), (tag, tb)
        assert not warn, tag
    tbs = [e["tb_tag"] for e in tm.map.values()]
    assert len(set(tbs)) == len(tbs) == len(FIXTURE["scalars"]) + len(added)
    by_bucket = Counter(tb_tag(t)[1] for t in added)
    assert by_bucket == {"1_operational": 59, "2_loss_accuracy": 128, "3_misc": 28}
    summary = [t for t in added if t.startswith("summary/")]
    assert len(summary) == 24 and all(tb_tag(t)[0].startswith("2_loss_accuracy/00_summary/") for t in summary)
    mini = [t for t in added if t.startswith("eval/mini/")]
    assert all(tb_tag(t)[0].split("/")[1].endswith("_mini") for t in mini if tb_tag(t)[1] == "2_loss_accuracy")
    assert all(tb_tag(t)[0].startswith(("1_operational/eval/mini/", "3_misc/eval/mini/"))
               for t in mini if tb_tag(t)[1] != "2_loss_accuracy")
    # the 00_ prefix sorts the headline first among the loss/accuracy sections TensorBoard shows
    sections = sorted({tb_tag(t)[0].split("/")[1] for t in [*FIXTURE["scalars"], *added]
                       if tb_tag(t)[1] == "2_loss_accuracy"})
    assert sections[0] == "00_summary"


# one tag per rule, with the exact TensorBoard tag
SPOT = [
    ("summary/full/val_cer", "scalars", "2_loss_accuracy/00_summary/full/val_cer"),
    ("summary/full/val_cer_pct", "scalars", "2_loss_accuracy/00_summary/full/val_cer_pct"),
    ("summary/mini/train_cer_vs_teacher", "scalars", "2_loss_accuracy/00_summary/mini/train_cer_vs_teacher"),
    ("summary/mini/val_loss", "scalars", "2_loss_accuracy/00_summary/mini/val_loss"),
    ("eval/mini/tf/eval_jsut/kl", "scalars", "2_loss_accuracy/val_loss_mini/eval_jsut/kl"),
    ("eval/mini/tf/all/ce_p1_lt_0.9", "scalars", "2_loss_accuracy/val_loss_mini/all/ce_p1_lt_0.9"),
    ("eval/mini/tf/eval_cv8/top1", "scalars", "2_loss_accuracy/val_accuracy_mini/eval_cv8/top1"),
    ("eval/mini/greedy/eval_reazon/cer_ref_corpus", "scalars",
     "2_loss_accuracy/val_accuracy_mini/eval_reazon/cer_ref_corpus"),
    ("eval/mini/greedy/all/cer_teacher_edits", "scalars", "2_loss_accuracy/val_accuracy_mini/all/cer_teacher_edits"),
    ("eval/mini/probe/galgame/kl_utt_mean", "scalars", "2_loss_accuracy/train_probe_loss_mini/galgame/kl_utt_mean"),
    ("eval/mini/probe/all/top1", "scalars", "2_loss_accuracy/train_probe_accuracy_mini/all/top1"),
    ("eval/mini/probe_greedy/all/cer_teacher_corpus", "scalars",
     "2_loss_accuracy/train_probe_accuracy_mini/all/cer_teacher_corpus"),
    ("eval/mini/greedy/eval_jsut/ref_chars", "scalars", "1_operational/eval/mini/greedy/eval_jsut/ref_chars"),
    ("eval/mini/probe_greedy/all/cer_teacher_chars", "scalars",
     "1_operational/eval/mini/probe_greedy/all/cer_teacher_chars"),
    ("eval/mini/greedy/rtf", "scalars", "1_operational/eval/mini/greedy/rtf"),
    ("eval/mini/tf/eval_jsut/n_tok", "scalars", "1_operational/eval/mini/tf/eval_jsut/n_tok"),
    ("eval/mini/wall_s", "scalars", "1_operational/eval/mini/wall_s"),
    ("eval/mini/probe/all/student_tail", "scalars", "3_misc/eval/mini/probe/all/student_tail"),
    ("eval/greedy/eval_cv8/cer_teacher_edits", "scalars", "2_loss_accuracy/val_accuracy/eval_cv8/cer_teacher_edits"),
    ("eval/greedy_full/all/cer_teacher_chars", "scalars", "1_operational/eval/greedy_full/all/cer_teacher_chars"),
    ("loss/kl", "scalars", "2_loss_accuracy/train_loss/kl"),
    ("loss/objective", "scalars", "2_loss_accuracy/train_loss/objective"),
    ("loss/total", "scalars", "2_loss_accuracy/train_loss_incl_l2sp/total"),
    ("loss/l2sp", "scalars", "2_loss_accuracy/train_loss_incl_l2sp/l2sp"),
    ("src/galgame/kl", "scalars", "2_loss_accuracy/train_loss/by_source/galgame/kl"),
    ("src/galgame/top1", "scalars", "2_loss_accuracy/train_accuracy/by_source/galgame/top1"),
    ("src/galgame/tokens", "scalars", "1_operational/src/galgame/tokens"),
    ("bucket/p1_lt_0.9/ce", "scalars", "2_loss_accuracy/train_loss/by_teacher_confidence/p1_lt_0.9/ce"),
    ("bucket/p1_gt_0.99/top1", "scalars", "2_loss_accuracy/train_accuracy/by_teacher_confidence/p1_gt_0.99/top1"),
    ("bucket/p1_gt_0.99/frac", "scalars", "3_misc/bucket/p1_gt_0.99/frac"),
    ("tok/top1", "scalars", "2_loss_accuracy/train_accuracy/top1_agreement"),
    ("eval/tf/eval_jsut/kl_utt_mean", "scalars", "2_loss_accuracy/val_loss/eval_jsut/kl_utt_mean"),
    ("eval/tf/all/top1_p1_gt_0.99", "scalars", "2_loss_accuracy/val_accuracy/all/top1_p1_gt_0.99"),
    ("eval/greedy/eval_cv8/cer_ref_corpus", "scalars", "2_loss_accuracy/val_accuracy/eval_cv8/cer_ref_corpus"),
    ("eval/greedy_full/eval_reazon/trunc_rate", "scalars", "2_loss_accuracy/val_accuracy_full/eval_reazon/trunc_rate"),
    ("eval/probe/all/ce", "scalars", "2_loss_accuracy/train_probe_loss/all/ce"),
    ("eval/probe/galgame/top1", "scalars", "2_loss_accuracy/train_probe_accuracy/galgame/top1"),
    ("eval/probe_greedy/all/cer_teacher_corpus", "scalars",
     "2_loss_accuracy/train_probe_accuracy/all/cer_teacher_corpus"),
    ("eval/greedy/all/ref_edits", "scalars", "2_loss_accuracy/val_accuracy/all/ref_edits"),
    ("eval/greedy/all/ref_chars", "scalars", "1_operational/eval/greedy/all/ref_chars"),
    ("eval/greedy_full/eval_reazon/ref_chars", "scalars", "1_operational/eval/greedy_full/eval_reazon/ref_chars"),
    ("eval/probe_greedy/galgame/ref_chars", "scalars", "1_operational/eval/probe_greedy/galgame/ref_chars"),
    ("early_stop/value", "scalars", "2_loss_accuracy/early_stop/value"),
    ("early_stop/best", "scalars", "2_loss_accuracy/early_stop/best"),
    ("early_stop/evals_since_best", "scalars", "1_operational/early_stop/evals_since_best"),
    ("early_stop/triggered", "scalars", "1_operational/early_stop/triggered"),
    ("eval/kl_gap_heldout_minus_probe", "scalars", "2_loss_accuracy/overfit_gap/kl_gap_heldout_minus_probe"),
    ("samples", "text", "2_loss_accuracy/samples"),
    ("samples/probe", "text", "2_loss_accuracy/samples/probe"),
    ("time/step_s", "scalars", "1_operational/time/step_s"),
    ("sys/gpu0/mem_used_gb", "scalars", "1_operational/sys/gpu0/mem_used_gb"),
    ("opt/lr", "scalars", "1_operational/opt/lr"),
    ("eval/epoch", "scalars", "1_operational/progress/epoch"),
    ("eval/greedy_full/rtf", "scalars", "1_operational/eval/greedy_full/rtf"),
    ("eval/tf/eval_jsut/n_tok", "scalars", "1_operational/eval/tf/eval_jsut/n_tok"),
    ("eval/wall_s", "scalars", "1_operational/eval/wall_s"),
    ("events/oom_fallback", "text", "1_operational/events/oom_fallback"),
    ("config", "text", "1_operational/config"),
    ("layers/grad_norm/encoder.layers.0", "scalars", "3_misc/layers/grad_norm/encoder.layers.0"),
    ("l2sp/rel/proj_out", "scalars", "3_misc/l2sp/rel/proj_out"),
    ("opt/clip_coef", "scalars", "3_misc/opt/clip_coef"),
    ("bn/max_abs_drift", "scalars", "3_misc/bn/max_abs_drift"),
    ("tok/tail_student", "scalars", "3_misc/tok/tail_student"),
    ("eval/tf/eval_jsut/teacher_entropy_coarse", "scalars", "3_misc/eval/tf/eval_jsut/teacher_entropy_coarse"),
    ("loss/kl", "histograms", "3_misc/loss/kl"),  # every histogram is 3_misc, whatever its name
    ("weight/encoder.layers.0", "histograms", "3_misc/weight/encoder.layers.0"),
]


@pytest.mark.parametrize("tag,plugin,want", SPOT, ids=[f"{p}:{t}" for t, p, _ in SPOT])
def test_rule_table(tag, plugin, want):
    assert tb_tag(tag, plugin) == (want, want.split("/", 1)[0])
    assert TagMapper().resolve(tag, plugin) == (want, False)


def test_unmatched_and_colliding_tags_go_to_misc_once():
    tm = TagMapper()
    assert tm.resolve("grad_norm") == ("3_misc/grad_norm", True)
    assert tm.resolve("grad_norm") == ("3_misc/grad_norm", False)  # reported once
    assert tm.resolve("eval/tf/all/sum_new_term") == ("3_misc/eval/tf/all/sum_new_term", True)
    assert tm.resolve("src/a/new_metric") == ("3_misc/src/a/new_metric", True)  # src/bucket rules name their metrics
    assert tm.resolve("bucket/b/new_metric") == ("3_misc/bucket/b/new_metric", True)
    assert tm.resolve("early_stop/anything_new") == ("1_operational/early_stop/anything_new", False)
    assert tm.resolve("loss/by_source/a/kl") == ("2_loss_accuracy/train_loss/by_source/a/kl", False)
    # src/a/kl's rule output is taken: it goes to 3_misc instead of sharing a TensorBoard tag
    assert tm.resolve("src/a/kl") == ("3_misc/src/a/kl", True)
    saved = tm.to_json()
    assert saved["grad_norm"] == {"tb_tag": "3_misc/grad_norm", "bucket": "3_misc", "unmapped": True,
                                  "plugin": "scalars"}
    assert tb_tag("anything/else") == ("3_misc/anything/else", "3_misc")
    again = TagMapper(saved)  # a restart: nothing reported twice, nothing rewritten
    assert again.resolve("grad_norm") == ("3_misc/grad_norm", False) and not again.dirty
    # the same tag as a scalar and as text keeps one entry per plugin
    tm.resolve("samples", "text")
    tm.resolve("samples", "scalars")
    e = tm.to_json()["samples"]
    assert e["plugin"] == "text" and e["other_plugins"]["scalars"]["tb_tag"] == "2_loss_accuracy/samples"
    assert TagMapper(tm.to_json()).lookup("samples", "scalars")["tb_tag"] == "2_loss_accuracy/samples"


# ---------------------------------------------------------------------------------------------- regroup


def tb_view(run: Path) -> dict[str, pd.DataFrame]:
    """What TensorBoard shows of run/tb, by logged tag (export_run's tables without wall times)."""
    exp = load_tool("export_run")
    t = exp.tb_tables(run / "tb", load_tag_map(run / "metrics" / "tag_map.json"))
    out = {}
    for k, df in t.items():
        df = df.drop(columns=["wall_time", *(["sum", "sum_squares"] if k == "tb_histograms" else [])])
        out[k] = df.sort_values(["tag", "step"], kind="stable").reset_index(drop=True)
    return out


def make_run(path: Path, *, flat: bool | str = False, monkeypatch=None, summary: dict | None = None) -> Path:
    """A run with a resume from step 3 (steps 4-5 logged twice), histograms, text, an unmapped tag. flat=True logs the
    way the logger did before the buckets (logged tags in TensorBoard, no tag_map.json); flat="first" only the first
    launch, so the resume with the bucketed logger mixes both layouts in tb/."""
    with monkeypatch.context() if flat else contextlib.nullcontext() as mp:
        if flat:
            mp.setattr(RunLogger, "_tb_tag", lambda self, tag, plugin: tag)
        g = torch.Generator().manual_seed(0)
        log = RunLogger(path, CFG, sync_every_min=60, capture=False, tee=False)
        state = None
        for s in range(1, 6):
            log.step_row({"loss/kl": 1.0 / s, "opt/lr": 1e-3 * s, "layers/grad_norm/enc0": 0.1 * s, "weird": 2.0}, s)
            log.hist("weight/enc0", torch.randn(1000, generator=g), s)
            if s == 3:
                state = log.state_dict()
        log.text("note", "hello", 5)
        log.close()
        if flat == "first":
            mp.undo()
        time.sleep(0.05)
        log = RunLogger(path, CFG, sync_every_min=60, capture=False, tee=False, resume=state)
        for s in range(4, 7):
            log.step_row({"loss/kl": 10.0 / s, "opt/lr": 1e-3 * s, "layers/grad_norm/enc0": 1.0 * s, "weird": 3.0}, s)
            log.hist("weight/enc0", torch.randn(1000, generator=g), s)
        log.samples(6, [dict(id="a", cer_ref=0.5, cer_teacher=0.5, ref="あい", teacher_hyp="あい", hyp="あう")])
        log.close(summary=summary if summary is not None else {"status": "complete"})
    return path


def test_regroup_a_flat_run(tmp_path, monkeypatch, capsys):
    run = make_run(tmp_path / "runs" / "flat", flat=True, monkeypatch=monkeypatch)
    assert not (run / "metrics" / "tag_map.json").exists()
    before = tb_view(run)
    assert "loss/kl" in set(before["tb_scalars"]["tag"])  # a flat run: bucket columns say where regroup puts it
    assert before["tb_scalars"].set_index("tag").loc["loss/kl", "tb_tag"].iloc[0] == "2_loss_accuracy/train_loss/kl"
    originals = sorted(p.name for p in (run / "tb").iterdir())
    rg = load_tool("regroup_tb")
    assert rg.main([str(run)]) == 0
    out = capsys.readouterr().out
    assert "2_loss_accuracy" in out and "1 resume purge" in out and "scalars:weird" in out

    files = sorted(p.name for p in (run / "tb").iterdir())
    assert [f for f in files if "tfevents" in f and f.endswith(".regrouped")] and len(files) == len(originals) + 1
    assert sorted(f for f in files if f.endswith(".bak")) == sorted(rg.backup_name(f) for f in originals)
    assert not any("tfevents" in f for f in files if f.endswith(".bak"))  # TensorBoard skips them
    after = tb_view(run)
    for k in before:  # the same points, steps 4-5 of the first launch purged as before; only the TB tags changed
        assert before[k].equals(after[k]), k
    s = after["tb_scalars"]
    assert s["step"][s["tag"] == "loss/kl"].tolist() == [1, 2, 3, 4, 5, 6]
    assert s["value"][s["tag"] == "loss/kl"].tolist() == pytest.approx([1, 1 / 2, 1 / 3, 10 / 4, 10 / 5, 10 / 6])

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run / "tb"))
    acc.Reload()
    assert set(acc.Tags()["scalars"]) == {"2_loss_accuracy/train_loss/kl", "1_operational/opt/lr",
                                          "3_misc/layers/grad_norm/enc0", "3_misc/weird"}
    assert acc.Tags()["histograms"] == ["3_misc/weight/enc0"]
    assert {"1_operational/config/text_summary", "2_loss_accuracy/samples/text_summary", "3_misc/note/text_summary",
            "1_operational/events/logger_start/text_summary"} <= set(acc.Tags()["tensors"])
    tag_map = load_tag_map(run / "metrics" / "tag_map.json")
    assert tag_map["loss/kl"]["tb_tag"] == "2_loss_accuracy/train_loss/kl" and tag_map["weird"]["unmapped"] is True

    # idempotent: a second run rebuilds from the open files, the originals stay as they are, never two copies
    res = rg.regroup(run)
    files2 = sorted(p.name for p in (run / "tb").iterdir())
    assert [f for f in files2 if f.endswith(".bak")] == [f for f in files if f.endswith(".bak")]
    assert len([f for f in files2 if f.endswith(".regrouped")]) == 1 and len(files2) == len(files)
    assert all(tb_view(run)[k].equals(after[k]) for k in after)
    assert res["counts"]["2_loss_accuracy"]["scalars"] == 1 and res["counts"]["3_misc"]["histograms"] == 1
    assert res["records"]["purge"] == 1 and res["unmapped"] == ["scalars:weird", "text:note"]


def test_regroup_a_bucketed_run_changes_nothing_visible(tmp_path):
    """A run the new logger wrote (tags bucketed already, tb_tag_unmapped events among its events) rebuilds to the
    same TensorBoard view."""
    run = make_run(tmp_path / "runs" / "new")
    ev = [json.loads(x)["kind"] for x in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ev.count("tb_tag_unmapped") == 2  # weird and note, each once, across both launches
    before = tb_view(run)
    assert set(before["tb_scalars"]["tag"]) == {"loss/kl", "opt/lr", "layers/grad_norm/enc0", "weird"}
    load_tool("regroup_tb").regroup(run)
    after = tb_view(run)
    for k in before:
        assert before[k].equals(after[k]), k


def test_regroup_refuses_a_live_run(tmp_path, monkeypatch, capsys):
    rg = load_tool("regroup_tb")
    live = make_run(tmp_path / "runs" / "live", summary={})  # just written, no final status
    assert "not final" in rg.live_reason(live)
    with pytest.raises(SystemExit, match="looks live"):
        rg.regroup(live)
    assert not list((live / "tb").glob("*.bak"))  # nothing was touched
    assert rg.live_reason(live, now=time.time() + 400) is None  # quiet for 5 minutes: done
    (live / "summary.json").unlink()
    assert "no summary.json" in rg.live_reason(live)
    done = make_run(tmp_path / "runs" / "done")
    assert rg.live_reason(done) is None
    # a resumed run keeps the failed launch's summary.json until it ends
    summary = (done / "summary.json").read_bytes()
    (done / "summary.json").write_text('{"status": "failed"}', encoding="utf-8")
    os.utime(done / "summary.json", (time.time() - 60, time.time() - 60))
    assert "resumed" in rg.live_reason(done)
    (done / "summary.json").write_bytes(summary)
    assert rg.regroup(live, force=True)["live"]  # --force
    assert rg.main(["--all", "--runs-root", str(tmp_path / "runs")]) == 0
    out = capsys.readouterr().out
    assert f"skipped: {live} looks live" in out and f"{done}: rebuilt" in out and "2 runs under" in out
    assert len(list((done / "tb").glob("*.regrouped"))) == len(list((live / "tb").glob("*.regrouped"))) == 1


def kinds(run: Path) -> list[str]:
    return [json.loads(x)["kind"] for x in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if x]


def test_regroup_refuses_an_open_logger(tmp_path):
    """A launch whose last logger_start has no logger_close is live while anything it writes changed in the last 5
    minutes, even before its first scalar (scalars.jsonl empty) and when only its TensorBoard file moves."""
    rg = load_tool("regroup_tb")
    run = tmp_path / "runs" / "open"
    log = RunLogger(run, CFG, sync_every_min=60, capture=False, tee=False)
    try:
        assert (run / "metrics" / "scalars.jsonl").stat().st_size == 0 and rg.logger_open(run) is True
        assert "no logger_close" in rg.live_reason(run)
        with pytest.raises(SystemExit, match="looks live"):
            rg.regroup(run)
        assert not list((run / "tb").glob("*.bak")) and not list(run.glob(".regroup-*"))
        assert rg.live_reason(run, now=time.time() + 400) is None  # a killed launch never logs its close: 5 min
        log.tb.flush()  # nothing left in the writer's queue to land in the event file later
        old = time.time() - 1000
        files = [run / f for f in rg.ACTIVITY] + rg.event_files(run / "tb")
        for f in files:
            os.utime(f, (old, old))
        assert rg.live_reason(run) is None
        os.utime(files[-1], None)  # only the event file was written just now
        assert f"tb/{files[-1].name} changed" in rg.live_reason(run)
    finally:
        log.close()
    assert rg.logger_open(run) is False and rg.live_reason(run) is None  # closed, no scalar: nothing to wait for
    bare = tmp_path / "runs" / "bare"  # no events.jsonl at all: its recent writes decide
    (bare / "metrics").mkdir(parents=True)
    (bare / "metrics" / "scalars.jsonl").write_text("", encoding="utf-8")
    assert rg.logger_open(bare) is None and "no logger_start" in rg.live_reason(bare)


def _sizes(run: Path) -> dict[str, int]:
    return {f.name: f.stat().st_size for f in (run / "tb").iterdir()}


def test_regroup_puts_back_a_file_that_grows_after_the_rename(tmp_path, monkeypatch, capsys):
    """--force skips the live check, never the one after the rename: a writer that keeps its event file open goes on
    writing into the renamed .bak (the rename succeeds on Linux, and on Windows under TensorFlow's writer), so the
    originals are renamed back and the tool exits non-zero."""
    rg = load_tool("regroup_tb")
    run = make_run(tmp_path / "runs" / "busy")
    before, tag_map = _sizes(run), (run / "metrics" / "tag_map.json").read_bytes()
    rename = Path.rename

    def rename_then_flush(self, target):  # the writer's next flush lands in the renamed file
        out = rename(self, target)
        if str(target).endswith(".bak"):
            with open(target, "ab") as f:
                f.write(b"\0")
        return out

    monkeypatch.setattr(Path, "rename", rename_then_flush)
    with pytest.raises(rg.BeingWritten, match="run is being written") as e:
        rg.regroup(run, force=True)
    assert "renamed back" in str(e.value) and isinstance(e.value, SystemExit)
    after = _sizes(run)
    assert after.keys() == before.keys() and all(after[n] == before[n] + 1 for n in before)  # back, with the byte
    assert not list(run.glob(".regroup-*")) and (run / "metrics" / "tag_map.json").read_bytes() == tag_map
    with pytest.raises(SystemExit, match="run is being written"):  # the command line: exit status 1 with the message
        rg.main([str(run), "--force"])
    assert rg.main(["--all", "--force", "--runs-root", str(tmp_path / "runs")]) == 1
    assert "1 being written" in capsys.readouterr().out
    monkeypatch.undo()
    assert rg.regroup(run)["unmapped"] == ["scalars:weird", "text:note"]  # once the writer is gone it goes through


def test_regroup_with_a_real_writer_holding_the_file(tmp_path):
    """A SummaryWriter the live check cannot see (no logger, --force) keeps writing to its file during the rebuild: on
    Windows without TensorFlow the rename itself fails, otherwise the file grows after it; either way tb/ is left as it
    was and the tool refuses."""
    from torch.utils.tensorboard import SummaryWriter

    rg = load_tool("regroup_tb")
    run = make_run(tmp_path / "runs" / "writer")
    w = SummaryWriter(str(run / "tb"), flush_secs=3600)
    stop = threading.Event()

    def pump():
        i = 0
        while not stop.is_set():
            w.add_scalar("x", float(i), i)
            w.flush()
            i += 1
            stop.wait(0.1)

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    try:
        names = sorted(_sizes(run))
        with pytest.raises(rg.BeingWritten, match="run is being written|holds it open"):
            rg.regroup(run, force=True)
        assert sorted(_sizes(run)) == names and not list(run.glob(".regroup-*"))
    finally:
        stop.set()
        t.join()
        w.close()


def test_resume_into_a_flat_run_warns_once_and_regroup_unifies_it(tmp_path, monkeypatch):
    """A run logged before the buckets and resumed with the bucketed logger: one tb_layout_mixed event (not again on
    the next restart, which finds the tag map), TensorBoard shows both layouts, and regroup_tb leaves one."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run = make_run(tmp_path / "runs" / "mixed", flat="first", monkeypatch=monkeypatch)
    ev = [json.loads(x) for x in (run / "events.jsonl").read_text(encoding="utf-8").splitlines() if x]
    mixed = [e for e in ev if e["kind"] == "tb_layout_mixed"]
    assert len(mixed) == 1 and len(mixed[0]["files"]) == 1 and "tools/regroup_tb.py" in mixed[0]["hint"]
    assert [e["kind"] for e in ev].index("tb_layout_mixed") == max(i for i, e in enumerate(ev)
                                                                   if e["kind"] == "logger_start") + 1
    twin = tmp_path / "twin" / run.name
    shutil.copytree(run, twin)
    RunLogger(twin, CFG, sync_every_min=60, capture=False, tee=False).close()  # a later restart: warned already
    assert kinds(twin).count("tb_layout_mixed") == 1 and kinds(twin)[-1] == "logger_close"

    acc = EventAccumulator(str(run / "tb"))
    acc.Reload()
    assert {"loss/kl", "2_loss_accuracy/train_loss/kl"} <= set(acc.Tags()["scalars"])  # both layouts
    before = tb_view(run)
    s = before["tb_scalars"]
    assert s["step"][s["tag"] == "loss/kl"].tolist() == [1, 2, 3, 4, 5, 6]  # one series by logged tag
    rg = load_tool("regroup_tb")
    rg.regroup(run)
    acc = EventAccumulator(str(run / "tb"))
    acc.Reload()
    assert all(t.split("/", 1)[0] in TB_BUCKETS for k in ("scalars", "histograms", "tensors") for t in acc.Tags()[k])
    after = tb_view(run)
    for k in before:
        assert before[k].equals(after[k]), k


def test_export_without_tag_map_gives_back_the_logged_tags(tmp_path):
    """A bucketed run whose metrics/tag_map.json is missing (a partial copy): the export maps the open files' tags by
    the rules and gives the logged tag back for every TensorBoard tag, as it does with the file."""
    exp = load_tool("export_run")
    run = make_run(tmp_path / "runs" / "nomap")
    with_map = exp.export(str(run), tmp_path / "with_map")
    (run / "metrics" / "tag_map.json").unlink()
    without = exp.export(str(run), tmp_path / "without")
    for k in ("tb_scalars", "tb_histograms", "tb_text"):
        assert len(without[k]) and without[k].equals(with_map[k]), k
    assert set(without["tb_scalars"]["tag"]) == set(without["scalars"]["tag"]) == {"loss/kl", "opt/lr",
                                                                                   "layers/grad_norm/enc0", "weird"}
    assert set(without["tb_histograms"]["tag"]) == {"weight/enc0"}
    text_tags = set(without["tb_text"]["tag"])
    assert {"config", "note", "samples", "events/logger_start", "events/tb_tag_unmapped"} <= text_tags
    assert not any(without[k]["tag"].str.match(r"\d_").any() for k in ("tb_scalars", "tb_histograms", "tb_text"))
    assert without["tag_map"].equals(with_map["tag_map"])
    assert not (tmp_path / "without" / "tag_map.json").exists()
