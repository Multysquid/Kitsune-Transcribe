"""scripts/05_evaluate.py with --quant (the full runs' quantised readouts; kitsune.quant) on the size study's tiny CTC
student (tests/test_evaluate_study.py's ctc_env, its own teacher on a study manifest), on CPU with the emulated
formats and fp16:

  in memory     --quant nvfp4-w4a4 --quant-impl emulate: the tables under study-p01@nvfp4-w4a4, study.json's quant
                block (format, impl, simulated, source, counters, non-finite counts, bytes) and the new macro metrics,
                summary.json's and evaluator.json's quant block, the quant / quant_counters / nonfinite events, no
                verdict (verdict_skipped), the quant block in the --out identity; a rerun evaluates nothing and keeps
                the block; another format into the same --out, a --system without the suffix, --anchor, --teachers or
                --from-evals with --quant, and the quant flags without --quant are refused
  from a file   python -m kitsune.quant export, then 05 --ckpt <variant>: exactly the in-memory eval's hypotheses and
                teacher-forced tables (compare --exact) for int8-w8a8, nvfp4-w4a4 and fp16 (--fp16); a variant with
                another --quant, or with --quant-scope, is refused; the readout CLI (export + 05) end to end, reusing a
                verified variant on its retry
  metrics       study_block's m4_nostyle, m4_all, m4_all_nostyle and m3 on synthetic tables, the teacher's and the
                ratio for the raw ones only, the old keys unchanged
  amp           kitsune.evaluate.amp_dtype / _amp stay backward compatible, and greedy_generate runs a fp16-applied
                tiny AED under CPU fp16 autocast

CPU only, synthetic data in the repo's formats; nothing needs the network or torchao."""
import json
import os
import sys
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from fixtures import load_script  # noqa: E402
from test_evaluate_study import ctc_env  # noqa: E402,F401  (the module-scoped fixture: a CTC student on a manifest)

from kitsune import evaluate as ev  # noqa: E402
from kitsune import quant as Q  # noqa: E402
from kitsune import study_stats as ss  # noqa: E402
from kitsune.store import ids_sha256  # noqa: E402


@pytest.fixture(scope="module")
def prereg(tmp_path_factory):
    """The rules with their sidecar fields pending (the toy corpus is not the study's selection: the committed
    PREREG.json would refuse its manifest), and Parakeet's CTC baselines taken as pending, as test_evaluate_study
    does for its own 05 runs."""
    from kitsune import prereg as pr

    path = tmp_path_factory.mktemp("prereg") / "PREREG.json"
    path.write_bytes(pr.rules_json(pr.rules()))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(ev, "PARAKEET_CTC_CER_PREREG", None)
        yield path


@pytest.fixture(scope="module")
def m05():
    return load_script("05_evaluate")


def run05(m05, env, prereg, out: Path, *extra, ckpt=None) -> int:
    return m05.main(["--root", str(env["root"]), "--config", str(env["config"]), "--ckpt", str(ckpt or env["student"]),
                     "--out", str(out), "--prereg", str(prereg), "--chunk-s", "4", *extra])


def events(out: Path, kind: str) -> list[dict]:
    return [r for r in (json.loads(x) for x in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()
                        if x.strip()) if r["kind"] == kind]


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mem_run(ctc_env, prereg, m05, tmp_path_factory):
    out = tmp_path_factory.mktemp("quant_mem") / "out"
    tables = out.parent / "tables"
    assert run05(m05, ctc_env, prereg, out, "--quant", "nvfp4-w4a4", "--quant-impl", "emulate", "--tables",
                 str(tables)) == 0
    return dict(out=out, tables=tables)


def test_05_quant_in_memory_end_to_end(ctc_env, prereg, m05, mem_run):
    out, tables = mem_run["out"], mem_run["tables"]
    system = "study-p01@nvfp4-w4a4"
    assert sorted(p.name for p in (tables / system).glob("*.parquet")) == sorted(
        f"{s}.parquet" for s in ctc_env["manifest"]["sets"])
    assert not (tables / "study-p01").exists()  # the base system's tables are never touched
    st = load(out / "study.json")
    assert st["system"] == system and st["family"] == "ctc"
    qb = st["quant"]
    assert qb["format"] == "nvfp4-w4a4" and qb["impl"] == "emulate" and qb["simulated"] is True
    assert qb["source"] == "memory" and qb["base_system"] == "study-p01" and qb["scope"] == "linear+pw"
    assert qb["export_dir"] is None and qb["file_bytes"] is None and 0 < qb["share_quantized"] < 1
    assert qb["counters"]["calls"] > 0 and qb["counters"]["padded"] == 0 and qb["counters"]["fallback_risk"] == 0
    assert qb["nonfinite"]["rows"] == 0 and qb["nonfinite"]["batches"] == 0 and qb["nonfinite"]["forwards"] > 0
    assert qb["uncalled"] == [] and qb["invocations"] == 1 and qb["fp32_fallbacks"] == {}
    assert qb["weights_bytes"]["quantized"] > 0 and "torchao" in qb
    assert st["weights"]["path"] == str(ctc_env["student"]) and st["weights"]["file_bytes"] > 0
    m = st["metrics"]
    for k in ("m4", "m4_nostyle", "m4_all", "m4_all_nostyle", "m3", "m4_all_teacher", "m4_all_ratio", "m3_teacher",
              "m3_ratio"):
        assert k in m, k
    assert "m4_nostyle_teacher" not in m and "m4_all_nostyle_ratio" not in m
    assert m["m3"] == pytest.approx(np.mean([st["strata"][s]["cer"] for s in ev.M3_STRATA]))
    assert m["m4_all"] == pytest.approx(np.mean([st["strata"][s]["cer"] for s in ev.M4_ALL_STRATA]))
    s = load(out / "summary.json")
    assert s["quant"]["format"] == "nvfp4-w4a4" and s["quant"]["counters"] == qb["counters"]
    assert not (out / "verdict.json").exists() and events(out, "verdict_skipped")[-1]["note"].startswith("quant")
    (inv,) = load(out / "evaluator.json")["invocations"]
    assert inv["status"] == "complete" and inv["quant"]["fmt"] == "nvfp4-w4a4" and inv["quant"]["impl"] == "emulate"
    assert inv["quant"]["counters"]["calls"] == qb["counters"]["calls"] and inv["quant"]["uncalled"] == []
    assert inv["system"] == system and "torchao" in inv["versions"]
    (qe,) = events(out, "quant")
    assert qe["fmt"] == "nvfp4-w4a4" and qe["source"] == "memory" and qe["layers"] > 0
    assert events(out, "quant_counters")[-1]["uncalled"] == [] and events(out, "nonfinite")[-1]["rows"] == 0
    ident = load(out / ".parts" / "identity.json")
    assert ident["quant"] == dict(fmt="nvfp4-w4a4", impl="emulate", scope="linear+pw", mx_rounding="rceil", schema=1)
    # a rerun: nothing left to evaluate, the quant block stays (from .parts/quant)
    assert run05(m05, ctc_env, prereg, out, "--quant", "nvfp4-w4a4", "--quant-impl", "emulate", "--tables",
                 str(tables)) == 0
    assert events(out, "todo")[-1]["sets"] == [] and load(out / "study.json")["quant"]["counters"] == qb["counters"]
    assert load(out / "summary.json")["quant"]["format"] == "nvfp4-w4a4"


def test_05_quant_refusals(ctc_env, prereg, m05, mem_run, tmp_path, monkeypatch):
    out = mem_run["out"]
    with pytest.raises(SystemExit, match="holds results of other weights or eval settings"):
        run05(m05, ctc_env, prereg, out, "--quant", "int8-w8a8", "--quant-impl", "emulate", "--tables",
              str(mem_run["tables"]))
    with pytest.raises(SystemExit, match="must end in @int8-w8a8"):
        run05(m05, ctc_env, prereg, tmp_path / "a", "--quant", "int8-w8a8", "--system", "study-p01")
    with pytest.raises(SystemExit, match="need --quant"):
        run05(m05, ctc_env, prereg, tmp_path / "b", "--quant-scope", "linear")
    for extra in (["--anchor"], ["--teachers"], ["--from-evals", str(out), "--system", "x@fp16"]):
        with pytest.raises(SystemExit) as e:
            m05.parse_args(["--config", "c.json", "--out", str(tmp_path / "c"), "--quant", "fp16", *extra])
        assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        m05.parse_args(["--config", "c.json", "--ckpt", "x", "--out", "o", "--fp16", "--quant", "fp16"])
    assert e.value.code == 2
    assert m05.parse_args(["--config", "c.json", "--ckpt", "x", "--out", "o", "--fp16"]).quant == "fp16"
    monkeypatch.setattr(Q, "torchao_available", lambda: False)
    with pytest.raises(SystemExit, match="REFUSED: .*needs torchao"):
        run05(m05, ctc_env, prereg, tmp_path / "d", "--quant", "int8-w8a8", "--quant-impl", "torchao")


@pytest.fixture(scope="module")
def variants(ctc_env, tmp_path_factory):
    root = tmp_path_factory.mktemp("variants")
    return {fmt: (Q.export(ctc_env["student"], root / fmt, fmt), root / fmt)[1]
            for fmt in ("int8-w8a8", "nvfp4-w4a4", "fp16")}


@pytest.mark.parametrize("fmt", ["int8-w8a8", "nvfp4-w4a4", "fp16"])
def test_export_then_eval_equals_in_memory(ctc_env, prereg, m05, variants, tmp_path, fmt):
    """What ships is what is scored: 05 on the exported variant dir gives exactly the in-memory eval's hypotheses and
    teacher-forced tables (fp16 in memory through the --fp16 alias)."""
    a, b = tmp_path / "file", tmp_path / "mem"
    assert run05(m05, ctc_env, prereg, a, "--tables", str(tmp_path / "ta"), ckpt=variants[fmt]) == 0
    flag = ["--fp16"] if fmt == "fp16" else ["--quant", fmt]
    assert run05(m05, ctc_env, prereg, b, *flag, "--tables", str(tmp_path / "tb")) == 0
    res = Q.compare_eval_dirs(a, b, exact=True)
    assert res["same"] and set(res["sets"]) == set(ctc_env["manifest"]["sets"]), res
    sa, sb = load(a / "study.json"), load(b / "study.json")
    assert sa["system"] == sb["system"] == f"study-p01@{fmt}"
    assert sa["quant"]["source"] == "file" and sb["quant"]["source"] == "memory"
    assert sa["quant"]["export_dir"] == str(variants[fmt]) and sa["quant"]["file_bytes"] == (
        variants[fmt] / Q.WEIGHTS_FILE).stat().st_size
    assert sa["quant"]["nonfinite"]["rows"] == 0 and sa["metrics"]["m4"] == sb["metrics"]["m4"]
    assert sa["quant"]["impl"] == ("native" if fmt == "fp16" else "emulate")
    if fmt != "fp16":
        other = "int8-w8a16" if fmt != "int8-w8a16" else "fp8-w8a8"
        with pytest.raises(SystemExit, match="does not match"):
            run05(m05, ctc_env, prereg, tmp_path / "x", "--quant", other, ckpt=variants[fmt])
        with pytest.raises(SystemExit, match="drop --quant-scope"):
            run05(m05, ctc_env, prereg, tmp_path / "y", "--quant-scope", "linear", ckpt=variants[fmt])


def test_the_readout_cli(ctc_env, prereg, tmp_path, monkeypatch, capsys):
    """python -m kitsune.quant readout: exports <out>/variant, then 05 on it into <out> with <out>/tables; a retry
    reuses the verified variant of the same checkpoint; the system is <run_name>@<fmt>."""
    real = Q._load_05

    def pending():
        mod = real()
        mod.PREREG_JSON = prereg
        return mod

    monkeypatch.setattr(Q, "_load_05", pending)
    out = tmp_path / "readout"
    argv = ["readout", "--config", str(ctc_env["config"]), "--ckpt", str(ctc_env["student"]), "--fmt", "int8-w8a16",
            "--out", str(out), "--cache-dir", str(tmp_path / "cache"), "--manifest", str(ctc_env["manifest_path"]),
            "--root", str(ctc_env["root"])]
    assert Q.main(argv) == 0
    assert Q.verify_export(out / "variant") == [] and Q.read_recipe(out / "variant")["format"] == "int8-w8a16"
    st = load(out / "study.json")
    assert st["system"] == "study-p01@int8-w8a16" and st["quant"]["source"] == "file"
    assert (out / "tables" / "study-p01@int8-w8a16").is_dir()
    before = (out / "variant" / Q.QUANT_FILE).stat().st_mtime_ns
    capsys.readouterr()
    assert Q.main(argv) == 0
    assert "is reused" in capsys.readouterr().out and (out / "variant" / Q.QUANT_FILE).stat().st_mtime_ns == before


def _manifest():
    sets = {s: [f"{s}/{i}" for i in range(4)] for s in ("eval_jsut", "eval_cv8", "eval_reazon", "galgame")}
    man = {"schema": 1, "sets": {s: {"n": len(i), "ids_sha256": ids_sha256(i), "ids": i} for s, i in sets.items()}}
    gal = sets["galgame"]
    man["galgame_views"] = {v: {"ids": x, "ids_sha256": ids_sha256(x), "n": len(x)}
                            for v, x in (("neutral", gal[:2]), ("all", gal), ("label_box", gal[1:]))}
    return ss.parse_manifest(man)


def test_study_block_new_metrics(m05):
    """m4_all = mean(JSUT, CV8, Reazon, Galgame-all), m4_nostyle over M4's sets no-style, m3 over the gate sets; the
    teacher's and the ratio for the raw ones only; m4 and the old keys unchanged."""
    man = _manifest()
    rng = np.random.default_rng(0)
    refs = {i: "あいうえおかきくけこ"[: 4 + k] for s, ids in man.sets.items() for k, i in enumerate(ids)}

    def table(s, err):
        ids = man.sets[s]
        hyps = [refs[i][: max(len(refs[i]) - err - int(rng.integers(0, 2)), 0)] + ("ア" if j % 2 else "")
                for j, i in enumerate(ids)]
        return ev.utterance_table(ids, s, [refs[i] for i in ids], hyps)

    tables = {s: table(s, e) for s, e in zip(man.sets, (1, 2, 0, 3))}
    teacher = {s: table(s, 1) for s in man.sets}
    blk = m05.study_block(man, tables, teacher)
    rows, m = blk["strata"], blk["metrics"]
    assert m["m4_all"] == pytest.approx(np.mean([rows[s]["cer"] for s in ev.M4_ALL_STRATA]))
    assert m["m4_all_nostyle"] == pytest.approx(np.mean([rows[s]["cer_nostyle"] for s in ev.M4_ALL_STRATA]))
    assert m["m4_nostyle"] == pytest.approx(np.mean([rows[s]["cer_nostyle"] for s in ss.M4_SETS]))
    assert m["m3"] == pytest.approx(np.mean([rows[s]["cer"] for s in ev.GATE_SETS]))
    assert m["m4"] == pytest.approx(np.mean([rows[s]["cer"] for s in ss.M4_SETS]))
    assert m["m4_all_ratio"] == pytest.approx(m["m4_all"] / m["m4_all_teacher"])
    assert m["m3_ratio"] == pytest.approx(m["m3"] / m["m3_teacher"])
    assert not any(k.startswith(("m4_nostyle_", "m4_all_nostyle_")) for k in m)
    assert {"m4", "m4_teacher", "m4_ratio", "ood", "ind", "jg", "jg_nostyle", "gate_pooled"} <= set(m)
    assert "m4_all" not in ss.METRICS and "m3" not in ss.METRICS  # the pre-registered report's list is untouched
    no_gal = {s: t for s, t in tables.items() if s != "galgame"}
    m2 = m05.study_block(man, no_gal, None)["metrics"]
    assert "m4_all" not in m2 and "m3" in m2 and "m3_teacher" not in m2


def test_amp_dtype_backward_compat():
    from test_quant import tiny_aed

    assert ev.amp_dtype(True) is torch.bfloat16 and ev.amp_dtype(False) is torch.bfloat16
    assert ev.amp_dtype(torch.float16) is torch.float16 and ev._amp("cpu", None) is False
    assert ev._amp("cpu", torch.float16) is torch.float16 and ev._amp("cuda", None) is True
    m = tiny_aed()
    Q.apply(m, "fp16")
    feats = torch.randn(2, 100, 128)
    fmask = torch.ones(2, 100, dtype=torch.long)
    with ev._eval_mode(m), ev._fp32_head(m):
        rows = ev.greedy_generate(m, feats, fmask, 1.0, prompt_ids=[13764, 7, 4, 16, 98, 98, 5, 9, 11, 13], eos=3,
                                  pad=2, amp=torch.float16)
    assert len(rows) == 2 and all(isinstance(ids, list) for ids, _ in rows)
