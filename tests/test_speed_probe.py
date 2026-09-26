"""tools/speed_probe.py on tiny models on CPU: every kind (a Transcribe student and the Cohere path, a CTC student,
the Parakeet teacher's CTC and TDT paths) times a batched pass and batch-1 latencies on one fixed id list, writes the
record tools/study_report.py's Pareto view reads, merges systems into one file and refuses another id list; the id
draw is reproducible; --require-idle refuses a busy GPU. The decode it times is the evaluator's: the CTC student's
CER on the list equals kitsune.evaluate.ctc_eval's.

CPU only, tiny random models, a synthetic corpus in the real formats; the Transcribe kinds need the gated teacher
processor in the local HF cache (skipped without it, as tests/test_evaluate_script.py)."""
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
sys.path.insert(0, str(ROOT / "tools"))

import pytest  # noqa: E402
import torch  # noqa: E402

import speed_probe as sp  # noqa: E402
import study_report  # noqa: E402
from fixtures import load_script, make_fake_corpus, make_fake_selection  # noqa: E402

from kitsune import study_stats as ss  # noqa: E402
from kitsune import trainset  # noqa: E402
from kitsune.store import ids_sha256  # noqa: E402

SETS = ["eval_jsut", "eval_cv8"]
FAST = ["--device", "cpu", "--warmup", "1", "--warmup-1", "1", "--batch-s", "6"]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """An eval store of two sets, a tiny CTC student dir and a tiny converted-Parakeet dir."""
    from fixtures_ctc import tiny_parakeet_dir, tiny_student_dir

    root = tmp_path_factory.mktemp("speed")
    fc = make_fake_corpus(root / "corpus", sources={"src_a": (6, "train"), **{s: (7, "eval") for s in SETS}},
                          dur_range=(0.4, 2.0), token_range=(256, 296), no_second=tuple(SETS), seed=5)
    sel = make_fake_selection(fc, greedy_n=3, probe_n=2)
    store = trainset.eval_store(sel, fc.data, fc.teacher_out, root / "cache" / "eval", SETS)
    ctc, _ = tiny_student_dir(root / "ctc_student")
    pk, _ = tiny_parakeet_dir(root / "parakeet")
    return dict(root=root, store=store, store_dir=root / "cache" / "eval", ctc=ctc, parakeet=pk)


def run(env, out: Path, *argv) -> int:
    return sp.main([*argv, "--store", str(env["store_dir"]), "--out", str(out), *FAST])


def check_record(r: dict, n: int):
    assert r["n_utts"] == n and r["n_latency"] == n and len(r["latencies_s"]) == n and r["batches"] >= 1
    assert r["rtf"] > 0 and r["wall_s"] > 0 and 0 < r["p50_s"] <= r["p95_s"] and r["mean_s"] > 0
    assert r["rtf"] == pytest.approx(r["wall_s"] / r["audio_s"], rel=1e-3) and r["rtf_1_p50"] > 0
    assert r["device"] == "cpu" and r["dtype"] == "fp32" and r["gpu"] is None and r["idle"] is None
    assert r["vram_gb"] is None and r["vram_peak_reserved_bytes"] is None  # CPU: no VRAM
    assert r["params_total"] > 0 and r["weights_bytes"] >= 4 * r["params_total"]
    assert 0 <= r["cer_ref_corpus"] and r["warmup_batches"] == 1 and r["warmup_1"] == 1


def test_ctc_and_parakeet_paths_and_the_merged_file(env, tmp_path):
    """A CTC student, then the Parakeet teacher's CTC and TDT paths into the same file: one id list for all (the
    per-set draw, reused from the file), each a complete record; tools/study_report.py reads it for its Pareto view."""
    out = tmp_path / "speed.json"
    assert run(env, out, "--kind", "ctc", "--model", str(env["ctc"]), "--system", "study-p01", "--per-set", "3") == 0
    assert run(env, out, "--kind", "parakeet-ctc", "--model", str(env["parakeet"])) == 0
    assert run(env, out, "--kind", "parakeet-tdt", "--model", str(env["parakeet"])) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert set(doc["systems"]) == {"study-p01", "parakeet-ctc", "parakeet-tdt"} and doc["schema"] == 1
    assert doc["ids"] == sp.pick_ids(env["store"], 3, 1234) and doc["ids_sha256"] == ids_sha256(doc["ids"])
    assert len(doc["ids"]) == 6 and {i.split("/")[0] for i in doc["ids"]} == set(SETS)
    for name, r in doc["systems"].items():
        check_record(r, 6)
    assert doc["systems"]["parakeet-tdt"]["kind"] == "parakeet-tdt"
    speed = study_report.load_speed(out)
    e = ss.speed_entry(speed["parakeet-ctc"])
    assert e["rtf"] == doc["systems"]["parakeet-ctc"]["rtf"] and e["vram_gb"] is None
    assert e["p50_s"] == doc["systems"]["parakeet-ctc"]["p50_s"] and "p95_s" in e
    # a record is replaced by the same system's next one, never duplicated
    assert run(env, out, "--kind", "parakeet-ctc", "--model", str(env["parakeet"]), "--latency-n", "2") == 0
    doc2 = json.loads(out.read_text(encoding="utf-8"))
    assert len(doc2["systems"]) == 3 and doc2["systems"]["parakeet-ctc"]["n_latency"] == 2


def test_another_id_list_is_refused(env, tmp_path):
    """Every system of one file is timed on the same audio: an --ids list other than the file's is refused (exit 2)
    before any model is loaded; the file's own list (as a JSON list or one id per line) is accepted."""
    out = tmp_path / "speed.json"
    assert run(env, out, "--kind", "ctc", "--model", str(env["ctc"]), "--system", "a", "--per-set", "2") == 0
    ids = json.loads(out.read_text(encoding="utf-8"))["ids"]
    other = tmp_path / "other.txt"
    other.write_text("\n".join(ids[:-1]) + "\n", encoding="utf-8")
    assert run(env, out, "--kind", "ctc", "--model", str(env["ctc"]), "--system", "b", "--ids", str(other)) == 2
    same = tmp_path / "same.json"
    same.write_text(json.dumps(ids), encoding="utf-8")
    assert sp.read_ids(same) == ids and sp.read_ids(other) == ids[:-1]
    assert run(env, out, "--kind", "ctc", "--model", str(env["ctc"]), "--system", "b", "--ids", str(same)) == 0
    assert set(json.loads(out.read_text(encoding="utf-8"))["systems"]) == {"a", "b"}
    missing = tmp_path / "missing.txt"
    missing.write_text("nope/1\n", encoding="utf-8")
    assert run(env, tmp_path / "x.json", "--kind", "ctc", "--model", str(env["ctc"]), "--system", "c", "--ids",
               str(missing)) == 2


def test_the_timed_decode_is_the_evaluators(env, tmp_path):
    """The CTC decode the probe times is kitsune.evaluate.ctc_eval's: the same corpus CER on the id list."""
    from transformers import AutoProcessor

    from kitsune import ctc_student as CS
    from kitsune import evaluate as ev

    out = tmp_path / "speed.json"
    assert run(env, out, "--kind", "ctc", "--model", str(env["ctc"]), "--system", "s", "--per-set", "4") == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    store = env["store"]
    model = CS.load_ctc_student(env["ctc"], "cpu")
    res = ev.ctc_eval(model, store, doc["ids"], CS.ctc_features(env["ctc"], "cpu").logmel, "cpu", 6.0,
                      tokenizer=AutoProcessor.from_pretrained(str(env["ctc"])).tokenizer)
    g = res["greedy"]
    assert doc["systems"]["s"]["cer_ref_corpus"] == ev.corpus_cer(g["hyp"].tolist(), g["ref"].tolist())["cer"]


def test_a_busy_gpu_is_refused(env, tmp_path, monkeypatch):
    """--require-idle: another compute process on the GPU refuses (exit 3) before anything is loaded; nvidia-smi's
    listing is parsed without this process."""
    monkeypatch.setattr(sp, "gpu_state", lambda device: dict(processes=[dict(pid=1, name="python", used_mib="900")],
                                                             utilization=["0, 97, 900"]))
    monkeypatch.setattr(sp, "load_runner", lambda *a, **k: pytest.fail("a model was loaded on a busy GPU"))
    assert run(env, tmp_path / "s.json", "--kind", "ctc", "--model", str(env["ctc"]), "--system", "x",
               "--require-idle") == 3
    monkeypatch.undo()

    class Done:
        def __init__(self, out):
            self.stdout = out

    outs = iter([Done(f"{os.getpid()}, python, 100\n4242, other, 2000\n"), Done("0, 55, 2100\n")])
    monkeypatch.setattr(sp.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(sp.subprocess, "run", lambda *a, **k: next(outs))
    st = sp.gpu_state(torch.device("cuda"))
    assert st["processes"] == [dict(pid=4242, name="other", used_mib="2000")] and st["utilization"] == ["0, 55, 2100"]
    assert sp.gpu_state(torch.device("cpu")) is None


def test_the_id_draw_and_the_pins():
    """The per-set draw depends on (seed, set) only and is sorted within each set; the Cohere revision is the one the
    labels came from (scripts/03_build_student.py)."""
    class U:
        def __init__(self, i, s):
            self.id, self.source = i, s

    class St:
        utts = [U(f"b/{i}", "b") for i in range(20)] + [U(f"a/{i}", "a") for i in range(5)]

    a = sp.pick_ids(St, 3, 7)
    assert a == sp.pick_ids(St, 3, 7) and a != sp.pick_ids(St, 3, 8) and len(a) == 6
    assert a[:3] == sorted(a[:3]) and all(x.startswith("b/") for x in a[:3])
    St.utts = St.utts[:20] + [U("c/0", "c")] + St.utts[20:]  # another set changes no other set's draw
    assert [x for x in sp.pick_ids(St, 3, 7) if not x.startswith("c/")] == a
    from kitsune import student as S

    assert sp.TEACHER_REVISION == load_script("03_build_student").TEACHER_REVISION and sp.TEACHER_ID == S.TEACHER_ID


@pytest.fixture(scope="module")
def aed_dir(tmp_path_factory):
    """A tiny Transcribe student saved as 03 saves one (the teacher processor from the local HF cache)."""
    try:
        from transformers import AutoProcessor

        from kitsune import student as S

        proc = AutoProcessor.from_pretrained(S.TEACHER_ID)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"teacher processor not in the local HF cache: {e}")
    from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

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
    d = tmp_path_factory.mktemp("aed") / "student"
    S.save_student(student, d, proc, dict(format=1, stage="complete", spec=dict(enc_layers=[0, 1], ffn_dim=128,
                                                                                dec_layers=[0], tie_head=True)))
    return d


def test_transcribe_kinds(env, aed_dir, tmp_path):
    """A Transcribe student (and the Cohere path, here on the same tiny weights through --model): the teacher pass's
    greedy decode timed like the others; its CER on the list equals kitsune.evaluate.greedy_eval's."""
    from kitsune import evaluate as ev
    from kitsune import student as S
    from kitsune.features import LogMel
    from transformers import AutoProcessor

    out = tmp_path / "speed.json"
    assert run(env, out, "--kind", "aed", "--model", str(aed_dir), "--system", "study-t005", "--per-set", "2") == 0
    assert run(env, out, "--kind", "cohere", "--model", str(aed_dir), "--system", "cohere-tiny") == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    for r in doc["systems"].values():
        check_record(r, 4)
    assert doc["systems"]["cohere-tiny"]["kind"] == "cohere"
    proc = AutoProcessor.from_pretrained(str(aed_dir))
    model = S.load_student(aed_dir, "cpu")
    summ, g = ev.greedy_eval(model, env["store"], doc["ids"], LogMel.from_feature_extractor(proc.feature_extractor),
                             "cpu", 6.0, tokenizer=proc.tokenizer)
    assert doc["systems"]["study-t005"]["cer_ref_corpus"] == ev.corpus_cer(g["hyp"].tolist(), g["ref"].tolist())["cer"]
