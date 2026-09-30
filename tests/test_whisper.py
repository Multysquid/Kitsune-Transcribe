"""The Whisper yardsticks (WP6; kitsune/whisper.py, tools/whisper_eval.py) on tiny offline models, CPU only.

  pins       WHISPER_MODELS: the four keys of the build contract, large-v3 and Kotoba at 02b's pins, turbo's and small's
             40-hex revisions, the row caps 64 / 128
  rules      max_new_tokens is the students' rule (443 at 448) and generate takes one more, never two; whisper_batches
             cuts the eval's own batches at the row cap in order (a 273-row batch -> 64, 64, 64, 64, 17);
             split_generated tells an EOS end from a repetition stop that pad == EOS hides, and from the length cap,
             and strips and counts timestamp ids; RecordingRepetitionStop records the first fire only
  decode     a tiny random Whisper (tests/fixtures_whisper.py): the prompt is asserted (drift refuses), generate runs
             once per batch with force_unique_generate_call and the fallback off, the stop sees the 4-token prompt, the
             fp32 head changes no id in fp32, an 80-mel model loads with its own extractor; scripted logits through a
             real generate call give the three ends (repetition / eos / length) with pad == EOS
  snapshots  fetch asks for the pinned revision and the allowed files only (a fake snapshot_download: no network),
             refuses a snapshot without its weights, and retries a transient Hub error (finish.hub_retry) but not a
             404 or an offline miss; resolve takes a key or a dir
  whisper_eval  end to end on an eval store: greedy_<set>.parquet (no teacher_hyp), whisper.json, the row-cap split,
             05's tables and study.json (family whisper, teacher null) that load in the study report; a second run
             decodes nothing; a stopped --force continues where it stopped (.parts/force.json); a partial re-run keeps
             every set's table; another --max-rows, a store that is not the manifest's, a frame store, a set outside
             the manifest and a --model typo refuse (exit 2) before any model is loaded, a store not built yet exits 1;
             --limit-per-set is speed_probe's draw and cannot be tabled (the box template's --tables is skipped with a
             note, not refused: smoke B's quick items); a heartbeat per batch and a bounded one around the load;
             --fetch-only; never exit 3
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tests"), str(ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import study_report  # noqa: E402
import whisper_eval as we  # noqa: E402
from fixtures import load_script  # noqa: E402
from fixtures_whisper import EOS, EVAL_SETS, MAX_TARGET, NO_TS, PROMPT, V, tiny_whisper_dir, whisper_env  # noqa: E402

from kitsune import evaluate as ev  # noqa: E402
from kitsune import heartbeat  # noqa: E402
from kitsune import study_stats as ss  # noqa: E402
from kitsune import trainset  # noqa: E402
from kitsune import whisper as W  # noqa: E402
from kitsune.store import ids_sha256  # noqa: E402

HEX40 = re.compile(r"^[0-9a-f]{40}$")


# ------------------------------------------------------------------------------------------------ pins and rules


def test_the_pins_and_the_row_caps():
    """The four systems of the build contract, each pinned to a 40-hex revision; large-v3 and Kotoba at the very
    commits scripts/02b_second_opinion.py pins (Kotoba's is the model that chose the Galgame-neutral rows); the row caps
    64 (large-v3) and 128 (the others)."""
    m = W.WHISPER_MODELS
    assert list(m) == ["whisper-large-v3", "whisper-large-v3-turbo", "kotoba-whisper-v2.0", "whisper-small"]
    assert all(k == s.system and HEX40.match(s.revision) for k, s in m.items())
    b02 = load_script("02b_second_opinion")
    assert (m["whisper-large-v3"].repo, m["whisper-large-v3"].revision) == (b02.WHISPER_TOK, b02.WHISPER_TOK_REVISION)
    assert (m["kotoba-whisper-v2.0"].repo, m["kotoba-whisper-v2.0"].revision) == (b02.MODEL2, b02.MODEL2_REVISION)
    assert m["whisper-large-v3-turbo"].revision == "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"
    assert (m["whisper-small"].repo, m["whisper-small"].revision) == ("openai/whisper-small",
                                                                      "973afd24965f72e36ca33b3055d56a652f456b4d")
    assert [s.max_rows for s in m.values()] == [64, 128, 128, 128]
    assert W.PROMPT_LEN == 4 and W.WHISPER_META == "whisper.json" and W.FAMILY == "whisper"
    assert "model.safetensors" in W.ALLOW_PATTERNS and not any("*" in p for p in W.ALLOW_PATTERNS)


def test_max_new_is_the_students_rule(tiny_dir):
    """max_new_tokens = kitsune.evaluate.greedy_generate's min(int(16 + 10 s), positions - prompt - 1): 443 at
    Whisper's 448. Whisper's generate itself takes one more new token (positions - 4) and refuses two more."""
    for s in (0.3, 1.0, 5.49, 29.9, 42.0, 43.0):
        assert W.max_new_tokens(s, 448) == min(int(16 + 10 * s), 448 - 4 - 1)
    assert W.max_new_tokens(43.0, 448) == 443 and W.max_new_tokens(29.9, 448) == 315
    assert W.max_new_tokens(1.0, 448) == 26
    wm = W.load_whisper(tiny_dir, "cpu", torch.float32)
    cap = W.max_new_tokens(99.0, MAX_TARGET)
    assert cap == MAX_TARGET - 5
    feats = W.features(wm, [np.zeros(8000, np.float32)])
    kw = dict(input_features=feats, language="ja", task="transcribe", return_timestamps=False, num_beams=1,
              do_sample=False, temperature=None, force_unique_generate_call=True)
    assert wm.model.generate(max_new_tokens=cap + 1, **kw).shape[0] == 1
    with pytest.raises(ValueError, match="max_target_positions"):
        wm.model.generate(max_new_tokens=cap + 2, **kw)


def _utts(durations):
    return [trainset.Utt(f"s/{i}", "s", float(d), 1, 0, 0, 0) for i, d in enumerate(durations)]


def test_whisper_batches_cut_the_eval_batches_at_the_row_cap():
    """A 273-row eval batch (the students' batch of short clips) is cut into 64, 64, 64, 64, 17 at cap 64; the pieces
    keep the eval's order and hold every row once; no cap leaves the batches as they are."""
    utts = _utts([200.0] * 3 + [1.0] * 273)  # three clips over half the budget each get a batch of their own
    pooled = trainset.eval_batches(utts, 273.0)
    assert [len(b) for b in pooled] == [1, 1, 1, 273]
    cut = W.whisper_batches(utts, 273.0, 64)
    assert [len(b) for b in cut] == [1, 1, 1, 64, 64, 64, 64, 17]
    assert [i for b in cut for i in b] == [i for b in pooled for i in b]
    assert sorted(i for b in cut for i in b) == list(range(len(utts)))
    assert W.whisper_batches(utts, 273.0, None) == pooled == W.whisper_batches(utts, 273.0, 0)
    sub = list(range(5, 200))
    assert [i for b in W.whisper_batches(utts, 273.0, 64, sub) for i in b] == [
        i for b in trainset.eval_batches(utts, 273.0, sub) for i in b]
    assert W.split_rows([[1, 2, 3], [4]], 2) == [[1, 2], [3], [4]]


def test_split_generated_with_pad_equal_to_eos():
    """pad == EOS: the first EOS ends the row only when it comes before the repetition stop fired; a row the stop cut is
    "repetition" (its EOS-coloured padding is not an end); neither is "length"; hyp_ids keep EOS as greedy_generate's
    do, text_ids drop it and every timestamp id (counted)."""
    ts = 64
    assert W.split_generated([10, 11, EOS, EOS, EOS], eos=EOS, timestamp_begin=ts, fired_at=None, max_new=9) == (
        [10, 11, EOS], [10, 11], "eos", 0)
    # an ended row's padding later looks periodic too: the EOS came first
    assert W.split_generated([10, EOS] + [EOS] * 30, eos=EOS, timestamp_begin=ts, fired_at=26, max_new=40)[2] == "eos"
    rep = [10] * 24 + [EOS] * 3
    assert W.split_generated(rep, eos=EOS, timestamp_begin=ts, fired_at=24, max_new=40) == (
        [10] * 24, [10] * 24, "repetition", 0)
    # the stop fired at the step whose next token is the first pad: still the stop's
    assert W.split_generated([12] * 25 + [EOS], eos=EOS, timestamp_begin=ts, fired_at=25, max_new=40)[2] == "repetition"
    assert W.split_generated([10, 11, 12], eos=EOS, timestamp_begin=ts, fired_at=None, max_new=3) == (
        [10, 11, 12], [10, 11, 12], "length", 0)
    assert W.split_generated([10, 70, 71, 11, EOS], eos=EOS, timestamp_begin=ts, fired_at=None, max_new=9) == (
        [10, 70, 71, 11, EOS], [10, 11], "eos", 2)


def test_the_recording_stop_keeps_the_first_fire():
    """RecordingRepetitionStop answers as RepetitionStop and records each row's generated length at its first fire
    (later calls on the padded row do not move it)."""
    stop = W.RecordingRepetitionStop(4)
    base = W.RepetitionStop(4)
    prompt = torch.tensor([PROMPT, PROMPT])
    loop, calm = [7] * 30, list(range(8, 38))
    for n in range(1, 31):
        ids = torch.cat([prompt, torch.tensor([loop[:n], calm[:n]])], dim=1)
        assert torch.equal(stop(ids, None), base(ids, None))
    assert stop.fired() == {0: 24}
    assert W.RecordingRepetitionStop(4).fired() == {}


# ------------------------------------------------------------------------------------------------ the decode


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    return tiny_whisper_dir(tmp_path_factory.mktemp("tiny_whisper") / "model")


def _waves(n, seed=0):
    rng = np.random.default_rng(seed)
    return [(0.1 * rng.standard_normal(int(16000 * d))).astype(np.float32) for d in rng.uniform(0.5, 2.0, n)]


def test_tiny_decode_on_cpu(tiny_dir, monkeypatch):
    """load_whisper reads the prompt ids from the generation config; greedy_whisper calls generate once per batch with
    the students' settings (language ja, transcribe, no timestamps, greedy, no temperature fallback and no thresholds,
    one call, the max_new rule), the stop sees the 4 prompt tokens in front of the generated ids, every row gets a
    stop kind; the fp32 head changes no id in fp32; a prompt the output does not start with refuses."""
    from transformers.generation.utils import GenerationMixin

    wm = W.load_whisper(tiny_dir, "cpu", torch.float32)
    assert wm.prompt_ids == PROMPT and wm.eos == EOS and wm.timestamp_begin == NO_TS + 1 == V
    assert wm.n_mels == 128 and wm.params_total == sum(p.numel() for p in wm.model.parameters())
    assert wm.weights_file_bytes == (tiny_dir / "model.safetensors").stat().st_size
    calls, seen, inner = [], [], []
    real_generate = type(wm.model).generate
    real_inner = GenerationMixin.generate
    real_stop = W.RecordingRepetitionStop.__call__

    def spy(self, *a, **k):
        calls.append(k)
        return real_generate(self, *a, **k)

    def spy_inner(self, *a, **k):
        inner.append(1)
        return real_inner(self, *a, **k)

    def spy_stop(self, input_ids, scores, **k):
        seen.append(input_ids[:, :4].tolist())
        return real_stop(self, input_ids, scores, **k)

    monkeypatch.setattr(type(wm.model), "generate", spy)
    monkeypatch.setattr(GenerationMixin, "generate", spy_inner)
    monkeypatch.setattr(W.RecordingRepetitionStop, "__call__", spy_stop)
    waves = _waves(3)
    longest = max(len(w) for w in waves) / 16000
    feats = W.features(wm, waves)
    assert tuple(feats.shape) == (3, 128, 3000) and feats.dtype == torch.float32
    rows = W.greedy_whisper(wm, feats, longest)
    assert len(calls) == 1 and len(inner) == 1
    k = calls[0]
    # only Whisper's own arguments next to the config (transformers 5.13 deprecates generation kwargs beside one)
    assert set(k) == {"input_features", "generation_config", "language", "task", "return_timestamps", "temperature",
                      "stopping_criteria", "force_unique_generate_call"}
    assert (k["language"], k["task"], k["return_timestamps"], k["temperature"], k["force_unique_generate_call"]) == (
        "ja", "transcribe", False, None, True)
    gc = k["generation_config"]
    assert (gc.max_new_tokens, gc.num_beams, gc.do_sample) == (W.max_new_tokens(longest, MAX_TARGET), 1, False)
    assert gc.no_speech_threshold is None and gc.logprob_threshold is None and gc.compression_ratio_threshold is None
    # the snapshot sets the reference decoding's thresholds (as a published config may): off on the loaded model, so
    # transformers cannot refill them into the call's config; its forced_decoder_ids are overridden by language / task
    saved = json.loads((tiny_dir / "generation_config.json").read_text(encoding="utf-8"))
    assert saved["no_speech_threshold"] == 0.6 and saved["forced_decoder_ids"][0] == [1, None]
    assert all(getattr(wm.model.generation_config, t) is None for t in W.FALLBACK_THRESHOLDS)
    assert seen and all(r == PROMPT for batch in seen for r in batch)
    assert len(rows) == 3
    for r in rows:
        assert r["stop"] in ("eos", "repetition", "length") and r["truncated"] == (r["stop"] != "eos")
        assert r["max_new"] == gc.max_new_tokens and len(r["hyp_ids"]) <= r["max_new"]
        assert r["n_timestamp_tokens"] == 0 and EOS not in r["text_ids"]
    texts = W.decode_texts(wm, rows)
    assert all(isinstance(t, str) and t == t.strip() for t in texts)
    monkeypatch.undo()
    off = W.greedy_whisper(wm, feats, longest, fp32_head=False)
    assert [r["hyp_ids"] for r in off] == [r["hyp_ids"] for r in rows]  # fp32 weights: the fp32 head is exact
    assert wm.model.proj_out.__class__.__name__ == "Linear"  # the head is back after the call
    wm.prompt_ids = [PROMPT[0], PROMPT[1], 7, PROMPT[3]]  # translate: not what generate was asked for
    with pytest.raises(RuntimeError, match="API changed"):
        W.greedy_whisper(wm, feats, longest)


def test_a_scripted_decode_tells_the_three_ends_apart(tiny_dir):
    """The pad == EOS trap through a real generate call (the LM head swapped for scripted logits): a row that loops is
    cut by the repetition stop at its first fire and padded with EOS, yet it is "repetition" with exactly its 24 loop
    ids; a row that emits EOS is "eos" with the EOS kept in hyp_ids; a row that never ends runs to max_new ("length").
    If transformers ran the criteria at another point of a step, or padded finished rows otherwise, these would move
    (and whisper.json's n_repetition / n_truncated with them)."""
    wm = W.load_whisper(tiny_dir, "cpu", torch.float32)

    class Scripted(torch.nn.Module):
        """Row 0: token 10 forever; row 1: 11, 12, then EOS; row 2: a new token every step (never periodic)."""

        def __init__(self):
            super().__init__()
            self.step = 0
            self.weight = torch.nn.Parameter(torch.zeros(V, 16))

        def forward(self, h):
            out = torch.full((h.shape[0], h.shape[1], V), -1e4)
            s, self.step = self.step, self.step + 1
            out[0, -1, 10] = 0
            out[1, -1, [11, 12, EOS][min(s, 2)]] = 0
            out[2, -1, 13 + s % 45] = 0
            return out

    wm.model.proj_out = Scripted()
    rows = W.greedy_whisper(wm, W.features(wm, [np.zeros(32000, np.float32)] * 3), 2.0, fp32_head=False)
    mx = W.max_new_tokens(2.0, MAX_TARGET)
    assert [r["stop"] for r in rows] == ["repetition", "eos", "length"]
    assert rows[0]["hyp_ids"] == rows[0]["text_ids"] == [10] * 24 and rows[0]["truncated"]
    assert rows[1]["hyp_ids"] == [11, 12, EOS] and rows[1]["text_ids"] == [11, 12] and not rows[1]["truncated"]
    assert rows[2]["hyp_ids"] == [13 + s for s in range(mx)] and rows[2]["max_new"] == mx and rows[2]["truncated"]
    assert all(r["n_timestamp_tokens"] == 0 for r in rows)


def test_an_80_mel_model_loads_its_own_extractor(tmp_path):
    """whisper-small has 80 mel bins: the extractor comes from the snapshot, never a fixed 128; a snapshot whose
    extractor and model disagree refuses."""
    d = tiny_whisper_dir(tmp_path / "m80", n_mels=80)
    wm = W.load_whisper(d, "cpu", torch.float32)
    assert wm.n_mels == 80
    feats = W.features(wm, _waves(2, 1))
    assert tuple(feats.shape) == (2, 80, 3000)
    assert len(W.greedy_whisper(wm, feats, 2.0)) == 2
    cfg = json.loads((d / "preprocessor_config.json").read_text(encoding="utf-8"))
    (d / "preprocessor_config.json").write_text(json.dumps(dict(cfg, feature_size=128)), encoding="utf-8")
    with pytest.raises(ValueError, match="mel bins"):
        W.load_whisper(d, "cpu", torch.float32)


def test_fetch_and_resolve_without_the_network(tiny_dir, tmp_path, monkeypatch):
    """fetch asks huggingface_hub for the pinned revision, the allowed files only, into the given cache; a snapshot
    without its weights refuses; resolve takes a key (fetched) or a model dir (as it is), nothing else."""
    import huggingface_hub

    asked = []

    def fake(**k):
        asked.append(k)
        return str(tiny_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake)
    spec = W.WHISPER_MODELS["whisper-small"]
    assert W.fetch(spec, tmp_path / "hf") == tiny_dir
    assert asked[-1] == dict(repo_id="openai/whisper-small", revision=spec.revision,
                             allow_patterns=list(W.ALLOW_PATTERNS), cache_dir=str(tmp_path / "hf"))
    assert W.resolve("kotoba-whisper-v2.0", None) == (tiny_dir, W.WHISPER_MODELS["kotoba-whisper-v2.0"])
    assert asked[-1]["revision"] == "7eb575277d18909a4af8a24e3ae8cce2e99794ae" and asked[-1]["cache_dir"] is None
    assert W.resolve(str(tiny_dir)) == (tiny_dir, None)
    with pytest.raises(ValueError, match="neither a Whisper key"):
        W.resolve("openai/whisper-medium")
    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **k: str(bare))
    with pytest.raises(RuntimeError, match="model.safetensors"):
        W.fetch(spec)


def test_fetch_retries_a_transient_hub_error(tiny_dir, monkeypatch):
    """The Whisper item's only download backs off like every other box download (vast/finish.py hub_retry): a
    transient error (snapshot_download's LocalEntryNotFoundError after a 5xx or a dropped connection on an empty
    cache) is retried and the second call's snapshot is used; a refusal the Hub repeats (a 404: no such revision) is
    raised at once; offline (HF_HUB_OFFLINE, the laptop and the tests) it is one call, no back-off."""
    import httpx
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError, RevisionNotFoundError

    from kitsune.study_queue import _finish

    monkeypatch.setattr(_finish(), "HUB_RETRY_WAITS", (0, 0))  # the back-off's waits, not its logic
    monkeypatch.setattr(huggingface_hub, "is_offline_mode", lambda: False)
    calls = []

    def flaky(**k):
        calls.append(k)
        if len(calls) == 1:
            raise LocalEntryNotFoundError("Got: ConnectError: connection reset\nAn error happened while trying to "
                                          "locate the files on the Hub")
        return str(tiny_dir)

    spec = W.WHISPER_MODELS["whisper-large-v3"]
    monkeypatch.setattr(huggingface_hub, "snapshot_download", flaky)
    assert W.fetch(spec) == tiny_dir and len(calls) == 2
    assert calls[0] == calls[1] and calls[0]["revision"] == spec.revision

    def gone(**k):
        calls.append(k)
        raise RevisionNotFoundError("404 Client Error: Revision Not Found", response=httpx.Response(
            404, request=httpx.Request("GET", "https://huggingface.co/api/models/openai/whisper-large-v3")))

    calls.clear()
    monkeypatch.setattr(huggingface_hub, "snapshot_download", gone)
    with pytest.raises(RevisionNotFoundError):
        W.fetch(spec)
    assert len(calls) == 1
    calls.clear()
    monkeypatch.setattr(huggingface_hub, "is_offline_mode", lambda: True)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", flaky)
    with pytest.raises(LocalEntryNotFoundError):
        W.fetch(spec)
    assert len(calls) == 1


# ------------------------------------------------------------------------------------------------ whisper_eval


@pytest.fixture(scope="module")
def wenv(tmp_path_factory):
    return whisper_env(tmp_path_factory.mktemp("whisper_env"))


def argv_of(env, model, out, *extra, tables=None, max_rows="2"):
    a = ["--model", str(model), "--system", "whisper-tiny", "--store", str(env["store_dir"]), "--manifest",
         str(env["manifest_path"]), "--prereg", str(env["prereg"]), "--out", str(out), "--device", "cpu",
         "--batch-s", "6", "--max-rows", max_rows, "--max-temp", "0"]
    return a + (["--tables", str(tables)] if tables else []) + list(extra)


@pytest.fixture(scope="module")
def first_run(wenv, tiny_dir, tmp_path_factory):
    """whisper_eval of the tiny model on the four sets, with 05's tables; the heartbeat calls counted."""
    root = tmp_path_factory.mktemp("whisper_run")
    out, tables = root / "out", root / "tables"
    beats, bounded = [], []
    real_beat, real_beating = heartbeat.beat, heartbeat.beating

    def beat(path=None, *, force=False):
        beats.append((path, force))
        return real_beat(path, force=force)

    def beating(path=None, every_s=30.0, max_s=None):
        bounded.append(max_s)
        return real_beating(path, every_s, max_s)

    hb = root / "hb" / "whisper-tiny"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(heartbeat, "beat", beat)
        mp.setattr(heartbeat, "beating", beating)
        mp.setenv(heartbeat.ENV, str(hb))
        mp.setenv("KITSUNE_MACHINE_ID", "m-7")
        rc = we.main(argv_of(wenv, tiny_dir, out, tables=tables))
    return dict(rc=rc, out=out, tables=tables, beats=beats, bounded=bounded, hb=hb)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def events(out: Path, kind: str) -> list[dict]:
    return [r for r in (json.loads(x) for x in (out / "events.jsonl").read_text(encoding="utf-8").splitlines()
                        if x.strip()) if r["kind"] == kind]


def test_whisper_eval_end_to_end(wenv, first_run):
    """Every manifest set decoded: greedy_<set>.parquet in the documented columns (no teacher_hyp), the manifest's
    ids, the store's references, n_tok / stop / truncated consistent; the eval batches were cut at --max-rows (the
    split is recorded); whisper.json's sets are the parquets' numbers."""
    assert first_run["rc"] == 0
    out = first_run["out"]
    man = wenv["manifest"]
    store = wenv["store"]
    refs = dict(zip((u.id for u in store.utts), store.frame()["ref"]))
    w = load(out / "whisper.json")
    assert w["schema"] == 1 and w["system"] == "whisper-tiny" and w["family"] == "whisper" and w["status"] == "complete"
    assert w["model"]["repo"] is None and w["model"]["path"] and w["model"]["fp32_head"] is True
    assert w["model"]["n_mels"] == 128 and w["model"]["dtype"] == "fp32" and w["model"]["attn"] == "sdpa"
    assert w["decode"]["prompt_ids"] == PROMPT and w["decode"]["force_unique_generate_call"] is True
    assert w["decode"]["repetition_stop"] == dict(prompt_len=4, window=24, max_period=12)
    assert w["manifest"]["sha256"] and w["store"]["fingerprint"] == store.info["fingerprint"]
    assert w["versions"]["machine_id"] == "m-7" and "cpu" in w["versions"] and w["limit_per_set"] is None
    b = w["batching"]
    assert b["max_rows"] == 2 and b["n_split"] >= 1 and b["pooled_max_rows_seen"] >= 3
    assert b["n_batches"] == sum(v["n_batches"] for v in w["sets"].values())
    assert set(w["sets"]) == set(EVAL_SETS)
    for s in EVAL_SETS:
        g = pd.read_parquet(out / f"greedy_{s}.parquet")
        assert tuple(g.columns) == we.COLUMNS and "teacher_hyp" not in g.columns
        assert sorted(g["id"]) == sorted(man["sets"][s]["ids"]) and (g["source"] == s).all()
        assert (g["ref"] == g["id"].map(refs)).all()
        assert set(g["stop"]) <= {"eos", "repetition", "length"}
        assert (g["truncated"] == (g["stop"] != "eos")).all() and (g["n_tok"] <= g["max_new"]).all()
        assert g.groupby("batch").size().max() <= 2
        v = w["sets"][s]
        assert v["n"] == len(g) and v["dropped"] == []
        assert v["cer_corpus"] == ev.corpus_cer(g["hyp"].tolist(), g["ref"].tolist())["cer"]
        assert v["n_truncated"] == int(g["truncated"].sum()) and v["n_repetition"] + v["n_length"] == v["n_truncated"]
        assert v["audio_s"] == pytest.approx(g["duration"].sum(), abs=1e-3)
        pooled = trainset.eval_batches(store.utts, 6.0, [i for i, u in enumerate(store.utts) if u.source == s])
        assert v["pooled_max_rows"] == max(len(p) for p in pooled)
        assert v["n_batches"] == g["batch"].nunique() == len(W.split_rows(pooled, 2))
    assert events(out, "whisper_done") and events(out, "whisper_start")[0]["max_rows"] == 2


def test_the_tables_and_study_json(wenv, first_run):
    """--tables ran 05 --from-evals: study.json is family whisper with no teacher (no teacher_cer, no *_teacher /
    *_ratio) and carries the model and decode blocks; the tables load in the study report and its corpus gives the
    same per-stratum CERs."""
    out, tables = first_run["out"], first_run["tables"]
    st = load(out / "study.json")
    assert st["system"] == "whisper-tiny" and st["family"] == "whisper" and st["teacher"] is None
    assert st["refused"] == {} and st["missing_sets"] == []
    assert not any("teacher_cer" in r for r in st["strata"].values())
    assert "m4" in st["metrics"] and not any(k.endswith(("_teacher", "_ratio")) for k in st["metrics"])
    assert st["whisper"]["model"] == load(out / "whisper.json")["model"]
    assert st["whisper"]["decode"]["prompt_ids"] == PROMPT
    man = ss.parse_manifest(wenv["manifest"])
    loaded, _ = study_report.load_tables(tables, man.sets)
    corpus = ss.build_corpus(loaded, man)
    assert list(corpus.systems) == ["whisper-tiny"]
    point = ss.point_sums(corpus)
    for name in corpus.strata:
        assert st["strata"][name]["cer"] == pytest.approx(float(ss.stratum_cer(point, name)[0]))
    for s in EVAL_SETS:
        t = pd.read_parquet(tables / "whisper-tiny" / f"{s}.parquet")
        g = pd.read_parquet(out / f"greedy_{s}.parquet").set_index("id").loc[man.sets[s]]
        assert t["id"].tolist() == man.sets[s] and t["hyp"].tolist() == g["hyp"].tolist()


def test_heartbeats(first_run, wenv):
    """One beat per decoded batch (whisper_eval's own, before the batch), the download + load under a bounded
    beating (3600 s), and the item's heartbeat file exists."""
    w = load(first_run["out"] / "whisper.json")
    per_batch = [b for b in first_run["beats"] if b == (None, False)]
    assert len(per_batch) == w["batching"]["n_batches"]
    assert W.LOAD_BEAT_MAX_S == 3600 and 3600 in first_run["bounded"]
    assert first_run["hb"].is_file()


def test_a_second_run_decodes_nothing_and_other_settings_refuse(wenv, tiny_dir, first_run, monkeypatch, tmp_path):
    """The same command again: every set is done, no model is loaded, the tables are rebuilt the same. Another
    --max-rows refuses (exit 2: the out dir holds the results of other settings). --force re-decodes the named set to
    the same rows."""
    out = first_run["out"]
    before = {s: pd.read_parquet(out / f"greedy_{s}.parquet") for s in EVAL_SETS}

    def no_model(*a, **k):
        raise AssertionError("a model was loaded")

    monkeypatch.setattr(W, "load_whisper", no_model)
    assert we.main(argv_of(wenv, tiny_dir, out, tables=first_run["tables"])) == 0
    assert events(out, "todo")[-1]["sets"] == []
    assert we.main(argv_of(wenv, tiny_dir, out, max_rows="3")) == 2
    monkeypatch.undo()
    assert we.main(argv_of(wenv, tiny_dir, out, "--force", "--sets", "eval_cv8")) == 0
    assert events(out, "todo")[-1]["sets"] == ["eval_cv8"]
    pd.testing.assert_frame_equal(pd.read_parquet(out / "greedy_eval_cv8.parquet"), before["eval_cv8"])
    assert set(load(out / "whisper.json")["sets"]) == set(EVAL_SETS)  # the earlier sets stay in the record
    assert not (out / we.PARTS / we.FORCE).exists() and events(out, "force_done")


def test_a_stopped_forced_run_continues_where_it_stopped(wenv, tiny_dir, first_run, tmp_path, monkeypatch):
    """--force is resumable, as 05's: a forced run stopped after its first set (a stall kill, a new host) leaves its
    record (.parts/force.json); the same forced command again deletes nothing and decodes only the sets not done since;
    the record goes once they are all done, and a later --force starts over."""
    out = tmp_path / "out"
    shutil.copytree(first_run["out"], out)
    before = {s: pd.read_parquet(out / f"greedy_{s}.parquet") for s in EVAL_SETS}
    real, decoded = we.decode_set, []

    def stop_after_one(args, wm, store, s, *a, **k):
        if decoded:
            raise RuntimeError("stall kill")
        decoded.append(s)
        return real(args, wm, store, s, *a, **k)

    forced = ("--force", "--sets", "eval_cv8", "eval_reazon", "galgame")
    monkeypatch.setattr(we, "decode_set", stop_after_one)
    with pytest.raises(RuntimeError, match="stall kill"):
        we.main(argv_of(wenv, tiny_dir, out, *forced))
    assert decoded == ["eval_cv8"] and [we.set_done(out, s) for s in EVAL_SETS] == [True, True, False, False]
    assert load(out / we.PARTS / we.FORCE)["sets"] == list(forced[2:])
    assert load(out / "whisper.json")["status"] == "failed"

    def spy(args, wm, store, s, *a, **k):
        decoded.append(s)
        return real(args, wm, store, s, *a, **k)

    decoded.clear()
    monkeypatch.setattr(we, "decode_set", spy)
    assert we.main(argv_of(wenv, tiny_dir, out, *forced)) == 0
    assert decoded == ["eval_reazon", "galgame"] and events(out, "force_continue")
    assert not (out / we.PARTS / we.FORCE).exists() and events(out, "force_done")
    for s in EVAL_SETS:
        pd.testing.assert_frame_equal(pd.read_parquet(out / f"greedy_{s}.parquet"), before[s])
    w = load(out / "whisper.json")
    assert w["status"] == "complete" and set(w["sets"]) == set(EVAL_SETS)
    decoded.clear()
    assert we.main(argv_of(wenv, tiny_dir, out, "--force", "--sets", "eval_cv8")) == 0
    assert decoded == ["eval_cv8"] and events(out, "force")[-1]["sets"] == ["eval_cv8"]


def test_a_partial_rerun_keeps_every_table(wenv, tiny_dir, first_run, tmp_path):
    """--tables tables every manifest set done in --out, not only the sets this command named: a forced re-run of one
    set keeps the other sets' tables and study.json's M4 (05's publish_tables deletes a table it was not handed). A
    fresh --sets eval_jsut run (smoke B's large-v3 item) tables that set only."""
    out, tables = tmp_path / "out", tmp_path / "tables"
    shutil.copytree(first_run["out"], out)
    assert we.main(argv_of(wenv, tiny_dir, out, tables=tables)) == 0
    m4 = load(out / "study.json")["metrics"]["m4"]
    assert m4 is not None
    assert we.main(argv_of(wenv, tiny_dir, out, "--force", "--sets", "eval_cv8", tables=tables)) == 0
    assert events(out, "todo")[-1]["sets"] == ["eval_cv8"]
    st = load(out / "study.json")
    assert sorted(p.stem for p in (tables / "whisper-tiny").glob("*.parquet")) == sorted(EVAL_SETS)
    assert st["missing_sets"] == [] and st["refused"] == {} and st["metrics"]["m4"] == m4
    assert events(out, "study_tables")[-1]["sets"] == sorted(EVAL_SETS)
    fresh, ftables = tmp_path / "fresh", tmp_path / "ftables"
    assert we.main(argv_of(wenv, tiny_dir, fresh, "--sets", "eval_jsut", tables=ftables)) == 0
    assert [p.stem for p in (ftables / "whisper-tiny").glob("*.parquet")] == ["eval_jsut"]
    assert events(fresh, "study_tables")[-1]["sets"] == ["eval_jsut"]


def test_refusals_before_any_model_is_loaded(wenv, tiny_dir, tmp_path, monkeypatch):
    """Exit 2, no model loaded: a store whose rows are not the manifest's, a set outside the manifest, a frame store, a
    manifest that does not hash to itself; a model dir without --system and a --model that is neither a key nor a model
    dir (a typo: refused before --out's identity is written, so the corrected command can use that --out) are argument
    errors (exit 2). A --store that is not built yet is no refusal: exit 1, as the queue retries it."""
    def no_model(*a, **k):
        raise AssertionError("a model was loaded before the refusal")

    monkeypatch.setattr(W, "load_whisper", no_model)
    man = json.loads(json.dumps(wenv["manifest"]))
    ids = man["sets"]["eval_cv8"]["ids"][:-1]
    man["sets"]["eval_cv8"].update(ids=ids, n=len(ids), ids_sha256=ids_sha256(ids))
    short = tmp_path / "short.json"
    short.write_text(json.dumps(man), encoding="utf-8")
    a = argv_of(wenv, tiny_dir, tmp_path / "o1")
    a[a.index("--manifest") + 1] = str(short)
    assert we.main(a) == 2
    assert we.main(argv_of(wenv, tiny_dir, tmp_path / "o2", "--sets", "eval_emilia")) == 2
    frames = tmp_path / "frames"
    shutil.copytree(wenv["store_dir"], frames)
    info = json.loads((frames / "stores.json").read_text(encoding="utf-8"))
    (frames / "stores.json").write_text(json.dumps(dict(info, kind="frames")), encoding="utf-8")
    a = argv_of(wenv, tiny_dir, tmp_path / "o3")
    a[a.index("--store") + 1] = str(frames)
    assert we.main(a) == 2
    bad = json.loads(json.dumps(wenv["manifest"]))
    bad["sets"]["eval_jsut"]["ids"] = bad["sets"]["eval_jsut"]["ids"][::-1]
    (tmp_path / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    a = argv_of(wenv, tiny_dir, tmp_path / "o4")
    a[a.index("--manifest") + 1] = str(tmp_path / "bad.json")
    assert we.main(a) == 2
    a = argv_of(wenv, tiny_dir, tmp_path / "o6")
    a[a.index("--system"):a.index("--system") + 2] = []
    assert we.main(a) == 2
    assert we.main(argv_of(wenv, "whisper-lage-v3", tmp_path / "o7")) == 2
    assert not (tmp_path / "o7").exists()
    a = argv_of(wenv, tiny_dir, tmp_path / "o8")
    a[a.index("--store") + 1] = str(tmp_path / "not-built")
    assert we.main(a) == 1
    assert not any((tmp_path / f"o{i}" / "greedy_eval_jsut.parquet").exists() for i in range(1, 9))


def test_limit_per_set_is_the_speed_probes_draw(wenv, tiny_dir, tmp_path):
    """--limit-per-set N scores speed_probe.pick_ids's seeded ids of each set (no manifest needed), recorded in
    whisper.json."""
    import speed_probe as sp

    out = tmp_path / "lim"
    a = ["--model", str(tiny_dir), "--system", "whisper-tiny", "--store", str(wenv["store_dir"]), "--out", str(out),
         "--device", "cpu", "--max-temp", "0", "--limit-per-set", "2", "--sets", "eval_jsut", "galgame"]
    assert we.main(a) == 0
    picked = sp.pick_ids(wenv["store"], 2, 1234)
    for s in ("eval_jsut", "galgame"):
        g = pd.read_parquet(out / f"greedy_{s}.parquet")
        assert sorted(g["id"]) == sorted(i for i in picked if i.startswith(s + "/"))
    w = load(out / "whisper.json")
    assert w["limit_per_set"] == 2 and w["manifest"] is None and set(w["sets"]) == {"eval_jsut", "galgame"}
    assert w["model"]["repo"] is None and w["batching"]["max_rows"] == W.DEFAULT_MAX_ROWS


def test_limit_per_set_skips_the_box_templates_tables(wenv, tiny_dir, tmp_path, capsys):
    """Smoke B's turbo / kotoba / small items are the box's Whisper argv template (contract 7: --manifest, --tables
    {out}/tables) plus --limit-per-set: exit 0, not an argument error, so check 15's 'item done' can pass. The limited
    sets are decoded and checked against the manifest; --tables is skipped with a note (stderr and a tables_skipped
    event): no tables, no study.json, 05 never runs."""
    import speed_probe as sp

    out = tmp_path / "lim"
    tables = out / "tables"
    assert we.main(argv_of(wenv, tiny_dir, out, "--limit-per-set", "2", tables=tables)) == 0
    assert "--tables" in capsys.readouterr().err
    assert not tables.exists() and not (out / "study.json").exists()
    assert [e["tables"] for e in events(out, "tables_skipped")] == [str(tables)]
    assert events(out, "tables_skipped")[0]["limit_per_set"] == 2 and not events(out, "study_tables")
    picked = sp.pick_ids(wenv["store"], 2, 1234)
    for s in EVAL_SETS:
        g = pd.read_parquet(out / f"greedy_{s}.parquet")
        assert list(g["id"]) and sorted(g["id"]) == sorted(i for i in picked if i in wenv["manifest"]["sets"][s]["ids"])
    w = load(out / "whisper.json")
    assert w["status"] == "complete" and w["limit_per_set"] == 2 and set(w["sets"]) == set(EVAL_SETS)
    assert w["manifest"]["sha256"]
    # the same command again (a queue retry): nothing decoded, the note again, still no tables
    assert we.main(argv_of(wenv, tiny_dir, out, "--limit-per-set", "2", tables=tables)) == 0
    assert len(events(out, "tables_skipped")) == 2 and not tables.exists()


def test_fetch_only_and_the_exit_codes(tiny_dir, monkeypatch):
    """--fetch-only fetches a key's pinned snapshot under a bounded heartbeat and exits 0 without a store or out; a
    model dir has nothing to fetch (exit 2); whatever stops the run, never the queue's exit 3."""
    got, bounded = [], []

    def resolve(model, cache=None):
        got.append((model, cache))
        return tiny_dir, W.WHISPER_MODELS[model]

    monkeypatch.setattr(W, "resolve", resolve)
    real = heartbeat.beating
    monkeypatch.setattr(heartbeat, "beating", lambda path=None, every_s=30.0, max_s=None: (bounded.append(max_s),
                                                                                         real(path, every_s, max_s))[1])
    assert we.main(["--model", "whisper-large-v3-turbo", "--fetch-only", "--hf-cache", "cache/hf"]) == 0
    assert got == [("whisper-large-v3-turbo", "cache/hf")] and bounded == [3600]
    assert we.main(["--model", str(tiny_dir), "--system", "x", "--fetch-only"]) == 2

    def stop3(args):
        raise SystemExit(3)

    monkeypatch.setattr(we, "run", stop3)
    assert we.main(["--model", "whisper-small", "--fetch-only"]) == 1
