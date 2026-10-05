"""kitsune/fullrun.py: the full-data runs' shared core (contract section 2): the constants, the recipe / dev-pick /
resume-flag / path / pointer helpers, the box registry's validation (every rule refused once), the box_* readers and
the CLI's exit codes. tests/fixtures_full.tiny_registry is the valid registry every rule test breaks. CPU only."""
import copy
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import ROOT, load_script  # noqa: E402
from fixtures_full import tiny_registry  # noqa: E402

from kitsune import fullrun as fr  # noqa: E402
from kitsune import heartbeat, prereg  # noqa: E402

RID = "full-p03-20260927T120000Z"


# ------------------------------------------------------------------------------------------------ constants


def test_constants_exact_values():
    assert fr.JOB == "full"
    assert (fr.BOXES_FILE, fr.ENV_REGISTRY) == ("configs/full/boxes.json", "KITSUNE_FULL_REGISTRY")
    assert fr.BOX_NAMES == ("full-smoke", "p01", "full", "smoke-b", "full-t", "full-p")
    assert fr.HUB_DIR == "full" and fr.STATE_DEFAULT == "/workspace/kitsune_state"
    assert (fr.TRAIN_HB, fr.HB_DIR, fr.RESUME_PLAN, fr.VERDICT_FILE, fr.ALERTS_FILE, fr.GATE_FILE, fr.SUMMARY_FILE,
            fr.DEADLINE_FILE) == ("train_hb", "hb", "resume_plan.json", "smoke_verdict.json",
                                  "watchdog_alerts.jsonl", "download_gate.json", "queue_summary.json", "deadline")
    assert (fr.TIMED_POINTER, fr.SCRATCH_MARK, fr.TIMED_REASON, fr.POINTER_FORMAT) == (
        "timed_state.json", ".scratch_pending", "timed", 1)
    assert fr.STATE_FILES_REQUIRED == ("model.pt", "optimizer.pt", "l2sp.pt", "trainer.pt", "trainer.json")
    assert fr.STATE_FILES_OPTIONAL == ("aux_ctc.pt",)
    assert fr.FROZEN_MANIFEST == "labels/full/selections/study_manifest.json"
    assert fr.FROZEN_MANIFEST_SHA256 == "ef56dec2b69bfe54f96a5ad1df36bde38255b68799873ab5f12346da78ac3796"
    assert fr.FULL_DIR == "labels/full/selections/full_study"
    assert fr.FULL_SELECTION == "labels/full/selections/full_study/full.parquet"
    assert fr.SMOKE_SELECTION == "labels/full/selections/full_study/smoke.parquet"
    assert (fr.DEV_SPLIT, fr.SPLITS, fr.DEV_RULE) == ("dev", ("train", "dev", "eval"), 1)
    assert fr.FULL_STUDY == {"f1a_max": 0.5, "dedup_min_chars": 15, "probe_n": 300, "draw_audio_s": None,
                             "dev_rule": 1}
    assert list(fr.FULL_STUDY) == ["f1a_max", "dedup_min_chars", "probe_n", "draw_audio_s", "dev_rule"]
    assert fr.SMOKE_DRAW_AUDIO_S == 360000 == 100 * 3600
    assert fr.SMOKE_STUDY == dict(fr.FULL_STUDY, draw_audio_s=360000)
    assert fr.DATA_KEYS == ("data_root", "teacher_root", "second_root", "parakeet_root", "selection", "extent",
                            "sources", "eval_sets", "selection_recipe", "pull_parakeet")
    assert fr.RUN_ID_RE == r"^[a-z0-9][a-z0-9.-]*-\d{8}T\d{6}Z(?:-\d+)?$"
    assert fr.ITEM_RE == r"^[a-z0-9][a-z0-9.-]*$"
    assert fr.RESUME_SET_KEYS == ("schedule.epochs", "early_stop.patience", "augment.enabled", "augment.truncate_p",
                                  "augment.concat_p", "augment.mix_p", "augment.truncate_min_row_s",
                                  "augment.end_trim_p", "augment.noise_p", "augment.noise_bank",
                                  "augment.noise_bank_sha256")
    assert fr.RESUME_SET_INT_MIN == {"schedule.epochs": 1, "early_stop.patience": 1}
    assert fr.RESUME_SET_KINDS == {"schedule.epochs": "int", "early_stop.patience": "int", "augment.enabled": "bool",
                                   "augment.truncate_p": "prob", "augment.concat_p": "prob", "augment.mix_p": "prob",
                                   "augment.truncate_min_row_s": "seconds", "augment.end_trim_p": "prob",
                                   "augment.noise_p": "prob", "augment.noise_bank": "path",
                                   "augment.noise_bank_sha256": "sha256"}
    assert fr.STALL_MIN_DEFAULT == {"stores": 360, "train": 45, "readout": 30, "speed": 30, "eval": 60}
    assert fr.ITEM_KINDS == ("stores", "train", "readout", "speed", "eval")
    assert fr.FAULT_ACTIONS == ("sigstop", "kill", "wipe_run_dir", "deadline", "freeze_controller_hb")
    assert fr.ITEM_STATUSES == ("pending", "running", "interrupted", "retry", "done", "failed", "skipped",
                                "not_needed")
    assert fr.OVERRUN_FACTOR == 2.0
    assert fr.PLACEHOLDERS == ("python", "root", "state", "box", "cache_dir", "hf_cache", "hub_cache", "manifest",
                               "out", "run_id", "run_name", "config", "ckpt")


def test_data_blocks():
    assert fr.FULL_DATA == {
        "data_root": "data", "teacher_root": "labels/full/teacher_out", "second_root": "labels/full/second_out",
        "parakeet_root": "labels/full/parakeet_out", "selection": fr.FULL_SELECTION,
        "extent": {"name": "full", "root": "labels/full", "inputs": {}},
        "sources": ["reazon_small", "reazon_large", "emilia_yodas", "emilia_nc", "galgame"],
        "eval_sets": ["eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia", "galgame"],
        "selection_recipe": {"agree_max": 0.5, "agree_max_source": ["emilia_yodas=0.2", "emilia_nc=0.2"],
                             "filter_eval_sets": [], "partial_second_opinion": [], "study": None,
                             "full_study": fr.FULL_STUDY}}
    want = copy.deepcopy(fr.FULL_DATA)
    want["selection"] = fr.SMOKE_SELECTION
    want["extent"]["inputs"] = {"reazon_large": 53, "emilia_yodas": "300h", "emilia_nc": 8, "galgame": 3}
    want["selection_recipe"]["full_study"] = fr.SMOKE_STUDY
    assert fr.SMOKE_DATA == want
    # the full blocks share the study's roots, sources, eval sets and F0 judges; the smoke keeps the study extent
    study = json.loads((ROOT / "study" / "data.json").read_text(encoding="utf-8"))
    for k in ("data_root", "teacher_root", "second_root", "parakeet_root", "sources", "eval_sets"):
        assert fr.FULL_DATA[k] == study[k] == fr.SMOKE_DATA[k]
    rec = {k: v for k, v in prereg.registered_recipe().items() if k != "study"}
    assert {k: v for k, v in fr.FULL_DATA["selection_recipe"].items() if k not in ("study", "full_study")} == rec
    assert fr.SMOKE_DATA["extent"] == study["extent"]
    assert fr.FULL_DATA["selection_recipe"]["full_study"] is not fr.FULL_STUDY  # no aliasing of the module constant


def test_every_environment_variable_has_a_constant():
    # contract 1.4 (+ KITSUNE_STATE of 1.3): the name -> value pairs other packages import
    names = ["KITSUNE_JOB", "KITSUNE_BOX", "KITSUNE_N_GPUS", "KITSUNE_CONFIG", "KITSUNE_SHA", "KITSUNE_DATA_REPO",
             "KITSUNE_DATA_REVISION", "KITSUNE_OUT_REPO", "KITSUNE_MAX_HOURS", "KITSUNE_DPH",
             "KITSUNE_REBUILD_TIMEOUT_MIN", "KITSUNE_PULL_TIMEOUT_MIN", "KITSUNE_SCRATCH_REPO", "KITSUNE_MACHINE_ID",
             "KITSUNE_GATE_BYTES", "KITSUNE_GATE_MAX_H", "KITSUNE_REBUILD_BYTES", "KITSUNE_PULL_BYTES",
             "KITSUNE_WATCHDOG_HB_FILE", "KITSUNE_WATCHDOG_ORPHAN_S", "KITSUNE_WATCHDOG_ORPHAN_ACTION",
             "KITSUNE_RESUME", "KITSUNE_RESUME_RESET", "KITSUNE_RESUME_SETS", "KITSUNE_CPU_QUOTA",
             "KITSUNE_THREADS_PER_GPU", "KITSUNE_CGROUP", "KITSUNE_HEARTBEAT", "KITSUNE_DEADLINE",
             "KITSUNE_QUEUE_ITEM", "KITSUNE_STATE"]
    for n in names:
        assert getattr(fr, "ENV_" + n[len("KITSUNE_"):]) == n
    for n in ("TZ", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS"):
        assert getattr(fr, "ENV_" + n) == n
    assert fr.ENV_REGISTRY == "KITSUNE_FULL_REGISTRY"
    assert fr.ENV_THREAD_POOLS == ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                   "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS")
    assert fr.ENV_HEARTBEAT == heartbeat.ENV


def test_upload_mark_is_the_trainers():
    src = (ROOT / "scripts" / "04_distill.py").read_text(encoding="utf-8")
    assert re.search(r'^UPLOAD_MARK = "(.*)"$', src, re.M).group(1) == fr._UPLOAD_MARK


def test_families_and_speed_kinds_follow_the_code():
    src = (ROOT / "scripts" / "04_distill.py").read_text(encoding="utf-8")
    assert re.search(r"^FAMILIES = \(\"aed\", \"ctc\"\)", src, re.M) and fr.FAMILIES == ("aed", "ctc")
    probe = (ROOT / "tools" / "speed_probe.py").read_text(encoding="utf-8")
    kinds = eval(re.search(r"^KINDS = (\(.*\))$", probe, re.M).group(1))  # noqa: S307  a literal tuple
    assert set(kinds) <= set(fr.SPEED_KINDS) and set(fr.SPEED_KINDS) - set(kinds) <= {"whisper"}


@pytest.mark.parametrize("imp", ["import kitsune.fullrun", "sys.path.insert(0, 'tests'); import fixtures_full"])
def test_stdlib_only_at_import(imp):
    # the fixture too: a test of a stdlib-only script (launch, bootstrap's helper) can use it without the heavy stack
    code = (f"import sys; {imp}; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('numpy', 'torch', 'pandas', 'pyarrow') "
            "or m == 'kitsune.prereg'))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"


def test_state_dir_and_item_hb_path(tmp_path, monkeypatch):
    monkeypatch.delenv("KITSUNE_STATE", raising=False)
    assert fr.state_dir() == Path("/workspace/kitsune_state")
    monkeypatch.setenv("KITSUNE_STATE", str(tmp_path))
    assert fr.state_dir() == tmp_path
    assert fr.item_hb_path("m4-full-p01") == tmp_path / "hb" / "m4-full-p01"
    assert fr.item_hb_path("stores-ctc", state=tmp_path / "s") == tmp_path / "s" / "hb" / "stores-ctc"
    for bad in ("../x", "A", "", "a/b"):
        with pytest.raises(ValueError):
            fr.item_hb_path(bad)


# ------------------------------------------------------------------------------------------------ helpers


def test_shard_split():
    assert fr.shard_split("dev") == "train"
    assert fr.shard_split("train") == "train" and fr.shard_split("eval") == "eval"


def test_full_recipe_problems():
    assert fr.full_recipe_problems(fr.FULL_STUDY) == []
    assert fr.full_recipe_problems(fr.SMOKE_STUDY) == []
    assert fr.full_recipe_problems(dict(fr.FULL_STUDY, draw_audio_s=1.5)) == []
    # the values the full recipe shares with the pre-registered study recipe
    for k in ("f1a_max", "dedup_min_chars", "probe_n"):
        assert fr.FULL_STUDY[k] == prereg.STUDY_SELECTION[k]
    bad = [dict(fr.FULL_STUDY, f1a_max=0.4), dict(fr.FULL_STUDY, dedup_min_chars=16), dict(fr.FULL_STUDY, probe_n=500),
           dict(fr.FULL_STUDY, probe_n=True), dict(fr.FULL_STUDY, draw_audio_s=0),
           dict(fr.FULL_STUDY, draw_audio_s=-5), dict(fr.FULL_STUDY, draw_audio_s=math.inf),
           dict(fr.FULL_STUDY, draw_audio_s="100h"), dict(fr.FULL_STUDY, draw_audio_s=True),
           dict(fr.FULL_STUDY, dev_rule=2), dict(fr.FULL_STUDY, dev_rule=True), dict(fr.FULL_STUDY, dev_rule=None),
           {k: v for k, v in fr.FULL_STUDY.items() if k != "dev_rule"},
           dict(fr.FULL_STUDY, neutral_max_cer=0.5),  # the study block's key: a study block is not a full one
           dict(prereg.STUDY_SELECTION), None, [], "full"]
    for b in bad:
        assert fr.full_recipe_problems(b), b


def test_seeded_subset_equals_make_selection():
    ms = load_script("make_selection")
    rng = np.random.default_rng(7)
    for trial in range(25):
        ids = [f"src/{int(x):07d}" for x in rng.choice(10**6, size=int(rng.integers(0, 400)), replace=False)]
        n, seed, tag = int(rng.integers(0, 500)), int(rng.integers(0, 2**31)), f"tag:{trial}"
        assert fr.seeded_subset(ids, n, seed, tag) == ms.seeded_subset(ids, n, seed, tag)
        assert fr.seeded_subset(list(reversed(ids)), n, seed, tag) == ms.seeded_subset(ids, n, seed, tag)


def _dev_rows():
    rng = np.random.default_rng(3)
    rows = [(f"{s}/{i:05d}", s) for s in ("reazon_small", "galgame", "emilia_nc") for i in range(40)]
    order = rng.permutation(len(rows))
    return [rows[i] for i in order]


def test_dev_pick_tag_order_and_sizes():
    rows = _dev_rows()
    got = fr.dev_pick(rows, ["reazon_small", "galgame"], 10, 1234)
    want = fr.seeded_subset([i for i, s in rows if s == "reazon_small"], 10, 1234, "dev_scored:reazon_small") | \
        fr.seeded_subset([i for i, s in rows if s == "galgame"], 10, 1234, "dev_scored:galgame")
    assert set(got) == want and len(got) == 20
    assert got == [i for i, _ in rows if i in want]  # selection order, not sorted or grouped
    assert not any(i.startswith("emilia_nc/") for i in got)  # a source not asked for is not picked
    assert fr.dev_pick(iter(rows), ["galgame"], 1000, 1) == [i for i, s in rows if s == "galgame"]  # all when fewer
    assert fr.dev_pick(rows, ["galgame"], 5, 1) != fr.dev_pick(rows, ["galgame"], 5, 2)  # the seed matters
    assert fr.dev_pick(rows, ["galgame"], 5, 1) == fr.dev_pick(list(reversed(rows)), ["galgame"], 5, 1)[::-1]


def test_dev_pick_refuses_a_source_without_rows():
    with pytest.raises(ValueError, match="eval_jsut"):
        fr.dev_pick(_dev_rows(), ["galgame", "eval_jsut"], 10, 1)
    with pytest.raises(ValueError):
        fr.dev_pick(_dev_rows(), ["galgame"], 0, 1)


def test_parse_resume_sets():
    assert fr.parse_resume_sets(None) == {} and fr.parse_resume_sets("") == {}
    assert fr.parse_resume_sets(f"{RID}:schedule.epochs=4") == {RID: ["schedule.epochs=4"]}
    two = "full-t06-20260927T120000Z-2:schedule.epochs=04,full-p005-20260928T000000Z:schedule.epochs=5"
    assert fr.parse_resume_sets(two) == {"full-t06-20260927T120000Z-2": ["schedule.epochs=4"],
                                         "full-p005-20260928T000000Z": ["schedule.epochs=5"]}
    # DECISIONS G3: P-0.1B's continuation sets its epochs and its early-stop patience, in the order given
    assert fr.parse_resume_sets(f"{RID}:early_stop.patience=12") == {RID: ["early_stop.patience=12"]}
    assert fr.parse_resume_sets(f"{RID}:early_stop.patience=012") == {RID: ["early_stop.patience=12"]}
    both = f"{RID}:schedule.epochs=8,{RID}:early_stop.patience=12"
    assert fr.parse_resume_sets(both) == {RID: ["schedule.epochs=8", "early_stop.patience=12"]}
    assert fr.parse_resume_sets(f"{RID}:early_stop.patience=12,{RID}:schedule.epochs=8") == {
        RID: ["early_stop.patience=12", "schedule.epochs=8"]}
    only = "only schedule.epochs, early_stop.patience, augment.enabled, augment.truncate_p, augment.concat_p, " \
           "augment.mix_p, augment.truncate_min_row_s, augment.end_trim_p, augment.noise_p, augment.noise_bank, " \
           "augment.noise_bank_sha256 may change on a resume"
    for bad in (f"{RID}:early_stop.patience=0", f"{RID}:early_stop.patience=-1", f"{RID}:early_stop.patience=1.5",
                f"{RID}:early_stop.patience=x", f"{RID}:early_stop.patience=",
                f"{RID}:early_stop.patience=12,{RID}:early_stop.patience=6",
                f"{RID}:early_stop.enabled=false", f"{RID}:early_stop.min_delta_rel=0.01"):
        with pytest.raises(ValueError, match=f"early_stop|{only}"):
            fr.parse_resume_sets(bad)
    with pytest.raises(ValueError, match=only):
        fr.parse_resume_sets(f"{RID}:optim.lr=0.001")
    with pytest.raises(ValueError, match="early_stop.patience must be an int >= 1, not '0'"):
        fr.parse_resume_sets(f"{RID}:early_stop.patience=0")
    for bad in (f"{RID}:optim.lr=0.001",  # outside the whitelist
                f"{RID}:schedule.epochs=0", f"{RID}:schedule.epochs=x", f"{RID}:schedule.epochs=1.5",
                f"{RID}:schedule.epochs=-2", f"{RID}:schedule.epochs=", f"{RID}:schedule.epochs",
                "full-p03:schedule.epochs=4",  # not a run id
                "runs/full-p03-20260927T120000Z:schedule.epochs=4", f"{RID}", f"{RID.upper()}:schedule.epochs=4",
                f"{RID}:schedule.epochs=4,{RID}:schedule.epochs=5", f"{RID}:schedule.epochs=4,"):
        with pytest.raises(ValueError):
            fr.parse_resume_sets(bad)


RECIPE_SETS = ["schedule.epochs=4", "augment.enabled=true", "augment.truncate_p=0.3", "augment.concat_p=0.5",
               "augment.mix_p=0.05"]  # DECISIONS H1, H4: the recipe test box's sets (make_full_configs CONTINUATIONS)


def test_parse_resume_sets_types_and_normalises_each_value():
    """The typed sets (RESUME_SET_KINDS): an int, a bool, a probability, seconds, a data-repo path and a sha256, each
    normalised to the one spelling the env word, the resume plan and the trainer's --set carry - so the env word
    round-trips to itself, and two spellings of one continuation are one continuation on the box (full_queue adopt's
    resume_sets_differ compares them)."""
    loose = [f"{RID}:schedule.epochs=04", f"{RID}:augment.enabled=True", f"{RID}:augment.truncate_p=.30",
             f"{RID}:augment.concat_p=5e-1", f"{RID}:augment.mix_p=.050"]
    got = fr.parse_resume_sets(",".join(loose))
    assert got == {RID: RECIPE_SETS}
    word = ",".join(f"{rid}:{kv}" for rid, kvs in got.items() for kv in kvs)  # vast/launch.py's env word
    assert re.fullmatch(fr._ENV_WORD, word) and fr.parse_resume_sets(word) == got
    for key, val, want in (("augment.enabled", "FALSE", "false"), ("augment.enabled", "true", "true"),
                           ("augment.mix_p", "0", "0.0"), ("augment.mix_p", "1", "1.0"), ("augment.mix_p", "1.", "1.0"),
                           ("augment.concat_p", "0.50", "0.5"), ("augment.truncate_p", "3E-1", "0.3"),
                           ("augment.truncate_p", "1e-7", "1e-07"), ("schedule.epochs", "008", "8"),
                           ("early_stop.patience", "12", "12"), ("augment.truncate_min_row_s", "3", "3.0"),
                           ("augment.truncate_min_row_s", "2.50", "2.5"), ("augment.truncate_min_row_s", "0", "0.0"),
                           ("augment.end_trim_p", ".3", "0.3"), ("augment.noise_p", "0.30", "0.3")):
        assert fr.resume_set_value(key, val) == want, (key, val)
        assert fr.resume_set_value(key, want) == want  # normalised once is normalised
        # ... and JSON of the key's type, as 04_distill's --set reads it (apply_set: json.loads)
        assert type(json.loads(want)) is {"int": int, "bool": bool, "prob": float,
                                          "seconds": float}[fr.RESUME_SET_KINDS[key]]
    # a path and a sha256 stay strings: 04_distill's --set keeps a value json.loads cannot read (or reads as a string)
    sha = "0123456789abcdef" * 4
    for key, val, want in (("augment.noise_bank", "aug/musan-bg-v1", "aug/musan-bg-v1"),
                           ("augment.noise_bank", "aug/v1.2_b", "aug/v1.2_b"),
                           ("augment.noise_bank_sha256", sha.upper(), sha), ("augment.noise_bank_sha256", sha, sha)):
        assert fr.resume_set_value(key, val) == want, (key, val)
        with pytest.raises(ValueError):
            json.loads(want)
    # the rule of every key, as launch's help says it and each refusal names it
    assert {k: fr.resume_set_rule(k) for k in fr.RESUME_SET_KEYS} == {
        "schedule.epochs": "an int >= 1", "early_stop.patience": "an int >= 1", "augment.enabled": "true or false",
        "augment.truncate_p": "a probability in [0, 1]", "augment.concat_p": "a probability in [0, 1]",
        "augment.mix_p": "a probability in [0, 1]", "augment.truncate_min_row_s": "a number of seconds >= 0",
        "augment.end_trim_p": "a probability in [0, 1]", "augment.noise_p": "a probability in [0, 1]",
        "augment.noise_bank": "a relative data-repo path (letters, digits, _ . - and /, no ..)",
        "augment.noise_bank_sha256": "a sha256 (64 lowercase hex digits)"}
    for bad, msg in ((f"{RID}:augment.enabled=yes", "augment.enabled must be true or false, not 'yes'"),
                     (f"{RID}:augment.enabled=1", "augment.enabled must be true or false"),
                     (f"{RID}:augment.enabled=", "augment.enabled must be true or false"),
                     (f"{RID}:augment.truncate_p=1.5", "augment.truncate_p must be a probability in \\[0, 1\\]"),
                     (f"{RID}:augment.concat_p=-0.1", "augment.concat_p must be a probability"),
                     (f"{RID}:augment.mix_p=+0.2", "augment.mix_p must be a probability"),
                     (f"{RID}:augment.mix_p=nan", "augment.mix_p must be a probability"),
                     (f"{RID}:augment.mix_p=inf", "augment.mix_p must be a probability"),
                     (f"{RID}:augment.mix_p=1_0", "augment.mix_p must be a probability"),
                     (f"{RID}:augment.mix_p=0.2.1", "augment.mix_p must be a probability"),
                     (f"{RID}:augment.mix_p=x", "augment.mix_p must be a probability"),
                     (f"{RID}:augment.mix_p=", "augment.mix_p must be a probability"),
                     (f"{RID}:augment.mix_p=0.2,{RID}:augment.mix_p=0.3", "augment.mix_p given twice"),
                     (f"{RID}:augment.truncate_min_row_s=-1", "truncate_min_row_s must be a number of seconds >= 0"),
                     (f"{RID}:augment.truncate_min_row_s=inf", "truncate_min_row_s must be a number of seconds"),
                     (f"{RID}:augment.end_trim_p=1.5", "augment.end_trim_p must be a probability"),
                     (f"{RID}:augment.noise_p=-0.3", "augment.noise_p must be a probability"),
                     (f"{RID}:augment.noise_bank=/abs/bank", "augment.noise_bank must be a relative data-repo path"),
                     (f"{RID}:augment.noise_bank=aug/../labels", "augment.noise_bank must be a relative"),
                     (f"{RID}:augment.noise_bank=aug//bank", "augment.noise_bank must be a relative"),
                     (f"{RID}:augment.noise_bank=aug/bank/", "augment.noise_bank must be a relative"),
                     (f"{RID}:augment.noise_bank=aug\\bank", "augment.noise_bank must be a relative"),
                     (f"{RID}:augment.noise_bank=12", "augment.noise_bank must be a relative"),  # JSON: a number
                     (f"{RID}:augment.noise_bank=true", "augment.noise_bank must be a relative"),  # JSON: a bool
                     (f"{RID}:augment.noise_bank=null", "augment.noise_bank must be a relative"),
                     (f"{RID}:augment.noise_bank=", "augment.noise_bank must be a relative"),
                     (f"{RID}:augment.noise_bank_sha256=abc", "augment.noise_bank_sha256 must be a sha256"),
                     (f"{RID}:augment.noise_bank_sha256={'g' * 64}", "augment.noise_bank_sha256 must be a sha256"),
                     (f"{RID}:augment.noise_bank_sha256={'1' * 64}", "augment.noise_bank_sha256 must be a sha256"),
                     (f"{RID}:augment.noise_bank_sha256={'0' * 65}", "augment.noise_bank_sha256 must be a sha256"),
                     # the recipe's other keys keep the trainer's defaults: another value would be another recipe
                     (f"{RID}:augment.seed=1", "may change on a resume"),
                     (f"{RID}:augment.concat_max_s=20", "may change on a resume"),
                     (f"{RID}:augment.mix_snr_db=[0,5]", "may change on a resume")):
        with pytest.raises(ValueError, match=msg):
            fr.parse_resume_sets(bad)
    # one run's int and recipe sets together, another run's apart, in the order given
    two = f"{RID}:augment.mix_p=0.2,full-p01-20261001T184145Z:augment.enabled=false,{RID}:schedule.epochs=4"
    assert fr.parse_resume_sets(two) == {RID: ["augment.mix_p=0.2", "schedule.epochs=4"],
                                         "full-p01-20261001T184145Z": ["augment.enabled=false"]}


def test_parse_resume_reset():
    assert fr.parse_resume_reset(None) == [] and fr.parse_resume_reset("") == []
    assert fr.parse_resume_reset(RID) == [RID]
    assert fr.parse_resume_reset(f"{RID},full-t06-20260927T120001Z,{RID}") == [RID, "full-t06-20260927T120001Z"]
    for bad in ("full-p03", f"{RID},", f"{RID},,x", "runs/" + RID, f"{RID}:schedule.epochs=4"):
        with pytest.raises(ValueError):
            fr.parse_resume_reset(bad)


def test_run_id_of():
    for rd in (f"runs/{RID}", f"runs/{RID}/", f"runs\\{RID}", RID, f"/workspace/kitsune/runs/{RID}"):
        assert fr.run_id_of(rd) == RID


def test_hub_and_scratch_paths():
    assert fr.box_summary_path("p01") == "full/box-p01/queue_summary.json"
    assert fr.box_verdict_path("full-smoke") == "full/box-full-smoke/smoke_verdict.json"
    assert fr.box_infra_dir("full", "12345") == "full/box-full/infra/12345"
    assert fr.scratch_state_dir(RID, 5200) == f"runs/{RID}/checkpoints/full_step_5200"
    assert fr.scratch_pointer(RID) == f"runs/{RID}/timed_state.json"


def _pointer(**over) -> dict:
    files = {f: {"size": 10 + i, "sha256": f"{i:x}" * 64} for i, f in enumerate(fr.STATE_FILES_REQUIRED)}
    p = {"format": 1, "run_id": RID, "name": "full_step_5200", "step": 5200, "epoch": 0.61, "wall": 7200.5,
         "time_utc": "2026-09-27T14:00:00Z", "kitsune_sha": "0" * 40, "planner_fingerprint": "ab12",
         "n_train_utts": 123456, "selection_sha256": "f" * 64, "micro_audio_s": 600.0, "files": files,
         "host": {"hostname": "c.1", "machine_id": "151760", "container_id": "C.2"}}
    p.update(over)
    return p


def test_pointer_problems_accepts_a_good_pointer():
    assert fr.pointer_problems(_pointer()) == []
    p = _pointer(kitsune_sha=None, selection_sha256=None, epoch=1, host={"hostname": "h", "machine_id": None,
                                                                          "container_id": None})
    p["files"]["aux_ctc.pt"] = {"size": 0, "sha256": "a" * 64}
    assert fr.pointer_problems(p) == []
    assert fr.pointer_problems(json.loads(json.dumps(_pointer()))) == []  # survives the JSON round trip


@pytest.mark.parametrize("change, match", [
    (lambda p: p["files"].update({".scratch_pending": {"size": 0, "sha256": "0" * 64}}), "marker"),
    (lambda p: p["files"].update({".upload_pending": {"size": 0, "sha256": "0" * 64}}), "marker"),
    (lambda p: p["files"].pop("trainer.json"), "lacks"),
    (lambda p: p["files"].pop("model.pt"), "lacks"),
    (lambda p: p["files"].update({"events.jsonl": {"size": 1, "sha256": "0" * 64}}), "not state files"),
    (lambda p: p["files"]["model.pt"].update(sha256="xyz"), "64 hex"),
    (lambda p: p["files"]["model.pt"].update(sha256="A" * 64), "64 hex"),
    (lambda p: p["files"]["model.pt"].update(size=-1), "size"),
    (lambda p: p["files"]["model.pt"].update(size=1.5), "size"),
    (lambda p: p.update(name="full_step_5100"), "full_step_5200"),
    (lambda p: p.update(format=2), "format"),
    (lambda p: p.update(step="5200"), "step"),
    (lambda p: p.update(step=True), "step"),
    (lambda p: p.pop("planner_fingerprint"), "no planner_fingerprint"),
    (lambda p: p.pop("host"), "no host"),
    (lambda p: p.update(micro_audio_s=None), "micro_audio_s"),
    (lambda p: p.update(kitsune_sha=5), "kitsune_sha"),
    (lambda p: p["host"].pop("machine_id"), "machine_id"),
    (lambda p: p.update(files=[]), "files"),
])
def test_pointer_problems_refuses(change, match):
    p = _pointer()
    change(p)
    probs = fr.pointer_problems(p)
    assert probs and any(match in x for x in probs), probs


def test_pointer_problems_non_dict():
    assert fr.pointer_problems(None) and fr.pointer_problems([])


# ------------------------------------------------------------------------------------------------ the registry


@pytest.fixture
def reg(tmp_path):
    return tiny_registry(tmp_path)


def _problems(reg, root, **kw):
    return fr.registry_problems(reg, root=root, **kw)


def test_tiny_registry_validates(tmp_path, reg):
    assert fr.registry_problems(reg, root=tmp_path) == []
    assert (tmp_path / "configs" / "full" / "data-p01.json").is_file()
    loaded = fr.load_registry(reg, root=tmp_path)
    assert set(loaded["boxes"]) == {"full-smoke", "p01", "full", "smoke-b"} < set(fr.BOX_NAMES)  # the four of section 7
    assert fr.load_registry(loaded, root=tmp_path) == loaded  # filling the defaults again changes nothing
    # every value a box env or command line carries is one env-string word (launch.env_string's rule)
    for b in loaded["boxes"].values():
        for v in (b["data_config"], *b["extra_files"], *b["extra_dirs"], *(i["name"] for i in b["items"])):
            assert re.fullmatch(r"[A-Za-z0-9_./:@+,=-]+", v)


def test_load_registry_does_not_change_its_argument(tmp_path, reg):
    before = copy.deepcopy(reg)
    fr.load_registry(reg, root=tmp_path)
    assert reg == before


def test_defaults_are_filled(tmp_path, reg):
    loaded = fr.load_registry(reg, root=tmp_path)
    full = loaded["boxes"]["full"]
    assert (full["gate"], full["smoke"], full["max_attempts"], full["extra_dirs"], full["faults"]) == (
        True, False, 4, [], [])
    items = {i["name"]: i for i in full["items"]}
    assert items["stores-ctc"]["stall_min"] == 360 and items["full-t06"]["stall_min"] == 45
    assert items["m4-full-t06"]["stall_min"] == 30 and items["whisper-small"]["stall_min"] == 60
    assert items["speed-full-t06"]["stall_min"] == 30
    assert items["full-t06"]["droppable"] is False and items["full-p005"]["droppable"] is True
    assert all(items[n]["droppable"] is True for n in ("stores-ctc", "m4-full-t06", "whisper-small"))
    assert items["stores-ctc"]["max_hours"] is None and items["stores-ctc"]["verdict"] == []
    assert (items["stores-aed"]["eval_only"], items["stores-aed"]["sets"]) == (False, [])
    assert (items["full-t06"]["plan_total_steps"], items["full-t06"]["plan_hours"]) == (None, None)
    assert (items["speed-full-t06"]["args"], items["speed-full-t06"]["only_if_new_machine"]) == ([], None)
    assert items["whisper-small"]["weights"] == [] and items["whisper-small"]["of"] is None
    # implicit needs: a readout's train item, a same-box `of`, never an `of_box`
    assert items["m4-full-t06"]["needs"] == ["full-t06"]
    assert items["quant-int8-w8a8-full-p03"]["needs"] == ["m4-full-p03", "full-p03"]
    assert items["speed-full-t06"]["needs"] == ["m4-full-t06", "full-t06"]
    assert items["quant-int8-w8a8-full-p01"]["needs"] == []
    smoke = loaded["boxes"]["full-smoke"]
    si = {i["name"]: i for i in smoke["items"]}
    assert si["speed-cohere"]["stall_min"] is None  # an explicit null stays null: no stall check
    assert si["smoke-t06"]["stall_min"] == 10
    f = {x["id"]: x for x in smoke["faults"]}
    assert f["F1"] == {"id": "F1", "action": "sigstop", "item": "smoke-p01", "at_step": 150, "after_event": None,
                       "min_attempt": 1, "seconds": None}
    assert f["F3"]["min_attempt"] == 2
    sb = {i["name"]: i for i in loaded["boxes"]["smoke-b"]["items"]}
    assert sb["cmp-int8-w8a8-study-p03"]["needs"] == ["quant-int8-w8a8-study-p03", "mem-int8-w8a8-study-p03"]
    assert sb["whisper-large-v3"]["verdict"] == reg["boxes"]["smoke-b"]["items"][5]["verdict"]  # specs unfilled


def _box(reg, name):
    return reg["boxes"][name]


def _item(reg, box, name):
    return next(i for i in reg["boxes"][box]["items"] if i["name"] == name)


def _refused(reg, root, match, **kw):
    probs = _problems(reg, root, **kw)
    assert probs and any(re.search(match, p) for p in probs), probs
    with pytest.raises(fr.RegistryError) as e:
        fr.load_registry(reg, root=root, **kw)
    assert e.value.problems == probs
    return probs


RULES = {
    "unknown kind": (lambda r: _item(r, "p01", "stores-ctc").update(kind="calib"), r"kind 'calib'"),
    "unknown placeholder": (lambda r: _item(r, "full", "whisper-small")["argv"].append("{outdir}"),
                            r"unknown placeholder \{outdir\}"),
    "unknown colon placeholder": (lambda r: _item(r, "full", "whisper-small")["argv"].append("{root:x}"),
                                  r"unknown placeholder \{root:x\}"),
    "unmatched brace": (lambda r: _item(r, "full", "whisper-small")["argv"].append("{out/x"), r"unmatched brace"),
    "{config} without of": (lambda r: _item(r, "full", "whisper-small")["argv"].append("{config}"), r"needs the item"),
    "{ckpt:<name>} not in weights": (lambda r: _item(r, "smoke-b", "selftest")["argv"].append("{ckpt:study-p01}"),
                                     r"names no entry of the item's weights"),
    "{out:<item>} of a later item": (lambda r: _item(r, "smoke-b", "selftest")["argv"].append("{out:whisper-large-v3}"),
                                     r"names no earlier item"),
    "needs a later item": (lambda r: _item(r, "p01", "stores-ctc").update(needs=["full-p01"]), r"not an earlier item"),
    "needs itself": (lambda r: _item(r, "p01", "full-p01").update(needs=["full-p01"]), r"not an earlier item"),
    "needs an unknown item": (lambda r: _item(r, "p01", "full-p01").update(needs=["stores-aed"]),
                              r"not an earlier item"),
    "faults on a non-smoke box": (lambda r: _box(r, "p01").update(
        faults=[{"id": "F1", "action": "kill", "item": "full-p01", "at_step": 10}]), r"not a smoke box"),
    "freeze fault on a stop box": (lambda r: _box(r, "full-smoke").update(watchdog={"orphan_s": 600,
                                                                                     "action": "stop"}),
                                   r"freeze_controller_hb on a box whose watchdog action is 'stop'"),
    "freeze shorter than orphan_s": (lambda r: _box(r, "full-smoke")["faults"][4].update(seconds=600),
                                     r"must exceed watchdog.orphan_s"),
    # box 53693389: the alert comes at the watchdog's first poll past orphan_s, which a 650 s window can miss
    "freeze inside one watchdog poll": (lambda r: _box(r, "full-smoke")["faults"][4].update(seconds=650),
                                        r"must be >= watchdog.orphan_s 600 \+ the watchdog's poll 60 s"),
    "kill without at_step": (lambda r: _box(r, "full-smoke")["faults"][1].pop("at_step"), r"kill needs at_step"),
    "wipe without after_event": (lambda r: _box(r, "full-smoke")["faults"][2].pop("after_event"),
                                 r"wipe_run_dir needs after_event"),
    "deadline without seconds": (lambda r: _box(r, "full-smoke")["faults"][3].pop("seconds"),
                                 r"deadline needs seconds"),
    "fault on a non-train item": (lambda r: _box(r, "full-smoke")["faults"][1].update(item="m4-smoke-p03"),
                                  r"not a train item"),
    "unknown fault action": (lambda r: _box(r, "full-smoke")["faults"][1].update(action="reboot"), r"action 'reboot'"),
    "duplicate fault id": (lambda r: _box(r, "full-smoke")["faults"][1].update(id="F1"), r"given twice"),
    "bad family": (lambda r: _item(r, "p01", "full-p01").update(family="rnnt"), r"family 'rnnt'"),
    "non-droppable readout": (lambda r: _item(r, "p01", "m4-full-p01").update(droppable=False), r"always droppable"),
    "non-droppable eval": (lambda r: _item(r, "full", "whisper-small").update(droppable=False), r"always droppable"),
    "data_config outside configs/full/": (lambda r: _box(r, "p01").update(data_config="configs/study/data.json"),
                                          r"not under configs/full/"),
    "data_config absolute": (lambda r: _box(r, "p01").update(data_config="/configs/full/data-p01.json"),
                             r"not a repo-relative"),
    "data_config with ..": (lambda r: _box(r, "p01").update(data_config="configs/full/../full/data-p01.json"),
                            r"not a repo-relative"),
    "null stall_min without max_hours": (lambda r: _item(r, "full-smoke", "speed-cohere").pop("max_hours"),
                                         r"stall_min null .* needs max_hours"),
    "train without max_hours": (lambda r: _item(r, "p01", "full-p01").pop("max_hours"), r"train item needs max_hours"),
    "stall_min 0": (lambda r: _item(r, "p01", "full-p01").update(stall_min=0), r"stall_min 0"),
    "unknown box": (lambda r: r["boxes"].update(p02=copy.deepcopy(r["boxes"]["p01"])), r"not in BOX_NAMES"),
    "version": (lambda r: r.update(version=2), r"version 2"),
    "unknown top-level key": (lambda r: r.update(box={}), r"unknown key"),
    "unknown box key": (lambda r: _box(r, "p01").update(max_hour=3), r"unknown key\(s\) \['max_hour'\]"),
    "unknown item key": (lambda r: _item(r, "p01", "full-p01").update(stall_mins=3), r"unknown key"),
    "a readout with argv": (lambda r: _item(r, "p01", "m4-full-p01").update(argv=["x"]), r"unknown key"),
    "missing gpus": (lambda r: _box(r, "p01").pop("gpus"), r"no gpus"),
    "gpus 0": (lambda r: _box(r, "p01").update(gpus=0), r"gpus 0"),
    "max_hours under est_hours": (lambda r: _box(r, "p01").update(max_hours=10), r"max_hours 10 < est_hours"),
    "bad watchdog action": (lambda r: _box(r, "p01").update(watchdog={"orphan_s": 3600, "action": "kill"}),
                            r"watchdog.action"),
    "watchdog without orphan_s": (lambda r: _box(r, "p01").update(watchdog={"action": "stop"}), r"watchdog"),
    "gate not a bool": (lambda r: _box(r, "p01").update(gate="yes"), r"gate 'yes'"),
    "max_attempts 0": (lambda r: _box(r, "p01").update(max_attempts=0), r"max_attempts 0"),
    "min_ram_gb 0": (lambda r: _box(r, "p01").update(min_ram_gb=0), r"min_ram_gb 0 is not null or a number > 0"),
    "min_ram_gb -1": (lambda r: _box(r, "p01").update(min_ram_gb=-1), r"min_ram_gb -1 is not null"),
    "min_ram_gb x": (lambda r: _box(r, "p01").update(min_ram_gb="x"), r"min_ram_gb 'x' is not null"),
    "no items": (lambda r: _box(r, "p01").update(items=[]), r"items"),
    "duplicate item name": (lambda r: _box(r, "p01")["items"].append(dict(_item(r, "p01", "m4-full-p01"))),
                            r"given twice"),
    "bad item name": (lambda r: _item(r, "p01", "m4-full-p01").update(name="M4 p01"), r"does not match"),
    "readout of a stores item": (lambda r: _item(r, "p01", "m4-full-p01").update(of="stores-ctc"),
                                 r"not an earlier train item"),
    "study_run not registered": (lambda r: _item(r, "p01", "full-p01").update(study_run="study-p02"),
                                 r"not a kitsune.prereg run"),
    "plan steps without hours": (lambda r: _item(r, "full-smoke", "smoke-t06").pop("plan_hours"),
                                 r"go together"),
    "of_box of a missing train item": (lambda r: _item(r, "full", "quant-int8-w8a8-full-p01").update(of="full-p03"),
                                       r"not a train item of box p01"),
    "of_box of the same box": (lambda r: _item(r, "full", "quant-int8-w8a8-full-p01").update(of_box="full"),
                               r"not another registry box"),
    "of_box not in the registry": (lambda r: r["boxes"].pop("p01"), r"of_box 'p01' is not a registry box"),
    "of_box without of": (lambda r: _item(r, "full", "quant-int8-w8a8-full-p01").pop("of"), r"without of"),
    "same-box of of a later item": (lambda r: _box(r, "full")["items"].insert(2, {
        "name": "early-eval", "kind": "eval", "of": "full-t06", "argv": ["x"], "max_hours": 0.3}),
        r"early-eval: of 'full-t06' is not an earlier train item"),
    "model not in extra_dirs": (lambda r: _item(r, "full-smoke", "speed-parakeet-tdt").update(model="models/x"),
                                r"not in the box's extra_dirs"),
    "bad weights run id": (lambda r: _item(r, "smoke-b", "selftest")["weights"][0].update(run_id="study-p03"),
                           r"not a run id"),
    "duplicate weights name": (lambda r: _item(r, "smoke-b", "selftest")["weights"][1].update(name="study-p03"),
                               r"given twice"),
    "speed with two sources": (lambda r: _item(r, "full", "speed-study-t06").update(of="full-t06"),
                               r"times one model"),
    "cohere with a model": (lambda r: _item(r, "full-smoke", "speed-cohere").update(
        model="models/parakeet-tdt_ctc-0.6b-ja-hf"), r"cohere"),
    # speed_probe refuses these kinds without --model at once: a dropped source must fail here, not on the rented box
    "aed speed item without a model source": (lambda r: _item(r, "full", "speed-study-t06").pop("weights"),
                                              r"speed_kind aed times one student"),
    "ctc speed item without a model source": (lambda r: _item(r, "full-smoke", "speed-study-p01").update(weights=[]),
                                              r"speed_kind ctc times one student"),
    "speed item with two weights entries": (lambda r: _item(r, "full-smoke", "speed-study-p01")["weights"].append(
        {"name": "study-p03", "run_id": "study-p03-20260926T172336Z", "step": 25120}), r"times one model .* got 2"),
    "parakeet speed item without model": (lambda r: _item(r, "full-smoke", "speed-parakeet-tdt").pop("model"),
                                          r"speed_kind parakeet-tdt times the Parakeet teacher: give model"),
    "parakeet speed item on student weights": (lambda r: _item(r, "full-smoke", "speed-parakeet-tdt").update(
        model=None, weights=_item(r, "full-smoke", "speed-study-p01")["weights"]),
        r"speed_kind parakeet-tdt times the Parakeet teacher"),
    "whisper speed item on student weights": (lambda r: _item(r, "full-smoke", "speed-study-p01").update(
        speed_kind="whisper"), r"speed_kind whisper times a Whisper model"),
    # no model source and no --model <key> in args: the queue's argv has no --model and speed_probe exits 2 on the box
    "whisper speed item without a model": (lambda r: _item(r, "full-smoke", "speed-cohere").update(
        speed_kind="whisper", system="whisper-small"), r"speed_kind whisper needs --model <key> in args \(or model\)"),
    "whisper speed item with only --hf-cache": (lambda r: _item(r, "full-smoke", "speed-cohere").update(
        speed_kind="whisper", system="whisper-small", args=["--hf-cache", "{hf_cache}"]),
        r"speed_kind whisper needs --model"),
    "whisper speed item with --model last": (lambda r: _item(r, "full-smoke", "speed-cohere").update(
        speed_kind="whisper", system="whisper-small", args=["--hf-cache", "{hf_cache}", "--model"]),
        r"speed_kind whisper needs --model"),
    "whisper speed item with --model before an option": (lambda r: _item(r, "full-smoke", "speed-cohere").update(
        speed_kind="whisper", system="whisper-small", args=["--model", "--hf-cache", "{hf_cache}"]),
        r"speed_kind whisper needs --model"),
    "whisper speed item with an empty --model=": (lambda r: _item(r, "full-smoke", "speed-cohere").update(
        speed_kind="whisper", system="whisper-small", args=["--model="]), r"speed_kind whisper needs --model"),
    "unknown speed kind": (lambda r: _item(r, "full", "speed-study-t06").update(speed_kind="tdt"),
                           r"speed_kind 'tdt'"),
    "only_if_new_machine unknown": (lambda r: _item(r, "smoke-b", "speed-study-p03").update(
        only_if_new_machine="box-a"), r"only_if_new_machine"),
    "only_if_new_machine itself": (lambda r: _item(r, "smoke-b", "speed-study-p03").update(
        only_if_new_machine="smoke-b"), r"only_if_new_machine"),
    "verdict on a non-smoke box": (lambda r: _item(r, "full", "whisper-small").update(verdict=[{"check": "15"}]),
                                   r"not a smoke box"),
    "verdict check 17": (lambda r: _item(r, "smoke-b", "speed-study-p03").update(verdict=[{"check": "17"}]),
                         r"check '17'"),
    "verdict check as an int": (lambda r: _item(r, "smoke-b", "speed-study-p03").update(verdict=[{"check": 16}]),
                                r"check 16"),
    "verdict condition without path": (lambda r: _item(r, "smoke-b", "speed-study-p03").update(
        verdict=[{"check": "16", "equals": True}]), r"condition needs"),
    "verdict json without path": (lambda r: _item(r, "smoke-b", "speed-study-p03").update(
        verdict=[{"check": "16", "json": "{out}/speed.json"}]), r"go together"),
    "verdict unknown placeholder": (lambda r: _item(r, "smoke-b", "whisper-large-v3")["verdict"][0].update(
        json="{outdir}/whisper.json"), r"unknown placeholder"),
    "stores set not key=value": (lambda r: _item(r, "smoke-b", "stores-eval")["sets"].append("optim.lr"),
                                 r"not key=value"),
    "a path with a space": (lambda r: _box(r, "p01")["extra_files"].append("labels/full/a b.json"),
                            r"env-string word"),
    "eval without argv": (lambda r: _item(r, "full", "whisper-small").pop("argv"), r"no argv"),
    "eval with an empty argv": (lambda r: _item(r, "full", "whisper-small").update(argv=[]), r"argv"),
    "not an object": (lambda r: r["boxes"].update(p01=[]), r"not an object"),
}


@pytest.mark.parametrize("name", list(RULES))
def test_each_registry_rule_is_refused(tmp_path, reg, name):
    change, match = RULES[name]
    change(reg)
    _refused(reg, tmp_path, match)


def test_data_key_mismatch_is_refused(tmp_path, reg):
    p = tmp_path / "configs" / "full" / "full-p03.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    cfg["selection"] = fr.SMOKE_SELECTION
    p.write_text(json.dumps(cfg), encoding="utf-8")
    _refused(reg, tmp_path, r"full-p03.json differs from the box data config .* \['selection'\]")
    assert _problems(reg, tmp_path, check_files=False) == []  # launch checks the configs at its sha itself


def test_pull_parakeet_missing_counts_as_false(tmp_path, reg):
    p = tmp_path / "configs" / "full" / "full-p01.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    assert "pull_parakeet" not in cfg
    cfg["pull_parakeet"] = False
    p.write_text(json.dumps(cfg), encoding="utf-8")
    assert _problems(reg, tmp_path) == []
    cfg["pull_parakeet"] = True
    p.write_text(json.dumps(cfg), encoding="utf-8")
    _refused(reg, tmp_path, r"\['pull_parakeet'\]")


def test_family_must_be_the_configs(tmp_path, reg):
    _item(reg, "full", "full-p03").update(family="aed")
    _refused(reg, tmp_path, r"family 'aed', but configs/full/full-p03.json trains 'ctc'")
    reg = tiny_registry(tmp_path)
    p = tmp_path / "configs" / "full" / "full-t06.json"
    cfg = json.loads(p.read_text(encoding="utf-8"))
    del cfg["family"]  # the trainer's default family is aed
    p.write_text(json.dumps(cfg), encoding="utf-8")
    assert _problems(reg, tmp_path) == []


def test_family_must_be_the_data_configs_without_pull_parakeet(tmp_path, reg):
    # p01's data config trains ctc without pull_parakeet: bootstrap pulls parakeet_out, and teacher_out only for the
    # eval stems (extent.pull_plan), so an aed item there would find no train labels after the paid rebuild
    folder = tmp_path / "configs" / "full"
    cfg = json.loads((folder / "full-p01.json").read_text(encoding="utf-8"))
    (folder / "p01-aed.json").write_text(json.dumps(dict(cfg, family="aed")), encoding="utf-8")
    _item(reg, "p01", "stores-ctc")["config"] = "configs/full/p01-aed.json"  # a stores item: it has no family field
    _refused(reg, tmp_path, r"stores-ctc: configs/full/p01-aed.json trains family aed, but the box data config "
                            r"configs/full/data-p01.json is family ctc without pull_parakeet")
    # the other way round: a ctc item on an aed data config without pull_parakeet (no parakeet_out is pulled)
    reg = tiny_registry(tmp_path)
    data = json.loads((folder / "data-p01.json").read_text(encoding="utf-8"))
    del data["family"]  # aed by default
    (folder / "data-p01.json").write_text(json.dumps(data), encoding="utf-8")
    probs = _refused(reg, tmp_path, r"full-p01: configs/full/full-p01.json trains family ctc, but the box data "
                                    r"config configs/full/data-p01.json is family aed")
    assert any(p.startswith("boxes.p01.items.stores-ctc:") for p in probs)
    # with pull_parakeet both teachers' labels come for every stem, so one box trains both families (box full)
    loaded = fr.load_registry(tiny_registry(tmp_path))
    assert {i["family"] for i in fr.train_items("full", loaded)} == {"aed", "ctc"}


def test_missing_configs_are_refused_only_with_check_files(tmp_path, reg):
    (tmp_path / "configs" / "full" / "smoke-b-t06.json").unlink()
    (tmp_path / "configs" / "full" / "data-p01.json").unlink()
    probs = _refused(reg, tmp_path, r"smoke-b-t06.json does not exist")
    assert any("data-p01.json does not exist" in p for p in probs)
    assert _problems(reg, tmp_path, check_files=False) == []


def test_item_configs_outside_configs_full_are_allowed(tmp_path, reg):
    # smoke-b's store build reads the frozen study config (contract 7); only data_config must be under configs/full/
    study = tmp_path / "configs" / "study"
    study.mkdir(parents=True)
    (study / "study-t06.json").write_text((tmp_path / "configs" / "full" / "smoke-b-t06.json").read_text("utf-8"),
                                          encoding="utf-8")
    _item(reg, "smoke-b", "stores-eval")["config"] = "configs/study/study-t06.json"
    assert _problems(reg, tmp_path) == []


def test_comments_are_allowed_anywhere(tmp_path, reg):
    _box(reg, "p01")["_comment"] = "box 1"
    _item(reg, "p01", "full-p01")["_why"] = "the plan's P-0.1B"
    # the blocks checked as exact key sets too: a hand-written boxes.json may annotate them
    _box(reg, "full-smoke")["watchdog"]["_comment"] = "600 s alert: the freeze fault must only alert"
    _item(reg, "smoke-b", "selftest")["weights"][0]["_why"] = "the study's final P-0.3B"
    _box(reg, "full-smoke")["faults"][0]["_c"] = "F1"
    _item(reg, "smoke-b", "selftest")["verdict"][0]["_c"] = "check 12"
    assert _problems(reg, tmp_path) == []
    loaded = fr.load_registry(reg, root=tmp_path)
    assert loaded["boxes"]["p01"]["_comment"] == "box 1"
    assert fr.box_env("full-smoke", loaded)["KITSUNE_WATCHDOG_ORPHAN_ACTION"] == "alert"
    # a comment never stands in for a real key
    del _box(reg, "full-smoke")["watchdog"]["orphan_s"]
    _refused(reg, tmp_path, r"watchdog .* is not \{orphan_s, action\}")


def test_speed_sources_that_pass(tmp_path, reg):
    # whisper: none (speed_probe --model <key> comes from args, WP6) or a data-repo `model`; aed/ctc: exactly one source
    items = _box(reg, "full-smoke")["items"]
    items.append({"name": "speed-whisper-small", "kind": "speed", "system": "whisper-small", "speed_kind": "whisper",
                  "args": ["--model", "whisper-small", "--hf-cache", "{hf_cache}"], "stall_min": None,
                  "max_hours": 0.3})
    items.append({"name": "speed-whisper-turbo", "kind": "speed", "system": "whisper-large-v3-turbo",
                  "speed_kind": "whisper", "args": ["--hf-cache", "{hf_cache}", "--model=whisper-large-v3-turbo"],
                  "stall_min": None, "max_hours": 0.3})
    items.append({"name": "speed-whisper-dir", "kind": "speed", "system": "whisper-dir", "speed_kind": "whisper",
                  "model": "models/parakeet-tdt_ctc-0.6b-ja-hf", "stall_min": None, "max_hours": 0.3})
    items.append({"name": "speed-student-dir", "kind": "speed", "system": "student-dir", "speed_kind": "ctc",
                  "model": "models/parakeet-tdt_ctc-0.6b-ja-hf", "stall_min": None, "max_hours": 0.3})
    items.append({"name": "speed-parakeet-ctc", "kind": "speed", "system": "parakeet-ctc", "speed_kind": "parakeet-ctc",
                  "model": "models/parakeet-tdt_ctc-0.6b-ja-hf", "stall_min": None, "max_hours": 0.3})
    assert _problems(reg, tmp_path) == []


def test_malformed_registries_do_not_crash(tmp_path):
    for bad in (None, [], "x", {}, {"version": 1}, {"version": 1, "boxes": []},
                {"version": 1, "boxes": {"p01": {"items": [None, 3, {"kind": ["train"]}, {"kind": "eval",
                                                                                        "weights": 5}]}}},
                {"version": 1, "boxes": {"full": {"items": [{"name": "a", "kind": "speed", "system": ["x"],
                                                             "speed_kind": {}, "only_if_new_machine": ["p01"],
                                                             "weights": [{"name": ["x"]}], "args": "x"}]}}}):
        assert fr.registry_problems(bad, root=tmp_path)


# ------------------------------------------------------------------------------------------------ load_registry


def test_load_registry_missing_file(tmp_path, monkeypatch):
    monkeypatch.delenv("KITSUNE_FULL_REGISTRY", raising=False)
    with pytest.raises(fr.RegistryError) as e:
        fr.load_registry(None, root=tmp_path)
    assert str(e.value) == "configs/full/boxes.json is not in this checkout (WP2c)"
    assert isinstance(e.value, ValueError)
    monkeypatch.setenv("KITSUNE_FULL_REGISTRY", str(tmp_path / "nope.json"))
    with pytest.raises(fr.RegistryError, match="KITSUNE_FULL_REGISTRY"):
        fr.load_registry()
    with pytest.raises(fr.RegistryError, match="does not exist"):
        fr.load_registry(tmp_path / "other.json")


def test_load_registry_from_the_env_resolves_against_its_checkout(tmp_path, monkeypatch):
    reg = tiny_registry(tmp_path, write_boxes=True)
    monkeypatch.setenv("KITSUNE_FULL_REGISTRY", str(tmp_path / "configs" / "full" / "boxes.json"))
    loaded = fr.load_registry()
    assert loaded == fr.load_registry(reg, root=tmp_path)
    # the box readers load it the same way, with its configs read in that checkout
    assert fr.box_students("p01") == ["students/study/p01"]
    monkeypatch.delenv("KITSUNE_FULL_REGISTRY")
    assert fr.load_registry(None, root=tmp_path) == loaded  # <root>/configs/full/boxes.json
    assert fr.load_registry(tmp_path / "configs" / "full" / "boxes.json") == loaded  # a path
    assert fr.load_registry(str(tmp_path / "configs" / "full" / "boxes.json")) == loaded


def test_load_registry_bad_json(tmp_path):
    f = tmp_path / "configs" / "full" / "boxes.json"
    f.parent.mkdir(parents=True)
    f.write_text("{not json", encoding="utf-8")
    with pytest.raises(fr.RegistryError, match="not readable JSON"):
        fr.load_registry(None, root=tmp_path)


def test_load_registry_problems_are_listed(tmp_path, reg):
    _box(reg, "p01").update(gpus=0, max_dph=-1)
    with pytest.raises(fr.RegistryError) as e:
        fr.load_registry(reg, root=tmp_path)
    assert len(e.value.problems) == 2 and "gpus 0" in str(e.value) and "max_dph -1" in str(e.value)


def test_a_loaded_registry_remembers_its_checkout(tmp_path, monkeypatch):
    # code that loads a registry file once and passes the dict on must read that file's checkout, not this repo
    monkeypatch.delenv("KITSUNE_FULL_REGISTRY", raising=False)
    tiny_registry(tmp_path, write_boxes=True)
    loaded = fr.load_registry(tmp_path / "configs" / "full" / "boxes.json")  # a path, no root=
    assert fr.box_students("p01", loaded) == ["students/study/p01"]
    assert fr.box_ctc_students("full", loaded) == ["students/study/p03", "students/study/p005"]
    assert fr.student_checks("full", lambda s: _metas()[s], loaded) == []
    assert fr.registry_problems(loaded) == []
    assert fr.load_registry(loaded) == loaded
    assert fr.box_students("p01", copy.deepcopy(loaded)) == ["students/study/p01"]  # a copy keeps it
    assert json.loads(json.dumps(loaded)) == loaded  # it is the plain registry otherwise
    with pytest.raises(fr.RegistryError, match="cannot read it under"):
        fr.box_students("p01", loaded, tmp_path / "other")  # an explicit root wins


def test_tiny_registry_remembers_its_root(tmp_path):
    reg = tiny_registry(tmp_path)
    assert fr.registry_problems(reg) == []  # no root=: the configs are read under tmp_path
    assert fr.box_students("p01", reg) == ["students/study/p01"]
    assert fr.load_registry(reg) == fr.load_registry(reg, root=tmp_path)


def test_read_json_reads_the_configs_instead_of_a_root(tmp_path, reg):
    # launch reads the configs at the sha it rents (git_show), never the working tree: a reader replaces root
    files = {f"configs/full/{p.name}": json.loads(p.read_text(encoding="utf-8"))
             for p in (tmp_path / "configs" / "full").glob("*.json")}
    seen = []

    def read(rel):
        seen.append(rel)
        return copy.deepcopy(files[rel])  # KeyError for a file the sha does not have
    plain = json.loads(json.dumps(reg))  # the registry JSON at the sha: it remembers no root
    nowhere = tmp_path / "nowhere"
    assert fr.registry_problems(plain, root=nowhere, read_json=read) == []
    assert "configs/full/data-p01.json" in seen and "configs/full/smoke-b-t06.json" in seen
    loaded = fr.load_registry(plain, root=nowhere, read_json=read)
    assert fr.box_students("full", loaded, read_json=read) == ["students/study/t06", "students/study/p03",
                                                               "students/study/p005"]
    assert fr.box_ctc_students("full", loaded, read_json=read) == ["students/study/p03", "students/study/p005"]
    assert fr.student_checks("full", lambda s: _metas()[s], loaded, read_json=read) == []
    # what the reader gives is checked like a file: a data-key mismatch at the sha is refused
    files["configs/full/full-p03.json"]["selection"] = fr.SMOKE_SELECTION
    assert any("full-p03.json differs from the box data config" in p
               for p in fr.registry_problems(plain, root=nowhere, read_json=read))
    # a file the reader cannot give is a problem (any error of the read), never a crash
    del files["configs/full/full-p03.json"]
    probs = fr.registry_problems(plain, root=nowhere, read_json=read)
    assert any("full-p03.json: not readable JSON (KeyError" in p for p in probs), probs
    with pytest.raises(fr.RegistryError, match="full-p03.json: cannot read it with read_json"):
        fr.box_students("full", loaded, read_json=read)

    def missing(rel):
        raise FileNotFoundError(rel)
    probs = fr.registry_problems(plain, read_json=missing)
    assert any("configs/full/data-p01.json does not exist (read_json)" in p for p in probs), probs


# ------------------------------------------------------------------------------------------------ box readers


def test_box_readers(tmp_path, reg):
    assert fr.box_spec("p01", reg)["gpus"] == 1
    with pytest.raises(fr.RegistryError, match="'smoke-c'"):
        fr.box_spec("smoke-c", reg)
    assert [i["name"] for i in fr.box_items("p01", reg)] == ["stores-ctc", "full-p01", "m4-full-p01"]
    assert [i["name"] for i in fr.train_items("full", reg)] == ["full-t06", "full-p03", "full-p005"]
    assert [i["name"] for i in fr.train_items("full-smoke", reg)] == ["smoke-t06", "smoke-p03", "smoke-p01",
                                                                      "smoke-p005", "smoke-nostart"]
    assert fr.box_configs("full", reg) == ["configs/full/data-full.json", "configs/full/full-p03.json",
                                           "configs/full/full-t06.json", "configs/full/full-p005.json",
                                           "configs/full/boxes.json"]
    assert fr.box_students("full", reg, tmp_path) == ["students/study/t06", "students/study/p03",
                                                      "students/study/p005"]
    assert fr.box_students("full-smoke", reg, tmp_path) == ["students/study/t06", "students/study/p03",
                                                            "students/study/p01", "students/study/p005"]
    assert fr.box_ctc_students("full", reg, tmp_path) == ["students/study/p03", "students/study/p005"]
    assert fr.box_ctc_students("smoke-b", reg, tmp_path) == []
    assert fr.box_students("smoke-b", reg, tmp_path) == []
    assert fr.box_extra_files("p01", reg) == [fr.FROZEN_MANIFEST, "labels/full/selections/full_study/full.json"]
    assert fr.box_extra_files("smoke-b", reg) == []
    assert fr.box_extra_dirs("full-smoke", reg) == ["models/parakeet-tdt_ctc-0.6b-ja-hf"]
    assert fr.box_extra_dirs("p01", reg) == []
    # copies: a caller that edits what it got does not change the registry
    fr.box_spec("p01", reg)["items"].clear()
    assert len(fr.box_items("p01", reg)) == 3


def test_box_env(reg):
    assert fr.box_env("p01", reg) == {"KITSUNE_N_GPUS": "1", "KITSUNE_WATCHDOG_HB_FILE": "train_hb",
                                      "KITSUNE_WATCHDOG_ORPHAN_S": "3600", "KITSUNE_WATCHDOG_ORPHAN_ACTION": "stop"}
    assert fr.box_env("full", reg)["KITSUNE_N_GPUS"] == "2"
    assert fr.box_env("full-smoke", reg)["KITSUNE_WATCHDOG_ORPHAN_ACTION"] == "alert"


def _meta(run: str, **over) -> dict:
    spec = prereg.RUNS[run]
    m = {"stage": "complete", **{k: spec[k] for k in ("family", "init_class", "seed", "params_total",
                                                       "params_non_embedding")},
         "closed_form_params": spec["params_total"], "calibration": {"ids_sha256": spec.get("calib_ids_sha256")}}
    m.update(over)
    return m


def _metas(**over) -> dict:
    by_student = {prereg.RUNS[r]["student"]: r for r in ("study-t06", "study-p03", "study-p01", "study-p005")}
    return {s: _meta(r, **over.get(r, {})) for s, r in by_student.items()}


def test_student_checks(tmp_path, reg):
    metas = _metas()
    seen = []

    def read(s):
        seen.append(s)
        return metas[s]
    assert fr.student_checks("full-smoke", read, reg, tmp_path) == []
    assert seen == ["students/study/t06", "students/study/p03", "students/study/p01", "students/study/p005"]  # once
    metas = _metas(**{"study-p03": {"params_total": 1}})
    probs = fr.student_checks("full", lambda s: metas[s], reg, tmp_path)
    assert probs and all(p.startswith("students/study/p03: study-p03: ") for p in probs)

    def boom(s):
        raise FileNotFoundError(s)
    probs = fr.student_checks("p01", boom, reg, tmp_path)
    assert probs == ["students/study/p01/student_meta.json: cannot read it (FileNotFoundError: students/study/p01)"]
    # a config that trains another student than its study_run registers
    _item(reg, "p01", "full-p01")["study_run"] = "study-p005"
    probs = fr.student_checks("p01", lambda s: _metas()[s], reg, tmp_path)
    assert any("full-p01: configs/full/full-p01.json trains students/study/p01, study-p005 registers "
               "students/study/p005" in p for p in probs)


# ------------------------------------------------------------------------------------------------ CLI


def _cli(argv, capsys):
    rc = fr.main(argv)
    out = capsys.readouterr()
    return rc, out.out, out.err


def test_cli(tmp_path, monkeypatch, capsys):
    tiny_registry(tmp_path, write_boxes=True)
    monkeypatch.delenv("KITSUNE_FULL_REGISTRY", raising=False)
    monkeypatch.delenv("KITSUNE_BOX", raising=False)
    root = ["--root", str(tmp_path)]
    assert _cli(["students", "--box", "full", *root], capsys)[:2] == (
        0, "students/study/t06\nstudents/study/p03\nstudents/study/p005\n")
    assert _cli(["extra-files", "--box", "p01", *root], capsys)[:2] == (
        0, f"{fr.FROZEN_MANIFEST}\nlabels/full/selections/full_study/full.json\n")
    assert _cli(["extra-dirs", "--box", "full-smoke", *root], capsys)[:2] == (
        0, "models/parakeet-tdt_ctc-0.6b-ja-hf\n")
    assert _cli(["extra-dirs", "--box", "p01", *root], capsys)[:2] == (0, "")
    rc, out, _ = _cli(["show", "--box", "p01", *root], capsys)
    shown = json.loads(out)
    assert rc == 0 and shown["box"] == "p01" and shown["env"]["KITSUNE_N_GPUS"] == "1"
    assert shown["spec"]["items"][2]["needs"] == ["full-p01"] and shown["students"] == ["students/study/p01"]
    monkeypatch.setenv("KITSUNE_BOX", "p01")  # --box defaults to the env
    assert _cli(["students", *root], capsys)[:2] == (0, "students/study/p01\n")


def test_cli_check_students(tmp_path, monkeypatch, capsys):
    tiny_registry(tmp_path, write_boxes=True)
    monkeypatch.delenv("KITSUNE_FULL_REGISTRY", raising=False)
    for s, m in _metas().items():
        (tmp_path / s).mkdir(parents=True)
        (tmp_path / s / "student_meta.json").write_text(json.dumps(m), encoding="utf-8")
    rc, out, _ = _cli(["check-students", "--box", "full", "--root", str(tmp_path)], capsys)
    assert rc == 0 and "3 student(s) are the registered builds" in out
    (tmp_path / "students/study/p005/student_meta.json").write_text(json.dumps(_meta("study-p005", seed=1)),
                                                                     encoding="utf-8")
    rc, _, err = _cli(["check-students", "--box", "full", "--root", str(tmp_path)], capsys)
    assert rc == 2 and "student refused: students/study/p005: study-p005: seed 1" in err
    (tmp_path / "students/study/p01/student_meta.json").unlink()
    rc, _, err = _cli(["check-students", "--box", "p01", "--root", str(tmp_path)], capsys)
    assert rc == 2 and "cannot read it" in err


def test_cli_refusals(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("KITSUNE_FULL_REGISTRY", raising=False)
    monkeypatch.delenv("KITSUNE_BOX", raising=False)
    rc, _, err = _cli(["students", "--box", "p01", "--root", str(tmp_path)], capsys)
    assert rc == 2 and "not in this checkout" in err  # no registry
    reg = tiny_registry(tmp_path)
    reg["boxes"].pop("smoke-b")
    (tmp_path / fr.BOXES_FILE).write_text(json.dumps(reg), encoding="utf-8")
    rc, _, err = _cli(["show", "--box", "smoke-b", "--root", str(tmp_path)], capsys)
    assert rc == 2 and "'smoke-b' is not in the registry" in err
    reg["boxes"]["p01"]["gpus"] = 0
    (tmp_path / fr.BOXES_FILE).write_text(json.dumps(reg), encoding="utf-8")
    rc, _, err = _cli(["check-students", "--box", "p01", "--root", str(tmp_path)], capsys)
    assert rc == 2 and "gpus 0" in err
    with pytest.raises(SystemExit) as e:
        fr.main(["students", "--root", str(tmp_path)])  # no --box and no KITSUNE_BOX
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        fr.main(["students", "--box", "box-a"])  # not a full box
    assert e.value.code == 2


def test_cli_as_a_module(tmp_path):
    tiny_registry(tmp_path, write_boxes=True)
    env = {k: v for k, v in __import__("os").environ.items() if k not in ("KITSUNE_FULL_REGISTRY", "KITSUNE_BOX")}
    out = subprocess.run([sys.executable, "-m", "kitsune.fullrun", "students", "--box", "p01", "--root",
                          str(tmp_path)], cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)
    assert (out.returncode, out.stdout) == (0, "students/study/p01\n"), out.stderr
