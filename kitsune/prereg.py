"""The size study's registered student builds and its selection constants, which the full-data pipeline reuses.

The size study (2026-09-25 to 09-27) pre-registered everything it fixed in advance in study/PREREG.json, committed
before its first box and frozen since (tests/test_prereg.py pins its sha256; its manifest block and the teachers'
baselines are read by scripts/05_evaluate.py, tools/full_report.py and kitsune.evaluate). What the full runs still read
from this module:

  RUNS          the registered student builds (students/study/<run>): family, init class, seed, the exact parameter
                counts and the calibration ids a pruned student's FFNs were ranked on. Every full run trains one of
                them (configs/full/boxes.json study_run); student_problems checks a pulled build against its entry
                (kitsune.fullrun.student_checks, at launch and in bootstrap), and tools/make_study_configs.py run_config
                builds every full config from it (with W_CTC)
  selection     STUDY_SELECTION, SELECTION_SEED, STUDY_SCHEMA, STUDY_REASONS, SELECTION_FILE, MANIFEST_FILE,
                ONE_ROOT_MAX_FRAC, M4_SETS, POOL_MIN_HOURS, cap_rule and study_files: the frozen study selection and
                its eval manifest (scripts/make_selection.py's study mode builds them; kitsune.extent, kitsune.devslice,
                vast/launch.py's selection checks and scripts/04_distill.py's selection_recipe.study read them)

Pure Python (stdlib only: kitsune.extent imports it at import); no torch.
"""

# importance_ids_sha256 of the first run's 1,000 calibration utterances (recovered 2026-09-26): every pruned student's
# FFNs are ranked on them (T-0.6B keeps the first run's selection, made on them; the others were rebuilt on them)
CALIB_IDS_SHA256 = "5e31cd68bebc2180a89089187b559e4b164a2394046c619a477b14c169a2a956"

# run -> what the study fixed about it (CONTRACT.md section 1, STUDY.md 1.1 and 2.2; study/PREREG.json "runs"). params_*
# are the exact meta-device counts every builder asserts; calib_ids_sha256 (pruned students) the ids their FFNs were
# ranked on. student_problems checks a built student's meta against its entry. lr_from, box, shape and bn are the
# study's record (its LR probe class, its box, its shape and BatchNorm mode). T-0.3B and the bridge are B8x2560 +
# decoder {0,2,5,7} by the owner's decision of 2026-09-26
RUNS = {
    "study-t06": dict(family="aed", init_class="pruned_kept", student="students/study/t06", seed=1234,
                      shape="encoder 20 of 48 layers [0,2,5,7,10,12,15,17,20,22,25,27,30,32,35,37,40,42,45,47], "
                            "d 1280, FFN 5120 -> 2560; decoder layers {0,2,5,7}, D 1024; vocab 16,384 tied",
                      params_total=616_963_328, params_non_embedding=599_137_536, lr_from="kept-t03",
                      warmup_steps=300, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                      micro_audio_s=600, box="A", calib_ids_sha256=CALIB_IDS_SHA256),
    "study-t03": dict(family="aed", init_class="pruned_kept", student="students/study/t03", seed=1234,
                      shape="encoder 8 of 48 layers [0,7,13,20,27,34,40,47] (evenly_spaced(8, 48)), d 1280, FFN 5120 "
                            "-> 2560; T-0.6B's decoder layers {0,2,5,7}, D 1024; vocab 16,384 tied",
                      params_total=301_822_208, params_non_embedding=283_996_416, lr_from="kept-t03",
                      warmup_steps=300, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                      micro_audio_s=1200, box="A", calib_ids_sha256=CALIB_IDS_SHA256),
    "study-bridge": dict(family="aed", init_class="scratch", student="students/study/bridge", seed=1234,
                         shape="the T-0.3B shape (encoder 8 layers, d 1280, 8 x 160, FFN 2560; decoder 4 layers, "
                               "D 1024, 8 x 128, FFN 4096; vocab 16,384 tied), from scratch",
                         params_total=301_822_208, params_non_embedding=283_996_416, lr_from="bridge",
                         warmup_steps=2000, weight_decay=1e-3, bn="train mode", aux_ctc=0.3, micro_audio_s=1200,
                         box="B"),
    "study-t01": dict(family="aed", init_class="scratch", student="students/study/t01", seed=1234,
                      shape="encoder 12 layers, d 512, 8 x 64, FFN 2048, kernel 9; decoder 4 layers, D 512, 8 x 64, "
                            "FFN 2048; vocab 16,384 tied",
                      params_total=103_996_416, params_non_embedding=95_083_520, lr_from="scratch",
                      warmup_steps=2000, weight_decay=1e-3, bn="train mode", aux_ctc=0.3, micro_audio_s=1600, box="A"),
    "study-t01-s1235": dict(family="aed", init_class="scratch", student="students/study/t01-s1235", seed=1235,
                            shape="the T-0.1B shape, seed 1235", params_total=103_996_416,
                            params_non_embedding=95_083_520, lr_from="scratch", warmup_steps=2000, weight_decay=1e-3,
                            bn="train mode", aux_ctc=0.3, micro_audio_s=1600, box="replicate"),
    "study-t005": dict(family="aed", init_class="scratch", student="students/study/t005", seed=1234,
                       shape="encoder 10 layers, d 384, 6 x 64, FFN 1536, kernel 9; decoder 3 layers, D 384, 6 x 64, "
                             "FFN 1536; vocab 16,384 tied",
                       params_total=51_209_600, params_non_embedding=44_524_928, lr_from="scratch",
                       warmup_steps=2000, weight_decay=1e-3, bn="train mode", aux_ctc=0.3, micro_audio_s=1600,
                       box="A"),
    "study-p03": dict(family="ctc", init_class="pruned_kept", student="students/study/p03", seed=1234,
                      shape="Parakeet encoder 16 of 24 layers [0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23], d 1024, "
                            "FFN 4096 -> 2560; the Parakeet CTC head verbatim (3,073 classes)",
                      params_total=308_524_033, params_non_embedding=305_374_208, lr_from="kept-p03",
                      warmup_steps=300, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                      micro_audio_s=1200, box="B", calib_ids_sha256=CALIB_IDS_SHA256),
    "study-p01": dict(family="ctc", init_class="pruned_lost", student="students/study/p01", seed=1234,
                      shape="Parakeet encoder 8 layers [0,3,7,10,13,16,20,23], FFN 768; CTC head",
                      params_total=98_468_865, params_non_embedding=95_319_040, lr_from="lost",
                      warmup_steps=1000, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                      micro_audio_s=1600, box="B", calib_ids_sha256=CALIB_IDS_SHA256),
    "study-p005": dict(family="ctc", init_class="pruned_lost", student="students/study/p005", seed=1234,
                       shape="Parakeet encoder 4 layers [0,8,15,23], FFN 768; CTC head",
                       params_total=52_190_209, params_non_embedding=49_040_384, lr_from="lost",
                       warmup_steps=1000, weight_decay=0.0, bn="teacher stats, frozen (decision 16)", aux_ctc=0.0,
                       micro_audio_s=1600, box="B", calib_ids_sha256=CALIB_IDS_SHA256),
}

W_CTC = 0.8  # decision 23: the CTC trainer's w_ctc for every Parakeet student (loss.w_ctc of their configs)

# ------------------------------------------------------------------------------------------------ the selection

# the recipe block every study run config carries as selection_recipe.study (scripts/make_selection.py applies it,
# vast/launch.py checks the selection was built with it and that it is this one). probe_n is per train source, as the
# existing probe convention draws it: 300 x the five sources = 1,500 kept rows
STUDY_SELECTION = {"f1a_max": 0.5, "dedup_min_chars": 15, "draw_audio_s": 3_600_000, "probe_n": 300,
                   "neutral_max_cer": 0.5}
SELECTION_SEED = 1234  # make_selection --seed of the study selection: the draw, the probe and the greedy subsets
STUDY_SCHEMA = 1  # of the study selection's sidecar and manifest (scripts/make_selection.py writes both)
# drop reasons of a study selection, after the existing ones (truncated, not_judged, no_agree, agree>A, no_audio)
STUDY_REASONS = ("not_in_parakeet", "f1a_disagree", "eval_dup", "ctc_infeasible", "not_drawn")
SELECTION_FILE = "labels/full/selections/study_1000h.parquet"
MANIFEST_FILE = "study_manifest.json"  # next to the selection
ONE_ROOT_MAX_FRAC = 0.001  # K5: train rows in teacher_out only (not_in_parakeet) above this share refuse the launch
M4_SETS = ("eval_jsut", "eval_cv8", "eval_reazon", "galgame_neutral")
POOL_MIN_HOURS = 1010  # the reazon_large cap: the smallest N whose pool after all filters holds >= this


def student_problems(run: str, meta: dict) -> list[str]:
    """Why a built student (its student_meta.json) is not the one the rules register for `run`: not complete, another
    family, init class, seed or parameter count (total, non-embedding, and the builder's closed form), or - for a
    pruned student - FFNs ranked on other calibration ids than the first run's (calibration.importance_ids_sha256 of a
    Transcribe student, calibration.ids_sha256 of a Parakeet one). Launch and bootstrap check every student a full box
    pulls with this (kitsune.fullrun.student_checks): a stale build (the B10 T-0.3B of 320.75M, the P students on the
    re-drawn ids) fails here."""
    spec = RUNS[run]
    p = []
    if meta.get("stage") != "complete":
        p.append(f"{run}: stage {meta.get('stage')!r}, not complete")
    for k in ("family", "init_class", "seed", "params_total", "params_non_embedding"):
        if meta.get(k) != spec[k]:
            p.append(f"{run}: {k} {meta.get(k)!r}, registered {spec[k]!r}")
    if meta.get("closed_form_params") != spec["params_total"]:
        p.append(f"{run}: closed_form_params {meta.get('closed_form_params')!r}, registered {spec['params_total']!r}")
    if spec["init_class"] != "scratch":
        cal = meta.get("calibration") or {}
        ids = cal.get("importance_ids_sha256") or cal.get("ids_sha256")
        if ids != spec["calib_ids_sha256"]:
            p.append(f"{run}: FFNs ranked on calibration ids {str(ids)[:12]}..., registered "
                     f"{spec['calib_ids_sha256'][:12]}...")
    return p


def study_files(selection: str) -> tuple[str, str]:
    """(sidecar, manifest) of a study selection, as repo paths: <selection without .parquet>.json and
    study_manifest.json in the selection's folder (make_selection writes both; pull_plan and launch need both)."""
    stem = selection[: -len(".parquet")] if selection.endswith(".parquet") else selection
    folder = selection.rsplit("/", 1)[0] + "/" if "/" in selection else ""
    return f"{stem}.json", f"{folder}{MANIFEST_FILE}"


def cap_rule(pool_hours_if_capped: dict) -> int | None:
    """The reazon_large cap the rule gives: the smallest N whose pool after all filters holds >= POOL_MIN_HOURS, from
    {N (int or its str, as JSON keys are): pool hours}; None when no N listed reaches it (rebuild at a larger cap)."""
    ok = [int(n) for n, h in pool_hours_if_capped.items() if isinstance(h, (int, float)) and h >= POOL_MIN_HOURS]
    return min(ok) if ok else None
