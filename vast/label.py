"""The label box controller: label the full extent on one rented GPU, upload it write-once, then end the box.

Started by vast/onstart.sh (KITSUNE_JOB=label) in place of supervise.py; it holds $KITSUNE_STATE/supervise.lock, so
`onstart.sh --rearm` sees it. The design is docs FINAL.md §1 and §10; in short:

  steps    plan -> pull -> models -> selftest -> roots -> [ingest lane] -> golden -> [GPU lanes]
           each a bounded child `python vast/label.py step <name>`, recorded in label.json (selftest re-runs on every
           start; the others never do, so a container restart makes no Hub call before the lanes restart).
           Step exit codes: 0 ok, 3 refusal, 65 integrity, 69 host failure, anything else transient (retried twice,
           after 60 s and 120 s, then a host failure).
  lanes    ingest (01 --extent-config), cohere-0..P-1 (02 --follow, crc32 partition), parakeet (02p --follow).
           Each touches $STATE/hb/<lane>; a heartbeat older than hang_min (ingest: ingest_hang_min) kills the lane's
           process group. Exit 0 = done, 65 = integrity, else a restart after 30/120/300 s; 4 failures in a row
           without progress (no new lanes/<lane>.jsonl line, no manifest growth for ingest) = host failure.
  janitor  every 60 s: a train shard whose Cohere and Parakeet npz both verify by ids on disk loses its audio (the id
           sidecar stays; data/pruned.jsonl records it); $STATE/ingest.hold holds 01 while the unpruned backlog is over
           backlog_gb or free disk is under 45 GB, and is removed below backlog_gb - 20 and above 60 GB free; under
           15 GB free while held = host failure.
  cycles   every sync_every_min: 02b (CPU, explicit sources, never the gate sets), then
           finish.py --job label --sync-only [--no-infra] (infra every 3rd cycle).
  rates    every 30 min from lanes/*.jsonl (non-adopted shards): x realtime per lane, remaining hours, ETA. After
           60 min of Cohere work the Cohere lanes together must reach min_xrt, else slow_host.
  budget   now >= deadline - end_margin_min -> budget end; lanes finished after deadline - finalize_margin_min ->
           budget end (too late to finalize).
  finalize F1 extent.json, F2 selections, F3 reports + provenance, F4 consumer check (local), F5 final sync + verify,
           F6 consumer check (Hub listing), F7 seal (COMPLETE.json, lease released), F8 finish --destroy.
  end()    label_end.json + `final` in label.json, SIGTERM the lanes, then finish.py: destroy for success, budget,
           host_failure and slow_host (finish verifies first and stops if it cannot); stop for refusal, integrity,
           unknown and sync_unverifiable. KITSUNE_LABEL_END=stop turns every destroy into a stop. A recorded final
           without the halt marker is replayed at the next start (as supervise.py does).

The controller touches $STATE/label_hb at least every 30 s (the watchdog's orphan rule), also while a bounded call
blocks. It imports no torch: the steps that need it (selftest, golden, models) run in their own child process.

Usage:
  python vast/label.py [--dry-run]           # the controller (finish.py only prints with --dry-run)
  python vast/label.py step <name> [...]     # one step, exits with its rc
"""
import argparse
import contextlib
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import supervise  # noqa: E402

EXIT_OK, EXIT_REFUSAL, EXIT_INTEGRITY, EXIT_HOST, EXIT_TIMEOUT = 0, 3, 65, 69, 124
STEPS = ("plan", "pull", "models", "selftest", "roots", "golden", "consumer-check", "seal")
BOOT_STEPS = ("plan", "pull", "models", "selftest", "roots")  # before the ingest lane; golden runs inside the loop
RERUN_STEPS = ("selftest",)  # local and cheap: re-run on every start
STEP_TIMEOUT_S = {"plan": 600, "pull": 3600, "models": 1800, "selftest": 600, "roots": 600, "golden": 3600,
                  "consumer-check": 7200, "seal": 1800}
STEP_RETRY_WAITS = (60, 120)
LANE_RESTART_S = (30, 120, 300)
MAX_FAILS_NO_PROGRESS = 4
TERM_GRACE_S = 60
LOOP_S = 15  # <= 30: the controller heartbeat
BEAT_S = 30
JANITOR_S = 60
JANITOR_MAX = 100  # shards verified per janitor pass
RATES_S = 1800
SLOW_AFTER_S = 3600  # Cohere work (non-adopted shards) before the slow-host floor applies
SECOND_TIMEOUT_S = 3600
SYNC_TIMEOUT_S = 1800
VERIFY_TIMEOUT_S = 3600
MAKE_SELECTION_TIMEOUT_S = 3600
INFRA_EVERY = 3  # every 3rd sync cycle also pushes the infra files
LOUD_SYNC_FAILS = 3
GB = 1e9
HOLD_FREE_GB, RELEASE_FREE_GB, CRITICAL_FREE_GB = 45, 60, 15
RELEASE_BELOW_GB = 20  # the hold is removed once the backlog is this far under backlog_gb
GOLDEN_TRIES = 3

GATE_SETS = ("eval_jsut", "eval_cv8", "eval_reazon")  # adopt-only: never recomputed on the box
SECOND_SOURCES = ("reazon_small", "reazon_large", "emilia_yodas", "emilia_nc", "galgame", "eval_emilia")
SEED_SOURCES = ("reazon_small", "galgame", "emilia_yodas", "eval_jsut", "eval_cv8", "eval_reazon", "eval_emilia")
TEACHER_SETTING_KEYS = ("model", "model_revision", "language", "punctuation", "k", "save_encoder")
PARAKEET_SETTING_KEYS = ("k_tdt", "k_ctc", "ctc_dense_thr", "max_symbols")

DESTROY_CLASSES = ("success", "budget", "host_failure", "slow_host")
STOP_CLASSES = ("refusal", "integrity", "unknown", "sync_unverifiable")

# pins copied from modules that import torch (a test compares them with the sources)
COHERE_MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"  # scripts/02_teacher_pass.py MODEL_ID
COHERE_REVISION = "b1eacc2686a3d08ceaae5f24a88b1d519620bc09"  # scripts/02_teacher_pass.py MODEL_REVISION
WHISPER_TOK_ID = "openai/whisper-large-v3"
WHISPER_TOK_REVISION = "06f233fe06e710322aca913c1bc4249a0d71fce1"  # scripts/02b_second_opinion.py
PARAKEET_PATH = "models/parakeet-tdt_ctc-0.6b-ja-hf"  # kitsune/parakeet.py PARAKEET_PATH

KNOBS = {  # configs/full.json "label"; a config's block overrides these (one level deep for cohere/parakeet)
    "cohere": {"procs": 2, "max_batch_seconds": 1200, "max_batch": 256, "workers": "auto", "torch_threads": 2,
               "vram_fraction": 0.33},
    "parakeet": {"max_batch_seconds": 2400, "max_batch": 512, "workers": "auto", "torch_threads": 2,
                 "vram_fraction": 0.22, "k_tdt": 8, "k_ctc": 8, "ctc_dense_thr": 0.95, "max_symbols": 10},
    "sync_every_min": 20, "backlog_gb": 80, "hang_min": 20, "ingest_hang_min": 30, "min_xrt": 400,
    "end_margin_min": 60, "finalize_margin_min": 90,
}


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [label] {msg}", flush=True)


class StepError(Exception):
    rc = 1


class Refusal(StepError):
    rc = EXIT_REFUSAL


class Integrity(StepError):
    rc = EXIT_INTEGRITY


class HostFailure(StepError):
    rc = EXIT_HOST


class Ended(Exception):
    """Raised by Controller.end() once the box's end is decided and acted on; carries the controller's rc."""

    def __init__(self, rc: int):
        super().__init__(rc)
        self.rc = rc


# --------------------------------------------------------------------------------------------------------- helpers


def knobs(cfg: dict) -> dict:
    block = cfg.get("label") or {}
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in KNOBS.items()}
    for k, v in block.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def effective_cores() -> int:
    """cgroup v2 cpu.max quota when set (a vast container often has fewer cores than nproc), else os.cpu_count()."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            return max(1, round(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def lane_workers(k: dict, lane: str, cores: int) -> int:
    w = k[lane]["workers"]
    if w == "auto":
        return max(2, round((0.3 if lane == "cohere" else 0.4) * (cores - 3)))
    return int(w)


def write_json_atomic(path: Path, obj, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=indent, ensure_ascii=False))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def jsonl_rows(path: Path) -> list[dict]:
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue  # a torn last line from a killed append
    except OSError:
        pass
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def npz_ids(path: Path) -> list[str] | None:
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as z:
            return [str(x) for x in z["ids"]]
    except Exception:
        return None


def teacher_done(npz: Path, shard_ids: list[str]) -> bool:
    """02's id-based done check on disk (scripts/02_teacher_pass.py shard_is_done without the argparse settings, which
    step roots pinned in meta.json): the npz loads, n_scanned == the data shard's rows, its ids are an ordered
    subsequence of the shard's ids, and the jsonl is there."""
    import numpy as np
    from kitsune.labelpass import ordered_subsequence

    try:
        with np.load(npz, allow_pickle=False) as z:
            ids, n = [str(x) for x in z["ids"]], int(z["n_scanned"])
    except Exception:
        return False
    return n == len(shard_ids) and ordered_subsequence(ids, shard_ids) and npz.with_suffix(".jsonl").is_file()


def http_status(e: BaseException) -> int | None:
    return getattr(getattr(e, "response", None), "status_code", None)


def is_transient(e: BaseException) -> bool:
    s = http_status(e)
    return s is not None and (s >= 500 or s == 429)


def refusal_from(e: BaseException, what: str) -> BaseException:
    """A Hub error as a Refusal when it is about access (gated, missing, 401/403/404), else the error itself (a 5xx or
    429 is transient: the step exits 1 and the controller retries it)."""
    if is_transient(e):
        return e
    if type(e).__name__ in ("GatedRepoError", "RepositoryNotFoundError") or http_status(e) in (401, 403, 404):
        return Refusal(f"{what}: {type(e).__name__}: {e}")
    return e


def corpus_cer(hyps: list[str], refs: list[str]) -> float:
    from kitsune.evaluate import corpus_cer as _cc

    return float(_cc(hyps, refs)["cer"]) if refs else 0.0


def golden_cer(box: list[dict], ref: dict[str, str], key: str = "hyp") -> tuple[float, int]:
    """Corpus CER of the box's hypotheses against reference hypotheses, on the ids both have; (cer, n compared)."""
    pairs = [(r[key], ref[r["id"]]) for r in box if r.get("id") in ref and r.get(key) is not None]
    if not pairs:
        return float("inf"), 0
    return corpus_cer([h for h, _ in pairs], [g for _, g in pairs]), len(pairs)


# ------------------------------------------------------------------------------------------ processes and the clock


class Proc:
    def __init__(self, popen: subprocess.Popen):
        self.p = popen

    def poll(self):
        return self.p.poll()

    def terminate_group(self, grace: float = TERM_GRACE_S):
        """SIGTERM the lane's process group (its prep threads, its children), SIGKILL after `grace` seconds."""
        if self.p.poll() is not None:
            return
        try:
            os.killpg(self.p.pid, signal.SIGTERM)
        except (AttributeError, OSError):  # Windows (no process groups), or already gone
            self.p.terminate()
        try:
            self.p.wait(grace)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.p.pid, signal.SIGKILL)
            except (AttributeError, OSError):
                self.p.kill()
            try:
                self.p.wait(30)
            except subprocess.TimeoutExpired:
                pass


class Runner:
    def __init__(self, cwd: Path = ROOT):
        self.cwd = Path(cwd)

    def start(self, name: str, argv: list[str], env: dict, log_path: Path) -> Proc:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"start {name}: {' '.join(map(str, argv))}")
        with open(log_path, "ab") as f:
            p = subprocess.Popen([str(a) for a in argv], cwd=self.cwd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=os.name != "nt")
        return Proc(p)


class Clock:
    def time(self) -> float:
        return time.time()

    def sleep(self, s: float):
        time.sleep(s)


# ------------------------------------------------------------------------------------------------------ controller


class Controller:
    def __init__(self, env: dict | None = None, runner=None, clock=None, final_finish=None, py: str = sys.executable,
                 dry_run: bool = False):
        self.env = dict(os.environ if env is None else env)
        self.kdir = Path(self.env.get("KITSUNE_DIR") or ROOT)
        self.state_dir = Path(self.env.get("KITSUNE_STATE") or "/workspace/kitsune_state")
        self.log_dir = self.state_dir.parent  # /workspace in production: lane_<name>.log
        self.cfg_path = self.env.get("KITSUNE_CONFIG") or "configs/full.json"
        self.cfg = json.loads((self.kdir / self.cfg_path).read_text(encoding="utf-8"))
        self.label_cfgs = [c for c in (self.env.get("KITSUNE_LABEL_CONFIGS") or self.cfg_path).split(",") if c]
        self.k = knobs(self.cfg)
        self.runner = runner or Runner(self.kdir)
        self.clock = clock or Clock()
        self._final_finish = final_finish or supervise.final_finish
        self.py = py
        self.dry_run = dry_run
        self.data_rel = self.cfg.get("data_root", "data")
        self.root_rel = (self.cfg.get("extent") or {}).get("root", "labels/full")
        self.teacher_rel = self.cfg.get("teacher_root", f"{self.root_rel}/teacher_out")
        self.parakeet_rel = self.cfg.get("parakeet_root") or f"{self.root_rel}/parakeet_out"
        self.second_rel = self.cfg.get("second_root", f"{self.root_rel}/second_out")
        self.data = self.kdir / self.data_rel
        self.root = self.kdir / self.root_rel
        self.teacher = self.kdir / self.teacher_rel
        self.parakeet = self.kdir / self.parakeet_rel
        self.second = self.kdir / self.second_rel
        self.state_path = self.state_dir / "label.json"
        self.procs: dict[str, Proc] = {}
        self.restart_at: dict[str, float] = {}
        self.cycle: dict | None = None
        self.golden: dict = {"proc": None, "tries": 0, "retry_at": 0.0}
        self.t_janitor = self.t_rates = float("-inf")
        self.t_sync = 0.0
        self.sync_fails = 0
        self.second_final_fails = 0
        self.pruned: set[str] | None = None
        self.ending = False
        self.state: dict = {}
        self._saved = None
        self.cores = int(self.env.get("KITSUNE_CORES") or 0) or effective_cores()
        self.cohere_procs = int(self.env.get("KITSUNE_COHERE_PROCS") or 0) or int(self.k["cohere"]["procs"])

    # ---- state, heartbeats, children

    def fresh_state(self) -> dict:
        return {"version": 1, "run_id": None, "phase": "boot", "steps_done": [], "lanes": {}, "sources": {},
                "syncs": [], "rates": {}, "finalize_done": [], "final": None, "lanes_started": False}

    def save(self, force: bool = False):
        blob = json.dumps(self.state, sort_keys=True)
        if force or blob != self._saved:
            supervise.save_state(self.state_path, self.state)
            self._saved = blob

    def beat(self):
        p = self.state_dir / "label_hb"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        now = self.clock.time()
        os.utime(p, (now, now))

    @contextlib.contextmanager
    def beating(self):
        """Keep label_hb fresh from a thread while a bounded blocking call runs (the final finish can take an hour;
        the watchdog's orphan rule fires after 15 min)."""
        stop = threading.Event()

        def run():
            while not stop.wait(BEAT_S):
                with contextlib.suppress(OSError):
                    self.beat()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        try:
            yield
        finally:
            stop.set()
            t.join(5)

    def child_env(self, offline: bool = False) -> dict:
        env = dict(self.env, OMP_NUM_THREADS="2", TQDM_MININTERVAL="60", KITSUNE_RUN_ID=self.state.get("run_id") or "")
        if offline:
            env["HF_HUB_OFFLINE"] = "1"
        return env

    def start(self, name: str, argv: list[str], env: dict | None = None) -> Proc:
        return self.runner.start(name, argv, env or self.child_env(), self.log_dir / f"lane_{name}.log")

    def run_bounded(self, name: str, argv: list[str], timeout: float, env: dict | None = None) -> int:
        """A child run to its end or its timeout (124), the controller heartbeat kept fresh meanwhile."""
        p = self.start(name, argv, env)
        t0 = self.clock.time()
        while True:
            rc = p.poll()
            if rc is not None:
                return rc
            if self.clock.time() - t0 > timeout:
                log(f"{name}: timed out after {timeout:.0f} s")
                p.terminate_group()
                return EXIT_TIMEOUT
            self.beat()
            self.clock.sleep(5)

    def wait(self, seconds: float):
        end = self.clock.time() + seconds
        while (left := end - self.clock.time()) > 0:
            self.beat()
            self.clock.sleep(min(BEAT_S, left))

    def now(self) -> float:
        return self.clock.time()

    def deadline(self) -> float:
        d = self.state_dir / "deadline"
        try:
            return float(d.read_text().strip())
        except (OSError, ValueError):
            if "deadline" not in self.state:  # no watchdog deadline (a local run): KITSUNE_MAX_HOURS from the start
                self.state["deadline"] = self.now() + float(self.env.get("KITSUNE_MAX_HOURS") or 30) * 3600
            return float(self.state["deadline"])

    # ---- entry

    def run(self) -> int:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        lock = supervise.acquire_lock(self.state_dir / "supervise.lock")  # held until this function returns
        if lock is None:
            log(f"another controller holds {self.state_dir / 'supervise.lock'}; not starting a second one")
            return 0
        try:
            return self._run()
        except Ended as e:
            return e.rc
        except Exception as e:  # a bug: best-effort final sync (inside finish --stop), then stop
            log(f"unexpected {type(e).__name__}: {e}")
            if self.ending:
                return 1
            try:
                self.end("unknown", repr(e)[:500])
            except Ended as e2:
                return e2.rc
            return 1

    def _run(self) -> int:
        st = supervise.load_state(self.state_path, required="phase")
        aside = st.pop("corrupt", None) if isinstance(st, dict) else None
        if not isinstance(st, dict) or st.get("phase") is None:  # none yet (supervise's fresh {"phase": None})
            st = dict(self.fresh_state(), final=st.get("final") if isinstance(st, dict) else None)
        self.state = st
        if aside:  # the step and lane history is unknown: a recorded stop (the disk stays for a human)
            self.end("unknown", f"label.json unreadable (moved to {aside}); state unknown")
        final = st.get("final")
        if final:
            log(f"a final decision is already recorded ({final}); not starting again")
            if not (self.state_dir / "halt").exists():
                log("no halt marker: the container restarted before finish acted on that decision; running it again")
                self.finish(final["action"], final.get("reason", ""), final.get("allow_empty", False))
            return 0
        if not st.get("run_id"):
            st["run_id"] = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(self.now()))}-" \
                           f"{self.env.get('CONTAINER_ID') or 'local'}"
        self.env["KITSUNE_RUN_ID"] = st["run_id"]
        self.save(force=True)
        log(f"run {st['run_id']}: config {self.cfg_path}, label configs {self.label_cfgs}, steps done "
            f"{st['steps_done']}")
        for name in BOOT_STEPS:
            if name in st["steps_done"] and name not in RERUN_STEPS:
                continue
            extra = ["--restart"] if name == "selftest" and ("selftest" in st["steps_done"] or self.work_on_disk())                 else []
            self.step(name, *extra)
        return self.loop()

    def work_on_disk(self) -> bool:
        """This root's work is already here (a stopped box resumed with onstart.sh --rearm moves label.json aside):
        selftest must then check the restart disk floor, not the fresh-box one."""
        # not data/manifest.jsonl: pull brings the eval gate shards (and their manifest) onto a fresh box too
        if (self.teacher / "meta.json").exists() or (self.parakeet / "meta.json").exists():
            return True
        try:
            led = json.loads((self.state_dir / "label_ledger.json").read_text(encoding="utf-8"))
            return isinstance(led, dict) and bool(led.get("remote") or led.get("hashes"))
        except (OSError, ValueError):
            return False

    def step(self, name: str, *extra: str, record: bool = True):
        argv = [self.py, "vast/label.py", "step", name, *extra]
        rc = None
        for attempt in range(len(STEP_RETRY_WAITS) + 1):
            rc = self.run_bounded(f"step-{name}", argv, STEP_TIMEOUT_S[name])
            if rc == EXIT_OK:
                if record and name not in self.state["steps_done"]:
                    self.state["steps_done"].append(name)
                self.save()
                return
            self.step_rc(name, rc)
            if attempt < len(STEP_RETRY_WAITS):
                log(f"step {name} exited {rc}; retrying in {STEP_RETRY_WAITS[attempt]} s")
                self.wait(STEP_RETRY_WAITS[attempt])
        self.end("host_failure", f"step {name} failed {len(STEP_RETRY_WAITS) + 1} times (last rc {rc})")

    def step_rc(self, name: str, rc: int):
        """End the box for a step's decisive exit codes; return for a transient one."""
        if rc == EXIT_REFUSAL:
            self.end("refusal", f"step {name} refused (rc 3)")
        if rc == EXIT_INTEGRITY:
            self.end("integrity", f"step {name}: integrity problem (rc 65)")
        if rc == EXIT_HOST:
            self.end("host_failure", f"step {name}: host failure (rc 69)")

    # ---- lanes

    def gpu_lanes(self) -> list[str]:
        return [f"cohere-{i}" for i in range(self.cohere_procs)] + ["parakeet"]

    def lane_argv(self, name: str) -> list[str]:
        st, k = self.state_dir, self.k
        common = ["--follow", str(st / "ingest.done"), "--heartbeat", str(st / "hb" / name),
                  "--progress", str(st / "lanes" / f"{name}.jsonl")]
        if name == "ingest":
            return [self.py, "scripts/01_prepare_data.py", "--data", self.data_rel, "--extent-config", self.cfg_path,
                    "--whisper-dir", f"{self.data_rel}/whisper", "--hold-file", str(st / "ingest.hold"),
                    "--heartbeat", str(st / "hb" / "ingest")]
        if name.startswith("cohere-"):
            c, w = k["cohere"], lane_workers(k, "cohere", self.cores)
            return [self.py, "scripts/02_teacher_pass.py", "--data", self.data_rel, "--out", self.teacher_rel,
                    "--split", "all", *common, "--shard-mod", str(self.cohere_procs), "--shard-rem", name.split("-")[1],
                    "--adopt-from", "seed/teacher_out", "--adopt-only", *GATE_SETS, "--strict-existing",
                    "--max-batch-seconds", str(c["max_batch_seconds"]), "--max-batch", str(c["max_batch"]),
                    "--workers", str(w), "--torch-threads", str(c["torch_threads"]), "--prefetch", str(2 * w),
                    "--vram-fraction", str(c["vram_fraction"])]
        if name == "parakeet":
            c, w = k["parakeet"], lane_workers(k, "parakeet", self.cores)
            return [self.py, "scripts/02p_parakeet_pass.py", "--data", self.data_rel, "--out", self.parakeet_rel,
                    "--model-dir", PARAKEET_PATH, "--split", "all", *common, "--strict-existing",
                    "--max-batch-seconds", str(c["max_batch_seconds"]), "--max-batch", str(c["max_batch"]),
                    "--workers", str(w), "--torch-threads", str(c["torch_threads"]), "--prefetch", str(2 * w),
                    "--vram-fraction", str(c["vram_fraction"]), "--k-tdt", str(c["k_tdt"]), "--k-ctc", str(c["k_ctc"]),
                    "--ctc-dense-thr", str(c["ctc_dense_thr"]), "--max-symbols", str(c["max_symbols"])]
        raise ValueError(name)

    def second_argv(self) -> list[str]:
        from kitsune import extent

        names = set(extent.names(self.cfg))
        sources = [s for s in SECOND_SOURCES if s in names]
        return [self.py, "scripts/02b_second_opinion.py", "--data", self.data_rel, "--teacher-out", self.teacher_rel,
                "--out", self.second_rel, "--judge", "parakeet", "--parakeet-out", self.parakeet_rel,
                "--whisper-dir", f"{self.data_rel}/whisper", "--require-npz", "--strict-existing",
                "--sources", *sources]

    def finish_argv(self, *args: str) -> list[str]:
        return [self.py, "vast/finish.py", "--job", "label", *args, *(["--dry-run"] if self.dry_run else [])]

    def progress_mark(self, name: str) -> int:
        p = self.data / "manifest.jsonl" if name == "ingest" else self.state_dir / "lanes" / f"{name}.jsonl"
        try:
            return p.stat().st_size
        except OSError:
            return 0

    def hang_s(self, name: str) -> float:
        return 60.0 * float(self.k["ingest_hang_min"] if name == "ingest" else self.k["hang_min"])

    def start_lane(self, name: str):
        now = self.now()
        rec = self.state["lanes"].setdefault(name, {"starts": 0, "fails_no_progress": 0, "last_progress_t": None,
                                                    "rc": None, "done": False})
        if name == "ingest":
            (self.state_dir / "ingest.done").unlink(missing_ok=True)
        hb = self.state_dir / "hb" / name
        hb.parent.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "lanes").mkdir(parents=True, exist_ok=True)
        hb.touch()
        os.utime(hb, (now, now))  # the hang clock starts now
        rec.update(starts=rec["starts"] + 1, rc=None, done=False, mark=self.progress_mark(name), t_start=now)
        self.procs[name] = self.start(name, self.lane_argv(name), self.child_env(offline=name != "ingest"))
        self.state["lanes_started"] = True
        self.save()

    def start_lanes(self, names):
        for n in names:
            if n not in self.procs and not self.state["lanes"].get(n, {}).get("done"):
                self.start_lane(n)

    def watch_lanes(self, now: float):
        for name in list(self.procs):
            p, rec = self.procs[name], self.state["lanes"][name]
            rc = p.poll()
            mark = self.progress_mark(name)
            if mark != rec.get("seen", rec["mark"]):
                rec["seen"], rec["last_progress_t"] = mark, now
            if rc is None:
                try:
                    age = now - (self.state_dir / "hb" / name).stat().st_mtime
                except OSError:
                    age = now - rec["t_start"]
                if age <= self.hang_s(name):
                    continue
                log(f"lane {name}: heartbeat {age / 60:.0f} min old; killing its process group")
                p.terminate_group()
                rc = "hang"
            del self.procs[name]
            rec["rc"] = rc
            if rc == 0:
                rec["done"] = True
                log(f"lane {name} finished")
                if name == "ingest":
                    (self.state_dir / "ingest.done").touch()
                continue
            if rc == EXIT_INTEGRITY:
                self.end("integrity", f"lane {name} exited 65 (see lane_{name}.log)")
            if rc == EXIT_REFUSAL:  # model files or meta that do not match: not the host's fault
                self.end("refusal", f"lane {name} exited 3 (see lane_{name}.log)")
            progressed = mark != rec["mark"]
            rec["fails_no_progress"] = 0 if progressed else rec["fails_no_progress"] + 1
            n = rec["fails_no_progress"]
            if n >= MAX_FAILS_NO_PROGRESS:
                self.end("host_failure", f"lane {name} failed {n} times in a row without progress (last: {rc})")
            delay = LANE_RESTART_S[min(max(n, 1), len(LANE_RESTART_S)) - 1]
            log(f"lane {name} exited {rc} ({'after progress' if progressed else f'{n} without progress'}); "
                f"restart in {delay} s")
            self.restart_at[name] = now + delay
        for name, t in list(self.restart_at.items()):
            if now >= t:
                del self.restart_at[name]
                self.start_lane(name)

    # ---- golden (inside the loop: it waits for eval_jsut/eval-00000 from the running ingest lane)

    def eval_jsut_ready(self) -> bool:
        from kitsune import store

        return any(s.source == "eval_jsut" and Path(s.path).stem == "eval-00000" for s in store.read_manifest(self.data))

    def golden_tick(self, now: float):
        g = self.golden
        if "golden" in self.state["steps_done"]:
            return
        if g["proc"] is None:
            if now < g["retry_at"] or not self.eval_jsut_ready():
                return
            g.update(proc=self.start("step-golden", [self.py, "vast/label.py", "step", "golden"]), t0=now,
                     tries=g["tries"] + 1)
            return
        rc = g["proc"].poll()
        if rc is None:
            if now - g["t0"] <= STEP_TIMEOUT_S["golden"]:
                return
            g["proc"].terminate_group()
            rc = EXIT_TIMEOUT
        g["proc"] = None
        if rc == EXIT_OK:
            self.state["steps_done"].append("golden")
            log("golden checks passed; starting the GPU lanes")
            self.start_lanes(self.gpu_lanes())
            return
        self.step_rc("golden", rc)
        if g["tries"] >= GOLDEN_TRIES:
            self.end("host_failure", f"step golden failed {g['tries']} times (last rc {rc})")
        g["retry_at"] = now + STEP_RETRY_WAITS[min(g["tries"], len(STEP_RETRY_WAITS)) - 1]

    # ---- janitor

    def teacher_done(self, npz: Path, ids: list[str]) -> bool:
        return teacher_done(npz, ids)

    def parakeet_done(self, npz: Path, ids: list[str]) -> bool:
        from kitsune import parakeet_targets

        p = self.k["parakeet"]
        return parakeet_targets.shard_done(npz, ids, {key: p[key] for key in PARAKEET_SETTING_KEYS})

    def disk_free(self) -> float:
        return shutil.disk_usage(self.kdir).free

    def pruned_paths(self) -> set[str]:
        if self.pruned is None:
            from kitsune import extent

            self.pruned = {r.get("path") for r in jsonl_rows(self.data / extent.PRUNED_FILE)}
        return self.pruned

    def janitor(self):
        from kitsune import extent, store

        pruned, n_checked, n_pruned, backlog = self.pruned_paths(), 0, 0, 0
        for sh in store.read_manifest(self.data):
            if sh.split != "train" or sh.path in pruned:  # eval shards are never pruned (3 GB)
                continue
            f = self.data / sh.path
            try:
                size = f.stat().st_size
            except OSError:
                continue
            stem = Path(sh.path).stem
            tn, pn = self.teacher / sh.source / f"{stem}.npz", self.parakeet / sh.source / f"{stem}.npz"
            if n_checked < JANITOR_MAX and tn.is_file() and pn.is_file():
                n_checked += 1
                ids = store.shard_ids(self.data, sh)
                if self.teacher_done(tn, ids) and self.parakeet_done(pn, ids):
                    f.unlink()
                    with open(self.data / extent.PRUNED_FILE, "a", encoding="utf-8") as g:
                        g.write(json.dumps({"path": sh.path, "bytes": size, "t": round(self.now())}) + "\n")
                    pruned.add(sh.path)
                    n_pruned += 1
                    continue
            backlog += size
        free = self.disk_free()
        hold = self.state_dir / "ingest.hold"
        limit = float(self.k["backlog_gb"])
        if backlog > limit * GB or free < HOLD_FREE_GB * GB:
            if not hold.exists():
                log(f"holding ingest: backlog {backlog / GB:.1f} GB, free {free / GB:.1f} GB")
            hold.touch()
        elif hold.exists() and backlog < (limit - RELEASE_BELOW_GB) * GB and free > RELEASE_FREE_GB * GB:
            log(f"releasing ingest: backlog {backlog / GB:.1f} GB, free {free / GB:.1f} GB")
            hold.unlink(missing_ok=True)
        self.state["disk"] = {"backlog_gb": round(backlog / GB, 2), "free_gb": round(free / GB, 2),
                              "held": hold.exists(), "pruned": len(pruned)}
        if n_pruned:
            log(f"janitor: pruned {n_pruned} shard(s); backlog {backlog / GB:.1f} GB, free {free / GB:.1f} GB")
        if hold.exists() and free < CRITICAL_FREE_GB * GB:
            self.end("host_failure", f"free disk {free / GB:.1f} GB < {CRITICAL_FREE_GB} GB with ingest held")

    # ---- sync cycles

    def cycle_tick(self, now: float):
        c = self.cycle
        if c is None:
            if now - self.t_sync >= 60.0 * float(self.k["sync_every_min"]):
                n = len(self.state["syncs"]) + 1
                self.t_sync = now
                # the sync runs first: it carries the lease heartbeat, which must not wait up to an hour behind 02b
                args = ["--sync-only"] + ([] if n % INFRA_EVERY == 0 else ["--no-infra"])
                self.cycle = {"stage": "sync", "t0": now, "n": n, "infra": n % INFRA_EVERY == 0,
                              "proc": self.start("sync", self.finish_argv(*args))}
            return
        rc = c["proc"].poll()
        if rc is None:
            if now - c["t0"] <= (SECOND_TIMEOUT_S if c["stage"] == "02b" else SYNC_TIMEOUT_S):
                return
            log(f"{c['stage']} timed out")
            c["proc"].terminate_group()
            rc = EXIT_TIMEOUT
        if c["stage"] == "sync":
            if rc == EXIT_INTEGRITY:
                self.end("integrity", "label sync refused a write-once conflict (see lane_sync.log)")
            c.update(stage="02b", t0=now, sync_rc=rc,
                     proc=self.start("second", self.second_argv(), self.child_env(offline=True)))
            return
        if rc == EXIT_INTEGRITY:
            self.end("integrity", "02b exited 65 (see lane_second.log)")
        if rc != 0:
            log(f"02b exited {rc}; retried next cycle")
        rc, second_rc = c["sync_rc"], rc
        self.state["syncs"].append({"t": round(now), "rc": rc, "second_rc": second_rc, "infra": c["infra"]})
        self.cycle = None
        self.sync_fails = 0 if rc == 0 else self.sync_fails + 1
        if self.sync_fails >= LOUD_SYNC_FAILS:
            log(f"WARNING: {self.sync_fails} sync cycles in a row failed (last rc {rc}); the labels stay on disk")

    # ---- rates

    def lane_rates(self) -> dict:
        out = {}
        for f in sorted((self.state_dir / "lanes").glob("*.jsonl")):
            h = w = 0.0
            n = 0
            for r in jsonl_rows(f):
                if r.get("adopted"):
                    continue
                h += float(r.get("hours") or 0)
                w += float(r.get("wall_s") or 0)
                n += 1
            out[f.stem] = {"shards": n, "hours": round(h, 3), "wall_s": round(w, 1),
                           "xrt": round(h * 3600 / w, 1) if w > 0 else None}
        return out

    def remaining_hours(self) -> dict:
        from kitsune import store

        rem = {"cohere": 0.0, "parakeet": 0.0}
        for sh in store.read_manifest(self.data):
            stem = Path(sh.path).stem
            if not (self.teacher / sh.source / f"{stem}.npz").is_file():
                rem["cohere"] += sh.hours
            if not (self.parakeet / sh.source / f"{stem}.npz").is_file():
                rem["parakeet"] += sh.hours
        return rem

    def rates(self, now: float):
        lanes = self.lane_rates()
        cohere = [v for n, v in lanes.items() if n.startswith("cohere")]
        cohere_xrt = sum(v["xrt"] or 0 for v in cohere)
        cohere_wall = max((v["wall_s"] for v in cohere), default=0.0)
        para_xrt = sum(v["xrt"] or 0 for n, v in lanes.items() if n == "parakeet")
        rem = self.remaining_hours()
        eta = max(rem["cohere"] / cohere_xrt if cohere_xrt else float("inf"),
                  rem["parakeet"] / para_xrt if para_xrt else float("inf"))
        left_h = (self.deadline() - now) / 3600
        self.state["rates"] = {"t": round(now), "lanes": lanes, "cohere_xrt": cohere_xrt, "parakeet_xrt": para_xrt,
                               "remaining_h": {k: round(v, 1) for k, v in rem.items()},
                               "eta_h": None if eta == float("inf") else round(eta, 2),
                               "deadline_in_h": round(left_h, 2)}
        log(f"rates: cohere {cohere_xrt:.0f}x, parakeet {para_xrt:.0f}x realtime; remaining (ingested only) "
            f"{rem['cohere']:.0f} h Cohere, {rem['parakeet']:.0f} h Parakeet; ETA {eta:.1f} h, deadline in "
            f"{left_h:.1f} h")
        min_xrt = float(self.k["min_xrt"])
        if cohere_wall >= SLOW_AFTER_S and cohere_xrt < min_xrt:
            self.end("slow_host", f"Cohere {cohere_xrt:.0f}x realtime < {min_xrt:.0f}x after "
                                  f"{cohere_wall / 60:.0f} min of Cohere work")

    # ---- the loop

    def tick(self):
        now = self.now()
        if now >= self.deadline() - 60.0 * float(self.k["end_margin_min"]):
            self.end("budget", f"deadline - {self.k['end_margin_min']} min reached with work left")
        self.watch_lanes(now)
        self.golden_tick(now)
        if now - self.t_janitor >= JANITOR_S:
            self.t_janitor = now
            self.janitor()
        self.cycle_tick(now)
        if now - self.t_rates >= RATES_S:
            self.t_rates = now
            self.rates(now)

    def loop(self) -> int:
        self.state["phase"] = "label"
        self.t_sync = self.now() - 60.0 * float(self.k["sync_every_min"])  # the first cycle (and lease beat) at once
        self.deadline()
        self.start_lanes(["ingest"])
        if "golden" in self.state["steps_done"]:
            self.start_lanes(self.gpu_lanes())
        while True:
            self.beat()
            self.tick()
            if self.finalize_ready():
                if self.now() > self.deadline() - 60.0 * float(self.k["finalize_margin_min"]):
                    self.end("budget", f"lanes finished after deadline - {self.k['finalize_margin_min']} min: "
                                       f"too late to finalize")
                return self.finalize()
            self.save()
            self.clock.sleep(LOOP_S)

    def unlabelled(self) -> list[str]:
        from kitsune import store

        out = []
        for sh in store.read_manifest(self.data):
            stem = Path(sh.path).stem
            for root in (self.teacher, self.parakeet):
                if not (root / sh.source / f"{stem}.npz").is_file():
                    out.append(f"{root.name}/{sh.source}/{stem}")
        return out

    def missing_second(self) -> list[str]:
        from kitsune import store

        srcs = set(self.second_argv()[self.second_argv().index("--sources") + 1:])
        return [f"{sh.source}/{Path(sh.path).stem}" for sh in store.read_manifest(self.data)
                if sh.source in srcs and not (self.second / sh.source / f"{Path(sh.path).stem}.jsonl").is_file()]

    def finalize_ready(self) -> bool:
        if not (self.state_dir / "ingest.done").exists() or "golden" not in self.state["steps_done"]:
            return False
        lanes = self.state["lanes"]
        if self.procs or self.restart_at or not all(lanes.get(n, {}).get("done") for n in self.gpu_lanes()):
            return False
        if self.cycle is not None:  # let the running sync cycle end first
            return False
        if missing := self.unlabelled():
            self.end("integrity", f"{len(missing)} shard(s) unlabelled after every lane finished: {missing[:5]}")
        if not self.state.get("second_final_ok"):
            rc = self.run_bounded("second", self.second_argv(), SECOND_TIMEOUT_S, self.child_env(offline=True))
            if rc == EXIT_INTEGRITY:
                self.end("integrity", "final 02b exited 65")
            if rc != 0:
                self.second_final_fails += 1
                if self.second_final_fails >= 3:
                    self.end("unknown", f"the final 02b failed 3 times (last rc {rc})")
                return False
            self.state["second_final_ok"] = True
            self.save()
        if missing := self.missing_second():
            self.end("integrity", f"{len(missing)} shard(s) without a second opinion after the final 02b: "
                                  f"{missing[:5]}")
        return True

    # ---- finalize F1-F8

    def finalize(self) -> int:
        self.state["phase"] = "finalize"
        self.save()
        done = self.state["finalize_done"]

        def sub(name, fn):
            if name in done:
                return
            log(f"finalize {name}")
            with self.beating():  # build_record / labels_report read millions of rows: keep label_hb fresh
                fn()
            done.append(name)
            self.save()

        sub("F1-extent", self.f_extent)
        sub("F2-selections", self.f_selections)
        sub("F3-reports", self.f_reports)
        sub("F4-lease", self.f_lease)
        sub("F4-consumer-check", lambda: self.step("consumer-check", record=False))
        sub("F5-final-sync", self.f_final_sync)
        sub("F6-consumer-check-hub", lambda: self.step("consumer-check", "--hub", record=False))
        sub("F7-seal", lambda: self.step("seal", record=False))
        self.end("success", "labels complete")
        return 0  # not reached: end() raises

    def run_ids(self) -> list[str]:
        return [self.state["run_id"]]

    def f_extent(self):
        from kitsune import extent

        try:
            record = extent.build_record(self.data, self.cfg, run_ids=self.run_ids(),
                                         kitsune_sha=self.env.get("KITSUNE_SHA") or "unknown")
        except ValueError as e:
            self.end("integrity", f"extent record: {e}")
        extent.write_record(self.root / extent.RECORD_FILE, record)

    def f_selections(self):
        main_sel = self.cfg["selection"]
        runs = [["--config", self.cfg_path]]
        runs += [["--config", c, "--from-selection", main_sel] for c in self.label_cfgs if c != self.cfg_path]
        for args in runs:
            rc = self.run_bounded("make_selection", [self.py, "scripts/make_selection.py", *args],
                                  MAKE_SELECTION_TIMEOUT_S, self.child_env(offline=True))
            if rc != 0:
                self.end("integrity", f"make_selection {' '.join(args)} exited {rc}")

    def f_reports(self):
        reports = self.root / "reports"
        rc = self.run_bounded("judge_report", self.judge_argv(), MAKE_SELECTION_TIMEOUT_S, self.child_env(offline=True))
        if rc != 0:  # a report, not a label: logged in labels.json, the run still seals
            log(f"tools/judge_report.py exited {rc}; reports/galgame_judge.json is missing")
        write_json_atomic(reports / "parakeet_baselines.json", self.parakeet_baselines())
        write_json_atomic(reports / "labels.json", dict(self.labels_report(), judge_report_rc=rc))
        write_json_atomic(reports / "throughput.json", {"rates": self.lane_rates(), "last": self.state.get("rates"),
                                                        "syncs": self.state["syncs"], "disk": self.state.get("disk")})
        prov = self.root / "provenance"
        write_json_atomic(prov / f"{self.state['run_id']}.json", self.provenance())
        write_json_atomic(prov / "adopted.json", self.adopted())

    def judge_argv(self) -> list[str]:
        recipe = self.cfg.get("selection_recipe") or {}
        thr = float(recipe.get("agree_max", 0.5))
        for item in recipe.get("agree_max_source") or []:
            src, _, val = str(item).partition("=")
            if src == "galgame":
                thr = float(val)
        return [self.py, "tools/judge_report.py", "--kotoba", "seed/second_out/galgame",
                "--parakeet", f"{self.second_rel}/galgame", "--teacher", f"{self.teacher_rel}/galgame",
                "--out", f"{self.root_rel}/reports/galgame_judge.json", "--threshold", str(thr)]

    def parakeet_baselines(self) -> dict:
        out = {}
        for s in self.cfg.get("eval_sets", []):
            rows = [r for f in sorted((self.parakeet / s).glob("eval-*.jsonl")) for r in jsonl_rows(f)]
            if not rows:
                continue
            refs = [r.get("ref") or "" for r in rows]
            out[s] = {"n": len(rows), "tdt_cer_corpus": corpus_cer([r.get("hyp") or "" for r in rows], refs),
                      "ctc_cer_corpus": corpus_cer([r.get("ctc_hyp") or "" for r in rows], refs)}
        try:
            from kitsune.evaluate import teacher_baselines

            cohere = teacher_baselines(self.teacher, [s for s in GATE_SETS if (self.teacher / s).is_dir()],
                                       check=False)
        except Exception as e:  # the consumer check runs it with check=True
            cohere = {"error": f"{type(e).__name__}: {e}"}
        return {"parakeet": out, "cohere": cohere}

    def labels_report(self) -> dict:
        from kitsune import store

        per: dict[str, dict] = {}
        for sh in store.read_manifest(self.data):
            d = per.setdefault(sh.source, {"shards": 0, "rows": 0, "hours": 0.0, "teacher_rows": 0,
                                           "teacher_truncated": 0, "parakeet_truncated": 0, "agree": [],
                                           "fallback": 0})
            stem = Path(sh.path).stem
            d["shards"] += 1
            d["rows"] += sh.rows
            d["hours"] += sh.hours
            t = jsonl_rows(self.teacher / sh.source / f"{stem}.jsonl")
            d["teacher_rows"] += len(t)
            d["teacher_truncated"] += sum(1 for r in t if r.get("truncated"))
            d["parakeet_truncated"] += sum(1 for r in jsonl_rows(self.parakeet / sh.source / f"{stem}.jsonl")
                                           if r.get("truncated"))
            for r in jsonl_rows(self.second / sh.source / f"{stem}.jsonl"):
                if r.get("agree") is not None:
                    d["agree"].append(float(r["agree"]))
                if str(r.get("model2", "")).startswith("parakeet-fallback"):
                    d["fallback"] += 1
        for d in per.values():
            a = sorted(d.pop("agree"))
            d["agree_quantiles"] = {q: a[min(len(a) - 1, int(q * len(a)))] for q in (0.1, 0.25, 0.5, 0.75, 0.9)} \
                if a else {}
            d["hours"] = round(d["hours"], 3)
        return {"sources": per}

    def provenance(self) -> dict:
        return {"run_id": self.state["run_id"], "kitsune_sha": self.env.get("KITSUNE_SHA"),
                "image": self.env.get("KITSUNE_IMAGE"), "machine_id": self.env.get("KITSUNE_MACHINE_ID"),
                "container_id": self.env.get("CONTAINER_ID"), "data_revision": self.env.get("KITSUNE_DATA_REVISION"),
                "config": self.cfg_path, "label_configs": self.label_cfgs, "settings": self.k,
                "cohere_procs": self.cohere_procs, "cores": self.cores,
                "selftest": read_json(self.state_dir / "selftest.json"),
                "golden": read_json(self.state_dir / "golden.json"),
                "timings": {"steps_done": self.state["steps_done"], "lanes": self.state["lanes"],
                            "syncs": len(self.state["syncs"])}}

    def adopted(self) -> dict:
        rows = []
        rev = self.env.get("KITSUNE_DATA_REVISION")
        for f in sorted((self.state_dir / "lanes").glob("cohere-*.jsonl")):
            for r in jsonl_rows(f):
                if not r.get("adopted"):
                    continue
                npz = self.teacher / r["source"] / f"{r['stem']}.npz"
                rows.append({"stem": f"{r['source']}/{r['stem']}", "laptop_path": f"teacher_out/{r['source']}/"
                             f"{r['stem']}.npz", "sha256": sha256_file(npz) if npz.is_file() else None,
                             "data_revision": rev})
        return {"data_revision": rev, "adopted": sorted(rows, key=lambda r: r["stem"])}

    def f_lease(self):
        """A sync (it carries the lease heartbeat) right before the long consumer check; a failure is not fatal here,
        F5 syncs again, but a write-once refusal is."""
        rc = self.run_bounded("sync", self.finish_argv("--sync-only", "--no-infra"), SYNC_TIMEOUT_S)
        if rc == EXIT_INTEGRITY:
            self.end("integrity", "label sync refused a write-once conflict (see lane_sync.log)")

    def f_final_sync(self):
        for attempt in range(2):
            rc = self.run_bounded("sync", self.finish_argv("--sync-only"), SYNC_TIMEOUT_S)
            if rc == EXIT_INTEGRITY:
                self.end("integrity", "the final sync refused a write-once conflict (see lane_sync.log)")
            vrc = self.run_bounded("verify", self.finish_argv("--verify-only"), VERIFY_TIMEOUT_S) if rc == 0 else None
            if rc == 0 and vrc == 0:
                return
            log(f"final sync rc {rc}, verify rc {vrc}" + ("; one more sync" if attempt == 0 else ""))
        self.end("sync_unverifiable", "the final sync could not be verified on the Hub")

    # ---- endings

    def stop_children(self):
        procs = list(self.procs.values())
        for extra in (self.cycle, self.golden):
            if extra and extra.get("proc") is not None:
                procs.append(extra["proc"])
        for p in procs:
            with contextlib.suppress(Exception):
                p.terminate_group()
        self.procs.clear()
        self.restart_at.clear()

    def finish(self, action: str, reason: str, allow_empty: bool = False, no_sync: bool = False):
        extra = ["--job", "label"] + (["--allow-empty"] if allow_empty else []) + (["--no-sync"] if no_sync else [])             + (["--dry-run"] if self.dry_run else [])
        with self.beating():
            self._final_finish(action, reason, extra)

    def end(self, cls: str, reason: str):
        """Record the end, SIGTERM the lanes, run finish.py (destroy or stop), then raise Ended."""
        if self.ending:
            raise Ended(1)
        self.ending = True
        action = "destroy" if cls in DESTROY_CLASSES else "stop"
        if action == "destroy" and self.env.get("KITSUNE_LABEL_END", "destroy") == "stop":
            action = "stop"
        allow_empty = not self.state.get("lanes_started")
        now = self.now()
        log(f"end: {cls} ({reason}) -> {action}")
        write_json_atomic(self.state_dir / "label_end.json",
                          {"class": cls, "reason": reason, "machine_id": self.env.get("KITSUNE_MACHINE_ID"),
                           "run_id": self.state.get("run_id"), "wall": now})
        if self.state:
            self.state["final"] = {"action": action, "class": cls, "reason": reason, "wall": now,
                                   "allow_empty": allow_empty}
            self.state["phase"] = "end"
            self.save(force=True)
        with self.beating():
            self.stop_children()
        # before plan took the lease this box has nothing unique on disk and must not write the root (a live lease of
        # another box, or a repo it refused); finish.py still pushes infra
        no_sync = "plan" not in (self.state or {}).get("steps_done", [])
        self.finish(action, f"{cls}: {reason}", allow_empty, no_sync)
        raise Ended(0 if cls == "success" else 1)


# ------------------------------------------------------------------------------------------------------------ steps


class StepCtx:
    """What a step needs, injectable for tests: env, the Hub api, a bounded subprocess runner."""

    def __init__(self, env: dict | None = None, api=None, run=None, now=time.time):
        self.env = dict(os.environ if env is None else env)
        self.kdir = Path(self.env.get("KITSUNE_DIR") or ROOT)
        self.state_dir = Path(self.env.get("KITSUNE_STATE") or "/workspace/kitsune_state")
        self.cfg_path = self.env.get("KITSUNE_CONFIG") or "configs/full.json"
        self.cfg = json.loads((self.kdir / self.cfg_path).read_text(encoding="utf-8"))
        self.label_cfgs = [c for c in (self.env.get("KITSUNE_LABEL_CONFIGS") or self.cfg_path).split(",") if c]
        self.k = knobs(self.cfg)
        self.root_rel = (self.cfg.get("extent") or {}).get("root", "labels/full")
        self.root = self.kdir / self.root_rel
        self.data = self.kdir / self.cfg.get("data_root", "data")
        self.teacher = self.kdir / self.cfg.get("teacher_root", f"{self.root_rel}/teacher_out")
        self.parakeet = self.kdir / (self.cfg.get("parakeet_root") or f"{self.root_rel}/parakeet_out")
        self.second = self.kdir / self.cfg.get("second_root", f"{self.root_rel}/second_out")
        self._api = api
        self.run = run or self._run
        self.now = now

    @property
    def api(self):
        if self._api is None:
            from finish import hf_api

            self._api = hf_api()
        return self._api

    @property
    def repo(self) -> str:
        repo = self.env.get("KITSUNE_DATA_REPO")
        if not repo:
            raise Refusal("KITSUNE_DATA_REPO is not set")
        return repo

    @property
    def revision(self) -> str:
        rev = self.env.get("KITSUNE_DATA_REVISION")
        if not rev:
            raise Refusal("KITSUNE_DATA_REVISION is not set")
        return rev

    def _run(self, argv: list[str], timeout: float, env: dict | None = None) -> int:
        log(f"run: {' '.join(map(str, argv))}")
        try:
            return subprocess.run([str(a) for a in argv], cwd=self.kdir, env=env or self.env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return EXIT_TIMEOUT


def guard(fn, what: str):
    try:
        return fn()
    except Exception as e:
        raise refusal_from(e, what) from e


def listing_entry(f) -> list:
    lfs = getattr(f, "lfs", None)
    sha = getattr(lfs, "sha256", None) if lfs is not None else None
    if sha is None and isinstance(lfs, dict):
        sha = lfs.get("sha256")
    return [getattr(f, "size", None), sha, getattr(f, "blob_id", None)]


def tree(api, repo: str, path: str, revision: str | None = None) -> dict[str, list]:
    """path -> [size, lfs sha256 or None, blob id] for every file under `path` (a missing folder is empty)."""
    kw = {"revision": revision} if revision else {}
    try:
        items = list(api.list_repo_tree(repo, path_in_repo=path, recursive=True, repo_type="dataset", **kw))
    except Exception as e:
        if type(e).__name__ == "EntryNotFoundError" or http_status(e) == 404:
            return {}
        raise
    return {f.path: listing_entry(f) for f in items if getattr(f, "size", None) is not None}


def download_json(ctx: StepCtx, path: str, revision: str | None = None) -> dict:
    kw = {"revision": revision} if revision else {}
    local = ctx.api.hf_hub_download(repo_id=ctx.repo, filename=path, repo_type="dataset", **kw)
    return json.loads(Path(local).read_text(encoding="utf-8"))


def step_plan(ctx: StepCtx):
    from label_sync import lease_bytes, lease_live

    api, repo, rev, root = ctx.api, ctx.repo, ctx.revision, ctx.root_rel
    guard(api.whoami, "whoami (the box's HF token)")
    guard(lambda: api.auth_check(repo, repo_type="dataset", write=True), f"the token cannot write {repo}")
    info = guard(lambda: api.dataset_info(repo), f"dataset_info {repo}")
    if getattr(info, "private", None) is not True:
        raise Refusal(f"{repo} is not private")
    guard(lambda: api.auth_check(COHERE_MODEL_ID, repo_type="model"), f"no read access to gated {COHERE_MODEL_ID}")
    listing = tree(api, repo, root)
    write_json_atomic(ctx.state_dir / "label_remote.json", listing)
    if f"{root}/COMPLETE.json" in listing:
        raise Refusal(f"{root}/COMPLETE.json exists: this root is sealed")
    cid = ctx.env.get("CONTAINER_ID") or "local"
    if f"{root}/LEASE.json" in listing:
        lease = download_json(ctx, f"{root}/LEASE.json")
        if lease_live(lease, cid, ctx.now()) and ctx.env.get("KITSUNE_STEAL_LEASE", "0") != "1":
            raise Refusal(f"{root}/LEASE.json is live for another box ({lease}); set KITSUNE_STEAL_LEASE=1 to take it")
    seeds = tree(api, repo, "teacher_out", revision=rev)
    problems = [] if "teacher_out/meta.json" in seeds else ["teacher_out/meta.json"]
    for g in GATE_SETS:
        npz = [p for p in seeds if p.startswith(f"teacher_out/{g}/") and p.endswith(".npz")]
        if not npz:
            problems.append(f"teacher_out/{g}/*.npz")
        problems += [p[:-4] + ".jsonl" for p in npz if p[:-4] + ".jsonl" not in seeds]
    if problems:
        raise Refusal(f"laptop seed files missing at {rev}: {problems[:10]}")
    from kitsune.parakeet import PARAKEET_FILES

    models = tree(api, repo, PARAKEET_PATH, revision=rev)
    for name, sha in PARAKEET_FILES.items():
        e = models.get(f"{PARAKEET_PATH}/{name}")
        if e is None:
            problems.append(f"{PARAKEET_PATH}/{name} missing")
        elif e[1] is not None and e[1] != sha:
            problems.append(f"{PARAKEET_PATH}/{name}: sha256 {e[1]} != pinned {sha}")
    if problems:
        raise Refusal(f"Parakeet model files at {rev}: {problems}")
    if f"{root}/parakeet_out/meta.json" in listing:  # a relaunch: the pulled labels must match this run's settings
        meta = download_json(ctx, f"{root}/parakeet_out/meta.json")
        p = ctx.k["parakeet"]
        diff = {key: (meta.get(key), p[key]) for key in PARAKEET_SETTING_KEYS if meta.get(key) != p[key]}
        if meta.get("files_sha256") not in (None, PARAKEET_FILES):
            diff["files_sha256"] = "differs"
        if diff:
            raise Refusal(f"{root}/parakeet_out/meta.json has other settings: {diff}")
    if f"{root}/teacher_out/meta.json" in listing:
        have, seed = download_json(ctx, f"{root}/teacher_out/meta.json"), download_json(ctx, "teacher_out/meta.json",
                                                                                       revision=rev)
        seed.setdefault("model_revision", COHERE_REVISION)
        diff = {key: (have.get(key), seed.get(key)) for key in TEACHER_SETTING_KEYS if have.get(key) != seed.get(key)}
        if diff:
            raise Refusal(f"{root}/teacher_out/meta.json has other settings than the seed: {diff}")
    from huggingface_hub import CommitOperationAdd

    data = lease_bytes(ctx.env.get("KITSUNE_RUN_ID") or "unknown", cid, ctx.env.get("KITSUNE_MACHINE_ID") or "",
                       ctx.env.get("KITSUNE_SHA") or "", released=False)
    api.create_commit(repo, repo_type="dataset", commit_message=f"label box lease ({cid})",
                      operations=[CommitOperationAdd(path_in_repo=f"{root}/LEASE.json", path_or_fileobj=data)])
    log(f"plan ok: {len(listing)} label file(s) already under {root}; lease written")


def step_pull(ctx: StepCtx):
    api, repo, rev = ctx.api, ctx.repo, ctx.revision
    listing = read_json(ctx.state_dir / "label_remote.json", {})
    lease = f"{ctx.root_rel}/LEASE.json"
    todo, bad = [], []
    for path, (size, _sha, _blob) in listing.items():
        if path == lease:
            continue
        local = ctx.kdir / path
        if local.is_file():
            if size is not None and local.stat().st_size != size:
                bad.append(f"{path}: local {local.stat().st_size} B != remote {size} B")
        else:
            todo.append(path)
    if bad:
        raise Integrity(f"local label files differ from the Hub: {bad[:10]}")

    def get(path):
        api.hf_hub_download(repo_id=repo, filename=path, repo_type="dataset", local_dir=str(ctx.kdir))
        size = listing[path][0]
        if size is not None and (ctx.kdir / path).stat().st_size != size:
            raise Integrity(f"{path}: downloaded size differs from the listing")

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(get, todo))
    log(f"pull: {len(todo)} label file(s) fetched, {len(listing) - len(todo)} already local")
    seeds = ["teacher_out/meta.json", "second_out/galgame/*"] + [f"teacher_out/{s}/*" for s in SEED_SOURCES]
    api.snapshot_download(repo_id=repo, repo_type="dataset", revision=rev, local_dir=str(ctx.kdir / "seed"),
                          allow_patterns=seeds)
    api.snapshot_download(repo_id=repo, repo_type="dataset", revision=rev, local_dir=str(ctx.kdir),
                          allow_patterns=[f"{PARAKEET_PATH}/*"])
    from label_sync import load_ledger, save_ledger

    ledger_path = ctx.state_dir / "label_ledger.json"
    ledger = load_ledger(ledger_path)
    ledger["remote"].update({p: e for p, e in listing.items() if p != lease})
    save_ledger(ledger_path, ledger)


def step_models(ctx: StepCtx):
    api = ctx.api
    api.snapshot_download(repo_id=COHERE_MODEL_ID, revision=COHERE_REVISION)
    api.snapshot_download(repo_id=WHISPER_TOK_ID, revision=WHISPER_TOK_REVISION, allow_patterns=["*.json", "*.txt"])
    from kitsune.parakeet import verify_model_dir

    if problems := verify_model_dir(ctx.kdir / PARAKEET_PATH):
        raise Refusal(f"Parakeet model dir: {problems}")


def step_selftest(ctx: StepCtx, restart: bool = False):
    res, problems = {"restart": restart}, []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
        name, driver, mem = [x.strip() for x in out.strip().splitlines()[0].split(",")[:3]]
        res.update(gpu=name, driver=driver, memory_mib=float(mem))
        if "5090" not in name:
            problems.append(f"GPU {name} is not an RTX 5090")
        if int(driver.split(".")[0]) < 580:
            problems.append(f"driver {driver} < 580")
        if float(mem) < 30 * 1024:
            problems.append(f"GPU memory {mem} MiB < 30 GiB")
    except Exception as e:
        problems.append(f"nvidia-smi: {type(e).__name__}: {e}")
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        cap = "sm_%d%d" % torch.cuda.get_device_capability()
        res.update(torch=torch.__version__, cuda=torch.version.cuda, arch=cap)
        if cap not in torch.cuda.get_arch_list():
            problems.append(f"{cap} not in torch's arch list {torch.cuda.get_arch_list()}")
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        ref = a @ b
        err = ((a.bfloat16() @ b.bfloat16()).float() - ref).norm() / ref.norm()
        res["bf16_matmul_rel_err"] = float(err)
        if not err < 1e-2:
            problems.append(f"bf16 matmul rel. error {float(err):.3g}")
        q = torch.randn(2, 8, 256, 64, device="cuda", dtype=torch.bfloat16)
        if not torch.isfinite(torch.nn.functional.scaled_dot_product_attention(q, q, q)).all():
            problems.append("bf16 SDPA not finite")
        lstm = torch.nn.LSTM(640, 640, 2, batch_first=True).cuda()
        if not torch.isfinite(lstm(torch.randn(4, 16, 640, device="cuda"))[0]).all():
            problems.append("fp32 cuDNN LSTM not finite")
    except Exception as e:
        problems.append(f"torch/CUDA: {type(e).__name__}: {e}")
    try:
        import soundfile

        res["libsndfile"] = soundfile.__libsndfile_version__
        if soundfile.__libsndfile_version__ != "1.2.2":
            problems.append(f"libsndfile {soundfile.__libsndfile_version__} != 1.2.2 (durations drive the filters)")
    except Exception as e:
        problems.append(f"soundfile: {e}")
    free = shutil.disk_usage(ctx.kdir).free / GB
    res["free_gb"] = round(free, 1)
    need = 30 if restart else 150
    if free < need:
        problems.append(f"free disk {free:.0f} GB < {need} GB")
    cores = effective_cores()
    res["cores"] = cores
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB
        res["ram_gb"] = round(ram, 1)
    except (AttributeError, ValueError, OSError):
        ram = None
    warnings = [w for w, bad in ((f"{cores} effective cores < 16", cores < 16),
                                 (f"RAM {ram} GB < 48", ram is not None and ram < 48)) if bad]
    res.update(problems=problems, warnings=warnings)
    write_json_atomic(ctx.state_dir / "selftest.json", res)
    for w in warnings:
        log(f"selftest warning: {w}")
    if problems:
        raise HostFailure(f"selftest: {problems}")


def step_roots(ctx: StepCtx):
    seed = read_json(ctx.kdir / "seed" / "teacher_out" / "meta.json")
    if seed is None:
        raise Refusal("seed/teacher_out/meta.json is missing (step pull)")
    want = dict(seed, model_revision=seed.get("model_revision", COHERE_REVISION),
                adopted_from=f"teacher_out/meta.json@{ctx.env.get('KITSUNE_DATA_REVISION', '')}")
    dst = ctx.teacher / "meta.json"
    if dst.is_file():
        have = read_json(dst, {})
        diff = {key: (have.get(key), want.get(key)) for key in TEACHER_SETTING_KEYS if have.get(key) != want.get(key)}
        if diff:
            raise Integrity(f"{dst} was written with other settings: {diff}")
        return
    write_json_atomic(dst, want)


def golden_reference(obj) -> dict[str, dict]:
    """kitsune/parakeet_golden.json ({"ids", "tdt", "ctc"} parallel lists) as id -> {"hyp", "ctc_hyp"}; also takes
    {"rows": [{id, hyp, ctc_hyp}]} and {id: {tdt, ctc}}."""
    if isinstance(obj, dict) and isinstance(obj.get("ids"), list):
        return {i: {"hyp": t, "ctc_hyp": c} for i, t, c in zip(obj["ids"], obj.get("tdt") or [None] * len(obj["ids"]),
                                                               obj.get("ctc") or [None] * len(obj["ids"]))}
    rows = obj.get("rows") if isinstance(obj, dict) and isinstance(obj.get("rows"), list) else None
    if rows is not None:
        return {r["id"]: {"hyp": r.get("hyp", r.get("tdt")), "ctc_hyp": r.get("ctc_hyp", r.get("ctc"))} for r in rows}
    return {i: {"hyp": v.get("hyp", v.get("tdt")), "ctc_hyp": v.get("ctc_hyp", v.get("ctc"))}
            for i, v in obj.items() if isinstance(v, dict)}


def step_golden(ctx: StepCtx):
    calib = Path(ctx.env.get("KITSUNE_CALIB") or "/workspace/calib")
    shutil.rmtree(calib, ignore_errors=True)
    calib.mkdir(parents=True)
    env = dict(ctx.env, HF_HUB_OFFLINE="1")
    data = ctx.cfg.get("data_root", "data")
    py = sys.executable
    res = {}
    rc = ctx.run([py, "scripts/02_teacher_pass.py", "--data", data, "--sources", "eval_jsut", "--limit-shards", "1",
                  "--limit-rows", "64", "--max-batch-seconds", "1200", "--max-batch", "256",
                  "--out", str(calib / "teacher")], 3600, env)
    if rc != 0:
        raise HostFailure(f"golden Cohere run exited {rc}")
    box = jsonl_rows(calib / "teacher" / "eval_jsut" / "eval-00000.jsonl")
    seed = {r["id"]: r["hyp"] for r in jsonl_rows(ctx.kdir / "seed" / "teacher_out" / "eval_jsut" / "eval-00000.jsonl")}
    cer, n = golden_cer(box, seed)
    res["cohere"] = {"cer": cer, "n": n, "tol": 0.01}
    rc = ctx.run([py, "scripts/02p_parakeet_pass.py", "--data", data, "--sources", "eval_jsut", "--limit-shards", "1",
                  "--limit-rows", "32", "--model-dir", PARAKEET_PATH, "--out", str(calib / "parakeet")], 3600, env)
    if rc != 0:
        raise HostFailure(f"golden Parakeet run exited {rc}")
    ref = golden_reference(read_json(ROOT / "kitsune" / "parakeet_golden.json", {}))
    pbox = jsonl_rows(calib / "parakeet" / "eval_jsut" / "eval-00000.jsonl")
    tdt, n_t = golden_cer(pbox, {i: v["hyp"] for i, v in ref.items() if v["hyp"] is not None})
    ctc, n_c = golden_cer(pbox, {i: v["ctc_hyp"] for i, v in ref.items() if v["ctc_hyp"] is not None}, "ctc_hyp")
    res["parakeet"] = {"tdt_cer": tdt, "ctc_cer": ctc, "n": n_t, "n_ctc": n_c, "tol": 0.02}
    problems = []
    if not cer <= 0.01:
        problems.append(f"Cohere CER {cer:.4f} vs the laptop hypotheses on {n} rows > 0.01")
    if not tdt <= 0.02 or not ctc <= 0.02:
        problems.append(f"Parakeet CER TDT {tdt:.4f} / CTC {ctc:.4f} vs the golden on {n_t} rows > 0.02")
    npz = calib / "parakeet" / "eval_jsut" / "eval-00000.npz"
    try:
        from kitsune.parakeet_targets import check_shard

        meta = read_json(calib / "parakeet" / "meta.json", {})
        res["parakeet"]["check_shard"] = check_shard(npz, meta)
        problems += [f"check_shard: {p}" for p in res["parakeet"]["check_shard"]]
    except Exception as e:
        problems.append(f"check_shard: {type(e).__name__}: {e}")
    res["problems"] = problems
    write_json_atomic(ctx.state_dir / "golden.json", res)
    if problems:
        raise HostFailure(f"golden: {problems}")


def consumer_local_problems(ctx: StepCtx) -> list[str]:
    """The per-file checks on the local tree: ids across the passes, second opinions, gate baselines, Cohere npz
    invariants and the Parakeet format."""
    import numpy as np

    from kitsune import store

    problems = []
    tmeta = read_json(ctx.teacher / "meta.json", {})
    pmeta = read_json(ctx.parakeet / "meta.json", {})
    names = set()
    try:
        from kitsune import extent

        names = set(extent.names(ctx.cfg))
    except Exception:
        pass
    second_sources = {s for s in SECOND_SOURCES if s in names}
    for sh in store.read_manifest(ctx.data):
        stem = Path(sh.path).stem
        where = f"{sh.source}/{stem}"
        ids = store.shard_ids(ctx.data, sh)
        tn, pn = ctx.teacher / sh.source / f"{stem}.npz", ctx.parakeet / sh.source / f"{stem}.npz"
        t_ids = npz_ids(tn)
        if t_ids is None:
            problems.append(f"{where}: teacher npz missing or unreadable")
            continue
        if not set(t_ids) <= set(ids):
            problems.append(f"{where}: teacher ids are not a subset of the shard's ids")
        if [r.get("id") for r in jsonl_rows(tn.with_suffix(".jsonl"))] != t_ids:
            problems.append(f"{where}: teacher jsonl ids != npz ids")
        try:
            with np.load(tn, allow_pickle=False) as z:
                if "k" in tmeta and int(z["k"]) != int(tmeta["k"]):
                    problems.append(f"{where}: teacher k {int(z['k'])} != meta {tmeta['k']}")
                prompt = tmeta.get("decoder_prompt_ids")
                if prompt is not None and [int(x) for x in z["prompt"]] != [int(x) for x in prompt]:
                    problems.append(f"{where}: teacher prompt differs from meta.json")
                if len(z["tokens"]) and not np.array_equal(z["topk_idx"][:, 0], z["tokens"]):
                    problems.append(f"{where}: topk_idx[:, 0] != tokens")
        except Exception as e:
            problems.append(f"{where}: teacher npz: {type(e).__name__}: {e}")
        p_ids = npz_ids(pn)
        if p_ids != t_ids:
            problems.append(f"{where}: Parakeet ids != teacher ids" if p_ids is not None else
                            f"{where}: Parakeet npz missing or unreadable")
        elif pmeta:
            try:
                from kitsune.parakeet_targets import check_shard

                problems += [f"{where}: {p}" for p in check_shard(pn, pmeta, ids)]
            except Exception as e:
                problems.append(f"{where}: check_shard: {type(e).__name__}: {e}")
        if sh.source in second_sources:
            s_ids = [r.get("id") for r in jsonl_rows(ctx.second / sh.source / f"{stem}.jsonl")]
            if s_ids != t_ids:
                problems.append(f"{where}: second opinion missing or its ids != teacher ids")
    try:
        from kitsune.evaluate import teacher_baselines

        teacher_baselines(ctx.teacher, GATE_SETS, check=True)
    except Exception as e:
        problems.append(f"gate baselines: {type(e).__name__}: {e}")
    return problems


def step_consumer_check(ctx: StepCtx, hub: bool = False):
    from kitsune import extent

    import launch

    inside = f"{ctx.root_rel}/"
    repo_files = list(ctx.api.list_repo_files(ctx.repo, repo_type="dataset"))  # students/, models/, the laptop roots
    if hub:
        files = sorted(repo_files)
    else:
        files = sorted([p for p in repo_files if not p.startswith(inside)] +
                       [p.relative_to(ctx.kdir).as_posix() for p in ctx.root.rglob("*")
                        if p.is_file() and not p.name.endswith(".tmp")])
    record = extent.load_record(ctx.root / extent.RECORD_FILE)
    problems = []
    for c in ctx.label_cfgs:
        cfg = json.loads((ctx.kdir / c).read_text(encoding="utf-8"))
        problems += [f"{c}: {p}" for p in extent.pull_plan(cfg, record, files).get("problems", [])]
        problems += [f"{c}: {p}" for p in launch.extent_problems(files, cfg, record)
                     if hub or "COMPLETE.json" not in p]  # the local tree is sealed only at F7
        problems += [f"{c}: {p}" for p in launch.selection_problems(ctx.kdir / cfg["selection"], cfg["selection"],
                                                                     cfg, set(files))]
    if not hub:
        problems += consumer_local_problems(ctx)
    out = {"hub": hub, "n_files": len(files), "problems": problems[:500], "n_problems": len(problems),
           "configs": ctx.label_cfgs}
    # the Hub check runs after the final sync: its result goes to the infra files, never into the sealed root
    write_json_atomic(ctx.state_dir / "consumer_check_hub.json" if hub else ctx.root / "reports" /
                      "consumer_check.json", out)
    if problems:
        raise Integrity(f"consumer check ({'Hub' if hub else 'local'}): {len(problems)} problem(s): {problems[:5]}")


def step_seal(ctx: StepCtx):
    import label_sync
    from kitsune import extent

    rec = ctx.root / extent.RECORD_FILE
    info = {"name": (ctx.cfg.get("extent") or {}).get("name"), "run_ids": [ctx.env.get("KITSUNE_RUN_ID")],
            "run_id": ctx.env.get("KITSUNE_RUN_ID"),
            "kitsune_sha": ctx.env.get("KITSUNE_SHA"), "image": ctx.env.get("KITSUNE_IMAGE"),
            "extent_sha256": sha256_file(rec), "configs": ctx.label_cfgs}
    label_sync.seal(ctx.api, ctx.repo, ctx.kdir, ctx.root_rel, info, ctx.state_dir / "label_ledger.json")


def run_step(name: str, argv: list[str], ctx: StepCtx | None = None) -> int:
    ap = argparse.ArgumentParser(prog=f"label.py step {name}")
    ap.add_argument("--restart", action="store_true", help="selftest: a restart (30 GB free disk is enough)")
    ap.add_argument("--hub", action="store_true", help="consumer-check: against the Hub listing")
    args = ap.parse_args(argv)
    ctx = ctx or StepCtx()
    fns = {"plan": step_plan, "pull": step_pull, "models": step_models,
           "selftest": lambda c: step_selftest(c, args.restart), "roots": step_roots, "golden": step_golden,
           "consumer-check": lambda c: step_consumer_check(c, args.hub), "seal": step_seal}
    try:
        fns[name](ctx)
    except StepError as e:
        log(f"step {name}: {type(e).__name__}: {e}")
        return e.rc
    except Exception as e:
        if type(e).__name__ == "IntegrityError":  # vast/label_sync.py: a write-once conflict or a sealed root
            log(f"step {name}: integrity: {e}")
            return EXIT_INTEGRITY
        log(f"step {name}: {type(e).__name__}: {e} (transient)")
        return 1
    log(f"step {name}: ok")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "step":
        if len(argv) < 2 or argv[1] not in STEPS:
            print(f"usage: label.py step {{{','.join(STEPS)}}} [--restart] [--hub]", file=sys.stderr)
            return 2
        return run_step(argv[1], argv[2:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="finish.py only prints (no stop/destroy)")
    args = ap.parse_args(argv)
    return Controller(dry_run=args.dry_run).run()


if __name__ == "__main__":
    sys.exit(main())
