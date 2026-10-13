"""The box queue's base: the process, state and upload machinery of kitsune/full_queue.py's FullQueue.

The size study's box queue (its box plans, calibration, LR probes, PREREG numbers, waves, anchor, speed and shakedown
phases) ran on 2026-09-26 and was removed after the study; the last commit with it is 76b28a1. What is left is the part
the full-data runs' queue builds on (it subclasses Queue and Settings; the module keeps its name: kitsune/full_queue.py
and kitsune/whisper.py import it):

  processes   Queue.start runs an item's command line (argv_for) with CUDA_VISIBLE_DEVICES pinned, in its own process
              group, logging to $KITSUNE_STATE/logs/<item>.log; stop_all and a restart's kill_orphans (ORPHAN_CMDS) end
              them; set_aside moves a run dir given up to runs/_abandoned/
  scheduling  Queue.execute runs a list of items on the free GPUs until each has ended, through the subclass's hooks
              (_next_for, resume_point, finished, drain_uploads)
  state       $KITSUNE_STATE/queue.json (atomic, fsynced; the subclass's _load_state) and events.jsonl (Queue.event)
  uploads     HubUploader: a run dir's lean sync and verification, one file up and verified, a file or a folder down,
              through vast/finish.py (_finish), every write backing off on a 429 or a raced commit (finish.hub_retry)
  readers     read_events (a run dir's events.jsonl) and StepTail (its metrics/scalars.jsonl, incrementally)
A subclass supplies _load_state, argv_for, child_env, _ready, _next_for, resume_point, finished and drain_uploads.
Pure Python at import (stdlib + kitsune.prereg); the trainer's modules only through _load_trainer.
"""
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from kitsune import prereg

REPO = Path(__file__).resolve().parents[1]
STATE_FILE, EVENTS_FILE = "queue.json", "events.jsonl"
EXIT_THROUGHPUT = 3  # a trainer's throughput floor (scripts/04_distill.py EXIT_THROUGHPUT)
SPEED_TOOL = "tools/speed_probe.py"
MAX_ATTEMPTS = 2  # per item, unless the subclass sets its own (the full queue takes its box's from the registry)
# the command lines of the processes a queue starts (kill_orphans): the trainer, the evaluator and, for the full-run
# queue (kitsune/full_queue.py), its store build and resume check, the speed probe, the Whisper eval and the quant CLI.
# Never the bare "kitsune.full_queue": that is the queue itself (its run command)
ORPHAN_CMDS = ("04_distill", "05_evaluate", "kitsune.full_queue build-stores", "kitsune.full_queue check-resume",
               "speed_probe", "whisper_eval", "kitsune.quant")
STAMP_RE = r"-\d{8}T\d{6}Z(?:-\d+)?"


class QueueError(RuntimeError):
    """Anything the queue cannot go on from."""


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [queue] {msg}", flush=True)


# ================================================================================================ metrics readers


def read_events(run_dir) -> list[dict]:
    """A run dir's events.jsonl (a torn last line is skipped)."""
    p = Path(run_dir) / "events.jsonl"
    out = []
    if p.is_file():
        for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


STEP_TAGS = {"sched/train_s": "train_s", "time/data_wait_s": "data_wait_s", "time/step_s": "step_s"}


def _merge_line(rows: dict, line: str):
    try:
        r = json.loads(line)
    except ValueError:
        return
    key = STEP_TAGS.get(r.get("tag"))
    if key is None or not isinstance(r.get("step"), int):
        return
    row = rows.setdefault(r["step"], {})
    row[key] = r.get("value")
    if key == "train_s":
        row["wall"] = r.get("wall")


class StepTail:
    """step -> {train_s (the loop clock when the step was logged), data_wait_s, step_s, wall} from a run's
    metrics/scalars.jsonl (flushed at every step; a later line of the same step, a resumed launch's, wins), read
    incrementally: each poll() reads only what was appended since (the full queue polls every running trainer's steps
    every few seconds, and a small run's file grows by ~7 KB a step)."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "metrics" / "scalars.jsonl"
        self.pos, self.rest, self.rows = 0, b"", {}

    def poll(self) -> dict[int, dict]:
        if not self.path.is_file():
            return self.rows
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
        self.pos += len(data)
        data = self.rest + data
        lines = data.split(b"\n")
        self.rest = lines.pop()  # a line still being written
        for line in lines:
            _merge_line(self.rows, line.decode("utf-8", "replace"))
        return self.rows


# ============================================================================================================ hub


def _finish():
    """vast/finish.py (stdlib-only at import): sync, verify, expected_files, hub_retry."""
    vast = str(REPO / "vast")
    if vast not in sys.path:
        sys.path.insert(0, vast)
    import finish

    return finish


class HubUploader:
    """The runs repo, through vast/finish.py: a run dir's lean sync and verification, one file up and verified, a file
    or a folder down."""

    def __init__(self, repo: str, repo_type: str = "model"):
        self.repo, self.repo_type, self._api = repo, repo_type, None

    def api(self):
        if self._api is None:
            self._api = _finish().hf_api()
        return self._api

    def sync_run(self, run_dir: Path) -> list[str]:
        """Upload the run dir and verify it; the problems (empty: every expected file is on the Hub). The upload
        backs off (finish.hub_retry: 5 s doubling to 10 min) like every other write of the queue: the box's trainers
        and its queue commit into the one runs repo, where a 429 or a commit that races another writer is the Hub
        asking for a later try, not a lost run (finish.sync raises after trying both of its parts; a re-upload of what
        is there already costs no commit)."""
        fin = _finish()
        try:
            fin.hub_retry(lambda: fin.sync(self.api(), self.repo, self.repo_type, run_dir, False),
                          f"sync {run_dir.name}")
        except Exception as e:  # noqa: BLE001  the verification below says what is missing
            log(f"sync of {run_dir.name} failed: {type(e).__name__}: {e}")
        return fin.verify(self.api(), self.repo, self.repo_type, fin.expected_files(run_dir))

    def put_file(self, local: Path, path_in_repo: str) -> list[str]:
        fin = _finish()
        fin.hub_retry(lambda: self.api().upload_file(path_or_fileobj=str(local), path_in_repo=path_in_repo,
                                                     repo_id=self.repo, repo_type=self.repo_type,
                                                     commit_message=f"box queue: {path_in_repo}"),
                      f"upload {path_in_repo}")
        return fin.verify(self.api(), self.repo, self.repo_type, {path_in_repo: Path(local)},
                          prefix_of=lambda p: p.rsplit("/", 1)[0])

    def download(self, path_in_repo: str, local_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download

        return Path(_finish().hub_retry(lambda: hf_hub_download(self.repo, path_in_repo, repo_type=self.repo_type,
                                                                local_dir=str(local_dir)), f"download {path_in_repo}"))

    def download_dir(self, prefix: str, local_dir: Path) -> Path:
        from huggingface_hub import snapshot_download

        _finish().hub_retry(lambda: snapshot_download(self.repo, repo_type=self.repo_type, local_dir=str(local_dir),
                                                      allow_patterns=[f"{prefix}/*"]), f"download {prefix}")
        return Path(local_dir) / prefix


# ============================================================================================================ queue


@dataclass
class Settings:
    """Everything the queue takes from its environment (the tests replace the commands, GPUs and uploader)."""
    root: Path = REPO
    state_dir: Path = field(default_factory=lambda: Path(os.environ.get("KITSUNE_STATE", "/workspace/kitsune_state")))
    out_repo: str | None = field(default_factory=lambda: os.environ.get("KITSUNE_OUT_REPO") or None)
    gpus: list[str] | None = None  # None: KITSUNE_GPUS (comma list), else nvidia-smi's indices
    # the GPUs the box was rented with (vast/launch.py KITSUNE_N_GPUS): the queue refuses to run on another count
    n_gpus: int | None = field(default_factory=lambda: int(os.environ["KITSUNE_N_GPUS"])
                               if os.environ.get("KITSUNE_N_GPUS") else None)
    python: str = sys.executable
    train_cmd: list[str] | None = None  # default: python scripts/04_distill.py
    stores_cmd: list[str] | None = None  # default: the subclass's (python -m kitsune.full_queue build-stores)
    eval_cmd: list[str] | None = None  # default: python scripts/05_evaluate.py
    speed_cmd: list[str] | None = None  # default: python tools/speed_probe.py
    uploader: object = None  # default: HubUploader(out_repo); None without a repo (nothing uploaded, nothing verified)
    poll_s: float = 10.0
    env: dict = field(default_factory=dict)  # extra env for every child


def detect_gpus() -> list[str]:
    """KITSUNE_GPUS (a comma list of CUDA_VISIBLE_DEVICES values), else nvidia-smi's indices; empty when neither says
    (the queue then refuses: a box that silently ran on one assumed GPU would train its items one after the other)."""
    if os.environ.get("KITSUNE_GPUS"):
        return [g.strip() for g in os.environ["KITSUNE_GPUS"].split(",") if g.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=60).stdout
        return [x.strip() for x in out.splitlines() if x.strip().isdigit()]
    except (OSError, subprocess.SubprocessError):
        return []


def _atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True, default=str))
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _pdeathsig():  # pragma: no cover - Linux only: a trainer dies with the queue that started it
    try:
        import ctypes

        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:  # noqa: BLE001
        pass


class Queue:
    """One box's items on its GPUs (the module docstring); kitsune/full_queue.py's FullQueue gives the items and the
    hooks."""

    def __init__(self, box: str, settings: Settings):
        self.box = box
        self.s = settings
        self.root = Path(self.s.root)
        self.max_attempts = MAX_ATTEMPTS  # per item (finished); the full-run queue takes its box's from the registry
        self.gpus = list(self.s.gpus or detect_gpus())
        if not self.gpus:
            raise QueueError("no GPU found: nvidia-smi listed none and KITSUNE_GPUS is not set")
        if self.s.n_gpus is not None and len(self.gpus) != int(self.s.n_gpus):
            raise QueueError(f"box {box} was rented with {self.s.n_gpus} GPU(s) (KITSUNE_N_GPUS), the queue found "
                             f"{len(self.gpus)}: {self.gpus}")
        self.state_path = Path(self.s.state_dir) / STATE_FILE
        self.state = self._load_state()
        self.uploader = self.s.uploader if self.s.uploader is not None else (
            HubUploader(self.s.out_repo) if self.s.out_repo else None)
        self.procs: dict[str, subprocess.Popen] = {}
        self._pending_uploads: list[str] = []

    # -------------------------------------------------------------------------------------------------- state

    def save(self):
        _atomic_json(self.state_path, self.state)

    def event(self, kind: str, **fields):
        """A lifecycle record in $KITSUNE_STATE/events.jsonl (finish.py's file; uploaded with the infra logs)."""
        p = Path(self.s.state_dir) / EVENTS_FILE
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"wall": time.time(), "source": "queue", "box": self.box, "kind": kind, **fields},
                               default=str) + "\n")
        log(f"{kind}: " + json.dumps(fields, default=str)[:600])

    def item(self, name: str) -> dict:
        return self.state["items"][name]

    def add_item(self, name: str, kind: str, config: str | None, sets: list[str] | None = None, *,
                 run: str | None = None) -> str:
        """Register an item once (a restart finds it registered, with its status)."""
        if name not in self.state["items"]:
            self.state["items"][name] = dict(kind=kind, config=config, sets=list(sets or []), run=run,
                                             status="pending", attempts=[], run_dir=None, verified=None, result=None)
        return name

    # ----------------------------------------------------------------------------------------- processes

    def _cmd(self, kind: str) -> list[str]:
        py = self.s.python
        if kind == "train":
            return list(self.s.train_cmd or [py, str(self.root / "scripts" / "04_distill.py")])
        if kind == "eval":
            return list(self.s.eval_cmd or [py, str(self.root / "scripts" / "05_evaluate.py")])
        return list(self.s.speed_cmd or [py, str(self.root / SPEED_TOOL)])

    def run_dirs_of(self, run_name: str) -> list[Path]:
        runs = self.root / "runs"
        pat = re.compile(re.escape(run_name) + STAMP_RE + "$")
        return sorted((d for d in runs.iterdir() if d.is_dir() and pat.match(d.name)),
                      key=lambda d: d.stat().st_mtime) if runs.is_dir() else []

    def find_run_dir(self, name: str) -> Path | None:
        """The run dir of the item's last attempt: its resumed dir, else the newest <name>-<stamp> dir created since the
        attempt started."""
        it = self.item(name)
        att = it["attempts"][-1] if it["attempts"] else None
        if att and att.get("resume"):
            return self.root / att["resume"]
        t0 = att["t0"] if att else 0
        cands = [d for d in self.run_dirs_of(name) if d.stat().st_mtime >= t0 - 5]
        return cands[-1] if cands else None

    def _rel(self, p: Path) -> str:
        return Path(p).resolve().relative_to(self.root.resolve()).as_posix()

    def start(self, name: str, gpu: str, resume: Path | None = None):
        it = self.item(name)
        kind = it["kind"]
        env = dict(os.environ, **self.s.env, CUDA_VISIBLE_DEVICES=str(gpu), KITSUNE_QUEUE_ITEM=name)
        env.update(self.child_env(name, it, gpu))
        argv = self.argv_for(name, it, resume)
        attempt = dict(t0=time.time(), gpu=str(gpu), resume=self._rel(resume) if resume is not None else None,
                       argv=argv)
        logs = Path(self.s.state_dir) / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        out = open(logs / f"{name}.log", "ab")
        kw = dict(cwd=str(self.root), env=env, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if os.name == "posix":
            kw.update(start_new_session=True, preexec_fn=_pdeathsig)
        proc = subprocess.Popen(argv, **kw)
        out.close()
        attempt["pid"] = proc.pid
        it["attempts"].append(attempt)
        it["status"] = "running"
        self.procs[name] = proc
        self.save()
        self.event("item_start", item=name, item_kind=kind, gpu=str(gpu), attempt=len(it["attempts"]),
                   resume=attempt["resume"], argv=argv)
        return proc

    def _signal(self, proc: subprocess.Popen, sig):
        try:
            if os.name == "posix":
                os.killpg(proc.pid, sig)
            else:
                proc.terminate()
        except (OSError, ProcessLookupError):
            pass

    def stop_all(self):
        """Terminate every trainer this queue still runs (a halt or an error): the process group, so its loader
        workers go too."""
        for name, proc in list(self.procs.items()):
            if proc.poll() is None:
                log(f"terminating {name} (pid {proc.pid})")
                self._signal(proc, signal.SIGTERM)
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    self._signal(proc, getattr(signal, "SIGKILL", signal.SIGTERM))
                it = self.item(name)
                if it["status"] == "running":
                    it["status"] = "interrupted"
        self.procs.clear()
        self.save()

    def kill_orphans(self):
        """A restart: the trainers the previous queue process left running (a crash of the queue itself, not of the
        container) are killed before anything starts, so no GPU runs two jobs; their items resume below. A pid is
        killed only when its command line is one the queues start (ORPHAN_CMDS: never the bare kitsune.full_queue,
        which is the queue itself) and never when it is this process or its parent (a recycled pid)."""
        for name, it in self.state["items"].items():
            if it["status"] != "running" or not it["attempts"]:
                continue
            pid = it["attempts"][-1].get("pid")
            it["status"] = "interrupted"
            if os.name != "posix" or not pid or pid in (os.getpid(), os.getppid()):
                continue
            try:
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if name in cmdline or any(c in cmdline for c in ORPHAN_CMDS):
                log(f"killing {name}'s orphaned process group {pid}")
                try:
                    os.killpg(pid, signal.SIGKILL)
                except OSError:
                    pass
        self.save()

    def set_aside(self, run_dir: Path | None, why: str):
        """A run dir given up (a run that cannot resume from it): moved to runs/_abandoned/, where finish.py's run-dir
        scan does not look."""
        if run_dir is None or not Path(run_dir).is_dir():
            return
        dest = self.root / "runs" / "_abandoned" / Path(run_dir).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(dest))
        self.state["abandoned"].append(dict(run_dir=Path(run_dir).name, why=why, wall=time.time()))
        self.event("run_dir_abandoned", run_dir=Path(run_dir).name, why=why)

    # -------------------------------------------------------------------------------------------- execution

    def execute(self, todo: list[str], *, monitor=None):
        """Run the items of `todo` on the free GPUs until each has ended (done, failed or skipped): per free GPU the
        item _next_for gives, started from its resume_point; monitor(running) runs at every poll (the full queue's
        stalls, faults and resources). A QueueError from a hook stops every running item and propagates."""
        running: dict[str, dict] = {}  # gpu -> {"name", "proc"}
        try:
            while True:
                for gpu in self.gpus:
                    if gpu in running:
                        continue
                    name = self._next_for(gpu, todo, running)
                    if name is None:
                        continue
                    resume = self.resume_point(name)
                    running[gpu] = dict(name=name, proc=self.start(name, gpu, resume))
                self.drain_uploads(one=True)
                if not running:
                    if not any(self._ready(n) for n in todo):
                        waiting = [n for n in todo if self.item(n)["status"] in ("pending", "interrupted", "retry")]
                        if waiting:  # their dependency failed: they can never run
                            for n in waiting:
                                self.item(n)["status"] = "skipped"
                                self.event("item_skipped", item=n)
                            self.save()
                        return
                    continue
                if monitor is not None:
                    monitor(running)
                time.sleep(self.s.poll_s)
                for gpu, r in list(running.items()):
                    rc = r["proc"].poll()
                    if rc is None:
                        continue
                    del running[gpu]
                    self.procs.pop(r["name"], None)
                    self.finished(r["name"], rc, todo)
        except BaseException:
            for r in running.values():
                if r["proc"].poll() is None:
                    self._signal(r["proc"], signal.SIGTERM)
            for r in running.values():
                try:
                    r["proc"].wait(60)
                except subprocess.TimeoutExpired:
                    self._signal(r["proc"], getattr(signal, "SIGKILL", signal.SIGTERM))
                if self.item(r["name"])["status"] == "running":
                    self.item(r["name"])["status"] = "interrupted"
                self.procs.pop(r["name"], None)
            self.save()
            raise

    @staticmethod
    def summary_of(rd: Path | None) -> dict | None:
        try:
            return json.loads((Path(rd) / "summary.json").read_text(encoding="utf-8")) if rd else None
        except (OSError, ValueError):
            return None

    def ensure_attribution(self, it: dict, rd: Path):
        """The CC-BY-4.0 attribution travels with a Parakeet-derived student (its dir's MODEL_CARD.md): into the run
        dir and every exported weights dir that lacks it, so the lean upload carries it."""
        run = it.get("run")
        spec = prereg.RUNS.get(run) if run else None
        if not spec or spec["family"] != "ctc":
            return
        card = self.root / spec["student"] / "MODEL_CARD.md"
        if not card.is_file():
            return
        fin = _finish()
        targets = [rd] + [p for p in (rd / "checkpoints").glob("*") if p.is_dir() and fin.WEIGHTS_RE.match(p.name)]
        for t in targets:
            if not (t / "MODEL_CARD.md").is_file():
                shutil.copyfile(card, t / "MODEL_CARD.md")


def _load_trainer():
    spec = importlib.util.spec_from_file_location("kitsune_script_04_distill", REPO / "scripts" / "04_distill.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
