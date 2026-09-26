"""End-of-run actions on the vast box: final upload, verification against the HF output repo, then destroy or stop.

Why the ordering (D49a): destroying the instance deletes its disk, so it is only safe once every file the run
produced is provably on the Hub. `--destroy` therefore uploads first, then lists the output repo and compares every
expected file by path, size and hash (sha256 for LFS/Xet files, git blob id for small ones). Only a clean comparison
destroys; any problem stops the instance instead, which keeps the disk (storage is still billed, GPU is not) so nothing
is lost and a human can look. The vast REST API is called with the per-instance key vast injects
(CONTAINER_API_KEY can only start/stop/destroy this one instance); the vastai CLI is the fallback.

Expected files for each run dir runs/<run_id>/ (see scripts/04_distill.py):
  every file outside checkpoints/                      -> runs/<run_id>/<same path>   (RunLogger sync)
  every weights dir       checkpoints/step_<N>/        -> runs/<run_id>/checkpoints/step_<N>/
    (all of them, not just the newest: an earlier upload that failed would otherwise go with the destroyed disk; the
    hub skips re-uploads of content it already has, though each byte is still read and hashed on the box, see sync)
  the newest full state   checkpoints/full_step_<N>/   -> runs/<run_id>/checkpoints/full_step_<N>/  (--expect-full)
  and every full state the trainer meant for the Hub whose upload has not succeeded (UPLOAD_MARK in it: the
    pre_cooldown one, which the trainer keeps from rotation for this; the marker itself is not uploaded)
The files outside checkpoints/ are uploaded from a snapshot copy: the watchdog's --sync-only runs while the trainer
still appends to its logs, and a file handed to the Hub by path is sized when it is listed but hashed and read later
(a growing file would go up as a size/hash/content mismatch). Checkpoint dirs are renamed into place complete and go
up in place; the one later write, the trainer's end save replacing trainer.pt/.json of an existing full_step_<N> (the
loop ended at the step of a full state), is over once the trainer has exited.
Supervisor/watchdog/bootstrap/portal logs are uploaded best-effort to runs/<run_id>/infra/, secrets redacted (scrub),
but not verified (they are still being written while this runs); on every path, --no-sync included, and once more
right before the stop/destroy call, so the last lifecycle records (verify, stop/destroy) are off the box before its
disk goes away or stays behind.

Every decision is appended to $KITSUNE_STATE/events.jsonl, and a `halt` marker is written before a stop/destroy so a
restarted container does not start a second run (vast/onstart.sh checks it).

Usage:
  python vast/finish.py --destroy                 # upload, verify, destroy (or stop if verification fails)
  python vast/finish.py --stop --reason "..."     # upload best-effort, stop
  python vast/finish.py --sync-only               # upload only (supervisor after a crash, watchdog before the cap)
  python vast/finish.py --verify-only             # print the comparison, exit 1 on problems
  add --dry-run to any of them to print the actions without uploading, stopping or destroying

Label box (--job label, the default when KITSUNE_JOB=label): the same modes over the config's label root
(extent.root of $KITSUNE_CONFIG, e.g. labels/full) in $KITSUNE_DATA_REPO, uploaded write-once by vast/label_sync.py
under $KITSUNE_STATE/sync.lock. --sync-only exits 1 when an upload failed and 65 on a write-once refusal; infra goes
to label_runs/<run_id>/. --no-infra skips the infra upload; --allow-empty lets --destroy go on with no finished label
file (a host failure before any lane started).

Study box (--job study, the default when KITSUNE_JOB=study; kitsune/study_queue.py runs the box): every mode is lean -
per run dir the logs, metrics, evals and summary, every exported weights dir and only the full states the run's config
uploads (ckpt.upload_full_at "frac:<f>" / "end", uploaded_fulls; plus a marked one), never the newest full state for
its own sake: the box keeps its resume states on its disk and the runs repo gets what the study reports (STUDY.md 5.5).
The infra logs and the queue's state go to study/box-<KITSUNE_BOX>/infra/<container>/, with the queue's per-item logs
($KITSUNE_STATE/logs/) and each re-armed state (rearm-<stamp>/) under it (infra_files deep).
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VAST_API = "https://console.vast.ai/api/v0"
STATE_DIR = Path(os.environ.get("KITSUNE_STATE", "/workspace/kitsune_state"))
INFRA_LOGS = ["/workspace/kitsune.log", "/workspace/watchdog.log", "/workspace/portal.log", "/workspace/tensorboard.log"]
INFRA_TIMEOUT_S = 180  # a hung infra upload must not keep a paid instance up
INFRA_MAX_FILE_BYTES = 32 << 20  # a study box's per-item log goes up as its last 32 MB (upload_infra deep)
# s before each retry of a hub call the library does not retry itself: exponential back-off from 5 s up to 10 min, ~21
# min in all (STUDY.md 5.5: the study's runs share one repo, and a 429 needs the Hub's rate window to pass). Only what a
# retry can fix is retried (retryable)
HUB_RETRY_WAITS = (5, 10, 20, 40, 80, 160, 320, 600)
# an env var whose name holds one of these is a secret: a mirror of kitsune/runlog.py's SECRET_MARKERS (finish stays
# stdlib-only; runlog imports pyarrow)
SECRET_MARKERS = ("TOKEN", "KEY", "SECRET", "PASS", "AUTH", "CRED", "COOKIE")
# what the base image's portal prints in the clear into portal.log once a PORTAL_CONFIG reaches the box (the account
# env, a template): caddy_config_manager's web credentials, open-button token and Bearer header (the password may be a
# generated uuid that is in no env var), syncthing's API key, a token in a URL
PRINTED_SECRET_RE = re.compile(rb"(credentials are: \S+ / |token is also valid: |Bearer |--gui-apikey=|[?&]token=)"
                               rb"[^\s\"')]+")
# checkpoint names written by the trainer; a directory or a single file (full_step_N.pt). In-progress writes end in
# .tmp/.partial and never match.
WEIGHTS_RE = re.compile(r"^step[_-]?(\d+)$")
FULL_RE = re.compile(r"^full[_-]?(?:step[_-]?)?(\d+)(?:\.(?!tmp$|partial$)[A-Za-z0-9]+)?$")
# scripts/04_distill.py's marker in a full state meant for the Hub (ckpt.upload_full_at), removed once its upload
# succeeded
UPLOAD_MARK = ".upload_pending"
HASH_CHUNK = 8 << 20
EXIT_INTEGRITY = 65  # label mode: a write-once conflict or a sealed root (vast/label.py ends such a box with a stop)
SYNC_LOCK_WAIT_S = 1500  # label mode: sync.lock is polled up to 25 min, then the sync is skipped


def log(msg: str):
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [finish] {msg}", flush=True)


def event(kind: str, **fields):
    """Append one lifecycle record to $KITSUNE_STATE/events.jsonl (uploaded with the infra logs)."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(STATE_DIR / "events.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"wall": time.time(), "source": "finish", "kind": kind, **fields}) + "\n")
    except OSError as e:
        log(f"could not write event {kind}: {e}")


# ---------------------------------------------------------------------------------------------------------- files

def run_dirs(runs_root: Path) -> list[Path]:
    """Run directories = subdirs of runs/ that the trainer created (they hold config.json or events.jsonl)."""
    if not runs_root.is_dir():
        return []
    return sorted(d for d in runs_root.iterdir()
                  if d.is_dir() and ((d / "config.json").exists() or (d / "events.jsonl").exists()))


def newest_checkpoint(ckpt_root: Path, pattern: re.Pattern) -> Path | None:
    """Highest-step checkpoint whose name matches; half-written ones (.tmp/.partial) never match the pattern."""
    best, best_step = None, -1
    if ckpt_root.is_dir():
        for p in ckpt_root.iterdir():
            m = pattern.match(p.name)
            if m and int(m.group(1)) > best_step:
                best, best_step = p, int(m.group(1))
    return best


def files_under(p: Path) -> list[Path]:
    return [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file())


def uploaded_fulls(run_dir: Path) -> list[Path]:
    """The full states a run's config sends to the Hub, which a lean verification expects (the study box's runs keep
    every other full state on the box): ckpt.upload_full_at "frac:<f>" -> checkpoints/full_step_<round(f x
    schedule.max_steps)> (none for a T/2 branch, which ignores its fractions) and "end" -> the newest full state; from
    the run's config.json. A missing or unreadable config.json: none."""
    try:
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["config"]
    except (OSError, ValueError, KeyError, TypeError):
        return []
    ckpt, out = run_dir / "checkpoints", []
    ups = (cfg.get("ckpt") or {}).get("upload_full_at") or []
    m = (cfg.get("schedule") or {}).get("max_steps")
    for u in ups:
        if isinstance(u, str) and u.startswith("frac:") and m and not (cfg.get("branch") or {}).get("parent"):
            try:
                out.append(ckpt / f"full_step_{int(round(float(u[5:]) * int(m)))}")
            except ValueError:
                continue
        elif u == "end":
            out.append(newest_checkpoint(ckpt, FULL_RE))
    return [p for p in out if p is not None and p.exists()]


def expected_files(run_dir: Path, expect_full: bool = True, lean: bool = False) -> dict[str, Path]:
    """repo path -> local file, for one run dir (see the module docstring). lean (the study box, KITSUNE_JOB=study):
    the logs, every weights dir, the marked full states and the ones the config uploads (uploaded_fulls), never the
    newest full state for its own sake."""
    prefix = f"runs/{run_dir.name}"
    out = {}
    for f in files_under(run_dir):
        rel = f.relative_to(run_dir).as_posix()
        if rel.split("/", 1)[0] in ("checkpoints", "infra") or rel.endswith((".tmp", ".lock")):
            continue
        out[f"{prefix}/{rel}"] = f
    ckpt = run_dir / "checkpoints"
    weights = sorted(p for p in ckpt.iterdir() if WEIGHTS_RE.match(p.name)) if ckpt.is_dir() else []
    # a full state whose trainer upload failed (the pre_cooldown one) has no other copy than this disk
    marked = sorted(p for p in ckpt.iterdir()
                    if FULL_RE.match(p.name) and (p / UPLOAD_MARK).is_file()) if ckpt.is_dir() else []
    picks = weights + (marked + [newest_checkpoint(ckpt, FULL_RE)] if expect_full else [])  # a dir twice: same keys
    if lean:
        picks += marked + uploaded_fulls(run_dir)
    for pick in picks:
        if pick is None:
            continue
        for f in files_under(pick):
            if f.name != UPLOAD_MARK:
                out[f"{prefix}/checkpoints/{f.relative_to(ckpt).as_posix()}"] = f
    return out


def git_blob_id(path: Path) -> str:
    """The git object id the Hub reports for a non-LFS file: sha1(b"blob <size>\\0" + content)."""
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def snapshot(src_root: Path, rels: list[str], dest: Path):
    """Copy each file as it is now: at most the size it had when opened (a file still being appended to), from the
    one version that was opened (a file replaced by os.replace meanwhile). Files that vanished are skipped."""
    for rel in rels:
        try:
            with open(src_root / rel, "rb") as f:
                left = os.fstat(f.fileno()).st_size
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                with open(dest / rel, "wb") as g:
                    while left > 0 and (chunk := f.read(min(left, HASH_CHUNK))):
                        g.write(chunk)
                        left -= len(chunk)
        except FileNotFoundError:
            continue


# ------------------------------------------------------------------------------------------------------------ hub

def bound_hub_http():
    """Give huggingface_hub's shared HTTP client a timeout (as scripts/04_distill.py does): its default has none, and a
    request the server accepted and never answered (the commit POST of a sync) would keep the instance up until the
    watchdog. The pinned client is kept otherwise (its request hook, redirects). list_repo_tree passes timeout=None
    itself and is not covered."""
    import httpx
    import huggingface_hub
    from huggingface_hub.utils import _http

    def factory():
        c = _http.default_client_factory()
        c.timeout = httpx.Timeout(60, read=300)  # read: well above the Hub's 60 s commit timeout on its side
        return c

    huggingface_hub.set_client_factory(factory)


def hf_api():
    from huggingface_hub import HfApi  # imported lazily: --help and the pure helpers work without the hub
    bound_hub_http()
    return HfApi()


def retryable(e: BaseException) -> bool:
    """A hub error a retry can fix: 429, 408 or 5xx, or one without an HTTP status (a dropped connection, a timeout);
    any other 4xx (a refused token, a missing repo or file) comes back the same."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    return not isinstance(status, int) or status in (408, 429) or status >= 500


def hub_retry(fn, what: str):
    """Call fn, retrying with HUB_RETRY_WAITS' exponential back-off on a retryable error: huggingface_hub does not
    retry the first page of list_repo_tree or the create_commit POST, so one transient 5xx/429 would stop a verified
    run (its disk billed until a human looks) or lose the box logs with the destroyed disk. A repeated infra commit is
    harmless."""
    for w in HUB_RETRY_WAITS:
        try:
            return fn()
        except Exception as e:
            if not retryable(e):
                raise
            log(f"{what} failed ({type(e).__name__}: {e}); retrying in {w} s")
            time.sleep(w)
    return fn()


def remote_listing(api, repo: str, repo_type: str, prefix: str, recursive: bool = True) -> dict:
    """repo path -> RepoFile for every file under prefix (folders are skipped: they have no size)."""
    out = {}
    for item in api.list_repo_tree(repo, path_in_repo=prefix, recursive=recursive, repo_type=repo_type):
        if getattr(item, "size", None) is not None:
            out[item.path] = item
    return out


def verify(api, repo: str, repo_type: str, expected: dict[str, Path], check_hash: bool = True,
           prefix_of=None) -> list[str]:
    """Compare expected local files with the repo. Returns problems; an empty list means every file is there.
    prefix_of(path) -> dir: list each returned dir non-recursively instead of runs/<run_id> recursively."""
    if not expected:
        return ["nothing to verify: no run directory with files was found"]
    problems = []
    listings: dict[str, dict] = {}
    for path, local in sorted(expected.items()):
        prefix = prefix_of(path) if prefix_of is not None else "/".join(path.split("/")[:2])  # runs/<run_id>
        if prefix not in listings:
            try:  # the whole listing: list_repo_tree's error comes while its pages are iterated
                listings[prefix] = hub_retry(lambda: remote_listing(api, repo, repo_type, prefix,
                                                                    recursive=prefix_of is None), f"listing {prefix}")
            except Exception as e:
                problems.append(f"{prefix}: cannot list the repo: {type(e).__name__}: {e}")
                listings[prefix] = {}
        remote = listings[prefix].get(path)
        size = local.stat().st_size
        if remote is None:
            problems.append(f"{path}: missing in {repo}")
        elif remote.size != size:
            problems.append(f"{path}: size {remote.size} in repo, {size} local")
        elif check_hash:
            lfs = getattr(remote, "lfs", None)
            if lfs is not None and getattr(lfs, "sha256", None):
                if sha256_file(local) != lfs.sha256:
                    problems.append(f"{path}: sha256 differs")
            elif getattr(remote, "blob_id", None) and git_blob_id(local) != remote.blob_id:
                problems.append(f"{path}: git blob id differs")
    return problems


def sync(api, repo: str, repo_type: str, run_dir: Path, expect_full: bool, dry_run: bool, lean: bool = False):
    """Upload the run dir (minus local-only checkpoints) plus the checkpoints finish will verify.

    Re-uploading files the trainer already pushed costs no network or commit: the hub skips content it already stores.
    hf_xet still reads and chunks every checkpoint byte locally (about 20 GB for the viability run: every weights dir
    and the newest full state, a minute or two at the launch filter's disk_bw>=500), and verify() reads them once more
    for sha256. This mostly repairs a checkpoint upload that failed and captures log lines written after the trainer's
    last sync. The logs go first: they are small, the watchdog's --sync-only has 10 minutes before the stop, which a
    ~9 GB full state not yet on the hub can take all of (the final eval and verdict of a trainer still waiting on its
    end-state upload are on this disk only). Each part gets its own try, logs first: a checkpoint upload that raises
    must not cost the logs, nor a log upload that raises (the unretried repo_info GET of an unchanged snapshot, say)
    the checkpoints, which may be the only copy off the box. Raises after both were tried if either failed. lean: the
    study box's files (expected_files)."""
    expected = expected_files(run_dir, expect_full, lean)
    rels = sorted(p[len(f"runs/{run_dir.name}/"):] for p in expected)
    ckpt = [r for r in rels if r.startswith("checkpoints/")]
    live = [r for r in rels if not r.startswith("checkpoints/")]
    dest = f"runs/{run_dir.name}"
    log(f"sync {run_dir} -> {repo}:{dest} ({len(live)} files + {len(ckpt)} checkpoint files)")
    if dry_run:
        return

    def upload_live():  # logs the trainer may still be writing: a consistent copy (module docstring)
        with tempfile.TemporaryDirectory(prefix="kitsune-finish-") as stage:
            snapshot(run_dir, live, Path(stage))
            api.upload_folder(repo_id=repo, repo_type=repo_type, folder_path=stage, path_in_repo=dest,
                              allow_patterns=live, commit_message=f"finish: sync {run_dir.name}")

    def upload_ckpt():  # renamed into place when complete (the end save may still replace trainer.pt/.json): in place
        api.upload_folder(repo_id=repo, repo_type=repo_type, folder_path=str(run_dir), path_in_repo=dest,
                          allow_patterns=ckpt, commit_message=f"finish: checkpoints {run_dir.name}")

    failed = []
    for what, files, upload in (("logs", live, upload_live), ("checkpoints", ckpt, upload_ckpt)):
        if files:
            try:
                upload()
            except Exception as e:  # noqa: BLE001  (main() logs the combined error and records sync_failed)
                log(f"sync of {run_dir.name} {what} failed: {type(e).__name__}: {e}")
                failed.append(f"{what}: {type(e).__name__}: {e}")
    if failed:
        raise RuntimeError("; ".join(failed))


def scrub(data: bytes) -> bytes:
    """data with every secret replaced by <redacted>: the value (8+ chars) of each env var with a SECRET_MARKERS part
    in its name (WEB_PASSWORD, OPEN_BUTTON_TOKEN, HF_TOKEN, CONTAINER_API_KEY, ...), longest first so a value inside
    another cannot leave part of the longer one, then what PRINTED_SECRET_RE matches."""
    values = {v for k, v in os.environ.items() if len(v) >= 8 and any(m in k.upper() for m in SECRET_MARKERS)}
    for v in sorted(values, key=len, reverse=True):
        data = data.replace(v.encode("utf-8", "surrogateescape"), b"<redacted>")
    return PRINTED_SECRET_RE.sub(rb"\1<redacted>", data)


def infra_files(deep: bool = False) -> list[tuple[str, Path]]:
    """(name under the infra folder, local file) of the infra upload: INFRA_LOGS and the top level of STATE_DIR; deep
    (the study box) also its logs/ (the queue's per-item output: the store build, the anchor, every speed-probe call,
    a trainer that died before its own logger started) and each rearm-<stamp>/ (the lifecycle state a re-arm moved
    aside: the queue's events.jsonl with its prereg_numbers record among it)."""
    out = [(Path(p).name, Path(p)) for p in INFRA_LOGS]
    if STATE_DIR.is_dir():
        out += [(f.name, f) for f in sorted(STATE_DIR.glob("*"))]
        if deep:
            for sub in [STATE_DIR / "logs", *sorted(STATE_DIR.glob("rearm-*"))]:
                if sub.is_dir():
                    out += [(f.relative_to(STATE_DIR).as_posix(), f) for f in sorted(sub.rglob("*"))]
    return [(n, f) for n, f in out if f.is_file() and not f.name.endswith(".lock")]


def _tail(f: Path, limit: int) -> bytes:
    """The file's last `limit` bytes (a trainer's console log of a whole run can grow large; its end says why)."""
    with open(f, "rb") as fh:
        size = fh.seek(0, os.SEEK_END)
        fh.seek(max(0, size - limit))
        return fh.read()


def upload_infra(api, repo: str, repo_type: str, dest: str, dry_run: bool, deep: bool = False):
    """Best effort: supervisor/bootstrap/watchdog logs and state go next to the run for later extraction, scrubbed:
    portal.log (the base image's boot output: its CUDA selection is recorded nowhere else) holds the portal's password
    and tokens as soon as a PORTAL_CONFIG reaches the box, and the runs repo keeps every file in its git history.
    deep: the study box's subfolders too (infra_files), each file at most INFRA_MAX_FILE_BYTES (its end)."""
    from huggingface_hub import CommitOperationAdd

    ops = [CommitOperationAdd(path_in_repo=f"{dest}/{name}",
                              path_or_fileobj=scrub(_tail(f, INFRA_MAX_FILE_BYTES) if "/" in name else f.read_bytes()))
           for name, f in infra_files(deep)]
    log(f"upload {len(ops)} infra files -> {repo}:{dest}")
    if ops and not dry_run:  # retried inside best_effort's INFRA_TIMEOUT_S, which bounds every try together
        hub_retry(lambda: api.create_commit(repo_id=repo, repo_type=repo_type, operations=ops,
                                           commit_message="finish: infra logs"), "infra commit")


def best_effort(fn, what: str, timeout: float = INFRA_TIMEOUT_S) -> bool:
    """Run fn in a daemon thread and wait at most `timeout` s; failures are logged, never raised."""
    ok = []

    def target():
        try:
            fn()
            ok.append(True)
        except Exception as e:
            log(f"{what} failed: {type(e).__name__}: {e}")

    th = threading.Thread(target=target, name=what, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        log(f"{what} still running after {timeout:.0f} s; going on without it")
    return bool(ok)


# ----------------------------------------------------------------------------------------------------------- vast

def vast_rest(action: str, timeout: float = 30) -> bool:
    """PUT {"state": "stopped"} or DELETE on /instances/<CONTAINER_ID>/ with the per-instance key. Retried like curl
    --retry: a 408, 429, 5xx or transport error; any other 4xx (a revoked key, no such instance) is logged with vast's
    reply and handed to the CLI fallback at once."""
    key, cid = os.environ.get("CONTAINER_API_KEY"), os.environ.get("CONTAINER_ID")
    if not key or not cid:
        log("CONTAINER_API_KEY/CONTAINER_ID not set (not on a vast instance?)")
        return False
    method, body = ("DELETE", {}) if action == "destroy" else ("PUT", {"state": "stopped"})
    req = urllib.request.Request(f"{VAST_API}/instances/{cid}/", method=method, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                reply = json.loads(r.read() or b"{}")
            if reply.get("success", True):
                return True
            log(f"vast {action}: {reply.get('msg') or reply}")
        except urllib.error.HTTPError as e:  # before URLError, its base class: the status and vast's msg say why
            try:
                body = e.read()[:200].decode("utf-8", "replace")
            except Exception:
                body = ""
            log(f"vast {action} attempt {attempt} failed: HTTP {e.code}: {body or e.reason}")
            if 400 <= e.code < 500 and e.code not in (408, 429):
                return False
        except Exception as e:  # URLError, timeouts, http.client errors, a bad reply: all mean "try again"
            log(f"vast {action} attempt {attempt} failed: {type(e).__name__}: {e}")
        if attempt < 3:
            time.sleep(5 * attempt)
    return False


def vast_cli(action: str) -> bool:
    """Fallback through the vastai CLI; the key goes through the environment, never argv. A hung or missing CLI is
    a failure like any other, so the caller's destroy -> stop fallback still runs. Success is the CLI's own success
    line ("destroying instance <id>." / "stopping instance <id>."), not its exit code: vastai 1.8.0 exits 0 when the
    API refuses (success=false) or answers with an HTTP error too. A substring, not a whole line: for destroy the
    confirmation prompt that input() prints shares the line."""
    exe, key, cid = shutil.which("vastai"), os.environ.get("CONTAINER_API_KEY"), os.environ.get("CONTAINER_ID")
    if not exe or not key or not cid:
        return False
    verb = "destroy" if action == "destroy" else "stop"
    try:
        r = subprocess.run([exe, verb, "instance", cid], env=dict(os.environ, VAST_API_KEY=key),
                           input="y\n", capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError) as e:
        log(f"vastai {verb} failed: {type(e).__name__}: {e}")
        return False
    done = f"{'destroying' if verb == 'destroy' else 'stopping'} instance {cid}."
    if r.returncode == 0 and done in (r.stdout or ""):
        return True
    log(f"vastai {verb} failed (exit {r.returncode}): {((r.stdout or '') + (r.stderr or '')).strip()[-300:]}")
    return False


def instance_action(action: str, reason: str, dry_run: bool, before=None) -> bool:
    """Write the halt marker, run `before` (the last infra upload: it carries this event and the marker), then
    stop/destroy. A destroy that fails falls back to stop."""
    marker = {"action": action, "reason": reason, "wall": time.time()}
    log(f"{action} instance {os.environ.get('CONTAINER_ID', '?')}: {reason}")
    event(action, reason=reason, dry_run=dry_run)
    if not dry_run:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        (STATE_DIR / "halt").write_text(json.dumps(marker) + "\n", encoding="utf-8")
    if before is not None:
        before()
    if dry_run:
        log(f"dry run: would {action} now")
        return True
    if vast_rest(action) or vast_cli(action):
        return True
    if action == "destroy":
        log("destroy failed; stopping instead")
        return instance_action("stop", f"destroy failed ({reason})", dry_run, before)
    log("stop failed; the watchdog will retry at the deadline")
    return False


# ----------------------------------------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--destroy", action="store_true", help="upload, verify, destroy (stop if verification fails)")
    mode.add_argument("--stop", action="store_true", help="upload best-effort, then stop")
    mode.add_argument("--sync-only", action="store_true", help="upload only, no instance action")
    mode.add_argument("--verify-only", action="store_true", help="compare only, no upload and no instance action")
    ap.add_argument("--reason", default="", help="recorded in the halt marker and events")
    ap.add_argument("--repo", default=os.environ.get("KITSUNE_OUT_REPO") or None, help="HF output repo (env KITSUNE_OUT_REPO)")
    ap.add_argument("--repo-type", default=os.environ.get("KITSUNE_OUT_REPO_TYPE", "model"))
    ap.add_argument("--runs-root", default=str(ROOT / "runs"))
    ap.add_argument("--run-dir", action="append", default=None, help="restrict to these run dirs (repeatable)")
    ap.add_argument("--no-sync", action="store_true", help="skip the upload step")
    ap.add_argument("--no-hash", action="store_true", help="compare sizes only")
    ap.add_argument("--no-full", action="store_true",
                    help="do not require the full training states in the repo (the newest, and any whose trainer "
                         "upload failed)")
    ap.add_argument("--dry-run", action="store_true", help="print the actions; no upload, stop or destroy")
    ap.add_argument("--job", choices=("train", "label", "study"), default=os.environ.get("KITSUNE_JOB") or "train",
                    help="train: runs/ in the output repo; label: the label root in the data repo; study: runs/ in the "
                         "output repo, lean (env KITSUNE_JOB)")
    ap.add_argument("--lean", action="store_true",
                    help="the study box's uploads (the default for --job study): logs, every weights dir and only the "
                         "full states the run's config uploads (expected_files lean), not the newest one")
    ap.add_argument("--no-infra", action="store_true", help="label: skip the infra log upload")
    ap.add_argument("--allow-empty", action="store_true",
                    help="label: --destroy with no finished label file (nothing unique on the disk yet)")
    args = ap.parse_args(argv)
    if args.job == "label":
        return label_main(args)

    dirs = [Path(d) for d in args.run_dir] if args.run_dir else run_dirs(Path(args.runs_root))
    lean = args.lean or args.job == "study"
    expect_full = not args.no_full and not lean
    api = None
    if args.repo:
        try:
            api = hf_api()
        except Exception as e:
            log(f"huggingface_hub unavailable: {e}")
    log(f"mode={'destroy' if args.destroy else 'stop' if args.stop else 'sync-only' if args.sync_only else 'verify-only'} "
        f"repo={args.repo} run_dirs={[d.name for d in dirs]} dry_run={args.dry_run}")

    if api is not None and not args.no_sync and not args.verify_only:
        for d in dirs:
            try:
                sync(api, args.repo, args.repo_type, d, expect_full, args.dry_run, lean)
            except Exception as e:  # a failed upload shows up as a verification problem below
                log(f"sync of {d.name} failed: {type(e).__name__}: {e}")
                event("sync_failed", run=d.name, error=f"{type(e).__name__}: {e}")
    if args.job == "study":  # one box, many run dirs: its logs and queue state under the box's own folder
        infra_dest = f"study/box-{os.environ.get('KITSUNE_BOX') or 'unknown'}/infra/" \
                     f"{os.environ.get('CONTAINER_ID', 'local')}"
    else:
        infra_dest = f"runs/{dirs[-1].name}/infra" if dirs else f"infra/{os.environ.get('CONTAINER_ID', 'local')}"

    def push_infra():  # best effort and bounded; --no-sync too (a failed bootstrap's reason is in these files)
        if api is not None and not args.verify_only:
            best_effort(lambda: upload_infra(api, args.repo, args.repo_type, infra_dest, args.dry_run,
                                             deep=args.job == "study"), "infra upload")

    if args.sync_only:
        push_infra()
        return 0
    if args.stop:
        return 0 if instance_action("stop", args.reason or "requested", args.dry_run, push_infra) else 1

    expected = {}
    for d in dirs:
        expected.update(expected_files(d, expect_full, lean))
    if api is None:
        problems = ["no HF output repo (KITSUNE_OUT_REPO unset) or huggingface_hub unavailable"]
    else:
        problems = verify(api, args.repo, args.repo_type, expected, check_hash=not args.no_hash)
    total_gb = sum(p.stat().st_size for p in expected.values()) / 1e9
    log(f"verification: {len(expected)} files ({total_gb:.2f} GB), {len(problems)} problem(s)")
    for p in problems[:50]:
        log(f"  {p}")
    event("verify", files=len(expected), gigabytes=round(total_gb, 3), problems=problems[:200])
    if args.verify_only:
        return 0 if not problems else 1
    if problems:
        why = f"verification failed ({len(problems)} problems); {args.reason}".rstrip("; ")
        return 2 if instance_action("stop", why, args.dry_run, push_infra) else 1
    return 0 if instance_action("destroy", args.reason or "run verified on the hub", args.dry_run, push_infra) else 1


# ---------------------------------------------------------------------------------------------------------- label

def label_prefix(config: str) -> str:
    """extent.root of the run config (a relative path: the cwd, else the checkout), e.g. labels/full."""
    p = Path(config)
    if not p.is_absolute() and not p.exists():
        p = ROOT / config
    root = json.loads(p.read_text(encoding="utf-8"))["extent"]["root"]
    if not isinstance(root, str) or not root.strip("/"):
        raise ValueError(f"{config}: extent.root {root!r} is not a repo path")
    return root.strip("/")


def label_main(args) -> int:
    """finish.py --job label (module docstring): sync the label root write-once, verify it per directory, then stop
    or destroy as the train path does."""
    import label_sync  # vast/ is on sys.path: this file's directory (python vast/finish.py), or the importer's

    repo = os.environ.get("KITSUNE_DATA_REPO") or args.repo
    repo_type = label_sync.REPO_TYPE
    local_root = Path(os.environ.get("KITSUNE_DIR") or ROOT)
    config = os.environ.get("KITSUNE_CONFIG") or "configs/full.json"
    try:
        prefix = label_prefix(config)
    except Exception as e:  # noqa: BLE001  (no sync and a verify problem; a stop still stops)
        log(f"cannot read the label root from {config}: {type(e).__name__}: {e}")
        prefix = None
    rid = label_sync.run_id(STATE_DIR)
    ledger = STATE_DIR / "label_ledger.json"
    infra_dest = f"label_runs/{rid}"
    mode = "destroy" if args.destroy else "stop" if args.stop else "sync-only" if args.sync_only else "verify-only"
    api = None
    if repo:
        try:
            api = hf_api()
        except Exception as e:
            log(f"huggingface_hub unavailable: {e}")
    log(f"mode={mode} job=label repo={repo} root={prefix} run_id={rid} dry_run={args.dry_run}")

    def push_infra():  # best effort and bounded, as in the train path
        if api is not None and not args.verify_only and not args.no_infra:
            best_effort(lambda: upload_infra(api, repo, repo_type, infra_dest, args.dry_run), "infra upload")

    lock = None  # held until this process exits: the sync and the verify after it share it
    if not args.verify_only and not args.no_sync:
        lock = label_sync.acquire_sync_lock(STATE_DIR / "sync.lock", SYNC_LOCK_WAIT_S)
        if lock is None:
            log(f"{STATE_DIR / 'sync.lock'} still busy after {SYNC_LOCK_WAIT_S} s; skipping the sync")
            event("sync_skipped", reason="sync.lock busy")

    def do_sync(released: bool) -> int:
        """0 synced, 1 an upload failed (or no sync was possible), EXIT_INTEGRITY a write-once refusal."""
        if lock is None:
            return 1
        if api is None or prefix is None:
            log("no data repo (KITSUNE_DATA_REPO unset), huggingface_hub unavailable or no label root: no sync")
            event("sync_failed", error="no repo, hub or label root")
            return 1
        lease = label_sync.lease_bytes(rid, os.environ.get("CONTAINER_ID") or "local",
                                       os.environ.get("KITSUNE_MACHINE_ID") or "", os.environ.get("KITSUNE_SHA") or "",
                                       released=released)
        try:
            res = label_sync.sync(api, repo, local_root, prefix, ledger, lease=lease, dry_run=args.dry_run, log=log)
        except label_sync.IntegrityError as e:
            log(f"label sync refused: {e}")
            event("sync_integrity", error=str(e)[:4000])
            return EXIT_INTEGRITY
        except Exception as e:  # noqa: BLE001
            log(f"label sync failed: {type(e).__name__}: {e}")
            event("sync_failed", error=f"{type(e).__name__}: {e}"[:4000])
            return 1
        event("label_sync", **res)
        return 0

    if args.sync_only:
        rc = 0 if args.no_sync else do_sync(False)
        push_infra()
        return rc
    if args.stop:
        if not args.no_sync:
            do_sync(True)
        return 0 if instance_action("stop", args.reason or "requested", args.dry_run, push_infra) else 1
    if args.destroy and not args.no_sync:
        do_sync(True)

    expected = label_sync.finished_files(local_root, prefix) if prefix is not None else {}
    if api is None or prefix is None:
        problems = ["no data repo (KITSUNE_DATA_REPO unset), huggingface_hub unavailable or no label root"]
    elif not expected:
        problems = [] if args.allow_empty else [f"nothing to verify: no finished label file under {local_root / prefix}"]
    else:
        problems = label_sync.verify(api, repo, expected, ledger, check_hash=not args.no_hash)
    total_gb = sum(p.stat().st_size for p in expected.values()) / 1e9
    log(f"verification: {len(expected)} files ({total_gb:.2f} GB), {len(problems)} problem(s)")
    for p in problems[:50]:
        log(f"  {p}")
    event("verify", files=len(expected), gigabytes=round(total_gb, 3), problems=problems[:200])
    if args.verify_only:
        return 0 if not problems else 1
    if problems:
        why = f"verification failed ({len(problems)} problems); {args.reason}".rstrip("; ")
        return 2 if instance_action("stop", why, args.dry_run, push_infra) else 1
    return 0 if instance_action("destroy", args.reason or "labels verified on the hub", args.dry_run, push_infra) else 1


if __name__ == "__main__":
    sys.exit(main())
