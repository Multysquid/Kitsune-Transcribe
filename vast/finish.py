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
    hub skips re-uploads of content it already has)
  the newest full state   checkpoints/full_step_<N>/   -> runs/<run_id>/checkpoints/full_step_<N>/  (--expect-full)
  and every full state the trainer meant for the Hub whose upload has not succeeded (UPLOAD_MARK in it: the
    pre_cooldown one, which the trainer keeps from rotation for this; the marker itself is not uploaded)
The files outside checkpoints/ are uploaded from a snapshot copy: the watchdog's --sync-only runs while the trainer
still appends to its logs, and a file handed to the Hub by path is sized when it is listed but hashed and read later
(a growing file would go up as a size/hash/content mismatch). Checkpoint dirs are renamed into place complete and go
up in place; the one later write, the trainer's end save replacing trainer.pt/.json of an existing full_step_<N> (the
loop ended at the step of a full state), is over once the trainer has exited.
Supervisor/watchdog/bootstrap logs are uploaded best-effort to runs/<run_id>/infra/ but not verified (they are still
being written while this runs); on every path, --no-sync included, and once more right before the stop/destroy call,
so the last lifecycle records (verify, stop/destroy) are off the box before its disk goes away or stays behind.

Every decision is appended to $KITSUNE_STATE/events.jsonl, and a `halt` marker is written before a stop/destroy so a
restarted container does not start a second run (vast/onstart.sh checks it).

Usage:
  python vast/finish.py --destroy                 # upload, verify, destroy (or stop if verification fails)
  python vast/finish.py --stop --reason "..."     # upload best-effort, stop
  python vast/finish.py --sync-only               # upload only (supervisor after a crash, watchdog before the cap)
  python vast/finish.py --verify-only             # print the comparison, exit 1 on problems
  add --dry-run to any of them to print the actions without uploading, stopping or destroying
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
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VAST_API = "https://console.vast.ai/api/v0"
STATE_DIR = Path(os.environ.get("KITSUNE_STATE", "/workspace/kitsune_state"))
INFRA_LOGS = ["/workspace/kitsune.log", "/workspace/watchdog.log", "/workspace/portal.log", "/workspace/tensorboard.log"]
INFRA_TIMEOUT_S = 180  # a hung infra upload must not keep a paid instance up
# checkpoint names written by the trainer; a directory or a single file (full_step_N.pt). In-progress writes end in
# .tmp/.partial and never match.
WEIGHTS_RE = re.compile(r"^step[_-]?(\d+)$")
FULL_RE = re.compile(r"^full[_-]?(?:step[_-]?)?(\d+)(?:\.(?!tmp$|partial$)[A-Za-z0-9]+)?$")
# scripts/04_distill.py's marker in a full state meant for the Hub (ckpt.upload_full_at), removed once its upload
# succeeded
UPLOAD_MARK = ".upload_pending"
HASH_CHUNK = 8 << 20


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


def expected_files(run_dir: Path, expect_full: bool = True) -> dict[str, Path]:
    """repo path -> local file, for one run dir (see the module docstring)."""
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


def remote_listing(api, repo: str, repo_type: str, prefix: str) -> dict:
    """repo path -> RepoFile for every file under prefix (folders are skipped: they have no size)."""
    out = {}
    for item in api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, repo_type=repo_type):
        if getattr(item, "size", None) is not None:
            out[item.path] = item
    return out


def verify(api, repo: str, repo_type: str, expected: dict[str, Path], check_hash: bool = True) -> list[str]:
    """Compare expected local files with the repo. Returns problems; an empty list means every file is there."""
    if not expected:
        return ["nothing to verify: no run directory with files was found"]
    problems = []
    listings: dict[str, dict] = {}
    for path, local in sorted(expected.items()):
        prefix = "/".join(path.split("/")[:2])  # runs/<run_id>
        if prefix not in listings:
            try:
                listings[prefix] = remote_listing(api, repo, repo_type, prefix)
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


def sync(api, repo: str, repo_type: str, run_dir: Path, expect_full: bool, dry_run: bool):
    """Upload the run dir (minus local-only checkpoints) plus the checkpoints finish will verify.

    Re-uploading files the trainer already pushed is cheap: the hub skips content it already stores, so this mostly
    repairs a checkpoint upload that failed and captures log lines written after the trainer's last sync. The logs go
    first: they are small, the watchdog's --sync-only has 10 minutes before the stop, which a ~9 GB full state not
    yet on the hub can take all of (the final eval and verdict of a trainer still waiting on its end-state upload
    are on this disk only), and a checkpoint upload that raises must not cost them."""
    expected = expected_files(run_dir, expect_full)
    rels = sorted(p[len(f"runs/{run_dir.name}/"):] for p in expected)
    ckpt = [r for r in rels if r.startswith("checkpoints/")]
    live = [r for r in rels if not r.startswith("checkpoints/")]
    dest = f"runs/{run_dir.name}"
    log(f"sync {run_dir} -> {repo}:{dest} ({len(live)} files + {len(ckpt)} checkpoint files)")
    if dry_run:
        return
    if live:  # logs the trainer may still be writing: a consistent copy (module docstring)
        with tempfile.TemporaryDirectory(prefix="kitsune-finish-") as stage:
            snapshot(run_dir, live, Path(stage))
            api.upload_folder(repo_id=repo, repo_type=repo_type, folder_path=stage, path_in_repo=dest,
                              allow_patterns=live, commit_message=f"finish: sync {run_dir.name}")
    if ckpt:  # renamed into place when complete (the end save may still replace trainer.pt/.json): uploaded in place
        api.upload_folder(repo_id=repo, repo_type=repo_type, folder_path=str(run_dir), path_in_repo=dest,
                          allow_patterns=ckpt, commit_message=f"finish: checkpoints {run_dir.name}")


def upload_infra(api, repo: str, repo_type: str, dest: str, dry_run: bool):
    """Best effort: supervisor/bootstrap/watchdog logs and state go next to the run for later extraction."""
    from huggingface_hub import CommitOperationAdd

    files = [Path(p) for p in INFRA_LOGS] + (sorted(STATE_DIR.glob("*")) if STATE_DIR.is_dir() else [])
    ops = [CommitOperationAdd(path_in_repo=f"{dest}/{f.name}", path_or_fileobj=f.read_bytes())
           for f in files if f.is_file() and not f.name.endswith(".lock")]
    log(f"upload {len(ops)} infra files -> {repo}:{dest}")
    if ops and not dry_run:
        api.create_commit(repo_id=repo, repo_type=repo_type, operations=ops, commit_message="finish: infra logs")


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
    """PUT {"state": "stopped"} or DELETE on /instances/<CONTAINER_ID>/ with the per-instance key."""
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
        except Exception as e:  # URLError, timeouts, http.client errors, a bad reply: all mean "try again"
            log(f"vast {action} attempt {attempt} failed: {type(e).__name__}: {e}")
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
    args = ap.parse_args(argv)

    dirs = [Path(d) for d in args.run_dir] if args.run_dir else run_dirs(Path(args.runs_root))
    expect_full = not args.no_full
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
                sync(api, args.repo, args.repo_type, d, expect_full, args.dry_run)
            except Exception as e:  # a failed upload shows up as a verification problem below
                log(f"sync of {d.name} failed: {type(e).__name__}: {e}")
                event("sync_failed", run=d.name, error=f"{type(e).__name__}: {e}")
    infra_dest = f"runs/{dirs[-1].name}/infra" if dirs else f"infra/{os.environ.get('CONTAINER_ID', 'local')}"

    def push_infra():  # best effort and bounded; --no-sync too (a failed bootstrap's reason is in these files)
        if api is not None and not args.verify_only:
            best_effort(lambda: upload_infra(api, args.repo, args.repo_type, infra_dest, args.dry_run), "infra upload")

    if args.sync_only:
        push_infra()
        return 0
    if args.stop:
        return 0 if instance_action("stop", args.reason or "requested", args.dry_run, push_infra) else 1

    expected = {}
    for d in dirs:
        expected.update(expected_files(d, expect_full))
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


if __name__ == "__main__":
    sys.exit(main())
