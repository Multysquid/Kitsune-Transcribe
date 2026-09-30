"""The download gate of a full-data box (fix 1): time a real upstream download before the paid multi-hour rebuild.

Why: the audio rebuild (scripts/01_prepare_data.py --extent-config) downloads the extent's upstream audio from the Hub,
~571 GB for the full extent. Study box A #1 (machine 151760) had a ~2.9 MB/s Hub link: three 67-minute rebuild attempts
failed before anyone noticed, all of it billed. So a full box times the same download path first, on three pinned
upstream files (~2.5 GB, the three biggest upstreams, exactly as 01 fetches them: hf_hub_download of the pinned
revision into a cache dir, one file at a time, with the bootstrap's HF_XET_HIGH_PERFORMANCE), and refuses a host that
could not pull the reference 571.2 GB within the gate's ceiling (5 h by default): 31.7 MB/s. The refusal exits 3, which
bootstrap's retry() does not repeat; onstart then runs `finish.py --abort`, which destroys a box that has no run dir yet
(nothing on its disk is unique) and records the slow gate, so vast/launch.py avoids that machine for GATE_BLOCK_DAYS.

The measured rate also sizes the bootstrap's timeouts (timeouts): the per-attempt rebuild timeout never below launch's
40 MB/s sizing (kitsune.extent: 01 is CPU-bound near that rate even on a fast link), the background label pull's never
below 30 min.

  python -m kitsune.netgate --out $STATE/download_gate.json --dir $STATE/netgate    # exit 0 pass, 3 slow, 1 error
  python -m kitsune.netgate --timeouts $STATE/download_gate.json                    # prints "PULL_MIN REBUILD_MIN"

The gate reads KITSUNE_GATE_BYTES (the bytes it judges: max(the extent's upstream bytes, GATE_REF_GB)), KITSUNE_GATE_MAX_H
(the ceiling in hours), KITSUNE_REBUILD_BYTES (this box's own upstream bytes) and KITSUNE_PULL_BYTES (its label bytes),
all set by vast/launch.py; KITSUNE_MACHINE_ID and KITSUNE_CPU_QUOTA go into the record. $STATE/download_gate.json
(format 1) reaches the runs repo with the box's infra files (full/box-<box>/infra/<container>/download_gate.json).

A slow host is cut early: the samples get CUTOFF_FACTOR x the time they may take at the minimum rate (~157 s), so a
2.9 MB/s host is refused after under three minutes instead of the ~15 its samples would take. The gate times the link,
never a cache: hf_xet keeps a chunk cache under HF_XET_CACHE (default $HF_HOME/xet) that outlives the --dir cache, so
the gate runs with that cache off and in a dir of its own (<--dir>.xet, removed with --dir); otherwise a gate retried
after an error (bootstrap's retry 2) could read the samples the first attempt fetched at disk speed and pass a slow
host. Stdlib only at import
(huggingface_hub is imported where the download runs; kitsune.extent's constants where the timeouts are computed).
"""
import argparse
import contextlib
import json
import math
import os
import shutil
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

GATE_FORMAT = 1
GATE_REF_GB = 571.2  # the full extent's upstream download (labels/full/extent.json): every full box is judged on it
DEFAULT_MAX_H = 5.0  # the ceiling: the reference download must fit in this many hours (decision 11)
CUTOFF_FACTOR = 2.0  # the samples may take this x their time at the minimum rate before the host counts as slow
POLL_S = 1.0  # how often a running download is checked against the cutoff


@dataclass(frozen=True)
class Sample:
    repo: str
    revision: str
    filename: str
    bytes: int


# the first input of the three biggest upstreams, at 01's pinned revisions (scripts/01_prepare_data.py REVISIONS; a
# test asserts they are equal) and with the sizes the label box recorded (labels/full/extent.json input bytes)
SAMPLES = (
    Sample("japanese-asr/whisper_transcriptions.reazonspeech.large", "4ad8d64a13594f0ce1f0622627a18ef99b42b5e8",
           "large/train-00000-of-00705.parquet", 493_409_818),
    Sample("litagin/Galgame_Speech_ASR_16kHz", "3fb86654222b3f0af0f7c332ae6a0ef9752a9451",
           "data/galgame-speech-asr-16kHz-train-000000.tar", 941_219_840),
    Sample("laion/Emolia", "4d375b4bf276834e555022bb7c937f87091e895e", "JA-B000000_standard.tar.gz", 1_049_805_733),
)

EXIT_PASS, EXIT_ERROR, EXIT_SLOW = 0, 1, 3
# hf_xet's chunk cache off for the gate (its dir is the gate's own too, see _own_xet_cache); read when the process's
# Xet session starts, i.e. at the first download
XET_CHUNK_CACHE_ENV = "HF_XET_CHUNK_CACHE_SIZE_BYTES"
XET_CACHE_ENV = "HF_XET_CACHE"


def min_rate(gate_bytes: float, max_h: float) -> float:
    """The slowest download rate (bytes/s) at which gate_bytes fit in max_h hours: 571.2e9 / 18000 = 31.7 MB/s."""
    return float(gate_bytes) / (float(max_h) * 3600.0)


def _hf_download(repo: str, filename: str, *, revision: str, cache_dir: str) -> str:
    from huggingface_hub import hf_hub_download  # 01's own call (Ingest.download), so the gate times the same path

    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=cache_dir)


@contextlib.contextmanager
def _own_xet_cache(path: Path):
    """While the samples download: hf_xet's chunk cache off (HF_XET_CHUNK_CACHE_SIZE_BYTES=0) and its cache dir
    HF_XET_CACHE = path, a dir of this gate run alone that the caller removes afterwards. Both win over any value the
    box set: a cache that survives a gate attempt would time the disk, not the link. The previous values come back on
    exit (the gate's own process ends there anyway)."""
    keep = {k: os.environ.get(k) for k in (XET_CHUNK_CACHE_ENV, XET_CACHE_ENV)}
    os.environ[XET_CHUNK_CACHE_ENV] = "0"
    os.environ[XET_CACHE_ENV] = str(path)
    try:
        yield
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _dir_bytes(d: Path) -> int:
    """Bytes under d now (a cut download's partial file included)."""
    try:
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    except OSError:
        return 0


def _delete(local: Path):
    """A timed file and the blob behind it (the cache's snapshot entry may be a symlink or a copy)."""
    try:
        real = local.resolve()
        local.unlink(missing_ok=True)
        if real != local:
            real.unlink(missing_ok=True)
    except OSError:
        pass


def measure(samples, dest: Path, *, cutoff_s: float, download=_hf_download, clock=time.monotonic,
            poll_s: float = POLL_S) -> dict:
    """Download the samples one after the other into dest (a cache dir), each in a daemon thread polled against the
    total cutoff_s; each file's size is checked against the sample's and the file deleted once timed. Returns
    {complete, error, samples: [{repo, revision, filename, bytes, seconds, mb_s}], bytes, seconds}: complete False
    when the cutoff came first (bytes then counts what had arrived, the cut sample's partial file included), error a
    string when a download raised or a file has the wrong size (the gate then says error, not slow: a retry may fix
    it)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = {"complete": False, "error": None, "samples": [], "bytes": 0, "seconds": 0.0}
    t_start = clock()
    for s in samples:
        box: dict = {}

        def run(s=s, box=box):
            try:
                box["path"] = download(s.repo, s.filename, revision=s.revision, cache_dir=str(dest))
            except BaseException as e:  # noqa: BLE001  reported, not raised, from the thread
                box["error"] = f"{type(e).__name__}: {e}"

        t0 = clock()
        th = threading.Thread(target=run, name="netgate-download", daemon=True)
        th.start()
        while th.is_alive():
            left = cutoff_s - (clock() - t_start)
            if left <= 0:
                break
            th.join(min(poll_s, left))
        if th.is_alive():  # the cutoff: the host is slower than the gate allows; the thread dies with the process
            out["seconds"] = clock() - t_start
            out["bytes"] += max(0, _dir_bytes(dest))
            out["cut"] = s.filename
            return out
        seconds = max(clock() - t0, 1e-9)
        if "error" in box:
            out["error"] = f"{s.repo}/{s.filename}: {box['error']}"
            out["seconds"] = clock() - t_start
            return out
        local = Path(box["path"])
        size = local.stat().st_size if local.is_file() else -1
        _delete(local)
        if size != s.bytes:
            out["error"] = f"{s.repo}/{s.filename}: {size} bytes on disk, the pinned file has {s.bytes}"
            out["seconds"] = clock() - t_start
            return out
        if clock() - t_start > cutoff_s:  # it arrived, but past the cutoff (between two polls): as slow as a cut one
            out["seconds"], out["cut"] = clock() - t_start, s.filename
            out["bytes"] += s.bytes
            return out
        out["samples"].append(dict(asdict(s), seconds=round(seconds, 3), mb_s=round(s.bytes / seconds / 1e6, 2)))
        out["bytes"] += s.bytes
    out["seconds"] = clock() - t_start
    out["complete"] = True
    return out


def timeouts(rate: float, rebuild_bytes: float, pull_bytes: float) -> tuple[int, int]:
    """(pull_min, rebuild_min) at the measured rate: the per-attempt timeout of the background label pull
    (max(30, ceil(30 + 1.5 x pull_bytes / rate / 60))) and of the audio rebuild (max(launch's sizing at 40 MB/s,
    ceil(30 + 1.5 x rebuild_bytes / rate / 60))). kitsune.extent's REBUILD_* constants are the ones launch sizes with."""
    from kitsune.extent import REBUILD_BASE_MIN, REBUILD_BYTES_PER_S, REBUILD_SLACK

    rate = max(float(rate), 1.0)

    def at(r: float, b: float) -> int:
        return int(math.ceil(REBUILD_BASE_MIN + REBUILD_SLACK * float(b) / r / 60))

    return max(30, at(rate, pull_bytes)), max(at(REBUILD_BYTES_PER_S, rebuild_bytes), at(rate, rebuild_bytes))


def verdict(meas: dict, *, gate_bytes: float, max_h: float, rebuild_bytes: float, pull_bytes: float) -> dict:
    """The gate record from a measurement: pass when the measured rate would pull gate_bytes within max_h hours, slow
    when not (or when the cutoff came first), error when a download failed. The timeouts come from the rate (for a
    slow or failed gate too, as a record of what they would have been)."""
    floor = min_rate(gate_bytes, max_h)
    seconds = max(float(meas.get("seconds") or 0.0), 1e-9)
    rate = float(meas.get("bytes") or 0) / seconds
    projected_h = gate_bytes / rate / 3600 if rate > 0 else None
    if meas.get("error"):
        v, reason = "error", f"download failed: {meas['error']}"
    elif not meas.get("complete"):
        v = "slow"
        reason = (f"cut after {seconds:.0f} s ({CUTOFF_FACTOR:g} x the samples' time at the minimum rate), "
                  f"{rate / 1e6:.1f} MB/s so far < {floor / 1e6:.1f} MB/s: {gate_bytes / 1e9:.1f} GB would take "
                  f"{projected_h if projected_h is None else round(projected_h, 1)} h > {max_h:g} h")
    elif rate < floor:
        v = "slow"
        reason = (f"{rate / 1e6:.1f} MB/s < {floor / 1e6:.1f} MB/s: {gate_bytes / 1e9:.1f} GB would take "
                  f"{projected_h:.1f} h > {max_h:g} h")
    else:
        v = "pass"
        reason = (f"{rate / 1e6:.1f} MB/s >= {floor / 1e6:.1f} MB/s: {gate_bytes / 1e9:.1f} GB in ~{projected_h:.1f} h "
                  f"<= {max_h:g} h")
    # the timeouts only from a rate the samples measured (a failed gate has none)
    pull_min, rebuild_min = timeouts(rate, rebuild_bytes, pull_bytes) if rate > 0 and v != "error" else (None, None)
    return {"format": GATE_FORMAT, "verdict": v, "reason": reason, "samples": meas.get("samples", []),
            "rate_bytes_s": round(rate, 1), "min_rate_bytes_s": round(floor, 1), "gate_bytes": int(gate_bytes),
            "gate_max_h": float(max_h), "projected_h": None if projected_h is None else round(projected_h, 3),
            "rebuild_bytes": int(rebuild_bytes), "rebuild_timeout_min": rebuild_min, "pull_bytes": int(pull_bytes),
            "pull_timeout_min": pull_min, "machine_id": os.environ.get("KITSUNE_MACHINE_ID") or None,
            "cpu_quota": os.environ.get("KITSUNE_CPU_QUOTA") or None, "nproc": os.cpu_count(), "wall": time.time(),
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _env_num(name: str, default=None) -> float | None:
    v = os.environ.get(name)
    if v in (None, ""):
        return default
    return float(v)


def _write(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv=None, *, download=_hf_download, clock=time.monotonic) -> int:
    ap = argparse.ArgumentParser(prog="python -m kitsune.netgate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="write the gate record here (the box: $STATE/download_gate.json)")
    ap.add_argument("--dir", default=None,
                    help="the download cache dir (removed after the gate, with the Xet cache dir <dir>.xet beside it)")
    ap.add_argument("--timeouts", default=None, metavar="GATE_JSON",
                    help="print the passed gate's 'PULL_MIN REBUILD_MIN' and exit")
    args = ap.parse_args(argv)
    if args.timeouts:
        try:
            rec = json.loads(Path(args.timeouts).read_text(encoding="utf-8"))
            if rec.get("verdict") != "pass":
                raise ValueError(f"verdict {rec.get('verdict')!r}, not pass")
            pull, rebuild = int(rec["pull_timeout_min"]), int(rec["rebuild_timeout_min"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"netgate: no timeouts from {args.timeouts}: {type(e).__name__}: {e}", file=sys.stderr)
            return EXIT_ERROR
        print(f"{pull} {rebuild}")
        return EXIT_PASS
    if not args.out or not args.dir:
        ap.error("--out and --dir are required (or --timeouts GATE_JSON)")
    try:
        gate_bytes = _env_num("KITSUNE_GATE_BYTES", GATE_REF_GB * 1e9)
        max_h = _env_num("KITSUNE_GATE_MAX_H", DEFAULT_MAX_H)
        rebuild_bytes = _env_num("KITSUNE_REBUILD_BYTES", gate_bytes)
        pull_bytes = _env_num("KITSUNE_PULL_BYTES", 0.0)
        if not (gate_bytes > 0 and max_h > 0 and rebuild_bytes >= 0 and pull_bytes >= 0):
            raise ValueError("gate bytes and hours must be > 0, the rebuild and pull bytes >= 0")
    except ValueError as e:
        print(f"netgate: bad KITSUNE_GATE_* / KITSUNE_*_BYTES env: {e}", file=sys.stderr)
        return EXIT_ERROR
    total = sum(s.bytes for s in SAMPLES)
    cutoff_s = CUTOFF_FACTOR * total / min_rate(gate_bytes, max_h)
    print(f"netgate: {len(SAMPLES)} samples, {total / 1e9:.2f} GB; the minimum is "
          f"{min_rate(gate_bytes, max_h) / 1e6:.1f} MB/s ({gate_bytes / 1e9:.1f} GB in {max_h:g} h), cutoff "
          f"{cutoff_s:.0f} s", flush=True)
    dest = Path(args.dir)
    xet = dest.with_name(dest.name + ".xet")  # beside dest, so a cut sample's partial bytes count only the sample
    try:
        with _own_xet_cache(xet):
            meas = measure(SAMPLES, dest, cutoff_s=cutoff_s, download=download, clock=clock)
    finally:
        shutil.rmtree(dest, ignore_errors=True)
        shutil.rmtree(xet, ignore_errors=True)
    rec = verdict(meas, gate_bytes=gate_bytes, max_h=max_h, rebuild_bytes=rebuild_bytes, pull_bytes=pull_bytes)
    for s in rec["samples"]:
        print(f"netgate:   {s['repo']}/{s['filename']}: {s['bytes'] / 1e9:.2f} GB in {s['seconds']:.1f} s "
              f"({s['mb_s']:.1f} MB/s)")
    print(f"netgate: {rec['verdict']}: {rec['reason']}", flush=True)
    try:
        _write(Path(args.out), rec)
    except OSError as e:
        print(f"netgate: cannot write {args.out}: {e}", file=sys.stderr)
        return EXIT_ERROR
    return {"pass": EXIT_PASS, "slow": EXIT_SLOW}.get(rec["verdict"], EXIT_ERROR)


if __name__ == "__main__":
    sys.exit(main())
