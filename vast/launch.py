"""Rent one A100 on vast.ai for the viability run: search offers, show them, and create the instance only with --yes.

Run this on the laptop. It
  1. searches on-demand offers with the strict host filter (D45a): 1x A100 SXM4 40 GB, verified, reliability >= 0.98,
     driver CUDA >= 13.0 (the image's torch is cu130), >= 12 effective CPU cores, >= 64 GB RAM, disk_bw, inet_down and
     inet_up >= 500, a direct port and room for the 150 GB disk, sorted by $/h; if none, the same filter on SXM4 80 GB;
  2. prints the offers and the exact `vastai create instance` command: the image by digest, --disk 150, --ssh --direct,
     the KITSUNE_* env, TZ=UTC and --onstart vast/onstart_stub.sh (which clones the repo and runs vast/onstart.sh);
  3. creates the instance only with --yes. Spending money is always the user's explicit step.
The git, image and HF checks run before the offer search, so they report problems even without the vastai CLI. The
image check reads the commit the image was built from (its org.opencontainers.image.revision label) and refuses it when
requirements-train.txt or docker/Dockerfile differ at the commit to run (the build is still running or failed). The HF
check refuses a data repo whose derived data is incomplete: a train source's teacher shard without its second opinion,
or a selection that still drops rows as no_agree (both mean training on a fraction of the planned hours); a selection
not built from the run config's sources, eval sets and selection_recipe, or with no kept rows for one of them
(make_selection.py --config builds the right one); and a selection whose kept rows point at teacher output the repo
lacks. It also refuses a data or output repo that is not private (both carry dataset reference transcripts).

HF_TOKEN is never an argument and never part of the command: the box gets it from the vast ACCOUNT-level environment
variables (D48a), so it does not appear in shell history, the process list or the instance config. The instance runs
the code at a pinned commit (KITSUNE_SHA), so this script refuses to rent for a commit GitHub does not have.

Usage:
  python vast/launch.py --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs        # look only
  python vast/launch.py --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --yes  # rent the cheapest
Add --no-self-stop to debug a fresh box: a failed on-start or bootstrap then leaves it running (KITSUNE_NO_SELF_STOP=1).

A config with an `extent` (configs/full.json, configs/full_sub3k.json) trains on the label box's labels: the HF check
then reads <extent.root>/extent.json at the pinned revision, requires COMPLETE.json and every file of the subset
(extent_problems), and sizes the disk, the rebuild timeout (KITSUNE_REBUILD_TIMEOUT_MIN) and --max-hours from it.

--job label rents one RTX 5090 for the label box (vast/label.py, docs in vast/README.md): its own tiers and host
filter, a 250 GB disk, offers ranked by the estimated total (GPU hours + traffic), machines that failed a label run
avoided, and a read-only preflight of the label root (lease, COMPLETE.json, seeds, Parakeet pins, extents):
  python vast/launch.py --job label --data-repo Multy123/kitsune-data --image-tag main        # look only

--job study --box A|B|replicate|shakedown rents a size-study box (kitsune/study_queue.py runs it; vast/README.md "Size
study"): the relaxed per-launch filter (study_filter: 4 GPUs for A and B, 1 for the replicate and the shakedown, any
A100 40 GB then 80 GB, reliability >= STUDY_RELIABILITY), the box's hours (STUDY_HOURS) as its cap, the disk from the
extent's sizing with both label stores and the box's checkpoints (study_queue.study_extra_gb), and study_preflight's
refusals: a PREREG with pending fields (not for the shakedown), a selection whose sha256 is not PREREG's, a student of
the box missing a file, a config of the box not committed, a numbers file of the box already written (or, for the
replicate, box A's missing or written under other rules); the queue's GPU count goes along (KITSUNE_N_GPUS):
  python vast/launch.py --job study --box A --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs \\
      --image-tag main                                                                        # look only
Needs the vastai CLI (`pip install vastai==1.8.0`, then `vastai set api-key <key>`); --help works without it.
"""
import argparse
import json
import math
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # kitsune.extent (pure Python) when run as `python vast/launch.py`
    sys.path.insert(1, str(ROOT))

VASTAI_PIN = "vastai==1.8.0"
IMAGE_REPO = "ghcr.io/multysquid/kitsune-train"
MANIFEST_ACCEPT = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.oci.image.manifest.v1+json",
                             "application/vnd.docker.distribution.manifest.list.v2+json",
                             "application/vnd.docker.distribution.manifest.v2+json"])
# what the image is built from (docker/smoke_import.py only runs in CI, never on the box)
IMAGE_INPUTS = ("requirements-train.txt", "docker/Dockerfile")
# the image (~12 GB unpacked, if vast counts it) + audio 23 GB + train/eval caches ~22 GB + derived data 2.3 GB + up to
# 4 full states x 8.6 GB while rotating/uploading + ~9 weights x 1.2 GB: 80 GB fills mid-run, 150 GB leaves headroom
DISK_GB = 150
# traffic of one run for the cost line: ~25 GB of upstream audio down, ~30 GB of checkpoints and logs up
EST_DOWN_GB, EST_UP_GB = 25, 30
ONSTART = Path(__file__).resolve().parent / "onstart_stub.sh"
# vast's OpenAPI create-instance doc caps the onstart field at 4048 chars (gzip+base64 beyond that), its CLI guide says
# 16 KB, and neither says whether a longer script is truncated or rejected: stay under the smaller limit
ONSTART_MAX_BYTES = 4000
DEFAULT_MAX_DPH = 2.0
# the student dir files the trainer loads (03_build_student saves them all; the trainer has no processor fallback and
# reads student_meta.json; README.md is the model card with the modification notice the uploads carry). The same list
# is STUDENT_FILES in vast/bootstrap.sh's helper. A Parakeet-derived student also carries CTC_CARD (its CC-BY-4.0
# attribution)
STUDENT_FILES = ("config.json", "model.safetensors", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
                 "student_meta.json", "README.md")
CTC_CARD = "MODEL_CARD.md"
# D45a strict filter; vast converts cpu_ram/gpu_ram GB to MB itself. rentable/verified are also CLI defaults, spelled
# out so the printed query is the whole truth. inet_up: the end phase's ~9.9 GB of checkpoint uploads must drain inside
# fit_budget's fixed end reserve (scripts/04_distill.py) before finish.py can destroy the box; a slow uplink overruns it
# and the watchdog stops the box at the deadline instead.
HOST_FILTER = [
    "num_gpus=1", "verified=true", "rentable=true", "reliability>=0.98", "cuda_vers>=13.0",
    "cpu_cores_effective>=12", "cpu_ram>=64", "disk_bw>=500", "inet_down>=500", "inet_up>=500",
    "direct_port_count>=1", f"disk_space>={DISK_GB}",
]
# vast names both SXM4 sizes "A100 SXM4"; gpu_ram tells them apart (40 GB cards report 39-40 GB)
TIERS = [
    ("A100 SXM4 40 GB", ["gpu_name in [A100_SXM4]", "gpu_ram<=48"]),
    ("A100 SXM4 80 GB (fallback)", ["gpu_name in [A100_SXM4]", "gpu_ram>=70"]),
]
SECRET_RE = re.compile(r"TOKEN|SECRET|PASSWORD|API_KEY", re.I)

# --job label (FINAL.md §11): one RTX 5090 (32 GB; the CLI token verified live), driver CUDA >= 13.0 (GeForce has no
# forward compatibility), >= 16 effective cores for the prep pools and 01, room for the 250 GB disk
LABEL_DISK_GB = 250
# the label disk, in GB: image, Cohere + Parakeet models, labels, sidecars/whisper/seeds, 01's unpruned-audio backlog
# cap (the hold file), in-flight downloads, 01's free-space floor and logs; LABEL_DISK_GB adds headroom
LABEL_DISK_PARTS = {"image": 12, "models": 7, "labels": 45, "sidecars": 3, "backlog": 80, "in_flight": 2.5,
                    "floor_01": 30, "logs": 1}
LABEL_DISK_MIN_GB = math.ceil(sum(LABEL_DISK_PARTS.values()))
LABEL_FILTER = [
    "num_gpus=1", "verified=true", "rentable=true", "cuda_vers>=13.0", "cpu_cores_effective>=16", "cpu_ram>=64",
    "inet_down>=500", "direct_port_count>=1", f"disk_space>={LABEL_DISK_GB}",
]
LABEL_TIERS = [
    ("RTX 5090 (strict)", ["gpu_name=RTX_5090", "gpu_ram>=30", "reliability>=0.97", "disk_bw>=300", "inet_up>=100"]),
    ("RTX 5090", ["gpu_name=RTX_5090", "gpu_ram>=30"]),
]
# ranking: a missing $/GB counts as this; label_end.json classes whose machine is avoided on the next launch
DEFAULT_GB_COST = 0.01
AVOID_CLASSES = ("host_failure", "slow_host")
LABEL_CONFIGS = "configs/full.json,configs/full_sub3k.json"
LABEL_WATCHDOG_ENV = {"KITSUNE_WATCHDOG_SYNC_LEAD_S": "1800", "KITSUNE_WATCHDOG_ORPHAN_S": "900"}
MODEL_ID = "CohereLabs/cohere-transcribe-03-2026"  # kitsune.features.HF_MODEL_ID (that module imports torch)
LEASE_MAX_AGE_S = 2700  # a lease heartbeat younger than this is a live box (label_sync.lease_live)
# HF storage (§16.4): what a full label run adds to the data repo at most, and the size launch warns above
LABEL_REPO_ADD_GB = 49
REPO_WARN_GB = 80


# --job study (STUDY.md 5.3-5.5, CONTRACT.md section 6): one size-study box, run by kitsune/study_queue.py. Boxes A and B
# rent 4 GPUs (one wave of four runs, each then its T/2 branch), the replicate and the shakedown one. Any A100, SXM4 or
# PCIe, 40 GB first: equal compute is measured on the box's own host (every box calibrates its reference run there), so
# the variant does not bias max_steps. The filter is relaxed per launch: reliability >= STUDY_RELIABILITY instead of the
# strict 0.98 (STUDY.md 5.3: at 0.95 the snapshot held 2 qualifying 1x offers and no 4x one; the 4x box it priced had
# 0.941); a box that dies costs a relaunch, and the queue's uploads keep everything it finished. Cores, RAM and the
# uplink scale with the GPUs (each trainer runs ~8 loader workers; every run uploads its logs and weights).
STUDY_RELIABILITY = 0.93
STUDY_BOXES = ("A", "B", "replicate", "shakedown")
STUDY_GPUS = {"A": 4, "B": 4, "replicate": 1, "shakedown": 1}
STUDY_TIERS = [
    ("A100 40 GB (SXM4 or PCIe)", ["gpu_name in [A100_SXM4,A100_PCIE]", "gpu_ram<=48"]),
    ("A100 80 GB (SXM4 or PCIe; fallback)", ["gpu_name in [A100_SXM4,A100_PCIE]", "gpu_ram>=70"]),
]
# box -> (central hours, watchdog cap before the rebuild timeout is added): boot and bootstrap with two stores ~0.55 h,
# the calibration ~0.2-0.4 h, the probes ~0.8-1.4 h, the wave (the longest run + its branch, STUDY.md 5.2: 3.8 central,
# 4.2 high), the anchor in a gap, the speed probes ~0.25 h (B), the end ~0.3 h; the replicate is one run + its branch;
# the shakedown ~1.5 h (STUDY.md 6.1)
STUDY_HOURS = {"A": (5.8, 8.0), "B": (6.3, 8.5), "replicate": (4.6, 6.0), "shakedown": (1.6, 3.0)}
STUDY_MAX_DPH = {4: 6.0, 1: 2.0}  # by GPU count
STUDY_UP_GB = {"A": 12, "B": 10, "replicate": 3, "shakedown": 3}  # lean uploads (STUDY.md 5.5)
STUDY_CONFIG = "study/data.json"


def study_filter(n_gpus: int, disk_gb: int = DISK_GB) -> list[str]:
    return [f"num_gpus={n_gpus}", "verified=true", "rentable=true", f"reliability>={STUDY_RELIABILITY}",
            "cuda_vers>=13.0", f"cpu_cores_effective>={12 * n_gpus}", f"cpu_ram>={64 * n_gpus}", "disk_bw>=500",
            "inet_down>=500", f"inet_up>={100 * n_gpus}", "direct_port_count>=1", f"disk_space>={disk_gb}"]


@dataclass(frozen=True)
class JobSpec:
    """What one kind of box rents: offer tiers and host filter, disk, the traffic and hours of its cost line, its
    caps, how offers are ranked (dph: $/h; est_total: GPU hours plus traffic) and the instance label's prefix."""
    tiers: list
    base_filter: list
    disk_gb: int
    est_down_gb: float
    est_up_gb: float
    est_hours: float | None
    max_hours: float
    max_dph: float
    sort: str
    label_prefix: str


JOBS = {
    "train": JobSpec(TIERS, HOST_FILTER, DISK_GB, EST_DOWN_GB, EST_UP_GB, None, 5.5, DEFAULT_MAX_DPH, "dph", "kitsune"),
    "label": JobSpec(LABEL_TIERS, LABEL_FILTER, LABEL_DISK_GB, 590, 45, 16, 30, 1.0, "est_total", "kitsune-label"),
}


def study_job(box: str) -> JobSpec:
    """The JobSpec of a size-study box: its GPU count in the relaxed filter, its hours, lean uploads."""
    n = STUDY_GPUS[box]
    est, cap = STUDY_HOURS[box]
    return JobSpec(STUDY_TIERS, study_filter(n), DISK_GB, EST_DOWN_GB, STUDY_UP_GB[box], est, cap, STUDY_MAX_DPH[n],
                   "dph", f"kitsune-study-{box}")


class LaunchError(RuntimeError):
    pass


def install_help() -> str:
    return (f"The vastai CLI is not on PATH. Install it and store your API key (you type the key yourself):\n"
            f"  pip install {VASTAI_PIN}\n"
            f"  vastai set api-key <your key from https://cloud.vast.ai/manage-keys/>\n"
            f"then re-run this script.")


def build_query(tier_terms: list[str]) -> str:
    return " ".join([*tier_terms, *HOST_FILTER])


def host_filter(terms: list[str], disk_gb: int) -> list[str]:
    """A host filter with its disk_space term set to `disk_gb` (the disk the box is created with)."""
    return [t for t in terms if not t.startswith("disk_space")] + [f"disk_space>={disk_gb}"]


def job_query(job: JobSpec, tier_terms: list[str], disk_gb: int) -> str:
    return " ".join([*tier_terms, *host_filter(job.base_filter, disk_gb)])


def search_args(query: str, disk_gb: int = DISK_GB) -> list[str]:
    return ["search", "offers", query, "--type", "on-demand", "-o", "dph", "--storage", str(disk_gb), "--raw"]


def env_string(env: dict[str, str]) -> str:
    """vast's --env takes docker-style options; values must be single shell words."""
    for k, v in env.items():
        if SECRET_RE.search(k):
            raise LaunchError(f"refusing to put {k} on the command line; secrets belong in the vast account env")
        if not re.fullmatch(r"[A-Za-z0-9_./:@+,=-]+", v):
            raise LaunchError(f"env value for {k} must be one word without quotes/spaces: {v!r}")
    return " ".join(f"-e {k}={v}" for k, v in env.items())


def create_args(offer_id: int | str, image: str, env: dict[str, str], onstart: Path, label: str,
                disk_gb: int = DISK_GB) -> list[str]:
    return ["create", "instance", str(offer_id), "--image", image, "--disk", str(disk_gb), "--ssh", "--direct",
            "--env", env_string(env), "--onstart", str(onstart), "--label", label, "--cancel-unavail", "--raw"]


def parse_json(text: str):
    """The CLI may print warnings before the JSON; take the first JSON value."""
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        raise LaunchError(f"no JSON in vastai output: {text[:300]!r}")
    return json.JSONDecoder().raw_decode(text[min(starts):])[0]


def vastai(exe: str, args: list[str]) -> str:
    """vastai 1.8.0 exits 0 on an API error (a 401 bad key on search, a 410 no_such_ask when the offer was taken before
    the create): stdout stays empty and --raw puts {"error": true, ...} on stderr. Both calls here print JSON on
    success (a search that finds nothing prints []), so an empty stdout is a failure too, reported with vast's reason.
    A create that times out may still have rented an instance, so the error says where to look before a re-run."""
    what = " ".join(args[:2])
    try:
        r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        hint = ""
        if args[:2] == ["create", "instance"]:
            label = args[args.index("--label") + 1] if "--label" in args else "kitsune-..."
            hint = (f"; the instance may still have been created: check `vastai show instances` for label {label} "
                    f"before re-running (or you may rent twice)")
        raise LaunchError(f"vastai {what} timed out after 180 s{hint}") from None
    if r.returncode != 0 or not r.stdout.strip() or '"error": true' in r.stderr:
        raise LaunchError(f"vastai {what} failed ({r.returncode}): "
                          f"{(r.stderr or r.stdout).strip()[:500] or 'no output'}")
    return r.stdout


def _gb_cost(offer: dict, key: str) -> float:
    v = offer.get(key)
    return v if isinstance(v, (int, float)) else DEFAULT_GB_COST


def est_total(offer: dict, job: JobSpec) -> float:
    """The run's expected cost at this offer: $/h (GPU + disk) x the job's hours, plus its traffic at the host's $/GB
    (a missing $/GB counts as DEFAULT_GB_COST)."""
    dph = offer.get("dph_total")
    dph = dph if isinstance(dph, (int, float)) else 1e9
    return (dph * (job.est_hours or 0) + _gb_cost(offer, "inet_down_cost") * job.est_down_gb
            + _gb_cost(offer, "inet_up_cost") * job.est_up_gb)


def rank_offers(offers: list[dict], job: JobSpec, avoid=frozenset(), max_dph: float | None = None) -> list[dict]:
    """Offers without the avoided machines, cheapest first by $/h (train) or by est_total (label). The label ranking
    drops offers over max_dph first: its winner is the cheapest total, which need not be the cheapest per hour."""
    avoid = {str(m) for m in avoid}
    kept = [o for o in offers if str(o.get("machine_id")) not in avoid]
    if job.sort == "est_total":
        if max_dph is not None:
            kept = [o for o in kept if isinstance(o.get("dph_total"), (int, float)) and o["dph_total"] <= max_dph]
        return sorted(kept, key=lambda o: est_total(o, job))
    return sorted(kept, key=lambda o: o.get("dph_total", 1e9))


def search_offers(exe: str, job: JobSpec | None = None, disk_gb: int | None = None,
                  avoid=frozenset(), max_dph: float | None = None) -> tuple[str, str, list[dict]]:
    """-> (tier name, query, ranked offers) for the first tier with any offer that is not on an avoided machine."""
    job = job or JOBS["train"]
    disk_gb = disk_gb or job.disk_gb
    for name, terms in job.tiers:
        query = job_query(job, terms, disk_gb)
        offers = parse_json(vastai(exe, search_args(query, disk_gb)))
        if isinstance(offers, dict):
            offers = offers.get("offers", [])
        ranked = rank_offers(offers, job, avoid, max_dph)
        if ranked:
            return name, query, ranked
        print(f"no offers for {name}" + (f" (all {len(offers)} on avoided machines)" if offers else "") + f": {query}")
    if job.sort == "est_total":  # --ssh --direct needs a direct port: show whether that is what empties the search
        hint = " ".join(t for t in job_query(job, job.tiers[-1][1], disk_gb).split(" ")
                        if not t.startswith("direct_port_count"))
        print(f"hint: the same search without direct_port_count: vastai {shlex.join(search_args(hint, disk_gb))}")
    return "", "", []


def offer_table(offers: list[dict], limit: int = 10, job: JobSpec | None = None) -> str:
    cols = [("id", "id", "{}"), ("gpu", "gpu_name", "{}"), ("GB", "gpu_ram", "{:.0f}"), ("$/h", "dph_total", "{:.3f}"),
            ("rel", "reliability", "{:.3f}"), ("cuda", "cuda_max_good", "{}"), ("cpu", "cpu_cores_effective", "{:.0f}"),
            ("ramGB", "cpu_ram", "{:.0f}"), ("down", "inet_down", "{:.0f}"), ("up", "inet_up", "{:.0f}"),
            ("diskMB/s", "disk_bw", "{:.0f}"), ("$/GBdn", "inet_down_cost", "{:.3f}"),
            ("$/GBup", "inet_up_cost", "{:.3f}"), ("where", "geolocation", "{}")]
    if job is not None and job.sort == "est_total":  # label: the machine, disk $/GB-month and the ranking's total
        cols[-1:-1] = [("machine", "machine_id", "{}"), ("$/GBmo", "storage_cost", "{:.3f}"),
                       ("est$", "_est", "{:.2f}")]
    rows = [[h for h, _, _ in cols]]
    for o in offers[:limit]:
        row = []
        for _, key, fmt in cols:
            v = est_total(o, job) if key == "_est" else o.get(key)
            if key in ("gpu_ram", "cpu_ram") and isinstance(v, (int, float)) and v > 1000:
                v = v / 1024  # the API reports MB
            try:
                row.append(fmt.format(v) if v is not None else "-")
            except (ValueError, TypeError):
                row.append(str(v))
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    return "\n".join("  ".join(c.ljust(w) for c, w in zip(r, widths)) for r in rows)


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True).stdout.strip()


def config_at(sha: str, config: str) -> dict:
    """The run config the box runs: the file committed at `sha` (KITSUNE_SHA), not the working tree's, which differs
    for a --sha other than HEAD. Read as UTF-8 bytes (git()'s text mode decodes with the locale's code page)."""
    out = subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:{config}"], capture_output=True, check=True).stdout
    return json.loads(out.decode("utf-8"))


def git_checks(sha: str, config: str) -> list[str]:
    """Problems that would make the box fail after it is already billing."""
    problems = []
    checks = [
        (("status", "--porcelain", "--untracked-files=no"), lambda out: out,
         "the working tree has uncommitted changes to tracked files (the box runs the commit, not them)"),
        (("branch", "-r", "--contains", sha), lambda out: not out,
         f"{sha[:12]} is not on any remote-tracking branch: push it first (the box clones it from GitHub)"),
        (("cat-file", "-e", f"{sha}:{config}"), lambda out: False, f"{config} does not exist at {sha[:12]}"),
    ]
    for cmd, bad, msg in checks:
        try:
            if bad(git(*cmd)):
                problems.append(msg)
        except (subprocess.CalledProcessError, OSError):
            problems.append(msg)
    return problems


def _ghcr_token(name: str, timeout: float) -> str:
    """Anonymous pull token for ghcr.io/<name> (the package must be public, D39a)."""
    with urllib.request.urlopen(f"https://ghcr.io/token?scope=repository:{name}:pull&service=ghcr.io", timeout=timeout) as r:
        return json.load(r)["token"]


def resolve_image_digest(tag: str, timeout: float = 20) -> str:
    """Tag -> immutable `repo@sha256:...` via the anonymous GHCR registry API (the package must be public, D39a)."""
    name = IMAGE_REPO.split("/", 1)[1]
    req = urllib.request.Request(f"https://ghcr.io/v2/{name}/manifests/{tag}", method="HEAD",
                                 headers={"Authorization": f"Bearer {_ghcr_token(name, timeout)}",
                                          "Accept": MANIFEST_ACCEPT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        digest = r.headers.get("Docker-Content-Digest")
    if not digest or not digest.startswith("sha256:"):
        raise LaunchError(f"GHCR returned no digest for {IMAGE_REPO}:{tag}")
    return f"{IMAGE_REPO}@{digest}"


def image_revision(image: str, timeout: float = 20) -> str:
    """The commit a pinned `ghcr.io/<name>@sha256:...` image was built from: the org.opencontainers.image.revision
    label docker/metadata-action writes in image.yml, read anonymously from the image config (index -> linux/amd64
    manifest -> config blob)."""
    name, digest = image.split("/", 1)[1].split("@", 1)
    auth = {"Authorization": f"Bearer {_ghcr_token(name, timeout)}"}

    def get(path: str, accept: str):
        req = urllib.request.Request(f"https://ghcr.io/v2/{name}/{path}", headers={**auth, "Accept": accept})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    m = get(f"manifests/{digest}", MANIFEST_ACCEPT)
    if "manifests" in m:  # an index: the build's image plus attestation entries (platform unknown/unknown)
        amd64 = [x["digest"] for x in m["manifests"] if x.get("platform", {}).get("architecture") == "amd64"]
        if not amd64:
            raise LaunchError(f"{image} has no linux/amd64 image")
        m = get(f"manifests/{amd64[0]}", MANIFEST_ACCEPT)
    labels = get(f"blobs/{m['config']['digest']}", "*/*").get("config", {}).get("Labels") or {}
    rev = labels.get("org.opencontainers.image.revision")
    if not rev:
        raise LaunchError(f"{image} carries no org.opencontainers.image.revision label")
    return rev


def image_problems(image: str, sha: str) -> list[str]:
    """The image must hold the dependencies of the commit the box runs. CI moves a branch tag only after the build and
    its smoke test pass, so while a requirements-train.txt / Dockerfile change is still building, or after its smoke
    failed, the tag still names the previous image, and nothing on the box compares the pins: the new code would run
    on the old libraries (a crash after the paid bootstrap, or results credited to pins they did not use)."""
    try:
        rev = image_revision(image)
    except (OSError, LaunchError, KeyError, ValueError) as e:
        return [f"cannot read the commit {image} was built from ({type(e).__name__}: {e}), so its dependencies cannot "
                f"be checked against {sha[:12]}"]
    try:
        git("cat-file", "-e", f"{rev}^{{commit}}")
    except (subprocess.CalledProcessError, OSError):
        return [f"{image} was built from {rev[:12]}, which this clone lacks: git fetch, then re-run"]
    try:
        changed = git("diff", "--name-only", rev, sha, "--", *IMAGE_INPUTS).split()
    except (subprocess.CalledProcessError, OSError):
        return [f"cannot compare {IMAGE_INPUTS} between the image's build commit {rev[:12]} and {sha[:12]}"]
    if changed:
        return [f"{image} was built from {rev[:12]} but {sha[:12]} changes {changed}: wait for the image build (or "
                f"check whether its smoke test failed in the Actions tab), then re-run"]
    return []


def data_problems(files: list[str], cfg: dict) -> list[str]:
    """What the run config needs from the data repo's file list (bootstrap.sh's plan() applies the same rules on the
    box, where a failure is already billed): the teacher/second-opinion meta, the selection, the student's weights,
    config, processor and tokenizer (STUDENT_FILES), teacher shards for every train source and eval set, and a second
    opinion for EVERY train source teacher shard. make_selection drops the rows of a shard without one as no_agree, so
    a partial 02b pass silently shrinks the train set. The run config's selection_recipe.partial_second_opinion lists
    the sources that train on their judged shards only, by decision: for them one judged shard is enough (the
    selection check makes sure the selection was built that way, with not_judged rows instead of no_agree).
    A CTC student (family "ctc") or pull_parakeet true also needs parakeet_root's meta.json and Parakeet npz for every
    train source and eval set (kitsune.extent.pull_plan's rule for extent configs), and a CTC run without
    pull_parakeet needs teacher npz only for its eval sets."""
    teacher_root, second_root = cfg.get("teacher_root", "teacher_out"), cfg.get("second_root", "second_out")
    have = set(files)
    student = cfg["student"].rstrip("/")
    ctc = cfg.get("family", "aed") == "ctc"
    problems = [f"no {f}" for f in (f"{teacher_root}/meta.json", f"{second_root}/meta.json", cfg["selection"],
                                    *(f"{student}/{n}" for n in STUDENT_FILES + ((CTC_CARD,) if ctc else ())))
                if f not in have]
    parakeet_root = cfg.get("parakeet_root") if ctc or cfg.get("pull_parakeet") else None
    if (ctc or cfg.get("pull_parakeet")) and not parakeet_root:
        problems.append("the config needs Parakeet targets (family ctc or pull_parakeet) but has no parakeet_root")
    elif parakeet_root and f"{parakeet_root}/meta.json" not in have:
        problems.append(f"no {parakeet_root}/meta.json")

    def stems(root: str, s: str, ext: str) -> set[str]:
        return {f.rsplit("/", 1)[1][: -len(ext)] for f in files if f.startswith(f"{root}/{s}/") and f.endswith(ext)}

    sources = list(cfg.get("sources", []))
    eval_sets = set(cfg.get("eval_sets", []))
    partial = set((cfg.get("selection_recipe") or {}).get("partial_second_opinion", []))
    for s in dict.fromkeys(sources + list(cfg.get("eval_sets", []))):
        if parakeet_root and not stems(parakeet_root, s, ".npz"):
            problems.append(f"no {parakeet_root}/{s}/*.npz")
        teacher = stems(teacher_root, s, ".npz")
        if not teacher and (s in eval_sets or not ctc or cfg.get("pull_parakeet")):
            problems.append(f"no {teacher_root}/{s}/*.npz")
        elif s in partial and s in sources and not teacher & stems(second_root, s, ".jsonl"):
            problems.append(f"{second_root}/{s}: none of its {len(teacher)} teacher shards has a second opinion")
        elif s in sources and s not in partial and (gap := teacher - stems(second_root, s, ".jsonl")):
            problems.append(f"{second_root}/{s}: {len(gap)} of {len(teacher)} teacher shards have no second opinion "
                            f"(e.g. {min(gap)}): finish scripts/02b_second_opinion.py, rebuild the selection, upload")
    return problems


def _by_source(items) -> dict[str, float]:
    """make_selection.py's --agree-max-source values (SOURCE=A) as {source: threshold}."""
    out = {}
    for item in items or []:
        src, _, val = str(item).partition("=")
        out[src] = float(val)
    return out


def selection_problems(path: Path, name: str, cfg: dict, have: set[str] | None = None) -> list[str]:
    """The selection must be the one the run config describes. make_selection.py records its arguments in the parquet
    (metadata b"kitsune_selection"): they must match the config's sources, eval_sets and selection_recipe (agreement
    thresholds, label-filtered hold-outs) - a rebuild with a flag forgotten would otherwise silently train on other
    rows or lose a hold-out. Every configured train source and eval set must keep rows (the trainer takes an empty one
    in silence). And a selection built before the second-opinion pass finished drops those rows as no_agree.
    With `have` (the data repo's file list) every kept row's teacher_file must be there as .npz and .jsonl: the
    trainer's build_stores raises on a missing one, after the paid bootstrap, and data_problems only asks for SOME
    teacher shard per source (a train npz also satisfies the galgame hold-out).
    Drop reasons must be ones the recipe produces: the study's (kitsune.prereg.STUDY_REASONS) only with a
    selection_recipe.study block, which must equal the one the selection was built with and the pre-registered one
    (kitsune.prereg.STUDY_SELECTION), with the pre-registered seed (SELECTION_SEED). A study selection also
    keeps every eval row in both label roots (an eval row not_in_parakeet means the Parakeet pass did not label the
    set whole, K6), leaves at most kitsune.prereg.ONE_ROOT_MAX_FRAC of its train rows in teacher_out only (K5), and,
    with `have`, has its sidecar (<selection>.json) and manifest (study_manifest.json) uploaded next to it. A CTC
    student (family "ctc") or pull_parakeet also needs parakeet_out's npz and jsonl of every kept row; a CTC run
    without pull_parakeet needs the teacher files of its kept eval rows only."""
    import pandas as pd
    import pyarrow.parquet as pq

    from kitsune import prereg

    rebuild = "rebuild it with scripts/make_selection.py --config <the run config> and upload it"
    problems = []
    raw = (pq.ParquetFile(path).schema_arrow.metadata or {}).get(b"kitsune_selection")
    args = json.loads(raw).get("args") if raw else None
    recipe = cfg.get("selection_recipe")
    if not isinstance(recipe, dict):
        problems.append("the run config has no selection_recipe to check the selection against")
    elif not isinstance(args, dict):
        problems.append(f"{name} has no record of how it was built (kitsune_selection metadata): {rebuild}")
    else:
        got = dict(sources=set(args.get("sources") or []), eval_sets=set(args.get("eval_sets") or []),
                   agree_max=args.get("agree_max"), agree_max_source=_by_source(args.get("agree_max_source")),
                   filter_eval_sets=set(args.get("filter_eval_sets") or []),
                   partial_second_opinion=set(args.get("partial_second_opinion") or []),
                   extent=args.get("extent") or None, study=args.get("study") or None)
        ext = cfg.get("extent") or None  # make_selection records {name, inputs} of the config's extent (None: none)
        want = dict(sources=set(cfg.get("sources", [])), eval_sets=set(cfg.get("eval_sets", [])),
                    agree_max=float(recipe["agree_max"]), agree_max_source=_by_source(recipe["agree_max_source"]),
                    filter_eval_sets=set(recipe["filter_eval_sets"]),
                    partial_second_opinion=set(recipe.get("partial_second_opinion", [])),
                    extent=dict(name=ext.get("name"), inputs=ext.get("inputs") or {}) if ext else None,
                    study=recipe.get("study") or None)
        for k, v in want.items():
            if got[k] != v:
                shown = [sorted(x.items()) if isinstance(x, dict) else sorted(x) if isinstance(x, set) else x
                         for x in (got[k], v)]
                problems.append(f"{name} was built with {k} {shown[0]}, the run config says {shown[1]}: {rebuild}")
        if want["study"] is not None:  # the study's data is pre-registered: one recipe, one seed for every run
            if want["study"] != prereg.STUDY_SELECTION:
                problems.append(f"the run config's selection_recipe.study {want['study']} is not the pre-registered "
                                f"{prereg.STUDY_SELECTION} (kitsune.prereg; study/data.json carries it)")
            if args.get("seed") != prereg.SELECTION_SEED:
                problems.append(f"{name} was built with seed {args.get('seed')!r}, the study selection's is "
                                f"pre-registered as {prereg.SELECTION_SEED}: {rebuild}")

    sel = pd.read_parquet(path, columns=["source", "split", "keep", "reason", "teacher_file"])
    kept = sel[sel["keep"]].groupby(["source", "split"]).size()
    for split, key in (("train", "sources"), ("eval", "eval_sets")):
        empty = [s for s in cfg.get(key, []) if not kept.get((s, split), 0)]
        if empty:
            problems.append(f"{name} keeps no {split} rows of {empty} (in the config's {key}): {rebuild}")
    bad = sel[sel["reason"] == "no_agree"]
    if not bad.empty:
        problems.append(f"{name} drops {len(bad)} rows as no_agree ({bad['source'].value_counts().to_dict()}): rebuild "
                        f"it with scripts/make_selection.py after the second-opinion pass and upload it")
    study = isinstance(recipe, dict) and recipe.get("study") is not None
    allowed = {"kept", "truncated", "not_judged", "no_agree", "no_audio"}
    allowed |= set(prereg.STUDY_REASONS) if study else set()
    if odd := sorted(r for r in sel["reason"].unique() if r not in allowed and not str(r).startswith("agree>")):
        problems.append(f"{name} drops rows as {odd}, which {'the study' if study else 'this'} recipe never does: "
                        f"{rebuild}")
    if study:
        ev = sel[(sel["split"] == "eval") & (sel["reason"] == "not_in_parakeet")]
        if not ev.empty:
            problems.append(f"{name}: {len(ev)} eval rows have no Parakeet labels "
                            f"({ev['source'].value_counts().to_dict()}): the eval sets must be labelled whole by both "
                            f"passes (K6)")
        train = sel[sel["split"] == "train"]
        n_one = int((train["reason"] == "not_in_parakeet").sum())
        if n_one > prereg.ONE_ROOT_MAX_FRAC * len(train):
            problems.append(f"{name}: {n_one} of {len(train)} train rows are in teacher_out only, more than "
                            f"{100 * prereg.ONE_ROOT_MAX_FRAC:g} % (K5): the Parakeet pass is incomplete")
    if have is not None:
        teacher_root = cfg.get("teacher_root", "teacher_out")
        ctc = cfg.get("family", "aed") == "ctc"
        rows = sel[sel["keep"] & ((sel["split"] == "eval") | (not ctc) | bool(cfg.get("pull_parakeet")))]
        need = sorted({f"{teacher_root}/{tf}{ext}" for tf in rows["teacher_file"].unique()
                       for ext in (".npz", ".jsonl")})
        absent = [f for f in need if f not in have]
        if absent:
            problems.append(f"{name} keeps rows whose teacher output is not in the data repo: {len(absent)} of "
                            f"{len(need)} files missing (e.g. {absent[0]}): upload the teacher_out the selection was "
                            f"built from")
        if (ctc or cfg.get("pull_parakeet")) and cfg.get("parakeet_root"):
            need = sorted({f"{cfg['parakeet_root']}/{tf}{ext}" for tf in sel.loc[sel["keep"], "teacher_file"].unique()
                           for ext in (".npz", ".jsonl")})
            if absent := [f for f in need if f not in have]:
                problems.append(f"{name} keeps rows whose Parakeet targets are not in the data repo: {len(absent)} of "
                                f"{len(need)} files missing (e.g. {absent[0]})")
        if study:
            for f in prereg.study_files(name):
                if f not in have:
                    problems.append(f"no {f}: upload the study selection's sidecar and manifest with it")
    return problems


def extent_problems(files: list[str], cfg: dict, record: dict) -> list[str]:
    """What a config with an `extent` needs from the data repo listing `files`, given the extent record: a valid
    extent that the record serves, the root sealed (<root>/COMPLETE.json: the label box verified and finished it), the
    teacher npz and jsonl of every subset stem, second_out jsonl for the subset stems of the train sources and
    filter_eval_sets, both meta files, the selection and the student's STUDENT_FILES. Pure: the label box's consumer
    check runs it against the Hub listing, the A100 launch (extent_preflight) before renting."""
    from kitsune import extent

    problems = extent.validate(cfg)
    if problems:
        return problems
    have = set(files)
    root = extent.extent_block(cfg)["root"].rstrip("/")
    if f"{root}/COMPLETE.json" not in have:
        problems.append(f"no {root}/COMPLETE.json: the label run of this root has not finished (relaunch it with "
                        f"vast/launch.py --job label)")
    problems += extent.pull_plan(cfg, record, files)["problems"]
    if cfg.get("student"):
        student = cfg["student"].rstrip("/")
        problems += [f"no {student}/{n}" for n in STUDENT_FILES if f"{student}/{n}" not in have]
    return problems


def _hub():
    """(HfApi(), hf_hub_download); one seam for the tests."""
    from huggingface_hub import HfApi, hf_hub_download

    return HfApi(), hf_hub_download


def hf_preflight(data_repo: str, out_repo: str, cfg: dict) -> tuple[str | None, list[str]]:
    """-> (data repo commit to pin, problems). Uses the laptop's own HF login, read-only (the selection, ~2 MB, is
    downloaded to a temporary dir). Both repos must be private: they hold dataset reference transcripts (`ref` in
    teacher_out/*.jsonl, the eval tables and the samples) whose terms forbid republishing them, and the trainer's
    hf.private only applies when it creates a repo, never to the existing ones launch requires."""
    import tempfile

    from huggingface_hub import HfApi, hf_hub_download

    api, problems, rev = HfApi(), [], None
    try:
        info = api.dataset_info(data_repo)
        rev = info.sha
        if info.private is not True:
            problems.append(f"{data_repo} is not private: teacher_out/second_out hold dataset transcripts (JSUT, CV, "
                            f"ReazonSpeech, Galgame) that must not be republished; hf repos settings {data_repo} "
                            f"--repo-type dataset --private")
        files = api.list_repo_files(data_repo, repo_type="dataset", revision=rev)
        if not cfg.get("extent"):  # an extent config: extent_preflight checks the files (extent_problems)
            problems += [f"{data_repo}@{rev[:12]}: {p}" for p in data_problems(files, cfg)]
        if cfg["selection"] in files:
            with tempfile.TemporaryDirectory(prefix="kitsune-launch-") as tmp:
                local = hf_hub_download(data_repo, cfg["selection"], repo_type="dataset", revision=rev, local_dir=tmp)
                problems += selection_problems(Path(local), cfg["selection"], cfg, set(files))
    except Exception as e:
        problems.append(f"cannot read dataset {data_repo}: {type(e).__name__}: {e}")
    try:
        private = api.model_info(out_repo).private
    except Exception as e:
        # the Hub answers 401/404 (RepositoryNotFoundError) both for a missing repo and for a private one this login
        # cannot see; a 5xx, 429 or dropped connection is the Hub's (model_info is not retried), so re-run
        status = getattr(getattr(e, "response", None), "status_code", None)
        if status in (401, 403, 404):
            problems.append(f"cannot read output model repo {out_repo} ({type(e).__name__}: {e}): create it first "
                            f"(hf repos create {out_repo} --private) or check the laptop's login (hf auth whoami)")
        else:
            problems.append(f"Hub error reading output model repo {out_repo} ({type(e).__name__}: {e}); re-run launch")
    else:
        if private is not True:
            problems.append(f"{out_repo} is not private: the run uploads eval and sample tables with reference "
                            f"transcripts; hf repos settings {out_repo} --private")
    return rev, problems


def _selection_kept(path: Path) -> dict | None:
    """make_selection's `kept` ({"<source>/<split>": {"utts", "hours"}}) from the selection's metadata."""
    import pyarrow.parquet as pq

    raw = (pq.ParquetFile(path).schema_arrow.metadata or {}).get(b"kitsune_selection")
    return json.loads(raw).get("kept") if raw else None


def extent_preflight(data_repo: str, data_rev: str | None, cfg: dict,
                     extra_gb: float = 0.0) -> tuple[list[str], dict | None]:
    """-> (problems, sizing) for a train config with an `extent`, read-only with the laptop's login at the pinned
    revision: <root>/extent.json must be there, extent_problems() empty, and kitsune.extent.sizing() (with the kept
    hours of the selection's metadata; extra_gb: a study box's checkpoints) gives the disk, the download and the
    rebuild timeout of the A100's bootstrap."""
    import tempfile

    from kitsune import extent

    problems, sizing = extent.validate(cfg), None
    if problems:
        return problems, None
    root = extent.extent_block(cfg)["root"].rstrip("/")
    record_path = f"{root}/{extent.RECORD_FILE}"
    try:
        api, download = _hub()
        files = api.list_repo_files(data_repo, repo_type="dataset", revision=data_rev)
        if record_path not in files:
            return [f"{data_repo}: no {record_path}: the label box writes it when it finishes the root"], None
        with tempfile.TemporaryDirectory(prefix="kitsune-launch-") as tmp:
            record = extent.load_record(Path(download(data_repo, record_path, repo_type="dataset", revision=data_rev,
                                                      local_dir=tmp)))
            kept = None
            if cfg["selection"] in files:
                kept = _selection_kept(Path(download(data_repo, cfg["selection"], repo_type="dataset",
                                                     revision=data_rev, local_dir=tmp)))
        problems += [f"{data_repo}@{(data_rev or 'head')[:12]}: {p}" for p in extent_problems(files, cfg, record)]
        if not extent.record_problems(record, cfg):
            sizing = extent.sizing(record, cfg, kept, extra_gb=extra_gb)
    except Exception as e:
        problems.append(f"cannot check the extent {record_path} of {data_repo}: {type(e).__name__}: {e}")
    return problems, sizing


def _heartbeat_s(v) -> float | None:
    """A lease heartbeat as epoch seconds: a number, or an ISO-8601 time (UTC when it has no zone)."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            t = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
        return t.timestamp() if t.tzinfo else t.replace(tzinfo=timezone.utc).timestamp()
    return None


def lease_live(lease: dict | None, now: float, max_age_s: int = LEASE_MAX_AGE_S) -> bool:
    """A lease another box still holds: not released and a heartbeat younger than max_age_s. From the laptop every
    holder is another container (the box to rent does not exist yet). An unreadable heartbeat counts as live."""
    if not lease or lease.get("released"):
        return False
    hb = _heartbeat_s(lease.get("heartbeat"))
    return hb is None or now - hb < max_age_s


def _lfs_sha256(info) -> str | None:
    lfs = getattr(info, "lfs", None)
    if lfs is None:
        return None
    return getattr(lfs, "sha256", None) or (lfs.get("sha256") if isinstance(lfs, dict) else None)


def label_preflight(data_repo: str, sha: str, cfg: dict, label_cfgs: dict[str, dict], *, steal_lease: bool = False,
                    now: float | None = None) -> tuple[str | None, list[str], list[str]]:
    """-> (data repo commit to pin, problems, notes) for --job label, read-only with the laptop's login; it replaces
    hf_preflight (never data_problems/selection_problems: the label box makes that data). Checks: the data repo is
    private; the label root is not sealed (COMPLETE.json) and no other box holds a live lease (unless steal_lease);
    the seed teacher meta and the gate sets' teacher files; the Parakeet model files with their pinned sha256; read
    access to the gated Cohere model (the box re-checks its own token in minute 1); every label config's extent valid
    and inside `cfg`'s with the same root. Notes: the resume plan (what the root already holds) and the repo size with
    the run's projected additions (a warning above REPO_WARN_GB). `sha` names the commit whose configs these are."""
    import tempfile

    from kitsune import extent

    now = time.time() if now is None else now
    problems, notes, rev = [], [], None
    problems += [f"the label config: {p}" for p in extent.validate(cfg)]
    ext = extent.extent_block(cfg) or {}
    root = str(ext.get("root") or "labels/full").rstrip("/")
    for name, other in label_cfgs.items():
        if other is cfg:
            continue
        problems += [f"{name}: {p}" for p in extent.validate(other)]
        problems += [f"{name}: {p}" for p in extent.within(cfg, other)]
    try:
        api, download = _hub()
        info = api.dataset_info(data_repo, files_metadata=True)
        rev = info.sha
        if info.private is not True:
            problems.append(f"{data_repo} is not private: the labels carry dataset reference transcripts; hf repos "
                            f"settings {data_repo} --repo-type dataset --private")
        files = api.list_repo_files(data_repo, repo_type="dataset", revision=rev)
        have = set(files)
        if f"{root}/COMPLETE.json" in have:
            problems.append(f"{data_repo}: {root}/COMPLETE.json exists: this root is sealed; a new label run needs a "
                            f"new extent root")
        if f"{root}/LEASE.json" in have:
            with tempfile.TemporaryDirectory(prefix="kitsune-launch-") as tmp:
                lease = json.loads(Path(download(data_repo, f"{root}/LEASE.json", repo_type="dataset", revision=rev,
                                                 local_dir=tmp)).read_text(encoding="utf-8"))
            if lease_live(lease, now):
                msg = (f"{root}/LEASE.json is held by a live box (container {lease.get('container_id')}, machine "
                       f"{lease.get('machine_id')}, heartbeat {lease.get('heartbeat')})")
                if steal_lease:
                    notes.append(f"{msg}: taken over (--steal-lease); make sure that box is gone")
                else:
                    problems.append(f"{msg}: two boxes must not write one root; destroy it or pass --steal-lease")
        counts: dict[str, int] = {}
        for f in files:
            parts = f.split("/")
            if f.startswith(f"{root}/") and len(parts) == root.count("/") + 4 and f.endswith(".npz"):
                key = "/".join(parts[-3:-1])
                counts[key] = counts.get(key, 0) + 1
        notes.append("resume plan: " + (", ".join(f"{k} {n} stems" for k, n in sorted(counts.items()))
                                        + f" already under {root} (the box pulls them, labels the rest)"
                                        if counts else f"{root} holds no labels yet: a fresh run"))
        seeds = ["teacher_out/meta.json"] + [f"teacher_out/{g}/" for g in extent.GATE_SETS]
        for s in seeds:
            if s.endswith("/"):
                if not any(f.startswith(s) and f.endswith(".npz") for f in files) or not any(
                        f.startswith(s) and f.endswith(".jsonl") for f in files):
                    problems.append(f"{data_repo}: no {s}*.npz/.jsonl seed (the gate sets are adopt-only)")
            elif s not in have:
                problems.append(f"{data_repo}: no {s} seed")
        try:
            from kitsune import parakeet
            pins, ppath = dict(parakeet.PARAKEET_FILES), parakeet.PARAKEET_PATH.rstrip("/")
        except (ImportError, AttributeError) as e:
            problems.append(f"no Parakeet pins in kitsune/parakeet.py ({type(e).__name__}: {e})")
            pins, ppath = {}, ""
        if pins:
            paths = [f"{ppath}/{n}" for n in pins]
            if missing := [x for x in paths if x not in have]:
                problems.append(f"{data_repo}: {len(missing)} Parakeet model files missing (e.g. {missing[0]}): run "
                                f"tools/publish_parakeet.py --upload on the laptop")
            present = [x for x in paths if x in have]
            for pi in (api.get_paths_info(data_repo, present, repo_type="dataset", revision=rev) if present else []):
                got = _lfs_sha256(pi)
                want = pins.get(pi.path.rsplit("/", 1)[1])
                if got and want and got != want:
                    problems.append(f"{data_repo}: {pi.path} has sha256 {got[:12]}..., kitsune/parakeet.py pins "
                                    f"{want[:12]}...")
        try:
            from huggingface_hub import auth_check
            auth_check(MODEL_ID, repo_type="model")
            notes.append(f"{MODEL_ID}: readable with the laptop login (the box re-checks its own HF_TOKEN in minute 1)")
        except Exception as e:
            problems.append(f"cannot read the gated {MODEL_ID} with the laptop login ({type(e).__name__}): accept its "
                            f"terms; the box's HF_TOKEN needs the same access")
        size = sum(getattr(x, "size", None) or 0 for x in (getattr(info, "siblings", None) or [])) / 1e9
        labelled = sum(getattr(x, "size", None) or 0 for x in (getattr(info, "siblings", None) or [])
                       if getattr(x, "rfilename", "").startswith(f"{root}/")) / 1e9
        projected = size + max(0.0, LABEL_REPO_ADD_GB - labelled)
        warn = projected > REPO_WARN_GB
        notes.append(f"{'WARNING: ' if warn else ''}{data_repo} holds {size:.1f} GB; after the label run up to "
                     f"~{projected:.0f} GB" + (f" (> {REPO_WARN_GB} GB: check the account's private storage quota)"
                                               if warn else ""))
    except Exception as e:
        problems.append(f"cannot read dataset {data_repo}: {type(e).__name__}: {e}")
    return rev, problems, notes


def avoided_machines(data_repo: str, rev: str | None) -> tuple[set[str], list[str]]:
    """-> (machine ids, notes): the machines of earlier label runs that ended as host_failure or slow_host
    (label_runs/*/label_end.json), best effort."""
    import tempfile

    out, notes = set(), []
    try:
        api, download = _hub()
        ends = [f for f in api.list_repo_files(data_repo, repo_type="dataset", revision=rev)
                if re.fullmatch(r"label_runs/[^/]+/label_end\.json", f)]
        with tempfile.TemporaryDirectory(prefix="kitsune-launch-") as tmp:
            for f in ends:
                end = json.loads(Path(download(data_repo, f, repo_type="dataset", revision=rev,
                                               local_dir=tmp)).read_text(encoding="utf-8"))
                if end.get("class") in AVOID_CLASSES and end.get("machine_id") not in (None, ""):
                    out.add(str(end["machine_id"]))
                    notes.append(f"avoiding machine {end['machine_id']}: {f} ended {end['class']}")
    except Exception as e:
        notes.append(f"could not read label_runs/*/label_end.json ({type(e).__name__}: {e}); only --avoid-machine "
                     f"applies")
    return out, notes


def git_show(sha: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(ROOT), "show", f"{sha}:{path}"], capture_output=True, check=True).stdout


def study_preflight(data_repo: str, data_rev: str | None, out_repo: str, sha: str, box: str,
                    cfg: dict) -> tuple[list[str], list[str]]:
    """-> (problems, notes) for --job study, read-only (local git and the laptop's HF login), on top of the extent and
    selection checks every extent config gets (hf_preflight, extent_preflight):
    - the box's plan (kitsune.prereg.rules()["boxes"], kitsune.study_queue.box_plan) and every config its queue reads
      (study_queue.box_configs) committed at `sha` (tools/make_study_configs.py writes them);
    - study/PREREG.json at `sha` without pending fields (the shakedown may run on a pending one), and the selection
      in the data repo (its Hub sha256 at the pinned revision) equal to PREREG's manifest.selection_sha256;
    - the box's students, and only those it pulls (study_queue.box_students), each with STUDENT_FILES and a
      Parakeet-derived one with CTC_CARD, its CC-BY-4.0 attribution; the other dirs it pulls (box_extra_dirs);
    - the runs repo: no numbers file of this box yet (a box writes its numbers once, before its first study step), and
      for a box with numbers_from (the replicate) that box's numbers file present and written under this commit's
      rules (its rules_sha256 = study/PREREG.json's at `sha`)."""
    import hashlib
    import tempfile

    from kitsune import prereg
    from kitsune import study_queue as Q

    problems, notes = [], []
    rules = prereg.rules()
    try:
        plan = Q.box_plan(box, rules)
        configs = Q.box_configs(box, rules)
    except (Q.QueueError, KeyError) as e:
        return [f"box {box}: {e}"], notes
    for c in configs:
        try:
            git("cat-file", "-e", f"{sha}:{c}")
        except (subprocess.CalledProcessError, OSError):
            problems.append(f"{c} does not exist at {sha[:12]}: python tools/make_study_configs.py, commit and push")
    try:
        pr = json.loads(git_show(sha, f"study/{prereg.RULES_JSON}").decode("utf-8"))
    except (subprocess.CalledProcessError, OSError, ValueError) as e:
        return problems + [f"cannot read study/{prereg.RULES_JSON} at {sha[:12]} ({type(e).__name__})"], notes
    left = prereg.pending(pr)
    shake = bool(plan.get("shakedown"))
    if left:
        msg = (f"study/{prereg.RULES_JSON} at {sha[:12]} has {len(left)} pending field(s) (e.g. {left[:3]}): the "
               f"filled rules (python -m kitsune.prereg --write study/ --sidecar ...) are committed before a study box")
        (notes if shake else problems).append(("the shakedown runs on it anyway: " if shake else "") + msg)
    want = (pr.get("manifest") or {}).get("selection_sha256")
    try:
        api, download = _hub()
        files = api.list_repo_files(data_repo, repo_type="dataset", revision=data_rev)
        have = set(files)
        ctc = set(Q.box_ctc_students(box, rules))
        for s in Q.box_students(box, rules):
            names = STUDENT_FILES + ((CTC_CARD,) if s in ctc else ())
            if missing := [n for n in names if f"{s}/{n}" not in have]:
                problems.append(f"{data_repo}: {s} lacks {missing}: upload the student dir (hf upload {data_repo} "
                                f". . --repo-type dataset --include '{s}/*')")
        for d in Q.box_extra_dirs(box, rules):
            if not any(f.startswith(f"{d}/") for f in files):
                problems.append(f"{data_repo}: no {d}/ (the box pulls it)")
        sel = cfg["selection"]
        if sel in have and want not in (None, prereg.PENDING):
            info = api.get_paths_info(data_repo, [sel], repo_type="dataset", revision=data_rev)
            got = _lfs_sha256(info[0]) if info else None
            if got is None:  # a small file in git, not LFS/Xet: hash it
                with tempfile.TemporaryDirectory(prefix="kitsune-launch-") as tmp:
                    got = prereg.file_sha256(download(data_repo, sel, repo_type="dataset", revision=data_rev,
                                                      local_dir=tmp))
            if got != want:
                problems.append(f"{data_repo}@{(data_rev or 'head')[:12]}: {sel} has sha256 {str(got)[:12]}..., "
                                f"PREREG's manifest.selection_sha256 is {want[:12]}...: the box would train on another "
                                f"selection than the pre-registered one")
            else:
                notes.append(f"selection {sel}: sha256 {got[:12]}... = PREREG's")
    except Exception as e:  # noqa: BLE001
        problems.append(f"cannot check the study data in {data_repo}: {type(e).__name__}: {e}")
    try:
        api, _ = _hub()
        mine = f"{Q.NUMBERS_DIR}/{plan['numbers_file']}" if plan.get("numbers_file") else None
        if mine and api.file_exists(out_repo, mine):
            problems.append(f"{out_repo} already holds {mine}: box {box} wrote its numbers on an earlier box, and a "
                            f"box writes them once, before its first study step (a relaunch after that is the owner's "
                            f"call: see vast/README.md, Size study)")
        if plan.get("numbers_from"):
            src = f"{Q.NUMBERS_DIR}/{rules['boxes'][plan['numbers_from']]['numbers_file']}"
            if not api.file_exists(out_repo, src):
                problems.append(f"{out_repo} has no {src}: box {box} takes its max_steps and LR from box "
                                f"{plan['numbers_from']}'s numbers; that box runs first")
            else:
                # the numbers are written under the rules of this commit (prereg.write_numbers refuses otherwise, on
                # the box, after its bootstrap and store build): check here, before anything is rented
                _, download = _hub()
                with tempfile.TemporaryDirectory(prefix="kitsune-launch-") as tmp:
                    src_rules = json.loads(Path(download(out_repo, src, local_dir=tmp)).read_text(
                        encoding="utf-8")).get("rules_sha256")
                here = hashlib.sha256(prereg.rules_json(pr)).hexdigest()
                if src_rules != here:
                    problems.append(f"{out_repo}/{src} was written under the rules {str(src_rules)[:12]}..., "
                                    f"study/{prereg.RULES_JSON} at {sha[:12]} is {here[:12]}...: box {box} would take "
                                    f"box {plan['numbers_from']}'s numbers under other rules (the box refuses them)")
                else:
                    notes.append(f"box {box} takes its numbers from {out_repo}/{src} (rules {here[:12]}... = "
                                 f"this commit's)")
    except Exception as e:  # noqa: BLE001
        problems.append(f"cannot check the numbers files in {out_repo}: {type(e).__name__}: {e}")
    return problems, notes


def sanitize_tag(branch: str) -> str:
    """The branch tag docker/metadata-action writes: '/' and other invalid characters become '-'."""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", branch)[:128]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", choices=sorted([*JOBS, "study"]), default="train",
                    help="train: the A100 run (default); label: the RTX 5090 label box (vast/label.py); study: a "
                         "size-study box (--box; kitsune/study_queue.py)")
    ap.add_argument("--box", choices=STUDY_BOXES, default=None,
                    help="study: A (the Cohere runs, 4 GPUs), B (Parakeet + bridge, 4 GPUs), replicate (1 GPU, after "
                         "A) or shakedown (1 GPU, first)")
    ap.add_argument("--data-repo", required=True, help="private HF dataset with the derived data (KITSUNE_DATA_REPO)")
    ap.add_argument("--out-repo", default=None, help="private HF model repo for runs/ (KITSUNE_OUT_REPO; train only)")
    ap.add_argument("--config", default=None,
                    help="run config, relative to the repo root (default: configs/viability.json; label: "
                         "configs/full.json)")
    ap.add_argument("--sha", default=None, help="commit to run (default: HEAD); must be pushed to GitHub")
    ap.add_argument("--image", default=None, help=f"full image ref with digest (default: resolve {IMAGE_REPO}:<tag>)")
    ap.add_argument("--image-tag", default=None, help="tag to resolve (default: the current branch, as CI tags it)")
    ap.add_argument("--offer-id", type=int, default=None, help="rent this offer from the search results")
    ap.add_argument("--max-dph", type=float, default=None,
                    help=f"refuse offers above this $/h (default {JOBS['train'].max_dph:g}; label "
                         f"{JOBS['label'].max_dph:g})")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="watchdog cap from first boot (KITSUNE_MAX_HOURS; default 5.5, plus the rebuild timeout for "
                         "an extent config; label 30)")
    ap.add_argument("--disk-gb", type=int, default=None,
                    help="disk to rent (default: 150; an extent config: its sizing; label: 250); refused below the "
                         "computed minimum")
    ap.add_argument("--label-configs", default=LABEL_CONFIGS,
                    help="label: comma list of the configs the label box's consumer check serves "
                         "(KITSUNE_LABEL_CONFIGS); each extent must lie inside --config's")
    ap.add_argument("--steal-lease", action="store_true",
                    help="label: take over a live LEASE.json (only when that box is surely gone; "
                         "KITSUNE_STEAL_LEASE=1)")
    ap.add_argument("--avoid-machine", action="append", default=[], metavar="ID",
                    help="label: never rent this vast machine_id (repeatable); label_runs/*/label_end.json with class "
                         "host_failure/slow_host adds its machine by itself")
    ap.add_argument("--cohere-procs", type=int, default=None,
                    help="label: Cohere processes on the GPU (KITSUNE_COHERE_PROCS; default: the config's)")
    ap.add_argument("--no-self-stop", action="store_true",
                    help="debugging: the box does not stop itself when its on-start or bootstrap fails "
                         "(KITSUNE_NO_SELF_STOP=1); the watchdog still stops it once onstart.sh has started it")
    ap.add_argument("--no-hf-check", action="store_true", help="skip the read-only HF preflight")
    ap.add_argument("--skip-git-checks", action="store_true",
                    help="rent even if the commit looks unpushed/dirty or the image was built from other dependencies")
    ap.add_argument("--dry-run", action="store_true", help="never create, even with --yes")
    ap.add_argument("--yes", action="store_true", help="actually create the instance (this spends money)")
    args = ap.parse_args(argv)
    label_job, study = args.job == "label", args.job == "study"
    if study and not args.box:
        ap.error("--job study needs --box (A, B, replicate or shakedown)")
    job = study_job(args.box) if study else JOBS[args.job]
    if not label_job and not args.out_repo:
        ap.error("the following arguments are required: --out-repo")
    if args.cohere_procs is not None and args.cohere_procs < 1:
        ap.error("--cohere-procs must be >= 1")
    config = args.config or ("configs/full.json" if label_job else STUDY_CONFIG if study else "configs/viability.json")
    label_configs = list(dict.fromkeys([config] + [c for c in args.label_configs.split(",") if c])) \
        if label_job else []
    max_dph = args.max_dph if args.max_dph is not None else job.max_dph
    will_create = args.yes and not args.dry_run
    errors: list[str] = []
    notes: list[str] = []

    sha = args.sha or git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise LaunchError(f"--sha must be a full 40-hex commit id, got {sha!r}")
    if not args.skip_git_checks:
        for c in label_configs or [config]:
            errors += [p for p in git_checks(sha, c) if p not in errors]

    image, pinned = args.image, True
    if image is None:
        tag = args.image_tag or sanitize_tag(git("rev-parse", "--abbrev-ref", "HEAD"))
        try:
            image = resolve_image_digest(tag)
        except (OSError, urllib.error.URLError, LaunchError, KeyError, ValueError) as e:
            errors.append(f"cannot resolve {IMAGE_REPO}:{tag} to a digest ({e}). CI tags an image only for branches "
                          f"whose pushes touch docker/**, requirements-train.txt or .github/workflows/image.yml (after "
                          f"its smoke test passes), so a code-only branch or a detached HEAD has none: pass --image-tag "
                          f"main (the build-revision check then confirms its dependencies match {sha[:12]}) or --image "
                          f"{IMAGE_REPO}@sha256:... (the package must be public)")
            image, pinned = f"{IMAGE_REPO}@sha256:<unresolved>", False
    elif "@sha256:" not in image:
        errors.append(f"--image must be pinned by digest (...@sha256:...), got {image}")
        pinned = False
    if pinned and not args.skip_git_checks:
        errors += image_problems(image, sha)

    data_rev, sizing, avoid = None, None, {str(m) for m in args.avoid_machine}
    if args.no_hf_check:
        if label_job:
            errors.append("--job label needs the HF preflight (it pins KITSUNE_DATA_REVISION and checks the lease): "
                          "drop --no-hf-check")
        elif study:
            errors.append("--job study needs the HF preflight (the extent's sizing, the PREREG and selection hashes, "
                          "the box's students, the numbers files): drop --no-hf-check")
        else:
            try:  # local git only: an extent config's sizing cannot be skipped with the Hub checks
                # --skip-git-checks: the sha may not be local; the working tree's config is the best guess
                c = json.loads((ROOT / config).read_text(encoding="utf-8")) if args.skip_git_checks                     else config_at(sha, config)
                has_extent = bool(c.get("extent"))
            except Exception:  # noqa: BLE001  an unreadable config is git_checks' finding (unless skipped)
                has_extent = False
            if has_extent and (args.disk_gb is None or args.max_hours is None):
                errors.append(f"{config} has an extent: its disk and hours come from the HF preflight; with "
                              f"--no-hf-check pass --disk-gb and --max-hours explicitly")
    if not args.no_hf_check:
        try:
            cfg = config_at(sha, config)
            label_cfgs = {c: (cfg if c == config else config_at(sha, c)) for c in label_configs}
        except (subprocess.CalledProcessError, OSError, ValueError) as e:
            errors.append(f"cannot read {', '.join(label_configs or [config])} at {sha[:12]} ({type(e).__name__}), so "
                          f"the HF preflight did not run")
        else:
            if label_job:
                data_rev, problems, pre_notes = label_preflight(args.data_repo, sha, cfg, label_cfgs,
                                                                steal_lease=args.steal_lease)
                errors += problems
                notes += pre_notes
                hub_avoid, avoid_notes = avoided_machines(args.data_repo, data_rev)
                avoid |= hub_avoid
                notes += avoid_notes
            else:
                data_rev, problems = hf_preflight(args.data_repo, args.out_repo, cfg)
                errors += problems
                extra_gb = 0.0
                if study:
                    from kitsune import study_queue

                    try:
                        extra_gb = study_queue.study_extra_gb(args.box, n_gpus=STUDY_GPUS[args.box])
                    except (study_queue.QueueError, KeyError) as e:
                        errors.append(f"box {args.box}: {e}")
                    problems, pre_notes = study_preflight(args.data_repo, data_rev, args.out_repo, sha, args.box,
                                                          cfg)
                    errors += problems
                    notes += pre_notes
                if cfg.get("extent"):
                    problems, sizing = extent_preflight(args.data_repo, data_rev, cfg, extra_gb=extra_gb)
                    errors += problems
                elif study:
                    errors.append(f"{config} has no extent: a study box rebuilds the sealed label extent")
    stub = ONSTART.read_bytes()
    if len(stub) > ONSTART_MAX_BYTES:
        errors.append(f"{ONSTART.name} is {len(stub)} bytes (> {ONSTART_MAX_BYTES}): vast may truncate the on-start "
                      f"field, and a truncated script starts nothing, not even the watchdog")

    # disk, hours and traffic: the job's, or an extent config's sizing (the A100 rebuilds the subset's audio)
    min_disk = LABEL_DISK_MIN_GB if label_job else sizing["disk_gb"] if sizing else DISK_GB
    disk_gb = args.disk_gb or (sizing["disk_gb"] if sizing else job.disk_gb)
    if disk_gb < min_disk:
        errors.append(f"--disk-gb {disk_gb} is below the {min_disk} GB this run needs")
    est_down = sizing["down_gb"] if sizing else job.est_down_gb
    max_hours = args.max_hours if args.max_hours is not None else \
        job.max_hours + (sizing["rebuild_timeout_min"] / 60 if sizing else 0)
    if sizing:
        stores = f" x {sizing['stores']} stores" if sizing.get("stores", 1) > 1 else ""
        extra = f", checkpoints ~{sizing['extra_gb']:.0f} GB" if sizing.get("extra_gb") else ""
        notes.append(f"extent sizing: ~{sizing['down_gb']:.0f} GB upstream down, shards ~{sizing['shard_gb']:.0f} GB, "
                     f"selected audio ~{sizing['sel_gb']:.0f} GB{stores}, labels ~{sizing['labels_gb']:.1f} GB"
                     f"{extra} -> disk {sizing['disk_gb']} GB, rebuild timeout {sizing['rebuild_timeout_min']} min, "
                     f"max hours {max_hours:g}")
    for n in notes:
        print(n)

    exe = shutil.which("vastai")
    if exe is None:
        print("\nproblems:\n  " + "\n  ".join(errors) if errors else "git, image and HF checks passed")
        print(install_help())
        return 2

    if label_job:
        env = {"KITSUNE_JOB": "label", "KITSUNE_SHA": sha, "KITSUNE_CONFIG": config,
               "KITSUNE_LABEL_CONFIGS": ",".join(label_configs), "KITSUNE_DATA_REPO": args.data_repo}
    elif study:
        # KITSUNE_N_GPUS: the queue refuses to run on another GPU count than the box was rented with
        env = {"KITSUNE_JOB": "study", "KITSUNE_BOX": args.box, "KITSUNE_N_GPUS": str(STUDY_GPUS[args.box]),
               "KITSUNE_SHA": sha, "KITSUNE_CONFIG": config, "KITSUNE_DATA_REPO": args.data_repo,
               "KITSUNE_OUT_REPO": args.out_repo}
    else:
        env = {"KITSUNE_SHA": sha, "KITSUNE_CONFIG": config, "KITSUNE_DATA_REPO": args.data_repo,
               "KITSUNE_OUT_REPO": args.out_repo}
    env["KITSUNE_MAX_HOURS"] = f"{max_hours:g}"
    if not label_job:
        env["TZ"] = "UTC"
    if data_rev:
        env["KITSUNE_DATA_REVISION"] = data_rev
    if sizing:
        env["KITSUNE_REBUILD_TIMEOUT_MIN"] = str(sizing["rebuild_timeout_min"])
    if label_job:
        env.update(LABEL_WATCHDOG_ENV)
        env["KITSUNE_IMAGE"] = image
        if args.steal_lease:
            env["KITSUNE_STEAL_LEASE"] = "1"
        if args.cohere_procs is not None:
            env["KITSUNE_COHERE_PROCS"] = str(args.cohere_procs)
    if args.no_self_stop:  # inside the one --env value: vastai has no -e option, and a second --env replaces the first
        env["KITSUNE_NO_SELF_STOP"] = "1"

    tier, query, offers = search_offers(exe, job, disk_gb, avoid, max_dph if label_job else None)
    print(f"\nsearch ({tier or 'nothing found'}): vastai "
          f"{shlex.join(search_args(query or job_query(job, job.tiers[0][1], disk_gb), disk_gb))}")
    if not offers:
        errors.append("no RTX 5090 offer matches the label filter (see the hint above); try again later" if label_job
                      else f"no {STUDY_GPUS[args.box]}x A100 offer matches the study filter; try again later"
                      if study else "no offer matches the strict filter (D45a); try again later or relax it by hand")
        offer = None
    else:
        print(offer_table(offers, job=job if label_job else None))
        offer = next((o for o in offers if o.get("id") == args.offer_id), None) if args.offer_id else offers[0]
        if offer is None:
            errors.append(f"offer {args.offer_id} is not in the results")
        elif offer.get("dph_total", 0) > max_dph:
            errors.append(f"offer {offer['id']} costs ${offer['dph_total']:.3f}/h > --max-dph {max_dph}")

    if offer and isinstance(offer.get("dph_total"), (int, float)):
        env["KITSUNE_DPH"] = f"{offer['dph_total']:.4f}"  # kitsune.runlog records it for the cost estimate
    if label_job:
        if offer and offer.get("machine_id") not in (None, ""):
            env["KITSUNE_MACHINE_ID"] = str(offer["machine_id"])  # label_end.json names it for the avoid list
        elif offer:
            errors.append(f"offer {offer['id']} lists no machine_id (the box records it for the avoid list)")
        env["TZ"] = "UTC"
    label = f"{job.label_prefix}-{Path(config).stem}-{sha[:7]}"
    cargs = create_args(offer["id"] if offer else "<offer-id>", image, env, ONSTART, label, disk_gb)
    assert not any("HF_TOKEN" in a for a in cargs), "HF_TOKEN must never be on the command line"
    print(f"\ncreate command:\n  vastai {shlex.join(cargs)}")
    print("HF_TOKEN is not passed: it must be set in vast Account -> Settings -> Environment Variables.")
    if offer and label_job:
        dph = offer.get("dph_total", 0)
        down, up = _gb_cost(offer, "inet_down_cost"), _gb_cost(offer, "inet_up_cost")
        traffic = job.est_down_gb * down + job.est_up_gb * up
        print(f"expected cost: ~${dph:.3f}/h (GPU + {disk_gb} GB) x ~{job.est_hours:g} h (12-24; cap {max_hours:g} h) "
              f"+ ~{job.est_down_gb:g} GB down x ${down:.3f}/GB + ~{job.est_up_gb:g} GB up x ${up:.3f}/GB "
              f"= ~${est_total(offer, job):.2f} (cap = ~${dph * max_hours + traffic:.2f})")
    elif offer and study:
        dph = offer.get("dph_total", 0)
        down, up = _gb_cost(offer, "inet_down_cost"), _gb_cost(offer, "inet_up_cost")
        traffic = est_down * down + job.est_up_gb * up
        print(f"expected cost: ~${dph:.3f}/h ({STUDY_GPUS[args.box]}x A100 + {disk_gb} GB) x ~{job.est_hours:g} h "
              f"(box {args.box}; watchdog cap {max_hours:g} h) + ~{est_down:.0f} GB down x ${down:.3f}/GB + "
              f"~{job.est_up_gb:g} GB up x ${up:.3f}/GB = ~${dph * job.est_hours + traffic:.2f} "
              f"(cap = ~${dph * max_hours + traffic:.2f})")
    elif offer:
        dph = offer.get("dph_total", 0)
        down, up = (offer.get(k) if isinstance(offer.get(k), (int, float)) else None
                    for k in ("inet_down_cost", "inet_up_cost"))
        bw = (f"~${est_down * down + EST_UP_GB * up:.2f} (~{est_down:.0f} GB down, ~{EST_UP_GB} GB up at this host's "
              f"$/GB)" if down is not None and up is not None else "unknown (the offer lists no $/GB)")
        print(f"expected cost: ${dph:.3f}/h (GPU + {disk_gb} GB disk) x <= {max_hours:g} h (watchdog cap) "
              f"= <= ${dph * max_hours:.2f}, plus bandwidth {bw}")

    if errors:
        print("\nproblems:\n  " + "\n  ".join(errors))
    if not will_create:
        print("\nnot creating anything" + (" (--dry-run)" if args.dry_run else "; re-run with --yes to rent this offer"))
        return 1 if errors else 0
    if errors:
        print("refusing to create the instance until the problems above are fixed")
        return 1

    reply = parse_json(vastai(exe, cargs))
    iid = reply.get("new_contract") if isinstance(reply, dict) else None
    if not iid:
        raise LaunchError(f"create did not return an instance id: {reply}")
    print(f"\ncreated instance {iid} (offer {offer['id']}, ${offer.get('dph_total', 0):.3f}/h). Next:\n"
          f"  vastai show instance {iid}          # wait for 'running'\n"
          f"  vastai ssh-url {iid}                 # then: ssh -p <port> root@<ip> -L 6006:localhost:6006\n"
          f"  tail -f /workspace/kitsune.log       # on the box\n"
          f"  vastai destroy instance {iid}       # by hand, if you ever need to abort")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LaunchError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
