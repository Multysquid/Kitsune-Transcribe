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
the box missing a file or not the registered build (kitsune.prereg.student_problems on its student_meta.json), a config
of the box not committed, a numbers file of the box already written under other rules than the commit's (under the
same rules a relaunched box reuses it), for the replicate box A's missing or written under other rules; the queue's GPU
count goes along (KITSUNE_N_GPUS). Boxes A and B may be live at the same time: nothing here refuses a second box:
  python vast/launch.py --job study --box A --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs \\
      --image-tag main                                                                        # look only

--job full --box full-smoke|p01|full|smoke-b rents a full-data box (kitsune/full_queue.py runs it; vast/README.md
"Full-data runs"). The box registry configs/full/boxes.json (kitsune/fullrun.py) is read at the commit the box runs
and is the one source of its GPU count (--gpus may only repeat it), data config (--config), hours (--max-hours; planned
hours est_hours), price cap (--max-dph), extra disk and watchdog (KITSUNE_N_GPUS, KITSUNE_WATCHDOG_*). The offers: RTX
5090s (--tier a100: A100s, option C, cap A100_MAX_DPH), full_filter per GPU, then a client filter (verified or
deverified hosts only, a rental that runs >= MIN_RENTAL_DAYS, >= 64 GB RAM per GPU, --machine), ranked by the estimated
total. Refused before renting (full_preflight): a box config not committed at the commit, a student not the
registered build, an extra file or dir the data repo lacks, an eval/speed tool the commit does not have, a scratch
repo (--scratch-repo, required for a box with timed states) that is not private, a selection sidecar that is not the
selection's, and with --resume a box whose Hub queue summary is missing or a --resume-reset/--resume-set run id no
train item of it ran. Every job avoids the machines of vast/blocklist.json; a full box also those whose download gate
said slow in the last GATE_BLOCK_DAYS (full/box-*/infra/*/download_gate.json) and the label runs' failed hosts. The
box times its Hub link first (kitsune.netgate: KITSUNE_GATE_BYTES, --gate-hours; 0 turns it off):
  python vast/launch.py --job full --box p01 --machine 54650 --image-tag main --data-repo Multy123/kitsune-data \\
      --out-repo Multy123/kitsune-runs --scratch-repo Multy123/kitsune-scratch                  # look only
--job full --box p01-chain rents a chain box (contract addendum E; kitsune/full_queue.py ChainController): smoke A and
smoke B, an automatic gate, then box 1, on one 1x RTX 5090. Its disk and download gate are sized on the last stage's
extent (box 1's, with the chain's extra_gb), the boot's rebuild timeout and bytes on stage 1's (KITSUNE_CONFIG is stage
1's rebuild, KITSUNE_CHAIN_STAGE=1, the watchdog's stage-1 env with KITSUNE_WATCHDOG_HANDOVER_S); every part is
preflighted as a box of its own, plus the chain's configs and stage files. Refused for a chain: --config,
--gate-hours 0, --max-hours below stage 1's hours + the last stage's est_hours (a warning below the chain's max_hours),
and any resume flag (fullrun.chain_resume_hint says what to run instead); --box p01 --resume is refused when the
chain's newer summary shows the Hub's p01 summary is another rental's.
Needs the vastai CLI (`pip install vastai==1.8.0`, then `vastai set api-key <key>`); --help works without it.
"""
import argparse
import importlib
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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # kitsune.extent (pure Python) when run as `python vast/launch.py`
    sys.path.insert(1, str(ROOT))

from kitsune import fullrun  # noqa: E402  stdlib only at import (the box registry and the full runs' names)
from kitsune.netgate import DEFAULT_MAX_H as DEFAULT_GATE_H, GATE_REF_GB  # noqa: E402  stdlib only at import

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
# the shakedown covers both families (CONTRACT.md 8): boot, bootstrap and the AED store ~0.55 h, the CTC store with its
# frame preflight ~0.25 h (MP3 decode), 18 trainer starts (8 smokes, 2 x resume twice, parent, branch, eval) at
# ~2.5 min set-up each ~0.75 h, their steps and evals ~0.35 h, the end ~0.1 h
STUDY_HOURS = {"A": (5.8, 8.0), "B": (6.3, 8.5), "replicate": (4.6, 6.0), "shakedown": (2.0, 3.5)}
STUDY_MAX_DPH = {4: 6.0, 1: 2.0}  # by GPU count
STUDY_UP_GB = {"A": 12, "B": 10, "replicate": 3, "shakedown": 5}  # lean uploads (STUDY.md 5.5)
STUDY_CONFIG = "study/data.json"

# --job full (plan v3; vast/README.md "Full-data runs"): one box of the registry configs/full/boxes.json, read at the
# commit the box runs. The registry is the one source of the box's GPU count, hours, price cap, extra disk and
# watchdog: launch keeps no box table. RTX 5090s (decision 10; 32 GB, driver CUDA >= 13.0), the A100s as option C.
# Per GPU: >= 16 effective cores (the trainer's loader workers and 01's prep pools), >= 100 Mbit/s up (lean syncs and
# the timed states to the scratch repo); verified=any in the query because vast's verified=true also drops
# "deverified" hosts, and the client filter then keeps verified and deverified ones, never an unverified one
FULL_TIERS = {
    "5090": [("RTX 5090", ["gpu_name=RTX_5090", "gpu_ram>=30"])],
    "a100": [("A100 40 GB (SXM4 or PCIe)", ["gpu_name in [A100_SXM4,A100_PCIE]", "gpu_ram<=48"]),
             ("A100 80 GB (SXM4 or PCIe; fallback)", ["gpu_name in [A100_SXM4,A100_PCIE]", "gpu_ram>=70"])],
}
A100_MAX_DPH = 2.60  # --tier a100's default price cap (option C: 2x A100 for box 2)
FULL_VERIFICATION = ("verified", "deverified")  # never "unverified" (decision 1)
MIN_RENTAL_DAYS = 4  # the host's max rental must outlast the box (decision 12; m52214 listed 0.4 d)
# vast converts the query's cpu_ram GB to MB itself, loosely (m54650's 1x lists 64,439 MB), so the query asks for 60 GB
# a GPU and the client filter for 64,000 MB a GPU (m140586's 2x at 126,367 MB fails, as planned)
FULL_RAM_MB_PER_GPU = 64_000
FULL_UP_GB_PER_GPU_HOUR = 3.0  # the cost line's upload: lean syncs plus the timed states, ~250 GB for box 2
GATE_BLOCK_DAYS = 30  # a machine whose download gate said slow is avoided this long (vast/blocklist.json: for good)
BLOCKLIST = Path(__file__).resolve().parent / "blocklist.json"
GATE_RE = re.compile(r"full/box-[^/]+/infra/[^/]+/download_gate\.json")
# the tools a registry item runs besides its argv template (kitsune/full_queue.py builds these argvs)
FULL_TOOLS = {"stores": "kitsune/full_queue.py", "train": "scripts/04_distill.py", "readout": "scripts/05_evaluate.py",
              "speed": "tools/speed_probe.py"}


def full_filter(n_gpus: int, disk_gb: int = DISK_GB) -> list[str]:
    """The full box's host filter for n GPUs (plan v3 section 3); the client filter (offer_problems) does the rest."""
    return [f"num_gpus={n_gpus}", "verified=any", "rentable=true", "reliability>=0.98", "cuda_vers>=13.0",
            f"cpu_cores_effective>={16 * n_gpus}", f"cpu_ram>={60 * n_gpus}", "disk_bw>=500", "inet_down>=500",
            f"inet_up>={100 * n_gpus}", "direct_port_count>=1", f"disk_space>={disk_gb}"]


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
    # the client filter (offer_problems); the defaults filter nothing, so train, label and study rank as before
    n_gpus: int = 1
    accept_verification: tuple | None = None  # the offer's "verification" must be one of these (None: any)
    min_rental_days: float = 0.0  # the host's max rental ("duration") at least this long
    ram_mb_per_gpu: int = 0  # cpu_ram (MB) at least this x n_gpus


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


def full_job(box: str, spec: dict, tier: str, plan_hours: float, max_hours: float, max_dph: float) -> JobSpec:
    """The JobSpec of a full box from its registry spec: its GPU count in full_filter, the tier's GPUs, the planned and
    capped hours, uploads at FULL_UP_GB_PER_GPU_HOUR, ranked by the estimated total, and the client filter. The
    download (est_down_gb) comes from the extent's sizing later."""
    n = spec["gpus"]
    return JobSpec(FULL_TIERS[tier], full_filter(n), DISK_GB, 0.0, FULL_UP_GB_PER_GPU_HOUR * n * plan_hours,
                   plan_hours, max_hours, max_dph, "est_total", f"kitsune-full-{box}", n_gpus=n,
                   accept_verification=FULL_VERIFICATION, min_rental_days=MIN_RENTAL_DAYS,
                   ram_mb_per_gpu=FULL_RAM_MB_PER_GPU)


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


def _duration_s(offer: dict, now: float) -> float | None:
    """The host's max rental from now, in seconds: the offer's `duration`, else end_date - now; None when neither."""
    d = offer.get("duration")
    if isinstance(d, (int, float)):
        return float(d)
    end = offer.get("end_date")
    return float(end) - now if isinstance(end, (int, float)) else None


def offer_problems(offer: dict, job: JobSpec, now: float | None = None) -> list[str]:
    """Why an offer fails the job's client-side filter (what vast's query cannot say exactly): its verification, its
    max rental, its RAM per GPU. Empty for every offer under the train, label and study jobs (JobSpec defaults)."""
    now = time.time() if now is None else now
    out = []
    if job.accept_verification is not None and offer.get("verification") not in job.accept_verification:
        out.append(f"verification {offer.get('verification')!r} (only {', '.join(job.accept_verification)})")
    if job.min_rental_days:
        d = _duration_s(offer, now)
        if d is None or d < job.min_rental_days * 86400:
            out.append(f"max rental {'unknown' if d is None else f'{d / 86400:.1f} d'} < {job.min_rental_days:g} d")
    if job.ram_mb_per_gpu:
        ram = offer.get("cpu_ram")
        if not isinstance(ram, (int, float)) or ram < job.ram_mb_per_gpu * job.n_gpus:
            out.append(f"cpu_ram {ram} MB < {job.ram_mb_per_gpu * job.n_gpus} MB")
    return out


def rank_offers(offers: list[dict], job: JobSpec, avoid=frozenset(), max_dph: float | None = None,
                machine: str | None = None, now: float | None = None) -> list[dict]:
    """Offers without the avoided machines (and only on `machine` when given) that pass the job's client filter
    (offer_problems), cheapest first by $/h (train) or by est_total (label, full). An est_total ranking drops offers
    over max_dph first: its winner is the cheapest total, which need not be the cheapest per hour."""
    avoid = {str(m) for m in avoid}
    kept = [o for o in offers if str(o.get("machine_id")) not in avoid]
    if machine is not None:
        kept = [o for o in kept if str(o.get("machine_id")) == str(machine)]
    kept = [o for o in kept if not offer_problems(o, job, now)]
    if job.sort == "est_total":
        if max_dph is not None:
            kept = [o for o in kept if isinstance(o.get("dph_total"), (int, float)) and o["dph_total"] <= max_dph]
        return sorted(kept, key=lambda o: est_total(o, job))
    return sorted(kept, key=lambda o: o.get("dph_total", 1e9))


def search_offers(exe: str, job: JobSpec | None = None, disk_gb: int | None = None,
                  avoid=frozenset(), max_dph: float | None = None,
                  machine: str | None = None) -> tuple[str, str, list[dict]]:
    """-> (tier name, query, ranked offers) for the first tier with any offer that is not on an avoided machine and
    passes the job's client filter (on `machine` only, when given)."""
    job = job or JOBS["train"]
    disk_gb = disk_gb or job.disk_gb
    for name, terms in job.tiers:
        query = job_query(job, terms, disk_gb)
        offers = parse_json(vastai(exe, search_args(query, disk_gb)))
        if isinstance(offers, dict):
            offers = offers.get("offers", [])
        ranked = rank_offers(offers, job, avoid, max_dph, machine)
        if ranked:
            return name, query, ranked
        why = ""
        if offers:  # (a job without a client filter, --machine or a cap only loses offers to the avoided machines)
            avoided = sum(str(o.get("machine_id")) in {str(m) for m in avoid} for o in offers)
            why = (f" (all {len(offers)} on avoided machines)" if avoided == len(offers) else
                   f" ({avoided} of {len(offers)} on avoided machines, the others dropped by the client filter, "
                   f"--machine or the price cap)")
        print(f"no offers for {name}{why}: {query}")
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
    if job is not None and job.accept_verification is not None:  # full: what the client filter looked at
        cols[-1:-1] = [("verif", "verification", "{}"), ("maxd", "_maxd", "{:.1f}")]
    rows = [[h for h, _, _ in cols]]
    now = time.time()
    for o in offers[:limit]:
        row = []
        for _, key, fmt in cols:
            if key == "_maxd":
                d = _duration_s(o, now)
                v = None if d is None else d / 86400
            else:
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
    without pull_parakeet needs the teacher files of its kept eval rows only.
    A full-data run's selection (selection_recipe.full_study, kitsune/devslice.py) must carry the same full_study
    block as the config, which must be the registered one (kitsune.fullrun.full_recipe_problems), and record the frozen
    study manifest's sha (fullrun.FROZEN_MANIFEST_SHA256) its eval rows equal, the study's seed (SELECTION_SEED) and
    greedy_n (kitsune.devslice.GREEDY_N), on which its dev slice and greedy subsets hang. Its splits are train, dev
    and eval (only a full_study selection may have dev rows), every train source keeps dev rows (the trainer's early
    stop scores them), its drop reasons are kitsune.devslice.FULL_REASONS (not_drawn only with a draw_audio_s), K6 and
    K5 hold with K5 counted over the train and dev rows (the dev rows were train rows), and with `have` its sidecar and
    the frozen manifest (devslice.selection_files) are in the data repo."""
    import pandas as pd
    import pyarrow.parquet as pq

    from kitsune import devslice, fullrun, prereg

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
                   extent=args.get("extent") or None, study=args.get("study") or None,
                   full_study=args.get("full_study") or None)  # older selections lack the key: None
        ext = cfg.get("extent") or None  # make_selection records {name, inputs} of the config's extent (None: none)
        want = dict(sources=set(cfg.get("sources", [])), eval_sets=set(cfg.get("eval_sets", [])),
                    agree_max=float(recipe["agree_max"]), agree_max_source=_by_source(recipe["agree_max_source"]),
                    filter_eval_sets=set(recipe["filter_eval_sets"]),
                    partial_second_opinion=set(recipe.get("partial_second_opinion", [])),
                    extent=dict(name=ext.get("name"), inputs=ext.get("inputs") or {}) if ext else None,
                    study=recipe.get("study") or None, full_study=recipe.get("full_study") or None)
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
        if want["full_study"] is not None:  # the full runs' recipe and the frozen manifest its eval rows equal
            problems += [f"the run config's {p}" for p in fullrun.full_recipe_problems(want["full_study"])]
            if args.get("manifest_sha256") != fullrun.FROZEN_MANIFEST_SHA256:
                problems.append(f"{name} was built against the manifest sha256 {args.get('manifest_sha256')!r}, not "
                                f"the frozen {fullrun.FROZEN_MANIFEST} ({fullrun.FROZEN_MANIFEST_SHA256}): {rebuild}")
            # the dev draw, the probe and the greedy subsets hang on the seed, the greedy subsets on greedy_n: the
            # study's, so the dev slice is the registered one and the greedy subsets are the study selection's
            if args.get("seed") != prereg.SELECTION_SEED:
                problems.append(f"{name} was built with seed {args.get('seed')!r}, a full selection's is the study's "
                                f"{prereg.SELECTION_SEED}: {rebuild}")
            if args.get("greedy_n") != devslice.GREEDY_N:
                problems.append(f"{name} was built with greedy_n {args.get('greedy_n')!r}, a full selection's is the "
                                f"study's {devslice.GREEDY_N} (kitsune.devslice.GREEDY_N): {rebuild}")

    full = isinstance(recipe, dict) and recipe.get("full_study") is not None
    sel = pd.read_parquet(path, columns=["source", "split", "keep", "reason", "teacher_file"])
    splits = {"train", fullrun.DEV_SPLIT, "eval"} if full else {"train", "eval"}
    if odd := sorted(set(sel["split"].unique()) - splits):
        problems.append(f"{name} has rows of split {odd}, which {'a full' if full else 'this'} recipe never "
                        f"writes{'' if full else ' (dev rows belong to a selection_recipe.full_study selection)'}: "
                        f"{rebuild}")
    kept = sel[sel["keep"]].groupby(["source", "split"]).size()
    for split, key in (("train", "sources"), ("eval", "eval_sets")) + ((fullrun.DEV_SPLIT, "sources"),) * full:
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
    if full:  # a draw only with draw_audio_s (the smoke selection); a malformed block is reported above
        fs = recipe["full_study"] if isinstance(recipe["full_study"], dict) else {}
        allowed |= set(devslice.FULL_REASONS) - (set() if fs.get("draw_audio_s") is not None else {"not_drawn"})
    if odd := sorted(r for r in sel["reason"].unique() if r not in allowed and not str(r).startswith("agree>")):
        problems.append(f"{name} drops rows as {odd}, which {'the study' if study else 'the full' if full else 'this'} "
                        f"recipe never does: {rebuild}")
    if study or full:
        ev = sel[(sel["split"] == "eval") & (sel["reason"] == "not_in_parakeet")]
        if not ev.empty:
            problems.append(f"{name}: {len(ev)} eval rows have no Parakeet labels "
                            f"({ev['source'].value_counts().to_dict()}): the eval sets must be labelled whole by both "
                            f"passes (K6)")
        # K5 over the rows that were train rows before the dev draw (a dev row keeps its not_in_parakeet reason)
        train = sel[sel["split"].isin(["train", fullrun.DEV_SPLIT] if full else ["train"])]
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
        if full:
            for f in devslice.selection_files(cfg):
                if f not in have:
                    problems.append(f"no {f}: a full selection needs its sidecar and the frozen study manifest")
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
    - each of those students the registered build: kitsune.prereg.student_problems on its student_meta.json in the data
      repo (study_queue.student_checks: stage, family, init class, seed, the exact parameter counts, a pruned
      student's calibration ids); bootstrap.sh checks the pulled copies again;
    - the runs repo: this box's numbers file, if it is there already (a relaunch), written under this commit's rules
      (the box then reuses it: numbers are written once, before a box's first study step), and for a box with
      numbers_from (the replicate) that box's numbers file present and written under this commit's rules (its
      rules_sha256 = study/PREREG.json's at `sha`).
    Nothing here looks at other live boxes: boxes A and B run at the same time (CONTRACT.md 8), each with its own run
    ids, queue summary and numbers file in the one runs repo."""
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
        # each student is the registered build (kitsune.prereg.student_problems on the data repo's student_meta.json:
        # counts, seed, init class, a pruned student's calibration ids): a stale upload is refused before renting
        with tempfile.TemporaryDirectory(prefix="kitsune-launch-") as tmp:
            def read_meta(s: str) -> dict:
                return json.loads(Path(download(data_repo, f"{s}/student_meta.json", repo_type="dataset",
                                                revision=data_rev, local_dir=tmp)).read_text(encoding="utf-8"))

            bad = Q.student_checks(box, read_meta, rules)
        problems += [f"{data_repo}: {p}" for p in bad]
        if not bad:
            notes.append(f"box {box}'s {len(Q.box_students(box, rules))} student(s) are the registered builds "
                         f"(kitsune.prereg.student_problems)")
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
        here = hashlib.sha256(prereg.rules_json(pr)).hexdigest()
        mine = f"{Q.NUMBERS_DIR}/{plan['numbers_file']}" if plan.get("numbers_file") else None
        if mine and api.file_exists(out_repo, mine):
            # a relaunch: the numbers are write-once, and the box reuses them (study_queue adopt_numbers: no second
            # calibration, no probes) - only under the rules they were written under, which the box checks again
            _, download = _hub()
            with tempfile.TemporaryDirectory(prefix="kitsune-launch-") as tmp:
                got = json.loads(Path(download(out_repo, mine, local_dir=tmp)).read_text(encoding="utf-8"))
            if got.get("rules_sha256") != here:
                problems.append(f"{out_repo}/{mine} was written under the rules "
                                f"{str(got.get('rules_sha256'))[:12]}..., study/{prereg.RULES_JSON} at {sha[:12]} is "
                                f"{here[:12]}...: a relaunched box {box} "
                                f"reuses its numbers only under the rules they were written under (it never writes "
                                f"them twice); relaunch from the commit they were written at")
            else:
                notes.append(f"relaunch: {out_repo}/{mine} is there (rules {here[:12]}... = this commit's): box {box} "
                             f"REUSES it and skips its calibration and probes")
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


# ------------------------------------------------------------------------------------------------- --job full


def load_blocklist(path: Path = BLOCKLIST) -> dict[str, str]:
    """vast/blocklist.json: {"_comment": str, "machines": {"<machine_id>": "<reason>"}}, the machines no job rents
    (study box A #1's 2.9 MB/s host, 151760, is the first). Malformed -> LaunchError: a broken committed file must never
    silently unblock a machine."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LaunchError(f"{path}: not readable JSON ({type(e).__name__}: {e})") from None
    machines = data.get("machines") if isinstance(data, dict) else None
    if (not isinstance(machines, dict) or set(data) - {"_comment", "machines"}
            or not all(re.fullmatch(r"\d+", str(k)) and isinstance(v, str) and v.strip() for k, v in machines.items())):
        raise LaunchError(f'{path}: not {{"_comment": str, "machines": {{"<machine_id digits>": "<reason>"}}}}')
    return {str(k): v for k, v in machines.items()}


def gate_refusals(out_repo: str, now: float | None = None,
                  max_age_days: float = GATE_BLOCK_DAYS) -> tuple[dict[str, str], list[str]]:
    """-> ({machine_id: why}, notes): the machines whose full box's download gate said slow in the last max_age_days
    (full/box-*/infra/*/download_gate.json in the runs repo: finish.py's infra upload after kitsune.netgate refused
    the host), best effort: a Hub it cannot read leaves a note and no machine."""
    import tempfile

    now = time.time() if now is None else now
    out, notes = {}, []
    try:
        api, download = _hub()
        gates = [f for f in api.list_repo_files(out_repo) if GATE_RE.fullmatch(f)]
        with tempfile.TemporaryDirectory(prefix="kitsune-launch-") as tmp:
            for f in gates:
                gate = json.loads(Path(download(out_repo, f, local_dir=tmp)).read_text(encoding="utf-8"))
                wall, mid = gate.get("wall"), gate.get("machine_id")
                if gate.get("verdict") != "slow" or mid in (None, "") or not isinstance(wall, (int, float)):
                    continue
                if now - wall > max_age_days * 86400:
                    notes.append(f"machine {mid}: {f} said slow {(now - wall) / 86400:.0f} d ago (> {max_age_days:g} "
                                 f"d): no longer avoided")
                    continue
                out[str(mid)] = f"its download gate said slow {(now - wall) / 86400:.1f} d ago ({f})"
    except Exception as e:  # noqa: BLE001
        notes.append(f"could not read the download gates in {out_repo} ({type(e).__name__}: {e}); only the blocklist "
                     f"and --avoid-machine apply")
    return out, notes


def worktree_file(rel: str) -> bytes | None:
    """The working tree's copy of a repo file (None: not there); a seam for the tests."""
    p = ROOT / rel
    return p.read_bytes() if p.is_file() else None


def full_registry(sha: str, *, skip_git_checks: bool = False) -> tuple[dict | None, object, list[str], list[str]]:
    """-> (registry, reader, problems, notes): configs/full/boxes.json as committed at `sha` (the box runs that commit),
    validated by kitsune.fullrun with its data and item configs read at the same sha (reader(rel) -> the parsed JSON of
    a repo file at sha). A working-tree copy that differs from the sha's is a problem (an uncommitted edit would be
    rented without its change), unless skip_git_checks; with skip_git_checks and no copy at the sha (it is not local),
    the working tree's is the best guess, with a note. registry None: unusable (the problems say why)."""
    problems, notes = [], []

    def reader(rel: str):
        return json.loads(git_show(sha, rel).decode("utf-8"))

    try:
        raw = git_show(sha, fullrun.BOXES_FILE)
    except (subprocess.CalledProcessError, OSError):
        raw, wt = None, worktree_file(fullrun.BOXES_FILE)
        if skip_git_checks and wt is not None:
            notes.append(f"{fullrun.BOXES_FILE} is not readable at {sha[:12]}: using the working tree's "
                         f"(--skip-git-checks)")
            raw = wt

            def reader(rel: str):  # noqa: F811  the working tree's files, as the registry's
                return json.loads((ROOT / rel).read_text(encoding="utf-8"))
        else:
            return None, None, [f"{fullrun.BOXES_FILE} does not exist at {sha[:12]}: the box registry is committed "
                                f"with the full configs (tools/make_full_configs.py); commit and push it"], notes
    else:
        wt = worktree_file(fullrun.BOXES_FILE)
        try:
            differs = wt is not None and json.loads(wt.decode("utf-8")) != json.loads(raw.decode("utf-8"))
        except ValueError:
            differs = wt is not None
        if differs and not skip_git_checks:
            problems.append(f"the working tree's {fullrun.BOXES_FILE} differs from the one at {sha[:12]}, which the "
                            f"box runs: commit and push the change (or pass --skip-git-checks)")
    try:
        reg = fullrun.load_registry(json.loads(raw.decode("utf-8")), read_json=reader)
    except ValueError as e:  # RegistryError, or JSON that does not parse
        return None, reader, problems + [f"{fullrun.BOXES_FILE} at {sha[:12]}: {e}"], notes
    return reg, reader, problems, notes


def argv_target(argv) -> str | None:
    """The repo file an eval item's argv template runs: `-m pkg.mod` -> pkg/mod.py, else its first *.py argument."""
    argv = [a for a in argv if isinstance(a, str)]
    for i, a in enumerate(argv):
        if a == "-m" and i + 1 < len(argv):
            return argv[i + 1].replace(".", "/") + ".py"
        if a.endswith(".py"):
            return a
    return None


def _at_sha(sha: str, path: str) -> bool:
    try:
        git("cat-file", "-e", f"{sha}:{path}")
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def full_preflight(data_repo: str, data_rev: str | None, out_repo: str, scratch_repo: str | None, sha: str, box: str,
                   reg: dict, cfg: dict, *, reader=None, resume: bool = False, resets=(),
                   sets: dict | None = None) -> tuple[list[str], list[str]]:
    """-> (problems, notes) for --job full, read-only (local git and the laptop's HF login), on top of hf_preflight and
    extent_preflight:
    - every config the box reads (fullrun.box_configs: its data config, its item configs, the registry) committed at
      `sha`, and every tool its items run: the queue (kitsune/full_queue.py), the trainer, 05, speed_probe (with the
      items' speed kinds and every --flag in their args) and each eval item's argv target (`-m pkg.mod` ->
      pkg/mod.py, or its *.py): a box whose items need a CLI the commit does not have yet is refused here, not after
      its paid bootstrap;
    - its students (fullrun.box_students, read at `sha`) with STUDENT_FILES (+ CTC_CARD for a Parakeet-derived one), each
      the registered build (fullrun.student_checks on the data repo's student_meta.json); its extra files and dirs;
    - the scratch repo (the trainers' timed states) readable and private;
    - a full selection's sidecar (<selection stem>.json) passing kitsune.devslice.sidecar_problems against the Hub's
      selection and frozen manifest;
    - with resume: the box's Hub queue summary, and every --resume-reset/--resume-set id the run dir of one of its train
      items; the notes say what will resume, with the newest state step found in the scratch and runs repos."""
    import hashlib
    import tempfile

    problems, notes = [], []
    reader = reader or (lambda rel: json.loads(git_show(sha, rel).decode("utf-8")))
    spec = fullrun.box_spec(box, reg)
    for c in fullrun.box_configs(box, reg):
        if not _at_sha(sha, c):
            problems.append(f"{c} does not exist at {sha[:12]}: python tools/make_full_configs.py, commit and push")
    tools = {"kitsune/full_queue.py": "the box's queue"}
    kinds, flags = set(), {}
    for it in spec["items"]:
        if it["kind"] in FULL_TOOLS:
            tools.setdefault(FULL_TOOLS[it["kind"]], f"{it['kind']} item {it['name']}")
        if it["kind"] == "speed":
            kinds.add(it["speed_kind"])
            # the flags an item adds to the queue's speed argv (WP5's --quant/--compile, WP6's --hf-cache): a sha
            # with the kind but not the flag would end the item in argparse's exit 2 on the rented box
            for a in it.get("args") or ():
                if isinstance(a, str) and a.startswith("--") and len(a) > 2:
                    flags.setdefault(a.split("=", 1)[0], it["name"])
        if it["kind"] == "eval":
            target = argv_target(it["argv"])
            if target is None:
                problems.append(f"item {it['name']}: its argv {it['argv'][:4]} names no -m module or *.py tool")
            else:
                tools.setdefault(target, f"item {it['name']}")
    for path, who in tools.items():
        if not _at_sha(sha, path) and not (path.endswith(".py") and _at_sha(sha, path[:-3] + "/__main__.py")):
            problems.append(f"{path} ({who}) does not exist at {sha[:12]}: this box needs a commit that has it")
    if (kinds or flags) and _at_sha(sha, FULL_TOOLS["speed"]):
        # textual: KINDS lists each kind as a "<kind>" literal, and argparse declares each flag as a "--flag" one
        probe = git_show(sha, FULL_TOOLS["speed"]).decode("utf-8", "replace")
        for k in sorted(kinds):
            if f'"{k}"' not in probe:
                problems.append(f"{FULL_TOOLS['speed']} at {sha[:12]} has no --kind {k} (a speed item of box {box})")
        for f, name in sorted(flags.items()):
            if f'"{f}"' not in probe:
                problems.append(f"{FULL_TOOLS['speed']} at {sha[:12]} has no {f} (in the args of speed item {name} "
                                f"of box {box})")
    try:
        api, download = _hub()
        files = api.list_repo_files(data_repo, repo_type="dataset", revision=data_rev)
        have = set(files)
        students = fullrun.box_students(box, reg, read_json=reader)
        ctc = set(fullrun.box_ctc_students(box, reg, read_json=reader))
        for s in students:
            names = STUDENT_FILES + ((CTC_CARD,) if s in ctc else ())
            if missing := [n for n in names if f"{s}/{n}" not in have]:
                problems.append(f"{data_repo}: {s} lacks {missing}: upload the student dir")
        for f in spec["extra_files"]:
            if f not in have:
                problems.append(f"{data_repo}: no {f} (box {box} pulls it)")
        for d in spec["extra_dirs"]:
            if not any(f.startswith(f"{d}/") for f in files):
                problems.append(f"{data_repo}: no {d}/ (box {box} pulls it)")
        with tempfile.TemporaryDirectory(prefix="kitsune-launch-") as tmp:
            def read_meta(s: str) -> dict:
                return json.loads(Path(download(data_repo, f"{s}/student_meta.json", repo_type="dataset",
                                                revision=data_rev, local_dir=tmp)).read_text(encoding="utf-8"))

            bad = fullrun.student_checks(box, read_meta, reg, read_json=reader)
            problems += [f"{data_repo}: {x}" for x in bad]
            if not bad:
                notes.append(f"box {box}'s {len(students)} student(s) are the registered builds")
            recipe = (cfg.get("selection_recipe") or {}).get("full_study")
            if recipe is not None:
                sel = cfg["selection"]
                sidecar = str(PurePosixPath(sel).with_suffix(".json"))
                if sidecar not in have or sel not in have or fullrun.FROZEN_MANIFEST not in have:
                    problems.append(f"{data_repo}: the full selection needs {sel}, its sidecar {sidecar} and "
                                    f"{fullrun.FROZEN_MANIFEST}")
                else:
                    shas = {}
                    for info in api.get_paths_info(data_repo, [sel, fullrun.FROZEN_MANIFEST], repo_type="dataset",
                                                   revision=data_rev):
                        shas[info.path] = _lfs_sha256(info)
                    for f in (sel, fullrun.FROZEN_MANIFEST):
                        if shas.get(f) is None:  # a small file in git, not LFS/Xet: hash it
                            local = download(data_repo, f, repo_type="dataset", revision=data_rev, local_dir=tmp)
                            shas[f] = hashlib.sha256(Path(local).read_bytes()).hexdigest()
                    side = json.loads(Path(download(data_repo, sidecar, repo_type="dataset", revision=data_rev,
                                                    local_dir=tmp)).read_text(encoding="utf-8"))
                    try:
                        # import_module, not `from kitsune import devslice`: that form returns the package attribute
                        # once an earlier import in the process set it, whatever sys.modules says now
                        devslice = importlib.import_module("kitsune.devslice")
                    except ImportError as e:
                        problems.append(f"cannot check {sidecar}: kitsune.devslice is not in this checkout ({e})")
                    else:
                        found = devslice.sidecar_problems(side, selection_sha256=shas[sel],
                                                          manifest_sha256=shas[fullrun.FROZEN_MANIFEST])
                        problems += [f"{data_repo}: {sidecar}: {x}" for x in found]
                        if not found:
                            notes.append(f"{sidecar}: the sidecar of {sel} (sha256 {shas[sel][:12]}...)")
    except Exception as e:  # noqa: BLE001
        problems.append(f"cannot check box {box}'s data in {data_repo}: {type(e).__name__}: {e}")
    if scratch_repo:
        try:
            api, _ = _hub()
            private = api.model_info(scratch_repo).private
        except Exception as e:  # noqa: BLE001
            problems.append(f"cannot read the scratch repo {scratch_repo} ({type(e).__name__}: {e}): the owner creates "
                            f"it on huggingface.co as a private model repo and gives the vast HF_TOKEN write access "
                            f"(vast/README.md, full-data runs)")
        else:
            if private is not True:
                problems.append(f"{scratch_repo} is not private: the timed states are the trainers' full states; "
                                f"hf repos settings {scratch_repo} --private")
    if resume:
        problems_r, notes_r = resume_preflight(out_repo, scratch_repo, box, resets, sets or {})
        problems += problems_r
        notes += notes_r
    return problems, notes


def resume_preflight(out_repo: str, scratch_repo: str | None, box: str, resets, sets: dict) -> tuple[list, list]:
    """--resume: the box's Hub queue summary (full/box-<box>/queue_summary.json, the source of truth) must exist, and
    every reset/set id must be fullrun.run_id_of(items[x].run_dir) of one of its train items; the notes list each
    train item (its status, run and the newest full state step in the scratch pointer and the runs repo)."""
    import tempfile

    problems, notes = [], []
    path = fullrun.box_summary_path(box)
    try:
        api, download = _hub()
        if not api.file_exists(out_repo, path):
            return [f"{out_repo} has no {path}: box {box} never put its queue summary up, so there is nothing to resume "
                    f"(launch it without --resume)"], notes
        with tempfile.TemporaryDirectory(prefix="kitsune-launch-") as tmp:
            summary = json.loads(Path(download(out_repo, path, local_dir=tmp)).read_text(encoding="utf-8"))
            items = summary.get("items") or {}
            ran = {fullrun.run_id_of(it["run_dir"]): name for name, it in items.items()
                   if isinstance(it, dict) and it.get("kind") == "train" and it.get("run_dir")}
            for rid in [*resets, *sets]:
                if rid not in ran:
                    problems.append(f"--resume-reset/--resume-set {rid}: no train item of box {box} ran it (its Hub "
                                    f"summary's runs: {sorted(ran) or 'none'})")
            for name, it in items.items():
                if not isinstance(it, dict) or it.get("kind") != "train":
                    continue
                rid = fullrun.run_id_of(it["run_dir"]) if it.get("run_dir") else None
                what = ("reset" if rid in resets else "") + (f" sets {sets[rid]}" if rid in sets else "")
                notes.append(f"resume: {name}: {it.get('status')}, run {rid or 'none yet (starts fresh)'}"
                             + (f", {what.strip()}" if what else "")
                             + (f"; newest state: {_state_steps(api, download, out_repo, scratch_repo, rid, tmp)}"
                                if rid else ""))
    except Exception as e:  # noqa: BLE001
        problems.append(f"cannot read {path} in {out_repo}: {type(e).__name__}: {e}")
    return problems, notes


def chain_preflight(data_repo: str, data_rev: str | None, sha: str, box: str, reg: dict, *,
                    reader=None) -> tuple[list[str], list[str]]:
    """-> (problems, notes) of a chain box (contract addendum E), on top of each part's full_preflight: every config
    of the chain (its parts', both stages' rebuild configs, the registry) committed at `sha`, and every file and dir of
    its stage views (fullrun.box_extra_files / box_extra_dirs, stage None: the union, incl. the selection files of a
    part whose data config is not its stage's rebuild: smoke-b's study_1000h.parquet and sidecar) in the data repo."""
    problems, notes = [], []
    reader = reader or (lambda rel: json.loads(git_show(sha, rel).decode("utf-8")))
    for c in fullrun.box_configs(box, reg):
        if not _at_sha(sha, c):
            problems.append(f"{c} does not exist at {sha[:12]}: python tools/make_full_configs.py, commit and push")
    try:
        files = fullrun.box_extra_files(box, reg, read_json=reader)
        dirs = fullrun.box_extra_dirs(box, reg, read_json=reader)
        api, _ = _hub()
        have = api.list_repo_files(data_repo, repo_type="dataset", revision=data_rev)
        problems += [f"{data_repo}: no {f} (chain {box} pulls it in a stage)" for f in files if f not in set(have)]
        problems += [f"{data_repo}: no {d}/ (chain {box} pulls it in a stage)" for d in dirs
                     if not any(f.startswith(f"{d}/") for f in have)]
    except Exception as e:  # noqa: BLE001
        problems.append(f"cannot check chain {box}'s stage files in {data_repo}: {type(e).__name__}: {e}")
    for st in fullrun.chain_stages(box, reg):
        notes.append(f"chain {box} stage {st['stage']}: parts {', '.join(st['parts'])}; rebuilds {st['rebuild']}"
                     + (f"; gate part {st['gate_box']} gates stage 2 (checks {fullrun.GATE_CHECKS[0]}-"
                        f"{fullrun.GATE_CHECKS[-1]}) by first boot + {st['gate_by_hours']:g} h; stage 1 ends by first "
                        f"boot + {st['max_hours']:g} h" if st["gate_box"] else ""))
    return problems, notes


def chain_resume_checks(out_repo: str, box: str, chain: str) -> tuple[list[str], list[str]]:
    """--job full --box <a chain's last-stage part> --resume, when the chain's summary is on the Hub (addendum E.8): the
    part's newest Hub summary must be the one the chain's rental wrote ("started on this rental": the chain summary's
    parts.<box>.status is not pending, the part summary has the chain's container_id, and its started is
    parts.<box>.queue_started) - else, when the chain summary is newer, the part summary is another rental's and the
    chain died before the part started: refused."""
    import tempfile

    problems, notes = [], []
    cpath, ppath = fullrun.box_summary_path(chain), fullrun.box_summary_path(box)
    try:
        api, download = _hub()
        if not api.file_exists(out_repo, cpath) or not api.file_exists(out_repo, ppath):
            return problems, notes
        with tempfile.TemporaryDirectory(prefix="kitsune-launch-") as tmp:
            cs = json.loads(Path(download(out_repo, cpath, local_dir=tmp)).read_text(encoding="utf-8"))
            ps = json.loads(Path(download(out_repo, ppath, local_dir=tmp)).read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [f"cannot read {cpath} / {ppath} in {out_repo}: {type(e).__name__}: {e}"], notes
    part = (cs.get("parts") or {}).get(box) or {}
    match = part.get("status") not in (None, "pending") and ps.get("container_id") == cs.get("container_id") \
        and ps.get("started") is not None and ps.get("started") == part.get("queue_started")
    if match:
        gate = cs.get("gate") or {}
        notes.append(f"--box {box} --resume continues stage 2 of chain {chain} (gate {gate.get('result')} "
                     f"{gate.get('time_utc')}, container {cs.get('container_id')})")
    elif float(cs.get("started") or 0) > float(ps.get("started") or 0):
        problems.append(f"the newest {box} summary on the Hub is from another rental (container "
                        f"{ps.get('container_id')}); chain {chain} on container {cs.get('container_id')} died before "
                        f"box {box} started: launch --box {box} fresh or the chain")
    return problems, notes


def _state_steps(api, download, out_repo: str, scratch_repo: str | None, rid: str, tmp: str) -> str:
    """The newest full state steps of run `rid` the laptop's login sees, best effort: the scratch pointer's step and
    the highest checkpoints/full_step_<N> in the runs repo (e.g. the pre_cooldown state)."""
    found = []
    if scratch_repo:
        try:
            ptr = json.loads(Path(download(scratch_repo, fullrun.scratch_pointer(rid), local_dir=tmp)).read_text(
                encoding="utf-8"))
            found.append(f"scratch step {ptr.get('step')}")
        except Exception:  # noqa: BLE001
            found.append("no scratch pointer")
    try:
        steps = [int(m.group(1)) for x in api.list_repo_tree(out_repo, path_in_repo=f"runs/{rid}/checkpoints")
                 if (m := re.fullmatch(r"full_step_(\d+)", str(getattr(x, "path", "")).rsplit("/", 1)[-1]))]
        found.append(f"runs repo full_step_{max(steps)}" if steps else "no full state in the runs repo")
    except Exception:  # noqa: BLE001
        found.append("runs repo not listed")
    return ", ".join(found)


def live_instances(exe: str, prefix: str) -> list[str]:
    """Labels of this account's instances that start with prefix (vastai show instances), best effort: two boxes of
    one full box would write the same run dirs."""
    try:
        rows = parse_json(vastai(exe, ["show", "instances", "--raw"]))
    except LaunchError:
        return []
    rows = rows.get("instances", []) if isinstance(rows, dict) else rows
    return [f"{r.get('label')} (instance {r.get('id')}, {r.get('actual_status') or r.get('cur_state') or '?'})"
            for r in rows if isinstance(r, dict) and str(r.get("label") or "").startswith(prefix)]


def sanitize_tag(branch: str) -> str:
    """The branch tag docker/metadata-action writes: '/' and other invalid characters become '-'."""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", branch)[:128]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", choices=sorted([*JOBS, "study", "full"]), default="train",
                    help="train: the A100 run (default); label: the RTX 5090 label box (vast/label.py); study: a "
                         "size-study box (--box; kitsune/study_queue.py); full: a full-data box (--box; "
                         "kitsune/full_queue.py, the registry configs/full/boxes.json)")
    ap.add_argument("--box", choices=[*STUDY_BOXES, *fullrun.ALL_BOX_NAMES], default=None,
                    help="study: A (the Cohere runs, 4 GPUs), B (Parakeet + bridge, 4 GPUs), replicate (1 GPU, after "
                         "A) or shakedown (1 GPU, first); full: full-smoke (smoke A), p01 (box 1), full (box 2), "
                         "smoke-b, or the chain p01-chain (smoke A + smoke B, then box 1 on one rental)")
    ap.add_argument("--data-repo", required=True, help="private HF dataset with the derived data (KITSUNE_DATA_REPO)")
    ap.add_argument("--out-repo", default=None, help="private HF model repo for runs/ (KITSUNE_OUT_REPO; train only)")
    ap.add_argument("--config", default=None,
                    help="run config, relative to the repo root (default: configs/viability.json; label: "
                         "configs/full.json; full: the box's data_config, the only one it takes)")
    ap.add_argument("--sha", default=None, help="commit to run (default: HEAD); must be pushed to GitHub")
    ap.add_argument("--image", default=None, help=f"full image ref with digest (default: resolve {IMAGE_REPO}:<tag>)")
    ap.add_argument("--image-tag", default=None, help="tag to resolve (default: the current branch, as CI tags it)")
    ap.add_argument("--offer-id", type=int, default=None, help="rent this offer from the search results")
    ap.add_argument("--max-dph", type=float, default=None,
                    help=f"refuse offers above this $/h (default {JOBS['train'].max_dph:g}; label "
                         f"{JOBS['label'].max_dph:g}; full: the registry's max_dph, {A100_MAX_DPH:g} with --tier a100)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="watchdog cap from first boot (KITSUNE_MAX_HOURS; default 5.5, plus the rebuild timeout for "
                         "an extent config; label 30; full: the registry's max_hours, no rebuild add-on)")
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
    ap.add_argument("--machine", default=None, metavar="ID",
                    help="rent on this vast machine_id only (a client-side filter on the search; a blocklisted or "
                         "avoided machine is refused)")
    ap.add_argument("--gpus", type=int, default=None,
                    help="full: the box's GPU count; the registry's gpus is the default and the only value taken")
    ap.add_argument("--scratch-repo", default=None,
                    help="full: the private HF model repo of the trainers' timed states (KITSUNE_SCRATCH_REPO); "
                         "required for a box with timed_states, never the output or the data repo")
    ap.add_argument("--resume", action="store_true",
                    help="full: relaunch a box on a new host: bootstrap pulls its runs from the Hub (the box's queue "
                         "summary) and the queue resumes them (KITSUNE_RESUME=1)")
    ap.add_argument("--resume-reset", action="append", default=[], metavar="RUN_ID",
                    help="full: continue this early-stopped run from its pre_cooldown state (repeatable; implies "
                         "--resume; KITSUNE_RESUME_RESET)")
    ap.add_argument("--resume-set", action="append", default=[], metavar="RUN_ID:schedule.epochs=E",
                    help="full: resume this run with another schedule.epochs (repeatable; implies --resume; "
                         "KITSUNE_RESUME_SETS)")
    ap.add_argument("--tier", choices=sorted(FULL_TIERS), default=None,
                    help=f"full: 5090 (default) or a100 (option C; default cap {A100_MAX_DPH:g} $/h)")
    ap.add_argument("--gate-hours", type=float, default=None,
                    help="full: the download gate's ceiling: the host must pull the reference 571.2 GB in this many "
                         "hours (default 5; 0 turns the gate off; a box whose registry says gate false has none)")
    ap.add_argument("--no-self-stop", action="store_true",
                    help="debugging: the box does not stop itself when its on-start or bootstrap fails "
                         "(KITSUNE_NO_SELF_STOP=1); the watchdog still stops it once onstart.sh has started it")
    ap.add_argument("--no-hf-check", action="store_true", help="skip the read-only HF preflight")
    ap.add_argument("--skip-git-checks", action="store_true",
                    help="rent even if the commit looks unpushed/dirty or the image was built from other dependencies")
    ap.add_argument("--dry-run", action="store_true", help="never create, even with --yes")
    ap.add_argument("--yes", action="store_true", help="actually create the instance (this spends money)")
    args = ap.parse_args(argv)
    label_job, study, full = args.job == "label", args.job == "study", args.job == "full"
    if study and not args.box:
        ap.error("--job study needs --box (A, B, replicate or shakedown)")
    if full and not args.box:
        ap.error(f"--job full needs --box ({', '.join(fullrun.ALL_BOX_NAMES)})")
    if args.box and not (study and args.box in STUDY_BOXES or full and args.box in fullrun.ALL_BOX_NAMES):
        ap.error(f"--box {args.box} is not a box of --job {args.job} (study: {', '.join(STUDY_BOXES)}; full: "
                 f"{', '.join(fullrun.ALL_BOX_NAMES)})")
    full_only = [f for f, v in (("--gpus", args.gpus is not None), ("--scratch-repo", args.scratch_repo),
                                ("--resume", args.resume), ("--resume-reset", args.resume_reset),
                                ("--resume-set", args.resume_set), ("--tier", args.tier),
                                ("--gate-hours", args.gate_hours is not None)) if v]
    if full_only and not full:
        ap.error(f"{', '.join(full_only)}: for --job full only")
    if args.machine is not None and not re.fullmatch(r"\d+", args.machine):
        ap.error(f"--machine takes a vast machine_id (digits), not {args.machine!r}")
    if args.gate_hours is not None and args.gate_hours < 0:
        ap.error("--gate-hours must be >= 0 (0 turns the gate off)")
    try:
        resets = fullrun.parse_resume_reset(",".join(args.resume_reset))
        sets = fullrun.parse_resume_sets(",".join(args.resume_set))
    except ValueError as e:
        ap.error(str(e))
    resume = args.resume or bool(resets or sets)
    if not label_job and not args.out_repo:
        ap.error("the following arguments are required: --out-repo")
    if args.cohere_procs is not None and args.cohere_procs < 1:
        ap.error("--cohere-procs must be >= 1")
    will_create = args.yes and not args.dry_run
    errors: list[str] = []
    notes: list[str] = []

    sha = args.sha or git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise LaunchError(f"--sha must be a full 40-hex commit id, got {sha!r}")

    reg = reader = spec = chain = None  # chain: a chain box's stages (contract addendum E), else None
    if full:  # the box registry at the commit the box runs: its GPUs, config, hours, price cap, disk and watchdog
        reg, reader, problems, reg_notes = full_registry(sha, skip_git_checks=args.skip_git_checks)
        errors += problems
        notes += reg_notes
        if reg is not None:
            try:
                spec = fullrun.box_spec(args.box, reg, read_json=reader)  # a chain's derived spec reads its parts
                if fullrun.is_chain(args.box, reg):
                    chain = fullrun.chain_stages(args.box, reg)
            except fullrun.RegistryError as e:
                errors.append(f"box {args.box}: {e} ({fullrun.BOXES_FILE} at {sha[:12]})")
        if spec is None:
            for n in notes:
                print(n)
            print("\nproblems:\n  " + "\n  ".join(errors) + "\nnot creating anything: box " + args.box
                  + " has no usable registry entry")
            return 1
        if chain is not None and resume:  # addendum E.8: a chain is not resumed as a chain
            print(f"\nproblems:\n  --resume/--resume-reset/--resume-set refused for chain box {args.box}\n"
                  + fullrun.chain_resume_hint(args.box, chain[-1]["parts"][-1]) + "\nnot creating anything")
            return 1
        if args.gpus is not None and args.gpus != spec["gpus"]:
            errors.append(f"box {args.box} is planned for {spec['gpus']} GPU(s) ({fullrun.BOXES_FILE}); --gpus "
                          f"{args.gpus} refused")
        if args.config is not None and chain is not None:
            errors.append(f"--config is refused for chain box {args.box}: stage 1 rebuilds {chain[0]['rebuild']}, its "
                          f"controller's stage-2 bootstrap {chain[-1]['rebuild']} ({fullrun.BOXES_FILE})")
        elif args.config is not None and args.config != spec["data_config"]:
            errors.append(f"box {args.box}'s data config is {spec['data_config']} ({fullrun.BOXES_FILE}): --config "
                          f"{args.config} refused (its items' configs share that data block)")
        if spec["timed_states"] and not args.scratch_repo:
            errors.append(f"box {args.box} keeps timed full states: --scratch-repo <the private scratch model repo> is "
                          f"required (vast/README.md, full-data runs)")
        if args.scratch_repo and args.scratch_repo in (args.out_repo, args.data_repo):
            errors.append(f"--scratch-repo {args.scratch_repo} is the output or the data repo: the trainers delete and "
                          f"squash the scratch repo's history, so it must be a repo of its own")
        if args.scratch_repo and not spec["timed_states"]:
            notes.append(f"box {args.box} keeps no timed states: --scratch-repo is not passed to it")
        tier = args.tier or "5090"
        max_hours = args.max_hours if args.max_hours is not None else float(spec["max_hours"])
        plan_hours = max_hours / 1.1 if args.max_hours is not None else float(spec["est_hours"])
        max_dph = args.max_dph if args.max_dph is not None else A100_MAX_DPH if tier == "a100" else spec["max_dph"]
        job = full_job(args.box, spec, tier, plan_hours, max_hours, max_dph)
        config = spec["data_config"]
        if chain is not None:
            # the boot bootstrap rebuilds stage 1's extent; the disk, the gate and the cap are the whole chain's. The
            # cap must hold stage 1 and the last stage's planned hours (E.6: 35 h covers the worst case the gate lets
            # pass)
            config = chain[0]["rebuild"]
            floor = float(chain[0]["max_hours"]) + sum(float(reg["boxes"][p]["est_hours"]) for p in chain[-1]["parts"])
            if max_hours < floor:
                errors.append(f"--max-hours {max_hours:g} is below chain {args.box}'s floor {floor:g} (stage 1's "
                              f"{chain[0]['max_hours']:g} h + the last stage's est_hours)")
            elif max_hours < float(spec["max_hours"]):
                notes.append(f"WARNING: --max-hours {max_hours:g} is below chain {args.box}'s {spec['max_hours']:g} h: "
                             f"a slow box 1 that the gate lets through may be compressed or stopped by the cap")
            if args.gate_hours == 0:
                errors.append(f"--gate-hours 0 is refused for chain box {args.box}: its gate part's check 4 reads the "
                              f"boot's download_gate.json")
    else:
        job = study_job(args.box) if study else JOBS[args.job]
        config = args.config or ("configs/full.json" if label_job else STUDY_CONFIG if study else "configs/viability.json")
        max_dph = args.max_dph if args.max_dph is not None else job.max_dph
    label_configs = list(dict.fromkeys([config] + [c for c in args.label_configs.split(",") if c])) \
        if label_job else []

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

    # the machines no launch rents: --avoid-machine, the committed blocklist (every job), then per job the Hub's
    avoid = {str(m): "--avoid-machine" for m in args.avoid_machine}
    for m, why in load_blocklist().items():
        if m not in avoid:
            avoid[m] = f"vast/blocklist.json: {why}"
            notes.append(f"avoiding machine {m}: {avoid[m]}")
    data_rev, sizing, boot_sizing = None, None, None  # boot_sizing: the boot bootstrap's (a chain: stage 1's extent)
    if args.no_hf_check:
        if label_job:
            errors.append("--job label needs the HF preflight (it pins KITSUNE_DATA_REVISION and checks the lease): "
                          "drop --no-hf-check")
        elif study:
            errors.append("--job study needs the HF preflight (the extent's sizing, the PREREG and selection hashes, "
                          "the box's students, the numbers files): drop --no-hf-check")
        elif full:
            errors.append("--job full needs the HF preflight (the extent's sizing, the selection, the box's students "
                          "and extra files, the scratch repo, the gate refusals): drop --no-hf-check")
        else:
            try:  # local git only: an extent config's sizing cannot be skipped with the Hub checks
                # --skip-git-checks: the sha may not be local; the working tree's config is the best guess
                c = json.loads((ROOT / config).read_text(encoding="utf-8")) if args.skip_git_checks \
                    else config_at(sha, config)
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
                for m in hub_avoid:
                    avoid.setdefault(m, "a label run ended there as host_failure/slow_host")
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
                if full and chain is not None:
                    # a chain: each part's own preflight (a smoke-b item whose CLI the sha lacks refuses here) and each
                    # distinct data config's selection at the same data revision, then the chain's own files
                    extra_gb = float(spec["extra_gb"])
                    checked = {config}
                    for part in [p for st in chain for p in st["parts"]]:
                        pspec = reg["boxes"][part]
                        try:
                            pcfg = cfg if pspec["data_config"] == config else config_at(sha, pspec["data_config"])
                        except (subprocess.CalledProcessError, OSError, ValueError) as e:
                            errors.append(f"part {part}: cannot read {pspec['data_config']} at {sha[:12]} "
                                          f"({type(e).__name__})")
                            continue
                        if pspec["data_config"] not in checked:
                            checked.add(pspec["data_config"])
                            rev_p, problems = hf_preflight(args.data_repo, args.out_repo, pcfg)
                            errors += [f"{pspec['data_config']}: {x}" for x in problems if x not in errors]
                            if rev_p and data_rev and rev_p != data_rev:
                                errors.append(f"{args.data_repo} moved from {data_rev[:12]} to {rev_p[:12]} during the "
                                              f"preflight: run it again")
                        problems, pre_notes = full_preflight(args.data_repo, data_rev, args.out_repo,
                                                             args.scratch_repo if pspec["timed_states"] else None, sha,
                                                             part, reg, pcfg, reader=reader)
                        errors += [f"part {part}: {x}" for x in problems]
                        notes += [f"part {part}: {x}" for x in pre_notes]
                    problems, pre_notes = chain_preflight(args.data_repo, data_rev, sha, args.box, reg, reader=reader)
                    errors += problems
                    notes += pre_notes
                elif full:
                    extra_gb = float(spec["extra_gb"])
                    problems, pre_notes = full_preflight(args.data_repo, data_rev, args.out_repo,
                                                         args.scratch_repo if spec["timed_states"] else None, sha,
                                                         args.box, reg, cfg, reader=reader, resume=resume,
                                                         resets=resets, sets=sets)
                    errors += problems
                    notes += pre_notes
                    for c in (fullrun.CHAIN_NAMES if resume else ()):  # E.8: a resume of a chain's last part
                        if c in reg["boxes"] and args.box in fullrun.chain_stages(c, reg)[-1]["parts"]:
                            problems, pre_notes = chain_resume_checks(args.out_repo, args.box, c)
                            errors += problems
                            notes += pre_notes
                if full:
                    hub_avoid, avoid_notes = avoided_machines(args.data_repo, data_rev)
                    for m in hub_avoid:
                        avoid.setdefault(m, "a label run ended there as host_failure/slow_host")
                    gates, gate_notes = gate_refusals(args.out_repo)
                    notes += avoid_notes + gate_notes
                    for m, why in gates.items():
                        if m not in avoid:
                            avoid[m] = why
                            notes.append(f"avoiding machine {m}: {why}")
                if chain is not None:
                    # the disk and the gate: the last stage's extent (box 1's; its extra_gb holds stage 1's leftovers);
                    # the boot bootstrap's rebuild and pull bytes and its rebuild timeout: stage 1's own extent
                    try:
                        cfg_last = config_at(sha, chain[-1]["rebuild"])
                    except (subprocess.CalledProcessError, OSError, ValueError) as e:
                        errors.append(f"cannot read {chain[-1]['rebuild']} at {sha[:12]} ({type(e).__name__})")
                    else:
                        problems, sizing = extent_preflight(args.data_repo, data_rev, cfg_last, extra_gb=extra_gb)
                        errors += problems
                    problems, boot_sizing = extent_preflight(args.data_repo, data_rev, cfg, extra_gb=0.0)
                    errors += [x for x in problems if x not in errors]
                elif cfg.get("extent"):
                    problems, sizing = extent_preflight(args.data_repo, data_rev, cfg, extra_gb=extra_gb)
                    errors += problems
                    boot_sizing = sizing
                elif study or full:
                    errors.append(f"{config} has no extent: a {args.job} box rebuilds the sealed label extent")
    if args.machine is not None and args.machine in avoid:
        errors.append(f"--machine {args.machine} is avoided ({avoid[args.machine]}): pick another machine")
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
    if full:  # the registry's cap is the whole box's: no rebuild add-on
        if sizing:
            job = replace(job, est_down_gb=sizing["down_gb"] + sizing["labels_gb"])
    else:
        max_hours = args.max_hours if args.max_hours is not None else \
            job.max_hours + (sizing["rebuild_timeout_min"] / 60 if sizing else 0)
    if sizing:
        stores = f" x {sizing['stores']} stores" if sizing.get("stores", 1) > 1 else ""
        extra = f", checkpoints ~{sizing['extra_gb']:.0f} GB" if sizing.get("extra_gb") else ""
        notes.append(f"extent sizing: ~{sizing['down_gb']:.0f} GB upstream down, shards ~{sizing['shard_gb']:.0f} GB, "
                     f"selected audio ~{sizing['sel_gb']:.0f} GB{stores}, labels ~{sizing['labels_gb']:.1f} GB"
                     f"{extra} -> disk {sizing['disk_gb']} GB, rebuild timeout {sizing['rebuild_timeout_min']} min, "
                     f"max hours {max_hours:g}")
    if chain is not None and boot_sizing:
        notes.append(f"chain stage 1 ({chain[0]['rebuild']}): ~{boot_sizing['down_gb']:.0f} GB upstream down, labels "
                     f"~{boot_sizing['labels_gb']:.1f} GB, rebuild timeout {boot_sizing['rebuild_timeout_min']} min; "
                     f"the disk, the gate and the cap above are the whole chain's (stage 2 rebuilds the rest of "
                     f"{chain[-1]['rebuild']})")
    gate_h = None
    if full:
        gate_h = args.gate_hours if args.gate_hours is not None else DEFAULT_GATE_H
        if not spec["gate"]:
            gate_h = None
            notes.append(f"box {args.box} has no download gate (registry gate false)")
        elif gate_h == 0:
            gate_h = None
            notes.append("WARNING: the download gate is OFF (--gate-hours 0): a slow Hub link is found only by the "
                         "rebuild's timeouts, after hours of billing")
        elif sizing is None:
            gate_h = None
    for n in notes:
        print(n)

    exe = shutil.which("vastai")
    if exe is None:
        print("\nproblems:\n  " + "\n  ".join(errors) if errors else "git, image and HF checks passed")
        print(install_help())
        return 2
    if full and resume:  # two boxes of one full box would write the same run dirs (its labels: the one below)
        prefixes = [f"{job.label_prefix}-{Path(config).stem}-"]
        # a chain whose last stage runs this box writes the same run dirs (addendum E.8): its labels too
        prefixes += [f"kitsune-full-{c}-" for c in fullrun.CHAIN_NAMES
                     if c in reg["boxes"] and args.box in fullrun.chain_stages(c, reg)[-1]["parts"]]
        for live in [x for p in prefixes for x in live_instances(exe, p)]:
            print(f"WARNING: a live instance of box {args.box}: {live}: destroy it (it keeps writing the runs this "
                  f"resume pulls) before renting")

    if label_job:
        env = {"KITSUNE_JOB": "label", "KITSUNE_SHA": sha, "KITSUNE_CONFIG": config,
               "KITSUNE_LABEL_CONFIGS": ",".join(label_configs), "KITSUNE_DATA_REPO": args.data_repo}
    elif study:
        # KITSUNE_N_GPUS: the queue refuses to run on another GPU count than the box was rented with
        env = {"KITSUNE_JOB": "study", "KITSUNE_BOX": args.box, "KITSUNE_N_GPUS": str(STUDY_GPUS[args.box]),
               "KITSUNE_SHA": sha, "KITSUNE_CONFIG": config, "KITSUNE_DATA_REPO": args.data_repo,
               "KITSUNE_OUT_REPO": args.out_repo}
    elif full:
        # the registry's part (KITSUNE_N_GPUS and the watchdog's train_hb rule) comes from kitsune.fullrun.box_env
        env = {"KITSUNE_JOB": fullrun.JOB, "KITSUNE_BOX": args.box, "KITSUNE_SHA": sha, "KITSUNE_CONFIG": config,
               "KITSUNE_DATA_REPO": args.data_repo, "KITSUNE_OUT_REPO": args.out_repo,
               **fullrun.box_env(args.box, reg)}
        if spec["timed_states"] and args.scratch_repo:
            env[fullrun.ENV_SCRATCH_REPO] = args.scratch_repo
        if resume:
            env[fullrun.ENV_RESUME] = "1"
            if resets:
                env[fullrun.ENV_RESUME_RESET] = ",".join(resets)
            if sets:
                env[fullrun.ENV_RESUME_SETS] = ",".join(f"{rid}:{kv}" for rid, kvs in sets.items() for kv in kvs)
        if gate_h is not None:  # kitsune.netgate on the box, before anything is pulled
            # a chain: the gate protects box 1's download (the last stage's), the timeouts are stage 1's rebuild's
            boot = boot_sizing or sizing
            env[fullrun.ENV_GATE_BYTES] = str(int(1e9 * max(sizing["down_gb"], GATE_REF_GB)))
            env[fullrun.ENV_GATE_MAX_H] = f"{gate_h:g}"
            env[fullrun.ENV_REBUILD_BYTES] = str(int(1e9 * boot["down_gb"]))
            env[fullrun.ENV_PULL_BYTES] = str(int(1e9 * (boot["labels_gb"] + 2)))
    else:
        env = {"KITSUNE_SHA": sha, "KITSUNE_CONFIG": config, "KITSUNE_DATA_REPO": args.data_repo,
               "KITSUNE_OUT_REPO": args.out_repo}
    env["KITSUNE_MAX_HOURS"] = f"{max_hours:g}"
    if not label_job:
        env["TZ"] = "UTC"
    if data_rev:
        env["KITSUNE_DATA_REVISION"] = data_rev
    if boot_sizing or sizing:  # the boot bootstrap's rebuild (a chain: stage 1's)
        env["KITSUNE_REBUILD_TIMEOUT_MIN"] = str((boot_sizing or sizing)["rebuild_timeout_min"])
    if label_job:
        env.update(LABEL_WATCHDOG_ENV)
        env["KITSUNE_IMAGE"] = image
        if args.steal_lease:
            env["KITSUNE_STEAL_LEASE"] = "1"
        if args.cohere_procs is not None:
            env["KITSUNE_COHERE_PROCS"] = str(args.cohere_procs)
    if args.no_self_stop:  # inside the one --env value: vastai has no -e option, and a second --env replaces the first
        env["KITSUNE_NO_SELF_STOP"] = "1"

    tier, query, offers = search_offers(exe, job, disk_gb, avoid, max_dph if label_job or full else None,
                                        args.machine)
    print(f"\nsearch ({tier or 'nothing found'}): vastai "
          f"{shlex.join(search_args(query or job_query(job, job.tiers[0][1], disk_gb), disk_gb))}")
    if not offers:
        if args.machine is not None:
            errors.append(f"machine {args.machine} has no offer passing the filter now (rented, or its ask changed): "
                          f"drop --machine to rank other machines")
        else:
            errors.append("no RTX 5090 offer matches the label filter (see the hint above); try again later" if label_job
                          else f"no {STUDY_GPUS[args.box]}x A100 offer matches the study filter; try again later"
                          if study else f"no {job.n_gpus}x {job.tiers[0][0]} offer matches the full-box filter (see "
                          f"above); try again later, or --tier a100 (option C)" if full
                          else "no offer matches the strict filter (D45a); try again later or relax it by hand")
        offer = None
    else:
        print(offer_table(offers, job=job if label_job or full else None))
        offer = next((o for o in offers if o.get("id") == args.offer_id), None) if args.offer_id else offers[0]
        if offer is None:
            errors.append(f"offer {args.offer_id} is not in the results")
        elif offer.get("dph_total", 0) > max_dph:
            errors.append(f"offer {offer['id']} costs ${offer['dph_total']:.3f}/h > --max-dph {max_dph}")

    if offer and isinstance(offer.get("dph_total"), (int, float)):
        env["KITSUNE_DPH"] = f"{offer['dph_total']:.4f}"  # kitsune.runlog records it for the cost estimate
    if label_job or full:
        if offer and offer.get("machine_id") not in (None, ""):
            # label_end.json names it for the avoid list; a full box's download gate record and queue summary too
            env["KITSUNE_MACHINE_ID"] = str(offer["machine_id"])
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
    elif offer and full:
        dph = offer.get("dph_total", 0)
        down, up = _gb_cost(offer, "inet_down_cost"), _gb_cost(offer, "inet_up_cost")
        traffic = job.est_down_gb * down + job.est_up_gb * up
        print(f"expected cost: ~${dph:.3f}/h ({job.n_gpus}x {tier} + {disk_gb} GB) x ~{job.est_hours:.3g} h (box "
              f"{args.box}; watchdog cap {max_hours:g} h) + ~{job.est_down_gb:.0f} GB down x ${down:.3f}/GB + "
              f"~{job.est_up_gb:.0f} GB up x ${up:.3f}/GB = ~${est_total(offer, job):.2f} "
              f"(cap = ~${dph * max_hours + traffic:.2f})")
        if chain is not None:  # addendum E.6: the chain's own deadlines, from first boot
            by = chain[0]["gate_by_hours"]
            print(f"chain {args.box}: the gate part must end by first boot + {by:g} h (the watchdog stops a stage 1 "
                  f"that has not handed over by + {by + 0.5:g} h), stage 1 ends by + {chain[0]['max_hours']:g} h, the "
                  f"cap is + {max_hours:g} h; a failed gate destroys the box after its records are verified on the "
                  f"Hub, a passed one runs box 1 on the same machine")
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
