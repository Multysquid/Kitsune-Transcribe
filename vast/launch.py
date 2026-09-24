"""Rent one A100 on vast.ai for the viability run: search offers, show them, and create the instance only with --yes.

Run this on the laptop. It
  1. searches on-demand offers with the strict host filter (D45a): 1x A100 SXM4 40 GB, verified, reliability >= 0.98,
     driver CUDA >= 13.0 (the image's torch is cu130), >= 12 effective CPU cores, >= 64 GB RAM, disk_bw and inet_down
     >= 500, a direct port and room for the 150 GB disk, sorted by $/h; if none, the same filter on SXM4 80 GB;
  2. prints the offers and the exact `vastai create instance` command: the image by digest, --disk 150, --ssh --direct,
     the KITSUNE_* env, TZ=UTC and --onstart vast/onstart_stub.sh (which clones the repo and runs vast/onstart.sh);
  3. creates the instance only with --yes. Spending money is always the user's explicit step.
The git, image and HF checks run before the offer search, so they report problems even without the vastai CLI. The HF
check refuses a data repo whose derived data is incomplete: a train source's teacher shard without its second opinion,
or a selection that still drops rows as no_agree (both mean training on a fraction of the planned hours).

HF_TOKEN is never an argument and never part of the command: the box gets it from the vast ACCOUNT-level environment
variables (D48a), so it does not appear in shell history, the process list or the instance config. The instance runs
the code at a pinned commit (KITSUNE_SHA), so this script refuses to rent for a commit GitHub does not have.

Usage:
  python vast/launch.py --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs        # look only
  python vast/launch.py --data-repo Multy123/kitsune-data --out-repo Multy123/kitsune-runs --yes  # rent the cheapest
Needs the vastai CLI (`pip install vastai==1.8.0`, then `vastai set api-key <key>`); --help works without it.
"""
import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VASTAI_PIN = "vastai==1.8.0"
IMAGE_REPO = "ghcr.io/multysquid/kitsune-train"
# the image (~12 GB unpacked, if vast counts it) + audio 23 GB + train/eval caches ~22 GB + derived data 2.3 GB + up to
# 4 full states x 8.6 GB while rotating/uploading + ~9 weights x 1.2 GB: 80 GB fills mid-run, 150 GB leaves headroom
DISK_GB = 150
# traffic of one run for the cost line: ~25 GB of upstream audio down, ~30 GB of checkpoints and logs up
EST_DOWN_GB, EST_UP_GB = 25, 30
ONSTART = Path(__file__).resolve().parent / "onstart_stub.sh"
ONSTART_MAX_BYTES = 4000  # the API's onstart field: vast documents 16 KB, one client SDK <= 4048 chars
DEFAULT_MAX_DPH = 2.0
# D45a strict filter; vast converts cpu_ram/gpu_ram GB to MB itself. rentable/verified are also CLI defaults, spelled
# out so the printed query is the whole truth.
HOST_FILTER = [
    "num_gpus=1", "verified=true", "rentable=true", "reliability>=0.98", "cuda_vers>=13.0",
    "cpu_cores_effective>=12", "cpu_ram>=64", "disk_bw>=500", "inet_down>=500", "direct_port_count>=1",
    f"disk_space>={DISK_GB}",
]
# vast names both SXM4 sizes "A100 SXM4"; gpu_ram tells them apart (40 GB cards report 39-40 GB)
TIERS = [
    ("A100 SXM4 40 GB", ["gpu_name in [A100_SXM4]", "gpu_ram<=48"]),
    ("A100 SXM4 80 GB (fallback)", ["gpu_name in [A100_SXM4]", "gpu_ram>=70"]),
]
SECRET_RE = re.compile(r"TOKEN|SECRET|PASSWORD|API_KEY", re.I)


class LaunchError(RuntimeError):
    pass


def install_help() -> str:
    return (f"The vastai CLI is not on PATH. Install it and store your API key (you type the key yourself):\n"
            f"  pip install {VASTAI_PIN}\n"
            f"  vastai set api-key <your key from https://cloud.vast.ai/manage-keys/>\n"
            f"then re-run this script.")


def build_query(tier_terms: list[str]) -> str:
    return " ".join([*tier_terms, *HOST_FILTER])


def search_args(query: str) -> list[str]:
    return ["search", "offers", query, "--type", "on-demand", "-o", "dph", "--storage", str(DISK_GB), "--raw"]


def env_string(env: dict[str, str]) -> str:
    """vast's --env takes docker-style options; values must be single shell words."""
    for k, v in env.items():
        if SECRET_RE.search(k):
            raise LaunchError(f"refusing to put {k} on the command line; secrets belong in the vast account env")
        if not re.fullmatch(r"[A-Za-z0-9_./:@+,=-]+", v):
            raise LaunchError(f"env value for {k} must be one word without quotes/spaces: {v!r}")
    return " ".join(f"-e {k}={v}" for k, v in env.items())


def create_args(offer_id: int | str, image: str, env: dict[str, str], onstart: Path, label: str) -> list[str]:
    return ["create", "instance", str(offer_id), "--image", image, "--disk", str(DISK_GB), "--ssh", "--direct",
            "--env", env_string(env), "--onstart", str(onstart), "--label", label, "--cancel-unavail", "--raw"]


def parse_json(text: str):
    """The CLI may print warnings before the JSON; take the first JSON value."""
    starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if not starts:
        raise LaunchError(f"no JSON in vastai output: {text[:300]!r}")
    return json.JSONDecoder().raw_decode(text[min(starts):])[0]


def vastai(exe: str, args: list[str]) -> str:
    r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        raise LaunchError(f"vastai {' '.join(args[:2])} failed ({r.returncode}): {(r.stderr or r.stdout).strip()[:500]}")
    return r.stdout


def search_offers(exe: str) -> tuple[str, str, list[dict]]:
    """-> (tier name, query, offers sorted by $/h) for the first tier with any offer."""
    for name, terms in TIERS:
        query = build_query(terms)
        offers = parse_json(vastai(exe, search_args(query)))
        if isinstance(offers, dict):
            offers = offers.get("offers", [])
        if offers:
            return name, query, sorted(offers, key=lambda o: o.get("dph_total", 1e9))
        print(f"no offers for {name}: {query}")
    return "", "", []


def offer_table(offers: list[dict], limit: int = 10) -> str:
    cols = [("id", "id", "{}"), ("gpu", "gpu_name", "{}"), ("GB", "gpu_ram", "{:.0f}"), ("$/h", "dph_total", "{:.3f}"),
            ("rel", "reliability", "{:.3f}"), ("cuda", "cuda_max_good", "{}"), ("cpu", "cpu_cores_effective", "{:.0f}"),
            ("ramGB", "cpu_ram", "{:.0f}"), ("down", "inet_down", "{:.0f}"), ("diskMB/s", "disk_bw", "{:.0f}"),
            ("$/GBdn", "inet_down_cost", "{:.3f}"), ("$/GBup", "inet_up_cost", "{:.3f}"), ("where", "geolocation", "{}")]
    rows = [[h for h, _, _ in cols]]
    for o in offers[:limit]:
        row = []
        for _, key, fmt in cols:
            v = o.get(key)
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


def resolve_image_digest(tag: str, timeout: float = 20) -> str:
    """Tag -> immutable `repo@sha256:...` via the anonymous GHCR registry API (the package must be public, D39a)."""
    name = IMAGE_REPO.split("/", 1)[1]
    with urllib.request.urlopen(f"https://ghcr.io/token?scope=repository:{name}:pull&service=ghcr.io", timeout=timeout) as r:
        token = json.load(r)["token"]
    accept = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.oci.image.manifest.v1+json",
                        "application/vnd.docker.distribution.manifest.list.v2+json",
                        "application/vnd.docker.distribution.manifest.v2+json"])
    req = urllib.request.Request(f"https://ghcr.io/v2/{name}/manifests/{tag}", method="HEAD",
                                 headers={"Authorization": f"Bearer {token}", "Accept": accept})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        digest = r.headers.get("Docker-Content-Digest")
    if not digest or not digest.startswith("sha256:"):
        raise LaunchError(f"GHCR returned no digest for {IMAGE_REPO}:{tag}")
    return f"{IMAGE_REPO}@{digest}"


def data_problems(files: list[str], cfg: dict) -> list[str]:
    """What the run config needs from the data repo's file list (bootstrap.sh's plan() applies the same rules on the
    box, where a failure is already billed): the teacher/second-opinion meta, the selection, the student config, teacher
    shards for every train source and eval set, and a second opinion for EVERY train source teacher shard.
    make_selection drops the rows of a shard without one as no_agree, so a partial 02b pass silently shrinks the train
    set."""
    teacher_root, second_root = cfg.get("teacher_root", "teacher_out"), cfg.get("second_root", "second_out")
    have = set(files)
    problems = [f"no {f}" for f in (f"{teacher_root}/meta.json", f"{second_root}/meta.json", cfg["selection"],
                                    f"{cfg['student'].rstrip('/')}/config.json") if f not in have]

    def stems(root: str, s: str, ext: str) -> set[str]:
        return {f.rsplit("/", 1)[1][: -len(ext)] for f in files if f.startswith(f"{root}/{s}/") and f.endswith(ext)}

    sources = list(cfg.get("sources", []))
    for s in dict.fromkeys(sources + list(cfg.get("eval_sets", []))):
        teacher = stems(teacher_root, s, ".npz")
        if not teacher:
            problems.append(f"no {teacher_root}/{s}/*.npz")
        elif s in sources and (gap := teacher - stems(second_root, s, ".jsonl")):
            problems.append(f"{second_root}/{s}: {len(gap)} of {len(teacher)} teacher shards have no second opinion "
                            f"(e.g. {min(gap)}): finish scripts/02b_second_opinion.py, rebuild the selection, upload")
    return problems


def selection_problems(path: Path, name: str) -> list[str]:
    """A selection built before the second-opinion pass finished drops those rows as no_agree."""
    import pandas as pd

    sel = pd.read_parquet(path, columns=["source", "reason"])
    bad = sel[sel["reason"] == "no_agree"]
    if bad.empty:
        return []
    return [f"{name} drops {len(bad)} rows as no_agree ({bad['source'].value_counts().to_dict()}): rebuild it with "
            f"scripts/make_selection.py after the second-opinion pass and upload it"]


def hf_preflight(data_repo: str, out_repo: str, cfg: dict) -> tuple[str | None, list[str]]:
    """-> (data repo commit to pin, problems). Uses the laptop's own HF login, read-only (the selection, ~2 MB, is
    downloaded to a temporary dir)."""
    import tempfile

    from huggingface_hub import HfApi, hf_hub_download

    api, problems, rev = HfApi(), [], None
    try:
        info = api.dataset_info(data_repo)
        rev = info.sha
        files = api.list_repo_files(data_repo, repo_type="dataset", revision=rev)
        problems += [f"{data_repo}@{rev[:12]}: {p}" for p in data_problems(files, cfg)]
        if cfg["selection"] in files:
            with tempfile.TemporaryDirectory(prefix="kitsune-launch-") as tmp:
                local = hf_hub_download(data_repo, cfg["selection"], repo_type="dataset", revision=rev, local_dir=tmp)
                problems += selection_problems(Path(local), cfg["selection"])
    except Exception as e:
        problems.append(f"cannot read dataset {data_repo}: {type(e).__name__}: {e}")
    try:
        api.model_info(out_repo)
    except Exception as e:
        problems.append(f"cannot read output model repo {out_repo} ({type(e).__name__}); create it first: "
                        f"hf repos create {out_repo} --private")
    return rev, problems


def sanitize_tag(branch: str) -> str:
    """The branch tag docker/metadata-action writes: '/' and other invalid characters become '-'."""
    return re.sub(r"[^A-Za-z0-9_.-]", "-", branch)[:128]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-repo", required=True, help="private HF dataset with the derived data (KITSUNE_DATA_REPO)")
    ap.add_argument("--out-repo", required=True, help="private HF model repo for runs/ (KITSUNE_OUT_REPO)")
    ap.add_argument("--config", default="configs/viability.json", help="run config, relative to the repo root")
    ap.add_argument("--sha", default=None, help="commit to run (default: HEAD); must be pushed to GitHub")
    ap.add_argument("--image", default=None, help=f"full image ref with digest (default: resolve {IMAGE_REPO}:<tag>)")
    ap.add_argument("--image-tag", default=None, help="tag to resolve (default: the current branch, as CI tags it)")
    ap.add_argument("--offer-id", type=int, default=None, help="rent this offer from the search results")
    ap.add_argument("--max-dph", type=float, default=DEFAULT_MAX_DPH, help="refuse offers above this $/h")
    ap.add_argument("--max-hours", type=float, default=5.5, help="watchdog cap from first boot (KITSUNE_MAX_HOURS)")
    ap.add_argument("--no-hf-check", action="store_true", help="skip the read-only HF preflight")
    ap.add_argument("--skip-git-checks", action="store_true", help="rent even if the commit looks unpushed/dirty")
    ap.add_argument("--dry-run", action="store_true", help="never create, even with --yes")
    ap.add_argument("--yes", action="store_true", help="actually create the instance (this spends money)")
    args = ap.parse_args(argv)
    will_create = args.yes and not args.dry_run
    errors: list[str] = []

    sha = args.sha or git("rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise LaunchError(f"--sha must be a full 40-hex commit id, got {sha!r}")
    if not args.skip_git_checks:
        errors += git_checks(sha, args.config)

    image = args.image
    if image is None:
        tag = args.image_tag or sanitize_tag(git("rev-parse", "--abbrev-ref", "HEAD"))
        try:
            image = resolve_image_digest(tag)
        except (OSError, urllib.error.URLError, LaunchError, KeyError, ValueError) as e:
            errors.append(f"cannot resolve {IMAGE_REPO}:{tag} to a digest ({e}); is the CI build done and the "
                          f"package public? Or pass --image {IMAGE_REPO}@sha256:...")
            image = f"{IMAGE_REPO}@sha256:<unresolved>"
    elif "@sha256:" not in image:
        errors.append(f"--image must be pinned by digest (...@sha256:...), got {image}")

    data_rev = None
    if not args.no_hf_check:
        cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
        data_rev, problems = hf_preflight(args.data_repo, args.out_repo, cfg)
        errors += problems
    stub = ONSTART.read_bytes()
    if len(stub) > ONSTART_MAX_BYTES:
        errors.append(f"{ONSTART.name} is {len(stub)} bytes (> {ONSTART_MAX_BYTES}): vast may truncate the on-start "
                      f"field, and a truncated script starts nothing, not even the watchdog")

    exe = shutil.which("vastai")
    if exe is None:
        print("\nproblems:\n  " + "\n  ".join(errors) if errors else "git, image and HF checks passed")
        print(install_help())
        return 2

    env = {"KITSUNE_SHA": sha, "KITSUNE_CONFIG": args.config, "KITSUNE_DATA_REPO": args.data_repo,
           "KITSUNE_OUT_REPO": args.out_repo, "KITSUNE_MAX_HOURS": f"{args.max_hours:g}", "TZ": "UTC"}
    if data_rev:
        env["KITSUNE_DATA_REVISION"] = data_rev

    tier, query, offers = search_offers(exe)
    print(f"\nsearch ({tier or 'nothing found'}): vastai {shlex.join(search_args(query or build_query(TIERS[0][1])))}")
    if not offers:
        errors.append("no offer matches the strict filter (D45a); try again later or relax it by hand")
        offer = None
    else:
        print(offer_table(offers))
        offer = next((o for o in offers if o.get("id") == args.offer_id), None) if args.offer_id else offers[0]
        if offer is None:
            errors.append(f"offer {args.offer_id} is not in the results")
        elif offer.get("dph_total", 0) > args.max_dph:
            errors.append(f"offer {offer['id']} costs ${offer['dph_total']:.3f}/h > --max-dph {args.max_dph}")

    if offer and isinstance(offer.get("dph_total"), (int, float)):
        env["KITSUNE_DPH"] = f"{offer['dph_total']:.4f}"  # kitsune.runlog records it for the cost estimate
    label = f"kitsune-{Path(args.config).stem}-{sha[:7]}"
    cargs = create_args(offer["id"] if offer else "<offer-id>", image, env, ONSTART, label)
    assert not any("HF_TOKEN" in a for a in cargs), "HF_TOKEN must never be on the command line"
    print(f"\ncreate command:\n  vastai {shlex.join(cargs)}")
    print("HF_TOKEN is not passed: it must be set in vast Account -> Settings -> Environment Variables.")
    if offer:
        dph = offer.get("dph_total", 0)
        down, up = (offer.get(k) if isinstance(offer.get(k), (int, float)) else None
                    for k in ("inet_down_cost", "inet_up_cost"))
        bw = (f"~${EST_DOWN_GB * down + EST_UP_GB * up:.2f} (~{EST_DOWN_GB} GB down, ~{EST_UP_GB} GB up at this host's "
              f"$/GB)" if down is not None and up is not None else "unknown (the offer lists no $/GB)")
        print(f"expected cost: ${dph:.3f}/h (GPU + {DISK_GB} GB disk) x <= {args.max_hours:g} h (watchdog cap) "
              f"= <= ${dph * args.max_hours:.2f}, plus bandwidth {bw}")

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
