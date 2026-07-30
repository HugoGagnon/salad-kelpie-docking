#!/usr/bin/env python3
"""
submit.py — build Kelpie job definitions and POST them to the queue.

Reads a JSON manifest and submits one Kelpie job per entry.  Credentials are
passed via environment variables (see config/.env.example).  The Kelpie API
key is fed over stdin to curl so it never appears in process arguments or logs.

Manifest schema differs by mode:

  --mode cpu   config/jobs.json  — one entry per docking job
  --mode gpu   config/matrix.json — one entry per (complex, replica) pair

Run prefix is REQUIRED and has no default.  Reusing a prefix deliberately
extends an existing run; change it whenever the image, inputs, or protocol
change.

Usage:
  python submit.py --mode cpu --manifest config/jobs.json --run-prefix 2024-01-campaign-v1
  python submit.py --mode gpu --manifest config/matrix.json --run-prefix 2024-01-md-v1 --prod-ns 10
  python submit.py --mode cpu --manifest config/jobs.json --run-prefix test-v1 --dry-run
"""
import argparse
import json
import os
import re
import subprocess
import sys

KELPIE_API_URL = os.getenv("KELPIE_API_URL", "https://kelpie.saladexamples.com")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# ── shared helpers ─────────────────────────────────────────────────────────────

def clean_prefix(value: str) -> str:
    value = value.strip().strip("/")
    if not value or ".." in value.split("/"):
        raise ValueError("must be a non-empty bucket-relative path without '..'")
    return value


def post_job(payload: dict, api_key: str, organization: str, project: str) -> tuple[int, str]:
    """POST one job definition to the Kelpie API via curl.

    curl is used instead of Python urllib because Cloudflare rejects Python's
    default user-agent on the Kelpie endpoint.  The API key is passed on stdin
    so it never appears in curl's process arguments or in shell history.
    """
    headers = (
        "Content-Type: application/json\n"
        f"Salad-Api-Key: {api_key}\n"
        f"Salad-Organization: {organization}\n"
        f"Salad-Project: {project}\n"
    )
    result = subprocess.run(
        [
            "curl", "--silent", "--show-error", "--fail-with-body",
            "--request", "POST", f"{KELPIE_API_URL}/jobs",
            "--header", "@-",
            "--data-binary", json.dumps(payload),
            "--write-out", "\n%{http_code}",
        ],
        input=headers, text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError((result.stderr + "\n" + result.stdout).strip())
    body, status_str = result.stdout.rsplit("\n", 1)
    return int(status_str), json.loads(body).get("id", "")


# ── CPU docking mode ───────────────────────────────────────────────────────────

def cpu_job_def(spec: dict, container_group_id: str, bucket: str, run_prefix: str) -> dict:
    """Build one Kelpie job definition for a CPU smina docking job.

    Input files must already exist in R2 at:
      <run_prefix>/inputs/<job_id>/{receptor.pdbqt,ligand.pdbqt,box.txt}

    Kelpie downloads them before the worker starts and uploads /app/outputs/
    afterward.
    """
    job_id = str(spec["id"])
    if not SAFE_ID.fullmatch(job_id):
        raise ValueError(f"invalid job id {job_id!r}")
    input_prefix = clean_prefix(spec.get("input_prefix", f"{run_prefix}/inputs/{job_id}"))
    output_prefix = f"{run_prefix}/outputs/{job_id}"
    return {
        "container_group_id": container_group_id,
        "command": "python3",
        "arguments": ["/app/worker/cpu_worker.py"],
        "environment": {
            "JOB_ID": job_id,
            # Right-size these down from the maximum available on a node.
            # Resource requests filter the pool of eligible nodes — asking for
            # more CPU threads means fewer machines match and allocation stalls.
            "CPU_THREADS": str(spec.get("cpu_threads", 2)),
            "EXHAUSTIVENESS": str(spec.get("exhaustiveness", 16)),
            "NUM_MODES": str(spec.get("num_modes", 10)),
            "SEED": str(spec.get("seed", 42)),
        },
        "sync": {
            "before": [{"bucket": bucket, "prefix": input_prefix + "/",
                        "local_path": "/app/input/", "direction": "download"}],
            "after": [{"bucket": bucket, "prefix": output_prefix + "/",
                       "local_path": "/app/outputs/", "direction": "upload"}],
        },
    }


def submit_cpu(args, bucket: str, cgid: str) -> None:
    run_prefix = clean_prefix(args.run_prefix)
    with open(args.manifest) as fh:
        specs = json.load(fh)["jobs"]
    jobs = [cpu_job_def(s, cgid, bucket, run_prefix) for s in specs]
    if args.max:
        jobs = jobs[: args.max]
    print(f"{len(jobs)} CPU docking job(s) | prefix: {run_prefix}/")
    if args.dry_run:
        print(json.dumps(jobs[0], indent=2))
        return
    _require_env(["SALAD_API_KEY", "SALAD_ORGANIZATION", "SALAD_PROJECT", "R2_BUCKET"])
    if not cgid:
        sys.exit("CPU_CONTAINER_GROUP_ID not set")
    _post_all(jobs, args)


# ── GPU MD mode ────────────────────────────────────────────────────────────────

def gpu_job_def(
    cx: dict, replica: int, prod_ns: float,
    container_group_id: str, bucket: str, run_prefix: str,
    base_seed: int = 42,
) -> dict:
    """Build one Kelpie job definition for a GPU MD + MM-GBSA job.

    Input data (receptor.pdb, ligand.sdf) must be baked into the container
    image at /app/data/<name>/.  Kelpie downloads the checkpoint prefix before
    the job and uploads it continuously during (sync.during) so the node can be
    preempted safely and resumed from the latest checkpoint on the next
    allocation.
    """
    name = cx["name"]
    target = cx["target"]
    ligand = cx["ligand"]
    root = f"{run_prefix}/{name}/rep{replica}"
    ckpt_prefix = f"{root}/checkpoints/"
    out_prefix = f"{root}/outputs/"
    # Per-replica seed: vary by replica index so trajectories diverge.
    seed = base_seed + replica
    return {
        "container_group_id": container_group_id,
        "command": "python",
        "arguments": [
            "worker/gpu_worker.py",
            target, ligand, str(replica), str(prod_ns),
            "checkpoints", "outputs", "/app/data",
        ],
        "environment": {
            "SYSTEM": target,
            "LIGAND_NAME": ligand,
            "REPLICA": str(replica),
            "TARGET_NS": str(prod_ns),
            "SEED": str(seed),
            "CKPT_DIR": "checkpoints",
            "OUT_DIR": "outputs",
            "DATA_DIR": "/app/data",
        },
        "sync": {
            "before": [{"bucket": bucket, "prefix": ckpt_prefix,
                        "local_path": "checkpoints/", "direction": "download"}],
            # Continuously upload the checkpoint directory while the job runs.
            # This is what makes GPU MD safely interruptible: the binary
            # checkpoint and progress JSON are durable in R2 even if the node
            # is preempted mid-trajectory.
            "during": [{"bucket": bucket, "prefix": ckpt_prefix,
                        "local_path": "checkpoints/", "direction": "upload"}],
            "after": [{"bucket": bucket, "prefix": out_prefix,
                       "local_path": "outputs/", "direction": "upload"}],
        },
    }


def submit_gpu(args, bucket: str, cgid: str) -> None:
    run_prefix = clean_prefix(args.run_prefix)
    with open(args.manifest) as fh:
        matrix = json.load(fh)
    complexes = matrix["complexes"]
    if args.targets:
        keep = {t.strip() for t in args.targets.split(",")}
        complexes = [c for c in complexes if c["target"] in keep]
        print(f"target filter {sorted(keep)} -> {len(complexes)} complex(es)")
    jobs = []
    for cx in complexes:
        for rep in range(args.n_reps):
            jobs.append(gpu_job_def(cx, rep, args.prod_ns, cgid, bucket, run_prefix))
    if args.max:
        jobs = jobs[: args.max]
    total_ns = len(jobs) * args.prod_ns
    print(f"{len(complexes)} complex(es) × {args.n_reps} rep(s) = {len(jobs)} GPU MD job(s)")
    print(f"prefix: {run_prefix}/ | total: {total_ns:.0f} ns")
    if args.dry_run:
        print(json.dumps(jobs[0], indent=2))
        return
    _require_env(["SALAD_API_KEY", "SALAD_ORGANIZATION", "SALAD_PROJECT", "R2_BUCKET"])
    if not cgid:
        sys.exit("GPU_CONTAINER_GROUP_ID not set")
    _post_all(jobs, args)


# ── submission loop ────────────────────────────────────────────────────────────

def _require_env(keys: list[str]) -> None:
    missing = [k for k in keys if not os.getenv(k)]
    if missing:
        sys.exit("missing environment variables: " + ", ".join(missing))


def _post_all(jobs: list[dict], args) -> None:
    api_key = os.environ["SALAD_API_KEY"]
    org = os.environ["SALAD_ORGANIZATION"]
    project = os.environ["SALAD_PROJECT"]
    submitted = []
    for job in jobs:
        label = (job.get("environment", {}).get("JOB_ID")
                 or f"{job['environment'].get('SYSTEM')} rep{job['environment'].get('REPLICA')}")
        try:
            status, kelpie_id = post_job(job, api_key, org, project)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            sys.exit(f"FAILED to submit {label}: {exc}")
        submitted.append(kelpie_id)
        print(f"  {label}: HTTP {status} | id={kelpie_id}")
    ids_path = "submitted_job_ids.txt"
    with open(ids_path, "w") as fh:
        fh.write("\n".join(x for x in submitted if x))
    print(f"\n{len(submitted)} job(s) submitted — IDs saved to {ids_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and submit Kelpie job definitions to a Salad container group."
    )
    parser.add_argument("--mode", choices=["cpu", "gpu"], required=True,
                        help="cpu = smina docking; gpu = OpenMM MD + MM-GBSA")
    parser.add_argument("--manifest", required=True,
                        help="job manifest JSON (jobs.json for cpu, matrix.json for gpu)")
    parser.add_argument("--run-prefix", required=True,
                        help="R2 campaign prefix — required, never defaulted. "
                             "Reuse only to extend the same run.")
    parser.add_argument("--container-group-id",
                        default=None,
                        help="Salad container group UUID (overrides CPU/GPU_CONTAINER_GROUP_ID)")
    parser.add_argument("--max", type=int, default=None,
                        help="submit only the first N jobs (smoke test)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build job definitions but do not POST")
    # GPU-only
    parser.add_argument("--prod-ns", type=float, default=10.0,
                        help="[gpu] MD production length in nanoseconds (default: 10)")
    parser.add_argument("--n-reps", type=int, default=3,
                        help="[gpu] replicas per complex (default: 3)")
    parser.add_argument("--targets", default=None,
                        help="[gpu] comma-separated target filter")
    args = parser.parse_args()

    bucket = os.getenv("R2_BUCKET", "R2_BUCKET")
    if args.mode == "cpu":
        cgid = (args.container_group_id
                or os.getenv("CPU_CONTAINER_GROUP_ID")
                or os.getenv("CONTAINER_GROUP_ID", ""))
        submit_cpu(args, bucket, cgid)
    else:
        cgid = (args.container_group_id
                or os.getenv("GPU_CONTAINER_GROUP_ID")
                or os.getenv("CONTAINER_GROUP_ID", ""))
        submit_gpu(args, bucket, cgid)


if __name__ == "__main__":
    main()
