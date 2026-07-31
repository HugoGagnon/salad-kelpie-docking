#!/usr/bin/env python3
"""
poll.py — poll object storage for job completion markers.

The queue's job status is a proxy and can be wrong in both directions.
Completion is defined as the existence of result.json in R2 for every job.
This script is that principle made concrete.

For CPU docking jobs:
  A job is complete when <run_prefix>/outputs/<job_id>/result.json exists.

For GPU MD jobs:
  A job is complete when <run_prefix>/<name>/rep<N>/outputs/<name>/rep<N>_mmgbsa.json exists.

Usage:
  python poll.py --mode cpu --manifest config/jobs.json --run-prefix 2024-01-campaign-v1
  python poll.py --mode gpu --manifest config/matrix.json --run-prefix 2024-01-md-v1 --n-reps 3
  python poll.py --mode cpu --manifest config/jobs.json --run-prefix test-v1 --watch --interval 60
"""
import argparse
import json
import os
import sys
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


def make_client():
    """Create an S3-compatible boto3 client for the configured R2 endpoint.

    max_attempts=5 gives automatic retries on transient network errors.
    Use the account-level endpoint exactly as issued; do not append the bucket name.
    """
    for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL"):
        if not os.getenv(key):
            sys.exit(f"missing environment variable: {key}")
    return boto3.client(
        "s3",
        endpoint_url=os.environ["AWS_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name=os.getenv("AWS_REGION", "auto"),
        config=Config(retries={"max_attempts": 5}),
    )


def object_exists(client, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return False
        raise


def fetch_result(client, bucket: str, key: str) -> dict | None:
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body)
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


# ── CPU docking polling ────────────────────────────────────────────────────────

def poll_cpu(manifest_path: str, run_prefix: str, bucket: str, client) -> dict:
    with open(manifest_path) as fh:
        specs = json.load(fh)["jobs"]
    results = {}
    for spec in specs:
        job_id = str(spec["id"])
        key = f"{run_prefix}/outputs/{job_id}/result.json"
        result = fetch_result(client, bucket, key)
        if result is None:
            results[job_id] = {"status": "pending"}
        elif result.get("exit_code") == 0:
            results[job_id] = {"status": "ok", "affinity_kcal_mol": result.get("best_affinity_kcal_mol")}
        else:
            results[job_id] = {"status": "failed",
                               "exit_code": result.get("exit_code")}
    return results


# ── GPU MD polling ─────────────────────────────────────────────────────────────

def poll_gpu(manifest_path: str, run_prefix: str, bucket: str, client, n_reps: int) -> dict:
    with open(manifest_path) as fh:
        matrix = json.load(fh)
    results = {}
    for cx in matrix["complexes"]:
        name = cx["name"]
        for rep in range(n_reps):
            label = f"{name}/rep{rep}"
            key = f"{run_prefix}/{name}/rep{rep}/outputs/{name}/rep{rep}_mmgbsa.json"
            result = fetch_result(client, bucket, key)
            if result is None:
                # Check whether a progress marker exists (job is running but not scored yet).
                ckpt_key = f"{run_prefix}/{name}/rep{rep}/checkpoints/progress.json"
                progress = fetch_result(client, bucket, ckpt_key)
                if progress:
                    pct = (progress.get("ns", 0) / max(progress.get("target_ns", 1), 1)) * 100
                    results[label] = {"status": "running", "progress_pct": round(pct, 1)}
                else:
                    results[label] = {"status": "pending"}
            elif result.get("error"):
                # gpu_worker.py records deterministic failures as an artifact
                # rather than a non-zero exit, so they are visible here instead
                # of looking like a job that never started.
                results[label] = {"status": "failed",
                                  "error": result["error"],
                                  "stage": result.get("stage")}
            else:
                results[label] = {
                    "status": "ok",
                    "dG_mean_kcal_mol": result.get("dG_mean"),
                    "dG_sem_kcal_mol": result.get("dG_sem"),
                }
    return results


# ── reporting ─────────────────────────────────────────────────────────────────

def report(results: dict) -> None:
    total = len(results)
    done = sum(1 for r in results.values() if r["status"] == "ok")
    failed = sum(1 for r in results.values() if r["status"] == "failed")
    running = sum(1 for r in results.values() if r["status"] == "running")
    pending = total - done - failed - running
    print(f"\n{'─'*60}")
    print(f"  Total: {total} | Done: {done} | Running: {running} | "
          f"Pending: {pending} | Failed: {failed}")
    print(f"{'─'*60}")
    for label, r in results.items():
        status = r["status"]
        if status == "ok":
            val = r.get("affinity_kcal_mol") or r.get("dG_mean_kcal_mol")
            suffix = f"  {val:.2f} kcal/mol" if val is not None else ""
            print(f"  ✓ {label}{suffix}")
        elif status == "running":
            print(f"  … {label}  ({r.get('progress_pct', '?')}%)")
        elif status == "failed":
            if r.get("error"):
                stage = f" [{r['stage']}]" if r.get("stage") else ""
                print(f"  ✗ {label}{stage}  {r['error']}")
            else:
                print(f"  ✗ {label}  exit={r.get('exit_code')}")
        else:
            print(f"  ○ {label}  pending")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Poll R2 object storage for job completion. "
                    "Completion is defined by the presence of result artifacts, "
                    "not by the queue's job status."
    )
    parser.add_argument("--mode", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--watch", action="store_true",
                        help="keep polling until all jobs are done or failed")
    parser.add_argument("--interval", type=int, default=60,
                        help="seconds between polls in --watch mode (default: 60)")
    parser.add_argument("--n-reps", type=int, default=3,
                        help="[gpu] replicas per complex (default: 3)")
    args = parser.parse_args()

    bucket = os.getenv("R2_BUCKET")
    if not bucket:
        sys.exit("R2_BUCKET not set")

    client = make_client()

    while True:
        if args.mode == "cpu":
            results = poll_cpu(args.manifest, args.run_prefix, bucket, client)
        else:
            results = poll_gpu(args.manifest, args.run_prefix, bucket, client, args.n_reps)

        report(results)

        if not args.watch:
            break

        terminal = {"ok", "failed"}
        if all(r["status"] in terminal for r in results.values()):
            print("\nAll jobs reached a terminal state.")
            break

        print(f"\n  next poll in {args.interval}s — Ctrl-C to stop")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            break


if __name__ == "__main__":
    main()
