#!/usr/bin/env python3
"""
cpu_worker.py — smina docking worker, invoked by Kelpie on each job.

Kelpie has already downloaded the input files to /app/input/ before this script
starts.  Expected inputs:
  /app/input/receptor.pdbqt
  /app/input/ligand.pdbqt
  /app/input/box.txt          # smina --config format

Writes to /app/outputs/:
  poses.pdbqt
  smina.log
  worker_stdout.log
  result.json

Kelpie uploads /app/outputs/ to R2 after the script exits zero.
"""
import datetime
import json
import os
import shutil
import subprocess
import sys

INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/outputs"
SMINA_BIN = os.getenv("SMINA_BIN", "smina")


def require_env(name: str) -> str:
    val = os.getenv(name, "").strip()
    if not val:
        sys.exit(f"required environment variable not set: {name}")
    return val


def require_positive_int(name: str, default: str) -> int:
    raw = os.getenv(name, default)
    try:
        n = int(raw)
        if n <= 0:
            raise ValueError
        return n
    except ValueError:
        sys.exit(f"{name} must be a positive integer, got: {raw!r}")


def main() -> None:
    job_id = require_env("JOB_ID")
    cpu_threads = require_positive_int("CPU_THREADS", "2")
    exhaustiveness = require_positive_int("EXHAUSTIVENESS", "16")
    num_modes = require_positive_int("NUM_MODES", "10")
    seed = require_positive_int("SEED", "42")

    # Clear and recreate the output directory so a prior failed run on the same
    # node does not contaminate results.
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    receptor = os.path.join(INPUT_DIR, "receptor.pdbqt")
    ligand = os.path.join(INPUT_DIR, "ligand.pdbqt")
    box_config = os.path.join(INPUT_DIR, "box.txt")
    for path in (receptor, ligand, box_config):
        if not os.path.isfile(path):
            sys.exit(f"required input not found: {path}")

    poses_out = os.path.join(OUTPUT_DIR, "poses.pdbqt")
    smina_log = os.path.join(OUTPUT_DIR, "smina.log")
    stdout_log = os.path.join(OUTPUT_DIR, "worker_stdout.log")

    cmd = [
        SMINA_BIN,
        "--receptor", receptor,
        "--ligand", ligand,
        "--config", box_config,
        "--out", poses_out,
        "--log", smina_log,
        "--num_modes", str(num_modes),
        "--exhaustiveness", str(exhaustiveness),
        "--cpu", str(cpu_threads),
        "--seed", str(seed),
    ]

    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"[cpu_worker] job_id={job_id}  cmd={' '.join(cmd)}")

    combined_output = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True) as proc:
        for line in proc.stdout:
            print(line, end="")
            combined_output.append(line)
        proc.wait()

    finish = datetime.datetime.now(datetime.timezone.utc).isoformat()
    exit_code = proc.returncode

    with open(stdout_log, "w") as fh:
        fh.writelines(combined_output)

    best_affinity = _parse_best_affinity(poses_out) if exit_code == 0 else None

    result = {
        "job_id": job_id,
        "start": start,
        "finish": finish,
        "exit_code": exit_code,
        "command": cmd,
        "params": {
            "cpu_threads": cpu_threads,
            "exhaustiveness": exhaustiveness,
            "num_modes": num_modes,
            "seed": seed,
        },
        "best_affinity_kcal_mol": best_affinity,
    }
    with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)

    if exit_code != 0:
        sys.exit(f"smina exited with code {exit_code}")


def _parse_best_affinity(poses_path: str) -> float | None:
    if not os.path.isfile(poses_path):
        return None
    with open(poses_path) as fh:
        for line in fh:
            if "minimizedAffinity" in line:
                try:
                    return float(line.split()[2])
                except (IndexError, ValueError):
                    pass
    return None


if __name__ == "__main__":
    main()
