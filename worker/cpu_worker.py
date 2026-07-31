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
    # Clear and recreate the output directory so a prior failed run on the same
    # node does not contaminate results.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _clear_directory_contents(OUTPUT_DIR)

    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    job_id = os.getenv("JOB_ID", "").strip() or "unknown"
    try:
        job_id = require_env("JOB_ID")
        cpu_threads = require_positive_int("CPU_THREADS", "2")
        exhaustiveness = require_positive_int("EXHAUSTIVENESS", "16")
        num_modes = require_positive_int("NUM_MODES", "10")
        seed = require_positive_int("SEED", "42")
    except SystemExit as exc:
        _write_result(
            job_id=job_id, start=start, exit_code=2, command=[], params={},
            best_affinity=None, error=str(exc),
        )
        print(f"[cpu_worker] preflight failed: {exc}")
        return

    receptor = os.path.join(INPUT_DIR, "receptor.pdbqt")
    ligand = os.path.join(INPUT_DIR, "ligand.pdbqt")
    box_config = os.path.join(INPUT_DIR, "box.txt")
    for path in (receptor, ligand, box_config):
        if not os.path.isfile(path):
            error = f"required input not found: {path}"
            _write_result(
                job_id=job_id, start=start, exit_code=2, command=[],
                params={
                    "cpu_threads": cpu_threads,
                    "exhaustiveness": exhaustiveness,
                    "num_modes": num_modes,
                    "seed": seed,
                },
                best_affinity=None, error=error,
            )
            print(f"[cpu_worker] preflight failed: {error}")
            return

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

    print(f"[cpu_worker] job_id={job_id}  cmd={' '.join(cmd)}")

    combined_output = []
    try:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True) as proc:
            for line in proc.stdout:
                print(line, end="")
                combined_output.append(line)
            proc.wait()
        exit_code = proc.returncode
    except OSError as exc:
        # smina missing or not executable.  Record it like any other failure so
        # the operator sees the cause in R2 instead of an unexplained silence.
        msg = f"could not execute {SMINA_BIN}: {exc}"
        print(f"[cpu_worker] {msg}")
        combined_output.append(msg + "\n")
        exit_code = 127

    with open(stdout_log, "w") as fh:
        fh.writelines(combined_output)

    best_affinity = _parse_best_affinity(poses_out) if exit_code == 0 else None
    error = None
    if exit_code == 0 and best_affinity is None:
        exit_code = 65
        error = "smina exited zero but no parseable minimizedAffinity was written"
    elif exit_code != 0:
        error = f"smina exited {exit_code}"

    _write_result(
        job_id=job_id,
        start=start,
        exit_code=exit_code,
        command=cmd,
        params={
            "cpu_threads": cpu_threads,
            "exhaustiveness": exhaustiveness,
            "num_modes": num_modes,
            "seed": seed,
        },
        best_affinity=best_affinity,
        error=error,
    )

    # Exit zero even when smina failed.  Kelpie runs sync.after only on a
    # zero exit, so exiting non-zero here would leave result.json stranded on
    # the node: poll.py would see no artifact, report the job as "pending"
    # forever, and --watch would never terminate.  The failure is not hidden —
    # it is recorded in result.json's exit_code, which is what poll.py reads.
    #
    # Nor does this lose retries where they matter: a preempted node kills the
    # process rather than returning an exit status, so Kelpie still requeues
    # those.  What it stops is the pointless re-running of a job that fails
    # deterministically on malformed input.
    if exit_code != 0:
        print(f"[cpu_worker] smina exited {exit_code} — recorded in result.json")


def _write_result(job_id: str, start: str, exit_code: int, command: list,
                  params: dict, best_affinity: float | None,
                  error: str | None) -> None:
    result = {
        "job_id": job_id,
        "start": start,
        "finish": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "exit_code": exit_code,
        "command": command,
        "params": params,
        "best_affinity_kcal_mol": best_affinity,
    }
    if error:
        result["error"] = error
    with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")


def _clear_directory_contents(path: str) -> None:
    """Clear a worker directory without deleting a possible mount point."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.path)
        else:
            os.unlink(entry.path)


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
