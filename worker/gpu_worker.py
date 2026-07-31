#!/usr/bin/env python3
"""
gpu_worker.py — GPU MD + MM-GBSA orchestrator, invoked by Kelpie on each job.

Kelpie has already downloaded the checkpoint directory from R2 (sync.before)
and will continuously upload it while this script runs (sync.during).  On a
fresh job the checkpoint directory is empty; on a resumed job it contains the
binary checkpoint, DCD trajectory, and progress.json from the previous
allocation.

Inputs baked into the container image at /app/data/<name>/:
  receptor.pdb
  ligand.sdf

Writes to checkpoints/ (continuously uploaded by Kelpie during sync.during):
  solvated.pdb, system.xml, system_metadata.json, trajectory.dcd,
  checkpoint.chk, progress.json

Writes to outputs/ (uploaded by Kelpie at sync.after):
  <name>/rep<N>_mmgbsa.json
  <name>/rep<N>_progress.json

If the MD run is incomplete (progress.json shows complete=false), the worker
exits non-zero so Kelpie retries the job.  The next allocation restores the
checkpoint via sync.before and the engine resumes from where it stopped.

Usage (invoked by Kelpie via job definition arguments):
  python worker/gpu_worker.py <target> <ligand> <replica> <prod_ns> <ckpt_dir> <out_dir> <data_dir>
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent


def main() -> None:
    if len(sys.argv) < 8:
        sys.exit(
            "usage: gpu_worker.py <target> <ligand> <replica> <prod_ns> "
            "<ckpt_dir> <out_dir> <data_dir>"
        )
    target, ligand, replica_s, prod_ns_s, ckpt_dir, out_dir, data_dir = sys.argv[1:8]
    try:
        replica = int(replica_s)
        prod_ns = float(prod_ns_s)
    except ValueError:
        sys.exit(f"replica must be an integer and prod_ns a float: {replica_s!r}, {prod_ns_s!r}")
    if replica < 0:
        sys.exit(f"replica must be zero or greater: {replica}")

    name = f"{target}__{ligand}"
    out_subdir = Path(out_dir) / name
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    try:
        seed = _env_int("SEED", 42 + replica, minimum=0)
        skip_ns = _env_float("SKIP_NS", 2.0, minimum=0.0)
        walltime_h = _env_float("WALLTIME_H", 20.0, minimum=0.000001)
        ckpt_ps = _env_int("CHECKPOINT_PS", 500, minimum=1)
        n_frames = _env_int("N_FRAMES", 50, minimum=1)
    except ValueError as exc:
        _record_failure(out_subdir, replica, stage="preflight", reason=str(exc))

    if not math.isfinite(prod_ns) or prod_ns <= 0:
        _record_failure(
            out_subdir, replica, stage="preflight",
            reason=f"trajectory length must be positive, got {prod_ns} ns",
        )

    # SKIP_NS is a container-group-wide constant but prod_ns is per job, so a
    # short run can ask the scorer to discard its whole trajectory.  Check that
    # here, in seconds, rather than discovering it after a full MD run.
    if skip_ns >= prod_ns:
        _record_failure(
            out_subdir, replica, stage="preflight",
            reason=(f"SKIP_NS ({skip_ns} ns) is >= the requested trajectory "
                    f"length ({prod_ns} ns); every frame would be discarded as "
                    f"equilibration and there would be nothing to score. Lower "
                    f"SKIP_NS on the container group or raise --prod-ns."),
        )

    if ckpt_ps > prod_ns * 1000:
        _record_failure(
            out_subdir, replica, stage="preflight",
            reason=(f"CHECKPOINT_PS ({ckpt_ps} ps) exceeds the requested trajectory "
                    f"length ({prod_ns * 1000:g} ps); lower CHECKPOINT_PS so at "
                    f"least one trajectory frame is written."),
        )

    try:
        receptor = _find_input(data_dir, name, "receptor.pdb")
        ligand_file = _find_input(data_dir, name, "ligand.sdf")
    except FileNotFoundError as exc:
        _record_failure(out_subdir, replica, stage="preflight", reason=str(exc))
    print(f"[gpu_worker] {name} rep{replica} | {prod_ns} ns | seed={seed}")

    # ── stage 1: MD run ───────────────────────────────────────────────────────
    engine = SCRIPTS_DIR / "md_engine.py"
    md_cmd = [
        sys.executable, str(engine),
        "--receptor", receptor,
        "--ligand", ligand_file,
        "--work-dir", ckpt_dir,
        "--target-ns", str(prod_ns),
        "--checkpoint-ps", str(ckpt_ps),
        "--walltime-h", str(walltime_h),
        "--seed", str(seed),
    ]
    ret = subprocess.run(md_cmd, check=False).returncode
    if ret != 0:
        _record_failure(
            out_subdir, replica, stage="md",
            reason=f"md_engine exited {ret}; inspect the node log before resubmitting",
        )

    # ── verify trajectory is complete ─────────────────────────────────────────
    progress_src = Path(ckpt_dir) / "progress.json"
    if not progress_src.exists():
        sys.exit("progress.json not found — MD did not write any output")
    try:
        with open(progress_src) as fh:
            progress = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        _record_failure(
            out_subdir, replica, stage="md",
            reason=f"could not read progress.json: {exc}",
        )
    if not progress.get("complete"):
        # The engine saved a checkpoint but did not reach the target duration.
        # Exit non-zero so Kelpie retries; sync.during has already pushed the
        # checkpoint to R2, so the next allocation resumes without re-running.
        print(f"[gpu_worker] trajectory incomplete ({progress.get('ns', 0):.1f} "
              f"/ {prod_ns} ns) — checkpoint synced, Kelpie will retry")
        sys.exit(1)

    # ── stage 2: MM-GBSA scoring ───────────────────────────────────────────────
    scorer = SCRIPTS_DIR / "mmgbsa.py"
    out_subdir.mkdir(parents=True, exist_ok=True)
    mmgbsa_json = out_subdir / f"rep{replica}_mmgbsa.json"

    score_cmd = [
        sys.executable, str(scorer),
        "--work-dir", ckpt_dir,
        "--output", str(mmgbsa_json),
        "--ligand", ligand_file,
        "--n-frames", str(n_frames),
        "--skip-ns", str(skip_ns),
    ]
    ret = subprocess.run(score_cmd, check=False).returncode
    if ret != 0:
        # The trajectory is finished and checkpointed; only scoring failed, and
        # it will fail the same way on every retry.  Exiting non-zero here would
        # skip Kelpie's sync.after so nothing reaches R2, leaving poll.py to
        # report "pending" while Kelpie re-allocates a GPU to repeat a job that
        # cannot succeed.  Record the failure instead and exit zero.
        _record_failure(
            out_subdir, replica, stage="mmgbsa",
            reason=f"mmgbsa scorer exited {ret}; the trajectory is complete and "
                   f"checkpointed, so re-running would fail identically. See the "
                   f"node log for the scorer's error.",
        )

    # ── copy progress JSON to outputs ─────────────────────────────────────────
    import shutil
    shutil.copy(progress_src, out_subdir / f"rep{replica}_progress.json")
    print(f"[gpu_worker] {name} rep{replica} complete")


def _record_failure(out_subdir: Path, replica: int, stage: str, reason: str) -> None:
    """Write a failure artifact to the output directory and exit zero.

    Deterministic failures must leave evidence in R2.  Kelpie runs sync.after
    only on a zero exit, so exiting non-zero would strand the explanation on a
    node that is about to be recycled, and Kelpie would retry a job whose
    outcome cannot change.  poll.py reads the "error" key and reports the job
    as failed, which is a terminal state, so --watch can finish.
    """
    out_subdir.mkdir(parents=True, exist_ok=True)
    payload = {"error": reason, "stage": stage, "replica": replica}
    with open(out_subdir / f"rep{replica}_mmgbsa.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[gpu_worker] FAILED ({stage}): {reason}")
    sys.exit(0)


def _env_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _env_float(name: str, default: float, minimum: float | None = None) -> float:
    raw = os.getenv(name, str(default))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {raw!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _find_input(data_dir: str, name: str, filename: str) -> str:
    path = Path(data_dir) / name / filename
    if path.is_file():
        return str(path)
    # Fallback: glob for the filename anywhere under data_dir/<name>/
    matches = list(Path(data_dir).glob(f"*{name}*/{filename}"))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(
        f"input not found: {filename} for complex {name} in {data_dir}"
    )


if __name__ == "__main__":
    main()
