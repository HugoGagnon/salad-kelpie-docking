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
  solvated.pdb, system.xml, trajectory.dcd, checkpoint.chk, progress.json

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

    name = f"{target}__{ligand}"
    seed = int(os.getenv("SEED", str(42 + replica)))
    out_subdir = Path(out_dir) / name

    # SKIP_NS is a container-group-wide constant but prod_ns is per job, so a
    # short run can ask the scorer to discard its whole trajectory.  Check that
    # here, in seconds, rather than discovering it after a full MD run.
    skip_ns = float(os.getenv("SKIP_NS", "2.0"))
    if skip_ns >= prod_ns:
        _record_failure(
            out_subdir, replica, stage="preflight",
            reason=(f"SKIP_NS ({skip_ns} ns) is >= the requested trajectory "
                    f"length ({prod_ns} ns); every frame would be discarded as "
                    f"equilibration and there would be nothing to score. Lower "
                    f"SKIP_NS on the container group or raise --prod-ns."),
        )

    receptor = _find_input(data_dir, name, "receptor.pdb")
    ligand_file = _find_input(data_dir, name, "ligand.sdf")
    print(f"[gpu_worker] {name} rep{replica} | {prod_ns} ns | seed={seed}")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # ── stage 1: MD run ───────────────────────────────────────────────────────
    engine = SCRIPTS_DIR / "md_engine.py"
    walltime_h = float(os.getenv("WALLTIME_H", "20.0"))
    ckpt_ps = int(os.getenv("CHECKPOINT_PS", "500"))

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
    ret = subprocess.run(md_cmd).returncode
    if ret != 0:
        sys.exit(f"md_engine exited {ret}")

    # ── verify trajectory is complete ─────────────────────────────────────────
    progress_src = Path(ckpt_dir) / "progress.json"
    if not progress_src.exists():
        sys.exit("progress.json not found — MD did not write any output")
    with open(progress_src) as fh:
        progress = json.load(fh)
    if not progress.get("complete"):
        # The engine saved a checkpoint but did not reach the target duration.
        # Exit non-zero so Kelpie retries; sync.during has already pushed the
        # checkpoint to R2, so the next allocation resumes without re-running.
        print(f"[gpu_worker] trajectory incomplete ({progress.get('ns', 0):.1f} "
              f"/ {prod_ns} ns) — checkpoint synced, Kelpie will retry")
        sys.exit(1)

    # ── stage 2: MM-GBSA scoring ───────────────────────────────────────────────
    scorer = SCRIPTS_DIR / "mmgbsa.py"
    n_frames = int(os.getenv("N_FRAMES", "50"))

    out_subdir.mkdir(parents=True, exist_ok=True)
    mmgbsa_json = out_subdir / f"rep{replica}_mmgbsa.json"

    score_cmd = [
        sys.executable, str(scorer),
        "--work-dir", ckpt_dir,
        "--output", str(mmgbsa_json),
        "--n-frames", str(n_frames),
        "--skip-ns", str(skip_ns),
    ]
    ret = subprocess.run(score_cmd).returncode
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


def _find_input(data_dir: str, name: str, filename: str) -> str:
    path = Path(data_dir) / name / filename
    if path.is_file():
        return str(path)
    # Fallback: glob for the filename anywhere under data_dir/<name>/
    matches = list(Path(data_dir).glob(f"*{name}*/{filename}"))
    if matches:
        return str(matches[0])
    sys.exit(f"input not found: {filename} for complex {name} in {data_dir}")


if __name__ == "__main__":
    main()
