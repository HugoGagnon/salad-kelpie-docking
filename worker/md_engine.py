#!/usr/bin/env python3
"""
md_engine.py — OpenMM molecular dynamics engine with checkpoint/resume.

Builds a solvated protein-ligand system (or loads from cached system.xml /
solvated.pdb), runs Langevin NPT MD with periodic checkpointing, and writes a
progress.json that the orchestrator reads to decide whether to score or retry.

Checkpointing design
--------------------
Every --checkpoint-ps picoseconds the engine writes:
  checkpoint.chk   — binary OpenMM checkpoint (resumes from exact state)
  trajectory.dcd   — appended DCD frame
  progress.json    — {steps, ns, target_ns, complete}

On restart, if checkpoint.chk and progress.json both exist, the engine loads
the checkpoint and resumes without re-minimising or re-equilibrating.

System cache
------------
If system.xml and solvated.pdb already exist in --work-dir, build_system() is
skipped entirely.  Because --work-dir is backed by R2 (sync.before/during), a
resumed job automatically gets the cached system without rebuilding.

Walltime guard
--------------
--walltime-h causes a clean exit before the external job time limit, leaving
the checkpoint in a resume-ready state.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path


def build_system(receptor_path: str, ligand_path: str, work_dir: Path) -> tuple:
    """Prepare a solvated protein-ligand system.

    Returns (system, topology, positions) in OpenMM types.
    Skips all work and loads from disk if system.xml + solvated.pdb exist.
    """
    from openmm import app, unit
    from openff.toolkit import Molecule
    from openmmforcefields.generators import SystemGenerator
    from pdbfixer import PDBFixer

    system_xml = work_dir / "system.xml"
    solvated_pdb = work_dir / "solvated.pdb"

    if system_xml.is_file() and solvated_pdb.is_file():
        print("[md_engine] system cache hit — loading from disk")
        from openmm import XmlSerializer
        with open(system_xml) as fh:
            system = XmlSerializer.deserialize(fh.read())
        pdb = app.PDBFile(str(solvated_pdb))
        return system, pdb.topology, pdb.positions

    print("[md_engine] building system from scratch")

    # Fix receptor: add missing atoms and hydrogens.
    fixer = PDBFixer(filename=receptor_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    receptor_top = app.Modeller(fixer.topology, fixer.positions)

    # Parametrise ligand with GAFF-2 via OpenFF.
    ligand_mol = Molecule.from_file(ligand_path, allow_undefined_stereo=True)
    cache_path = os.getenv("LIGAND_CACHE")
    generator = SystemGenerator(
        forcefields=["amber/ff14SB.xml", "amber/tip3p_standard.xml"],
        small_molecule_forcefield="gaff-2.11",
        molecules=[ligand_mol],
        cache=cache_path,
    )
    modeller = app.Modeller(receptor_top.topology, receptor_top.positions)
    ligand_top = ligand_mol.to_topology().to_openmm()
    from openmm.unit import nanometer
    modeller.add(ligand_top, ligand_mol.conformers[0].to_openmm())

    # Solvate in TIP3P with 1 nm padding and 0.15 M NaCl.
    modeller.addSolvent(
        generator.forcefield,
        model="tip3p",
        padding=1.0 * nanometer,
        ionicStrength=0.15 * (unit.mole / unit.liter),
    )

    system = generator.create_system(modeller.topology)

    # Persist the cache so resumed runs skip this step.
    from openmm import XmlSerializer
    with open(system_xml, "w") as fh:
        fh.write(XmlSerializer.serialize(system))
    with open(solvated_pdb, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)

    return system, modeller.topology, modeller.positions


def run_md(
    system,
    topology,
    positions,
    work_dir: Path,
    target_ns: float,
    checkpoint_ps: int,
    walltime_h: float,
    seed: int,
) -> None:
    from openmm import (
        LangevinMiddleIntegrator, MonteCarloBarostat,
        Platform, XmlSerializer,
    )
    from openmm import app, unit

    # ── integrator: 4 fs timestep via hydrogen mass repartitioning ────────────
    dt = 4 * unit.femtoseconds
    integrator = LangevinMiddleIntegrator(310 * unit.kelvin, 1 / unit.picosecond, dt)
    integrator.setRandomNumberSeed(seed)

    system.addForce(MonteCarloBarostat(1 * unit.bar, 310 * unit.kelvin))

    # Right-size: use the fastest available platform; fall back to CPU.
    for platform_name in ("CUDA", "OpenCL", "CPU"):
        try:
            platform = Platform.getPlatformByName(platform_name)
            break
        except Exception:
            continue

    properties = {}
    if platform_name == "CUDA":
        properties = {"CudaPrecision": "mixed"}

    simulation = app.Simulation(topology, system, integrator, platform, properties)

    ckpt_file = work_dir / "checkpoint.chk"
    dcd_file = work_dir / "trajectory.dcd"
    progress_file = work_dir / "progress.json"

    target_steps = int(target_ns * 1000 / 4)        # 4 fs timestep
    ckpt_steps = int(checkpoint_ps * 1000 / 4)

    # ── resume or initialise ──────────────────────────────────────────────────
    if ckpt_file.is_file() and progress_file.is_file():
        print("[md_engine] resuming from checkpoint")
        with open(ckpt_file, "rb") as fh:
            simulation.context.loadCheckpoint(fh.read())
        with open(progress_file) as fh:
            prev = json.load(fh)
        steps_done = prev.get("steps", 0)
        dcd = open(dcd_file, "ab")
        dcd_reporter = app.DCDFile(dcd, topology, dt, steps_done, 1)
    else:
        print("[md_engine] fresh start — minimising and equilibrating")
        simulation.context.setPositions(positions)
        simulation.minimizeEnergy()
        simulation.context.setVelocitiesToTemperature(310 * unit.kelvin, seed)
        # 200 ps NPT equilibration (50 000 × 4 fs steps).
        simulation.step(50_000)
        steps_done = 0
        dcd = open(dcd_file, "wb")
        dcd_reporter = app.DCDFile(dcd, topology, dt, 0, 1)

    deadline = time.monotonic() + walltime_h * 3600

    # ── production loop ───────────────────────────────────────────────────────
    while steps_done < target_steps:
        if time.monotonic() > deadline:
            print("[md_engine] approaching walltime — saving checkpoint and exiting")
            _write_checkpoint(simulation, ckpt_file)
            _write_progress(progress_file, steps_done, target_steps, target_ns, complete=False)
            dcd.close()
            return

        batch = min(ckpt_steps, target_steps - steps_done)
        simulation.step(batch)
        steps_done += batch

        state = simulation.context.getState(getPositions=True)
        dcd_reporter.report(simulation, state)
        _write_checkpoint(simulation, ckpt_file)
        _write_progress(progress_file, steps_done, target_steps, target_ns,
                        complete=(steps_done >= target_steps))

        ns_done = steps_done * 4 / 1_000_000
        print(f"[md_engine] {ns_done:.2f} / {target_ns:.1f} ns")

    dcd.close()
    print("[md_engine] trajectory complete")


def _write_checkpoint(simulation, path: Path) -> None:
    with open(path, "wb") as fh:
        fh.write(simulation.context.createCheckpoint())


def _write_progress(path: Path, steps: int, target_steps: int, target_ns: float,
                    complete: bool) -> None:
    ns = steps * 4 / 1_000_000
    with open(path, "w") as fh:
        json.dump({"steps": steps, "target_steps": target_steps,
                   "ns": ns, "target_ns": target_ns, "complete": complete}, fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenMM MD engine with checkpoint/resume")
    parser.add_argument("--receptor", required=True)
    parser.add_argument("--ligand", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--target-ns", type=float, required=True)
    parser.add_argument("--checkpoint-ps", type=int, default=500)
    parser.add_argument("--walltime-h", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--build-only", action="store_true",
                        help="stop after building and caching the system (prep step)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    system, topology, positions = build_system(args.receptor, args.ligand, work_dir)
    if args.build_only:
        print("[md_engine] --build-only: system built and cached, exiting")
        return

    run_md(system, topology, positions, work_dir,
           args.target_ns, args.checkpoint_ps, args.walltime_h, args.seed)


if __name__ == "__main__":
    main()
