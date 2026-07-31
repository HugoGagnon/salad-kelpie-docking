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

On restart, checkpoint.chk, progress.json, and trajectory.dcd must all exist;
the engine loads the checkpoint and appends without re-minimising or
re-equilibrating.

System cache
------------
If system.xml, solvated.pdb, and system_metadata.json already exist in
--work-dir, build_system() is skipped entirely.  Because --work-dir is backed
by R2 (sync.before/during), a resumed job automatically gets the cached system
without rebuilding.

Walltime guard
--------------
--walltime-h causes a clean exit before the external job time limit, leaving
the checkpoint in a resume-ready state.
"""
import argparse
import json
import os
import time
from pathlib import Path

TIMESTEP_FS = 4.0
FS_PER_NS = 1_000_000.0
FS_PER_PS = 1_000.0
SYSTEM_METADATA_VERSION = 1


def steps_for_duration(value: float, unit_fs: float, timestep_fs: float = TIMESTEP_FS) -> int:
    """Convert a positive duration to an exact, positive integration step count."""
    if value <= 0:
        raise ValueError("duration must be positive")
    if unit_fs <= 0 or timestep_fs <= 0:
        raise ValueError("time units and timestep must be positive")
    steps = round(value * unit_fs / timestep_fs)
    if steps <= 0:
        raise ValueError("duration is shorter than one integration step")
    return steps


def build_system(receptor_path: str, ligand_path: str, work_dir: Path) -> tuple:
    """Prepare a solvated protein-ligand system.

    Returns (system, topology, positions) in OpenMM types.
    Skips all work and loads from disk if system.xml + solvated.pdb exist.
    """
    from openff.toolkit import Molecule
    from openmm import app, unit
    from openmmforcefields.generators import SystemGenerator
    from pdbfixer import PDBFixer

    system_xml = work_dir / "system.xml"
    solvated_pdb = work_dir / "solvated.pdb"
    metadata_json = work_dir / "system_metadata.json"

    cache_state = [path.is_file() for path in (system_xml, solvated_pdb, metadata_json)]
    if all(cache_state):
        print("[md_engine] system cache hit — loading from disk")
        from openmm import XmlSerializer
        with open(system_xml) as fh:
            system = XmlSerializer.deserialize(fh.read())
        pdb = app.PDBFile(str(solvated_pdb))
        return system, pdb.topology, pdb.positions
    if any(cache_state):
        raise RuntimeError(
            "incomplete or legacy system cache detected; use a new RUN_PREFIX so "
            "system.xml, solvated.pdb, and system_metadata.json are rebuilt together"
        )

    print("[md_engine] building system from scratch")

    # Fix receptor: add missing atoms and hydrogens.
    fixer = PDBFixer(filename=receptor_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    receptor_top = app.Modeller(fixer.topology, fixer.positions)
    receptor_atom_count = receptor_top.topology.getNumAtoms()

    # Parametrise ligand with GAFF-2 via OpenFF.
    ligand_mol = Molecule.from_file(ligand_path)
    cache_path = os.getenv("LIGAND_CACHE")
    generator = SystemGenerator(
        forcefields=["amber/ff14SB.xml", "amber/tip3p_standard.xml"],
        small_molecule_forcefield="gaff-2.11",
        molecules=[ligand_mol],
        cache=cache_path,
        forcefield_kwargs={
            "constraints": app.HBonds,
            "rigidWater": True,
            "removeCMMotion": False,
            "hydrogenMass": 4 * unit.amu,
        },
    )
    modeller = app.Modeller(receptor_top.topology, receptor_top.positions)
    ligand_top = ligand_mol.to_topology().to_openmm()
    ligand_atom_count = ligand_top.getNumAtoms()
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

    # Persist the cache as a complete tuple.  Metadata is written last and acts
    # as the cache-complete marker, preventing a preemption during setup from
    # mixing a partial cache with a later checkpoint.
    from openmm import XmlSerializer
    system_tmp = system_xml.with_suffix(".xml.tmp")
    pdb_tmp = solvated_pdb.with_suffix(".pdb.tmp")
    with open(system_tmp, "w") as fh:
        fh.write(XmlSerializer.serialize(system))
    with open(pdb_tmp, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)
    os.replace(system_tmp, system_xml)
    os.replace(pdb_tmp, solvated_pdb)
    _atomic_write_json(metadata_json, {
        "version": SYSTEM_METADATA_VERSION,
        "receptor_atom_count": receptor_atom_count,
        "ligand_atom_count": ligand_atom_count,
        "timestep_fs": TIMESTEP_FS,
        "forcefield": "amber/ff14SB.xml",
        "water_model": "amber/tip3p_standard.xml",
        "ligand_forcefield": "gaff-2.11",
        "hydrogen_mass_amu": 4.0,
    })

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
        LangevinMiddleIntegrator,
        MonteCarloBarostat,
        OpenMMException,
        Platform,
        app,
        unit,
    )

    # ── integrator: 4 fs timestep via hydrogen mass repartitioning ────────────
    dt = TIMESTEP_FS * unit.femtoseconds
    integrator = LangevinMiddleIntegrator(310 * unit.kelvin, 1 / unit.picosecond, dt)
    integrator.setRandomNumberSeed(seed)

    system.addForce(MonteCarloBarostat(1 * unit.bar, 310 * unit.kelvin))

    # Right-size: use the fastest available platform; fall back to CPU.
    for platform_name in ("CUDA", "OpenCL", "CPU"):
        try:
            platform = Platform.getPlatformByName(platform_name)
            break
        except OpenMMException:
            continue

    properties = {}
    if platform_name == "CUDA":
        properties = {"CudaPrecision": "mixed"}

    simulation = app.Simulation(topology, system, integrator, platform, properties)

    ckpt_file = work_dir / "checkpoint.chk"
    dcd_file = work_dir / "trajectory.dcd"
    progress_file = work_dir / "progress.json"

    target_steps = steps_for_duration(target_ns, FS_PER_NS)
    ckpt_steps = steps_for_duration(checkpoint_ps, FS_PER_PS)
    if target_steps < ckpt_steps:
        raise ValueError(
            f"target duration ({target_ns} ns) is shorter than CHECKPOINT_PS "
            f"({checkpoint_ps} ps); lower CHECKPOINT_PS so at least one trajectory frame is written"
        )

    # ── resume or initialise ──────────────────────────────────────────────────
    resume_state = [ckpt_file.is_file(), progress_file.is_file(), dcd_file.is_file()]
    if all(resume_state):
        print("[md_engine] resuming from checkpoint")
        with open(ckpt_file, "rb") as fh:
            simulation.context.loadCheckpoint(fh.read())
        with open(progress_file) as fh:
            prev = json.load(fh)
        steps_done = int(prev.get("steps", 0))
        if steps_done < 0 or steps_done > target_steps:
            raise RuntimeError(f"invalid checkpoint progress: {steps_done} steps")
        if prev.get("checkpoint_steps") != ckpt_steps:
            raise RuntimeError(
                "CHECKPOINT_PS changed for an existing trajectory; use the original "
                "value or start a new RUN_PREFIX"
            )
        if prev.get("timestep_fs") != TIMESTEP_FS:
            raise RuntimeError("trajectory timestep changed; start a new RUN_PREFIX")
        dcd_mode = "r+b"
        append_dcd = True
    elif any(resume_state):
        raise RuntimeError(
            "partial checkpoint detected; checkpoint.chk, progress.json, and "
            "trajectory.dcd must be restored together"
        )
    else:
        print("[md_engine] fresh start — minimising and equilibrating")
        simulation.context.setPositions(positions)
        simulation.minimizeEnergy()
        simulation.context.setVelocitiesToTemperature(310 * unit.kelvin, seed)
        # 200 ps NPT equilibration (50 000 × 4 fs steps).
        simulation.step(50_000)
        steps_done = 0
        dcd_mode = "wb"
        append_dcd = False

    with open(dcd_file, dcd_mode) as dcd:
        dcd_writer = app.DCDFile(
            dcd, topology, dt, firstStep=ckpt_steps,
            interval=ckpt_steps, append=append_dcd,
        )
        deadline = time.monotonic() + walltime_h * 3600

        # ── production loop ───────────────────────────────────────────────────
        while steps_done < target_steps:
            if time.monotonic() > deadline:
                print("[md_engine] approaching walltime — saving checkpoint and exiting")
                _write_checkpoint(simulation, ckpt_file)
                _write_progress(
                    progress_file, steps_done, target_steps, target_ns,
                    checkpoint_steps=ckpt_steps, complete=False,
                )
                return

            batch = min(ckpt_steps, target_steps - steps_done)
            simulation.step(batch)
            steps_done += batch

            state = simulation.context.getState(getPositions=True)
            # DCD stores a fixed reporting interval.  Do not write an irregular
            # final partial interval, since doing so would give it a false timestamp.
            if steps_done % ckpt_steps == 0:
                dcd_writer.writeModel(
                    state.getPositions(),
                    periodicBoxVectors=state.getPeriodicBoxVectors(),
                )
            _write_checkpoint(simulation, ckpt_file)
            _write_progress(
                progress_file, steps_done, target_steps, target_ns,
                checkpoint_steps=ckpt_steps, complete=(steps_done >= target_steps),
            )

            ns_done = steps_done * TIMESTEP_FS / FS_PER_NS
            print(f"[md_engine] {ns_done:.2f} / {target_ns:.1f} ns")

    print("[md_engine] trajectory complete")


def _write_checkpoint(simulation, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(simulation.context.createCheckpoint())
    os.replace(tmp, path)


def _write_progress(path: Path, steps: int, target_steps: int, target_ns: float,
                    checkpoint_steps: int, complete: bool) -> None:
    ns = steps * TIMESTEP_FS / FS_PER_NS
    _atomic_write_json(path, {
        "steps": steps,
        "target_steps": target_steps,
        "ns": ns,
        "target_ns": target_ns,
        "checkpoint_steps": checkpoint_steps,
        "complete": complete,
        "timestep_fs": TIMESTEP_FS,
    })


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


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

    if args.target_ns <= 0:
        parser.error("--target-ns must be positive")
    if args.checkpoint_ps <= 0:
        parser.error("--checkpoint-ps must be positive")
    if args.walltime_h <= 0:
        parser.error("--walltime-h must be positive")

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
