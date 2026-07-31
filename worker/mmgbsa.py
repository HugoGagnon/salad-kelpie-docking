#!/usr/bin/env python3
"""
mmgbsa.py — PBC-corrected single-trajectory MM-GBSA scorer.

Reads the trajectory written by md_engine.py and computes an implicit-solvent
MM-GBSA binding free energy estimate using OpenMM's GBn2 model.

The ligand is first placed in the same periodic image as the receptor.  Atom
partitioning comes from build-time metadata, so multi-chain receptors remain
intact, and reusable GBn2/GAFF contexts are built once per scoring job.

Protocol: single-trajectory (receptor and ligand extracted from the same MD
frame), so receptor internal energy cancels in ΔG = E_complex − E_rec − E_lig.

Output JSON:
  {dG_mean, dG_sem, dG_frames, n_frames, skip_ns}
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def load_trajectory(work_dir: Path, skip_ns: float, n_frames: int):
    import mdtraj as md

    dcd = work_dir / "trajectory.dcd"
    top = work_dir / "solvated.pdb"
    if not dcd.is_file():
        sys.exit(f"trajectory not found: {dcd}")
    if not top.is_file():
        sys.exit(f"topology not found: {top}")

    if skip_ns < 0:
        raise ValueError("skip_ns must be zero or greater")
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")

    traj = md.load(str(dcd), top=str(top))
    # DCD time is expressed in ps.  Select by the recorded frame times rather
    # than assuming a frame interval, which remains correct after append/resume.
    times_ns = traj.time / 1000.0
    traj = traj[times_ns >= skip_ns]
    if len(traj) == 0:
        sys.exit(f"no frames remain after skipping {skip_ns} ns equilibration")

    stride = max(1, len(traj) // n_frames)
    traj = traj[::stride][:n_frames]
    return traj


def _unwrap_ligand(positions, box_vectors, rec_indices, lig_indices):
    """Minimum-image convention: shift ligand COM into same image as receptor COM."""
    rec_com = positions[rec_indices].mean(axis=0)
    lig_com = positions[lig_indices].mean(axis=0)
    delta = lig_com - rec_com
    # Assume orthorhombic box.
    box = np.diag(box_vectors)
    shift = np.round(delta / box) * box
    positions[lig_indices] -= shift
    return positions


def load_system_metadata(work_dir: Path) -> dict:
    path = work_dir / "system_metadata.json"
    if not path.is_file():
        sys.exit(
            f"system metadata not found: {path}; this is a legacy/incompatible "
            "checkpoint, so submit with a new RUN_PREFIX"
        )
    with open(path) as fh:
        metadata = json.load(fh)
    if metadata.get("version") != 1:
        sys.exit(f"unsupported system metadata version: {metadata.get('version')!r}")
    return metadata


def partition_indices(metadata: dict, topology_atom_count: int) -> tuple[list[int], list[int]]:
    """Recover receptor and ligand atoms from build-time counts, not chain IDs."""
    try:
        receptor_count = int(metadata["receptor_atom_count"])
        ligand_count = int(metadata["ligand_atom_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid receptor/ligand atom counts in system metadata") from exc
    if receptor_count <= 0 or ligand_count <= 0:
        raise ValueError("receptor and ligand atom counts must be positive")
    if receptor_count + ligand_count > topology_atom_count:
        raise ValueError(
            "system metadata atom counts exceed the solvated topology atom count"
        )
    receptor = list(range(receptor_count))
    ligand = list(range(receptor_count, receptor_count + ligand_count))
    return receptor, ligand


def build_energy_contexts(topology_openmm, ligand_path: str,
                          rec_indices: list[int], lig_indices: list[int]):
    """Build reusable GBn2/GAFF contexts for complex, receptor, and ligand."""
    from openff.toolkit import Molecule
    from openmm import Context, Platform, app
    from openmmforcefields.generators import SystemGenerator

    ligand_mol = Molecule.from_file(ligand_path)
    generator = SystemGenerator(
        forcefields=["amber/ff14SB.xml", "implicit/gbn2.xml"],
        small_molecule_forcefield="gaff-2.11",
        molecules=[ligand_mol],
        nonperiodic_forcefield_kwargs={"nonbondedMethod": app.NoCutoff},
    )
    platform = Platform.getPlatformByName("CPU")
    selections = {
        "complex": rec_indices + lig_indices,
        "receptor": rec_indices,
        "ligand": lig_indices,
    }
    contexts = {}
    for label, atom_indices in selections.items():
        sub_top = _subset_topology(topology_openmm, atom_indices)
        molecules = [ligand_mol] if label in ("complex", "ligand") else None
        system = generator.create_system(sub_top, molecules=molecules)
        integrator = _dummy_integrator()
        context = Context(system, integrator, platform)
        # Retain the integrator reference with the context for the lifetime of
        # the scorer; OpenMM contexts own their integrator but Python wrappers
        # can otherwise be collected unexpectedly in some versions.
        contexts[label] = (context, integrator, atom_indices)
    return contexts


def score_frame(positions_nm, contexts):
    """Compute ΔG_bind = E_complex − E_receptor − E_ligand (GBn2, CPU)."""
    from openmm import unit

    energies = {}
    for label, (context, _integrator, atom_indices) in contexts.items():
        context.setPositions(positions_nm[atom_indices] * unit.nanometers)
        state = context.getState(getEnergy=True)
        energies[label] = state.getPotentialEnergy().value_in_unit(
            unit.kilocalories_per_mole
        )
    return energies["complex"] - energies["receptor"] - energies["ligand"]


def _subset_topology(full_top, atom_indices):
    from openmm import app
    new_top = app.Topology()
    index_map = {old: new for new, old in enumerate(atom_indices)}
    chains = {}
    residues = {}
    for atom in full_top.atoms():
        if atom.index not in index_map:
            continue
        chain = atom.residue.chain
        if chain not in chains:
            chains[chain] = new_top.addChain(chain.id)
        res = atom.residue
        if res not in residues:
            residues[res] = new_top.addResidue(res.name, chains[chain], res.id)
        new_top.addAtom(atom.name, atom.element, residues[res])
    atoms = list(new_top.atoms())
    for bond in full_top.bonds():
        a1, a2 = bond.atom1.index, bond.atom2.index
        if a1 in index_map and a2 in index_map:
            new_top.addBond(atoms[index_map[a1]], atoms[index_map[a2]])
    return new_top


def _dummy_integrator():
    from openmm import VerletIntegrator, unit
    return VerletIntegrator(0.001 * unit.picoseconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="PBC-corrected MM-GBSA scorer")
    parser.add_argument("--work-dir", required=True,
                        help="directory containing trajectory.dcd, solvated.pdb, system.xml")
    parser.add_argument("--output", required=True, help="path for output JSON")
    parser.add_argument("--ligand", required=True,
                        help="source ligand SDF used to regenerate the GAFF template")
    parser.add_argument("--n-frames", type=int, default=50,
                        help="number of frames to score (default: 50)")
    parser.add_argument("--skip-ns", type=float, default=2.0,
                        help="ns of equilibration to skip at scoring (default: 2.0)")
    args = parser.parse_args()

    if args.n_frames <= 0:
        parser.error("--n-frames must be positive")
    if args.skip_ns < 0:
        parser.error("--skip-ns must be zero or greater")

    work_dir = Path(args.work_dir)
    metadata = load_system_metadata(work_dir)

    print(f"[mmgbsa] loading trajectory from {work_dir}")
    traj = load_trajectory(work_dir, args.skip_ns, args.n_frames)
    print(f"[mmgbsa] scoring {len(traj)} frames")

    from openmm import app

    pdb = app.PDBFile(str(work_dir / "solvated.pdb"))
    try:
        rec_indices, lig_indices = partition_indices(
            metadata, pdb.topology.getNumAtoms()
        )
    except ValueError as exc:
        sys.exit(str(exc))
    contexts = build_energy_contexts(
        pdb.topology, args.ligand, rec_indices, lig_indices
    )

    dg_frames = []
    for i, frame in enumerate(traj):
        pos = frame.xyz[0].copy()  # nm
        box = frame.unitcell_vectors[0] if frame.unitcell_vectors is not None else None
        if box is not None:
            pos = _unwrap_ligand(pos, box, rec_indices, lig_indices)
        dg = score_frame(pos, contexts)
        dg_frames.append(dg)
        print(f"[mmgbsa] frame {i+1}/{len(traj)}: ΔG = {dg:.2f} kcal/mol")

    arr = np.array(dg_frames)
    result = {
        "dG_mean": float(arr.mean()),
        "dG_sem": float(arr.std(ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "dG_frames": dg_frames,
        "n_frames": len(dg_frames),
        "skip_ns": args.skip_ns,
        "method": "single-trajectory GBn2 endpoint estimate with GAFF-2.11 ligand parameters",
    }
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[mmgbsa] ΔG = {result['dG_mean']:.2f} ± {result['dG_sem']:.2f} kcal/mol")
    print(f"[mmgbsa] result written to {args.output}")


if __name__ == "__main__":
    main()
