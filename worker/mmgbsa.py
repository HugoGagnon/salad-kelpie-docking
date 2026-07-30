#!/usr/bin/env python3
"""
mmgbsa.py — PBC-corrected single-trajectory MM-GBSA scorer.

Reads the trajectory written by md_engine.py and computes an implicit-solvent
MM-GBSA binding free energy estimate using OpenMM's GBn2 model.

Two corrections applied before scoring each frame:
  1. PBC minimum-image unwrapping: places the ligand COM in the same periodic
     image as the receptor so near-zero artefacts from a wrapped ligand are
     avoided.
  2. Bond re-imposition: re-applies bonds from the reference topology before
     every createSystem call, restoring N-terminal H-N bonds and ligand
     intramolecular bonds that MDTraj's DCD slice inference drops.

Protocol: single-trajectory (receptor and ligand extracted from the same MD
frame), so receptor internal energy cancels in ΔG = E_complex − E_rec − E_lig.

Output JSON:
  {dG_mean, dG_sem, dG_frames, n_frames, skip_ns}
"""
import argparse
import json
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

    traj = md.load(str(dcd), top=str(top))
    dt_ns = traj.timestep / 1000.0   # MDTraj reports timestep in ps
    skip_frames = int(skip_ns / dt_ns) if dt_ns > 0 else 0
    traj = traj[skip_frames:]
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


def score_frame(positions_nm, topology_openmm, system_xml_path: Path,
                rec_indices, lig_indices):
    """Compute ΔG_bind = E_complex − E_receptor − E_ligand (GBn2, CPU)."""
    from openmm import XmlSerializer, Context, Platform
    from openmm import app, unit
    from openmm.app import GBn2

    with open(system_xml_path) as fh:
        system_full = XmlSerializer.deserialize(fh.read())

    # GBn2 implicit solvent system.
    forcefield = app.ForceField("amber/ff14SB.xml", "implicit/gbn2.xml")
    platform = Platform.getPlatformByName("CPU")

    all_indices = set(range(topology_openmm.getNumAtoms()))
    rec_set = set(rec_indices)
    lig_set = set(lig_indices)

    def _energy(indices):
        """Extract sub-topology, re-impose bonds, create implicit system, compute energy."""
        import mdtraj as md
        atom_list = sorted(indices)
        sub_pos = positions_nm[atom_list] * unit.nanometers

        pdb = app.PDBFile(str(system_xml_path.parent / "solvated.pdb"))
        sub_top = _subset_topology(pdb.topology, atom_list)

        # Re-impose bonds from the reference PDB topology before createSystem.
        # MDTraj DCD loading infers bonds and may drop N-terminal H-N bonds and
        # ligand intramolecular bonds; this step restores them.
        _restore_bonds(sub_top, pdb.topology, atom_list)

        sys_implicit = forcefield.createSystem(
            sub_top,
            nonbondedMethod=app.NoCutoff,
            implicitSolvent=GBn2,
        )
        ctx = Context(sys_implicit, _dummy_integrator(), platform)
        ctx.setPositions(sub_pos)
        state = ctx.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    e_complex = _energy(list(all_indices - set(_solvent_indices(topology_openmm))))
    e_rec = _energy(list(rec_set))
    e_lig = _energy(list(lig_set))
    return e_complex - e_rec - e_lig


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
    for bond in full_top.bonds():
        a1, a2 = bond.atom1.index, bond.atom2.index
        if a1 in index_map and a2 in index_map:
            atoms = list(new_top.atoms())
            new_top.addBond(atoms[index_map[a1]], atoms[index_map[a2]])
    return new_top


def _restore_bonds(sub_top, ref_top, atom_indices):
    """Add any bonds from ref_top that are missing in sub_top."""
    index_map = {old: new for new, old in enumerate(atom_indices)}
    existing = {(b.atom1.index, b.atom2.index) for b in sub_top.bonds()}
    atoms = list(sub_top.atoms())
    for bond in ref_top.bonds():
        a1, a2 = bond.atom1.index, bond.atom2.index
        if a1 in index_map and a2 in index_map:
            n1, n2 = index_map[a1], index_map[a2]
            if (n1, n2) not in existing and (n2, n1) not in existing:
                sub_top.addBond(atoms[n1], atoms[n2])


def _solvent_indices(topology):
    return [a.index for a in topology.atoms()
            if a.residue.name in ("HOH", "WAT", "NA", "CL", "SOD", "CLA")]


def _dummy_integrator():
    from openmm import VerletIntegrator
    from openmm import unit
    return VerletIntegrator(0.001 * unit.picoseconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="PBC-corrected MM-GBSA scorer")
    parser.add_argument("--work-dir", required=True,
                        help="directory containing trajectory.dcd, solvated.pdb, system.xml")
    parser.add_argument("--output", required=True, help="path for output JSON")
    parser.add_argument("--n-frames", type=int, default=50,
                        help="number of frames to score (default: 50)")
    parser.add_argument("--skip-ns", type=float, default=2.0,
                        help="ns of equilibration to skip at scoring (default: 2.0)")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    system_xml = work_dir / "system.xml"
    if not system_xml.is_file():
        sys.exit(f"system.xml not found in {work_dir}")

    print(f"[mmgbsa] loading trajectory from {work_dir}")
    traj = load_trajectory(work_dir, args.skip_ns, args.n_frames)
    print(f"[mmgbsa] scoring {len(traj)} frames")

    import mdtraj as md
    from openmm import app

    pdb = app.PDBFile(str(work_dir / "solvated.pdb"))
    solvent = set(_solvent_indices(pdb.topology))
    all_idx = set(range(pdb.topology.getNumAtoms()))
    non_solvent = sorted(all_idx - solvent)

    # Identify receptor and ligand atoms: ligand is the last non-protein chain.
    rec_indices = [a.index for a in pdb.topology.atoms()
                   if a.index in all_idx - solvent and a.residue.chain.index == 0]
    lig_indices = [a.index for a in pdb.topology.atoms()
                   if a.index in all_idx - solvent and a.residue.chain.index != 0
                   and a.index not in solvent]

    dg_frames = []
    for i, frame in enumerate(traj):
        pos = frame.xyz[0].copy()  # nm
        box = frame.unitcell_vectors[0] if frame.unitcell_vectors is not None else None
        if box is not None:
            pos = _unwrap_ligand(pos, box, rec_indices, lig_indices)
        dg = score_frame(pos, pdb.topology, system_xml, rec_indices, lig_indices)
        dg_frames.append(dg)
        print(f"[mmgbsa] frame {i+1}/{len(traj)}: ΔG = {dg:.2f} kcal/mol")

    arr = np.array(dg_frames)
    result = {
        "dG_mean": float(arr.mean()),
        "dG_sem": float(arr.std() / len(arr) ** 0.5),
        "dG_frames": dg_frames,
        "n_frames": len(dg_frames),
        "skip_ns": args.skip_ns,
    }
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[mmgbsa] ΔG = {result['dG_mean']:.2f} ± {result['dG_sem']:.2f} kcal/mol")
    print(f"[mmgbsa] result written to {args.output}")


if __name__ == "__main__":
    main()
