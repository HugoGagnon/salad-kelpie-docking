"""
Consistency tests for the worked example.

These guard a failure mode that unit tests on submit.py cannot catch: a
manifest that references a job id or complex whose input files do not exist.
Such a job submits cleanly, is dispatched to a node, and only then fails —
after the allocation has been paid for.

No network access.
"""
import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE_DIR = os.path.join(REPO_ROOT, "examples", "docking")
DATA_DIR = os.path.join(EXAMPLE_DIR, "data")

# Files each mode's worker requires to be present in a job's input directory.
CPU_REQUIRED = ("receptor.pdbqt", "ligand.pdbqt", "box.txt")
GPU_REQUIRED = ("receptor.pdb", "ligand.sdf")

# Heavy-atom counts for the four HIV-1 protease inhibitors in the example,
# derived from their published molecular formulae.  A ligand file that misses
# these is a truncated or wrong structure, which docks to a plausible-looking
# but meaningless affinity.
EXPECTED_HEAVY_ATOMS = {
    "1hsg__indinavir": 45,   # C36H47N5O4
    "1hsg__ritonavir": 50,   # C37H48N6O5S2
    "1hsg__nelfinavir": 40,  # C32H45N3O4S
    "1hsg__lopinavir": 46,   # C37H48N4O5
}


def _load(name):
    with open(os.path.join(EXAMPLE_DIR, name)) as fh:
        return json.load(fh)


def _count_heavy_atoms(pdbqt_path):
    """Count non-hydrogen atoms.  The PDBQT autodock type is the last field."""
    n = 0
    with open(pdbqt_path) as fh:
        for line in fh:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            fields = line.split()
            if fields and fields[-1] not in ("H", "HD", "HS"):
                n += 1
    return n


def _job_ids():
    return [job["id"] for job in _load("jobs.json")["jobs"]]


def _complex_names():
    return [cx["name"] for cx in _load("matrix.json")["complexes"]]


@pytest.mark.parametrize("job_id", _job_ids())
@pytest.mark.parametrize("filename", CPU_REQUIRED)
def test_cpu_manifest_job_has_required_input(job_id, filename):
    path = os.path.join(DATA_DIR, job_id, filename)
    assert os.path.isfile(path), (
        f"jobs.json lists job {job_id!r} but {filename} is missing from "
        f"examples/docking/data/{job_id}/ — the job would fail on the node"
    )


@pytest.mark.parametrize("name", _complex_names())
@pytest.mark.parametrize("filename", GPU_REQUIRED)
def test_gpu_manifest_complex_has_required_input(name, filename):
    path = os.path.join(DATA_DIR, name, filename)
    assert os.path.isfile(path), (
        f"matrix.json lists complex {name!r} but {filename} is missing from "
        f"examples/docking/data/{name}/ — gpu_worker.py would exit "
        f"'input not found'"
    )


@pytest.mark.parametrize("job_id", _job_ids())
def test_ligand_is_the_complete_molecule(job_id):
    """A truncated ligand still docks and still returns a number — silently wrong."""
    expected = EXPECTED_HEAVY_ATOMS.get(job_id)
    if expected is None:
        pytest.skip(f"no reference heavy-atom count for {job_id}")
    path = os.path.join(DATA_DIR, job_id, "ligand.pdbqt")
    assert os.path.isfile(path), f"missing ligand.pdbqt for {job_id}"
    got = _count_heavy_atoms(path)
    assert got == expected, (
        f"{job_id}: ligand.pdbqt has {got} heavy atoms, expected {expected}. "
        f"The structure is truncated or wrong; re-run prepare_example.sh"
    )


@pytest.mark.parametrize("job_id", _job_ids())
def test_box_config_is_smina_format(job_id):
    """smina --config accepts only 'key = value' lines; anything else aborts the run."""
    path = os.path.join(DATA_DIR, job_id, "box.txt")
    assert os.path.isfile(path), f"missing box.txt for {job_id}"
    with open(path) as fh:
        keys = set()
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            assert "=" in line, f"{job_id}: malformed box.txt line: {line!r}"
            keys.add(line.split("=")[0].strip())
    required = {"center_x", "center_y", "center_z", "size_x", "size_y", "size_z"}
    assert required <= keys, f"{job_id}: box.txt missing {sorted(required - keys)}"


# ── 3D geometry ───────────────────────────────────────────────────────────────
#
# Open Babel's --gen3d silently emitted flat 2D coordinates for lopinavir while
# exiting 0.  A flat conformer has no stereochemistry: it docks to a meaningless
# score, and OpenFF rejects it outright, so GPU jobs die after allocation.
# Verifying the input SMILES does not catch this -- the written file must be
# checked.

MIN_Z_RANGE_A = 1.0


def _z_range_sdf(path):
    with open(path) as fh:
        lines = fh.read().splitlines()
    n_atoms = int(lines[3][:3])
    zs = [float(l[20:30]) for l in lines[4:4 + n_atoms]]
    return max(zs) - min(zs)


def _z_range_pdbqt(path):
    zs = []
    with open(path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                zs.append(float(line[46:54]))
    return max(zs) - min(zs) if zs else 0.0


@pytest.mark.parametrize("job_id", _job_ids())
def test_ligand_sdf_has_real_3d_coordinates(job_id):
    path = os.path.join(DATA_DIR, job_id, "ligand.sdf")
    assert os.path.isfile(path), f"missing ligand.sdf for {job_id}"
    z = _z_range_sdf(path)
    assert z >= MIN_Z_RANGE_A, (
        f"{job_id}: ligand.sdf spans only {z:.2f} A in z — the conformer is flat, "
        f"so its stereochemistry is undefined and OpenFF will reject it"
    )


@pytest.mark.parametrize("job_id", _job_ids())
def test_ligand_pdbqt_has_real_3d_coordinates(job_id):
    path = os.path.join(DATA_DIR, job_id, "ligand.pdbqt")
    assert os.path.isfile(path), f"missing ligand.pdbqt for {job_id}"
    z = _z_range_pdbqt(path)
    assert z >= MIN_Z_RANGE_A, (
        f"{job_id}: ligand.pdbqt spans only {z:.2f} A in z — docking a flat "
        f"conformer of a chiral drug returns a meaningless affinity"
    )


# Published InChIKeys (connectivity + stereochemistry) for the example ligands.
EXPECTED_INCHIKEY = {
    "1hsg__indinavir": "CBVCZFGXHXORBI-PXQQMZJSSA-N",
    "1hsg__ritonavir": "NCDNCNXCDXHOMX-XGKFQTDJSA-N",
    "1hsg__nelfinavir": "QAGYKUNXZHXKMR-HKWSIXNMSA-N",
    "1hsg__lopinavir": "KJHKTHWMRKYKJE-SUGCFTRWSA-N",
}


@pytest.mark.parametrize("job_id", _job_ids())
def test_ligand_stereochemistry_matches_published_structure(job_id):
    """The strongest check: re-derive the InChIKey from the 3D coordinates."""
    rdkit = pytest.importorskip("rdkit", reason="RDKit not installed")
    from rdkit import Chem

    expected = EXPECTED_INCHIKEY.get(job_id)
    if expected is None:
        pytest.skip(f"no reference InChIKey for {job_id}")
    path = os.path.join(DATA_DIR, job_id, "ligand.sdf")
    mol = Chem.MolFromMolFile(path, removeHs=False)
    assert mol is not None, f"{job_id}: ligand.sdf could not be parsed"
    Chem.AssignStereochemistryFrom3D(mol)
    got = Chem.inchi.MolToInchiKey(mol)
    assert got == expected, (
        f"{job_id}: conformer is not the published structure.\n"
        f"  expected {expected}\n  got      {got}\n"
        f"A differing stereochemistry block means the 3D geometry is wrong or "
        f"undefined; re-run prepare_example.sh"
    )
