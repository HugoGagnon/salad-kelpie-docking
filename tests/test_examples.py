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
