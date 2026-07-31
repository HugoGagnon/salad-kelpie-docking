"""Tests for scorer atom partitioning that do not require OpenMM."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "worker"))
import mmgbsa


def test_partition_uses_build_time_counts_not_chain_count():
    metadata = {"receptor_atom_count": 1514, "ligand_atom_count": 45}
    receptor, ligand = mmgbsa.partition_indices(metadata, topology_atom_count=10_000)
    assert receptor == list(range(1514))
    assert ligand == list(range(1514, 1559))


def test_partition_rejects_counts_larger_than_topology():
    metadata = {"receptor_atom_count": 100, "ligand_atom_count": 20}
    with pytest.raises(ValueError, match="exceed"):
        mmgbsa.partition_indices(metadata, topology_atom_count=119)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"receptor_atom_count": 0, "ligand_atom_count": 10},
        {"receptor_atom_count": 10, "ligand_atom_count": -1},
    ],
)
def test_partition_rejects_invalid_metadata(metadata):
    with pytest.raises(ValueError):
        mmgbsa.partition_indices(metadata, topology_atom_count=100)

