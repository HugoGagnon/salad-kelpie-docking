"""Unit tests for MD duration math and durable progress metadata."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "worker"))
import md_engine


def test_ten_nanoseconds_is_two_point_five_million_steps():
    assert md_engine.steps_for_duration(10.0, md_engine.FS_PER_NS) == 2_500_000


def test_five_hundred_picoseconds_is_125000_steps():
    assert md_engine.steps_for_duration(500, md_engine.FS_PER_PS) == 125_000


@pytest.mark.parametrize("value", [0, -1])
def test_steps_for_duration_rejects_nonpositive_values(value):
    with pytest.raises(ValueError, match="positive"):
        md_engine.steps_for_duration(value, md_engine.FS_PER_NS)


def test_progress_uses_the_same_timestep_conversion(tmp_path):
    path = tmp_path / "progress.json"
    md_engine._write_progress(
        path,
        steps=2_500_000,
        target_steps=2_500_000,
        target_ns=10.0,
        checkpoint_steps=125_000,
        complete=True,
    )
    progress = json.loads(path.read_text())
    assert progress["ns"] == pytest.approx(10.0)
    assert progress["timestep_fs"] == 4.0
    assert progress["checkpoint_steps"] == 125_000
    assert progress["complete"] is True
