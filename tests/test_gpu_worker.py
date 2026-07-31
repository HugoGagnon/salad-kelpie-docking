"""Tests for GPU worker preflight and terminal failure artifacts."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "worker"))
import gpu_worker


def test_invalid_integer_environment_value_is_rejected(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_PS", "not-an-integer")
    with pytest.raises(ValueError, match="CHECKPOINT_PS must be an integer"):
        gpu_worker._env_int("CHECKPOINT_PS", 500, minimum=1)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_float_environment_value_is_rejected(monkeypatch, value):
    monkeypatch.setenv("SKIP_NS", value)
    with pytest.raises(ValueError, match="SKIP_NS must be finite"):
        gpu_worker._env_float("SKIP_NS", 2.0, minimum=0.0)


def test_missing_baked_input_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="input not found"):
        gpu_worker._find_input(str(tmp_path), "target__ligand", "ligand.sdf")


def test_record_failure_writes_terminal_artifact(tmp_path):
    out_subdir = tmp_path / "target__ligand"
    with pytest.raises(SystemExit) as exc:
        gpu_worker._record_failure(
            out_subdir, replica=2, stage="preflight", reason="bad input"
        )
    assert exc.value.code == 0
    payload = json.loads((out_subdir / "rep2_mmgbsa.json").read_text())
    assert payload == {
        "error": "bad input",
        "stage": "preflight",
        "replica": 2,
    }
