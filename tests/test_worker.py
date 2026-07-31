"""Tests for the CPU worker — argument validation and result structure.  No network calls."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "worker"))


# ── require_positive_int ───────────────────────────────────────────────────────

def test_require_positive_int_valid(monkeypatch):
    monkeypatch.setenv("CPU_THREADS", "4")
    import cpu_worker
    assert cpu_worker.require_positive_int("CPU_THREADS", "2") == 4


def test_require_positive_int_default(monkeypatch):
    monkeypatch.delenv("CPU_THREADS", raising=False)
    import cpu_worker
    assert cpu_worker.require_positive_int("CPU_THREADS", "2") == 2


def test_require_positive_int_rejects_zero(monkeypatch):
    monkeypatch.setenv("CPU_THREADS", "0")
    import cpu_worker
    with pytest.raises(SystemExit, match="positive integer"):
        cpu_worker.require_positive_int("CPU_THREADS", "2")


def test_require_positive_int_rejects_negative(monkeypatch):
    monkeypatch.setenv("EXHAUSTIVENESS", "-1")
    import cpu_worker
    with pytest.raises(SystemExit, match="positive integer"):
        cpu_worker.require_positive_int("EXHAUSTIVENESS", "16")


def test_require_positive_int_rejects_non_numeric(monkeypatch):
    monkeypatch.setenv("SEED", "abc")
    import cpu_worker
    with pytest.raises(SystemExit, match="positive integer"):
        cpu_worker.require_positive_int("SEED", "42")


# ── parse_best_affinity ────────────────────────────────────────────────────────

def test_parse_best_affinity_from_pdbqt(tmp_path):
    import cpu_worker
    pdbqt = tmp_path / "poses.pdbqt"
    pdbqt.write_text(
        "REMARK VINA RESULT:     -8.5      0.000      0.000\n"
        "REMARK minimizedAffinity   -8.5\n"
        "ATOM      1  C   LIG A   1       0.000   0.000   0.000\n"
    )
    val = cpu_worker._parse_best_affinity(str(pdbqt))
    assert val == pytest.approx(-8.5)


def test_parse_best_affinity_missing_file():
    import cpu_worker
    assert cpu_worker._parse_best_affinity("/nonexistent/poses.pdbqt") is None


def test_parse_best_affinity_no_affinity_line(tmp_path):
    import cpu_worker
    pdbqt = tmp_path / "poses.pdbqt"
    pdbqt.write_text("ATOM      1  C   LIG A   1       0.000   0.000   0.000\n")
    assert cpu_worker._parse_best_affinity(str(pdbqt)) is None


# ── result.json structure ──────────────────────────────────────────────────────

def test_result_json_contains_required_fields(tmp_path, monkeypatch):
    """Full integration test using a fake smina that exits 0 and writes an empty PDBQT."""
    import cpu_worker

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    (input_dir / "receptor.pdbqt").write_text("ATOM\n")
    (input_dir / "ligand.pdbqt").write_text("ATOM\n")
    (input_dir / "box.txt").write_text("center_x = 0\ncenter_y = 0\ncenter_z = 0\nsize_x = 20\n")

    monkeypatch.setenv("JOB_ID", "test-job")
    monkeypatch.setenv("CPU_THREADS", "1")
    monkeypatch.setenv("EXHAUSTIVENESS", "4")
    monkeypatch.setenv("NUM_MODES", "1")
    monkeypatch.setenv("SEED", "42")

    def fake_popen(cmd, **kwargs):
        poses_path = Path(cmd[cmd.index("--out") + 1])
        poses_path.write_text("REMARK minimizedAffinity   -8.5\n")
        mock = MagicMock()
        mock.__enter__ = lambda s: s
        mock.__exit__ = MagicMock(return_value=False)
        mock.stdout = iter(["smina output\n"])
        mock.wait.return_value = None
        mock.returncode = 0
        return mock

    with patch.object(cpu_worker, "INPUT_DIR", str(input_dir)), \
         patch.object(cpu_worker, "OUTPUT_DIR", str(output_dir)), \
         patch("subprocess.Popen", side_effect=fake_popen):
        cpu_worker.main()

    result_path = output_dir / "result.json"
    assert result_path.exists(), "result.json must be written"
    result = json.loads(result_path.read_text())
    for field in ("job_id", "start", "finish", "exit_code", "command", "params"):
        assert field in result, f"result.json missing field: {field}"
    assert result["job_id"] == "test-job"
    assert result["exit_code"] == 0
    assert result["best_affinity_kcal_mol"] == pytest.approx(-8.5)


def test_missing_input_writes_terminal_failure_artifact(tmp_path, monkeypatch):
    import cpu_worker

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    monkeypatch.setenv("JOB_ID", "missing-input-job")

    with patch.object(cpu_worker, "INPUT_DIR", str(input_dir)), \
         patch.object(cpu_worker, "OUTPUT_DIR", str(output_dir)):
        cpu_worker.main()

    result = json.loads((output_dir / "result.json").read_text())
    assert result["exit_code"] != 0
    assert "required input not found" in result["error"]


def test_clear_directory_contents_preserves_root(tmp_path):
    import cpu_worker

    (tmp_path / "old.txt").write_text("stale")
    nested = tmp_path / "old-dir"
    nested.mkdir()
    (nested / "old.txt").write_text("stale")

    cpu_worker._clear_directory_contents(str(tmp_path))

    assert tmp_path.is_dir()
    assert list(tmp_path.iterdir()) == []
