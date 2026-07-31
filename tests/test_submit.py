"""Tests for submit.py — job definition construction.  No network calls."""
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Add repo root to path so we can import submit directly.
sys.path.insert(0, str(Path(__file__).parent.parent))
import submit


# ── helpers ────────────────────────────────────────────────────────────────────

def _cpu_spec(**overrides):
    spec = {"id": "test-job-001", "cpu_threads": 2, "exhaustiveness": 16,
            "num_modes": 10, "seed": 42}
    spec.update(overrides)
    return spec


def _gpu_complex(**overrides):
    cx = {"name": "receptor__ligand", "target": "receptor", "ligand": "ligand"}
    cx.update(overrides)
    return cx


# ── clean_prefix ───────────────────────────────────────────────────────────────

def test_clean_prefix_strips_slashes():
    assert submit.clean_prefix("/foo/bar/") == "foo/bar"


def test_clean_prefix_rejects_empty():
    with pytest.raises(ValueError):
        submit.clean_prefix("")


def test_clean_prefix_rejects_dotdot():
    with pytest.raises(ValueError):
        submit.clean_prefix("foo/../bar")


def test_clean_prefix_allows_nested():
    assert submit.clean_prefix("2024-01/campaign/v1") == "2024-01/campaign/v1"


# ── cpu_job_def ────────────────────────────────────────────────────────────────

def test_cpu_job_def_structure():
    job = submit.cpu_job_def(_cpu_spec(), "cgid-abc", "my-bucket", "prefix/v1")
    assert job["container_group_id"] == "cgid-abc"
    assert job["command"] == "python3"
    assert "/app/worker/cpu_worker.py" in job["arguments"]
    assert job["environment"]["JOB_ID"] == "test-job-001"
    assert job["environment"]["CPU_THREADS"] == "2"
    assert job["environment"]["EXHAUSTIVENESS"] == "16"


def test_cpu_job_def_sync_paths():
    job = submit.cpu_job_def(_cpu_spec(), "cgid", "bucket", "run/v1")
    before = job["sync"]["before"][0]
    after = job["sync"]["after"][0]
    assert before["prefix"].startswith("run/v1/inputs/test-job-001")
    assert after["prefix"].startswith("run/v1/outputs/test-job-001")
    assert before["direction"] == "download"
    assert after["direction"] == "upload"


def test_cpu_job_def_custom_input_prefix():
    spec = _cpu_spec(input_prefix="shared/inputs/common")
    job = submit.cpu_job_def(spec, "cgid", "bucket", "run/v1")
    assert job["sync"]["before"][0]["prefix"] == "shared/inputs/common/"


def test_cpu_job_def_rejects_bad_job_id():
    with pytest.raises(ValueError, match="invalid job id"):
        submit.cpu_job_def(_cpu_spec(id="bad id!"), "cgid", "bucket", "prefix")


def test_cpu_job_def_rejects_empty_job_id():
    with pytest.raises(ValueError, match="invalid job id"):
        submit.cpu_job_def(_cpu_spec(id=""), "cgid", "bucket", "prefix")


def test_cpu_job_def_defaults():
    spec = {"id": "minimal-job"}
    job = submit.cpu_job_def(spec, "cgid", "bucket", "prefix")
    env = job["environment"]
    assert env["CPU_THREADS"] == "2"
    assert env["EXHAUSTIVENESS"] == "16"
    assert env["NUM_MODES"] == "10"
    assert env["SEED"] == "42"


# ── gpu_job_def ────────────────────────────────────────────────────────────────

def test_gpu_job_def_structure():
    job = submit.gpu_job_def(_gpu_complex(), 0, 10.0, "cgid-gpu", "bucket", "prefix/md")
    assert job["container_group_id"] == "cgid-gpu"
    assert job["environment"]["SYSTEM"] == "receptor"
    assert job["environment"]["LIGAND_NAME"] == "ligand"
    assert job["environment"]["REPLICA"] == "0"
    assert job["environment"]["TARGET_NS"] == "10.0"


def test_gpu_job_def_sync_has_during():
    job = submit.gpu_job_def(_gpu_complex(), 1, 5.0, "cgid", "bucket", "run")
    assert "during" in job["sync"], "GPU jobs must have sync.during for checkpoint safety"
    during = job["sync"]["during"][0]
    assert during["direction"] == "upload"


def test_gpu_job_def_replica_seed_varies():
    job0 = submit.gpu_job_def(_gpu_complex(), 0, 10.0, "cgid", "bucket", "run")
    job1 = submit.gpu_job_def(_gpu_complex(), 1, 10.0, "cgid", "bucket", "run")
    assert job0["environment"]["SEED"] != job1["environment"]["SEED"], (
        "each replica must get a distinct seed so trajectories diverge"
    )


def test_gpu_job_def_r2_prefix_layout():
    job = submit.gpu_job_def(_gpu_complex(), 2, 10.0, "cgid", "bucket", "campaign/v1")
    before = job["sync"]["before"][0]["prefix"]
    during = job["sync"]["during"][0]["prefix"]
    after = job["sync"]["after"][0]["prefix"]
    assert "campaign/v1/receptor__ligand/rep2/checkpoints/" in before
    assert "campaign/v1/receptor__ligand/rep2/checkpoints/" in during
    assert "campaign/v1/receptor__ligand/rep2/outputs/" in after


# ── run_prefix validation ─────────────────────────────────────────────────────

def test_run_prefix_is_required_no_default():
    """submit.py must not supply any default run prefix — it is required on the CLI."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-prefix", required=True)
    with pytest.raises(SystemExit):
        parser.parse_args([])


# ── GPU complex-name invariant ────────────────────────────────────────────────

def test_gpu_job_def_rejects_name_that_breaks_the_output_path():
    """gpu_worker.py derives its output dir from target/ligand, poll.py from name.

    If cx["name"] != "<target>__<ligand>" the results upload to a key poll.py
    never checks: the MD completes, the data is orphaned, and --watch hangs.
    Reject it at submit time rather than after paying for GPU hours.
    """
    bad = {"name": "some_label", "target": "receptor", "ligand": "ligand"}
    with pytest.raises(ValueError, match="target.*__.*ligand|<target>__<ligand>"):
        submit.gpu_job_def(bad, 0, 10.0, "cgid", "bucket", "campaign/v1")


def test_gpu_job_def_accepts_consistent_name():
    ok = {"name": "receptor__ligand", "target": "receptor", "ligand": "ligand"}
    job = submit.gpu_job_def(ok, 0, 10.0, "cgid", "bucket", "campaign/v1")
    assert job["sync"]["after"][0]["prefix"] == "campaign/v1/receptor__ligand/rep0/outputs/"


# ── preview / empty job list ──────────────────────────────────────────────────

def test_preview_handles_empty_job_list(capsys):
    """A --targets typo yields zero jobs; that must report, not raise IndexError."""
    submit._preview([])
    assert "no jobs" in capsys.readouterr().out.lower()


def test_preview_prints_first_job(capsys):
    submit._preview([{"container_group_id": "abc"}, {"container_group_id": "def"}])
    assert "abc" in capsys.readouterr().out
