"""Tests for poll.py — polling logic and result interpretation.  No network calls."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import poll


# ── object_exists ─────────────────────────────────────────────────────────────

def test_object_exists_returns_true_on_success():
    client = MagicMock()
    client.head_object.return_value = {}
    assert poll.object_exists(client, "bucket", "some/key") is True


def test_object_exists_returns_false_on_404():
    from botocore.exceptions import ClientError
    client = MagicMock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )
    assert poll.object_exists(client, "bucket", "missing/key") is False


def test_object_exists_propagates_other_errors():
    from botocore.exceptions import ClientError
    client = MagicMock()
    client.head_object.side_effect = ClientError(
        {"Error": {"Code": "403", "Message": "Forbidden"}}, "HeadObject"
    )
    with pytest.raises(ClientError):
        poll.object_exists(client, "bucket", "forbidden/key")


# ── fetch_result ──────────────────────────────────────────────────────────────

def test_fetch_result_returns_dict_on_success():
    client = MagicMock()
    payload = {"exit_code": 0, "job_id": "test"}
    client.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps(payload).encode())}
    result = poll.fetch_result(client, "bucket", "key/result.json")
    assert result == payload


def test_fetch_result_returns_none_on_404():
    from botocore.exceptions import ClientError
    client = MagicMock()
    client.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": ""}}, "GetObject"
    )
    assert poll.fetch_result(client, "bucket", "missing.json") is None


# ── poll_cpu ──────────────────────────────────────────────────────────────────

def _make_manifest_file(tmp_path, jobs):
    p = tmp_path / "jobs.json"
    p.write_text(json.dumps({"jobs": jobs}))
    return str(p)


def test_poll_cpu_pending(tmp_path):
    manifest = _make_manifest_file(tmp_path, [{"id": "job-001"}])
    client = MagicMock()
    client.get_object.side_effect = __import__("botocore.exceptions", fromlist=["ClientError"]).ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": ""}}, "GetObject"
    )
    results = poll.poll_cpu(manifest, "run/v1", "bucket", client)
    assert results["job-001"]["status"] == "pending"


def test_poll_cpu_ok(tmp_path):
    manifest = _make_manifest_file(tmp_path, [{"id": "job-002"}])
    client = MagicMock()
    payload = {"exit_code": 0, "best_affinity_kcal_mol": -8.5}
    client.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps(payload).encode())}
    results = poll.poll_cpu(manifest, "run/v1", "bucket", client)
    assert results["job-002"]["status"] == "ok"
    assert results["job-002"]["affinity_kcal_mol"] == -8.5


def test_poll_cpu_failed(tmp_path):
    """A failed job must reach a terminal state, or --watch never exits.

    This depends on cpu_worker.py exiting zero after recording a failure, so
    that Kelpie's sync.after uploads the result.json carrying exit_code != 0.
    """
    manifest = _make_manifest_file(tmp_path, [{"id": "job-003"}])
    client = MagicMock()
    payload = {"exit_code": 1, "best_affinity_kcal_mol": None}
    client.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps(payload).encode())}
    results = poll.poll_cpu(manifest, "run/v1", "bucket", client)
    assert results["job-003"]["status"] == "failed"
    assert results["job-003"]["exit_code"] == 1
    # "failed" must be in the terminal set poll.py's --watch loop checks.
    assert results["job-003"]["status"] in {"ok", "failed"}


# ── poll_gpu ──────────────────────────────────────────────────────────────────

def _make_matrix_file(tmp_path, complexes):
    p = tmp_path / "matrix.json"
    p.write_text(json.dumps({"complexes": complexes}))
    return str(p)


def test_poll_gpu_pending(tmp_path):
    from botocore.exceptions import ClientError
    matrix = _make_matrix_file(tmp_path, [{"name": "target__ligand", "target": "target", "ligand": "ligand"}])
    client = MagicMock()
    client.get_object.side_effect = ClientError(
        {"Error": {"Code": "NoSuchKey", "Message": ""}}, "GetObject"
    )
    results = poll.poll_gpu(matrix, "run/v1", "bucket", client, n_reps=1)
    assert results["target__ligand/rep0"]["status"] == "pending"


def test_poll_gpu_ok(tmp_path):
    matrix = _make_matrix_file(tmp_path, [{"name": "target__ligand", "target": "target", "ligand": "ligand"}])
    payload = {"dG_mean": -6.2, "dG_sem": 0.4}

    call_count = [0]
    def side_effect(**kwargs):
        call_count[0] += 1
        return {"Body": MagicMock(read=lambda: json.dumps(payload).encode())}

    client = MagicMock()
    client.get_object.side_effect = side_effect
    results = poll.poll_gpu(matrix, "run/v1", "bucket", client, n_reps=1)
    assert results["target__ligand/rep0"]["status"] == "ok"
    assert results["target__ligand/rep0"]["dG_mean_kcal_mol"] == -6.2
