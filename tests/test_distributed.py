import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ascendep.launch import run_local


def test_real_two_process_training():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=str(root))
    try:
        result = run_local([str(root / "tests/distributed_worker.py")], cwd=root, env=env, capture_output=True)
    except subprocess.CalledProcessError as error:
        pytest.fail((error.stdout or "") + (error.stderr or ""))
    line = next(line for line in result.stdout.splitlines() if line.startswith("VALIDATION="))
    results = json.loads(line.split("=", 1)[1])
    assert len(results) == 14
    assert all(row["status"] == "pass" and row["steps"] == 2 for row in results.values())
