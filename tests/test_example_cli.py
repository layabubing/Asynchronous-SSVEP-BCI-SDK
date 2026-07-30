from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tests.test_detector import synthetic_ssvep


def test_external_npy_cli_emits_json_activation(tmp_path: Path) -> None:
    input_path = tmp_path / "external_eeg.npy"
    np.save(input_path, synthetic_ssvep(12.0, seconds=2.0))

    completed = subprocess.run(
        [
            sys.executable,
            "examples/process_npy.py",
            str(input_path),
            "--sampling-rate",
            "250",
            "--frequencies",
            "10",
            "12",
            "15",
            "--algorithm",
            "fbcca",
            "--chunk-samples",
            "37",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    messages = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(messages) == 1
    assert messages[0]["event"] == "activated"
    assert messages[0]["command_frequency"] == 12.0
