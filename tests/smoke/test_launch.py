"""Smoke tests: start the real entry points in a fresh process, as a user would.

These catch packaging and import-order breakage that in-process tests can hide.
"""

import os
import subprocess
import sys
import time

import pytest

from easy_mask import __version__


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "easy_mask", *args], capture_output=True, text=True, timeout=60
    )


def test_version():
    result = run("--version")
    assert result.returncode == 0 and __version__ in result.stdout


def test_help_lists_options():
    assert "--labels" in run("--help").stdout
    assert "--format" in run("export", "--help").stdout


def test_bad_arguments_exit_nonzero():
    assert run("export", "/nonexistent", "--format", "nope", "--out", "x").returncode == 2


@pytest.mark.slow
def test_gui_starts_and_stays_up(image_dir):
    """Launch the window headless on a folder; it must still be running after 3 s."""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "easy_mask", str(image_dir)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(3)
        alive = proc.poll() is None
    finally:
        proc.terminate()
        _, stderr = proc.communicate(timeout=10)
    assert alive, f"app exited early:\n{stderr}"
    assert "Traceback" not in stderr
