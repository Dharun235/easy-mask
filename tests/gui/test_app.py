"""Application bootstrap: `run()` and `python -m easy_mask`, without blocking."""

import runpy

import pytest
from PySide6.QtWidgets import QApplication

from easy_mask.gui import app
from easy_mask.gui.main_window import MainWindow, Settings


def test_run_opens_path_and_returns_exit_code(qapp, image_dir, monkeypatch):
    opened = []
    monkeypatch.setattr(QApplication, "exec", lambda self: 0)
    monkeypatch.setattr(MainWindow, "open_path", lambda self, path: opened.append(path))
    assert app.run(image_dir, Settings()) == 0
    assert opened == [image_dir]
    for widget in QApplication.topLevelWidgets():
        widget.close()


def test_cli_starts_gui_with_parsed_settings(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(app, "run", lambda path, settings: seen.update(path=path, s=settings) or 0)
    from easy_mask.cli import main

    assert main([str(tmp_path), "--labels", "a,b", "--flags", "x", "--autosave"]) == 0
    assert seen["path"] == tmp_path
    assert (seen["s"].labels, seen["s"].flags, seen["s"].autosave) == (["a", "b"], ["x"], True)


def test_python_dash_m_entry_point(monkeypatch):
    monkeypatch.setattr("sys.argv", ["easy_mask", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("easy_mask", run_name="__main__")
    assert exc.value.code == 0
