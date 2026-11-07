"""Architecture tests: keep the layering in docs/architecture.md true as the code grows.

``core`` and ``io`` must never import Qt or ``gui``; ``core`` must not import ``io``.
Violations make logic untestable without a display, so we fail the build instead.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[2] / "src" / "easy_mask"

FORBIDDEN = {
    "core": ("PySide6", "easy_mask.gui", "easy_mask.io"),
    "io": ("PySide6", "easy_mask.gui"),
}


def imported_modules(path: Path) -> set[str]:
    """Every module name imported by a Python file (absolute imports only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize("layer", sorted(FORBIDDEN))
def test_layer_does_not_import_forbidden_modules(layer):
    violations = []
    for path in sorted((PACKAGE / layer).rglob("*.py")):
        for module in imported_modules(path):
            if module.startswith(FORBIDDEN[layer]):
                violations.append(f"{path.relative_to(PACKAGE)} imports {module}")
    assert violations == []


def test_no_relative_imports():
    """Absolute imports only: easier to read and to grep."""
    offenders = [
        str(path.relative_to(PACKAGE))
        for path in PACKAGE.rglob("*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert offenders == []


def test_headless_export_never_loads_qt():
    """`easy-mask export` must work on servers without Qt libraries."""
    code = "import sys, easy_mask.cli, easy_mask.io.export;sys.exit('PySide6' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
