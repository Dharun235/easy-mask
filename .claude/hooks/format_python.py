"""Claude Code PostToolUse hook: auto-fix and format any Python file Claude just edited.

Reads the hook payload (JSON) from stdin, runs `ruff check --fix` and `ruff format` on the
edited file, and never blocks Claude: lint errors that can't be auto-fixed are printed
to stderr for Claude to see, and the exit code stays 0.
"""

import json
import subprocess
import sys

payload = json.load(sys.stdin)
path = (payload.get("tool_input") or {}).get("file_path", "")
if path.endswith(".py"):
    subprocess.run(["uv", "run", "--quiet", "ruff", "check", "--fix", "--quiet", path], check=False)
    subprocess.run(["uv", "run", "--quiet", "ruff", "format", "--quiet", path], check=False)
