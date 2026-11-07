# Architecture

easy-mask is split into three layers with a one-way dependency: `gui -> io -> core`.

| Layer | Knows about | Must not import |
| --- | --- | --- |
| `core/` | numpy, Pillow, stdlib | Qt, `io`, `gui` |
| `io/` | `core`, numpy, Pillow | Qt, `gui` |
| `gui/` | everything | — |

Keeping Qt out of `core` and `io` means the annotation logic (geometry, file format,
rasterizing, export) is plain Python that runs and tests without a display.

The layer contents are filled in by the `feat/core-annotation` branch.
