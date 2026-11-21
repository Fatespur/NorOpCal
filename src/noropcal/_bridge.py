"""Bridge to the frozen scientific implementation (delegation only).

This module NEVER reimplements NorOpCal mathematics. It locates the frozen
PB7-P1 implementation shipped verbatim under ``tools/`` and imports it, so
every public call delegates to the exact code that produced the paper numbers.

Frozen constants are read from the sealed specification artifact
``reports/pb7_p1_safety_actuator/E_FROZEN_METHOD_SPEC.json`` — never hard-coded
here and never recomputed.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


def _repo_root() -> Path:
    """Locate the repository root that carries the frozen implementation.

    Works for source checkouts, editable installs, and (when launched from
    inside a checkout) regular installs. Raises an actionable ImportError
    otherwise - the facade never reimplements the science, so it REQUIRES the
    real repository tree.
    """
    # 1) source-layout resolution: .../github/src/noropcal/_bridge.py
    candidate = Path(__file__).resolve().parents[2]
    if _is_repo_root(candidate):
        return candidate
    # 2) walk up from the current working directory
    here = Path.cwd().resolve()
    for c in [here, *here.parents][:5]:
        if _is_repo_root(c):
            return c
    raise ImportError(
        "noropcal facade could not locate the frozen scientific implementation "
        "(expected 'tools/pb7_p1_safety_actuator' and "
        "'reports/pb7_p1_safety_actuator/E_FROZEN_METHOD_SPEC.json'). "
        "Run from within a NorOpCal repository checkout."
    )


def _is_repo_root(candidate: Path) -> bool:
    return (
        (candidate / "reports" / "pb7_p1_safety_actuator" / "E_FROZEN_METHOD_SPEC.json").exists()
        and (candidate / "tools" / "pb7_p1_safety_actuator").is_dir()
    )


def frozen_spec_path() -> Path:
    return _repo_root() / "reports" / "pb7_p1_safety_actuator" / "E_FROZEN_METHOD_SPEC.json"


def load_frozen_spec() -> dict:
    """Load the sealed PB7-P1 method specification (single legal source)."""
    return json.loads(frozen_spec_path().read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def scientific_impl() -> ModuleType:
    """Import the frozen ``pb7_p1_common`` module (verbatim scientific code)."""
    root = _repo_root()
    tools = root / "tools"
    stage = tools / "pb7_p1_safety_actuator"
    p0 = tools / "pb7_p0_threshold_correctability"
    for p in (tools, p0, stage):
        if not p.is_dir():
            raise ImportError(
                f"frozen scientific implementation not found at {p}; "
                "run from a repository checkout or editable install"
            )
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
    import pb7_p1_common as impl  # type: ignore[import-not-found]  # noqa: E402

    return impl
