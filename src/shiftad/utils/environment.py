"""Privacy-conscious environment and source-state capture."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

DEFAULT_PACKAGES = (
    "shiftad",
    "torch",
    "torchvision",
    "numpy",
    "Pillow",
    "PyYAML",
    "scikit-learn",
    "matplotlib",
    "onnx",
    "onnxruntime",
)


def _package_versions(packages: Sequence[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _git_metadata(repo_root: Path | None) -> dict[str, Any]:
    if repo_root is None:
        return {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": bool(status.strip())}


def collect_environment(
    *,
    repo_root: Path | None = None,
    command: Sequence[str] | None = None,
    packages: Sequence[str] = DEFAULT_PACKAGES,
) -> dict[str, Any]:
    """Collect reproducibility metadata without personal host identifiers.

    Args:
        repo_root: Optional repository used for Git revision capture.
        command: Optional explicit command tokens to record.
        packages: Distribution names whose installed versions are recorded.

    Returns:
        A JSON-serializable environment mapping.

    Assumptions:
        Usernames, hostnames, environment variables, home paths, and Python
        executable paths are intentionally excluded.
    """

    import torch

    gpu_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "captured_at_utc": datetime.now(tz=UTC).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "cpu": {
            "description": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
            "logical_count": os.cpu_count(),
        },
        "gpu": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "count": torch.cuda.device_count(),
            "devices": gpu_names,
        },
        "packages": _package_versions(packages),
        "git": _git_metadata(repo_root),
        "command": list(command) if command is not None else None,
    }


def write_environment(snapshot: dict[str, Any], path: Path) -> None:
    """Write environment metadata as indented UTF-8 JSON.

    Args:
        snapshot: Output from :func:`collect_environment`.
        path: Destination JSON path. Parent directories are created.

    Raises:
        OSError: If the destination cannot be created or written.
        TypeError: If the snapshot is not JSON serializable.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
