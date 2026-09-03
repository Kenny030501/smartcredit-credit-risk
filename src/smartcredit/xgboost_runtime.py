"""Load XGBoost and its OpenMP runtime safely across platforms."""

from __future__ import annotations

import ctypes
import platform
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xgboost import XGBClassifier

_OPENMP_HANDLE: ctypes.CDLL | None = None


def ensure_openmp_runtime() -> Path | None:
    """Preload an available libomp on macOS to avoid XGBoost dynamic-link failures."""
    global _OPENMP_HANDLE
    if platform.system() != "Darwin" or _OPENMP_HANDLE is not None:
        return None

    candidates = [
        Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
        Path("/usr/local/opt/libomp/lib/libomp.dylib"),
        Path("/opt/anaconda3/lib/libomp.dylib"),
    ]
    for candidate in candidates:
        if candidate.exists():
            _OPENMP_HANDLE = ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return candidate

    raise RuntimeError(
        "macOS缺少XGBoost所需的libomp.dylib；请安装libomp或在Anaconda中安装llvm-openmp"
    )


def get_xgboost_classifier() -> type[XGBClassifier]:
    """Return XGBClassifier after ensuring native dependencies are available."""
    ensure_openmp_runtime()
    from xgboost import XGBClassifier

    return XGBClassifier
