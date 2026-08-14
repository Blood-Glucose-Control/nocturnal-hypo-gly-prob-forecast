"""Shared hardware detection helpers for workflow orchestration."""

from __future__ import annotations

import logging
from typing import Any, Dict


def get_gpu_info(logger: logging.Logger | None = None) -> Dict[str, Any]:
    """Return CUDA availability metadata used by workflow runtime decisions."""
    info: Dict[str, Any] = {"gpu_available": False, "gpu_count": 0}

    try:
        import torch
    except ImportError:
        if logger is not None:
            logger.warning("PyTorch is not installed; defaulting to CPU execution.")
        return info

    if not torch.cuda.is_available():
        return info

    gpu_count = torch.cuda.device_count()
    info["gpu_available"] = True
    info["gpu_count"] = gpu_count
    info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    return info


def clear_cuda_cache(logger: logging.Logger | None = None, *, context: str) -> None:
    """Clear CUDA allocator cache when CUDA is available."""
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if logger is not None:
            logger.info(f"GPU memory cleared ({context})")
