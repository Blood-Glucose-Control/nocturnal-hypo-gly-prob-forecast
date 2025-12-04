# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: cjrisi/christopher AT uwaterloo/gluroo DOT ca/com
"""Logging Helper Utilities

Provides standardized logging functions with function name context and distributed training awareness.

The debug_print function respects the DEBUG environment variable and checks it dynamically,
allowing for runtime control of debug output. This is especially useful for testing and
distributed training scenarios.

To run in debug mode, set the DEBUG environment variable:
DEBUG=1 python your_script.py
# or
export DEBUG=true
python your_script.py

Supported DEBUG values: "1", "true", "yes", "on" (case-insensitive)
"""

import inspect
import sys

# Import torch at module level to avoid repeated imports
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Global debug flag - checked dynamically to support runtime environment changes
import os


def _is_debug_enabled():
    """Check if debug mode is enabled via environment variable."""
    return os.environ.get("DEBUG", "").lower() in ("1", "true", "yes", "on")


def _get_caller_name():
    """Get the calling function name, filtering out common non-useful names."""
    frame = inspect.currentframe()
    try:
        # Go back two frames: _get_caller_name -> {info/error/debug}_print -> actual_caller
        if frame is not None and hasattr(frame, "f_back") and frame.f_back is not None:
            caller_frame = getattr(frame.f_back, "f_back", None)
            if caller_frame is not None and hasattr(caller_frame, "f_code"):
                caller_name = getattr(caller_frame.f_code, "co_name", None)
                # Filter out non-useful function names
                if caller_name and caller_name not in ["<module>", "wrapper", "main"]:
                    return f"[{caller_name}]"
    except (AttributeError, TypeError):
        pass  # Handle cases where frame is None or doesn't have expected attributes
    finally:
        if frame is not None:
            del frame
    return ""


def _should_print_on_rank():
    """Check if current process should print (only rank 0 in distributed settings)."""
    if not TORCH_AVAILABLE:
        # torch not available, always print
        return True
    # print(f'torch dist avail: {torch.distributed.is_available()}, \t torch dist init: {torch.distributed.is_initialized()}')
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            # In distributed training, only rank 0 should print
            rank = torch.distributed.get_rank()
            return rank == 0
        else:
            # Not in distributed mode, always print
            return True
    except Exception:
        # Any error in detection, always print (safer)
        return True


def info_print(*args, rank_zero_only=True, **kwargs):
    """
    Print informational messages to stderr with function name.

    Args:
        *args: Arguments to print
        rank_zero_only (bool): If True (default), only print on rank 0 in distributed settings.
                              If False, print on all ranks.
        **kwargs: Additional keyword arguments for print()
    """
    if rank_zero_only and not _should_print_on_rank():
        return

    caller = _get_caller_name()
    if caller:
        print("INFO:", caller, *args, file=sys.stderr, flush=True, **kwargs)
    else:
        print("INFO:", *args, file=sys.stderr, flush=True, **kwargs)


def error_print(*args, rank_zero_only=True, **kwargs):
    """
    Print error messages to stderr with function name.

    Args:
        *args: Arguments to print
        rank_zero_only (bool): If True (default), only print on rank 0 in distributed settings.
                              If False, print on all ranks (useful for debugging).
        **kwargs: Additional keyword arguments for print()
    """
    if rank_zero_only and not _should_print_on_rank():
        return

    caller = _get_caller_name()
    if caller:
        print("ERROR:", caller, *args, file=sys.stderr, flush=True, **kwargs)
    else:
        print("ERROR:", *args, file=sys.stderr, flush=True, **kwargs)


def debug_print(*args, rank_zero_only=True, **kwargs):
    """
    Print debug messages to stderr with function name.
    Only prints if DEBUG environment variable is set to true/1/yes/on.

    Args:
        *args: Arguments to print
        rank_zero_only (bool): If True (default), only print on rank 0 in distributed settings.
                              If False, print on all ranks (useful for debugging).
        **kwargs: Additional keyword arguments for print()
    """
    # Check debug flag first - if debug is disabled, don't print at all
    if not _is_debug_enabled():
        return

    if rank_zero_only and not _should_print_on_rank():
        return

    caller = _get_caller_name()
    if caller:
        print("DEBUG:", caller, *args, file=sys.stderr, flush=True, **kwargs)
    else:
        print("DEBUG:", *args, file=sys.stderr, flush=True, **kwargs)
