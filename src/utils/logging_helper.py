# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: cjrisi/christopher AT uwaterloo/gluroo DOT ca/com

import inspect
import sys


def _get_caller_name():
    """Get the calling function name, filtering out common non-useful names."""
    frame = inspect.currentframe()
    try:
        # Go back two frames: _get_caller_name -> {info/error/debug}_print -> actual_caller
        caller_frame = frame.f_back.f_back
        if caller_frame:
            caller_name = caller_frame.f_code.co_name
            # Filter out non-useful function names
            if caller_name not in ["<module>", "wrapper", "main"]:
                return f"[{caller_name}]"
    except (AttributeError, TypeError):
        pass  # Handle cases where frame is None or doesn't have expected attributes
    finally:
        del frame
    return ""


def info_print(*args, **kwargs):
    """Print informational messages to stderr with function name (so they show up in slurm error file)"""
    caller = _get_caller_name()
    if caller:
        print("INFO:", caller, *args, file=sys.stderr, flush=True, **kwargs)
    else:
        print("INFO:", *args, file=sys.stderr, flush=True, **kwargs)


def error_print(*args, **kwargs):
    """Print error messages to stderr with function name (so they show up in slurm error file)"""
    caller = _get_caller_name()
    if caller:
        print("ERROR:", caller, *args, file=sys.stderr, flush=True, **kwargs)
    else:
        print("ERROR:", *args, file=sys.stderr, flush=True, **kwargs)


def debug_print(*args, **kwargs):
    """Print debug messages to stderr with function name (so they show up in slurm error file)"""
    caller = _get_caller_name()
    if caller:
        print("DEBUG:", caller, *args, file=sys.stderr, flush=True, **kwargs)
    else:
        print("DEBUG:", *args, file=sys.stderr, flush=True, **kwargs)