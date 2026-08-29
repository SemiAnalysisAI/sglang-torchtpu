"""Standalone stacked RPA decode implementation.

The public entry point lives in ``wrapper``. Sliding-window and global decode
each own their schedule, buffered refs, and kernel implementation. Prefill is
intentionally supplied by the existing RPA v3.1 backend.
"""
