"""Shared helpers for the manual NGram-on-TPU tests.

These tests are run directly (``python test/manual/tpu/test_ngram_*.py``), so the
script's directory is on ``sys.path`` and a plain ``from _ngram_tpu_common import
...`` resolves.
"""

# Target model for every NGram-on-TPU manual test (small, single-chip friendly).
MODEL = "Qwen/Qwen3-0.6B"


def skip_if_no_tpu() -> bool:
    """True when torch_tpu is not importable, so the test can be skipped."""
    try:
        import importlib.util

        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True
