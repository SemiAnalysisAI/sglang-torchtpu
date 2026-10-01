# SPDX-License-Identifier: Apache-2.0
"""Out-of-tree TPU platform plugin for SGLang.

Wave 0 of the plugin migration: this package owns the platform *identity* and
is the home for all new TPU code. No existing TPU code has moved yet; see the
package README for the migration state.

Registered under the ``sglang.srt.platforms`` entry-point group as ``tpu_oot``.
"""

import importlib.util
import logging
import os

logger = logging.getLogger(__name__)

__all__ = ["activate", "ENTRY_POINT_NAME", "PLATFORM_CLASS_PATH"]

ENTRY_POINT_NAME = "tpu_oot"
PLATFORM_CLASS_PATH = "sglang_torchtpu.platform:TpuOOTPlatform"


def activate():
    """Return the OOT TPU platform class path, or ``None`` to decline.

    Opt-in by design. Declaring ``PlatformEnum.OOT`` is not behaviour-neutral:
    the fork's core tree has 27 ``is_out_of_tree()`` gates, and the ones that do
    not already read ``or is_tpu()`` change which branch they take. Two known
    divergences are still open (see README "Known divergences"), so until they
    are closed this platform activates only when asked for by name.

    SGLang's loader already front-loads the filter when ``SGLANG_PLATFORM`` is
    set, so the environment check here is belt-and-braces: it also keeps this
    entry point from competing with the in-tree ``tpu`` one during automatic
    discovery, which raises when two platforms activate.
    """
    if os.environ.get("SGLANG_PLATFORM") != ENTRY_POINT_NAME:
        return None

    try:
        torch_tpu_available = importlib.util.find_spec("torch_tpu") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        logger.debug("torch_tpu activation check failed.", exc_info=True)
        return None

    if not torch_tpu_available:
        logger.warning(
            "SGLANG_PLATFORM=%s was requested but torch_tpu is not importable.",
            ENTRY_POINT_NAME,
        )
        return None

    logger.info("TPU out-of-tree platform selected (%s).", PLATFORM_CLASS_PATH)
    return PLATFORM_CLASS_PATH
