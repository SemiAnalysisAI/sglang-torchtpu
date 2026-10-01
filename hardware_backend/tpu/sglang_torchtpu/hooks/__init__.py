# SPDX-License-Identifier: Apache-2.0
"""General-plugin hook registration (`sglang.srt.plugins` entry-point group).

Deliberately empty. A hook is a monkeypatch keyed by dotted name: it breaks
silently when upstream renames the target, and the registry logs rather than
raises. Every hook added here is a tracked debt and must carry a header:

    Why:            the behaviour that cannot be expressed through a seam
    How:            BEFORE / AFTER / AROUND / REPLACE, and the dotted target
    Upstream:       the PR or issue that would make the hook unnecessary
    Remove when:    the condition under which this file is deleted

A REPLACE hook on a method longer than ~30 lines means "write the seam instead".
``tools/plugin/hooks_report.py`` counts what lives here; the count is a
release-blocking metric (<= 40 at Gate B, < 20 at cutover).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# (target, hook, hook_type) triples. Empty by design -- see module docstring.
HOOKS: list[tuple] = []


def register() -> None:
    """Entry point for ``sglang.srt.plugins``."""
    if not HOOKS:
        logger.debug("sglang_torchtpu: no hooks registered.")
        return

    from sglang.srt.plugins.hook_registry import HookRegistry

    for target, hook, hook_type in HOOKS:
        HookRegistry.register(target, hook, hook_type)
    logger.info("sglang_torchtpu: registered %d hook(s).", len(HOOKS))
