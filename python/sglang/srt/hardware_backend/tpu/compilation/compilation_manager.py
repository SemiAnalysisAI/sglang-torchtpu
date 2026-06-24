"""In-process TPU precompilation to warm up the PJRT/HLO compilation cache.

Target-model modes (compiled when ``is_draft_worker=False``):

* EXTEND: configured token-count and batch-size bucket combinations.
* DECODE: configured batch-size buckets (tokens == batch_size).
* TARGET_VERIFY: batch-size buckets × draft_token_num (speculative decoding).
* EXTEND(FULL): same grid as EXTEND but with CaptureHiddenMode.FULL so the
  Dynamo graph that captures all hidden states is pre-compiled.

Draft-model modes (compiled when ``is_draft_worker=True`` and spec decode is
enabled):

* DRAFT_EXTEND: variable-length "absorb accepted tokens" pass that runs once
  per verify step. Compiled at the same fixed batch cap as the target's
  EXTEND, with ``tokens_per_req = draft_token_num`` so the worst-case query
  total is covered. The runtime ``pad_draft_extend_inputs`` ensures incoming
  batches match this shape.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Optional

import torch
import tqdm

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)

if TYPE_CHECKING:
    from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = get_logger(__name__)

# Pure-text progress format (no animated bar) so log files and
# multi-process wrappers don't garble the output.
_COMPILE_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}% Completed | "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)


class CompilationManager:
    """Precompiles model forward paths across token and batch-size buckets."""

    # The libtpu HLO cache is process-global; multiple CompilationManager
    # instances in the same process (target worker + draft worker) all see
    # the same ``torch.tpu._get_cache_stats()``. Sharing the baseline across
    # all managers prevents the same compile from being reported by every
    # manager that runs a forward after it.
    #
    # We track two baselines:
    #   * ``_shared_baseline_misses``: ``num_cache_reqs - num_cache_hits`` at
    #     last check. The delta on each call is the number of true cache
    #     misses (= real HLO compiles) since last check.
    #   * ``_shared_baseline_graph_count``: ``len(per_entry_stats)`` at last
    #     check. Used only to slice ``per_entry_stats`` so we can report the
    #     per-compile durations of the new entries.
    _shared_baseline_misses: int = -1
    _shared_baseline_graph_count: int = -1

    def __init__(
        self,
        model_runner: ModelRunner,
        compiled_fns: dict[str, Callable],
        t_bucketer: Bucketer,
        b_bucketer: Bucketer,
        speculative_num_steps: int = 0,
        speculative_algorithm: Optional[str] = None,
        is_draft_worker: bool = False,
        speculative_eagle_topk: int = 1,
    ):
        self.model_runner = model_runner
        self.compiled_fns = compiled_fns
        self.t_bucketer = t_bucketer
        self.b_bucketer = b_bucketer
        self.speculative_num_steps = speculative_num_steps
        self.speculative_algorithm = speculative_algorithm
        self.is_draft_worker = is_draft_worker
        self.speculative_eagle_topk = speculative_eagle_topk

        if speculative_algorithm and speculative_algorithm != "EAGLE3":
            raise ValueError(
                f"TPU torch.compile precompilation only supports EAGLE3, "
                f"got {speculative_algorithm!r}."
            )

        self.draft_token_num = speculative_num_steps + 1 if speculative_num_steps else 0

        # Width of ``spec_info.hidden_states`` the EAGLE3 draft model expects.
        # Inferred from ``model.model.fc.in_features`` (= target_hidden_size *
        # num_aux). 0 for non-draft workers.
        self.eagle_hidden_size = self._infer_eagle_hidden_size()

        # TODO(wyzhang): support ``forward_split_prefill`` on TPU. Its
        # signature includes a ``(split_index, next_split_index)`` tuple
        # that Dynamo specializes per value, so precompile needs an
        # inner layer-range loop here. Until that exists, the
        # split-prefill entry point cold-compiles on first invocation
        # if the scheduler ever routes a request through it.
        if hasattr(self.model_runner.model, "forward_split_prefill"):
            logger.warning(
                "[compilation manager] %s exposes forward_split_prefill "
                "but this path is not precompiled yet. Requests routed "
                "through split-prefill will cold-compile on first use.",
                type(self.model_runner.model).__name__,
            )

    def _infer_eagle_hidden_size(self) -> int:
        """Derive the hidden-state width the EAGLE3 draft model's forward expects.

        EAGLE3 models expose an ``fc`` layer whose ``in_features`` is
        ``target_hidden_size * num_aux_hidden_states``.  Fall back to
        ``model_config.hidden_size`` if the ``fc`` layer is absent.
        Returns 0 for non-draft workers (the target model never reads
        ``spec_info.hidden_states``).
        """
        if not self.is_draft_worker:
            return 0
        model = self.model_runner.model
        # EAGLE3: ``model.model.fc`` is the projection layer.
        fc = getattr(getattr(model, "model", None), "fc", None)
        if fc is not None and hasattr(fc, "in_features"):
            return fc.in_features
        # Fallback: use the model's hidden_size.
        return getattr(self.model_runner.model_config, "hidden_size", 0) or 0

    @property
    def enable_spec_decoding(self) -> bool:
        return self.speculative_num_steps > 0

    def precompile_all(self) -> None:
        """Run precompilation for all valid extend and decode bucket shapes.

        For the **target** worker, compiles EXTEND / DECODE plus the
        EAGLE3-specific TARGET_VERIFY and EXTEND(FULL) modes.

        For the **draft** worker (EAGLE3), the draft model runs in three
        distinct modes that all need precompilation:

        * ``EXTEND``: initial ``forward_draft_extend`` runs once per request
          right after the target's prefill, with ``tokens_per_req = 1`` and
          a target-hidden_states feed.
        * ``DECODE``: ``draft_forward`` unrolls ``speculative_num_steps - 1``
          decode steps; one token per draft batch row each step.
        * ``DRAFT_EXTEND``: ``forward_draft_extend_after_decode`` absorbs the
          accepted prefix after each verify; query total is the per-bs
          worst case ``bs * draft_token_num``.
        """
        overall = time.perf_counter()
        if self.is_draft_worker:
            if self.enable_spec_decoding:
                self._precompile_extend()
                self._precompile_decode()
                self._precompile_draft_extend()
            else:
                logger.info(
                    "[compilation manager] DRAFT: nothing to precompile "
                    "(spec decode disabled)."
                )
        else:
            self._precompile_extend()
            self._precompile_decode()
            if self.enable_spec_decoding:
                self._precompile_target_verify()
                self._precompile_extend_full()
        logger.info(
            "[compilation manager] precompile_all done in %.1fs.",
            time.perf_counter() - overall,
        )

        # Snapshot the XLA cache state so serving-time recompilation can be
        # detected. The scheduler calls check_recompilation() periodically.
        # Stored on the class so the target and draft managers share a single
        # baseline (the libtpu cache is process-global).
        CompilationManager._shared_baseline_graph_count = self._get_xla_graph_count()
        reqs, hits = self._get_xla_cache_perf()
        CompilationManager._shared_baseline_misses = (
            max(reqs - hits, 0) if reqs >= 0 and hits >= 0 else -1
        )
        logger.info(
            "[compilation manager] XLA cache baseline: %d entries, %d misses",
            CompilationManager._shared_baseline_graph_count,
            CompilationManager._shared_baseline_misses,
        )

        # Optional: enable libtpu's "dump StableHLO on cache miss" so each
        # post-readiness compile prints its module name, op signature, and
        # source location. Enabled by SGLANG_TPU_DUMP_HLO_ON_MISS=1; off by
        # default because the dump volume is large.
        import os

        if os.environ.get("SGLANG_TPU_DUMP_HLO_ON_MISS") == "1":
            try:
                torch.tpu._set_dump_on_cache_miss(True)
                logger.warning(
                    "[compilation manager] enabled libtpu StableHLO dump-on-"
                    "cache-miss. Each post-readiness recompile will print "
                    "its HLO module and source location."
                )
            except (AttributeError, RuntimeError) as exc:
                logger.warning(
                    "[compilation manager] could not enable dump-on-cache-miss"
                    " (this torch_tpu build may not support it): %s",
                    exc,
                )

    @staticmethod
    def _get_xla_graph_count() -> int:
        """Return the number of compiled XLA programs in the cache."""
        try:
            stats = torch.tpu._get_cache_stats()
            return len(stats.per_entry_stats)
        except (AttributeError, RuntimeError):
            return -1

    @staticmethod
    def _get_xla_cache_entries():
        """Return the full per-entry stats list (compilation_duration, etc.)."""
        try:
            return torch.tpu._get_cache_stats().per_entry_stats
        except (AttributeError, RuntimeError):
            return None

    @staticmethod
    def _get_xla_cache_perf():
        """Return (num_cache_reqs, num_cache_hits) for the libtpu cache."""
        try:
            stats = torch.tpu._get_cache_stats()
            return int(stats.num_cache_reqs), int(stats.num_cache_hits)
        except (AttributeError, RuntimeError):
            return -1, -1

    def check_recompilation(self, context: str = "") -> int:
        """Check if new XLA programs were compiled since precompilation.

        Args:
            context: human-readable description of the current forward
                (e.g. "target LlamaForCausalLM mode=DECODE") for the warning.

        Returns the number of new compilations (0 = no recompilation).
        """
        baseline_misses = CompilationManager._shared_baseline_misses
        baseline_entries = CompilationManager._shared_baseline_graph_count
        if baseline_misses < 0 or baseline_entries < 0:
            return 0  # cache stats unavailable

        reqs, hits = self._get_xla_cache_perf()
        if reqs < 0 or hits < 0:
            return 0
        current_misses = max(reqs - hits, 0)
        new_misses = current_misses - baseline_misses
        if new_misses <= 0:
            # No real compile happened. Still update the entry-count baseline
            # so the per-entry slice stays bounded if entries get appended
            # for cache-hit-only requests.
            entries_now = self._get_xla_graph_count()
            if entries_now > baseline_entries:
                CompilationManager._shared_baseline_graph_count = entries_now
            return 0

        # Real compile(s) happened. Report durations of the new entries.
        entries = self._get_xla_cache_entries() or []
        current_entries = len(entries)
        new_entries = entries[baseline_entries:current_entries]
        durations_us: list[int] = []
        for e in new_entries:
            dur = getattr(e, "compilation_duration", None)
            if dur is None:
                continue
            try:
                us = dur.seconds * 1_000_000 + dur.microseconds
            except AttributeError:
                continue
            if us > 0:
                durations_us.append(us)
        # Advance the SHARED baselines.
        CompilationManager._shared_baseline_misses = current_misses
        CompilationManager._shared_baseline_graph_count = current_entries

        # Sort and show distribution so we can distinguish a single big
        # compile from many small ones.
        durations_us.sort()
        sample = (
            durations_us[:5]
            + (["..."] if len(durations_us) > 10 else [])
            + (durations_us[-5:] if len(durations_us) > 10 else [])
        )
        total_us = sum(durations_us)
        logger.warning(
            "[compilation manager] %d real XLA program(s) compiled at "
            "serving time [%s] (new_misses=%d, total_compile_ms=%.1f, "
            "new_entries=%d, cache_reqs=%d hits=%d miss_rate=%.2f%%, "
            "durations_us_sample=%s). This causes latency spikes.",
            new_misses,
            context,
            new_misses,
            total_us / 1000.0,
            len(new_entries),
            reqs,
            hits,
            100.0 * (reqs - hits) / max(reqs, 1),
            sample,
        )
        return new_misses

    def _precompile_extend(self) -> None:
        """Compile extend at each token bucket and the fixed batch cap.

        The runtime pad shim pads extend batch metadata to
        ``b_bucketer.max_val``. Keeping precompile on the same policy avoids
        the full token_bucket x batch_bucket compile grid while still covering
        the runtime extend shapes.
        """
        bs = self.b_bucketer.max_val
        valid_shapes = [(t, bs) for t in self.t_bucketer.buckets if t >= bs]
        if not valid_shapes:
            raise ValueError(
                "No TPU extend compile shapes are valid: token buckets "
                f"{list(self.t_bucketer.buckets)} are all smaller than the "
                f"extend batch cap {bs}."
            )
        n = len(valid_shapes)
        logger.info(
            "[compilation manager] EXTEND: %d shapes (bs=%d, token_buckets=%s)",
            n,
            bs,
            list(self.t_bucketer.buckets),
        )
        compile_range = (
            tqdm.tqdm(
                valid_shapes, desc="Compiling EXTEND", bar_format=_COMPILE_BAR_FORMAT
            )
            if get_tensor_model_parallel_rank() == 0
            else valid_shapes
        )
        for num_tokens, bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(num_tokens=num_tokens, bs=bs, mode=ForwardMode.EXTEND)
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled EXTEND (tokens={num_tokens}, bs={bs})"
                    f" in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_decode(self) -> None:
        """Compile decode mode: vary batch size (where tokens == batch_size)."""
        bs_buckets = list(self.b_bucketer.buckets)
        n = len(bs_buckets)
        logger.info(
            "[compilation manager] DECODE: %d shapes (bs=%s)",
            n,
            bs_buckets,
        )
        compile_range = (
            tqdm.tqdm(
                bs_buckets, desc="Compiling DECODE", bar_format=_COMPILE_BAR_FORMAT
            )
            if get_tensor_model_parallel_rank() == 0
            else bs_buckets
        )
        for bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(num_tokens=bs, bs=bs, mode=ForwardMode.DECODE)
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled DECODE (bs={bs}) in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_target_verify(self) -> None:
        """Compile TARGET_VERIFY at the fixed batch cap.

        Like extend, verify is padded to ``b_bucketer.max_val`` at runtime
        (non-decode branch of the pad shim), so only that one batch size needs
        precompiling -- smaller buckets re-pad to it and cache-hit.
        """
        draft_token_num = self.draft_token_num
        bs_buckets = [self.b_bucketer.max_val]
        n = len(bs_buckets)
        logger.info(
            "[compilation manager] TARGET_VERIFY: %d shapes "
            "(bs=%s, draft_token_num=%d)",
            n,
            bs_buckets,
            draft_token_num,
        )
        compile_range = (
            tqdm.tqdm(
                bs_buckets,
                desc="Compiling TARGET_VERIFY",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_tensor_model_parallel_rank() == 0
            else bs_buckets
        )
        for bs in compile_range:
            num_tokens = bs * draft_token_num
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.TARGET_VERIFY,
                # The EAGLE worker sets CaptureHiddenMode.FULL during
                # verify so the target model captures all hidden states
                # for the draft model.  Precompile with the same mode
                # so the Dynamo graph matches at runtime.
                capture_hidden_mode=CaptureHiddenMode.FULL,
                spec_info=self._make_verify_spec_info(),
            )
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled TARGET_VERIFY (bs={bs}, tokens={num_tokens})"
                    f" in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_extend_full(self) -> None:
        """Compile EXTEND with CaptureHiddenMode.FULL at the fixed batch cap."""
        bs = self.b_bucketer.max_val
        valid_shapes = [(t, bs) for t in self.t_bucketer.buckets if t >= bs]
        n = len(valid_shapes)
        logger.info(
            "[compilation manager] EXTEND(FULL): %d shapes (bs=%d, token_buckets=%s)",
            n,
            bs,
            list(self.t_bucketer.buckets),
        )
        compile_range = (
            tqdm.tqdm(
                valid_shapes,
                desc="Compiling EXTEND(FULL)",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_tensor_model_parallel_rank() == 0
            else valid_shapes
        )
        for num_tokens, bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.EXTEND,
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled EXTEND(FULL) (tokens={num_tokens}, bs={bs})"
                    f" in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_draft_extend(self) -> None:
        """Compile DRAFT_EXTEND once per bs bucket at the worst-case token count.

        For each ``bs`` in ``b_bucketer.buckets`` we compile a single shape:
        ``token_bucket = smallest_bucket >= bs * draft_token_num``. That is
        the upper bound for the per-step query total (sum of per-seq accept
        lengths, each in ``[1, draft_token_num]``). The runtime
        ``pad_draft_extend_inputs`` matches this policy: it pads batch to
        ``b_bucketer.next_bucket(real_b)`` and tokens to the worst case for
        that bs, so every (real_b, real_q) combination hits a precompiled
        shape.

        Falling back to the worst case per bs (rather than every (bs, t)
        cross product) keeps the startup compile budget bounded at
        ``|b_buckets|`` shapes. If profiling shows the worst-case pad
        costs significant latency for small accept totals, expand here
        to multiple token buckets per bs.
        """
        draft_token_num = self.draft_token_num
        max_token_bucket = self.t_bucketer.max_val
        shapes: list[tuple[int, int]] = []  # (num_tokens, bs)
        for bs in self.b_bucketer.buckets:
            worst_q = bs * draft_token_num
            # Token bucket must hold both the worst-case q and pad_b
            # (matches the extend pad policy in _pad_forward_batch_inputs).
            need = max(worst_q, bs)
            candidates = [t for t in self.t_bucketer.buckets if t >= need]
            token_bucket = candidates[0] if candidates else max_token_bucket
            shapes.append((token_bucket, bs))
        n = len(shapes)
        logger.info(
            "[compilation manager] DRAFT_EXTEND: %d shapes "
            "(per-bs worst case, draft_token_num=%d, shapes=%s)",
            n,
            draft_token_num,
            shapes,
        )
        compile_range = (
            tqdm.tqdm(
                shapes,
                desc="Compiling DRAFT_EXTEND",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_tensor_model_parallel_rank() == 0
            else shapes
        )
        for num_tokens, bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.DRAFT_EXTEND,
                # EAGLE3 draft-extend captures LAST hidden state (one row per
                # request) so the next draft-decode step can read it.
                capture_hidden_mode=CaptureHiddenMode.LAST,
                spec_info=self._make_draft_extend_spec_info(
                    num_tokens=num_tokens, bs=bs
                ),
            )
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled DRAFT_EXTEND (tokens={num_tokens}, bs={bs})"
                    f" in {elapsed:.1f}s{hbm_str}"
                )

    def _make_draft_extend_spec_info(self, num_tokens: int, bs: int):
        """Build an ``EagleDraftExtendInput`` stub for DRAFT_EXTEND precompile.

        The EAGLE3 draft model's forward reads ``spec_info.hidden_states`` and
        projects it through ``model.fc`` (and optionally ``fc_norm``), so the
        stub needs a real-shaped ``(num_tokens, eagle_hidden_size)`` tensor.
        Other fields are bs-shaped but zero-filled; the compiled graph only
        cares about shapes.
        """
        from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

        device = self.model_runner.device
        # Width matches what the draft model's fc layer expects (validated in
        # __init__ via _infer_eagle_hidden_size).
        hidden_size = self.eagle_hidden_size or 1
        dtype = self.model_runner.model_config.dtype
        return EagleDraftExtendInput(
            hidden_states=torch.zeros(
                (num_tokens, hidden_size), device=device, dtype=dtype
            ),
            num_correct_drafts=torch.zeros(bs, device=device, dtype=torch.int32),
            num_accept_tokens=torch.full(
                (bs,), self.draft_token_num, device=device, dtype=torch.int32
            ),
            num_accept_tokens_cpu=[self.draft_token_num] * bs,
            input_ids=torch.zeros(num_tokens, device=device, dtype=torch.long),
            seq_lens=torch.ones(bs, device=device, dtype=torch.int32),
            seq_lens_cpu=torch.ones(bs, dtype=torch.int32),
            req_pool_indices=torch.ones(bs, device=device, dtype=torch.int64),
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

    def _make_null_spec_info(self):
        """Build a minimal spec_info stub for non-spec forward modes.

        When speculative decoding is enabled, Dynamo traces both branches
        of ``if spec_info is not None`` guards in the model forward.
        This stub carries the attributes Dynamo probes without affecting
        the forward result.
        """
        if not self.enable_spec_decoding:
            return None
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        return EagleDraftInput.create_idle_input(
            device=self.model_runner.device,
            hidden_size=self.eagle_hidden_size or 1,
            dtype=torch.bfloat16,
            topk=self.speculative_eagle_topk,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    def _make_verify_spec_info(self):
        """Build an ``EagleVerifyInput`` for TARGET_VERIFY precompilation."""
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        return EagleVerifyInput.create_idle_input(
            topk=self.speculative_eagle_topk,
            spec_steps=self.speculative_num_steps,
            num_verify_tokens=self.draft_token_num,
            device=self.model_runner.device,
        )

    def _run_one(
        self,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        *,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: Optional[object] = None,
    ) -> None:
        """Build and execute one dummy forward pass for compilation."""
        backend = self.model_runner.attn_backend
        forward_batch = self._make_dummy_forward_batch(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=(
                spec_info if spec_info is not None else self._make_null_spec_info()
            ),
            attn_backend=backend,
        )
        # Set up req_to_token rows 1..bs with valid page indices for this
        # model's KV pool. The dummy batch uses
        # ``req_pool_indices=arange(1, bs+1)`` and varied ``out_cache_loc``
        # values; the attention metadata will gather page entries from each
        # of these rows, so they all need valid (non-OOB) slot indices.
        # Filling with KV slot 1 keeps the data simple but valid: slot 1 is
        # in-range for both target and draft KV pools.
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        req_to_token[1 : bs + 1, :num_tokens] = 1

        model = self.model_runner.model
        if hasattr(model, "prepare_forward_batch"):
            model.prepare_forward_batch(forward_batch)
        backend.init_forward_metadata(forward_batch)

        # Match TpuGraphRunner's no_grad wrapper for Dynamo cache reuse.
        with torch.no_grad():
            out = model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )
            # Materialize lazy outputs and backend-marked side effects.
            _materialize_out(out, forward_batch)

    def _make_dummy_forward_batch(
        self,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: Optional[object] = None,
        attn_backend: Optional[object] = None,
    ) -> ForwardBatch:
        """Construct a dummy ForwardBatch at the target bucket shape."""
        device = self.model_runner.device
        if attn_backend is None:
            attn_backend = self.model_runner.attn_backend
        # TARGET_VERIFY is extend-like (is_extend() returns True) so it
        # needs the extend metadata fields populated.
        is_extend = mode.is_extend()

        # Token-level inputs. Use ``arange``-based values rather than zeros so
        # the precompiled HLO sees the same value-distribution patterns that
        # runtime produces. With constant zero/one tensors, libtpu's sub-graph
        # cache keys (which fingerprint on tensor content patterns, not just
        # shape) cluster on the all-equal case; switching to distinct values
        # better matches what real serving traffic produces and avoids
        # tier-1-miss-tier-2-hit churn on the first real-data request.
        input_ids = torch.arange(1, num_tokens + 1, dtype=torch.int64, device=device)
        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)

        # Use slot 1 (not 0) for out_cache_loc. Slot 0 is the permanent
        # padding slot shared between target and draft models. If the
        # target model's precompilation writes page indices to
        # req_to_token row 0 that are valid for the target's KV pool
        # but out-of-bounds for the draft's smaller KV pool, the draft's
        # attention kernel would crash. Using slot >= 1 avoids this conflict.
        # arange(1, ...) gives distinct slot indices matching real allocator
        # output rather than the constant-1 dummy.
        out_cache_loc = torch.arange(
            1, num_tokens + 1, dtype=torch.int64, device=device
        )

        # Batch-level inputs. Use distinct values matching the runtime
        # distribution (each request has its own seq_len and slot). The
        # actual values are still synthetic but no longer all-equal.
        # seq_lens: realistic mid-range values (>1) so attention metadata
        # builds with the same value patterns as real KV-occupied requests.
        seq_lens = torch.arange(2, bs + 2, dtype=torch.int64, device=device)
        seq_lens_cpu = seq_lens.cpu()
        # req_pool_indices: start at 1 (slot 0 reserved for padding).
        req_pool_indices = torch.arange(1, bs + 1, dtype=torch.int64, device=device)

        if is_extend:
            if mode.is_target_verify() and spec_info is not None:
                tokens_per_req = spec_info.draft_token_num
            elif mode.is_draft_extend() and self.draft_token_num:
                tokens_per_req = self.draft_token_num
            else:
                tokens_per_req = 1
            extend_seq_lens = torch.full(
                (bs,), tokens_per_req, dtype=torch.int32, device=device
            )
            extend_prefix_lens = torch.zeros(bs, dtype=torch.int32, device=device)
            extend_start_loc = torch.arange(
                0,
                bs * tokens_per_req,
                tokens_per_req,
                dtype=torch.int32,
                device=device,
            )
            extend_num_tokens = bs * tokens_per_req
            extend_seq_lens_cpu = [tokens_per_req] * bs
            extend_prefix_lens_cpu = [0] * bs
            extend_logprob_start_lens_cpu = [0] * bs
        else:
            extend_seq_lens = None
            extend_prefix_lens = None
            extend_start_loc = None
            extend_num_tokens = None
            extend_seq_lens_cpu = None
            extend_prefix_lens_cpu = None
            extend_logprob_start_lens_cpu = None

        return ForwardBatch(
            forward_mode=mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=bs,
            orig_seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            return_logprob=False,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_logprob_start_lens_cpu=extend_logprob_start_lens_cpu,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=attn_backend,
            # Model accesses ``capture_hidden_mode.need_capture()``; default
            # of None would raise AttributeError inside the compiled region.
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
            # At runtime, text-only requests produce mm_inputs=[None]*bs
            # (one None per request), NOT mm_inputs=None.  Dynamo guards
            # on ``mm_inputs is None`` so the dummy batch must match.
            mm_inputs=[None] * bs,
        )


def _materialize_out(out, forward_batch) -> None:
    """Sync returned tensors and backend-marked KV side effects."""
    tensors: list[torch.Tensor] = []

    next_token_logits = getattr(out, "next_token_logits", None)
    if isinstance(next_token_logits, torch.Tensor):
        tensors.append(next_token_logits)

    hidden_states = getattr(out, "hidden_states", None)
    if isinstance(hidden_states, torch.Tensor):
        tensors.append(hidden_states)

    if getattr(forward_batch.attn_backend, "requires_kv_sync_after_compile", False):
        pool = forward_batch.token_to_kv_pool
        for layer_id in range(pool.start_layer, pool.start_layer + pool.layer_num):
            tensors.append(pool.get_fused_kv_buffer(layer_id))

    if not tensors:
        return
    from torch_tpu._internal.sync.sync import synchronize as tpu_sync

    tpu_sync(tensors, wait=True)


def _get_hbm_usage_mb() -> Optional[tuple[float, float]]:
    """Return (used_mb, total_mb) for the current TPU device, or None."""
    try:
        info = torch.accelerator.get_memory_info(torch.device("tpu"))
        free = info["free"] if isinstance(info, dict) else info[0]
        total = info["total"] if isinstance(info, dict) else info[1]
        used = total - free
        return (used / 1024 / 1024, total / 1024 / 1024)
    except Exception:
        return None


def _hbm_delta_str(
    before: Optional[tuple[float, float]],
    after: Optional[tuple[float, float]],
) -> str:
    """Format HBM delta + total as a suffix string, or empty if unavailable."""
    if before is None or after is None:
        return " HBM: N/A"
    delta = after[0] - before[0]
    return f" HBM: +{delta:.0f} MB ({after[0]:.0f}/{after[1]:.0f} MB)"
