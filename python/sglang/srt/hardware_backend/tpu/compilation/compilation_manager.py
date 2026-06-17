"""In-process TPU precompilation to warm up the PJRT/HLO compilation cache.

Iterates over shape buckets to compile:

Target model (``is_draft_worker=False``):
* EXTEND: configured token-count and batch-size bucket combinations.
* DECODE: configured batch-size buckets (tokens == batch_size).
* TARGET_VERIFY: batch-size buckets × draft_token_num (speculative decoding).
* EXTEND(FULL): same grid as EXTEND but with CaptureHiddenMode.FULL so the
  Dynamo graph that captures all hidden states is pre-compiled.

Draft model (``is_draft_worker=True``):
* EXTEND(LAST): same grid as EXTEND but with CaptureHiddenMode.LAST.
* DECODE(draft): batch-size buckets with tokens = bs × topk, spec_info
  stub attached so Dynamo sees the draft-decode guard.  Runs without
  ``prepare_forward_batch`` to match the runtime path that uses
  ``skip_attn_backend_init=True``.
* DRAFT_EXTEND: batch-size buckets with tokens =
  (speculative_num_steps + 1) × bs.  Also runs without
  ``prepare_forward_batch``.
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

    def __init__(
        self,
        model_runner: ModelRunner,
        compiled_fns: dict[str, Callable],
        t_bucketer: Bucketer,
        b_bucketer: Bucketer,
        *,
        speculative_num_steps: int = 0,
        speculative_algorithm: Optional[str] = None,
        is_draft_worker: bool = False,
        topk: int = 1,
    ):
        self.model_runner = model_runner
        self.compiled_fns = compiled_fns
        self.t_bucketer = t_bucketer
        self.b_bucketer = b_bucketer
        self.speculative_num_steps = speculative_num_steps
        self.speculative_algorithm = speculative_algorithm
        self.is_draft_worker = is_draft_worker
        self.topk = topk

        # draft_token_num is the number of tokens the target model
        # verifies per request in a single TARGET_VERIFY forward.
        # For EAGLE-family algorithms this equals speculative_num_steps + 1.
        self.draft_token_num = speculative_num_steps + 1 if speculative_num_steps else 0

        # spec_hidden_size is the width of ``spec_info.hidden_states``
        # that the draft model's forward expects.  For EAGLE3, this
        # equals ``fc.in_features`` (= target_hidden_size * num_aux).
        # Inferred from the model if not provided.
        self.spec_hidden_size = self._infer_spec_hidden_size()

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

    def _infer_spec_hidden_size(self) -> int:
        """Derive the hidden-state width the draft model's forward expects.

        EAGLE3 models expose an ``fc`` layer whose ``in_features`` is
        ``target_hidden_size * num_aux_hidden_states``.  Fall back to
        ``model_config.hidden_size`` if the ``fc`` layer is absent
        (e.g., non-EAGLE draft models).  Returns 0 for non-draft
        workers (the target model never reads ``spec_info.hidden_states``).
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
    def _is_speculative(self) -> bool:
        return self.speculative_num_steps > 0

    @property
    def _extend_width(self) -> int:
        """Hidden width for draft EXTEND(LAST) / DRAFT_EXTEND (aux concat)."""
        return self.spec_hidden_size

    @property
    def _decode_width(self) -> int:
        """Hidden width for the draft DECODE loop (draft self-chain output)."""

        return self.spec_hidden_size

    def precompile_all(self) -> None:
        """Run precompilation for all valid extend and decode bucket shapes."""
        overall = time.perf_counter()
        if self.is_draft_worker:
            self._precompile_draft_all()
        else:
            self._precompile_target_all()
        logger.info(
            "[compilation manager] precompile_all done in %.1fs.",
            time.perf_counter() - overall,
        )

    def _precompile_target_all(self) -> None:
        """Precompile shapes used by the target model."""
        self._precompile_extend()
        self._precompile_decode()
        if self._is_speculative:
            self._precompile_target_verify()
            self._precompile_extend_capture_hidden()

    def _precompile_draft_all(self) -> None:
        """Precompile shapes used by the draft model.

        A draft model only exists when speculative decoding is enabled.
        If ``_is_speculative`` is ``False``, this is a no-op — there is
        no draft model to precompile.

        The draft model sees three distinct forward paths:

        1. **EXTEND(LAST)** — initial draft-extend after target prefill.
           Runs through the normal ``prepare_forward_batch`` path
           (``skip_attn_backend_init=False``).

        2. **DECODE(draft)** — multi-step decode loop.  The EAGLE worker
           pre-initializes attention metadata and swaps per-step backends
           directly, using ``skip_attn_backend_init=True``.  Token dim
           is ``bs × topk``.

        3. **DRAFT_EXTEND** — extend after verify.  On TPU the EAGLE
           worker pads to ``(speculative_num_steps + 1) × bs`` and runs
           with ``skip_attn_backend_init=True``.

        Draft-side precompilation is intentionally **skipped** as serving
        behavior.  The sglang startup server warmup (``skip_server_warmup=False``,
        the default) drives the full v1 speculative loop — target EXTEND/FULL +
        TARGET_VERIFY and draft EXTEND/LAST + per-step DECODE/LAST + DRAFT_EXTEND
        — before the server reports ready, so all these shapes are already
        compiled when the first real request arrives.  Measured: 0 post-ready
        recompiles on both Qwen3-1.7B and Llama-3.1-8B.  Enabling the dormant
        precompile here is therefore unnecessary, and it is unsafe for shared
        target/draft model families: on Llama-3.1-8B it core-halts during warmup
        with a tensor-parallel collective desync (``error_0200``,
        "Group didn't close all-reduce" in the EAGLE3 hidden-states-cat path).
        See ``work_dirs/runs/round6_task2_crash_RESULTS.md``.  The dormant
        ``_precompile_*`` helpers are kept only for the
        ``SGLANG_TPU_DRAFT_PRECOMPILE_ISOLATE`` diagnostic path; do not enable
        them as serving behavior without resolving that desync.
        """
        if not self._is_speculative:
            return
        logger.info(
            "[compilation manager] DRAFT: skipping precompilation "
            "(startup server warmup compiles these shapes before readiness)."
        )

    def _precompile_extend(self) -> None:
        """Compile every (token_bucket, bs_bucket) combination."""
        valid_shapes = [
            (t, b)
            for b in self.b_bucketer.buckets
            for t in self.t_bucketer.buckets
            if t >= b
        ]
        n = len(valid_shapes)
        logger.info(
            "[compilation manager] EXTEND: %d shapes (bs_buckets=%s, token_buckets=%s)",
            n,
            list(self.b_bucketer.buckets),
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
        """Compile TARGET_VERIFY mode for each batch-size bucket.

        In TARGET_VERIFY the target model verifies ``draft_token_num``
        tokens per request in a single forward pass.  The total token
        count is ``bs * draft_token_num``, bucketed by the token
        bucketer.  Each unique (pad_tokens, pad_bs) shape is a separate
        HLO cache entry.

        A minimal ``spec_info`` stub carrying ``draft_token_num`` is
        attached so the attention backend can read it during compilation.
        """
        draft_token_num = self.draft_token_num
        bs_buckets = list(self.b_bucketer.buckets)
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

    def _precompile_extend_capture_hidden(self) -> None:
        """Compile EXTEND with CaptureHiddenMode.FULL.

        During speculative decoding the first target-model prefill runs
        with ``CaptureHiddenMode.FULL`` to capture all hidden states for
        the draft model.  The logits processor branches on
        ``capture_hidden_mode.need_capture()`` / ``.is_full()``, which
        Dynamo specializes, so EXTEND+FULL is a distinct graph from the
        normal EXTEND+NULL already compiled by ``_precompile_extend``.
        """
        valid_shapes = [
            (t, b)
            for b in self.b_bucketer.buckets
            for t in self.t_bucketer.buckets
            if t >= b
        ]
        n = len(valid_shapes)
        logger.info(
            "[compilation manager] EXTEND(FULL): %d shapes "
            "(bs_buckets=%s, token_buckets=%s)",
            n,
            list(self.b_bucketer.buckets),
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

    def _make_null_spec_info(self):
        """Build a minimal spec_info stub for non-spec forward modes.

        When speculative decoding is enabled, Dynamo traces both branches
        of ``if spec_info is not None`` guards in the model forward.
        Without a real object, ``spec_info.hidden_states`` raises
        AttributeError and Dynamo emits a fatal graph break under
        fullgraph=True. This stub carries the attributes Dynamo probes
        without affecting the forward result (the runtime guard skips
        the spec path for EXTEND/DECODE).
        """
        if not self._is_speculative:
            return None
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        return EagleDraftInput.create_idle_input(
            device=self.model_runner.device,
            hidden_size=self._extend_width or 1,
            dtype=torch.bfloat16,
            topk=self.topk,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    def _make_verify_spec_info(self):
        """Build an ``EagleVerifyInput`` for TARGET_VERIFY precompilation."""
        from sglang.srt.speculative.eagle_info import EagleVerifyInput

        return EagleVerifyInput.create_idle_input(
            topk=self.topk,
            spec_steps=self.speculative_num_steps,
            num_verify_tokens=self.draft_token_num,
            device=self.model_runner.device,
        )

    def _make_draft_spec_info(self, num_tokens: int, hidden_size: Optional[int] = None):
        """Build an ``EagleDraftInput`` for draft-model precompilation.

        ``hidden_size`` selects the ``hidden_states`` width.  The draft EXTEND
        and DECODE phases consume different widths: EXTEND(LAST) reads the
        target aux concat (``_extend_width``) while the DECODE loop reads the
        draft self-chain output (``_decode_width``).  Defaults to the extend
        width to match the prior single-width behavior.
        """
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        device = self.model_runner.device
        if hidden_size is None:
            hidden_size = self._extend_width
        return EagleDraftInput(
            hidden_states=torch.zeros(
                num_tokens,
                hidden_size,
                dtype=torch.bfloat16,
                device=device,
            ),
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int32),
            topk_p=torch.empty(
                (num_tokens, self.topk), device=device, dtype=torch.float32
            ),
            topk_index=torch.empty(
                (num_tokens, self.topk), device=device, dtype=torch.int64
            ),
            capture_hidden_mode=CaptureHiddenMode.LAST,
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
        )

    def _make_draft_extend_input(self, num_tokens: int, bs: int):
        """Build an ``EagleDraftExtendInput`` for DRAFT_EXTEND precompilation.

        Mirrors the runtime ``forward_draft_extend_after_decode`` input: the
        worker pads to ``(speculative_num_steps + 1) * bs`` tokens and installs
        an ``EagleDraftExtendInput`` (not a generic ``EagleDraftInput``) carrying
        per-request accept counts and batch-state slices.  The ``hidden_states``
        width is the EAGLE3 aux-concat width (``_extend_width``).
        """
        from sglang.srt.speculative.eagle_info import EagleDraftExtendInput

        device = self.model_runner.device
        draft_token_num = self.draft_token_num or 1
        return EagleDraftExtendInput(
            hidden_states=torch.zeros(
                num_tokens,
                self._extend_width,
                dtype=torch.bfloat16,
                device=device,
            ),
            num_correct_drafts=torch.zeros(bs, dtype=torch.int32, device=device),
            num_accept_tokens=torch.ones(bs, dtype=torch.int32, device=device),
            num_accept_tokens_cpu=[1] * bs,
            input_ids=torch.zeros(num_tokens, dtype=torch.long, device=device),
            seq_lens=torch.ones(bs, dtype=torch.int32, device=device),
            seq_lens_cpu=torch.ones(bs, dtype=torch.int32),
            req_pool_indices=torch.ones(bs, dtype=torch.int64, device=device),
            positions=torch.zeros(num_tokens, dtype=torch.int64, device=device),
            bonus_tokens=torch.zeros(bs, dtype=torch.int32, device=device),
            capture_hidden_mode=CaptureHiddenMode.LAST,
            num_tokens_per_req=draft_token_num,
        )

    def _precompile_extend_capture_last(self) -> None:
        """Compile EXTEND with CaptureHiddenMode.LAST.

        The draft model's initial extend (``forward_draft_extend``) runs
        after the target model's prefill and captures the last hidden
        state for the next draft decode iteration.  This path does NOT
        use ``skip_attn_backend_init``, so ``prepare_forward_batch``
        runs normally and shapes follow the standard extend bucketing.
        """
        valid_shapes = [
            (t, b)
            for b in self.b_bucketer.buckets
            for t in self.t_bucketer.buckets
            if t >= b
        ]
        n = len(valid_shapes)
        logger.info(
            "[compilation manager] EXTEND(LAST): %d shapes "
            "(bs_buckets=%s, token_buckets=%s)",
            n,
            list(self.b_bucketer.buckets),
            list(self.t_bucketer.buckets),
        )
        compile_range = (
            tqdm.tqdm(
                valid_shapes,
                desc="Compiling EXTEND(LAST)",
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
                capture_hidden_mode=CaptureHiddenMode.LAST,
                spec_info=self._make_draft_spec_info(num_tokens),
            )
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled EXTEND(LAST) (tokens={num_tokens}, bs={bs})"
                    f" in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_draft_decode(self) -> None:
        """Compile the draft decode loop shapes.

        In the EAGLE draft decode loop, each step runs with:
        * ``forward_mode = DECODE``
        * ``capture_hidden_mode = CaptureHiddenMode.LAST``
        * ``spec_info`` is an ``EagleDraftInput`` (``is_draft_input()``
          returns ``True``) — Dynamo guards on its presence.
        * ``skip_attn_backend_init=True`` — ``prepare_forward_batch`` is
          NOT called at runtime, so we skip it here too.
        * Token dim = ``bs × topk`` (for topk=1, same as bs).

        The EAGLE worker initializes attention metadata once before the
        loop via ``TpuPallasMultiStepDraftBackend.init_forward_metadata``
        and swaps per-step backends directly.  We still call
        ``attn_backend.init_forward_metadata`` here to mirror the
        metadata state the kernel expects.
        """
        bs_buckets = list(self.b_bucketer.buckets)
        topk = self.topk
        n = len(bs_buckets)
        # At runtime the draft decode loop swaps a distinct per-step backend
        # object (``draft_attn_backend.attn_backends[i]``) on each of the
        # ``speculative_num_steps`` steps.  The compiled graph is keyed on the
        # backend object identity, so each step is a separate cache entry and
        # must be precompiled against its own backend.  When the runtime
        # backends are not bound (the naive isolate path), fall back to a single
        # compile against ``model_runner.attn_backend``.
        step_backends = [None]
        logger.info(
            "[compilation manager] DECODE(draft): %d shapes (bs=%s, topk=%d, steps=%d)",
            n,
            bs_buckets,
            topk,
            len(step_backends),
        )
        compile_range = (
            tqdm.tqdm(
                bs_buckets,
                desc="Compiling DECODE(draft)",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_tensor_model_parallel_rank() == 0
            else bs_buckets
        )
        for bs in compile_range:
            num_tokens = bs * topk
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            for step, backend in enumerate(step_backends):
                self._run_one(
                    num_tokens=num_tokens,
                    bs=bs,
                    mode=ForwardMode.DECODE,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                    spec_info=self._make_draft_spec_info(
                        num_tokens, self._decode_width
                    ),
                    skip_prepare=True,
                    attn_backend=backend,
                )
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled DECODE(draft) (bs={bs}, tokens={num_tokens})"
                    f" in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_draft_extend_after_decode(self) -> None:
        """Compile DRAFT_EXTEND shapes used after target-model verify.

        On TPU the EAGLE worker pads ``input_ids``, ``positions``,
        ``out_cache_loc``, and ``hidden_states`` to the fixed size
        ``(speculative_num_steps + 1) × bs`` so XLA sees a single
        compiled graph regardless of how many draft tokens were accepted.

        The forward runs with ``skip_attn_backend_init=True`` — the EAGLE
        worker initializes the draft-extend attention backend directly.
        """
        draft_token_num = self.draft_token_num
        bs_buckets = list(self.b_bucketer.buckets)
        n = len(bs_buckets)
        logger.info(
            "[compilation manager] DRAFT_EXTEND: %d shapes (bs=%s, draft_token_num=%d)",
            n,
            bs_buckets,
            draft_token_num,
        )
        compile_range = (
            tqdm.tqdm(
                bs_buckets,
                desc="Compiling DRAFT_EXTEND",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_tensor_model_parallel_rank() == 0
            else bs_buckets
        )
        for bs in compile_range:
            num_tokens = draft_token_num * bs
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.DRAFT_EXTEND,
                capture_hidden_mode=CaptureHiddenMode.LAST,
                spec_info=self._make_draft_extend_input(num_tokens, bs),
                skip_prepare=True,
                attn_backend=None,
            )
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled DRAFT_EXTEND (bs={bs}, tokens={num_tokens})"
                    f" in {elapsed:.1f}s{hbm_str}"
                )

    def _run_one(
        self,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        *,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: Optional[object] = None,
        skip_prepare: bool = False,
        attn_backend: Optional[object] = None,
    ) -> None:
        """Build and execute one dummy forward pass for compilation.

        Parameters
        ----------
        skip_prepare:
            When ``True``, skip ``prepare_forward_batch`` to match
            runtime paths that use ``skip_attn_backend_init=True``
            (e.g., EAGLE draft decode and draft-extend-after-decode).
            Attention metadata is still initialized via
            ``attn_backend.init_forward_metadata``.
        attn_backend:
            The attention backend to bind for this forward.  When ``None``
            (target paths), falls back to ``model_runner.attn_backend``.
            The EAGLE draft paths pass the exact runtime per-step decode
            backend or the draft-extend backend so the compiled graph is
            keyed on the same backend object the worker drives at runtime.
        """
        backend = (
            attn_backend if attn_backend is not None else self.model_runner.attn_backend
        )
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
        # Set up req_to_token row 1 with valid page indices for this
        # model's KV pool. The dummy batch uses req_pool_indices=1 and
        # out_cache_loc=1 to avoid conflicts with row/slot 0 (which may
        # have stale state from another model's precompilation).
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        req_to_token[1, :num_tokens] = 1  # all positions map to KV slot 1

        model = self.model_runner.model
        if not skip_prepare and hasattr(model, "prepare_forward_batch"):
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
        *,
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

        # Token-level inputs
        input_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        input_ids[:bs] = 1
        positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)

        # Use slot 1 (not 0) for out_cache_loc. Slot 0 is the permanent
        # padding slot shared between target and draft models. If the
        # target model's precompilation writes page indices to
        # req_to_token row 0 that are valid for the target's KV pool
        # but out-of-bounds for the draft's smaller KV pool, the draft's
        # attention kernel would crash. Using slot 1 avoids this conflict.
        out_cache_loc = torch.ones(num_tokens, dtype=torch.int64, device=device)

        # Batch-level inputs — use req_pool_indices=1 (not 0) for the
        # same reason: row 0 of req_to_token may have stale page indices
        # from a different model's precompilation.
        seq_lens = torch.ones(bs, dtype=torch.int64, device=device)
        seq_lens_cpu = torch.ones(bs, dtype=torch.int64)
        req_pool_indices = torch.ones(bs, dtype=torch.int64, device=device)

        if is_extend:
            # For TARGET_VERIFY and DRAFT_EXTEND the extend fields
            # describe each request's contribution in tokens.  For
            # normal EXTEND each request contributes 1 token.
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
