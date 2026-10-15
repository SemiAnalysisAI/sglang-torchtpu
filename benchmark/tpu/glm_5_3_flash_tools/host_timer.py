"""Temporary host-side phase timer for the overlap scheduler (pod-only, never committed).

Enable with QH_HOST_TIMER=1; scheduler.event_loop_overlap calls install(self).
Wraps instance methods with perf_counter timers (inclusive wall time) and prints
mean ms per call every N loop iterations on TP0.
"""
import collections, os, sys, time

N = int(os.environ.get("QH_HOST_TIMER_EVERY", "400"))
_acc = collections.defaultdict(float)
_cnt = collections.Counter()
_state = {"last": None, "iters": 0}


def _wrap(obj, attr, name):
    fn = getattr(obj, attr, None)
    if fn is None or getattr(fn, "_qh_wrapped", False):
        return
    def timed(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            _acc[name] += time.perf_counter() - t; _cnt[name] += 1
    timed._qh_wrapped = True
    setattr(obj, attr, timed)


def _report():
    it = _state["iters"]
    lines = [f"HOSTTIMER {it} iters, mean loop period {_acc['loop.period'] / max(_cnt['loop.period'], 1) * 1e3:.3f} ms"]
    for k in sorted(_acc, key=lambda k: -_acc[k]):
        if k == "loop.period":
            continue
        lines.append(f"HOSTTIMER   {_acc[k] / it * 1e3:7.3f} ms/iter  {_cnt[k] / it:5.2f} calls/iter  {k}")
    print("\n".join(lines), file=sys.stderr, flush=True)
    _acc.clear(); _cnt.clear(); _state["iters"] = 0


def install(sched):
    if getattr(sched, "tp_rank", 0) != 0:
        return
    for a in ("ingest_requests", "get_next_batch_to_run", "process_batch_result",
              "_apply_war_barrier", "launch_batch_sample_if_needed", "_relay_forward_payload"):
        _wrap(sched, a, "sched." + a)
    run = sched.run_batch
    def run_batch(batch):
        t = time.perf_counter()
        if _state["last"] is not None:
            _acc["loop.period"] += t - _state["last"]; _cnt["loop.period"] += 1
        _state["last"] = t
        try:
            return run(batch)
        finally:
            if not batch.forward_mode.is_decode():
                # Keep windows to steady decode: drop prefill and the gap before it.
                _acc.clear(); _cnt.clear(); _state["iters"] = 0; _state["last"] = None
            else:
                _acc["sched.run_batch"] += time.perf_counter() - t; _cnt["sched.run_batch"] += 1
                _state["iters"] += 1
                if _state["iters"] >= N:
                    _report()
    sched.run_batch = run_batch
    fm = getattr(sched, "future_map", None)
    if fm is not None:
        for a in ("publish", "stash", "resolve_forward_inputs"):
            _wrap(fm, a, "future_map." + a)
    import sglang.srt.managers.scheduler as S
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    _wrap(S, "resolve_forward_inputs", "resolve_forward_inputs")
    _wrap(ForwardBatch, "init_new", "ForwardBatch.init_new")
    cw = getattr(sched, "tpu_async_copy_worker", None)
    if cw is not None:
        _wrap(cw, "submit", "tpu_async_copy_worker.submit")
    spec = sched.model_worker
    if spec is not sched.tp_worker:
        for a in ("forward_batch_generation", "_append_target_hidden_to_draft_kv_by_loc",
                  "_accept_block", "_accept_block_on_bucket", "_propose_selector_block",
                  "_greedy_sample_from_vocab_parallel_head", "_update_target_mamba_state_after_verify",
                  "_make_next_draft_input_decode", "embed_draft_block_ids",
                  "_fill_verify_block_cache_locs", "_rebuild_compact_draft_cache"):
            _wrap(spec, a, "dflash." + a)
        dmr = spec.draft_model_runner
        _wrap(dmr, "forward", "draft_model_runner.forward")
        dm = dmr.model
        _wrap(dm, "forward", "draft_model.forward")
        from sglang.srt.speculative.dflash_info import DFlashVerifyInput
        _wrap(DFlashVerifyInput, "prepare_for_verify", "DFlashVerifyInput.prepare_for_verify")
    w = sched.tp_worker
    _wrap(w, "forward_batch_generation", "worker.forward_batch_generation")
    mr = w.model_runner
    for a in ("forward", "sample"):
        _wrap(mr, a, "model_runner." + a)
    ab = getattr(mr, "attn_backend", None)
    if ab is not None:
        _wrap(ab, "init_forward_metadata", "attn_backend.init_forward_metadata")
    m = getattr(mr, "model", None)
    if m is not None:
        shim = m.forward
        print(f"HOSTTIMER model.forward is {type(shim).__name__}", file=sys.stderr, flush=True)
        _wrap(shim, "fn", "shim.fn(compiled call)")
        _wrap(shim, "wait_for_hicache_reload", "shim.wait_for_hicache_reload")
        pol = getattr(shim, "policy", None)
        if pol is not None:
            _wrap(pol, "finalize_forward_batch", "policy.finalize_forward_batch")
        ck = getattr(shim, "_recompilation_checker", None)
        if ck is not None:
            _wrap(ck, "check", "recompilation_checker.check")
        _wrap(m, "forward", "model.forward(graph launch)")
    try:
        from torch_tpu._internal.compile import torch_tpu_compiled_executable as X
        cls = X.TorchTpuCompiledExecutable
        orig = cls.__call__
        def call(self, *a, **k):
            t = time.perf_counter()
            try:
                return orig(self, *a, **k)
            finally:
                n = len(a[0]) if len(a) == 1 and isinstance(a[0], (list, tuple)) else len(a)
                key = "tt_executable(big graph)" if n > 64 else "tt_executable(small)"
                _acc[key] += time.perf_counter() - t; _cnt[key] += 1
        cls.__call__ = call
    except Exception as e:  # pragma: no cover
        print(f"HOSTTIMER tt wrap failed: {e}", file=sys.stderr, flush=True)
    print("HOSTTIMER installed", file=sys.stderr, flush=True)
