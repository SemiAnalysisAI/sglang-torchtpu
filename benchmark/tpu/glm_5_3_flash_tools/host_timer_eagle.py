"""Pod-only host phase timer for the EAGLE/NEXTN overlap round (never committed).

Enable with QH_HOST_TIMER=1; the patched scheduler.event_loop_overlap calls install(self).
Inclusive perf_counter wall time per wrapped call, mean ms per decode iteration, every N iters on TP0.
"""
import collections, os, sys, time

N = int(os.environ.get("QH_HOST_TIMER_EVERY", "300"))
_acc = collections.defaultdict(float)
_cnt = collections.Counter()
_state = {"last": None, "iters": 0}
_tag = "HOSTTIMER"


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
    lines = [f"{_tag} {it} iters, mean loop period {_acc['loop.period'] / max(_cnt['loop.period'], 1) * 1e3:.3f} ms"]
    for k in sorted(_acc, key=lambda k: -_acc[k]):
        if k != "loop.period":
            lines.append(f"{_tag}   {_acc[k] / it * 1e3:7.3f} ms/iter  {_cnt[k] / it:6.2f} calls/iter  {k}")
    print("\n".join(lines), file=sys.stderr, flush=True)
    _acc.clear(); _cnt.clear(); _state["iters"] = 0


_prof = {"p": None, "n": 0, "done": False}
CPROF = os.environ.get("QH_CPROFILE")


def _cprofile_step(is_decode):
    """Profile decode rounds 100..300 of the first long-enough request on this rank."""
    import cProfile, io, pstats
    if _prof["done"]:
        return
    if not is_decode:
        if _prof["p"] is not None:
            _prof["p"].disable(); _prof["p"] = None
        _prof["n"] = 0
        return
    _prof["n"] += 1
    if _prof["n"] == 50:
        _prof["p"] = cProfile.Profile(); _prof["p"].enable()
    elif _prof["n"] == 250 and _prof["p"] is not None:
        _prof["p"].disable()
        for key in ("tottime", "cumulative"):
            buf = io.StringIO()
            pstats.Stats(_prof["p"], stream=buf).sort_stats(key).print_stats(70)
            print(f"{_tag} CPROFILE by {key} (200 rounds)\n" + buf.getvalue(), file=sys.stderr, flush=True)
        buf = io.StringIO()
        st = pstats.Stats(_prof["p"], stream=buf).sort_stats("tottime")
        st.print_callers(r"method 'to' of|method 'tolist' of|torch.tensor|compile_wrapper|method 'copy_' of")
        print(f"{_tag} CPROFILE callers (200 rounds)\n" + buf.getvalue(), file=sys.stderr, flush=True)
        _prof["p"] = None; _prof["done"] = True


def _is_rank0():
    from sglang.srt.runtime_context import get_parallel
    return get_parallel().tp_rank == 0


def install(sched):
    global _tag
    _tag = f"HOSTTIMER[{os.getpid()}]"
    for a in ("recv_requests", "process_input_requests", "get_next_batch_to_run",
              "process_batch_result", "launch_batch_sample_if_needed"):
        _wrap(sched, a, "sched." + a)
    run = sched.run_batch
    def run_batch(batch, *a, **k):
        is_decode = batch.forward_mode.is_decode()
        if CPROF and _is_rank0():
            _cprofile_step(is_decode)
        t = time.perf_counter()
        if _state["last"] is not None:
            _acc["loop.period"] += t - _state["last"]; _cnt["loop.period"] += 1
        _state["last"] = t
        try:
            return run(batch, *a, **k)
        finally:
            if not is_decode:
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
    import sglang.srt.speculative.eagle_worker_common as EC
    import sglang.srt.speculative.eagle_worker_v2 as EV
    import sglang.srt.speculative.spec_utils as SU
    for mod, names in ((EC, ("eagle_prepare_for_verify", "eagle_sample", "commit_mamba_states_after_verify",
                             "verify_bookkeeping_func", "record_stream_for_v2_verify", "build_grammar_vocab_mask")),
                       (EV, ("build_eagle_verify_input", "prepare_for_draft", "prepare_for_draft_extend",
                             "run_eagle_verify", "fused_draft_extend_select_argmax_func"))):
        for n in names:
            _wrap(mod, n, f"{mod.__name__.rsplit('.', 1)[1]}.{n}")
    for n in ("eagle_prepare_for_verify",):
        _wrap(SU, n, "spec_utils." + n)
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    _wrap(ForwardBatch, "init_new", "ForwardBatch.init_new")
    spec = sched.model_worker
    _wrap(spec, "forward_batch_generation", "spec.forward_batch_generation")
    _wrap(spec, "verify", "spec.verify")
    dw = getattr(spec, "draft_worker", None)
    if dw is not None:
        for a in ("draft", "_draft_extend_for_decode", "draft_forward"):
            _wrap(dw, a, "draft_worker." + a)
        dr = getattr(dw, "draft_runner", None)
        if dr is not None:
            _wrap(dr, "forward", "draft_runner.forward")
            if getattr(dr, "attn_backend", None) is not None:
                _wrap(dr.attn_backend, "init_forward_metadata", "draft_attn.init_forward_metadata")
    w = sched.tp_worker
    _wrap(w, "forward_batch_generation", "target_worker.forward_batch_generation")
    mr = w.model_runner
    _wrap(mr, "forward", "target_runner.forward")
    if getattr(mr, "attn_backend", None) is not None:
        _wrap(mr.attn_backend, "init_forward_metadata", "target_attn.init_forward_metadata")
        lin = getattr(mr.attn_backend, "linear_attn_backend", None)
        if lin is not None:
            _wrap(lin, "init_forward_metadata", "target_linear_attn.init_forward_metadata")
        full = getattr(mr.attn_backend, "full_attn_backend", None)
        if full is not None:
            _wrap(full, "init_forward_metadata", "target_full_attn.init_forward_metadata")
    m = getattr(mr, "model", None)
    if m is not None:
        _wrap(m, "forward", "target_model.forward(launch)")
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
