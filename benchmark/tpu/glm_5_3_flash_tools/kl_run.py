"""Run the three kl_test_utils consistency checks against a local server and print KL
summaries: python kl_run.py [port] [samples] [new_tokens]."""
import sys, traceback
import numpy as np
from sglang.test import kl_test_utils as kl

PORT = sys.argv[1] if len(sys.argv) > 1 else "30000"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 32
NEW = int(sys.argv[3]) if len(sys.argv) > 3 else 512
MODEL = "/weights/GLM-5.3-Flash"; URL = f"http://127.0.0.1:{PORT}"
_orig = kl.compare_kl_divergence
def summarize(inp, out, thr, model, name):
    per = []; allk = []
    for a, b in zip(inp, out):
        logr = np.array(a) - np.array(b); k = np.exp(logr) - 1 - logr
        per.append(float(np.mean(k))); allk.extend(k.tolist())
    allk = np.array(allk)
    print(f"RESULT {name}: avg_kl={np.mean(per):.6f} max_tok={allk.max():.4f} "
          f"frac>1e-3={np.mean(allk > 1e-3):.4f} frac>1e-1={np.mean(allk > 1e-1):.4f} "
          f"exact_zero_tokens={np.mean(allk == 0):.4f} worst_samples={sorted(per)[-3:]}", flush=True)
kl.compare_kl_divergence = summarize
thr = {MODEL: {"kl_div": 1e9}}
for fn in (kl.test_input_output_logprobs_match_helper,
           kl.test_input_output_logprobs_match_prefill_cache_hit_helper,
           kl.test_input_output_logprobs_match_decode_cache_hit_helper):
    try:
        fn(URL, thr, MODEL, max_samples=N, max_new_tokens=NEW, trust_remote_code=True)
    except Exception:
        print(f"FAILED {fn.__name__}"); traceback.print_exc(limit=3)
print("KL_RUN_DONE")
