"""AC-2 runtime guard: a real TPU NGram greedy decode must not launch any raw
CUDA/Triton kernel.

A temporary ``sitecustomize`` is placed on ``PYTHONPATH`` so that *every* process
in the server (launcher + spawned scheduler) installs an import hook. When
``SGLANG_NGRAM_TPU_KERNEL_GUARD=1`` the hook replaces the raw kernels used by the
NGram verify/prepare/free path with objects that raise on call:

  - ngram_info.create_flashinfer_kv_indices_triton  (generate_attn_arg_prefill)
  - ngram_info.get_target_cache_loc                 (page_size>1 free path)
  - NgramVerifyInput.generate_attn_arg_prefill      (GPU-only attn-arg build)
  - spec_utils.assign_req_to_token_pool             (raw Triton; CUDA branch of
                                                     the dispatch wrapper)

A successful greedy decode through ``pallas_tpu_rpa_v3`` then proves none of those
paths were reached on TPU (on TPU Triton silently rolls back to CPU, so this
instrumented guard is stronger than relying on a crash).

Usage:
    SGLANG_PLATFORM=tpu python test/manual/tpu/test_ngram_tpu_kernel_guard.py
"""

import os
import tempfile
import textwrap
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from _ngram_tpu_common import MODEL, skip_if_no_tpu

_SITECUSTOMIZE = textwrap.dedent(
    '''
    import builtins
    import os
    import sys

    if os.environ.get("SGLANG_NGRAM_TPU_KERNEL_GUARD") == "1":
        _real_import = builtins.__import__
        _in_hook = False

        class _Forbidden:
            def __init__(self, name):
                self._name = name

            def __getitem__(self, *a):
                raise AssertionError("FORBIDDEN raw kernel launched on TPU: " + self._name)

            def __call__(self, *a, **k):
                raise AssertionError("FORBIDDEN raw kernel called on TPU: " + self._name)

        _done = set()

        def _patch():
            # Uses the module-level `sys` captured before the hook is installed,
            # so it never triggers an import (which would re-enter this hook).
            su = sys.modules.get("sglang.srt.speculative.spec_utils")
            if su is not None and "su" not in _done:
                su.assign_req_to_token_pool = _Forbidden("assign_req_to_token_pool")
                _done.add("su")

            ni = sys.modules.get("sglang.srt.speculative.ngram_info")
            if ni is not None and "ni" not in _done:
                ni.create_flashinfer_kv_indices_triton = _Forbidden(
                    "create_flashinfer_kv_indices_triton"
                )
                ni.get_target_cache_loc = _Forbidden("get_target_cache_loc")

                def _boom(*a, **k):
                    raise AssertionError("FORBIDDEN generate_attn_arg_prefill called on TPU")

                ni.NgramVerifyInput.generate_attn_arg_prefill = _boom
                _done.add("ni")

        def _guarded_import(*a, **k):
            global _in_hook
            mod = _real_import(*a, **k)
            if not _in_hook:
                _in_hook = True
                try:
                    _patch()
                except Exception:
                    pass
                finally:
                    _in_hook = False
            return mod

        builtins.__import__ = _guarded_import
    '''
)


@unittest.skipIf(skip_if_no_tpu(), "torch_tpu not available")
class TestNgramTpuKernelGuard(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_PLATFORM"] = "tpu"
        cls.tmpdir = tempfile.mkdtemp(prefix="ngram_tpu_guard_")
        with open(os.path.join(cls.tmpdir, "sitecustomize.py"), "w") as f:
            f.write(_SITECUSTOMIZE)

        env = dict(os.environ)
        env["SGLANG_NGRAM_TPU_KERNEL_GUARD"] = "1"
        env["PYTHONPATH"] = cls.tmpdir + os.pathsep + env.get("PYTHONPATH", "")

        cls.process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=600,
            env=env,
            other_args=[
                "--device", "tpu",
                "--dtype", "bfloat16",
                "--attention-backend", "pallas_tpu_rpa_v3",
                "--page-size", "1",
                "--grammar-backend", "none",
                "--mem-fraction-static", "0.4",
                "--max-running-requests", "8",
                "--speculative-algorithm", "NGRAM",
                "--speculative-num-draft-tokens", "4",
                "--speculative-ngram-max-bfs-breadth", "1",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_greedy_decode_launches_no_raw_kernel(self):
        # If any guarded raw kernel were reached the scheduler would crash and the
        # request would not return 200 with real output.
        for prompt in ["The capital of France is", "The first three primes are 2, 3,"]:
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                    "stream": False,
                },
                timeout=300,
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            ids = resp.json().get("output_ids")
            self.assertTrue(ids, f"no output for {prompt!r}: {resp.text[:200]}")


if __name__ == "__main__":
    unittest.main()
