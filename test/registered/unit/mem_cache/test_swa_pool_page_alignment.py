"""CPU-only unit tests for the TPU hybrid-SWA 8-page alignment floor.

The two-pool hybrid-SWA KV cache miscompiles under torch.compile on TPU unless
the SWA sub-pool's page count is a multiple of the sublane tiling factor (8);
_derive_pool_sizes rounds down to enforce that. The round-down is guarded by
`aligned > 0`, which was meant to stop a small pool being zeroed -- but its
effect is that a pool of fewer than 8 pages skips the alignment entirely and is
allocated at its unaligned size, which is exactly the state the round-down
exists to prevent. Nothing logs, and the failure surfaces much later as a
serving hang rather than a startup error.

That floor is reachable from ordinary flags: `--max-total-tokens 32768` with
the TPU recipes' `--swa-full-tokens-ratio 0.10` and page_size 512 gives a
3072-token SWA pool -- 6 pages -- and the server then hangs on the first
request large enough to touch the upper pages.

These tests pin the three regimes: above the floor the alignment fires, at an
exact multiple it is a no-op, and below the floor the pool must not be silently
left unaligned.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PAGE_SIZE = 512
ALIGN_PAGES = 8


def _swa_pool_tokens(swa_tokens, *, device="tpu", page_size=PAGE_SIZE):
    """Run _derive_pool_sizes over a hybrid-SWA config and report the SWA size."""
    from sglang.srt import runtime_context as rc
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator

    # A draft worker zeroes the compression pools instead of reading them off
    # config, which keeps the fake to the fields the alignment path touches.
    fake = SimpleNamespace(
        device=device,
        is_hybrid_swa=True,
        is_draft_worker=True,
        loc_space_scale=1,
        pool_page_size=page_size,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
    )
    config = SimpleNamespace(
        max_total_num_tokens=32768,
        max_running_requests=64,
        full_max_total_num_tokens=32768,
        swa_max_total_num_tokens=swa_tokens,
    )
    with rc.get_context().override_server_args(page_size=page_size):
        sizes = KVCacheConfigurator._derive_pool_sizes(fake, config=config)
    return sizes.swa_max_total_num_tokens


class TestSWAPageAlignment(unittest.TestCase):
    def test_unaligned_pool_above_floor_is_rounded_down(self):
        """34 pages -> 32. The documented behaviour, and the observed one."""
        self.assertEqual(_swa_pool_tokens(34 * PAGE_SIZE), 32 * PAGE_SIZE)

    def test_already_aligned_pool_is_untouched(self):
        for pages in (8, 32, 64):
            with self.subTest(pages=pages):
                self.assertEqual(_swa_pool_tokens(pages * PAGE_SIZE), pages * PAGE_SIZE)

    def test_non_tpu_device_is_untouched(self):
        """The tiling constraint is a TPU property; other devices keep their size."""
        self.assertEqual(
            _swa_pool_tokens(34 * PAGE_SIZE, device="cuda"), 34 * PAGE_SIZE
        )

    def test_pool_below_the_floor_is_not_left_unaligned(self):
        """A sub-8-page pool cannot satisfy the invariant and must not pass through.

        6 pages is what `--max-total-tokens 32768 --swa-full-tokens-ratio 0.10`
        produces at page_size 512. Rounding down yields 0, so the `aligned > 0`
        guard declines to act and the pool is allocated at 6 pages. Any resolution
        is acceptable here -- raise at startup, or round up to 8 -- except
        returning an unaligned size, which is the one outcome the alignment code
        exists to rule out.
        """
        try:
            result = _swa_pool_tokens(6 * PAGE_SIZE)
        except ValueError:
            return  # raising at startup is a valid resolution

        pages = result // PAGE_SIZE
        self.assertEqual(
            pages % ALIGN_PAGES,
            0,
            f"SWA pool left at {pages} pages, which is not a multiple of "
            f"{ALIGN_PAGES}; compiled decode reads wrong K/V from this pool",
        )


if __name__ == "__main__":
    unittest.main()
