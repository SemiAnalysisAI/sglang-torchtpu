"""Validate `_page_aligned_free_count_topk1` (eagle_info) against an independent
reference for the page-aligned KV-free count on the topk==1 verify path.

The verify free at page_size>1 replaces the dynamic-shape gather
``out_cache_loc[align(evict_mask)]`` with a static compaction that frees a
``[:free_count]`` slice. ``free_count`` must EXACTLY equal the number of True
entries in the aligned evict_mask, or the KV allocator frees the wrong slots
(silent cache corruption). This test cross-checks the helper against a
pure-python set-based reference over random shapes.
"""

import random

import torch

from sglang.srt.speculative.eagle_info import _page_aligned_free_count_topk1


def _ref_free_count(seq_lens, num_accept, page_size, num_draft_tokens):
    """Independent reference: front-packed accepts, evict the tail [c, D), then the
    page-align keeps [lo, hi); freed = evicted minus kept, summed over requests."""
    total = 0
    for i in range(len(num_accept)):
        c = num_accept[i]
        evicted = set(range(c, num_draft_tokens))
        sl = int(seq_lens[i])
        start = (sl + c - 1) // page_size * page_size - sl
        lo = max(start, 0)
        hi = min(start + page_size, num_draft_tokens)
        kept = set(range(lo, hi))
        total += len(evicted - kept)
    return total


def test_page_aligned_free_count_matches_reference():
    rng = random.Random(0)
    for page_size in (2, 4, 8, 16, 64):
        for num_draft_tokens in (2, 4, 5, 8):
            for _ in range(200):
                bs = rng.randint(1, 8)
                seq_lens = [rng.randint(0, 300) for _ in range(bs)]
                # num_accept in [1, num_draft_tokens] (>=1: the bonus token always accepts)
                num_accept = [rng.randint(1, num_draft_tokens) for _ in range(bs)]
                seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
                got = _page_aligned_free_count_topk1(
                    seq_lens_t, num_accept, page_size, num_draft_tokens
                )
                exp = _ref_free_count(seq_lens, num_accept, page_size, num_draft_tokens)
                assert got == exp, (
                    f"free_count mismatch: got={got} exp={exp} "
                    f"page_size={page_size} D={num_draft_tokens} "
                    f"seq_lens={seq_lens} num_accept={num_accept}"
                )


def test_helper_never_frees_kept_or_negative():
    """Sanity: free_count is in [0, total evicted] for random inputs."""
    rng = random.Random(1)
    for _ in range(500):
        page_size = rng.choice([2, 4, 8, 64])
        D = rng.choice([2, 4, 8])
        bs = rng.randint(1, 6)
        seq_lens = [rng.randint(0, 200) for _ in range(bs)]
        num_accept = [rng.randint(1, D) for _ in range(bs)]
        total_evicted = sum(D - c for c in num_accept)
        got = _page_aligned_free_count_topk1(
            torch.tensor(seq_lens, dtype=torch.int32), num_accept, page_size, D
        )
        assert 0 <= got <= total_evicted, (got, total_evicted, seq_lens, num_accept)
