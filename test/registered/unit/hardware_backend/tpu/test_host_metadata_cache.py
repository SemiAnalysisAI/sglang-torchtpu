import torch

from sglang.srt.hardware_backend.tpu.host_metadata_cache import HostMetadataCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def test_reuse_and_invalidation_after_page_change():
    cache = HostMetadataCache()
    pages = torch.tensor([[3, 4], [8, 9]], dtype=torch.int32)
    first = cache.get("pages", pages, "cpu")
    assert cache.get("pages", pages.clone(), "cpu") is first
    pages[0, 1] = 10
    second = cache.get("pages", pages, "cpu")
    assert second is not first
    assert first[0, 1] == 4
    assert second[0, 1] == 10
    third = cache.get("pages", pages.flip(0), "cpu")
    assert third is not second
    assert third[0, 0] == 8


def test_names_shapes_and_dtypes_are_part_of_cache_identity():
    cache = HostMetadataCache()
    x = torch.ones(4, dtype=torch.int32)
    a = cache.get("q", x, "cpu")
    assert cache.get("kv", x, "cpu") is not a
    assert cache.get("q", x.long(), "cpu").dtype == torch.int64
    assert cache.get("q", x[:2], "cpu").shape == (2,)
    assert len(cache._entries) == 2


def test_active_scalar_is_owned_across_batch_sizes_and_source_mutation():
    from sglang.srt.hardware_backend.tpu.attention_pallas_rpa_base import (
        _make_active_num_tokens,
    )

    cache = HostMetadataCache()
    previous = None
    for bs, live in ((16, 16), (4, 4), (16, 4), (4, 0)):
        query = torch.arange(bs + 1, dtype=torch.int32)
        scalar = cache.get(
            "active_num_tokens", _make_active_num_tokens(query, live), "cpu"
        )
        assert scalar.shape == ()
        assert scalar._base is None
        assert scalar.item() == live
        if previous is not None and previous.item() == live:
            assert scalar is previous
        query.fill_(999)
        assert scalar.item() == live
        previous = scalar
