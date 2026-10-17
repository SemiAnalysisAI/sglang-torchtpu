"""Platform overrides of the shared mHC functions (``MhcOverrides``).

A platform plugin installs fused hc_pre / hc_post / hc_post_pre
implementations; each gets the public function's keyword arguments and may
decline a call by returning None, which then takes the torch path. Nothing is
installed in-tree, so hc_post_pre is None and hc_pre / hc_post run as before.
"""

import inspect
from contextlib import ExitStack

import pytest
import torch

import sglang.kernels.ops.layernorm.mhc as mhc
from sglang.kernels.ops.layernorm.mhc import (
    MhcOverrides,
    get_mhc_overrides,
    hc_post,
    hc_post_pre,
    hc_pre,
    set_mhc_overrides,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HC, D, TOKENS = 4, 8, 3
SITE = dict(hc_mult=HC, rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20)


@pytest.fixture(autouse=True)
def _torch_path_and_restore():
    saved = get_mhc_overrides()
    with ExitStack() as stack:
        for env in (
            envs.SGLANG_OPT_USE_TILELANG_MHC_PRE,
            envs.SGLANG_OPT_USE_TILELANG_MHC_POST,
            envs.SGLANG_OPT_FUSE_MHC_POST_PRE,
        ):
            stack.enter_context(env.override(False))
        yield
    set_mhc_overrides(saved)


@pytest.fixture
def site():
    gen = torch.Generator().manual_seed(11)
    mix = (2 + HC) * HC
    return dict(
        x=torch.randn(TOKENS, HC * D, generator=gen),
        hc_fn=torch.randn(mix, HC * D, generator=gen) * 0.1,
        hc_scale=torch.tensor([0.8, 1.1, 1.3]),
        hc_base=torch.randn(mix, generator=gen) * 0.3,
        branch=torch.randn(TOKENS, D, generator=gen),
        norm_weight=1 + 0.1 * torch.randn(D, generator=gen),
    )


def _pre(site, **kwargs):
    return hc_pre(
        site["x"], site["hc_fn"], site["hc_scale"], site["hc_base"], **SITE, **kwargs
    )


def _post(site, h_post, h_res):
    return hc_post(site["branch"], site["x"], h_post, h_res, hc_mult=HC)


def _post_pre(site, h_post, h_res, **kwargs):
    return hc_post_pre(
        site["branch"],
        site["x"],
        h_post,
        h_res,
        site["hc_fn"],
        site["hc_scale"],
        site["hc_base"],
        **SITE,
        **kwargs,
    )


def _assert_same(actual, expected):
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        if isinstance(e, torch.Tensor):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        else:
            assert a == e


def test_nothing_registered_in_tree():
    assert mhc._mhc_overrides == MhcOverrides()


def test_default_runs_torch_path_and_post_pre_declines(site):
    layer_input, h_res, h_post, norm_fused = _pre(site)
    assert not norm_fused
    post_mix, comb, expected_in = mhc._mhc_pre_torch(
        site["x"].view(TOKENS, HC, D),
        site["hc_fn"],
        site["hc_scale"],
        site["hc_base"],
        SITE["rms_eps"],
        SITE["hc_eps"],
        SITE["hc_eps"],
        2.0,
        SITE["sinkhorn_iters"],
    )
    torch.testing.assert_close(layer_input, expected_in)
    torch.testing.assert_close(h_res, comb.reshape(TOKENS, -1))
    torch.testing.assert_close(h_post, post_mix.reshape(TOKENS, -1))
    expected_out = mhc._mhc_post_torch(
        site["branch"], site["x"].view(TOKENS, HC, D), post_mix, comb
    ).view(TOKENS, -1)
    torch.testing.assert_close(_post(site, h_post, h_res), expected_out)
    assert _post_pre(site, h_post, h_res) is None


def test_override_receives_public_kwargs_and_wins(site):
    seen = {}
    pre_result = (torch.zeros(TOKENS, D), torch.zeros(TOKENS, HC * HC), None, True)
    post_result = torch.ones(TOKENS, HC * D)
    post_pre_result = (torch.zeros(TOKENS, D), post_result, None, None, True)

    def make(name, result):
        def override(**kwargs):
            seen[name] = kwargs
            return result

        return override

    set_mhc_overrides(
        MhcOverrides(
            hc_pre=make("hc_pre", pre_result),
            hc_post=make("hc_post", post_result),
            hc_post_pre=make("hc_post_pre", post_pre_result),
        )
    )
    h_post, h_res = torch.zeros(TOKENS, HC), torch.zeros(TOKENS, HC * HC)
    norm = dict(out_norm_weight=site["norm_weight"], out_norm_eps=1e-6)
    assert _pre(site, **norm) is pre_result
    assert _post(site, h_post, h_res) is post_result
    assert _post_pre(site, h_post, h_res, **norm) is post_pre_result

    for name, fn in (("hc_pre", hc_pre), ("hc_post", hc_post)):
        assert set(seen[name]) == set(inspect.signature(fn).parameters)
    assert set(seen["hc_post_pre"]) == set(inspect.signature(hc_post_pre).parameters)
    assert seen["hc_pre"]["out_norm_weight"] is site["norm_weight"]
    assert seen["hc_pre"]["post_mult_value"] == 2.0
    assert seen["hc_post_pre"]["out_norm_eps"] == 1e-6


def test_declined_call_falls_back_to_torch(site):
    expected_pre = _pre(site, out_norm_weight=site["norm_weight"], out_norm_eps=1e-6)
    expected_post = _post(site, expected_pre[2], expected_pre[1])
    calls = []

    def decline(**kwargs):
        calls.append(kwargs)
        return None

    set_mhc_overrides(MhcOverrides(hc_pre=decline, hc_post=decline))
    _assert_same(
        _pre(site, out_norm_weight=site["norm_weight"], out_norm_eps=1e-6),
        expected_pre,
    )
    _assert_same(
        (_post(site, expected_pre[2], expected_pre[1]),),
        (expected_post,),
    )
    # The torch path does not apply the norm; the caller still owes it.
    assert expected_pre[3] is False
    assert len(calls) == 2
    # An unset slot declines like a registered one that returns None.
    assert _post_pre(site, expected_pre[2], expected_pre[1]) is None


def test_empty_batch_never_reaches_the_override(site):
    def fail(**kwargs):
        raise AssertionError("override called on an empty batch")

    set_mhc_overrides(MhcOverrides(hc_pre=fail, hc_post=fail, hc_post_pre=fail))
    empty = {**site, "x": site["x"][:0], "branch": site["branch"][:0]}
    layer_input, h_res, h_post, norm_fused = _pre(empty)
    assert layer_input.shape == (0, D) and not norm_fused
    assert _post(empty, h_post, h_res).shape == (0, HC * D)
    assert _post_pre(empty, h_post, h_res) is None
