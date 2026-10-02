"""The functional Ulysses all-to-all must match the in-place flat exchange.

Both collectives are emulated in-process with the same chunk semantics (equal
chunks of the flattened buffer, reversed across a 4-rank group), so only the
call each platform preference selects and the result layout are checked.
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers import usp as usp_mod

_USP = "sglang.multimodal_gen.runtime.layers.usp"
_WORLD = 4


def _exchange(flat: torch.Tensor) -> torch.Tensor:
    return torch.cat(list(reversed(flat.chunk(_WORLD))))


def _inplace(output, input, group=None):
    output.copy_(_exchange(input.flatten()))


def _functional(input, output_split_sizes, input_split_sizes, group):
    assert output_split_sizes is None and input_split_sizes is None
    return _exchange(input.flatten()).reshape(input.shape)


def _run(prefers_functional: bool, x: torch.Tensor):
    group = SimpleNamespace(ulysses_group=object())
    with (
        patch(f"{_USP}.get_sp_group", return_value=group),
        patch(
            f"{_USP}.current_platform.prefers_functional_collectives",
            return_value=prefers_functional,
        ),
        patch(
            f"{_USP}.torch.distributed.all_to_all_single", side_effect=_inplace
        ) as c10d,
        patch(f"{_USP}.ft_c.all_to_all_single", side_effect=_functional) as funcol,
    ):
        out = usp_mod._usp_all_to_all_single(x)
    return out, c10d, funcol


def test_functional_exchange_matches_the_flat_exchange():
    # [h_global, b, s_local, d] after the input permute; heads split evenly.
    x = torch.arange(8 * 2 * 3 * 5, dtype=torch.float32).reshape(8, 2, 3, 5)

    flat, c10d, funcol = _run(False, x)
    c10d.assert_called_once()
    funcol.assert_not_called()

    functional, c10d, funcol = _run(True, x)
    funcol.assert_called_once()
    c10d.assert_not_called()

    assert functional.shape == x.shape
    assert torch.equal(functional, flat)
