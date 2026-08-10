"""Deprecated compatibility shim for the former ``MultiPlatformOp``.

The multi-platform operator abstraction was unified into
:class:`sglang.kernels.fused_op.BaseFusedOp` (RFC #29630): one class now
covers kernel-backend selection (``forward_aot`` / ``forward_jit`` / ...),
platform dispatch (``forward_cuda`` / ``forward_hip`` / ``forward_npu`` /
...), out-of-tree platform overrides (:meth:`BaseFusedOp.register_oot_forward`),
and the torch.compile enter/leave protocol.

<<<<<<< HEAD
from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.platforms import current_platform
from sglang.srt.utils import (
    cpu_has_amx_support,
    is_cpu,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    is_tpu,
    is_xpu,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_musa = is_musa()
_is_tpu = is_tpu()
=======
In-repo code must subclass ``BaseFusedOp`` directly. This alias exists only so
out-of-tree platform plugins and external users keep importing from the old
path while they migrate; it will be removed in a future release.
"""

import warnings

from sglang.kernels.fused_op import BaseFusedOp
>>>>>>> upstream/main


class MultiPlatformOp(BaseFusedOp):
    """Deprecated alias of :class:`sglang.kernels.fused_op.BaseFusedOp`.

    Kept attribute-compatible with the original class: ``forward_native`` is
    concrete here (raising ``NotImplementedError``) so existing plugin
    subclasses that only define platform forwards keep instantiating, and the
    old per-platform default methods (``forward_hip`` -> ``forward_cuda``,
    ``forward_cpu`` -> ``forward_native``, ...) remain callable for plugin
    code that invokes them directly.
    """

<<<<<<< HEAD
    @classmethod
    def register_oot_forward(cls, op_cls: type, fn: Callable, platform_key: str):
        """Register an OOT forward implementation for a specific op class and platform."""
        cls._oot_forward_registry.setdefault(platform_key, {})[op_cls] = fn

    def __init__(self):
        super().__init__()
        self._forward_method: Callable = self.dispatch_forward()

        # States for torch.compile
        self._original_forward_method = None
        self.is_torch_compile = False

    def enter_torch_compile(self, num_tokens: int):
        # Skip if Op is already entered compile mode.
        # NOTE(alcanderian): Some Ops(for example RotaryEmbedding) will be reused
        # among layers and `enter_torch_compile` will be called many times.
        # We should prevent `self._original_forward_method` from being overridden when
        # it is not the first time `enter_torch_compile` called.
        if self.is_torch_compile:
            return

        self._original_forward_method = self._forward_method
        # NOTE: Temporarily workaround MoE
        # The performance of torch.compile on this layer is not always good when bs > 1,
        # so we decide to only use torch.compile when bs=1
        if "FusedMoE" in self.__class__.__name__:
            if num_tokens == 1 and not _is_tpu:
                from sglang.srt.layers.moe.fused_moe_native import (
                    fused_moe_forward_native,
                )

                self._forward_method = fused_moe_forward_native
        elif "TopK" in self.__class__.__name__:
            if num_tokens == 1:
                self._forward_method = self.forward_native
        else:
            self._forward_method = self.forward_native
        self.is_torch_compile = True

    def leave_torch_compile(self):
        # Skip if Op is already exited compile mode.
        if not self.is_torch_compile:
            return

        self._forward_method = self._original_forward_method
        self._original_forward_method = None
        self.is_torch_compile = False

    # Please do not override this method, because `self._forward_method` can change when in torch compile mode
    @debug_kernel_api
    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)
=======
    def __init_subclass__(cls, **kwargs):
        warnings.warn(
            "MultiPlatformOp is deprecated; subclass "
            "sglang.kernels.fused_op.BaseFusedOp instead (RFC #29630).",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init_subclass__(**kwargs)
>>>>>>> upstream/main

    def forward_native(self, *args, **kwargs):
        raise NotImplementedError

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_hip(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_musa(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)

    def forward_npu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_xpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)
<<<<<<< HEAD

    def forward_tpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def dispatch_forward(self):
        # OOT platform dispatch: check registry then method lookup
        if current_platform.is_out_of_tree():
            key = current_platform.get_dispatch_key_name()
            oot = self._oot_forward_registry.get(key, {})
            if type(self) in oot:
                return oot[type(self)].__get__(self)
            method = getattr(self, f"forward_{key}", None)
            if method is not None:
                return method
            return self.forward_native

        if _is_cuda:
            return self.forward_cuda
        elif _is_hip:
            return self.forward_hip
        elif _is_tpu:
            return self.forward_tpu
        elif _is_cpu and _is_cpu_amx_available:
            return self.forward_cpu
        elif _is_npu:
            return self.forward_npu
        elif _is_xpu:
            return self.forward_xpu
        elif _is_musa:
            return self.forward_musa
        else:
            return self.forward_native
=======
>>>>>>> upstream/main
