"""Experimental raw VMEM views; every new view must be overwritten before reading."""
import dataclasses
import math
from jax import tree_util
from jax._src.state import types
from jax._src.pallas.mosaic import lowering
from jax._src.lib import tpu
from jax._src.lib.mlir import ir

@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class Reinterpret(types.ReshapeTransform):
    pass

# SGLANG-PATCH: document and guard this process-wide JAX patch.
# What: importing this module replaces the private
#   jax._src.pallas.mosaic.lowering._reshape_memref(ref, reshaper, ref_aval,
#   ref_block_shape) for the whole process, so every Pallas kernel lowered
#   afterwards (attention and MoE included) goes through _lower below.
# Why: Mosaic has no public way to view a VMEM ref under another shape, dtype
#   and tiled layout; the decode megakernel reuses one VMEM pool for every
#   layer's weight tiles through Reinterpret views. Transforms that are not
#   Reinterpret go to the original function unchanged.
# Targets: upstream pins jax 0.11.1; SGLang runs it on jax 0.10.2, which has
#   the same four-argument private function. The check below fails the import
#   instead of mis-lowering if a JAX bump changes it.
# When: only the Kimi-K3 megakernel imports this module, and only when
#   SGLANG_TPU_ENABLE_K3_MEGAKERNEL is set.
# Remove when: Pallas TPU exposes a public layout-reinterpreting ref view.
import inspect as _inspect
if len(_inspect.signature(lowering._reshape_memref).parameters) != 4:
    raise ImportError(
        "pool_alias replaces jax._src.pallas.mosaic.lowering._reshape_memref("
        "ref, reshaper, ref_aval, ref_block_shape); this JAX version changed it"
    )
_original = lowering._reshape_memref

def _lower(ref, transform, aval, block_shape):
    if not isinstance(transform, Reinterpret):
        return _original(ref, transform, aval, block_shape)
    transform._validate_shape(tuple(block_shape))
    original_type = ir.MemRefType(ref.type)
    bits = aval.dtype.itemsize * 8
    packing = 32 // bits
    rows = 8 * packing
    columns = transform.shape[-1] // 128
    strides = [columns, 1]
    minor_dimension = transform.shape[-2]
    for dimension in reversed(transform.shape[:-2]):
        minor_tiles = (minor_dimension + rows - 1) // rows
        strides.insert(0, strides[0] * minor_tiles)
        minor_dimension = dimension
    inner = f"({packing},1)" if packing > 1 else ""
    # SGLANG-PATCH: no reused quote inside the f-string; SGLang supports Python 3.10.
    layout = ir.Attribute.parse(f"#tpu.tiled<({rows},128){inner},[{','.join(map(str, strides))}]>")
    result_type = ir.MemRefType.get(transform.shape, original_type.element_type, layout=layout, memory_space=original_type.memory_space)
    alias = tpu.reinterpret_cast(result_type, ref)
    erased = ir.MemRefType.get(transform.shape, original_type.element_type, memory_space=original_type.memory_space)
    return tpu.erase_memref_layout(erased, alias), transform.shape

lowering._reshape_memref = _lower

def storage_size(shape, dtype):
    bits = dtype.itemsize * 8
    packing = 32 // bits
    rows = 8 * packing
    row_tiles = (shape[-2] + rows - 1) // rows
    return math.prod(shape[:-2]) * row_tiles * (shape[-1] // 128) * 4096

def view(ref, shape, dtype):
    ref = ref.bitcast(dtype)
    return types.TransformedRef(ref.ref, (*ref.transforms, Reinterpret(tuple(shape))))
