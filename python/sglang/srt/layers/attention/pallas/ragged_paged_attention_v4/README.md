# Ragged-paged attention v4

These files are copied from
`msl_tpu_kernel/kernels/ragged_paged_attention/kernel` at
`msl-tpu-kernel@384dac35824dce1fe5e26a5b135bb85630bb2a61` (MIT).

That is the commit used by the qualified Tier5 vLLM TPU deployment. The RPA
subtree is byte-identical at local source-of-truth commit
`3f3e0164efe53d6a0edf73f9ab24a49bd2162f37`. Only package-relative imports,
the package docstring, the environment-variable prefix, and the SGLang module
location differ from the pinned source.
