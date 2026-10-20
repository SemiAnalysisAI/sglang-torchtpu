# HiCache write_back copy-ahead: open reproduction

On a hybrid sliding-window model under an agentic multi-turn load, sglang-torchtpu's `write_back`
copy-ahead window ([sgl-project/sglang-torchtpu#1517](https://github.com/sgl-project/sglang-torchtpu/pull/1517))
lowers the cache hit rate from the first minutes. The hypothesis behind
[sgl-project/sglang-torchtpu#1586](https://github.com/sgl-project/sglang-torchtpu/pull/1586): the window
host-locks every component of each victim, and the SWA host lock takes a sliding window of ancestors off
the SWA host LRU, so SWA host eviction deletes host leaves instead. #1586's first commit locks only the
Full copy; that is the fix measured here.

This package reproduces both with open components: `openai/gpt-oss-120b` on one tpu7x-8 host, and a
synthetic stdlib load generator.

- [REPRODUCE.md](REPRODUCE.md): step-by-step instructions, expected results and pitfalls.
- [results.md](results.md): what we measured.
- `agentic_hicache.py`: the server config (copy into `benchmark/tpu/recipe/configs/gpt_oss_120b/`).
- `agentic_multiturn_client.py`: the load generator.
- `compare.py`: turns client output and server logs into the comparison table.
- `arm_d_full_pin.patch`: arm D's code change, in case its commit is unreachable.
