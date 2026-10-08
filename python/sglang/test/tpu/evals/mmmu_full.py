"""MMMU-Pro on a chosen row list: stride shards of the full set, or a fixed sample.

Only the row list changes: sgl-eval's own loader builds every example and its
own grader scores it, so a shard or sample is comparable with an unsharded run
of the same rows. The shards partition the set by stride, and the collector
rejects a missing or duplicated id before reporting the combined score.

Sharding is for a model too slow to finish the set inside one CI job; a sample
scores the same rows every night at a fraction of the cost. The caller supplies
the model and its sampling.
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

SHARD_RESULT_PREFIX = "MMMU_SHARD_RESULT "


def shard_rows(rows, shard, shards):
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("Invalid shard index/count")
    return list(rows)[shard::shards]


def sample_rows(rows, count, seed):
    """The same `count` rows for a given seed, in dataset order."""
    rows = list(rows)
    if not 0 < count <= len(rows):
        raise ValueError(f"Cannot sample {count} of {len(rows)} rows")
    picked = sorted(random.Random(seed).sample(range(len(rows)), count))
    return [rows[i] for i in picked]


def _score(base_url, selection, *, sampling, tag):
    """Run sgl-eval on one row selection; return its metrics and output dir."""
    output = Path(tempfile.mkdtemp(prefix=f"mmmu-pro-{tag}-"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            __name__,
            "--base-url",
            base_url,
            *selection,
            "--output",
            str(output),
            "--sampling",
            json.dumps(sampling),
        ],
        check=True,
    )
    files = list(output.glob("sgl_eval_mmmu_pro_*/metrics.json"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one metrics file, found {files}")
    return json.loads(files[0].read_text()), files[0].parent


def run_sample(
    base_url: str,
    *,
    count: int,
    seed: int,
    model: str,
    max_tokens: int,
    num_threads: int,
    temperature: float,
    top_p: float,
    chat_template_kwargs: dict | None = None,
) -> dict:
    """Score a fixed `count`-row sample of MMMU-Pro."""
    result, _ = _score(
        base_url,
        ["--sample", str(count), "--seed", str(seed)],
        sampling={
            "model": model,
            "max_tokens": max_tokens,
            "num_threads": num_threads,
            "temperature": temperature,
            "top_p": top_p,
            "chat_template_kwargs": chat_template_kwargs or {},
        },
        tag=f"sample-{count}",
    )
    if result["num_examples"] != count:
        raise RuntimeError(
            f"MMMU-Pro sample scored {result['num_examples']} of {count} rows"
        )
    return {
        **result["aggregate"],
        "latency": result["latency_seconds"],
        "output_throughput": result["output_throughput_tps"],
    }


def run_shard(
    base_url: str,
    *,
    shard: int,
    shards: int,
    model: str,
    max_tokens: int,
    num_threads: int,
    temperature: float,
    top_p: float,
    chat_template_kwargs: dict | None = None,
) -> dict:
    """Score one shard of MMMU-Pro and print its summary for the collector."""
    result, run_dir = _score(
        base_url,
        ["--shard", str(shard), "--shards", str(shards)],
        sampling={
            "model": model,
            "max_tokens": max_tokens,
            "num_threads": num_threads,
            "temperature": temperature,
            "top_p": top_p,
            "chat_template_kwargs": chat_template_kwargs or {},
        },
        tag=f"shard-{shard}",
    )
    ids = [
        json.loads(line)["id"]
        for line in (run_dir / "output-rs0.jsonl").read_text().splitlines()
    ]
    if result["aggregate"]["error_rate"] != 0:
        raise RuntimeError("MMMU-Pro shard contains request errors")
    if result["num_examples"] != len(ids):
        raise RuntimeError("MMMU-Pro shard is incomplete")

    from sglang.test.tpu.evals.full_eval_artifacts import archive_results

    archive_results(run_dir)
    summary = {
        "shard": shard,
        "shards": shards,
        "questions": result["num_examples"],
        "score": result["aggregate"]["score"],
        "correct": round(result["aggregate"]["score"] * result["num_examples"]),
        "ids": ids,
    }
    print(SHARD_RESULT_PREFIX + json.dumps(summary), flush=True)
    return {
        **result["aggregate"],
        "latency": result["latency_seconds"],
        "output_throughput": result["output_throughput_tps"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--shard", type=int)
    selection.add_argument("--sample", type=int, help="Score this many rows.")
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="Seed for --sample.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sampling", type=json.loads, required=True)
    args = parser.parse_args()

    from sgl_eval.cli import main as eval_main
    from sgl_eval.evals import _registry
    from sgl_eval.evals._mmmu_pro import load_mmmu_pro

    from sglang.test.tpu.evals.eval_support import sgl_eval_args

    def load_rows(split="test", num_examples=None):
        if num_examples is not None:
            raise ValueError("The row selection replaces sgl-eval's sample limit")
        examples = load_mmmu_pro(split)
        if not examples:
            raise ValueError("MMMU-Pro loaded no examples")
        if args.sample is not None:
            return sample_rows(examples, args.sample, args.seed)
        return shard_rows(examples, args.shard, args.shards)

    # Process-local: no installed package or other benchmark is modified.
    _registry.load_mmmu_pro = load_rows
    return eval_main(
        sgl_eval_args(
            "mmmu_pro",
            base_url=args.base_url,
            out_dir=args.output,
            n_repeats=1,
            **args.sampling,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
