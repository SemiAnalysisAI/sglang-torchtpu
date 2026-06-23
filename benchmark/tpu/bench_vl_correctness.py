#!/usr/bin/env python3
# Correctness (accuracy) helper for VL models served by an sglang torch_tpu
# server. Companion to ./bench_vl_serving.sh — that script measures
# throughput/TTFT/latency but does NOT grade answers; this one actually checks
# whether the model answers the multiple-choice question correctly.
#
# It is a *thin* driver over the existing MMMU harness in ../mmmu:
#   - sample loading / prompt+image construction  -> mmmu.eval_utils.prepare_samples
#   - multiple-choice answer parsing (robust to long Thinking reasoning)
#                                                  -> mmmu.eval_utils.parse_multi_choice_response
#   - open-question parsing / grading              -> mmmu.eval_utils.{parse_open_response,
#                                                       eval_multi_choice, eval_open}
# We do NOT modify ../mmmu — it is imported read-only. The only thing this file
# adds on top of mmmu/bench_sglang.py is a clean --num-questions cap (so a tiny
# labeled sample can be graded against a slow server) and a fixed
# concurrency=1, sequential send loop (required while the TPU server crashes on
# concurrent multimodal batches).
#
# Invoked by ./bench_vl_correctness.sh; see that script's header for the env
# contract. Can also be run directly:
#   python3 bench_vl_correctness.py --model Qwen/Qwen3-VL-30B-A3B-Thinking \
#       --port 30000 --num-questions 10 --output-dir ./results/.../ts

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path

import openai

# Import the existing MMMU harness read-only. Its modules use bare imports
# (e.g. `from data_utils import ...`), so its directory must be on sys.path.
_MMMU_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "mmmu"))
if _MMMU_DIR not in sys.path:
    sys.path.insert(0, _MMMU_DIR)

from eval_utils import (  # noqa: E402  (must follow sys.path tweak above)
    EvalArgs,
    eval_multi_choice,
    eval_open,
    get_sampling_params,
    parse_multi_choice_response,
    parse_open_response,
    prepare_samples,
)


def _split_prefix_suffix(prompt):
    """Split an MMMU prompt around its image placeholder, same as bench_sglang.py."""
    prefix = prompt.split("<")[0]
    suffix = prompt.split(">", 1)[1]
    return prefix, suffix


def _image_url(sample):
    """Build a data: URL (or pass through a real URL) for the sample's image."""
    image_path = sample["image_path"]
    if image_path and not image_path.startswith(("http://", "https://", "data:")):
        p = Path(image_path)
        mime = mimetypes.guess_type(str(p))[0] or "image/png"
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{b64}"
    return image_path


async def _query_one(client, sample, sampling_params, model, reasoning_effort):
    """Send one image+question prompt; return the raw model text (or None)."""
    prefix, suffix = _split_prefix_suffix(sample["final_input_prompt"])
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prefix},
                    {"type": "image_url", "image_url": {"url": _image_url(sample)}},
                    {"type": "text", "text": suffix},
                ],
            }
        ],
        **sampling_params,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
    response = await client.chat.completions.create(**payload)
    msg = response.choices[0].message
    content = msg.content
    # Thinking models may put everything in reasoning_content with empty content.
    if not content:
        content = getattr(msg, "reasoning_content", None)
    return content


def _grade(sample, response, response_answer_regex):
    """Parse the model answer and grade it. Returns (parsed_pred, is_correct)."""
    if response is None:
        return None, False
    answer = re.search(response_answer_regex, response)
    response_to_parse = answer.group(1).strip() if answer else response
    if sample["question_type"] == "multiple-choice":
        parsed = parse_multi_choice_response(
            response_to_parse, sample["all_choices"], sample["index2ans"]
        )
        correct = eval_multi_choice(sample["answer"], parsed)
    else:  # open question
        parsed = parse_open_response(response_to_parse)
        correct = eval_open(sample["answer"], parsed)
    return parsed, correct


async def run(args):
    eval_args = EvalArgs.from_cli_args(args)
    sampling_params = get_sampling_params(eval_args)
    reasoning_effort = eval_args.reasoning_effort

    samples = prepare_samples(eval_args)  # deterministic: sorted by prompt
    if args.num_questions and args.num_questions > 0:
        samples = samples[: args.num_questions]
    total = len(samples)
    print(f"Grading {total} MMMU question(s) at concurrency=1 against "
          f"{args.base_url} (model={args.model})")

    client = openai.AsyncOpenAI(
        api_key="sk",
        base_url=args.base_url.rstrip("/") + "/v1",
        timeout=args.request_timeout,
    )

    records = []
    num_correct = 0
    num_error = 0
    start = time.perf_counter()
    for i, sample in enumerate(samples):
        t0 = time.perf_counter()
        try:
            response = await _query_one(
                client, sample, sampling_params, args.model, reasoning_effort
            )
            err = None
        except Exception as e:  # noqa: BLE001 - record and keep going
            response, err = None, repr(e)
            num_error += 1
        parsed, correct = _grade(
            sample, response, eval_args.response_answer_regex
        )
        num_correct += int(correct)
        dt = time.perf_counter() - t0
        print(
            f"[{i + 1}/{total}] id={sample['id']} "
            f"gt={sample['answer']!r} pred={parsed!r} "
            f"{'OK ' if correct else 'X  '} ({dt:.1f}s)"
            + (f" ERROR={err}" if err else "")
        )
        records.append(
            {
                "id": sample["id"],
                "question_type": sample["question_type"],
                "ground_truth": sample["answer"],
                "parsed_pred": parsed,
                "correct": correct,
                "latency_s": round(dt, 2),
                "error": err,
                "response": response,
            }
        )

    elapsed = time.perf_counter() - start
    accuracy = (num_correct / total) if total else 0.0
    summary = {
        "model": args.model,
        "port": args.port,
        "dataset": eval_args.dataset_path,
        "split": eval_args.split,
        "num_questions": total,
        "num_correct": num_correct,
        "num_error": num_error,
        "accuracy": round(accuracy, 4),
        "max_new_tokens": eval_args.max_new_tokens,
        "temperature": eval_args.temperature,
        "reasoning_effort": reasoning_effort,
        "response_answer_regex": eval_args.response_answer_regex,
        "elapsed_s": round(elapsed, 2),
        "concurrency": 1,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    detail_path = os.path.join(args.output_dir, "details.jsonl")
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("==== VL correctness summary ====")
    print(json.dumps(summary, indent=2))
    print(f"-> per-question detail: {detail_path}")
    print(f"-> summary:             {summary_path}")
    print(f"Accuracy: {num_correct}/{total} = {accuracy:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True,
                        help="Model name sent in API requests; match the served model.")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL of the running sglang server.")
    parser.add_argument("--port", type=int, default=30000,
                        help="Deprecated; used only when --base-url is unset.")
    parser.add_argument("--num-questions", type=int, default=10,
                        help="Cap on number of MMMU questions to grade (0 = all). "
                             "Samples are taken in a deterministic order.")
    parser.add_argument("--output-dir", type=str, default="./results/vl-correctness",
                        help="Directory for summary.json + details.jsonl.")
    parser.add_argument("--request-timeout", type=float, default=1800.0,
                        help="Per-request client timeout in seconds (image prefills "
                             "can be slow on TPU).")
    # Reuse the MMMU eval args (--dataset-path, --split, --max-new-tokens,
    # --temperature, --response-answer-regex, --reasoning-effort, --seed, etc.).
    EvalArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.base_url = args.base_url or f"http://127.0.0.1:{args.port}"
    return args


def main():
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
