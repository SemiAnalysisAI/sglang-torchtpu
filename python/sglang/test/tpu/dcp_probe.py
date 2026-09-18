"""Direct validation that TPU decode context parallelism is on and correct.

An accuracy gate alone cannot carry this feature. DCP changes where each
token's KV lives, not what the model is asked to compute, so a server that
silently ignored ``--dcp-size`` would score exactly the same GSM8K as one that
honoured it -- and a merge that drops one rank's contribution still answers
most questions, because the surviving shard holds a strided sample of the
context. The gate therefore checks the mechanism, and leaves accuracy to the
GSM8K gate that runs beside it.

Three checks, none of which compares a measurement to a tuned constant:

1. ``/server_info`` reports the DCP group size the launch asked for. Catches a
   flag that never reached the engine.
2. For each prompt, a cold read and a read on a full prefix hit decode
   identical text. Under DCP those two take different paths -- the cold read
   runs the chunk-local causal pass and the cached pass and merges them, the
   warm read is almost all cached pass -- so a wrong position shard or a wrong
   log-sum-exp merge separates them. Content-independent: there is nothing to
   recalibrate when the model's output legitimately changes.
3. The answer buried in the prompt comes back in the opening of each reply.
   The needle sits in the middle of the filler, so the positions that carry it
   land on one rank of the group and the reply proves the merge brought that
   rank's contribution back.

The prompt set deliberately includes one prompt shorter than the group size's
stride, where some ranks of the group own no key at all -- the shard the
kernel has to carry through one guaranteed block iteration.
"""

from typing import Any, Dict, List

import requests

from sglang.test.test_utils import dump_metric

# Paragraphs used as filler. Each prompt rotates the order and carries its own
# leading tag, so no two prompts share an opening: a nested prompt set would
# measure incremental extends against a warm radix cache rather than prompts
# of that length.
_PARAS = (
    "The logistics team reviewed the quarterly inventory and found the pallet "
    "counts consistent with the warehouse ledger.",
    "Maintenance scheduled the compressor overhaul for the third week, pending "
    "the arrival of the replacement seals.",
    "The compliance office circulated the revised handling procedure and asked "
    "every shift lead to acknowledge receipt.",
    "Weather delays pushed two inbound containers to the following Monday "
    "without affecting the downstream assembly plan.",
    "A sampling audit of the cold chain sensors showed no excursions beyond "
    "the permitted temperature band.",
    "The night crew reported a jammed conveyor at station four, cleared within "
    "the hour and logged as a minor stoppage.",
    "Procurement consolidated three small orders into a single shipment to "
    "reduce the per-unit freight charge.",
    "The safety briefing covered the new lockout tags and the revised muster "
    "point behind the loading dock.",
)

# (case, filler paragraphs, needle). "tiny" is the degenerate shard case: at
# any group size above one, several ranks own no key for it.
_CASES = (
    ("tiny", 0, "QUJ-417"),
    ("short", 8, "TKR-882"),
    ("mid", 80, "HVN-536"),
    ("long", 260, "PLD-193"),
)

_HEAD_CHARS = 80
_MAX_NEW_TOKENS = 48


def _build_prompt(index: int, case: str, num_paras: int, needle: str) -> str:
    rotation = index + 1
    body: List[str] = []
    for i in range(num_paras):
        body.append(_PARAS[(i + rotation) % len(_PARAS)])
        if i == num_paras // 2:
            body.append(f"Operations note: the access code for this memo is {needle}.")
    if not body:
        body.append(f"Operations note: the access code for this memo is {needle}.")
    return (
        f"Memo {case}-{index:02d}-{rotation * 7 + 3}. Read the following record.\n\n"
        + "\n".join(body)
        + "\n\nQuestion: what is the access code for this memo? "
        "Reply with the code first, then anything else you want to say.\n"
    )


def _generate(base_url: str, prompt: str) -> Dict[str, Any]:
    response = requests.post(
        base_url.rstrip("/") + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": _MAX_NEW_TOKENS,
            },
        },
        timeout=1800,
    )
    response.raise_for_status()
    return response.json()


def _server_dcp_size(base_url: str) -> Any:
    response = requests.get(base_url.rstrip("/") + "/server_info", timeout=60)
    response.raise_for_status()
    info = response.json()
    # /server_info nests the resolved args under "server_args" on some builds
    # and flattens them on others; both are the engine's own echo.
    if "dcp_size" in info:
        return info["dcp_size"]
    return info.get("server_args", {}).get("dcp_size")


def dcp_runner(*, test_name: str, dcp_size: int):
    """Build a gate runner that proves DCP is engaged and self-consistent.

    ``dcp_size`` is the group size the launch env asked for; the runner fails
    if the engine reports anything else.
    """
    if dcp_size < 2:
        raise ValueError(f"dcp_runner is for a real group, got dcp_size={dcp_size}")

    def run(base_url: str) -> Dict[str, Any]:
        failures: List[str] = []

        reported = _server_dcp_size(base_url)
        if reported != dcp_size:
            failures.append(
                f"/server_info reports dcp_size={reported!r}, launch asked for "
                f"{dcp_size}: the flag never reached the engine"
            )

        needle_hits = 0
        identical = 0
        for index, (case, num_paras, needle) in enumerate(_CASES):
            prompt = _build_prompt(index, case, num_paras, needle)

            flush = requests.post(base_url.rstrip("/") + "/flush_cache", timeout=60)
            flush.raise_for_status()

            cold = _generate(base_url, prompt)
            warm = _generate(base_url, prompt)
            cold_text, warm_text = cold["text"], warm["text"]
            cold_cached = cold["meta_info"].get("cached_tokens")
            warm_cached = warm["meta_info"].get("cached_tokens")
            prompt_tokens = cold["meta_info"]["prompt_tokens"]

            print(
                f"[dcp] {test_name}/{case}: prompt_tokens={prompt_tokens} "
                f"cached cold={cold_cached} warm={warm_cached} "
                f"cold_head={cold_text[:_HEAD_CHARS]!r}",
                flush=True,
            )

            if cold_text == warm_text:
                identical += 1
            else:
                failures.append(
                    f"{case}: the cold read and the prefix-hit read of the same "
                    f"prompt decoded different text "
                    f"(cold={cold_text[:_HEAD_CHARS]!r}, "
                    f"warm={warm_text[:_HEAD_CHARS]!r})"
                )

            if needle in cold_text[:_HEAD_CHARS]:
                needle_hits += 1
            else:
                failures.append(
                    f"{case}: the access code {needle} buried at position "
                    f"~{prompt_tokens // 2} of {prompt_tokens} did not come back "
                    f"in the opening of the reply "
                    f"({cold_text[:_HEAD_CHARS]!r})"
                )

        labels = {"model": test_name, "kind": "info", "probe": "dcp"}
        dump_metric(f"{test_name}.dcp_size", float(dcp_size), labels=labels)
        dump_metric(f"{test_name}.dcp_needle_hits", float(needle_hits), labels=labels)
        dump_metric(
            f"{test_name}.dcp_warm_cold_identical", float(identical), labels=labels
        )
        print(
            f"[dcp] {test_name}: dcp_size={reported} "
            f"needle_hits={needle_hits}/{len(_CASES)} "
            f"warm_cold_identical={identical}/{len(_CASES)}",
            flush=True,
        )

        return {
            "score": float(needle_hits),
            "_passed": not failures,
            "_failures": failures,
        }

    return run
