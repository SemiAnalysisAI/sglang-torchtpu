"""Direct validation that TPU decode context parallelism is on and correct.

An accuracy gate alone cannot carry this feature. DCP changes where each
token's KV lives, not what the model is asked to compute, so a server that
silently ignored ``--dcp-size`` would score exactly the same GSM8K as one that
honoured it -- and a merge that drops one rank's contribution still answers
fluently, because the surviving shards hold a strided sample of the context.
The gate therefore checks the mechanism and leaves accuracy to the GSM8K gate
beside it.

Each prompt buries THREE access codes at spread positions and asks for all
three. That is the shape that tests DCP specifically: rank ``r`` of a group of
``N`` owns absolute position ``p`` iff ``p % N == r``, so codes at different
offsets are carried by different ranks and a lost rank costs the codes it
held. The prompt set includes one prompt shorter than the group's stride, the
degenerate shard where several ranks own no key at all.

Three checks, none of which compares a measurement to a tuned constant:

1. ``/server_info`` reports the DCP group size the launch asked for. Catches a
   flag that never reached the engine.
2. Every access code comes back. The model under test may be a reasoning
   model, so the reply opens with chain of thought: the budget reaches the
   answer and the match is over the whole reply, not its opening.
3. Two reads served entirely from the prefix cache decode identical text.

Note what is deliberately NOT asserted: that a cold read and a prefix-hit read
agree. That is not an invariant even on a TP control -- the cold read consumes
the prefix at compute dtype while the hit reads it back from the KV cache, and
a reasoning model turns that epsilon into different wording. Measured on
gpt-oss-120B TP8, cdk job j-db97feb2, where two of four prompts diverged with
no DCP in the picture at all.
"""

from typing import Any, Dict, List, Sequence, Tuple

import requests

from sglang.test.test_utils import dump_metric

# Filler paragraphs. Each prompt rotates the order and carries its own leading
# tag, so no two prompts share an opening: a nested prompt set would measure
# incremental extends against a warm radix cache rather than prompts of that
# length.
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

# (case, filler paragraphs, codes). "tiny" is the degenerate shard case.
_CASES: Tuple[Tuple[str, int, Tuple[str, str, str]], ...] = (
    ("tiny", 0, ("QUJ-417", "QUJ-418", "QUJ-419")),
    ("short", 12, ("TKR-880", "TKR-881", "TKR-882")),
    ("mid", 90, ("HVN-534", "HVN-535", "HVN-536")),
    ("long", 280, ("PLD-191", "PLD-192", "PLD-193")),
)

# Fractions of the filler at which the three codes sit.
_CODE_POSITIONS = (0.15, 0.5, 0.85)
# Enough budget to get past a reasoning model's chain of thought.
_MAX_NEW_TOKENS = 384
_TAIL_CHARS = 200


def _build_prompt(index: int, case: str, num_paras: int, codes: Sequence[str]) -> str:
    rotation = index + 1
    if not num_paras:
        body = [f"Operations note: access codes {', '.join(codes)} apply."]
    else:
        at = {
            max(0, int(num_paras * f)): code
            for f, code in zip(_CODE_POSITIONS, codes)
        }
        body = []
        for i in range(num_paras):
            if i in at:
                body.append(
                    f"Operations note: access code {at[i]} applies to this section."
                )
            body.append(_PARAS[(i + rotation) % len(_PARAS)])
    return (
        f"Memo {case}-{index:02d}-{rotation * 7 + 3}. Read the following record.\n\n"
        + "\n".join(body)
        + "\n\nQuestion: list all three access codes that appear in this memo.\n"
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
    # /server_info flattens the resolved args on some builds and nests them
    # under "server_args" on others; both are the engine's own echo.
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

        recalled = 0
        wanted = 0
        agreed = 0
        for index, (case, num_paras, codes) in enumerate(_CASES):
            prompt = _build_prompt(index, case, num_paras, codes)

            flush = requests.post(base_url.rstrip("/") + "/flush_cache", timeout=60)
            flush.raise_for_status()

            cold = _generate(base_url, prompt)
            warm_a = _generate(base_url, prompt)
            warm_b = _generate(base_url, prompt)

            found = [code for code in codes if code in cold["text"]]
            missing = [code for code in codes if code not in found]
            recalled += len(found)
            wanted += len(codes)
            warm_cached = warm_a["meta_info"].get("cached_tokens")
            prompt_tokens = cold["meta_info"]["prompt_tokens"]

            print(
                f"[dcp] {test_name}/{case}: prompt_tokens={prompt_tokens} "
                f"warm_cached={warm_cached} codes={len(found)}/{len(codes)} "
                f"missing={missing} "
                f"cold_tail={cold['text'][-_TAIL_CHARS:]!r}",
                flush=True,
            )

            if missing:
                failures.append(
                    f"{case}: access codes {missing} buried in a "
                    f"{prompt_tokens}-token context did not come back "
                    f"(reply tail {cold['text'][-_TAIL_CHARS:]!r})"
                )

            if warm_a["text"] == warm_b["text"]:
                agreed += 1
            else:
                failures.append(
                    f"{case}: two reads served from the prefix cache decoded "
                    f"different text "
                    f"(a={warm_a['text'][-_TAIL_CHARS:]!r}, "
                    f"b={warm_b['text'][-_TAIL_CHARS:]!r})"
                )

            if not warm_cached:
                failures.append(
                    f"{case}: the second read reported cached_tokens="
                    f"{warm_cached!r}, so the prefix-cache comparison above "
                    f"did not test the cached path"
                )

        labels = {"model": test_name, "kind": "info", "probe": "dcp"}
        dump_metric(f"{test_name}.dcp_size", float(dcp_size), labels=labels)
        dump_metric(f"{test_name}.dcp_codes_recalled", float(recalled), labels=labels)
        dump_metric(f"{test_name}.dcp_cached_reads_agree", float(agreed), labels=labels)
        print(
            f"[dcp] {test_name}: dcp_size={reported} codes={recalled}/{wanted} "
            f"cached_reads_agree={agreed}/{len(_CASES)}",
            flush=True,
        )

        return {
            "score": float(recalled),
            "_passed": not failures,
            "_failures": failures,
        }

    return run
