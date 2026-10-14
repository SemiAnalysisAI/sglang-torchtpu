"""Session-affinity routing policy for MiniLB.

Places requests the way the production P/D router does, as reconstructed from
its decision logs, so PD tests behind MiniLB see production-like placement:

- Prefill is sticky per routing key (the x-routing-key request header). A known
  key stays on its pinned engine until a load gate trips, then spills to the
  least loaded sampled engine and is re-pinned there. An unknown key goes to the
  less loaded of the sampled engines and is pinned there. A request without a
  key is placed by load alone.
- Decode has no affinity. Each request goes to the less loaded of the sampled
  engines.
- Loads come from polling every engine's /v1/loads once per interval, so picks
  act on loads up to one interval old.
- An engine over a guardrail is never picked. When no sampled engine passes,
  the request is rejected with 429.
- A failed request holds its prefill/decode pair out of decode selection for a
  cooldown. A later success on the pair releases it.

Not reproduced: the production "halo" load score (raw /v1/loads counts are
used instead), engine ejection after repeated failures, and the sticky shortcut
that skips sampling at low in-flight load.
"""

import asyncio
import collections
import logging
import random
import time
from typing import Callable, Dict, FrozenSet, List, NamedTuple, Optional, Set, Tuple

import aiohttp
import orjson

logger = logging.getLogger(__name__)

POLICY_STICKY = "sticky"
POLICY_NO_AFFINITY = "fallback_no_affinity"
POLICY_OVERLOADED = "fallback_overloaded"


class SpilloverGate(NamedTuple):
    """When a pinned engine counts as overloaded against sampled references.

    The pinned engine spills once its load is at least min_load and above
    multiplier times the least loaded reference.
    """

    multiplier: float
    min_load: float


class PoolConfig(NamedTuple):
    """Placement rules for one engine pool."""

    name: str
    # /v1/loads core field that ranks engines, summed over DP ranks.
    load_metric: str
    num_reference_engines: int
    # (metric, threshold) pairs. An engine with metric > threshold is not picked.
    guardrails: Tuple[Tuple[str, float], ...]
    # Pins routing keys to engines when set. None places every request by load.
    spillover_gate: Optional[SpilloverGate]


# Values follow the production router's configuration. The gate's min_load is
# inferred from its rejections, it never spilled a pinned load below about 6.
PREFILL_POOL = PoolConfig(
    name="prefill",
    load_metric="num_waiting_reqs",
    num_reference_engines=2,
    guardrails=(("num_waiting_uncached_tokens", 2_000_000),),
    spillover_gate=SpilloverGate(multiplier=3.0, min_load=6.0),
)
DECODE_POOL = PoolConfig(
    name="decode",
    load_metric="num_running_reqs",
    num_reference_engines=2,
    guardrails=(("num_waiting_reqs", 24),),
    spillover_gate=None,
)


def _format_load(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.2f}"


class EngineSample(NamedTuple):
    url: str
    load: float


class PoolDecision(NamedTuple):
    """Which engine of a pool serves a request, and why."""

    pool: str
    engine: str
    # POLICY_STICKY, POLICY_NO_AFFINITY or POLICY_OVERLOADED.
    policy: str
    load_name: str
    load: float
    sampled_references: Tuple[EngineSample, ...]
    # Why the pinned engine was refused: "primary:load" or "guardrail:<metric>".
    rejection: Optional[str]

    def to_log_json(self, *, room: Optional[int], routing_key: Optional[str]) -> str:
        return orjson.dumps(
            {
                "room": room,
                "pool": self.pool,
                "decision": self.engine,
                "policy": self.policy,
                "rejection": self.rejection,
                "key": routing_key,
                "load_name": self.load_name,
                "load": self.load,
                "sampled_references": [
                    [sample.url, sample.load] for sample in self.sampled_references
                ],
            }
        ).decode()


class EngineOverloaded(Exception):
    """No sampled engine of a pool passes its guardrails."""

    def __init__(self, *, engine: str, metric: str, load: float):
        super().__init__(
            f"engine {engine} overloaded on {metric}: load={_format_load(load)}"
        )


class EngineLoads:
    """Latest polled load metrics per engine URL, and which engines are down.

    An engine never polled reads as idle and reachable.
    """

    def __init__(self):
        self._metrics: Dict[str, Dict[str, float]] = {}
        self._unreachable: Set[str] = set()

    def update(self, url: str, metrics: Dict[str, float]) -> None:
        self._metrics[url] = metrics
        self._unreachable.discard(url)

    def mark_unreachable(self, url: str) -> None:
        self._unreachable.add(url)

    def is_reachable(self, url: str) -> bool:
        return url not in self._unreachable

    def get(self, url: str, metric: str) -> float:
        return self._metrics.get(url, {}).get(metric, 0.0)


def _sum_over_dp_ranks(payload: dict, metric_names: FrozenSet[str]) -> Dict[str, float]:
    totals = dict.fromkeys(metric_names, 0.0)
    for rank_load in payload["loads"]:
        for name in metric_names:
            totals[name] += rank_load[name]
    return totals


class LoadPoller:
    """Polls every engine's /v1/loads once per interval into an EngineLoads.

    Each engine has at most one poll in flight, so a slow engine delays only its
    own refresh. A connection error marks the engine unreachable until a poll
    succeeds again. A timeout or an error status keeps its last load.
    """

    def __init__(
        self,
        *,
        loads: EngineLoads,
        metric_names: FrozenSet[str],
        interval_secs: float,
        timeout_secs: float,
    ):
        self._loads = loads
        self._metric_names = metric_names
        self._interval_secs = interval_secs
        self._timeout_secs = timeout_secs

    async def run(self, get_engine_urls: Callable[[], List[str]]) -> None:
        polls: Dict[str, asyncio.Task] = {}
        timeout = aiohttp.ClientTimeout(total=self._timeout_secs)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                while True:
                    for url in get_engine_urls():
                        poll = polls.get(url)
                        if poll is None or poll.done():
                            polls[url] = asyncio.create_task(
                                self._poll(session=session, url=url)
                            )
                    await asyncio.sleep(self._interval_secs)
            finally:
                for poll in polls.values():
                    poll.cancel()
                await asyncio.gather(*polls.values(), return_exceptions=True)

    async def _poll(self, *, session: aiohttp.ClientSession, url: str) -> None:
        try:
            async with session.get(
                f"{url}/v1/loads", params={"include": "core,queues"}
            ) as response:
                if response.status != 200:
                    return
                payload = await response.json()
        except asyncio.TimeoutError:
            return
        except aiohttp.ClientError as e:
            if self._loads.is_reachable(url):
                logger.warning(f"[MiniLB] engine {url} is unreachable: {e!r}")
            self._loads.mark_unreachable(url)
            return

        try:
            metrics = _sum_over_dp_ranks(payload, self._metric_names)
        except (KeyError, TypeError) as e:
            logger.warning(f"[MiniLB] malformed /v1/loads from {url}: {e!r}")
            return
        if not self._loads.is_reachable(url):
            logger.info(f"[MiniLB] engine {url} is reachable again")
        self._loads.update(url, metrics)


class SessionTable:
    """Routing key to pinned engine URL, least recently used key evicted first."""

    def __init__(self, *, max_keys: int):
        self._max_keys = max_keys
        self._pins: "collections.OrderedDict[str, str]" = collections.OrderedDict()

    def lookup(self, key: str) -> Optional[str]:
        engine = self._pins.get(key)
        if engine is not None:
            self._pins.move_to_end(key)
        return engine

    def pin(self, key: str, engine: str) -> None:
        self._pins[key] = engine
        self._pins.move_to_end(key)
        if len(self._pins) > self._max_keys:
            self._pins.popitem(last=False)


class PairBreaker:
    """Holds failed prefill/decode pairs out of decode selection for a cooldown."""

    def __init__(
        self, *, holdout_secs: float, clock: Callable[[], float] = time.monotonic
    ):
        self._holdout_secs = holdout_secs
        self._clock = clock
        self._held_until: Dict[Tuple[str, str], float] = {}

    def is_held_out(self, prefill_url: str, decode_url: str) -> bool:
        held_until = self._held_until.get((prefill_url, decode_url))
        return held_until is not None and held_until > self._clock()

    def trip(self, prefill_url: str, decode_url: str, reason: str) -> None:
        self._held_until[(prefill_url, decode_url)] = self._clock() + self._holdout_secs
        logger.warning(
            f'[MiniLB] pair breaker tripped for ("{prefill_url}", "{decode_url}"): '
            f"holding out for {self._holdout_secs:g}s after {reason}"
        )

    def reset(self, prefill_url: str, decode_url: str) -> None:
        if self._held_until.pop((prefill_url, decode_url), None) is not None:
            logger.info(
                f'[MiniLB] pair breaker reset for ("{prefill_url}", "{decode_url}"): '
                "retry succeeded"
            )


class SessionAffinityPolicy:
    """Picks a prefill and a decode engine per request, as production does.

    The module docstring lists the placement rules. Picks read the loads that
    run_load_polling keeps refreshing, and record_success / record_failure feed
    the pair breaker.
    """

    def __init__(
        self,
        *,
        prefill_pool: PoolConfig = PREFILL_POOL,
        decode_pool: PoolConfig = DECODE_POOL,
        pair_holdout_secs: float = 300.0,
        load_poll_interval_secs: float = 1.0,
        load_poll_timeout_secs: float = 5.0,
        # Arbitrary bound that keeps memory flat when every request has a new key.
        max_session_keys: int = 1_000_000,
        rng: Optional[random.Random] = None,
    ):
        self.loads = EngineLoads()
        self._prefill_pool = prefill_pool
        self._decode_pool = decode_pool
        self._sessions = SessionTable(max_keys=max_session_keys)
        self._pair_breaker = PairBreaker(holdout_secs=pair_holdout_secs)
        self._load_poller = LoadPoller(
            loads=self.loads,
            metric_names=frozenset(
                name
                for pool in (prefill_pool, decode_pool)
                for name in (pool.load_metric, *(m for m, _ in pool.guardrails))
            ),
            interval_secs=load_poll_interval_secs,
            timeout_secs=load_poll_timeout_secs,
        )
        self._rng = rng if rng is not None else random.Random()
        # Pools currently routing to all engines because none is eligible.
        self._pools_failing_open: Set[str] = set()

    async def run_load_polling(self, get_engine_urls: Callable[[], List[str]]) -> None:
        await self._load_poller.run(get_engine_urls)

    def select(
        self,
        *,
        prefill_urls: List[str],
        decode_urls: List[str],
        routing_key: Optional[str],
    ) -> Tuple[PoolDecision, PoolDecision]:
        """Picks the prefill and decode engines of one request.

        Raises EngineOverloaded when either pool has no engine to take it. The
        routing key is pinned only when both pools accept the request.
        """
        prefill = self._select_prefill(urls=prefill_urls, routing_key=routing_key)
        decode_candidates = self._eligible(
            pool=self._decode_pool,
            urls=decode_urls,
            is_excluded=lambda url: (
                not self.loads.is_reachable(url)
                or self._pair_breaker.is_held_out(prefill.engine, url)
            ),
        )
        decode = self._select_by_load(
            pool=self._decode_pool, candidates=decode_candidates
        )
        if routing_key is not None:
            self._sessions.pin(routing_key, prefill.engine)
        return prefill, decode

    def record_success(self, *, prefill_url: str, decode_url: str) -> None:
        self._pair_breaker.reset(prefill_url, decode_url)

    def record_failure(self, *, prefill_url: str, decode_url: str, reason: str) -> None:
        self._pair_breaker.trip(prefill_url, decode_url, reason)

    def _select_prefill(
        self, *, urls: List[str], routing_key: Optional[str]
    ) -> PoolDecision:
        pool = self._prefill_pool
        candidates = self._eligible(
            pool=pool,
            urls=urls,
            is_excluded=lambda url: not self.loads.is_reachable(url),
        )
        pinned = None
        if routing_key is not None and pool.spillover_gate is not None:
            pinned = self._sessions.lookup(routing_key)
        if pinned is None or pinned not in candidates:
            return self._select_by_load(pool=pool, candidates=candidates)

        references = self._sample(pool=pool, candidates=candidates)
        rejection = self._gate_rejection(
            pool=pool, pinned=pinned, references=references
        )
        if rejection is None:
            return self._decision(
                pool=pool, engine=pinned, policy=POLICY_STICKY, references=references
            )

        spill_targets = tuple(ref for ref in references if ref.url != pinned)
        spill = self._least_loaded_passing(pool=pool, samples=spill_targets)
        if spill is None:
            raise self._overloaded(pool=pool, samples=spill_targets or references)
        return self._decision(
            pool=pool,
            engine=spill.url,
            policy=POLICY_OVERLOADED,
            references=references,
            rejection=rejection,
        )

    def _select_by_load(
        self, *, pool: PoolConfig, candidates: List[str]
    ) -> PoolDecision:
        samples = self._sample(pool=pool, candidates=candidates)
        choice = self._least_loaded_passing(pool=pool, samples=samples)
        if choice is None:
            raise self._overloaded(pool=pool, samples=samples)
        return self._decision(
            pool=pool, engine=choice.url, policy=POLICY_NO_AFFINITY, references=samples
        )

    def _eligible(
        self,
        *,
        pool: PoolConfig,
        urls: List[str],
        is_excluded: Callable[[str], bool],
    ) -> List[str]:
        """Engines not excluded, or every engine when all are (fail-open)."""
        eligible = [url for url in urls if not is_excluded(url)]
        if eligible:
            self._pools_failing_open.discard(pool.name)
            return eligible
        if pool.name not in self._pools_failing_open:
            self._pools_failing_open.add(pool.name)
            logger.warning(
                f"[MiniLB] pool {pool.name} fail-open: all engines excluded, "
                "routing anyway"
            )
        return list(urls)

    def _sample(
        self, *, pool: PoolConfig, candidates: List[str]
    ) -> Tuple[EngineSample, ...]:
        count = min(pool.num_reference_engines, len(candidates))
        return tuple(
            EngineSample(url, self.loads.get(url, pool.load_metric))
            for url in self._rng.sample(candidates, count)
        )

    def _guardrail_violation(
        self, *, pool: PoolConfig, url: str
    ) -> Optional[Tuple[str, float]]:
        for metric, threshold in pool.guardrails:
            value = self.loads.get(url, metric)
            if value > threshold:
                return metric, value
        return None

    def _gate_rejection(
        self,
        *,
        pool: PoolConfig,
        pinned: str,
        references: Tuple[EngineSample, ...],
    ) -> Optional[str]:
        violation = self._guardrail_violation(pool=pool, url=pinned)
        if violation is not None:
            return f"guardrail:{violation[0]}"
        gate = pool.spillover_gate
        pinned_load = self.loads.get(pinned, pool.load_metric)
        least_reference_load = min(ref.load for ref in references)
        if (
            pinned_load >= gate.min_load
            and pinned_load > gate.multiplier * least_reference_load
        ):
            return "primary:load"
        return None

    def _least_loaded_passing(
        self, *, pool: PoolConfig, samples: Tuple[EngineSample, ...]
    ) -> Optional[EngineSample]:
        passing = [
            sample
            for sample in samples
            if self._guardrail_violation(pool=pool, url=sample.url) is None
        ]
        return min(passing, key=lambda sample: sample.load, default=None)

    def _overloaded(
        self, *, pool: PoolConfig, samples: Tuple[EngineSample, ...]
    ) -> EngineOverloaded:
        """Names the least loaded sample, which every caller found over a limit."""
        least_loaded = min(samples, key=lambda sample: sample.load)
        violation = self._guardrail_violation(pool=pool, url=least_loaded.url)
        if violation is None:
            return EngineOverloaded(
                engine=least_loaded.url, metric=pool.load_metric, load=least_loaded.load
            )
        metric, value = violation
        return EngineOverloaded(engine=least_loaded.url, metric=metric, load=value)

    def _decision(
        self,
        *,
        pool: PoolConfig,
        engine: str,
        policy: str,
        references: Tuple[EngineSample, ...],
        rejection: Optional[str] = None,
    ) -> PoolDecision:
        return PoolDecision(
            pool=pool.name,
            engine=engine,
            policy=policy,
            load_name=pool.load_metric,
            load=self.loads.get(engine, pool.load_metric),
            sampled_references=references,
            rejection=rejection,
        )
