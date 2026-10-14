"""MiniLB's session_affinity policy: sticky prefill with load-gated spillover,
power-of-two decode, guardrail 429s and the prefill/decode pair breaker."""

import asyncio
import json
import random
from types import SimpleNamespace

import pytest
from aiohttp import web
from sglang_router import mini_lb
from sglang_router.launch_router import parse_router_args
from sglang_router.mini_lb_policy import (
    POLICY_NO_AFFINITY,
    POLICY_OVERLOADED,
    POLICY_STICKY,
    EngineLoads,
    EngineOverloaded,
    LoadPoller,
    SessionAffinityPolicy,
)

P1, P2 = "http://p1:30000", "http://p2:30000"
D1, D2 = "http://d1:30000", "http://d2:30000"


def _policy():
    return SessionAffinityPolicy(rng=random.Random(0))


def _set_load(policy, url, **metrics):
    base = {
        "num_waiting_reqs": 0,
        "num_running_reqs": 0,
        "num_waiting_uncached_tokens": 0,
    }
    base.update(metrics)
    policy.loads.update(url, base)


def _select(policy, routing_key, prefill_urls=(P1, P2), decode_urls=(D1, D2)):
    return policy.select(
        prefill_urls=list(prefill_urls),
        decode_urls=list(decode_urls),
        routing_key=routing_key,
    )


@pytest.mark.unit
def test_new_key_goes_to_less_loaded_prefill_and_then_sticks():
    policy = _policy()
    _set_load(policy, P1, num_waiting_reqs=4)
    _set_load(policy, P2, num_waiting_reqs=1)

    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_NO_AFFINITY)

    # Sticky while the pinned load stays within the gate, even when not least loaded.
    _set_load(policy, P1, num_waiting_reqs=4)
    _set_load(policy, P2, num_waiting_reqs=10)
    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_STICKY)


@pytest.mark.unit
def test_overloaded_pin_spills_and_repins():
    policy = _policy()
    _select(policy, "s-1", prefill_urls=(P1,))
    _set_load(policy, P1, num_waiting_reqs=7)
    _set_load(policy, P2, num_waiting_reqs=2)

    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_OVERLOADED)
    assert prefill.rejection == "primary:load"

    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_STICKY)


@pytest.mark.unit
def test_gate_ignores_imbalance_below_min_load():
    policy = _policy()
    _select(policy, "s-1", prefill_urls=(P1,))
    _set_load(policy, P1, num_waiting_reqs=5)

    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P1, POLICY_STICKY)


@pytest.mark.unit
def test_pinned_engine_over_guardrail_spills():
    policy = _policy()
    _select(policy, "s-1", prefill_urls=(P1,))
    _set_load(policy, P1, num_waiting_uncached_tokens=2_000_001)

    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_OVERLOADED)
    assert prefill.rejection == "guardrail:num_waiting_uncached_tokens"


@pytest.mark.unit
def test_prefill_429_when_spill_target_is_also_over_guardrail():
    policy = _policy()
    _select(policy, "s-1", prefill_urls=(P1,))
    _set_load(policy, P1, num_waiting_uncached_tokens=2_000_001)
    _set_load(policy, P2, num_waiting_uncached_tokens=3_000_000)

    with pytest.raises(EngineOverloaded) as raised:
        _select(policy, "s-1")
    assert str(raised.value) == (
        f"engine {P2} overloaded on num_waiting_uncached_tokens: load=3000000"
    )


@pytest.mark.unit
def test_decode_picks_fewer_running_and_rejects_over_guardrail():
    policy = _policy()
    _set_load(policy, D1, num_running_reqs=9)
    _set_load(policy, D2, num_running_reqs=3)
    _, decode = _select(policy, None)
    assert (decode.engine, decode.policy) == (D2, POLICY_NO_AFFINITY)

    _set_load(policy, D1, num_running_reqs=1, num_waiting_reqs=25)
    _set_load(policy, D2, num_running_reqs=5, num_waiting_reqs=30)
    with pytest.raises(
        EngineOverloaded, match="overloaded on num_waiting_reqs: load=25"
    ):
        _select(policy, "s-1")
    # A request rejected at decode leaves its key unpinned.
    _set_load(policy, D1)
    _set_load(policy, D2)
    _set_load(policy, P1, num_waiting_reqs=1)
    prefill, _ = _select(policy, "s-1")
    assert prefill.policy == POLICY_NO_AFFINITY


@pytest.mark.unit
def test_request_without_key_is_never_pinned():
    policy = _policy()
    for _ in range(5):
        prefill, _ = _select(policy, None)
        assert prefill.policy == POLICY_NO_AFFINITY


@pytest.mark.unit
def test_unreachable_pin_is_replaced():
    policy = _policy()
    _select(policy, "s-1", prefill_urls=(P1,))
    policy.loads.mark_unreachable(P1)

    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_NO_AFFINITY)
    prefill, _ = _select(policy, "s-1")
    assert (prefill.engine, prefill.policy) == (P2, POLICY_STICKY)


@pytest.mark.unit
def test_pair_breaker_holds_out_failed_pair_until_success():
    policy = _policy()
    policy.record_failure(prefill_url=P1, decode_url=D1, reason="decode returned 500")
    for _ in range(10):
        _, decode = _select(policy, None, prefill_urls=(P1,))
        assert decode.engine == D2

    # Every decode held out with this prefill: route anyway.
    policy.record_failure(prefill_url=P1, decode_url=D2, reason="decode returned 500")
    _, decode = _select(policy, None, prefill_urls=(P1,))
    assert decode.engine in (D1, D2)

    policy.record_success(prefill_url=P1, decode_url=D2)
    for _ in range(10):
        _, decode = _select(policy, None, prefill_urls=(P1,))
        assert decode.engine == D2


@pytest.mark.unit
def test_load_poller_sums_dp_ranks_and_marks_dead_engines():
    async def loads_handler(request):
        assert request.query["include"] == "core,queues"
        rank = {"num_waiting_reqs": 2, "num_running_reqs": 5}
        return web.json_response({"loads": [rank, rank]})

    async def run():
        app = web.Application()
        app.router.add_get("/v1/loads", loads_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        live_url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        dead_url = "http://127.0.0.1:1"
        loads = EngineLoads()
        poller = LoadPoller(
            loads=loads,
            metric_names=frozenset({"num_waiting_reqs", "num_running_reqs"}),
            interval_secs=0.05,
            timeout_secs=1.0,
        )
        polling = asyncio.create_task(poller.run(lambda: [live_url, dead_url]))
        try:
            await asyncio.sleep(0.3)
        finally:
            polling.cancel()
            await asyncio.gather(polling, return_exceptions=True)
            await runner.cleanup()
        return loads, live_url, dead_url

    loads, live_url, dead_url = asyncio.run(run())
    assert loads.get(live_url, "num_waiting_reqs") == 4
    assert loads.get(live_url, "num_running_reqs") == 10
    assert loads.is_reachable(live_url)
    assert not loads.is_reachable(dead_url)


def _event(payload):
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stream", "error"),
    [
        (_event({"text": "fine"}) + b"data: [DONE]\n\n", None),
        (
            _event({"error": {"message": "bad input", "code": 400}}),
            None,
        ),
        (
            _event({"error": {"message": "KV transfer failed", "code": 500}}),
            "in-band error 500: KV transfer failed",
        ),
        (
            _event(
                {
                    "text": "",
                    "meta_info": {
                        "finish_reason": {
                            "type": "abort",
                            "message": "bootstrap timeout",
                            "status_code": 503,
                        }
                    },
                }
            ),
            "in-band error 503: bootstrap timeout",
        ),
        # Generated text quoting an error object is escaped JSON, not an event field.
        (_event({"text": '{"error": {"code": 500}}'}), None),
    ],
)
def test_inband_error_scanner(stream, error):
    for chunk_size in (len(stream), 7, 1):
        scanner = mini_lb._InbandErrorScanner()
        for start in range(0, len(stream), chunk_size):
            scanner.feed(stream[start : start + chunk_size])
        assert scanner.error == error


async def _serve(handler):
    app = web.Application()
    app.router.add_post("/generate", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _streamer(final_event):
    """Engine /generate handler streaming one token, then final_event()."""

    async def handler(request):
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        await resp.write(_event({"text": "x"}))
        await resp.write(final_event())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    return handler


def _lb(prefill_urls, decode_urls):
    args = SimpleNamespace(
        host="127.0.0.1",
        port=0,
        policy="random",
        mini_lb_policy="session_affinity",
        pd_disaggregation=True,
        prefill_urls=[(url, 8998) for url in prefill_urls],
        decode_urls=list(decode_urls),
        test_external_dp_routing=False,
        request_timeout_secs=10,
    )
    return mini_lb.MiniLoadBalancer(args)


async def _stream_through(load_balancer, prefill_url, decode_url):
    response = await load_balancer.generate_stream(
        {"text": "hello", "stream": True}, prefill_url, decode_url
    )
    async for _ in response.body_iterator:
        pass


@pytest.mark.unit
def test_stream_abort_trips_pair_breaker_and_success_resets_it():
    decode_fails = [True]

    def flaky_decode_event():
        if not decode_fails[0]:
            return _event({"text": "y"})
        finish_reason = {
            "type": "abort",
            "message": "KV transfer failed",
            "status_code": 500,
        }
        return _event({"text": "", "meta_info": {"finish_reason": finish_reason}})

    async def run():
        prefill_runner, prefill_url = await _serve(_streamer(lambda: _event({})))
        flaky_runner, flaky_decode = await _serve(_streamer(flaky_decode_event))
        decode_urls = [flaky_decode, D2]
        load_balancer = _lb([prefill_url], decode_urls)
        policy = load_balancer.affinity_policy

        def decode_picks():
            return {
                _select(policy, None, [prefill_url], decode_urls)[1].engine
                for _ in range(20)
            }

        try:
            await _stream_through(load_balancer, prefill_url, flaky_decode)
            held_out_picks = decode_picks()
            decode_fails[0] = False
            await _stream_through(load_balancer, prefill_url, flaky_decode)
            released_picks = decode_picks()
        finally:
            await prefill_runner.cleanup()
            await flaky_runner.cleanup()
        return held_out_picks, released_picks, flaky_decode

    held_out_picks, released_picks, flaky_decode = asyncio.run(run())
    assert held_out_picks == {D2}
    assert released_picks == {flaky_decode, D2}


@pytest.mark.unit
def test_overloaded_request_gets_429(monkeypatch):
    load_balancer = _lb([P1], [D1])
    load_balancer.affinity_policy.loads.update(
        D1, {"num_running_reqs": 0, "num_waiting_reqs": 25}
    )
    monkeypatch.setattr(mini_lb, "lb", load_balancer)

    response = asyncio.run(
        mini_lb._forward_to_backend({"text": "hello"}, "generate", "s-1")
    )
    assert response.status_code == 429
    assert json.loads(response.body) == {
        "error": {"message": f"engine {D1} overloaded on num_waiting_reqs: load=25"}
    }


@pytest.mark.unit
def test_routing_key_header_reaches_the_policy(monkeypatch):
    load_balancer = _lb([P1], [D1])
    monkeypatch.setattr(mini_lb, "lb", load_balancer)
    seen_keys = []

    def select(*, prefill_urls, decode_urls, routing_key):
        seen_keys.append(routing_key)
        raise EngineOverloaded(engine=D1, metric="num_waiting_reqs", load=25)

    monkeypatch.setattr(load_balancer.affinity_policy, "select", select)

    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    with TestClient(mini_lb.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": []},
            headers={"x-routing-key": "s-42"},
        )
    assert response.status_code == 429
    assert seen_keys == ["s-42"]


@pytest.mark.unit
def test_cli_flag_selects_policy():
    args = parse_router_args(
        [
            "--mini-lb",
            "--mini-lb-policy",
            "session_affinity",
            "--pd-disaggregation",
            "--prefill",
            P1,
            "--decode",
            D1,
        ]
    )
    assert args.mini_lb_policy == "session_affinity"
    assert parse_router_args(["--mini-lb"]).mini_lb_policy == "session_affinity"
    random_args = parse_router_args(["--mini-lb", "--mini-lb-policy", "random"])
    assert random_args.mini_lb_policy == "random"
