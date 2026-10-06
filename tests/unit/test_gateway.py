"""The OpenAI-compatible gateway (pickspin serve) with a custom model catalog."""

import json
import random
import threading
import urllib.request
from http import HTTPStatus

import pytest

from pickspin.config import ModelSpec, Tier, tiers_of
from pickspin.live.gateway import Gateway, GatewayError, last_user_text, serve
from pickspin.pick.classifier import HybridClassifier
from pickspin.pick.router import Pick
from pickspin.spin.lifecycle import ModelState, Spin

CATALOG = {
    "tiny": ModelSpec("tiny", "Tiny", Tier.SIMPLE, "org/tiny", 1, 10, 1),
    "mid": ModelSpec("mid", "Mid", Tier.MEDIUM, "org/mid", 5, 20, 1),
    "big": ModelSpec("big", "Big", Tier.COMPLEX, "org/big", 20, 30, 2),
}
ENDPOINTS = {k: {"base_url": f"http://{k}:8000", "model": s.hf_id, "deployment": k} for k, s in CATALOG.items()}


class RecordingActuator:
    def __init__(self):
        self.measured = {}
        self.calls = []

    def scale(self, model, replicas):
        self.calls.append((model, replicas))

    def ready_replicas(self, model):
        return 0

    def wait_ready(self, model, t0):
        return None

    def load_estimate(self, model, now=None):
        return 1.0


def make_gateway(post, *, static=False):
    clock = iter(float(i) for i in range(10_000))
    now = lambda: next(clock)  # noqa: E731
    spin = Spin(CATALOG, catalog=CATALOG, scale_to_zero=not static, now=now())
    pick = Pick(HybridClassifier(None), spin, rng=random.Random(0), tiers=tiers_of(CATALOG), models=CATALOG)
    actuator = None if static else RecordingActuator()
    gw = Gateway(
        pick=pick,
        spin=spin,
        catalog=CATALOG,
        endpoints=ENDPOINTS,
        actuator=actuator,
        static=static,
        clock=now,
        post=post,
    )
    return gw, actuator


def ok_post(url, body, headers, timeout):
    return 200, {"choices": [{"message": {"role": "assistant", "content": "hi"}}], "model": body["model"], "url": url}


def test_auto_routes_by_tier_and_cold_starts_the_model():
    gw, actuator = make_gateway(ok_post)
    reply = gw.complete({"model": "auto", "messages": [{"role": "user", "content": "What is the capital of France?"}]})
    assert reply["pickspin"]["tier"] == "SIMPLE"
    assert reply["pickspin"]["model"] == "tiny"
    assert reply["pickspin"]["cold_start"] is True
    assert reply["model"] == "org/tiny"
    assert reply["url"] == "http://tiny:8000/v1/chat/completions"
    assert actuator.calls == [("tiny", 1)]
    assert gw.spin.status("tiny") is ModelState.WARM
    again = gw.complete({"messages": [{"role": "user", "content": "Define entropy."}]})
    assert again["pickspin"]["cold_start"] is False


def test_a_request_may_name_a_model():
    gw, _ = make_gateway(ok_post, static=True)
    reply = gw.complete({"model": "big", "messages": [{"role": "user", "content": "hello"}]})
    assert reply["pickspin"]["model"] == "big"
    assert reply["pickspin"]["tier"] == "COMPLEX"
    with pytest.raises(GatewayError) as e:
        gw.complete({"model": "gpt-9", "messages": [{"role": "user", "content": "hello"}]})
    assert e.value.status == HTTPStatus.NOT_FOUND


def test_streaming_and_bad_messages_are_rejected():
    gw, _ = make_gateway(ok_post, static=True)
    with pytest.raises(GatewayError):
        gw.complete({"stream": True, "messages": [{"role": "user", "content": "x"}]})
    with pytest.raises(GatewayError):
        last_user_text([{"role": "system", "content": "x"}])
    assert (
        last_user_text([{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}])
        == "a b"
    )


def test_models_and_stats_reflect_the_catalog():
    gw, _ = make_gateway(ok_post, static=True)
    ids = [m["id"] for m in gw.models()["data"]]
    assert ids == ["auto", "tiny", "mid", "big"]
    gw.complete({"model": "big", "messages": [{"role": "user", "content": "x"}]})
    stats = gw.stats()
    assert stats["requests_served"] == 1
    assert stats["states"] == {"tiny": "WARM", "mid": "WARM", "big": "WARM"}
    assert stats["per_model"]["big"]["gpu_hours"] > 0


def test_http_server_end_to_end():
    gw, _ = make_gateway(ok_post, static=True)
    server = serve(gw, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert json.loads(urllib.request.urlopen(base + "/healthz").read())["status"] == "ok"
        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps(
                {"model": "auto", "messages": [{"role": "user", "content": "Prove that 2 is prime."}]}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        body = json.loads(urllib.request.urlopen(req).read())
        assert body["pickspin"]["tier"] == "COMPLEX"
        assert body["choices"][0]["message"]["content"] == "hi"
    finally:
        server.shutdown()
        gw.stop.set()
