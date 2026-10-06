"""The OpenAI-compatible gateway (pickspin serve) with a custom model catalog."""

import json
import random
import threading
import time
import urllib.request
from http import HTTPStatus

import pytest
import requests

from pickspin.config import ModelSpec, Tier, tiers_of
from pickspin.live.gateway import Gateway, GatewayError, _ReplyError, last_user_text, serve
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
    """Loads at once, or when `gate` is set; wait_ready raises `fail` if set; alive() returns `is_alive`."""

    def __init__(self, *, fail=None, gate=None):
        self.measured = {}
        self.calls = []
        self.fail = fail
        self.gate = gate
        self.is_alive = True

    def scale(self, model, replicas):
        self.calls.append((model, replicas))

    def ready_replicas(self, model):
        return 0

    def wait_ready(self, model, t0):
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.fail is not None:
            raise self.fail

    def alive(self, model):
        return self.is_alive

    def load_estimate(self, model, now=None):
        return 1.0


class TickClock:
    """Moves on one second at every reading; the bring-up threads read it too."""

    def __init__(self):
        self.t = 0.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            self.t += 1.0
            return self.t


def make_gateway(post, *, static=False, actuator=None, max_wait_s=600.0):
    clock = TickClock()
    spin = Spin(CATALOG, catalog=CATALOG, scale_to_zero=not static, now=clock())
    pick = Pick(HybridClassifier(None), spin, rng=random.Random(0), tiers=tiers_of(CATALOG), models=CATALOG)
    if actuator is None and not static:
        actuator = RecordingActuator()
    gw = Gateway(
        pick=pick,
        spin=spin,
        catalog=CATALOG,
        endpoints=ENDPOINTS,
        actuator=actuator,
        static=static,
        clock=clock,
        max_wait_s=max_wait_s,
        post=post,
    )
    return gw, actuator


def ask(model):
    return {"model": model, "messages": [{"role": "user", "content": "hello"}]}


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


def test_a_failed_load_answers_503_and_the_next_request_starts_the_model_again():
    actuator = RecordingActuator(fail=RuntimeError("tiny server failed: its pod stopped: OOMKilled (exit code 137)"))
    gw, _ = make_gateway(ok_post, actuator=actuator)

    with pytest.raises(_ReplyError) as e:
        gw.complete(ask("tiny"))
    assert e.value.status == HTTPStatus.SERVICE_UNAVAILABLE
    assert e.value.body["error"]["type"] == "model_unavailable"
    assert e.value.headers == {"Retry-After": "30"}
    assert e.value.body["pickspin"]["model"] == "tiny"
    assert gw.spin.status("tiny") is ModelState.COLD
    assert actuator.calls == [("tiny", 1), ("tiny", 0)]  # the failed server is removed
    assert (gw.requests_unavailable, gw.requests_served) == (1, 0)

    actuator.fail = None
    reply = gw.complete(ask("tiny"))
    assert reply["pickspin"]["cold_start"] is True
    assert actuator.calls[-1] == ("tiny", 1)
    assert gw.spin.status("tiny") is ModelState.WARM
    assert gw.spin.summary(gw.clock()).per_model["tiny"].cold_starts == 2


def test_a_request_gives_up_on_a_slow_load_which_goes_on():
    gate = threading.Event()
    gw, _ = make_gateway(ok_post, actuator=RecordingActuator(gate=gate), max_wait_s=0.05)

    with pytest.raises(_ReplyError) as e:
        gw.complete(ask("mid"))
    assert e.value.status == HTTPStatus.SERVICE_UNAVAILABLE
    assert e.value.body["error"]["type"] == "model_loading"
    assert e.value.headers == {"Retry-After": "30"}
    assert gw.spin.status("mid") is ModelState.LOADING

    gate.set()
    deadline = time.monotonic() + 5
    while gw.spin.status("mid") is not ModelState.WARM and time.monotonic() < deadline:
        time.sleep(0.01)
    assert gw.spin.status("mid") is ModelState.WARM
    assert gw.spin.idle_expired(1e9) == ["mid"]  # the request that gave up holds nothing
    assert gw.complete(ask("mid"))["pickspin"]["cold_start"] is False


def test_a_server_that_died_is_started_again_by_the_next_request():
    refuse = {"on": False}

    def post(url, body, headers, timeout):
        if refuse["on"]:
            raise requests.ConnectionError("connection refused")
        return ok_post(url, body, headers, timeout)

    gw, actuator = make_gateway(post)
    gw.complete(ask("tiny"))
    refuse["on"] = True

    # Unreachable but still alive (say, between readiness probes): 502, and the server is kept.
    with pytest.raises(_ReplyError) as e:
        gw.complete(ask("tiny"))
    assert e.value.status == HTTPStatus.BAD_GATEWAY
    assert gw.spin.status("tiny") is ModelState.WARM

    # Gone for good (its Job failed or hit its deadline): 503, and the model is COLD again.
    actuator.is_alive = False
    with pytest.raises(_ReplyError) as e:
        gw.complete(ask("tiny"))
    assert e.value.status == HTTPStatus.SERVICE_UNAVAILABLE
    assert "its server is gone" in e.value.body["error"]["message"]
    assert gw.spin.status("tiny") is ModelState.COLD
    assert actuator.calls == [("tiny", 1), ("tiny", 0)]

    refuse["on"], actuator.is_alive = False, True
    assert gw.complete(ask("tiny"))["pickspin"]["cold_start"] is True
    assert actuator.calls[-1] == ("tiny", 1)


def test_release_all_scales_every_model_to_zero():
    gw, actuator = make_gateway(ok_post)
    gw.release_all()
    assert actuator.calls == [("tiny", 0), ("mid", 0), ("big", 0)]
    static, _ = make_gateway(ok_post, static=True)
    static.release_all()  # a static gateway scales nothing and has no actuator


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
