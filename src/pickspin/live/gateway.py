"""The Pick and Spin gateway: an OpenAI-compatible service in front of the model servers.

A client sends a chat completion to the gateway as it would to any OpenAI-compatible server. With
"model": "auto" (or no model), Pick classifies the last user message and selects a model; a request may
also name one of the gateway's models directly. Spin brings a cold model up, holds the request until
the server is ready, forwards it, and scales models that stay idle for T_cooldown back to zero.

Endpoints:
    POST /v1/chat/completions   the vLLM response, plus a "pickspin" object (tier, model, cold start, wait)
    GET  /v1/models             the models the gateway serves, and "auto"
    GET  /healthz               200 while the gateway runs
    GET  /stats                 Spin's accounting: GPU-hours, utilization, cold starts, model states

The models come from the server file (deploy/nautilus/servers.json), so replacing a model means editing
that file. Streaming responses are not supported.

The gateway recovers on its own: a model whose server fails to load, or dies while WARM, goes back to
COLD and is started afresh by the next request. A request waits at most max_wait_s for a cold model;
after that it gets 503 with Retry-After while the model keeps loading.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import requests

from pickspin.config import ModelSpec
from pickspin.live.actuator import Actuator
from pickspin.live.runner import ModelLifecycle, ModelUnavailable
from pickspin.live.vllm import Endpoint, chat_completions_url
from pickspin.pick.router import Pick
from pickspin.spin.lifecycle import ModelState, Spin

log = logging.getLogger(__name__)

AUTO = "auto"
RETRY_AFTER_S = 30


class GatewayError(Exception):
    """A request the gateway rejects, with the HTTP status to answer."""

    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def last_user_text(messages: Any) -> str:
    """Return the text of the last user message (string content or the text parts of a list)."""
    if not isinstance(messages, list) or not messages:
        raise GatewayError(HTTPStatus.BAD_REQUEST, "'messages' must be a non-empty list")
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    raise GatewayError(HTTPStatus.BAD_REQUEST, "no user message to route")


class Gateway(ModelLifecycle):
    """Routes chat completions with Pick and serves them through Spin's lifecycle.

    catalog describes the models; endpoints gives each model's server URL and served model name.
    timeout_s bounds a forwarded request and max_wait_s the wait for a cold model. post sends a JSON
    body to a URL and returns (status, body); it defaults to requests.post and is replaced in tests.
    """

    def __init__(
        self,
        *,
        pick: Pick,
        spin: Spin,
        catalog: Mapping[str, ModelSpec],
        endpoints: Mapping[str, Endpoint],
        actuator: Actuator | None,
        headers: Mapping[str, str] | None = None,
        static: bool = False,
        clock: Callable[[], float] = time.monotonic,
        timeout_s: float = 300.0,
        max_wait_s: float = 600.0,
        post: Callable[[str, dict[str, Any], Mapping[str, str], float], tuple[int, Any]] | None = None,
    ) -> None:
        super().__init__(spin=spin, actuator=actuator, static=static, clock=clock, recover=True)
        self.pick = pick
        self.catalog = catalog
        self.endpoints = endpoints
        self.headers = dict(headers or {})
        self.timeout_s = timeout_s
        self.max_wait_s = max_wait_s
        self.post = post or _post_json
        self._requests_lock = threading.Lock()
        self.requests_served = 0
        self.requests_unavailable = 0  # answered 503 because the model failed to load or was still loading

    def models(self) -> dict[str, Any]:
        """The /v1/models answer: "auto" plus every model of the catalog."""
        data = [{"id": AUTO, "object": "model", "owned_by": "pickspin"}]
        data += [
            {"id": key, "object": "model", "owned_by": "pickspin", "tier": str(spec.tier), "hf_id": spec.hf_id}
            for key, spec in self.catalog.items()
        ]
        return {"object": "list", "data": data}

    def stats(self) -> dict[str, Any]:
        """The /stats answer: Spin's accounting now, each model's state and the requests served."""
        summary = self.spin.summary(self.clock()).to_dict()
        summary["states"] = {m: str(self.spin.status(m)) for m in self.spin.models}
        summary["requests_served"] = self.requests_served
        summary["requests_unavailable"] = self.requests_unavailable
        if self.actuator is not None:
            summary["measured_load_s"] = self.actuator.measured
            phases = getattr(self.actuator, "phases", None)
            if phases is not None:
                summary["load_phases"] = phases
        return summary

    def complete(self, body: Mapping[str, Any]) -> dict[str, Any]:
        """Serve one chat completion request and return the response body."""
        if body.get("stream"):
            raise GatewayError(HTTPStatus.BAD_REQUEST, "streaming is not supported by the gateway")
        requested = body.get("model") or AUTO
        text = last_user_text(body.get("messages"))
        t_arrive = self.clock()
        if requested == AUTO:
            route = self.pick.route(text, t_arrive)
            model, tier, stage = route.model, route.tier, route.stage
        elif requested in self.catalog:
            model, tier, stage = requested, self.catalog[requested].tier, None
        else:
            raise GatewayError(HTTPStatus.NOT_FOUND, f"unknown model {requested!r}; see /v1/models")
        meta: dict[str, Any] = {"model": model, "tier": str(tier), "stage": str(stage) if stage is not None else None}
        try:
            before = self.acquire(model, t_arrive, timeout=self.max_wait_s)
        except ModelUnavailable as e:
            with self._requests_lock:
                self.requests_unavailable += 1
            if not e.still_loading:
                self.pick.update(model, tier, False)
            meta["wait_s"] = round(self.clock() - t_arrive, 3)
            kind = "model_loading" if e.still_loading else "model_unavailable"
            message = f"{e}; retry in {RETRY_AFTER_S} s"
            raise _ReplyError(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"error": {"message": message, "type": kind}, "pickspin": meta},
                {"Retry-After": str(RETRY_AFTER_S)},
            ) from None
        load = self.loads[model]
        t_start = self.clock()
        self.spin.start(model, t_start)
        endpoint = self.endpoints[model]
        forward = dict(body)
        forward["model"] = endpoint.get("model", self.catalog[model].hf_id)
        ok = False
        try:
            status, reply = self.post(chat_completions_url(endpoint), forward, self.headers, self.timeout_s)
            ok = status == HTTPStatus.OK
        except requests.ConnectionError as e:
            # The server may be gone (its Job failed or hit its deadline); if so, the next request restarts it.
            if self.check_server(model, load):
                status, why = HTTPStatus.SERVICE_UNAVAILABLE, "its server is gone; the next request starts it again"
            else:
                status, why = HTTPStatus.BAD_GATEWAY, type(e).__name__
            reply = {"error": {"message": f"{model} unreachable: {why}"}}
        except requests.RequestException as e:
            status, reply = HTTPStatus.BAD_GATEWAY, {"error": {"message": f"{model} unreachable: {type(e).__name__}"}}
        finally:
            t_end = self.clock()
            self.spin.finish(model, t_end, t_end - t_start, t_end - t_arrive)
            self.pick.update(model, tier, ok)
            with self._requests_lock:
                self.requests_served += 1
        if not isinstance(reply, dict):
            reply = {"error": {"message": str(reply)}}
        reply["pickspin"] = meta | {
            "cold_start": before is ModelState.COLD,
            "wait_s": round(t_start - t_arrive, 3),
            "total_s": round(t_end - t_arrive, 3),
        }
        if not ok:
            try:
                code = HTTPStatus(status)
            except ValueError:
                code = HTTPStatus.BAD_GATEWAY
            raise _ReplyError(code, reply)
        return reply


class _ReplyError(Exception):
    """A reply that is not a success, passed on to the client with its status and extra headers."""

    def __init__(self, status: HTTPStatus, body: dict[str, Any], headers: Mapping[str, str] | None = None) -> None:
        super().__init__(status)
        self.status = status
        self.body = body
        self.headers = dict(headers or {})


def _post_json(url: str, body: dict[str, Any], headers: Mapping[str, str], timeout_s: float) -> tuple[int, Any]:
    """POST a JSON body and return (status, decoded body or text)."""
    r = requests.post(url, json=body, headers=dict(headers), timeout=timeout_s)
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, r.text


def make_handler(gateway: Gateway) -> type[BaseHTTPRequestHandler]:
    """Return the HTTP request handler class bound to a gateway."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "pickspin"

        def log_message(self, fmt: str, *args: Any) -> None:
            log.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: HTTPStatus | int, body: Any, headers: Mapping[str, str] | None = None) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(HTTPStatus.OK, {"status": "ok"})
            elif self.path == "/v1/models":
                self._send(HTTPStatus.OK, gateway.models())
            elif self.path == "/stats":
                self._send(HTTPStatus.OK, gateway.stats())
            else:
                self._send(HTTPStatus.NOT_FOUND, {"error": {"message": f"no route {self.path}"}})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send(HTTPStatus.NOT_FOUND, {"error": {"message": f"no route {self.path}"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise GatewayError(HTTPStatus.BAD_REQUEST, "the request body must be a JSON object")
                self._send(HTTPStatus.OK, gateway.complete(body))
            except json.JSONDecodeError:
                self._send(HTTPStatus.BAD_REQUEST, {"error": {"message": "invalid JSON"}})
            except GatewayError as e:
                self._send(e.status, {"error": {"message": str(e)}})
            except _ReplyError as e:
                self._send(e.status, e.body, e.headers)

    return Handler


def serve(gateway: Gateway, host: str, port: int) -> ThreadingHTTPServer:
    """Start the reaper and return a threading HTTP server for the gateway (call serve_forever on it)."""
    gateway.start_reaper()
    server = ThreadingHTTPServer((host, port), make_handler(gateway))
    server.daemon_threads = True
    log.info("Pick and Spin gateway listening on http://%s:%d (%d models)", host, port, len(gateway.catalog))
    return server
