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
from pickspin.live.runner import ModelLifecycle
from pickspin.live.vllm import Endpoint, chat_completions_url
from pickspin.pick.router import Pick
from pickspin.spin.lifecycle import ModelState, Spin

log = logging.getLogger(__name__)

AUTO = "auto"


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
    post sends a JSON body to a URL and returns (status, body); it defaults to requests.post and is
    replaced in tests.
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
        post: Callable[[str, dict[str, Any], Mapping[str, str], float], tuple[int, Any]] | None = None,
    ) -> None:
        super().__init__(spin=spin, actuator=actuator, static=static, clock=clock)
        self.pick = pick
        self.catalog = catalog
        self.endpoints = endpoints
        self.headers = dict(headers or {})
        self.timeout_s = timeout_s
        self.post = post or _post_json
        self._requests_lock = threading.Lock()
        self.requests_served = 0

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
        if self.actuator is not None:
            summary["measured_load_s"] = self.actuator.measured
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
        before = self.acquire(model, t_arrive)
        t_start = self.clock()
        self.spin.start(model, t_start)
        endpoint = self.endpoints[model]
        forward = dict(body)
        forward["model"] = endpoint.get("model", self.catalog[model].hf_id)
        ok = False
        try:
            status, reply = self.post(chat_completions_url(endpoint), forward, self.headers, self.timeout_s)
            ok = status == HTTPStatus.OK
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
        reply["pickspin"] = {
            "model": model,
            "tier": str(tier),
            "stage": str(stage) if stage is not None else None,
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
    """A failed upstream reply that is passed on to the client with its status."""

    def __init__(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        super().__init__(status)
        self.status = status
        self.body = body


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

        def _send(self, status: HTTPStatus | int, body: Any) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
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
                self._send(e.status, e.body)

    return Handler


def serve(gateway: Gateway, host: str, port: int) -> ThreadingHTTPServer:
    """Start the reaper and return a threading HTTP server for the gateway (call serve_forever on it)."""
    gateway.start_reaper()
    server = ThreadingHTTPServer((host, port), make_handler(gateway))
    server.daemon_threads = True
    log.info("Pick and Spin gateway listening on http://%s:%d (%d models)", host, port, len(gateway.catalog))
    return server
