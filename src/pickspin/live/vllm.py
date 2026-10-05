"""The endpoint map and the OpenAI-compatible chat call to vLLM used by live runs.

load_endpoints reads a JSON file that maps each model key to its base_url, served model name and
Deployment (see deploy/endpoints.example.json). call_vllm sends one query and reports whether it
succeeded, the response text, the completion tokens and the seconds it took. chat_completions_url and
bearer_headers are shared with the static baseline, whose HTTP call is otherwise separate.

call_vllm posts with requests.post and no Session on purpose: a Session keeps at most 10 pooled
connections per host, which would change how a run with 250 worker threads connects.
"""

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple, Required, TypedDict

import requests


class Endpoint(TypedDict, total=False):
    """One model's server: its base URL, the served model name and the Kubernetes Deployment."""

    base_url: Required[str]
    model: str
    deployment: str


def load_endpoints(path: Path) -> dict[str, Endpoint]:
    """Read the endpoint map: model key -> endpoint."""
    with path.open(encoding="utf-8") as f:
        endpoints: dict[str, Endpoint] = json.load(f)
    return endpoints


def chat_completions_url(endpoint: Endpoint) -> str:
    """Return the endpoint's OpenAI-compatible chat completions URL."""
    return endpoint["base_url"].rstrip("/") + "/v1/chat/completions"


def bearer_headers(api_key: str | None) -> dict[str, str]:
    """Return the Authorization header for an API key, or no headers without one."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


class ChatResult(NamedTuple):
    """The outcome of one chat call: the response text on success, else a short error description."""

    success: bool
    text: str
    completion_tokens: int
    seconds: float


def call_vllm(
    endpoint: Endpoint, query: str, max_tokens: int, headers: Mapping[str, str], timeout: float = 120
) -> ChatResult:
    """Send one query to the endpoint and time it.

    The endpoint must name its served model. A reply other than HTTP 200 gives the text 'HTTP <code>',
    and a requests error (connection refused, timeout, invalid JSON) gives the exception's class name;
    both count as failures with 0 tokens. The seconds run on time.monotonic from just before the
    request until the reply arrives, or until the error.
    """
    payload = {
        "model": endpoint["model"],
        "messages": [{"role": "user", "content": query}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    t0 = time.monotonic()
    try:
        r = requests.post(chat_completions_url(endpoint), json=payload, headers=dict(headers), timeout=timeout)
        took = time.monotonic() - t0
        if r.status_code == 200:
            data = r.json()
            return ChatResult(
                True, data["choices"][0]["message"]["content"], data.get("usage", {}).get("completion_tokens", 0), took
            )
        return ChatResult(False, f"HTTP {r.status_code}", 0, took)
    except requests.RequestException as e:
        return ChatResult(False, type(e).__name__, 0, time.monotonic() - t0)
