"""Spin: per-model WARM/COLD state and query execution against vLLM endpoints.

State machine (Sec. V-A):
    COLD -> WARM  when a query is routed to a cold model
    WARM -> COLD  when the model has been idle longer than T_cooldown

Total latency = inference latency + cold-start penalty if the model was cold (Eq. 6).
The penalty is the fixed per-model value in config.COLD_START_TIMES.
"""

import os
import threading
import time

import requests

from config import COLD_START_TIMES, MODELS, ROUTING, load_endpoints


class ColdStartManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.cooldown = ROUTING["cooldown_seconds"]
        self.state = {m: {"status": "COLD", "last_used": 0.0} for m in MODELS}
        self.cold_hits = 0
        self.warm_hits = 0

    def check_and_update(self, model):
        """Return (was_cold, penalty_seconds) and mark the model WARM."""
        now = time.time()
        with self.lock:
            s = self.state[model]
            if s["status"] == "WARM" and now - s["last_used"] > self.cooldown:
                s["status"] = "COLD"
            was_cold = s["status"] == "COLD"
            s["status"] = "WARM"
            s["last_used"] = now
            if was_cold:
                self.cold_hits += 1
            else:
                self.warm_hits += 1
        return was_cold, COLD_START_TIMES.get(model, 30) if was_cold else 0

    def get_stats(self):
        with self.lock:
            total = self.cold_hits + self.warm_hits
            return {"cold_hits": self.cold_hits, "warm_hits": self.warm_hits,
                    "cold_rate": self.cold_hits / total if total else 0}


class ModelExecutor:
    def __init__(self, endpoints=None):
        self.cold_start_mgr = ColdStartManager()
        self.endpoints = endpoints or load_endpoints()
        key = os.environ.get("VLLM_API_KEY")
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}

    def call_api(self, model, query, max_tokens=256, timeout=90):
        """Return (success, text, usage, latency_seconds)."""
        ep = self.endpoints.get(model)
        if not ep:
            return False, f"Unknown model: {model}", {}, 0
        payload = {"model": ep.get("model", MODELS[model]["hf_id"]),
                   "messages": [{"role": "user", "content": query}],
                   "max_tokens": max_tokens, "temperature": 0.1}
        start = time.time()
        try:
            r = requests.post(ep["base_url"].rstrip("/") + "/v1/chat/completions",
                              json=payload, headers=self.headers, timeout=timeout)
            latency = time.time() - start
            if r.status_code == 200:
                data = r.json()
                return True, data["choices"][0]["message"]["content"], data.get("usage", {}), latency
            return False, f"HTTP {r.status_code}", {}, latency
        except requests.exceptions.Timeout:
            return False, "Timeout", {}, time.time() - start
        except Exception as e:  # network errors are recorded, not raised
            return False, str(e), {}, time.time() - start

    def execute(self, model, query, max_tokens=256):
        was_cold, penalty = self.cold_start_mgr.check_and_update(model)
        ok, text, usage, latency = self.call_api(model, query, max_tokens)
        if not ok:  # one retry
            time.sleep(0.3)
            ok, text, usage, latency = self.call_api(model, query, max_tokens)
        return {"success": ok, "response": text, "tokens": usage.get("completion_tokens", 0),
                "latency": latency, "was_cold": was_cold, "cold_penalty": penalty,
                "total_latency": latency + penalty}

    def get_cold_stats(self):
        return self.cold_start_mgr.get_stats()
