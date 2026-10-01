"""Unit tests for Pick, Spin and the simulator.   Run: python -m pytest tests"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "pickspin"))

import simulate  # noqa: E402
from classifier import HybridClassifier, keyword_tier  # noqa: E402
from config import MODELS  # noqa: E402
from pick import Pick  # noqa: E402
from spin import COLD, LOADING, WARM, SharedStorage, Spin, init_seconds  # noqa: E402


class FixedBeta(random.Random):
    """Every Beta sample is 0.5, so Eq. 4 depends only on latency and exploration."""

    def betavariate(self, alpha, beta):
        return 0.5


def serve(spin, m, t_request, t_ready, infer_s):
    """Route one query to m at t_request; the model is ready at t_ready; the query runs infer_s."""
    before = spin.request(m, t_request)
    if before == COLD:
        spin.loaded(m, t_ready)
    spin.start(m, t_ready)
    spin.finish(m, t_ready + infer_s, infer_s, t_ready + infer_s - t_request)
    return t_ready + infer_s


def test_keyword_lists_and_precedence():
    assert keyword_tier("Prove that x > 0. What is x?") == "COMPLEX"
    assert keyword_tier("Calculate how many apples are left") == "MEDIUM"
    assert keyword_tier("What is the capital of France?") == "SIMPLE"
    assert keyword_tier("Finish the story") is None


def test_distilbert_only_sees_unmatched_queries():
    class Stub:
        seen = []

        def predict(self, queries):
            self.seen.append(list(queries))
            return ["COMPLEX"] * len(queries)

    stub = Stub()
    clf = HybridClassifier(distilbert=stub)
    assert clf.classify("What is 2 + 2?") == ("SIMPLE", "keyword")
    assert clf.classify("Finish the story") == ("COMPLEX", "distilbert")
    assert clf.classify_many(["define entropy", "foo", "bar"]) == [
        ("SIMPLE", "keyword"), ("COMPLEX", "distilbert"), ("COMPLEX", "distilbert")]
    assert stub.seen[-1] == ["foo", "bar"]


def test_state_machine_and_gpu_accounting():
    spin = Spin(cooldown_s=300, scale_to_zero=True, now=0.0)
    m = "llama3.2_1B"
    assert spin.status(m) == COLD
    assert spin.request(m, 10.0) == COLD and spin.status(m) == LOADING
    assert spin.idle_expired(1000.0) == []
    spin.loaded(m, 42.0)
    spin.start(m, 42.0)
    assert spin.status(m) == WARM
    spin.finish(m, 50.0, infer_s=8.0, total_s=40.0)
    assert spin.idle_expired(349.0) == [] and spin.idle_expired(350.0) == [m]
    assert spin.stop(m, 350.0) and spin.status(m) == COLD
    s = spin.summary(400.0)
    pm = s["per_model"][m]
    assert pm["gpu_hours"] == pytest.approx(340.0 / 3600)          # GPU held from 10 s to 350 s
    assert pm["busy_gpu_hours"] == pytest.approx(8.0 / 3600)       # serving from 42 s to 50 s
    assert pm["loading_gpu_hours"] == pytest.approx(32.0 / 3600)   # loading from 10 s to 42 s
    assert s["cold_starts"] == 1 and s["cold_start_rate"] == 1.0


def test_a_routed_query_blocks_scale_down():
    spin = Spin(cooldown_s=300, scale_to_zero=True, now=0.0)
    m = "qwen2.5_7B"
    serve(spin, m, 0.0, 48.0, 2.0)
    assert spin.request(m, 400.0) == WARM
    assert spin.idle_expired(400.0) == [] and not spin.stop(m, 400.0)


def test_static_deployment_holds_every_gpu():
    spin = Spin(scale_to_zero=False, now=0.0)
    assert all(spin.status(m) == WARM for m in MODELS)
    assert spin.idle_expired(10_000.0) == []
    assert spin.summary(3600.0)["gpu_hours"] == pytest.approx(sum(c["gpus"] for c in MODELS.values()))


def test_latency_estimate_follows_lifecycle_state():
    spin = Spin(scale_to_zero=True, now=0.0)
    m = "gemma3_27B"
    assert spin.latency_estimate(m, 0.0) is None
    serve(spin, m, 0.0, 95.0, 10.0)
    assert spin.latency_estimate(m, 106.0, "spin") == pytest.approx(10.0)
    assert spin.latency_estimate(m, 106.0, "observed") == pytest.approx(105.0)
    assert spin.stop(m, 500.0)
    assert spin.latency_estimate(m, 501.0, "spin") == pytest.approx(10.0 + MODELS[m]["cold_start_s"])
    assert spin.latency_estimate(m, 501.0, "inference") == pytest.approx(10.0)
    spin.request(m, 600.0)
    assert spin.latency_estimate(m, 650.0, "spin") == pytest.approx(10.0 + 600.0 + MODELS[m]["cold_start_s"] - 650.0)


def test_eq4_routes_around_a_cold_model():
    spin = Spin(scale_to_zero=True, now=0.0)
    serve(spin, "qwen2.5_7B", 0.0, 1.0, 3.0)
    serve(spin, "llama3.1_8B", 0.0, 1.0, 4.0)
    assert spin.stop("qwen2.5_7B", 1000.0)       # the faster 7B model is cold again, 8B is warm
    tier = {"MEDIUM": ["qwen2.5_7B", "llama3.1_8B"]}
    aware = Pick(None, spin, "spin", rng=FixedBeta(), tiers=tier)
    unaware = Pick(None, spin, "inference", rng=FixedBeta(), tiers=tier)
    for p in (aware, unaware):
        p.sampler.n.update({"qwen2.5_7B": 1, "llama3.1_8B": 1})
    assert aware.route(None, 1001.0, tier="MEDIUM")["model"] == "llama3.1_8B"
    assert unaware.route(None, 1001.0, tier="MEDIUM")["model"] == "qwen2.5_7B"


def test_eq4_score():
    spin = Spin(scale_to_zero=False, now=0.0)
    serve(spin, "qwen2.5_7B", 0.0, 0.0, 2.0)
    serve(spin, "llama3.1_8B", 0.0, 0.0, 8.0)
    pick = Pick(None, spin, "spin", rng=FixedBeta(), tiers={"MEDIUM": ["qwen2.5_7B", "llama3.1_8B"]})
    pick.sampler.n.update({"qwen2.5_7B": 3, "llama3.1_8B": 3})
    model, score = pick.sampler.select("MEDIUM", lambda m: spin.latency_estimate(m, 10.0))
    # S = 0.7 * 0.5 + 0.3 * (1 - 2/8) + 0.1 / sqrt(4)
    assert model == "qwen2.5_7B" and score == pytest.approx(0.35 + 0.225 + 0.05)


def test_loads_share_storage_bandwidth():
    st = SharedStorage(gbps=1.0)
    st.begin("qwen2.5_7B", 0.0)                       # 14 GB
    assert st.next_transfer_done() == pytest.approx((14.0, "qwen2.5_7B"))
    st.begin("llama3.1_8B", 4.0)                      # 16 GB; 10 GB of the 7B left, each now at 0.5 GB/s
    t, m = st.next_transfer_done()
    assert m == "qwen2.5_7B" and t == pytest.approx(24.0)
    st.transfer_done("qwen2.5_7B", 24.0)              # the 8B moved 10 GB, 6 GB left at full speed
    t, m = st.next_transfer_done()
    assert m == "llama3.1_8B" and t == pytest.approx(30.0)
    assert init_seconds("gemma3_27B") == pytest.approx(95 - 54 / 1.2)


def tiny_dataset(n=40, latency=2.0):
    queries = [{"id": f"q{i}", "benchmark": "b", "query": "x"} for i in range(n)]
    runs = {(q["id"], m): (True, latency) for q in queries for m in MODELS}
    correct = {(q["id"], m): i % 2 == 0 for i, q in enumerate(queries) for m in MODELS}
    tiers = {q["id"]: ("SIMPLE", "keyword") for q in queries}
    return queries, runs, correct, tiers


def test_simulator_static_and_scale_to_zero():
    data = tiny_dataset()
    recs, s = simulate.simulate("static", 0, *data, workers=4)
    assert len(recs) == 40 and s["cold_starts"] == 0 and all(r["wait_s"] == 0 for r in recs)
    assert s["gpu_hours"] == pytest.approx(len(MODELS) * s["makespan_s"] / 3600)
    recs, s = simulate.simulate("pick-and-spin", 0, *data, workers=4)
    first = min(recs, key=lambda r: r["order"])
    assert s["cold_starts"] >= 1 and first["cold_start"]
    assert first["wait_s"] >= MODELS[first["model"]]["cold_start_s"] - 1e-6
    assert all(MODELS[r["model"]]["tier"] == "SIMPLE" for r in recs)


def test_sparse_load_scales_models_down():
    data = tiny_dataset(n=20)
    recs, s = simulate.simulate("pick-and-spin", 0, *data, arrival_rate=1 / 1000)
    static = simulate.simulate("static", 0, *data, arrival_rate=1 / 1000)[1]
    assert s["cold_starts"] >= 10
    assert s["gpu_hours"] < 0.2 * static["gpu_hours"]


def test_simulator_is_deterministic():
    data = tiny_dataset()
    a = simulate.simulate("pick-and-spin", 3, *data, arrival_rate=0.01)[1]
    b = simulate.simulate("pick-and-spin", 3, *data, arrival_rate=0.01)[1]
    assert (a["gpu_hours"], a["cold_starts"]) == (b["gpu_hours"], b["cold_starts"])
