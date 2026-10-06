"""Spin: the lifecycle state machine, its GPU accounting and the latency it reports to Pick (Sec. V)."""

import dataclasses
import json
import math
import threading
from collections.abc import Callable

import pytest

from pickspin.config import DEFAULT_SPIN, MODELS
from pickspin.spin import LatencySignal, ModelState, ModelUsage, Spin, SpinSummary

COLD, LOADING, WARM = ModelState.COLD, ModelState.LOADING, ModelState.WARM

# The serve fixture of tests/conftest.py: serve(spin, model, t_request, t_ready, infer_s) -> end time.
Serve = Callable[[Spin, str, float, float, float], float]


# --- ported from tests/test_pickspin.py at v1.1.0 ------------------------------------------------


def test_state_machine_and_gpu_accounting() -> None:
    spin = Spin(cooldown_s=300, scale_to_zero=True, now=0.0)
    m = "llama3.2_1B"
    assert spin.status(m) == COLD
    assert spin.request(m, 10.0) == COLD
    assert spin.status(m) == LOADING
    assert spin.idle_expired(1000.0) == []
    spin.loaded(m, 42.0)
    spin.start(m, 42.0)
    assert spin.status(m) == WARM
    spin.finish(m, 50.0, infer_s=8.0, total_s=40.0)
    assert spin.idle_expired(349.0) == []
    assert spin.idle_expired(350.0) == [m]
    assert spin.stop(m, 350.0)
    assert spin.status(m) == COLD
    s = spin.summary(400.0)
    pm = s.per_model[m]
    assert pm.gpu_hours == pytest.approx(340.0 / 3600)  # GPU held from 10 s to 350 s
    assert pm.busy_gpu_hours == pytest.approx(8.0 / 3600)  # serving from 42 s to 50 s
    assert pm.loading_gpu_hours == pytest.approx(32.0 / 3600)  # loading from 10 s to 42 s
    assert s.cold_starts == 1
    assert s.cold_start_rate == 1.0


def test_a_routed_query_blocks_scale_down(serve: Serve) -> None:
    spin = Spin(cooldown_s=300, scale_to_zero=True, now=0.0)
    m = "qwen2.5_7B"
    serve(spin, m, 0.0, 48.0, 2.0)
    assert spin.request(m, 400.0) == WARM
    assert spin.idle_expired(400.0) == []
    assert not spin.stop(m, 400.0)


def test_a_lost_load_returns_the_model_to_cold_and_releases_its_gpus() -> None:
    spin = Spin(cooldown_s=300, scale_to_zero=True, now=0.0)
    m = "llama3.2_1B"
    assert spin.request(m, 10.0) == COLD
    assert spin.request(m, 12.0) == LOADING
    assert spin.lost(m, 40.0)  # the load failed after 30 s
    assert spin.status(m) == COLD
    assert not spin.lost(m, 41.0)  # already COLD
    spin.cancel(m)
    spin.cancel(m)  # both waiting queries give up
    assert spin.request(m, 50.0) == COLD  # the next query is a new cold start
    spin.loaded(m, 60.0)
    assert spin.idle_expired(360.0) == []  # that query is still pending
    spin.start(m, 60.0)
    spin.finish(m, 61.0, infer_s=1.0, total_s=11.0)
    assert spin.idle_expired(361.0) == [m]
    usage = spin.summary(61.0).per_model[m]
    assert usage.cold_starts == 2
    assert usage.loading_gpu_hours == pytest.approx((30.0 + 10.0) / 3600)
    assert usage.gpu_hours == pytest.approx((30.0 + 11.0) / 3600)


def test_a_warm_model_whose_server_died_is_lost_even_while_busy() -> None:
    spin = Spin(cooldown_s=300, scale_to_zero=True, now=0.0)
    m = "qwen2.5_7B"
    spin.request(m, 0.0)
    spin.loaded(m, 10.0)
    spin.start(m, 10.0)
    assert spin.lost(m, 20.0)  # the query in flight still finishes, as a failure
    assert spin.status(m) == COLD
    spin.finish(m, 21.0, infer_s=11.0, total_s=21.0)
    assert spin.inflight(m) == 0
    usage = spin.summary(100.0).per_model[m]
    assert usage.gpu_hours == pytest.approx(20.0 / 3600)
    assert usage.busy_gpu_hours == pytest.approx(11.0 / 3600)


def test_static_deployment_holds_every_gpu() -> None:
    spin = Spin(scale_to_zero=False, now=0.0)
    assert all(spin.status(m) == WARM for m in MODELS)
    assert spin.idle_expired(10_000.0) == []
    assert spin.summary(3600.0).gpu_hours == pytest.approx(sum(c.gpus for c in MODELS.values()))


def test_latency_estimate_follows_lifecycle_state(serve: Serve) -> None:
    spin = Spin(scale_to_zero=True, now=0.0)
    m = "gemma3_27B"
    assert spin.latency_estimate(m, 0.0) is None
    serve(spin, m, 0.0, 95.0, 10.0)
    assert spin.latency_estimate(m, 106.0, "spin") == pytest.approx(10.0)
    assert spin.latency_estimate(m, 106.0, "observed") == pytest.approx(105.0)
    assert spin.stop(m, 500.0)
    assert spin.latency_estimate(m, 501.0, "spin") == pytest.approx(10.0 + MODELS[m].cold_start_s)
    assert spin.latency_estimate(m, 501.0, "inference") == pytest.approx(10.0)
    spin.request(m, 600.0)
    assert spin.latency_estimate(m, 650.0, "spin") == pytest.approx(10.0 + 600.0 + MODELS[m].cold_start_s - 650.0)


# --- construction and the state machine -----------------------------------------------------------


def test_defaults_follow_the_paper_configuration() -> None:
    spin = Spin()
    assert spin.models == list(MODELS)
    assert spin.cooldown_s == DEFAULT_SPIN.cooldown_s == 300.0
    assert spin.scale_to_zero
    assert spin.t0 == 0.0
    assert spin.queries == 0
    for m, spec in MODELS.items():
        assert spin.status(m) == COLD
        assert spin.inflight(m) == 0
        estimate = spin.load_estimate(m, 123.0)  # the stated cold-start time, whatever the time
        assert type(estimate) is float
        assert estimate == spec.cold_start_s
    subset = Spin(iter(["gemma2_9B", "llama3.2_1B"]), scale_to_zero=False, now=7.5)
    assert subset.models == ["gemma2_9B", "llama3.2_1B"]
    assert subset.t0 == 7.5
    assert list(subset.summary(10.0).per_model) == ["gemma2_9B", "llama3.2_1B"]


def test_inflight_counts_executing_queries(serve: Serve) -> None:
    spin = Spin(now=0.0)
    m = "llama3.1_8B"
    spin.request(m, 0.0)
    spin.request(m, 1.0)
    assert spin.inflight(m) == 0  # routed but still waiting for the load
    spin.loaded(m, 42.0)
    spin.start(m, 42.0)
    spin.start(m, 42.0)
    assert spin.inflight(m) == 2
    spin.finish(m, 45.0, 3.0, 45.0)
    assert spin.inflight(m) == 1
    spin.finish(m, 46.0, 4.0, 45.0)
    assert spin.inflight(m) == 0
    assert spin.inflight("qwen2.5_7B") == 0
    serve(spin, "qwen2.5_7B", 50.0, 98.0, 1.0)
    assert spin.inflight("qwen2.5_7B") == 0
    assert spin.queries == 3


def test_pending_and_running_queries_block_scale_down_until_the_last_one_finishes() -> None:
    spin = Spin(cooldown_s=300.0, now=0.0)
    m = "gemma2_2B"
    spin.request(m, 0.0)
    spin.loaded(m, 38.0)
    assert spin.idle_expired(1000.0) == []  # the query that triggered the load has not started
    spin.start(m, 38.0)
    assert spin.idle_expired(1000.0) == []  # it is running
    spin.finish(m, 40.0, 2.0, 40.0)
    assert spin.idle_expired(339.5) == []
    assert spin.idle_expired(340.0) == [m]  # T_cooldown after the last query finished
    spin.request(m, 340.0)  # a query routed in time keeps the model up: pending, then running
    assert spin.idle_expired(1000.0) == []
    assert not spin.stop(m, 1000.0)
    spin.start(m, 1000.0)
    assert spin.idle_expired(2000.0) == []
    assert not spin.stop(m, 2000.0)
    spin.finish(m, 2001.0, 1001.0, 1661.0)
    assert spin.idle_expired(2300.5) == []  # the cooldown restarted at 2001 s
    assert spin.idle_expired(2301.0) == [m]
    assert spin.stop(m, 2301.0)
    assert not spin.stop(m, 2302.0)  # already COLD
    assert spin.summary(2400.0).per_model[m].gpu_hours == pytest.approx(2301.0 / 3600)


def test_idle_time_is_measured_as_now_minus_idle_since() -> None:
    """idle_expired compares now - idle_since with T_cooldown, exactly as v1.1.0 did.

    The simulator checks a model at finish + T_cooldown, and (212.3 + 300.0) - 212.3 is
    299.99999999999994, so that model is not scaled down then. Comparing now >= idle_since + T_cooldown
    instead would scale it down and change the seeded simulation results.
    """
    spin = Spin(cooldown_s=300.0, now=0.0)
    m = "llama3.2_1B"
    spin.request(m, 100.0)
    spin.loaded(m, 132.0)
    spin.start(m, 132.0)
    spin.finish(m, 212.3, 80.3, 112.3)
    assert (212.3 + 300.0) - 212.3 < 300.0
    assert spin.idle_expired(212.3 + 300.0) == []
    assert spin.idle_expired(math.nextafter(212.3 + 300.0, math.inf)) == [m]


def test_idle_models_are_listed_in_model_order(serve: Serve) -> None:
    spin = Spin(cooldown_s=10.0, now=0.0)
    for k, m in enumerate(reversed(MODELS)):
        serve(spin, m, float(k), 100.0, 1.0)
    assert spin.idle_expired(111.0) == list(MODELS)
    assert Spin(scale_to_zero=False).idle_expired(1e9) == []


def test_loaded_ignores_a_model_that_is_not_loading() -> None:
    spin = Spin(now=0.0)
    m = "qwen2.5_14B"
    spin.loaded(m, 5.0)
    assert spin.status(m) == COLD
    spin.request(m, 10.0)
    spin.loaded(m, 75.0)
    spin.loaded(m, 80.0)  # a second ready signal changes nothing
    assert spin.status(m) == WARM
    assert spin.summary(100.0).per_model[m].loading_gpu_hours == pytest.approx(65.0 / 3600)


# --- the latency Pick scores on ---------------------------------------------------------------------


def test_load_estimate_is_consulted_only_on_the_cold_paths(serve: Serve) -> None:
    """Spin calls its load estimator exactly where v1.1.0 did; the simulator's estimator has side effects."""
    calls: list[tuple[str, float]] = []

    def estimate(model: str, now: float) -> float:
        calls.append((model, now))
        return 30.0

    spin = Spin(cooldown_s=300.0, now=0.0, load_estimate=estimate)
    m = "qwen2.5_7B"
    assert spin.latency_estimate(m, 1.0) is None  # nothing served yet: not consulted
    serve(spin, m, 2.0, 32.0, 2.0)  # COLD -> LOADING consults it once, at the request
    assert calls == [(m, 2.0)]
    assert spin.latency_estimate(m, 40.0) == 2.0  # WARM: the inference latency only
    assert spin.stop(m, 400.0)
    assert spin.latency_estimate(m, 401.0, LatencySignal.OBSERVED) == 32.0  # the query waited 30 s for the load
    assert spin.latency_estimate(m, 401.0, LatencySignal.INFERENCE) == 2.0
    assert calls == [(m, 2.0)]
    assert spin.latency_estimate(m, 402.0) == 2.0 + 30.0  # COLD under 'spin': consulted at now
    assert calls == [(m, 2.0), (m, 402.0)]
    assert spin.request(m, 500.0) == COLD  # the load is expected to be done at 500 + 30 s
    assert spin.request(m, 501.0) == LOADING  # a second query does not start another load
    assert calls == [(m, 2.0), (m, 402.0), (m, 500.0)]
    assert spin.latency_estimate(m, 510.0) == 2.0 + 20.0  # LOADING: the time left on the load
    assert spin.latency_estimate(m, 600.0) == 2.0 + 0.0  # an overdue load adds nothing
    assert calls == [(m, 2.0), (m, 402.0), (m, 500.0)]


@pytest.mark.parametrize("signal", list(LatencySignal), ids=str)
def test_latency_signal_members_and_strings_agree(signal: LatencySignal, serve: Serve) -> None:
    spin = Spin(now=0.0)
    serve(spin, "gemma2_9B", 0.0, 48.0, 3.0)
    serve(spin, "gemma2_9B", 60.0, 60.0, 5.0)
    assert spin.stop("gemma2_9B", 400.0)
    expected = {LatencySignal.SPIN: 4.0 + 48.0, LatencySignal.OBSERVED: 28.0, LatencySignal.INFERENCE: 4.0}
    assert spin.latency_estimate("gemma2_9B", 401.0, signal) == expected[signal]
    assert spin.latency_estimate("gemma2_9B", 401.0, signal.value) == expected[signal]
    assert spin.latency_estimate("gemma2_9B", 401.0) == expected[LatencySignal.SPIN]


# --- accounting ---------------------------------------------------------------------------------------


def test_summary_counts_intervals_still_open_at_now() -> None:
    spin = Spin(now=0.0)
    loading, busy = "llama3.2_1B", "qwen2.5_7B"
    spin.request(loading, 10.0)  # still loading at 100 s
    spin.request(busy, 0.0)
    spin.loaded(busy, 48.0)
    spin.start(busy, 48.0)  # still serving at 100 s
    s = spin.summary(100.0)
    assert s.per_model[loading] == ModelUsage(
        gpu_hours=90.0 / 3600, busy_gpu_hours=0.0, loading_gpu_hours=90.0 / 3600, cold_starts=1
    )
    assert s.per_model[busy] == ModelUsage(
        gpu_hours=100.0 / 3600, busy_gpu_hours=52.0 / 3600, loading_gpu_hours=48.0 / 3600, cold_starts=1
    )
    assert s.gpu_utilization == pytest.approx(52.0 / 190.0)
    assert s.cold_start_rate == 1.0
    # A summary is a snapshot: the intervals stay open.
    assert spin.summary(200.0).per_model[busy].busy_gpu_hours == pytest.approx(152.0 / 3600)


def test_open_interval_is_added_to_the_closed_ones(serve: Serve) -> None:
    """GPU time is closed + (now - since), in seconds, then * gpus / 3600, as in v1.1.0 (bit for bit)."""
    spin = Spin(["qwen2.5_7B"], now=0.0)
    m = "qwen2.5_7B"
    serve(spin, m, 0.0, 0.05, 0.02)
    assert spin.stop(m, 0.1)
    spin.request(m, 0.2)
    gpu_hours = spin.summary(0.3).per_model[m].gpu_hours
    assert gpu_hours == (0.1 + (0.3 - 0.2)) * 1 / 3600 == 5.555555555555555e-05
    assert gpu_hours != ((0.1 + 0.3) - 0.2) / 3600


def test_summary_without_gpu_time_or_queries_reports_zero_rates() -> None:
    s = Spin(now=0.0).summary(1000.0)
    assert s.gpu_hours == 0.0
    assert s.busy_gpu_hours == 0.0
    assert s.gpu_utilization == 0.0
    assert s.cold_starts == 0
    assert s.cold_start_rate == 0.0
    assert all(u == ModelUsage(0.0, 0.0, 0.0, 0) for u in s.per_model.values())


def test_static_deployment_counts_gpu_time_from_t0(serve: Serve) -> None:
    spin = Spin(scale_to_zero=False, now=100.0)
    serve(spin, "gemma3_27B", 1000.0, 1000.0, 360.0)
    s = spin.summary(3700.0)
    assert s.gpu_hours == pytest.approx(len(MODELS) * 1.0)
    assert s.busy_gpu_hours == pytest.approx(0.1)
    assert s.gpu_utilization == pytest.approx(0.1 / len(MODELS))
    assert s.cold_starts == 0
    assert s.cold_start_rate == 0.0
    assert all(u.loading_gpu_hours == 0.0 for u in s.per_model.values())


def test_gpu_hours_count_every_gpu_of_a_model(serve: Serve) -> None:
    """Every model of the paper's pool has one GPU; a model on two GPUs holds both while it is up."""
    two_gpus = dataclasses.replace(MODELS["gemma3_27B"], gpus=2)
    spin = Spin(scale_to_zero=False, now=0.0, catalog={**MODELS, "gemma3_27B": two_gpus})
    serve(spin, "gemma3_27B", 0.0, 0.0, 360.0)
    s = spin.summary(3600.0)
    assert s.per_model["gemma3_27B"] == ModelUsage(
        gpu_hours=2.0, busy_gpu_hours=0.2, loading_gpu_hours=0.0, cold_starts=0
    )
    assert s.per_model["llama3.2_1B"].gpu_hours == 1.0
    assert s.gpu_hours == pytest.approx(len(MODELS) + 1.0)


def test_summary_totals_use_the_builtin_sum_in_model_order() -> None:
    """Totals are sum() over the models in order: compensated since Python 3.12, unlike a += loop."""
    spin = Spin(["qwen2.5_7B", "llama3.1_8B", "gemma2_9B"], now=0.0)
    for m, t in zip(spin.models, (3240.0, 2880.0, 2520.0)):
        spin.request(m, t)  # GPUs held for 360, 720 and 1080 s by 3600 s
    s = spin.summary(3600.0)
    hours = [u.gpu_hours for u in s.per_model.values()]
    assert hours == [0.1, 0.2, 0.3]
    assert s.gpu_hours == sum(hours)  # 0.6 on 3.12+; 0.1 + 0.2 + 0.3 is 0.6000000000000001
    assert s.cold_starts == 3
    assert type(s.cold_starts) is int


def test_summary_to_dict_has_the_old_keys_in_order(serve: Serve) -> None:
    spin = Spin(now=0.0)
    serve(spin, "llama3.2_3B", 0.0, 32.0, 1.5)
    s = spin.summary(60.0)
    d = s.to_dict()
    assert list(d) == ["gpu_hours", "busy_gpu_hours", "gpu_utilization", "cold_starts", "cold_start_rate", "per_model"]
    assert [f.name for f in dataclasses.fields(SpinSummary)] == list(d)
    assert list(d["per_model"]) == list(MODELS)
    for m, usage in d["per_model"].items():
        assert list(usage) == ["gpu_hours", "busy_gpu_hours", "loading_gpu_hours", "cold_starts"]
        assert usage == dataclasses.asdict(s.per_model[m])
    assert d["gpu_hours"] == s.gpu_hours
    assert d["per_model"]["llama3.2_3B"]["cold_starts"] == 1
    # The live runner adds measured_load_s to the dict and writes it with json.dump(..., indent=1).
    d["measured_load_s"] = {}
    assert "measured_load_s" not in s.to_dict()
    assert json.loads(json.dumps(d, indent=1)) == d


def test_summary_records_are_frozen() -> None:
    s = Spin(now=0.0).summary(1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.gpu_hours = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.per_model["llama3.2_1B"].cold_starts = 1  # type: ignore[misc]


# --- threads ------------------------------------------------------------------------------------------


def test_concurrent_queries_are_counted_exactly() -> None:
    """The live runner calls Spin from many threads; one lock guards every change."""
    spin = Spin(scale_to_zero=False, now=0.0)
    m = "llama3.2_1B"
    threads, rounds = 8, 500
    barrier = threading.Barrier(threads)
    states: list[ModelState] = []
    errors: list[Exception] = []

    def worker() -> None:
        try:
            barrier.wait()
            for _ in range(rounds):
                states.append(spin.request(m, 1.0))
                spin.start(m, 1.0)
                spin.latency_estimate(m, 1.0)
                spin.finish(m, 2.0, 1.0, 1.0)
                spin.summary(2.0)
        except Exception as e:
            errors.append(e)

    pool = [threading.Thread(target=worker) for _ in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    assert errors == []
    assert states == [WARM] * (threads * rounds)
    assert spin.queries == threads * rounds
    assert spin.inflight(m) == 0
    assert spin.latency_estimate(m, 3.0, LatencySignal.OBSERVED) == 1.0  # total_sum / n: no update was lost
    assert spin.summary(3.0).cold_starts == 0
