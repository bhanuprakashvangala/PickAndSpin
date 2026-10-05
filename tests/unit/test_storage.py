"""Eq. 5 with contention: weight transfers that share the storage bandwidth, and L_init."""

import pytest

from pickspin.config import DEFAULT_SPIN, MODELS
from pickspin.spin import ModelState, SharedStorage, Spin, init_seconds

# --- ported from tests/test_pickspin.py at v1.1.0 ------------------------------------------------


def test_loads_share_storage_bandwidth() -> None:
    st = SharedStorage(gbps=1.0)
    st.begin("qwen2.5_7B", 0.0)  # 14 GB
    assert st.next_transfer_done() == pytest.approx((14.0, "qwen2.5_7B"))
    st.begin("llama3.1_8B", 4.0)  # 16 GB; 10 GB of the 7B left, each now at 0.5 GB/s
    t, m = st.next_transfer_done()
    assert m == "qwen2.5_7B"
    assert t == pytest.approx(24.0)
    st.transfer_done("qwen2.5_7B", 24.0)  # the 8B moved 10 GB, 6 GB left at full speed
    t, m = st.next_transfer_done()
    assert m == "llama3.1_8B"
    assert t == pytest.approx(30.0)
    assert init_seconds("gemma3_27B") == pytest.approx(95 - 54 / 1.2)


# --- estimate() advances the storage: a load-bearing side effect -----------------------------------


def test_estimate_advances_the_transfers_in_flight() -> None:
    """estimate(m, now) first moves the transfers in flight to now. The simulator depends on it.

    The simulator uses storage.estimate as Spin's load estimator, so every cold request and every
    'spin' latency of a cold model advances the storage clock. Keep the call and the mutation.
    """
    storage = SharedStorage()  # 1.2 GB/s
    storage.begin("qwen2.5_7B", 0.0)  # 14 GB
    storage.begin("llama3.1_8B", 2.0)  # 16 GB; the 7B moved 2.4 GB alone
    assert storage.t == 2.0
    assert storage.remaining == {"qwen2.5_7B": 11.6, "llama3.1_8B": 16.0}
    # A load beginning now would be the third transfer: 18 GB * 3 / 1.2 GB/s, then L_init = 48 - 18 / 1.2.
    assert storage.estimate("gemma2_9B", 3.0) == 78.0
    assert storage.t == 3.0  # the clock moved to now ...
    assert storage.remaining == {"qwen2.5_7B": 11.0, "llama3.1_8B": 15.4}  # ... and each transfer by 0.6 GB
    assert "gemma2_9B" not in storage.remaining  # nothing began


def test_estimate_changes_later_transfer_times_at_the_bit_level() -> None:
    """In exact arithmetic the extra advance changes nothing; in floating point it changes the last bit.

    These are exact IEEE-754 results (only +, -, * and / are involved), the same on every platform.
    """
    plain, consulted = SharedStorage(), SharedStorage()
    for storage in (plain, consulted):
        storage.begin("qwen2.5_7B", 0.0)
        storage.begin("llama3.1_8B", 2.0)
    consulted.estimate("gemma2_9B", 3.0)
    assert plain.next_transfer_done() == (21.333333333333332, "qwen2.5_7B")
    assert consulted.next_transfer_done() == (21.333333333333336, "qwen2.5_7B")


def test_a_cold_request_through_spin_advances_the_storage() -> None:
    """The simulator's wiring: Spin.request consults storage.estimate before storage.begin is called."""
    storage = SharedStorage(gbps=1.0)
    spin = Spin(now=0.0, load_estimate=storage.estimate)
    assert spin.request("qwen2.5_7B", 0.0) == ModelState.COLD
    storage.begin("qwen2.5_7B", 0.0)  # 14 GB
    assert spin.request("llama3.1_8B", 4.0) == ModelState.COLD
    assert storage.t == 4.0
    assert storage.remaining == {"qwen2.5_7B": 10.0}  # advanced by the request, before the 8B began
    storage.begin("llama3.1_8B", 4.0)
    assert storage.next_transfer_done() == (24.0, "qwen2.5_7B")
    assert spin.request("qwen2.5_7B", 5.0) == ModelState.LOADING  # no new load, no estimate
    assert storage.t == 4.0
    assert storage.remaining == {"qwen2.5_7B": 10.0, "llama3.1_8B": 16.0}


# --- the fluid sharing model ------------------------------------------------------------------------


def test_transfer_arithmetic_matches_v1_1_0_bit_for_bit() -> None:
    """Exact results of the v1.1.0 expressions, which must not be reassociated:

    (now - t) * gbps / len(remaining), t + max(0.0, rem) * len(remaining) / gbps and
    weight_gb * k / gbps + init_seconds(m, gbps). Writing (now - t) * (gbps / len(remaining)),
    rem / (gbps / len(remaining)) or weight_gb / gbps * k changes the last bits below. Only +, -, *
    and / are involved, so these results are the same on every platform.
    """
    storage = SharedStorage()  # 1.2 GB/s
    storage.begin("qwen2.5_7B", 0.0)
    storage.begin("llama3.1_8B", 0.1)
    storage.begin("gemma2_9B", 0.6)
    assert storage.estimate("qwen2.5_14B", 3.0) == 135.0
    assert storage.remaining == {
        "qwen2.5_7B": 12.620000000000001,
        "llama3.1_8B": 14.739999999999998,
        "gemma2_9B": 17.04,
    }
    assert storage.next_transfer_done() == (34.55, "qwen2.5_7B")
    crowded = SharedStorage()
    for m in ("llama3.1_8B", "gemma2_9B", "qwen2.5_14B", "gemma3_27B"):
        crowded.begin(m, 0.0)
    # A fifth transfer: 14 GB * 5 / 1.2 GB/s, then L_init = 48 - 14 / 1.2.
    assert crowded.estimate("qwen2.5_7B", 6.0) == 94.66666666666666


def test_uncontended_load_takes_the_stated_cold_start() -> None:
    assert SharedStorage().gbps == DEFAULT_SPIN.storage_gbps == 1.2
    for m, spec in MODELS.items():
        assert init_seconds(m) == init_seconds(m, 1.2) == max(0.0, spec.cold_start_s - spec.weight_gb / 1.2)
        assert spec.weight_gb / 1.2 + init_seconds(m) == pytest.approx(spec.cold_start_s)
        assert SharedStorage().estimate(m, 10.0) == pytest.approx(spec.cold_start_s)


def test_init_seconds_is_never_negative() -> None:
    # On 0.5 GB/s the 27B transfer alone (108 s) takes longer than the stated 95 s cold start.
    assert init_seconds("gemma3_27B", 0.5) == 0.0
    assert init_seconds("llama3.2_1B", 0.5) == 32 - 4.0


def test_estimate_counts_a_transfer_in_flight_once() -> None:
    storage = SharedStorage(gbps=1.0)
    storage.begin("qwen2.5_7B", 0.0)
    # The 7B is already transferring, so it is not counted twice: one transfer.
    assert storage.estimate("qwen2.5_7B", 0.0) == 14 * 1 / 1.0 + init_seconds("qwen2.5_7B", 1.0)
    # Another model would be the second transfer.
    assert storage.estimate("llama3.1_8B", 0.0) == 16 * 2 / 1.0 + init_seconds("llama3.1_8B", 1.0)


def test_ties_go_to_the_transfer_that_began_first() -> None:
    storage = SharedStorage(gbps=1.0)
    storage.begin("llama3.2_3B", 0.0)  # 6 GB
    storage.begin("gemma2_2B", 2.0)  # 4 GB, as much as the 3B has left
    assert storage.remaining == {"llama3.2_3B": 4.0, "gemma2_2B": 4.0}
    assert storage.next_transfer_done() == (10.0, "llama3.2_3B")
    storage.transfer_done("llama3.2_3B", 10.0)
    assert storage.next_transfer_done() == (10.0, "gemma2_2B")


def test_overdue_transfer_finishes_at_the_storage_clock() -> None:
    storage = SharedStorage(gbps=1.0)
    storage.begin("llama3.2_1B", 0.0)  # 2 GB, done at 2 s
    storage.estimate("qwen2.5_7B", 5.0)  # advances past the end of the transfer
    assert storage.remaining == {"llama3.2_1B": -3.0}
    assert storage.next_transfer_done() == (5.0, "llama3.2_1B")


def test_storage_clock_never_moves_backwards() -> None:
    storage = SharedStorage(gbps=1.0)
    storage.begin("qwen2.5_7B", 5.0)
    storage.estimate("llama3.1_8B", 3.0)
    assert storage.t == 5.0
    assert storage.remaining == {"qwen2.5_7B": 14.0}
    storage.begin("llama3.1_8B", 4.0)  # counted from the storage clock, 5 s
    assert storage.next_transfer_done() == (5.0 + 14.0 * 2 / 1.0, "qwen2.5_7B")


def test_next_transfer_done_without_transfers_is_none() -> None:
    storage = SharedStorage()
    assert storage.next_transfer_done() is None
    storage.begin("gemma2_9B", 1.0)
    storage.transfer_done("gemma2_9B", 16.0)
    assert storage.next_transfer_done() is None
    assert storage.remaining == {}
    assert storage.t == 16.0
