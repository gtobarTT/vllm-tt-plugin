# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Unit tests for ``max_prefill_requests_per_step``.

The key caps how many new requests one prefill step admits, summed over all
lanes, oldest waiting request first. A prompt's first token is sampled at the
end of the step that finishes its prefill, so admitting fewer prompts per step
lets each one finish (and stream its first token) sooner. Partial-prefill
continuations are not new requests and are never held back.

These tests drive real ``TTScheduler`` lanes built by the host-only fixture in
``tests/test_block_scheduler.py``; nothing touches a device.
"""

from types import SimpleNamespace

import pytest

from tests.test_block_scheduler import BLOCK_SIZE, _request
from tests.test_block_scheduler import _scheduler as _real_scheduler
from vllm_tt_plugin.config import (
    get_tt_max_prefill_requests_per_step,
    store_tt_lane_count,
)
from vllm_tt_plugin.lane_scheduler import TTLaneCoordinator
from vllm_tt_plugin.scheduler import TTScheduler

_KEY = "max_prefill_requests_per_step"


def _config(**tt_keys):
    return SimpleNamespace(additional_config={"tt": dict(tt_keys)} if tt_keys else {})


# --- Configuration ---------------------------------------------------------


def test_cap_is_absent_by_default():
    assert get_tt_max_prefill_requests_per_step(_config()) is None


def test_cap_reads_a_positive_integer():
    assert get_tt_max_prefill_requests_per_step(_config(**{_KEY: 3})) == 3


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "1", None])
def test_invalid_cap_values_raise(value):
    with pytest.raises(ValueError, match=_KEY):
        get_tt_max_prefill_requests_per_step(_config(**{_KEY: value}))


# --- Helpers ---------------------------------------------------------------


def _set_tt_keys(vllm_config, tt_keys):
    tt = vllm_config.additional_config.setdefault("tt", {})
    # Interleave off: these tests watch admission order across prefill steps,
    # and an inserted decode step would only add empty rows to that record.
    tt["decode_interleave_enabled"] = False
    tt.update(tt_keys)


def _chunked(vllm_config, budget):
    """Let a prompt longer than ``budget`` tokens span several prefill steps."""
    scheduler_config = vllm_config.scheduler_config
    scheduler_config.enable_chunked_prefill = True
    scheduler_config.max_num_batched_tokens = budget
    scheduler_config.max_num_scheduled_tokens = budget


def _single(*, max_num_seqs=4, max_model_len=256, budget=None, **tt_keys):
    """A real single (non-lane) TT scheduler with ``tt_keys`` configured."""
    base = _real_scheduler(
        output_width=1, max_num_seqs=max_num_seqs, max_model_len=max_model_len
    )
    _set_tt_keys(base.vllm_config, tt_keys)
    if budget is not None:
        _chunked(base.vllm_config, budget)
    return TTScheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=base.kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=base.structured_output_manager,
    )


def _coordinator(*, lanes=3, per_lane=4, max_model_len=256, budget=None, **tt_keys):
    """A real lane coordinator over real ``TTScheduler`` lanes."""
    base = _real_scheduler(
        output_width=1, max_num_seqs=lanes * per_lane, max_model_len=max_model_len
    )
    store_tt_lane_count(base.vllm_config, lanes)
    _set_tt_keys(base.vllm_config, tt_keys)
    if budget is not None:
        _chunked(base.vllm_config, budget)
    return TTLaneCoordinator(
        base.vllm_config,
        base.kv_cache_config,
        base.structured_output_manager,
        BLOCK_SIZE,
    )


def _req(request_id, arrival_time, prompt_length=32):
    request = _request(4, request_id=request_id, prompt_length=prompt_length)
    request.arrival_time = arrival_time
    return request


def _admit_to_lane(coordinator, lane, request):
    """Bind ``request`` to ``lane``, as least-loaded routing would on its own."""
    coordinator._req_to_lane[request.request_id] = lane
    coordinator.add_request(request)


def _steps(scheduler, n):
    """The requests (and token counts) each of ``n`` steps scheduled."""
    return [dict(scheduler.schedule().num_scheduled_tokens) for _ in range(n)]


def _new_ids(steps):
    return [sorted(step) for step in steps]


# Six short prompts over three lanes, two per lane. Each lane's queue is in
# arrival order, but the global arrival order interleaves the lanes:
# r1 (lane 1) < r2 (lane 2) < r5 (lane 2) < r0 (lane 0) < r3 (lane 0) < r4.
_ARRIVALS = {"r0": 30, "r1": 10, "r2": 20, "r3": 40, "r4": 50, "r5": 25}
_LANE_OF = {"r0": 0, "r3": 0, "r1": 1, "r4": 1, "r2": 2, "r5": 2}


def _six_prompt_coordinator(**tt_keys):
    coordinator = _coordinator(**tt_keys)
    for request_id in ("r0", "r1", "r2", "r3", "r4", "r5"):
        _admit_to_lane(
            coordinator,
            _LANE_OF[request_id],
            _req(request_id, _ARRIVALS[request_id]),
        )
    return coordinator


# --- Lane mode -------------------------------------------------------------


def test_lanes_without_a_cap_admit_every_waiting_prompt_in_one_step():
    # Today's behaviour, which an absent key must keep: every lane enters the
    # shared prefill step and admits everything that fits.
    coordinator = _six_prompt_coordinator()

    assert _new_ids(_steps(coordinator, 1)) == [["r0", "r1", "r2", "r3", "r4", "r5"]]


def test_cap_of_one_admits_one_prompt_per_step_oldest_first_across_lanes():
    coordinator = _six_prompt_coordinator(**{_KEY: 1})

    assert _new_ids(_steps(coordinator, 6)) == [
        ["r1"],
        ["r2"],
        ["r5"],
        ["r0"],
        ["r3"],
        ["r4"],
    ]


def test_cap_of_two_takes_the_two_oldest_even_from_one_lane():
    coordinator = _six_prompt_coordinator(**{_KEY: 2})

    assert _new_ids(_steps(coordinator, 3)) == [
        ["r1", "r2"],
        ["r0", "r5"],
        ["r3", "r4"],
    ]


def test_a_lane_partial_prefill_advances_beside_the_capped_admission():
    # A (lane 0) needs two steps of a 256-token budget. In step 2 its
    # continuation is not a new request: it runs alongside the one prompt the
    # cap admits, and the cap goes to B, the oldest waiting prompt.
    coordinator = _coordinator(max_model_len=1024, budget=256, **{_KEY: 1})
    _admit_to_lane(coordinator, 0, _req("A", 1, prompt_length=384))
    _admit_to_lane(coordinator, 1, _req("B", 2))
    _admit_to_lane(coordinator, 2, _req("C", 3))

    assert _steps(coordinator, 3) == [
        {"A": 256},
        {"A": 128, "B": 32},
        {"C": 32},
    ]


def test_a_full_lane_does_not_swallow_the_cap():
    # Lane 0's only slot is taken, so its waiting prompt cannot be admitted
    # this step even though it is the oldest. The cap goes to the next-oldest
    # prompt that can run instead of leaving the step empty.
    coordinator = _coordinator(lanes=2, per_lane=1, **{_KEY: 1})
    _admit_to_lane(coordinator, 0, _req("X", 0))
    assert _new_ids(_steps(coordinator, 1)) == [["X"]]
    _admit_to_lane(coordinator, 0, _req("Y", 1))
    _admit_to_lane(coordinator, 1, _req("Z", 2))

    assert _new_ids(_steps(coordinator, 1)) == [["Z"]]


# --- Single-scheduler mode -------------------------------------------------


def test_single_scheduler_without_a_cap_admits_every_waiting_prompt():
    scheduler = _single()
    for i in range(3):
        scheduler.add_request(_req(f"s{i}", i))

    assert _new_ids(_steps(scheduler, 1)) == [["s0", "s1", "s2"]]


def test_single_scheduler_cap_of_one_admits_one_prompt_per_step():
    scheduler = _single(**{_KEY: 1})
    for i in range(3):
        scheduler.add_request(_req(f"s{i}", i))

    assert _new_ids(_steps(scheduler, 3)) == [["s0"], ["s1"], ["s2"]]


def test_single_scheduler_partial_prefill_is_not_held_back_by_the_cap():
    scheduler = _single(max_model_len=1024, budget=256, **{_KEY: 1})
    scheduler.add_request(_req("A", 1, prompt_length=384))
    scheduler.add_request(_req("B", 2))
    scheduler.add_request(_req("C", 3))

    assert _steps(scheduler, 3) == [
        {"A": 256},
        {"A": 128, "B": 32},
        {"C": 32},
    ]
