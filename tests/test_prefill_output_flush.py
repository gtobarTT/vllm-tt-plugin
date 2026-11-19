# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Unit tests for ``flush_prefill_output``.

Under async scheduling vLLM's ``EngineCore.step_with_batch_queue`` keeps up to
two steps in its batch queue and publishes a step's tokens only after the next
step has been submitted. On TT a prefill submit runs the whole prefill before
it returns, so a prefill step's first tokens wait for the following step: a
whole second prefill when two prefill steps run back to back. With the key on,
a queued prefill step's output is published before the next step is scheduled.
Decode steps keep their one-step overlap.

The engine under test is vLLM's real ``EngineCore.step_with_batch_queue`` over
a real ``TTScheduler`` or lane coordinator from the host-only fixtures. Only
the executor, the device boundary, is faked; it runs each step at once, as the
TT uniproc executor does.
"""

import queue
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

from tests.test_block_scheduler import BLOCK_SIZE, _request
from tests.test_block_scheduler import _scheduler as _real_scheduler
from vllm_tt_plugin.config import (
    get_tt_arrival_wait_ms,
    get_tt_flush_prefill_output,
    store_tt_lane_count,
)
from vllm_tt_plugin.lane_scheduler import TTLaneCoordinator
from vllm_tt_plugin.platform import _install_tt_prefill_output_flush_patch
from vllm_tt_plugin.scheduler import TTScheduler

_KEY = "flush_prefill_output"


def _config(**tt_keys):
    return SimpleNamespace(additional_config={"tt": dict(tt_keys)} if tt_keys else {})


# --- Configuration ---------------------------------------------------------


def test_flush_is_off_by_default():
    assert get_tt_flush_prefill_output(_config()) is False


@pytest.mark.parametrize("value", [True, False])
def test_flush_reads_a_boolean(value):
    assert get_tt_flush_prefill_output(_config(**{_KEY: value})) is value


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_non_boolean_flush_values_raise(value):
    with pytest.raises(ValueError, match=_KEY):
        get_tt_flush_prefill_output(_config(**{_KEY: value}))


# --- Engine harness --------------------------------------------------------


class _Device:
    """Executor stand-in. Runs each step when it is submitted and logs it.

    The TT uniproc executor runs ``execute_model`` and ``sample_tokens`` inline,
    so both futures are already resolved when vLLM queues them; this keeps that.
    A step is logged as ``("run", kind, req_ids)``, where ``kind`` is
    ``"prefill"`` when the step carries a new request and ``"decode"`` otherwise
    (the scenarios here neither chunk nor preempt).
    """

    def __init__(self, log):
        self.log = log
        self._step = None
        self._token = 100

    def execute_model(self, scheduler_output, non_block=False):
        req_ids = tuple(sorted(scheduler_output.num_scheduled_tokens))
        future = Future()
        if not req_ids:
            future.set_result(EMPTY_MODEL_RUNNER_OUTPUT)
            return future
        kind = "prefill" if scheduler_output.scheduled_new_reqs else "decode"
        self.log.append(("run", kind, req_ids))
        self._step = req_ids
        future.set_result(None)
        return future

    def sample_tokens(self, grammar_output, non_block=False):
        req_ids, self._step = self._step, None
        tokens = []
        for _ in req_ids:
            self._token += 1
            tokens.append([self._token])
        future = Future()
        future.set_result(
            ModelRunnerOutput(
                req_ids=list(req_ids),
                req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
                sampled_token_ids=tokens,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )
        )
        return future


def _engine(scheduler, vllm_config, log):
    """vLLM's ``EngineCore`` with only what ``step_with_batch_queue`` reads.

    ``batch_queue_size`` is 2, what vLLM uses under async scheduling.
    """
    _install_tt_prefill_output_flush_patch()
    engine = object.__new__(EngineCore)
    engine.vllm_config = vllm_config
    engine.scheduler = scheduler
    engine.model_executor = _Device(log)
    engine.batch_queue_size = 2
    engine.batch_queue = deque(maxlen=2)
    engine.is_mm_encoder_only = False
    engine.is_pooling_model = False
    engine.aborts_queue = queue.Queue()
    engine.log_stats = False
    return engine


def _drive(engine, log, calls):
    """Call the engine step ``calls`` times; log each publish as it is returned."""
    for _ in range(calls):
        outputs, _ = engine.step_with_batch_queue()
        published = sorted(
            output.request_id
            for eco in (outputs or {}).values()
            for output in eco.outputs
            if output.new_token_ids
        )
        if published:
            log.append(("publish", tuple(published)))


def _set_tt_keys(vllm_config, tt_keys):
    tt = vllm_config.additional_config.setdefault("tt", {})
    tt.update(tt_keys)


def _single(**tt_keys):
    """A real single (non-lane) TT scheduler under async scheduling."""
    base = _real_scheduler(output_width=1, max_num_seqs=4, async_scheduling=True)
    _set_tt_keys(base.vllm_config, tt_keys)
    return TTScheduler(
        vllm_config=base.vllm_config,
        kv_cache_config=base.kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=base.structured_output_manager,
    )


def _coordinator(**tt_keys):
    """A real two-lane coordinator over real ``TTScheduler`` lanes."""
    base = _real_scheduler(output_width=1, max_num_seqs=4, async_scheduling=True)
    store_tt_lane_count(base.vllm_config, 2)
    _set_tt_keys(base.vllm_config, tt_keys)
    return TTLaneCoordinator(
        base.vllm_config,
        base.kv_cache_config,
        base.structured_output_manager,
        BLOCK_SIZE,
    )


def _req(request_id, arrival_time):
    request = _request(16, request_id=request_id)
    request.arrival_time = arrival_time
    return request


# Two prompts, admitted one per prefill step (the prefix row's setting), so
# the two prefill steps run back to back.
_CAP_ONE = {"max_prefill_requests_per_step": 1}


def _two_prompts_single(flush):
    scheduler = _single(**_CAP_ONE, **{_KEY: flush})
    scheduler.add_request(_req("A", 1))
    scheduler.add_request(_req("B", 2))
    return scheduler, scheduler.vllm_config


def _two_prompts_lanes(flush):
    coordinator = _coordinator(**_CAP_ONE, **{_KEY: flush})
    for lane, request in enumerate((_req("A", 1), _req("B", 2))):
        coordinator._req_to_lane[request.request_id] = lane
        coordinator.add_request(request)
    return coordinator, coordinator.vllm_config


_TWO_PROMPTS = pytest.mark.parametrize(
    "two_prompts", [_two_prompts_single, _two_prompts_lanes], ids=["single", "lanes"]
)


# --- Behaviour -------------------------------------------------------------


@_TWO_PROMPTS
def test_without_the_key_a_prefill_token_waits_for_the_next_prefill(two_prompts):
    # vLLM's order, which an absent or false key must keep: A's first token is
    # published only after B's whole prefill has run.
    scheduler, vllm_config = two_prompts(False)
    log = []
    _drive(_engine(scheduler, vllm_config, log), log, 2)

    assert log == [
        ("run", "prefill", ("A",)),
        ("run", "prefill", ("B",)),
        ("publish", ("A",)),
    ]


@_TWO_PROMPTS
def test_with_the_key_a_prefill_token_is_published_before_the_next_step(
    two_prompts,
):
    scheduler, vllm_config = two_prompts(True)
    log = []
    _drive(_engine(scheduler, vllm_config, log), log, 4)

    assert log == [
        ("run", "prefill", ("A",)),
        ("publish", ("A",)),
        ("run", "prefill", ("B",)),
        ("publish", ("B",)),
    ]


@_TWO_PROMPTS
def test_with_the_key_decode_steps_keep_their_one_step_overlap(two_prompts):
    # After both prefills, each decode step's tokens are still published only
    # once the following decode step has been submitted.
    scheduler, vllm_config = two_prompts(True)
    log = []
    _drive(_engine(scheduler, vllm_config, log), log, 7)

    assert log[4:] == [
        ("run", "decode", ("A", "B")),
        ("run", "decode", ("A", "B")),
        ("publish", ("A", "B")),
        ("run", "decode", ("A", "B")),
        ("publish", ("A", "B")),
    ]


@pytest.mark.parametrize("flush", [False, True])
def test_a_prefill_submitted_behind_a_decode_is_flushed_on_the_next_call(flush):
    # A decode step is in flight when C arrives. C's prefill submit publishes
    # that decode (vLLM's own pop); with the key on, C's own token follows
    # before the next step runs instead of after it.
    scheduler = _single(**{_KEY: flush})
    scheduler.add_request(_req("A", 1))
    log = []
    engine = _engine(scheduler, scheduler.vllm_config, log)
    _drive(engine, log, 3 if flush else 2)
    log.clear()
    scheduler.add_request(_req("C", 2))
    _drive(engine, log, 3 if flush else 2)

    expected_tail = (
        [("publish", ("C",)), ("run", "decode", ("A", "C"))]
        if flush
        else [("run", "decode", ("A", "C")), ("publish", ("C",))]
    )
    assert log == [
        ("run", "prefill", ("C",)),
        ("publish", ("A",)),
        *expected_tail,
    ]


@pytest.mark.parametrize("flush", [False, True])
def test_the_engine_patch_is_installed_only_when_the_key_is_on(flush):
    import vllm.v1.engine.core as engine_core

    from vllm_tt_plugin.platform import _install_prefill_output_flush_if_requested

    upstream = EngineCore.step_with_batch_queue
    _install_prefill_output_flush_if_requested(_config(**{_KEY: flush}))

    assert (EngineCore.step_with_batch_queue is not upstream) is flush
    assert hasattr(engine_core, "_tt_original_step_with_batch_queue") is flush


# --- Experimental arrival wait (arrival_wait_ms) ----------------------------

_WAIT_KEY = "arrival_wait_ms"


def test_arrival_wait_is_off_by_default():
    assert get_tt_arrival_wait_ms(_config()) is None


@pytest.mark.parametrize("value", [0, 8, 2.5])
def test_arrival_wait_reads_a_non_negative_number_of_ms(value):
    margin = get_tt_arrival_wait_ms(_config(**{_WAIT_KEY: value}))

    assert margin == value
    assert isinstance(margin, float)


@pytest.mark.parametrize(
    "value", [-1, -0.5, float("nan"), float("inf"), "8", True, None]
)
def test_invalid_arrival_wait_values_raise(value):
    with pytest.raises(ValueError, match=_WAIT_KEY):
        get_tt_arrival_wait_ms(_config(**{_WAIT_KEY: value}))


@pytest.mark.parametrize("flush", [None, False])
def test_arrival_wait_without_flush_fails_at_startup(flush):
    from vllm_tt_plugin.platform import _install_prefill_output_flush_if_requested

    tt_keys = {_WAIT_KEY: 8} if flush is None else {_WAIT_KEY: 8, _KEY: flush}
    upstream = EngineCore.step_with_batch_queue
    with pytest.raises(ValueError, match=rf"{_WAIT_KEY}.*{_KEY}"):
        _install_prefill_output_flush_if_requested(_config(**tt_keys))
    assert EngineCore.step_with_batch_queue is upstream


def test_arrival_wait_with_flush_validates_at_startup():
    from vllm_tt_plugin.platform import _install_prefill_output_flush_if_requested

    with pytest.raises(ValueError, match=_WAIT_KEY):
        _install_prefill_output_flush_if_requested(
            _config(**{_KEY: True, _WAIT_KEY: -1})
        )


def _steady_decode_engine(monkeypatch, margin_ms, **tt_keys):
    """A single-lane engine in steady decode for A, flush on.

    The arrival wait is on when ``margin_ms`` is not None. The fake device runs
    each step at once, so the measured pop-to-pop intervals are ~0; the wait's
    state is set as if each decode step took 200 ms and the last pop happened
    just now.
    """
    import time

    from vllm_tt_plugin.platform import _ArrivalWait

    if margin_ms is not None:
        tt_keys[_WAIT_KEY] = margin_ms
    scheduler = _single(**{_KEY: True}, **tt_keys)
    scheduler.add_request(_req("A", 1))
    log = []
    engine = _engine(scheduler, scheduler.vllm_config, log)
    engine.input_queue = queue.Queue()
    engine._handle_client_request = lambda _type, request: scheduler.add_request(
        request
    )
    _drive(engine, log, 6)
    # One decode step in flight: where the wait applies.
    assert len(engine.batch_queue) == 1
    assert ("run", "decode", ("A",)) in log
    wait = engine._tt_arrival_wait
    if margin_ms is None:
        assert wait is False
        log.clear()
        return engine, scheduler, log
    assert isinstance(wait, _ArrivalWait)
    wait.intervals.clear()
    wait.intervals.append(0.2)
    wait.last_pop = time.perf_counter()
    wait.last_pop_decode = True
    log.clear()
    return engine, scheduler, log


def test_arrival_wait_schedules_a_request_that_arrives_before_the_deadline(
    monkeypatch,
):
    import threading

    engine, _, log = _steady_decode_engine(monkeypatch, 50)
    threading.Timer(0.02, engine.input_queue.put, args=((None, _req("C", 2)),)).start()
    _drive(engine, log, 1)

    # C's prefill takes the slot the next decode step would have had.
    assert log[0] == ("run", "prefill", ("C",))
    assert engine._tt_arrival_wait.n_hit == 1


def test_arrival_wait_without_a_request_submits_decode_at_the_deadline(monkeypatch):
    import time

    engine, _, log = _steady_decode_engine(monkeypatch, 50)
    start = time.perf_counter()
    _drive(engine, log, 1)
    waited = time.perf_counter() - start

    # Deadline = last pop + 200 ms - 50 ms margin.
    assert 0.12 < waited < 0.2
    assert log[0] == ("run", "decode", ("A",))
    assert engine._tt_arrival_wait.n_hit == 0


def test_arrival_wait_is_skipped_when_a_request_already_waits(monkeypatch):
    import time

    engine, scheduler, log = _steady_decode_engine(monkeypatch, 50)
    scheduler.add_request(_req("C", 2))
    start = time.perf_counter()
    _drive(engine, log, 1)

    assert time.perf_counter() - start < 0.05
    assert log[0] == ("run", "prefill", ("C",))
    assert engine._tt_arrival_wait.n_wait == 0


def test_without_the_key_decode_is_submitted_at_once(monkeypatch):
    import time

    engine, _, log = _steady_decode_engine(monkeypatch, None)
    start = time.perf_counter()
    _drive(engine, log, 1)

    assert time.perf_counter() - start < 0.05
    assert log[0] == ("run", "decode", ("A",))
