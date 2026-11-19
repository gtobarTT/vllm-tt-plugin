# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Unit tests for ``keep_decode_chain``.

Under async scheduling on the lane-DP path, a decode step whose only layout
change is a finished request's freed slot drains the pipeline and reloads every
input from the host, and the reloading step is never overlap-ok, so the step
after it drains too. With the key on, such a release keeps the device-resident
chain and a plain device-sampled reloading decode may be overlapped by the
next steady step. With the key absent or false neither happens.

The controller under test is the real ``TTAsyncDecodeController``; the runner
and the model adapter are host fakes (the device boundary). The startup checks
run through the real ``TTPlatform.check_and_update_config``.
"""

import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_tt_plugin.async_decode import SubmittedStepContext, TTAsyncDecodeController
from vllm_tt_plugin.config import get_tt_keep_decode_chain
from vllm_tt_plugin.model_input import TTSamplingParams
from vllm_tt_plugin.platform import TTPlatform

_KEY = "keep_decode_chain"


def _config(**tt_keys):
    return SimpleNamespace(additional_config={"tt": dict(tt_keys)} if tt_keys else {})


# --- Configuration ---------------------------------------------------------


def test_keep_decode_chain_is_off_by_default():
    assert get_tt_keep_decode_chain(_config()) is False


@pytest.mark.parametrize("value", [True, False])
def test_keep_decode_chain_reads_a_boolean(value):
    assert get_tt_keep_decode_chain(_config(**{_KEY: value})) is value


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_non_boolean_keep_decode_chain_values_raise(value):
    with pytest.raises(ValueError, match=_KEY):
        get_tt_keep_decode_chain(_config(**{_KEY: value}))


# --- Startup validation ----------------------------------------------------


def _model_class(*, supports_async_decode=True):
    return type(
        "DummyModel",
        (),
        {
            "__module__": "models.tt_transformers.tt.generator_vllm",
            "model_capabilities": {
                "supports_async_decode": supports_async_decode,
                "supports_sample_on_device": True,
            },
        },
    )


def _check_and_update_config(
    monkeypatch,
    vllm_config,
    *,
    tt_keys,
    lanes=2,
    async_scheduling=True,
    supports_async_decode=True,
):
    """Run the platform hook on the conftest config in lane mode by default."""
    vllm_config.additional_config = {"tt": dict(tt_keys)}
    if lanes > 1:
        vllm_config.additional_config["_tt_resolved_lane_count"] = lanes
    vllm_config.scheduler_config.async_scheduling = async_scheduling
    model_class = _model_class(supports_async_decode=supports_async_decode)
    monkeypatch.setattr(
        "vllm_tt_plugin.platform.register_tt_models", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        "vllm_tt_plugin.platform._resolve_standard_dp_visible_device_groups",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        "vllm.model_executor.models.registry.ModelRegistry.get_supported_archs",
        lambda: ["TTDummyModel"],
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.utils.get_model_architecture",
        lambda _model_config: (model_class, None),
    )
    TTPlatform.check_and_update_config(vllm_config)


_ON = {_KEY: True, "sample_on_device_mode": "all"}


def test_keep_decode_chain_starts_with_what_it_needs(monkeypatch, vllm_config):
    _check_and_update_config(monkeypatch, vllm_config, tt_keys=_ON)

    assert get_tt_keep_decode_chain(vllm_config) is True


@pytest.mark.parametrize("tt_keys", [{}, {_KEY: False}], ids=["absent", "false"])
def test_keep_decode_chain_off_adds_no_startup_requirement(
    monkeypatch, vllm_config, tt_keys
):
    # Synchronous, single engine, host sampling: none of it is refused when the
    # key is off.
    _check_and_update_config(
        monkeypatch, vllm_config, tt_keys=tt_keys, lanes=1, async_scheduling=False
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"async_scheduling": False}, "async scheduling"),
        # The model gate turns async scheduling off for a model that does not
        # declare supports_async_decode; the key must not survive that quietly.
        ({"supports_async_decode": False}, "async scheduling"),
        ({"lanes": 1}, "lane mode"),
    ],
    ids=["no-async", "async-disabled-by-model", "not-lane-mode"],
)
def test_keep_decode_chain_without_a_requirement_fails_at_startup(
    monkeypatch, vllm_config, overrides, message
):
    with pytest.raises(ValueError, match=rf"{_KEY}.*{message}"):
        _check_and_update_config(monkeypatch, vllm_config, tt_keys=_ON, **overrides)


def test_keep_decode_chain_without_device_sampling_fails_at_startup(
    monkeypatch, vllm_config
):
    with pytest.raises(ValueError, match=rf"{_KEY}.*sample_on_device_mode"):
        _check_and_update_config(monkeypatch, vllm_config, tt_keys={_KEY: True})


# --- Behaviour -------------------------------------------------------------


class _Model:
    """A version-1 adapter whose decode returns a device output read async."""

    decode_input_update_contract = 1
    model_capabilities = {"supports_async_decode": True}

    def decode_forward(self, **kwargs):
        return torch.zeros((2, 1))

    def read_decode_output(self, tt_out, async_read=False):
        return (tt_out, []) if async_read else tt_out


def _runner():
    """Two lane slots: A in row 0, B in row 1, both decoding."""
    return SimpleNamespace(
        model=_Model(),
        trace_mode="decode_only",
        kv_caches=object(),
        request_specific_rope=False,
        supports_stage_outputs=False,
        async_decode_scheduling=True,
        _num_speculative_tokens=0,
        _decode_layout_changed_since_last_decode=False,
        cache_config=SimpleNamespace(block_size=16),
        max_num_blocks_per_req=16,
        model_config=SimpleNamespace(max_model_len=256),
        lane_batch=SimpleNamespace(
            req_id_to_index={"A": 0, "B": 1}, num_tokens=np.array([20, 30])
        ),
        _steady_decode_lock=threading.Lock(),
        _pending_async_steps=deque(),
        _pending_async_overlap_ok=deque(),
        _completed_decode_steps=deque(),
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
    )


def _sampling_params(rows=2):
    return TTSamplingParams(
        temperature=torch.ones(rows),
        top_k=torch.full((rows,), 32),
        top_p=torch.ones(rows),
        presence_penalty=torch.zeros(rows),
        frequency_penalty=torch.zeros(rows),
        repetition_penalty=torch.ones(rows),
        seed=torch.full((rows,), -1),
        num_logprobs=torch.full((rows,), -2),
        enable_log_probs=torch.zeros(rows, dtype=torch.bool),
    )


def _decode_input(*, layout_changed):
    """A plain device-sampled lane decode over slots 0 and 1."""
    page_table = torch.zeros((2, 1), dtype=torch.int32)
    return SimpleNamespace(
        input_tokens=torch.zeros((2, 1), dtype=torch.int32),
        input_positions=torch.zeros((2,), dtype=torch.int32),
        block_tables=page_table,
        block_tables_per_group=[page_table],
        block_tables_per_layer=None,
        unpadded_batch_size=2,
        tt_sampling_params=_sampling_params(),
        perform_device_sampling=True,
        defer_device_sampling=False,
        prompt_lens=None,
        prompt_tokens=None,
        output_tokens=None,
        allowed_token_ids_mask_list=[None],
        bad_words_token_ids_list=[[]],
        max_num_logprobs=[None],
        decode_layout_changed=layout_changed,
        num_valid_drafts=None,
        accepted_counts=None,
        spec_mode=None,
        slot_remap=None,
    )


def _controller(keep_decode_chain):
    runner = _runner()
    controller = TTAsyncDecodeController(runner, keep_decode_chain=keep_decode_chain)
    # A device-sampled decode is resident: the chain a release could keep.
    model_input = _decode_input(layout_changed=False)
    controller.commit_decode_submission(
        model_input, controller.plan_decode_reload(model_input)
    )
    return controller, runner


def _b_finishes():
    """The lane step after B finished: A keeps decoding, B's slot is freed."""
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
        preempted_req_ids=set(),
        finished_req_ids={"B"},
    )
    plan = SimpleNamespace(
        is_decode=True, decode_layout_changed=True, scheduled_req_ids=["A"]
    )
    return scheduler_output, plan


@pytest.mark.parametrize("keep", [False, True])
def test_a_finished_only_release_keeps_the_resident_chain_only_with_the_key(keep):
    controller, _ = _controller(keep)
    scheduler_output, plan = _b_finishes()

    assert controller.exp_release_keeps_chain(scheduler_output, plan) is keep
    # A kept release, and only a kept one, suppresses B's late output from a
    # step submitted before the release; without the key the step drains.
    assert ("B" in controller._exp_released_req_ids) is keep


@pytest.mark.parametrize("keep", [False, True])
def test_a_reloading_decode_is_overlap_ok_only_with_the_key(keep):
    controller, runner = _controller(keep)
    context = SubmittedStepContext(
        req_ids=["A", "B"], req_id_to_index={"A": 0, "B": 1}, submit_time_ns=1
    )

    controller.submit_async_lane_decode(
        _decode_input(layout_changed=True), context, scheduled_rows=[0, 1]
    )

    # The step reloaded its inputs (layout changed), so it is not steady; the
    # key alone lets the next steady step be submitted over it.
    assert list(runner._pending_async_overlap_ok) == [keep]
