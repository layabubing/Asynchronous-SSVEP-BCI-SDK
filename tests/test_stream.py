from __future__ import annotations

import numpy as np
import pytest

from async_ssvep import (
    AsyncSSVEPProcessor,
    DecisionEvent,
    DecisionState,
    SSVEPConfig,
)
from tests.test_detector import synthetic_ssvep


def _push_irregular(processor: AsyncSSVEPProcessor, eeg: np.ndarray) -> list:
    chunk_sizes = (13, 41, 7, 89, 25, 3, 61)
    outputs = []
    start = 0
    chunk_index = 0
    while start < eeg.shape[1]:
        size = chunk_sizes[chunk_index % len(chunk_sizes)]
        outputs.extend(processor.push(eeg[:, start : start + size]))
        start += size
        chunk_index += 1
    return outputs


def test_stream_emits_one_debounced_activation() -> None:
    config = SSVEPConfig(
        sampling_rate=250.0,
        frequencies=(10.0, 12.0, 15.0),
        algorithm="fbcca",
        window_seconds=1.0,
        step_seconds=0.1,
        dwell_windows=3,
        release_windows=2,
    )
    processor = AsyncSSVEPProcessor(config)
    signal = synthetic_ssvep(12.0, seconds=2.0)
    outputs = _push_irregular(processor, signal)
    activations = [item for item in outputs if item.event is DecisionEvent.ACTIVATED]
    assert len(activations) == 1
    assert activations[0].command_frequency == 12.0
    assert activations[0].state is DecisionState.ACTIVE
    assert all(
        item.command_frequency is None
        for item in outputs
        if item.event is not DecisionEvent.ACTIVATED
    )


def test_stream_releases_and_enters_refractory_state() -> None:
    config = SSVEPConfig(
        sampling_rate=250.0,
        frequencies=(10.0, 12.0),
        dwell_windows=2,
        release_windows=2,
        refractory_seconds=0.3,
    )
    processor = AsyncSSVEPProcessor(config)
    active = synthetic_ssvep(10.0, seconds=2.0)
    idle = np.zeros((active.shape[0], 2 * config.window_samples))
    outputs = _push_irregular(processor, np.concatenate((active, idle), axis=1))
    assert any(item.event is DecisionEvent.ACTIVATED for item in outputs)
    releases = [item for item in outputs if item.event is DecisionEvent.RELEASED]
    assert len(releases) == 1
    assert releases[0].state is DecisionState.REFRACTORY


def test_timestamp_continuity_is_checked() -> None:
    config = SSVEPConfig(sampling_rate=250.0, frequencies=(10.0, 12.0))
    processor = AsyncSSVEPProcessor(config)
    processor.push(np.ones((2, 25)), first_sample_time=100.0)
    processor.push(np.ones((2, 25)), first_sample_time=100.1)
    with pytest.raises(ValueError, match="timestamp discontinuity"):
        processor.push(np.ones((2, 25)), first_sample_time=101.0)


def test_channel_count_change_requires_reset() -> None:
    config = SSVEPConfig(sampling_rate=250.0, frequencies=(10.0, 12.0))
    processor = AsyncSSVEPProcessor(config)
    processor.push(np.ones((2, 25)))
    with pytest.raises(ValueError, match="channel count changed"):
        processor.push(np.ones((3, 25)))


def test_confirmed_target_changes_activate_without_releasing_control() -> None:
    config = SSVEPConfig(
        sampling_rate=250.0,
        frequencies=(7.5, 9.0, 10.0, 12.0, 15.0, 20.0),
        dwell_windows=3,
        release_windows=2,
    )
    processor = AsyncSSVEPProcessor(config)
    targets = (10.0, 7.5, 10.0, 9.0, 20.0, 9.0)
    signal = np.concatenate(
        [synthetic_ssvep(frequency, seconds=3.0) for frequency in targets], axis=1
    )
    outputs = _push_irregular(processor, signal)
    activated = [item for item in outputs if item.event is DecisionEvent.ACTIVATED]
    assert [item.command_frequency for item in activated] == list(targets)
    assert not any(item.event is DecisionEvent.RELEASED for item in outputs)
    switching = [
        item for item in outputs
        if item.active_frequency is not None
        and item.candidate_frequency is not None
        and item.active_frequency != item.candidate_frequency
    ]
    assert switching
    assert all(item.event is DecisionEvent.NONE for item in switching)
