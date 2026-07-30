"""Minimal adapter for a real EEG SDK callback."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from async_ssvep import AsyncSSVEPProcessor, DecisionEvent, SSVEPConfig

FloatArray = NDArray[np.float64]


class BCIController:
    """Connect an acquisition callback to an application command callback."""

    def __init__(self, on_command: Callable[[float], None]) -> None:
        self._on_command = on_command
        self._processor = AsyncSSVEPProcessor(
            SSVEPConfig(
                sampling_rate=250.0,
                frequencies=(8.0, 10.0, 12.0, 15.0),
                algorithm="fbcca",
                notch_hz=50.0,
            )
        )

    def on_eeg_chunk(
        self,
        eeg_channels_by_samples: FloatArray,
        first_sample_time: float,
    ) -> None:
        """Call this method directly from the acquisition SDK."""

        for decision in self._processor.push(
            eeg_channels_by_samples,
            first_sample_time=first_sample_time,
        ):
            if decision.event is DecisionEvent.ACTIVATED:
                assert decision.command_frequency is not None
                self._on_command(decision.command_frequency)

    def on_acquisition_gap(self) -> None:
        """Reset after a device reconnect, packet loss, or timestamp discontinuity."""

        self._processor.reset()
