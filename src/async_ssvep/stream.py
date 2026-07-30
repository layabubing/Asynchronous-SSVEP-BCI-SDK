"""Chunk-oriented streaming interface for acquisition systems."""

from __future__ import annotations

from math import isfinite

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import SSVEPConfig
from .detector import TrainingFreeSSVEPDetector
from .state_machine import AsynchronousGate
from .types import AsyncDecision

FloatArray = NDArray[np.float64]


class AsyncSSVEPProcessor:
    """Accept arbitrary EEG chunks and emit zero or more asynchronous decisions.

    The first dimension is always channels. ``first_sample_time`` is optional and
    may be a device/monotonic timestamp in seconds for the chunk's first sample.
    Without it, output times are relative to the beginning of the stream.
    """

    def __init__(self, config: SSVEPConfig) -> None:
        self.config = config
        self.detector = TrainingFreeSSVEPDetector(config)
        self._gate = AsynchronousGate(config)
        self.reset()

    def reset(self) -> None:
        self._buffer: FloatArray | None = None
        self._buffer_start_sample = 0
        self._total_received = 0
        self._next_window_end = self.config.window_samples
        self._channel_count: int | None = None
        self._time_origin: float | None = None
        self._gate.reset()

    def _update_clock(self, first_sample_time: float | None) -> None:
        if first_sample_time is None:
            return
        first_sample_time = float(first_sample_time)
        if not isfinite(first_sample_time):
            raise ValueError("first_sample_time must be finite")

        inferred_origin = first_sample_time - self._total_received / self.config.sampling_rate
        if self._time_origin is None:
            self._time_origin = inferred_origin
            return

        expected = self._time_origin + self._total_received / self.config.sampling_rate
        tolerance = max(2.0 / self.config.sampling_rate, 1e-6)
        if abs(first_sample_time - expected) > tolerance:
            raise ValueError(
                "timestamp discontinuity: reset the processor after an acquisition gap"
            )

    def _decision_time(self, sample_end: int) -> float:
        relative = sample_end / self.config.sampling_rate
        return relative if self._time_origin is None else self._time_origin + relative

    def push(
        self,
        eeg_chunk: ArrayLike,
        *,
        first_sample_time: float | None = None,
    ) -> list[AsyncDecision]:
        """Append ``[channels, samples]`` EEG and return newly available decisions."""

        chunk = np.asarray(eeg_chunk, dtype=np.float64)
        if chunk.ndim != 2:
            raise ValueError("eeg_chunk must have shape [channels, samples]")
        if chunk.shape[0] == 0:
            raise ValueError("eeg_chunk must contain at least one channel")
        if chunk.shape[1] == 0:
            return []
        if self._channel_count is None:
            self._channel_count = int(chunk.shape[0])
        elif chunk.shape[0] != self._channel_count:
            raise ValueError(
                f"channel count changed from {self._channel_count} to {chunk.shape[0]}"
            )

        self._update_clock(first_sample_time)
        if self._buffer is None:
            self._buffer = chunk.copy()
        else:
            self._buffer = np.concatenate((self._buffer, chunk), axis=1)
        self._total_received += int(chunk.shape[1])

        decisions: list[AsyncDecision] = []
        while self._total_received >= self._next_window_end:
            assert self._buffer is not None
            window_start = self._next_window_end - self.config.window_samples
            local_start = window_start - self._buffer_start_sample
            local_end = self._next_window_end - self._buffer_start_sample
            window = self._buffer[:, local_start:local_end]
            if window.shape[1] != self.config.window_samples:
                raise RuntimeError("internal ring-buffer invariant violated")

            detection = self.detector.detect(window)
            decisions.append(
                self._gate.update(
                    detection,
                    sample_end=self._next_window_end,
                    time_seconds=self._decision_time(self._next_window_end),
                )
            )
            self._next_window_end += self.config.step_samples

        # Keep exactly the history needed for the next overlapping window.
        earliest_needed = max(0, self._next_window_end - self.config.window_samples)
        if self._buffer is not None and earliest_needed > self._buffer_start_sample:
            drop = earliest_needed - self._buffer_start_sample
            self._buffer = self._buffer[:, drop:]
            self._buffer_start_sample = earliest_needed

        return decisions
