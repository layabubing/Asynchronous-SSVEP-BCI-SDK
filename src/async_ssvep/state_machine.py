"""Deterministic asynchronous control/no-control state machine."""

from __future__ import annotations

from .config import SSVEPConfig
from .types import (
    AsyncDecision,
    DecisionEvent,
    DecisionState,
    DetectionResult,
)


class AsynchronousGate:
    """Convert window candidates into edge-triggered, debounced commands."""

    def __init__(self, config: SSVEPConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._state = DecisionState.IDLE
        self._candidate: float | None = None
        self._candidate_count = 0
        self._active: float | None = None
        self._release_count = 0
        self._refractory_remaining = 0

    def _decision(
        self,
        detection: DetectionResult,
        *,
        sample_end: int,
        time_seconds: float,
        event: DecisionEvent = DecisionEvent.NONE,
        command_frequency: float | None = None,
    ) -> AsyncDecision:
        return AsyncDecision(
            state=self._state,
            event=event,
            command_frequency=command_frequency,
            active_frequency=self._active,
            candidate_frequency=detection.candidate_frequency,
            sample_end=sample_end,
            time_seconds=time_seconds,
            detection=detection,
        )

    def update(
        self,
        detection: DetectionResult,
        *,
        sample_end: int,
        time_seconds: float,
    ) -> AsyncDecision:
        observed = detection.candidate_frequency

        if self._state is DecisionState.REFRACTORY:
            self._refractory_remaining -= 1
            if self._refractory_remaining <= 0:
                self._state = DecisionState.IDLE
            return self._decision(
                detection,
                sample_end=sample_end,
                time_seconds=time_seconds,
            )

        if self._state is DecisionState.ACTIVE:
            if observed == self._active:
                self._release_count = 0
                return self._decision(
                    detection,
                    sample_end=sample_end,
                    time_seconds=time_seconds,
                )

            self._release_count += 1
            if self._release_count < self.config.release_windows:
                return self._decision(
                    detection,
                    sample_end=sample_end,
                    time_seconds=time_seconds,
                )

            self._active = None
            self._release_count = 0
            self._candidate = None
            self._candidate_count = 0
            self._refractory_remaining = self.config.refractory_windows
            self._state = (
                DecisionState.REFRACTORY if self._refractory_remaining > 0 else DecisionState.IDLE
            )
            return self._decision(
                detection,
                sample_end=sample_end,
                time_seconds=time_seconds,
                event=DecisionEvent.RELEASED,
            )

        if observed is None:
            self._state = DecisionState.IDLE
            self._candidate = None
            self._candidate_count = 0
            return self._decision(
                detection,
                sample_end=sample_end,
                time_seconds=time_seconds,
            )

        if observed == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = observed
            self._candidate_count = 1

        if self._candidate_count < self.config.dwell_windows:
            self._state = DecisionState.CANDIDATE
            return self._decision(
                detection,
                sample_end=sample_end,
                time_seconds=time_seconds,
            )

        self._state = DecisionState.ACTIVE
        self._active = observed
        self._candidate = None
        self._candidate_count = 0
        return self._decision(
            detection,
            sample_end=sample_end,
            time_seconds=time_seconds,
            event=DecisionEvent.ACTIVATED,
            command_frequency=observed,
        )
