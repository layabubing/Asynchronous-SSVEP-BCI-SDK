"""Public result types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class DecisionState(str, Enum):
    """Current asynchronous control state."""

    IDLE = "idle"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    REFRACTORY = "refractory"


class DecisionEvent(str, Enum):
    """Edge-triggered event emitted by the asynchronous gate."""

    NONE = "none"
    ACTIVATED = "activated"
    RELEASED = "released"


@dataclass(frozen=True, slots=True)
class SignalQuality:
    """Fail-closed signal-quality assessment for one analysis window."""

    valid: bool
    reason: str | None
    usable_channels: tuple[int, ...]
    rejected_channels: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "usable_channels": list(self.usable_channels),
            "rejected_channels": list(self.rejected_channels),
        }


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Stateless CCA/FBCCA result for one window."""

    frequencies: tuple[float, ...]
    scores: tuple[float, ...]
    candidate_frequency: float | None
    best_score: float
    second_score: float
    relative_margin: float
    quality: SignalQuality

    @property
    def score_by_frequency(self) -> dict[float, float]:
        return dict(zip(self.frequencies, self.scores, strict=True))

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_frequency": self.candidate_frequency,
            "best_score": self.best_score,
            "second_score": self.second_score,
            "relative_margin": self.relative_margin,
            "scores": {
                str(frequency): score
                for frequency, score in zip(self.frequencies, self.scores, strict=True)
            },
            "quality": self.quality.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class AsyncDecision:
    """One timestamped state-machine output.

    ``command_frequency`` is populated only on an ``activated`` event. Consumers
    should act on that edge-triggered field rather than repeatedly acting while
    ``state == active``.
    """

    state: DecisionState
    event: DecisionEvent
    command_frequency: float | None
    active_frequency: float | None
    candidate_frequency: float | None
    sample_end: int
    time_seconds: float
    detection: DetectionResult

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "event": self.event.value,
            "command_frequency": self.command_frequency,
            "active_frequency": self.active_frequency,
            "candidate_frequency": self.candidate_frequency,
            "sample_end": self.sample_end,
            "time_seconds": self.time_seconds,
            "detection": self.detection.as_dict(),
        }
