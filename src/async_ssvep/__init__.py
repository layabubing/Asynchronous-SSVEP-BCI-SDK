"""Training-free asynchronous SSVEP detection."""

from .config import SSVEPConfig
from .detector import TrainingFreeSSVEPDetector
from .stream import AsyncSSVEPProcessor
from .types import (
    AsyncDecision,
    DecisionEvent,
    DecisionState,
    DetectionResult,
    SignalQuality,
)

__all__ = [
    "AsyncDecision",
    "AsyncSSVEPProcessor",
    "DecisionEvent",
    "DecisionState",
    "DetectionResult",
    "SSVEPConfig",
    "SignalQuality",
    "TrainingFreeSSVEPDetector",
]

__version__ = "0.1.0"
