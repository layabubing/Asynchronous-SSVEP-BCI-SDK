"""Validated configuration for the SSVEP detector and asynchronous gate."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Literal


@dataclass(frozen=True, slots=True)
class SSVEPConfig:
    """Complete, immutable configuration for one acquisition stream.

    EEG arrays use ``[channels, samples]`` throughout the package.
    """

    sampling_rate: float
    frequencies: tuple[float, ...]
    algorithm: Literal["cca", "fbcca"] = "fbcca"
    window_seconds: float = 1.0
    step_seconds: float = 0.1
    harmonics: int = 3

    # Common preprocessing.
    notch_hz: float | None = None
    notch_quality_factor: float = 30.0
    cca_band_hz: tuple[float, float] = (6.0, 90.0)

    # Chen et al. M3-inspired filter bank. Invalid high sub-bands are safely omitted
    # when the acquisition Nyquist frequency is lower than 90 Hz.
    filterbank_low_hz: tuple[float, ...] = (6.0, 14.0, 22.0, 30.0, 38.0)
    filterbank_high_hz: float = 90.0
    filter_order: int = 4
    filter_ripple_db: float = 1.0
    filter_weight_exponent: float = 1.25
    filter_weight_offset: float = 0.25

    # Fail-closed quality gate.
    min_usable_channels: int = 1
    flat_channel_std: float = 1e-10
    rank_tolerance: float | None = None

    # Training-free asynchronous decision gate.
    score_threshold: float = 0.18
    margin_threshold: float = 0.10
    dwell_windows: int = 3
    release_windows: int = 2
    refractory_seconds: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequencies", tuple(float(x) for x in self.frequencies))
        object.__setattr__(self, "cca_band_hz", tuple(float(x) for x in self.cca_band_hz))
        object.__setattr__(
            self, "filterbank_low_hz", tuple(float(x) for x in self.filterbank_low_hz)
        )

        if not isfinite(self.sampling_rate) or self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be finite and > 0")
        if not self.frequencies:
            raise ValueError("frequencies must contain at least one target frequency")
        if any(not isfinite(f) or f <= 0 for f in self.frequencies):
            raise ValueError("all target frequencies must be finite and > 0")
        if len(set(self.frequencies)) != len(self.frequencies):
            raise ValueError("target frequencies must be unique")
        if self.algorithm not in {"cca", "fbcca"}:
            raise ValueError("algorithm must be 'cca' or 'fbcca'")
        if self.harmonics < 1:
            raise ValueError("harmonics must be >= 1")

        nyquist = self.sampling_rate / 2.0
        if max(self.frequencies) * self.harmonics >= nyquist:
            raise ValueError(
                "highest target harmonic must be below Nyquist; reduce harmonics "
                "or increase sampling_rate"
            )

        if not isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("window_seconds must be finite and > 0")
        if not isfinite(self.step_seconds) or self.step_seconds <= 0:
            raise ValueError("step_seconds must be finite and > 0")
        if self.step_seconds > self.window_seconds:
            raise ValueError("step_seconds must not exceed window_seconds")
        if self.window_samples < 16:
            raise ValueError("analysis window must contain at least 16 samples")
        if self.step_samples < 1:
            raise ValueError("step_seconds is too short for the sampling rate")

        if self.notch_hz is not None:
            if not isfinite(self.notch_hz) or not 0 < self.notch_hz < nyquist:
                raise ValueError("notch_hz must be between 0 and Nyquist")
        if not isfinite(self.notch_quality_factor) or self.notch_quality_factor <= 0:
            raise ValueError("notch_quality_factor must be finite and > 0")

        if len(self.cca_band_hz) != 2:
            raise ValueError("cca_band_hz must contain (low_hz, high_hz)")
        cca_low, cca_high = self.effective_cca_band_hz
        if not 0 < cca_low < cca_high:
            raise ValueError("effective CCA band must satisfy 0 < low < high")

        if not self.filterbank_low_hz:
            raise ValueError("filterbank_low_hz must not be empty")
        if any(not isfinite(x) or x <= 0 for x in self.filterbank_low_hz):
            raise ValueError("filter-bank low cutoffs must be finite and > 0")
        if tuple(sorted(set(self.filterbank_low_hz))) != self.filterbank_low_hz:
            raise ValueError("filterbank_low_hz must be unique and strictly increasing")
        if not isfinite(self.filterbank_high_hz) or self.filterbank_high_hz <= 0:
            raise ValueError("filterbank_high_hz must be finite and > 0")
        if not self.effective_filterbank_low_hz:
            raise ValueError("no filter-bank sub-band fits below Nyquist")
        if self.filter_order < 1:
            raise ValueError("filter_order must be >= 1")
        if not isfinite(self.filter_ripple_db) or self.filter_ripple_db <= 0:
            raise ValueError("filter_ripple_db must be finite and > 0")
        if not isfinite(self.filter_weight_exponent) or self.filter_weight_exponent <= 0:
            raise ValueError("filter_weight_exponent must be finite and > 0")
        if not isfinite(self.filter_weight_offset) or self.filter_weight_offset < 0:
            raise ValueError("filter_weight_offset must be finite and >= 0")

        if self.min_usable_channels < 1:
            raise ValueError("min_usable_channels must be >= 1")
        if not isfinite(self.flat_channel_std) or self.flat_channel_std < 0:
            raise ValueError("flat_channel_std must be finite and >= 0")
        if self.rank_tolerance is not None and (
            not isfinite(self.rank_tolerance) or self.rank_tolerance <= 0
        ):
            raise ValueError("rank_tolerance must be finite and > 0 when provided")

        if not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be in [0, 1]")
        if not 0 <= self.margin_threshold <= 1:
            raise ValueError("margin_threshold must be in [0, 1]")
        if self.dwell_windows < 1:
            raise ValueError("dwell_windows must be >= 1")
        if self.release_windows < 1:
            raise ValueError("release_windows must be >= 1")
        if not isfinite(self.refractory_seconds) or self.refractory_seconds < 0:
            raise ValueError("refractory_seconds must be finite and >= 0")

    @property
    def nyquist_hz(self) -> float:
        return self.sampling_rate / 2.0

    @property
    def window_samples(self) -> int:
        return int(round(self.window_seconds * self.sampling_rate))

    @property
    def step_samples(self) -> int:
        return int(round(self.step_seconds * self.sampling_rate))

    @property
    def refractory_windows(self) -> int:
        if self.refractory_seconds == 0:
            return 0
        return ceil(self.refractory_seconds / self.step_seconds)

    @property
    def effective_filter_high_hz(self) -> float:
        return min(self.filterbank_high_hz, self.nyquist_hz * 0.95)

    @property
    def effective_filterbank_low_hz(self) -> tuple[float, ...]:
        high = self.effective_filter_high_hz
        return tuple(low for low in self.filterbank_low_hz if low < high - 1.0)

    @property
    def effective_cca_band_hz(self) -> tuple[float, float]:
        low, high = self.cca_band_hz
        return low, min(high, self.nyquist_hz * 0.95)
