from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import cheby1, detrend, iirnotch, sosfiltfilt, tf2sos

from .config import SSVEPConfig

FloatArray = NDArray[np.float64]


def _safe_sosfiltfilt(sos: FloatArray, data: FloatArray) -> FloatArray:
    # A bounded pad length makes behavior deterministic for short, valid windows.
    estimated_default = 3 * (2 * sos.shape[0] + 1)
    padlen = min(data.shape[-1] - 1, estimated_default)
    if padlen < 1:
        return data.copy()
    return np.asarray(sosfiltfilt(sos, data, axis=-1, padlen=padlen), dtype=np.float64)


def detrend_and_notch(data: FloatArray, config: SSVEPConfig) -> FloatArray:
    return Preprocessor(config).apply(data)


class Preprocessor:
    def __init__(self, config: SSVEPConfig) -> None:
        self._notch: FloatArray | None = None
        if config.notch_hz is not None:
            numerator, denominator = iirnotch(
                config.notch_hz,
                config.notch_quality_factor,
                fs=config.sampling_rate,
            )
            self._notch = np.asarray(tf2sos(numerator, denominator), dtype=np.float64)

    def apply(self, data: FloatArray) -> FloatArray:
        processed = np.asarray(detrend(data, axis=-1, type="linear"), dtype=np.float64)
        if self._notch is None:
            return processed
        return _safe_sosfiltfilt(self._notch, processed)


def design_bandpass(
    low_hz: float,
    high_hz: float,
    config: SSVEPConfig,
) -> FloatArray:
    return np.asarray(
        cheby1(
            config.filter_order,
            config.filter_ripple_db,
            (low_hz, high_hz),
            btype="bandpass",
            fs=config.sampling_rate,
            output="sos",
        ),
        dtype=np.float64,
    )


class FilterBank:
    """Precomputed, second-order-section M3-style filter bank."""

    def __init__(self, config: SSVEPConfig) -> None:
        self._filters = tuple(
            design_bandpass(low, config.effective_filter_high_hz, config)
            for low in config.effective_filterbank_low_hz
        )
        indices = np.arange(1, len(self._filters) + 1, dtype=np.float64)
        weights = indices ** (-config.filter_weight_exponent)
        weights += config.filter_weight_offset
        self.weights = weights / weights.sum()

    def apply(self, data: FloatArray) -> tuple[FloatArray, ...]:
        return tuple(_safe_sosfiltfilt(sos, data) for sos in self._filters)


class CCAFilter:
    """Single preprocessing band for standard CCA."""

    def __init__(self, config: SSVEPConfig) -> None:
        low_hz, high_hz = config.effective_cca_band_hz
        self._sos = design_bandpass(low_hz, high_hz, config)

    def apply(self, data: FloatArray) -> FloatArray:
        return _safe_sosfiltfilt(self._sos, data)
