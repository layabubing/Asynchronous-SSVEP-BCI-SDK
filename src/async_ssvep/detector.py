from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .cca import (
    canonical_correlation_from_bases,
    harmonic_reference,
    orthonormal_time_basis,
)
from .config import SSVEPConfig
from .filters import CCAFilter, FilterBank, Preprocessor
from .types import DetectionResult, SignalQuality

FloatArray = NDArray[np.float64]


class TrainingFreeSSVEPDetector:
    """Stateless, calibration-free SSVEP detector for fixed-size windows."""

    def __init__(self, config: SSVEPConfig) -> None:
        self.config = config
        self._preprocessor = Preprocessor(config)
        self._references = tuple(
            harmonic_reference(
                frequency,
                config.window_samples,
                config.sampling_rate,
                config.harmonics,
            )
            for frequency in config.frequencies
        )
        self._reference_bases = tuple(
            orthonormal_time_basis(reference, config.rank_tolerance)
            for reference in self._references
        )
        self._filterbank = FilterBank(config) if config.algorithm == "fbcca" else None
        self._cca_filter = CCAFilter(config) if config.algorithm == "cca" else None

    def _scores(self, filtered: FloatArray) -> FloatArray:
        basis = orthonormal_time_basis(filtered, self.config.rank_tolerance)
        return np.asarray(
            [
                canonical_correlation_from_bases(basis, reference) ** 2
                for reference in self._reference_bases
            ],
            dtype=np.float64,
        )

    def _quality(self, eeg: FloatArray) -> SignalQuality:
        channel_count = eeg.shape[0]
        finite_mask = np.isfinite(eeg).all(axis=1)
        channel_std = np.zeros(channel_count, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            channel_std[finite_mask] = np.std(eeg[finite_mask], axis=1)
        usable_mask = (
            finite_mask & np.isfinite(channel_std) & (channel_std > self.config.flat_channel_std)
        )
        usable = tuple(int(index) for index in np.flatnonzero(usable_mask))
        rejected = tuple(int(index) for index in np.flatnonzero(~usable_mask))
        if len(usable) < self.config.min_usable_channels:
            reason = (
                "insufficient_valid_channels"
                if not finite_mask.all()
                else "insufficient_non_flat_channels"
            )
            return SignalQuality(
                valid=False,
                reason=reason,
                usable_channels=usable,
                rejected_channels=rejected,
            )
        return SignalQuality(
            valid=True,
            reason=None,
            usable_channels=usable,
            rejected_channels=rejected,
        )

    def _invalid_result(self, quality: SignalQuality) -> DetectionResult:
        zeros = tuple(0.0 for _ in self.config.frequencies)
        return DetectionResult(
            frequencies=self.config.frequencies,
            scores=zeros,
            candidate_frequency=None,
            best_score=0.0,
            second_score=0.0,
            relative_margin=0.0,
            quality=quality,
        )

    def detect(self, eeg_window: ArrayLike) -> DetectionResult:
        """Score one EEG window shaped ``[channels, samples]``.

        Signal-quality failures return a no-control result. Structural API errors
        (wrong rank, zero channels, or wrong window length) raise ``ValueError``.
        """

        eeg = np.asarray(eeg_window, dtype=np.float64)
        if eeg.ndim != 2:
            raise ValueError("eeg_window must have shape [channels, samples]")
        if eeg.shape[0] == 0:
            raise ValueError("eeg_window must contain at least one channel")
        if eeg.shape[1] != self.config.window_samples:
            raise ValueError(f"expected {self.config.window_samples} samples, got {eeg.shape[1]}")

        quality = self._quality(eeg)
        if not quality.valid:
            return self._invalid_result(quality)

        usable_eeg = eeg[np.asarray(quality.usable_channels)]
        preprocessed = self._preprocessor.apply(usable_eeg)

        if self.config.algorithm == "cca":
            assert self._cca_filter is not None
            filtered = self._cca_filter.apply(preprocessed)
            scores = self._scores(filtered)
        else:
            assert self._filterbank is not None
            subbands = self._filterbank.apply(preprocessed)
            scores = np.zeros(len(self._references), dtype=np.float64)
            for weight, subband in zip(
                self._filterbank.weights,
                subbands,
                strict=True,
            ):
                scores += weight * self._scores(subband)

        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        if scores.size > 1:
            second_score = float(np.partition(scores, -2)[-2])
        else:
            second_score = 0.0
        relative_margin = float(
            np.clip(
                (best_score - second_score) / max(best_score, np.finfo(float).eps),
                0.0,
                1.0,
            )
        )
        accepted = (
            best_score >= self.config.score_threshold
            and relative_margin >= self.config.margin_threshold
        )
        candidate = self.config.frequencies[best_index] if accepted else None

        return DetectionResult(
            frequencies=self.config.frequencies,
            scores=tuple(float(score) for score in scores),
            candidate_frequency=candidate,
            best_score=best_score,
            second_score=second_score,
            relative_margin=relative_margin,
            quality=quality,
        )
