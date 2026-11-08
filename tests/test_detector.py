from __future__ import annotations

import numpy as np
import pytest

from async_ssvep import SSVEPConfig, TrainingFreeSSVEPDetector
from async_ssvep.cca import harmonic_reference, largest_canonical_correlation
from async_ssvep.filters import FilterBank, detrend_and_notch


def synthetic_ssvep(
    frequency: float,
    *,
    sampling_rate: float = 250.0,
    seconds: float = 1.0,
    channels: int = 4,
    noise: float = 0.20,
) -> np.ndarray:
    rng = np.random.default_rng(20260730)
    time = np.arange(round(sampling_rate * seconds)) / sampling_rate
    output = []
    for channel in range(channels):
        phase = 0.17 * channel
        signal = np.sin(2 * np.pi * frequency * time + phase)
        signal += 0.45 * np.sin(2 * np.pi * 2 * frequency * time + phase / 2)
        signal += 0.20 * np.sin(2 * np.pi * 3 * frequency * time - phase)
        output.append((1.0 - channel * 0.08) * signal + noise * rng.standard_normal(time.size))
    return np.asarray(output)


@pytest.mark.parametrize("algorithm", ["cca", "fbcca"])
def test_detector_finds_frequency_without_training(algorithm: str) -> None:
    config = SSVEPConfig(
        sampling_rate=250.0,
        frequencies=(10.0, 12.0, 15.0),
        algorithm=algorithm,
        dwell_windows=2,
    )
    result = TrainingFreeSSVEPDetector(config).detect(synthetic_ssvep(12.0))
    assert result.candidate_frequency == 12.0
    assert result.best_score > 0.5
    assert result.relative_margin > 0.2
    assert result.quality.valid


def test_cached_bases_match_direct_cca_at_four_channels_and_1000_hz() -> None:
    config = SSVEPConfig(
        sampling_rate=1000,
        frequencies=(6.0, 7.5, 9.0, 10.0, 12.0, 15.0, 20.0),
        notch_hz=50.0,
    )
    eeg = synthetic_ssvep(15.0, sampling_rate=1000)
    result = TrainingFreeSSVEPDetector(config).detect(eeg)
    bank = FilterBank(config)
    expected = np.zeros(len(config.frequencies))
    for weight, subband in zip(
        bank.weights, bank.apply(detrend_and_notch(eeg, config)), strict=True
    ):
        expected += weight * np.asarray(
            [
                largest_canonical_correlation(
                    subband, harmonic_reference(frequency, 1000, 1000, config.harmonics)
                )
                ** 2
                for frequency in config.frequencies
            ]
        )
    np.testing.assert_allclose(result.scores, expected, rtol=1e-12, atol=1e-12)


def test_flat_window_fails_closed() -> None:
    config = SSVEPConfig(sampling_rate=250.0, frequencies=(10.0, 12.0))
    result = TrainingFreeSSVEPDetector(config).detect(np.zeros((4, 250)))
    assert result.candidate_frequency is None
    assert not result.quality.valid
    assert result.quality.reason == "insufficient_non_flat_channels"


def test_one_non_finite_channel_is_removed() -> None:
    config = SSVEPConfig(sampling_rate=250.0, frequencies=(10.0, 12.0))
    eeg = synthetic_ssvep(10.0)
    eeg[0, 100] = np.nan
    result = TrainingFreeSSVEPDetector(config).detect(eeg)
    assert result.candidate_frequency == 10.0
    assert result.quality.valid
    assert result.quality.rejected_channels == (0,)


def test_non_finite_window_fails_closed_when_no_channel_remains() -> None:
    config = SSVEPConfig(
        sampling_rate=250.0,
        frequencies=(10.0, 12.0),
        min_usable_channels=2,
    )
    eeg = np.full((2, 250), np.nan)
    result = TrainingFreeSSVEPDetector(config).detect(eeg)
    assert result.candidate_frequency is None
    assert result.quality.reason == "insufficient_valid_channels"


def test_flat_channels_are_removed_when_enough_channels_remain() -> None:
    config = SSVEPConfig(
        sampling_rate=250.0,
        frequencies=(10.0, 12.0),
        min_usable_channels=2,
    )
    eeg = np.vstack((synthetic_ssvep(10.0, channels=2), np.zeros((2, 250))))
    result = TrainingFreeSSVEPDetector(config).detect(eeg)
    assert result.candidate_frequency == 10.0
    assert result.quality.usable_channels == (0, 1)
    assert result.quality.rejected_channels == (2, 3)


def test_structural_errors_are_explicit() -> None:
    config = SSVEPConfig(sampling_rate=250.0, frequencies=(10.0, 12.0))
    detector = TrainingFreeSSVEPDetector(config)
    with pytest.raises(ValueError, match="shape"):
        detector.detect(np.zeros(250))
    with pytest.raises(ValueError, match="expected 250"):
        detector.detect(np.zeros((4, 200)))
