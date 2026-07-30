from __future__ import annotations

import numpy as np
import pytest

from async_ssvep.cca import harmonic_reference, largest_canonical_correlation


def test_harmonic_reference_shape_and_range() -> None:
    reference = harmonic_reference(12.0, 250, 250.0, 3)
    assert reference.shape == (6, 250)
    assert np.max(np.abs(reference)) <= 1.0


def test_canonical_correlation_is_one_for_shared_component() -> None:
    reference = harmonic_reference(10.0, 250, 250.0, 2)
    eeg = np.vstack((2.0 * reference[0] + reference[1], reference[2]))
    correlation = largest_canonical_correlation(eeg, reference)
    assert correlation == pytest.approx(1.0, abs=1e-10)


def test_rank_deficient_input_is_handled() -> None:
    time = np.arange(250) / 250.0
    signal = np.sin(2 * np.pi * 12.0 * time)
    eeg = np.vstack((signal, signal, signal))
    reference = harmonic_reference(12.0, 250, 250.0, 3)
    assert largest_canonical_correlation(eeg, reference) > 0.999
