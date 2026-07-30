"""Numerically stable canonical-correlation primitives."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def harmonic_reference(
    frequency: float,
    n_samples: int,
    sampling_rate: float,
    harmonics: int,
) -> FloatArray:
    """Create the standard sine/cosine harmonic reference matrix.

    Returns an array shaped ``[2 * harmonics, samples]``.
    """

    time = np.arange(n_samples, dtype=np.float64) / sampling_rate
    rows = []
    for harmonic in range(1, harmonics + 1):
        phase = 2.0 * np.pi * harmonic * frequency * time
        rows.extend((np.sin(phase), np.cos(phase)))
    return np.asarray(rows, dtype=np.float64)


def _orthonormal_time_basis(matrix: FloatArray, tolerance: float | None) -> FloatArray:
    observations = np.asarray(matrix, dtype=np.float64).T
    observations = observations - observations.mean(axis=0, keepdims=True)
    left, singular_values, _ = np.linalg.svd(observations, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0:
        return np.empty((observations.shape[0], 0), dtype=np.float64)

    if tolerance is None:
        threshold = np.finfo(np.float64).eps * max(observations.shape) * float(singular_values[0])
    else:
        threshold = tolerance * float(singular_values[0])
    rank = int(np.count_nonzero(singular_values > threshold))
    return left[:, :rank]


def largest_canonical_correlation(
    first: FloatArray,
    second: FloatArray,
    *,
    tolerance: float | None = None,
) -> float:
    """Return the largest canonical correlation between two variable sets.

    A compact SVD subspace formulation avoids explicit covariance inversion and
    remains stable for correlated EEG channels or rank-deficient references.
    Both inputs are shaped ``[variables, samples]``.
    """

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("CCA inputs must both be two-dimensional")
    if first.shape[1] != second.shape[1]:
        raise ValueError("CCA inputs must have the same number of samples")

    first_basis = _orthonormal_time_basis(first, tolerance)
    second_basis = _orthonormal_time_basis(second, tolerance)
    if first_basis.shape[1] == 0 or second_basis.shape[1] == 0:
        return 0.0

    singular_values = np.linalg.svd(
        first_basis.T @ second_basis,
        compute_uv=False,
    )
    if singular_values.size == 0:
        return 0.0
    return float(np.clip(singular_values[0], 0.0, 1.0))
