from __future__ import annotations

import pytest

from async_ssvep import SSVEPConfig


def test_config_rejects_harmonics_above_nyquist() -> None:
    with pytest.raises(ValueError, match="harmonic"):
        SSVEPConfig(
            sampling_rate=100.0,
            frequencies=(10.0, 20.0),
            harmonics=3,
        )


def test_low_sampling_rate_omits_invalid_filter_bands() -> None:
    config = SSVEPConfig(
        sampling_rate=128.0,
        frequencies=(8.0, 10.0, 12.0),
        harmonics=3,
    )
    assert config.effective_filter_high_hz < config.nyquist_hz
    assert all(low < config.effective_filter_high_hz for low in config.effective_filterbank_low_hz)
