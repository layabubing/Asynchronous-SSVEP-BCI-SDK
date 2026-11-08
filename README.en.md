[中文](README.md) | [English](README.en.md)

# Training-Free Asynchronous SSVEP-BCI

![Static Badge](https://img.shields.io/badge/License-Apache2.0-green)
![Static Badge](https://img.shields.io/badge/Python-3.x-blue)

A Python implementation aimed at real-time engineering integration: CCA or
FBCCA performs training-free SSVEP frequency recognition, and a separate
asynchronous state machine distinguishes control from non-control states. The
system never forces a command output in every analysis window; an `activated`
event is emitted only when a candidate frequency passes the absolute score, the
inter-class margin, and the continuous dwell criteria at the same time.

## Features

- Standard CCA: sine/cosine multi-harmonic reference signals, purely analytic
  computation, no model files.
- FBCCA: M3-style multi-subband, zero-phase Chebyshev-I SOS filtering, and the
  weight fusion from the paper.
- Asynchronous control: `idle → candidate → active → refractory` state machine.
- False-trigger protection: absolute score, first/second relative margin,
  dwell, release, and refractory period.
- Channel-level fault tolerance: rejects channels containing NaN/Inf or flat
  signals; fails closed when usable channels are insufficient, returning a
  non-control output without emitting commands.
- Numerical stability: SVD-subspace-based CCA without explicit covariance
  inversion; supports rank-deficient channels.
- Streaming interface: accepts arbitrary-length `[channels, samples]` data
  chunks with automatic sliding windows.
- Temporal integrity: checks device timestamp continuity; explicitly requires
  `reset()` after an acquisition gap.

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

On Linux/macOS, use `.venv/bin/python` instead.

## Minimal real-time interface

```python
from async_ssvep import AsyncSSVEPProcessor, DecisionEvent, SSVEPConfig

config = SSVEPConfig(
    sampling_rate=250.0,
    frequencies=(8.0, 10.0, 12.0, 15.0),
    algorithm="fbcca",
    window_seconds=1.0,
    step_seconds=0.1,
    harmonics=3,
    notch_hz=50.0,
)
processor = AsyncSSVEPProcessor(config)


# eeg_chunk comes from the acquisition card/SDK and must be [channels, samples];
# no synthetic data is used here.
def on_eeg_chunk(eeg_chunk, first_sample_time):
    for decision in processor.push(
        eeg_chunk,
        first_sample_time=first_sample_time,
    ):
        if decision.event is DecisionEvent.ACTIVATED:
            send_command(decision.command_frequency)
```

Input contract:

- `eeg_chunk`: a two-dimensional numeric array with a fixed channel count,
  shaped `[channels, samples]`.
- `first_sample_time`: optional; device or monotonic clock seconds for the
  first sample of the current chunk.
- Amplitude units are unrestricted (V, µV, ...) but must stay consistent within
  one stream.
- Call `processor.reset()` after packet loss, reconnection, channel changes, or
  timestamp discontinuities.

Output contract:

- `push()` returns `list[AsyncDecision]`; it returns an empty list when less
  than one analysis window of data is available.
- `command_frequency` is a one-shot control command only when
  `event == "activated"`.
- `state == "active"` only reflects the current locked state; do not issue
  commands repeatedly based on it.
- A different valid target activates directly after `dwell_windows` consecutive windows.
  During confirmation, `active_frequency` retains the old target and
  `candidate_frequency` identifies the pending target. Consumers must bound how long
  an old action remains valid during this transition.
- `released` follows `release_windows` windows without a valid candidate and begins
  the configurable refractory period. Consumers must stop and reset on invalid signal quality.
- `decision.as_dict()` converts directly to JSON, message-queue, or logging
  structures.

For offline external data, see
[`examples/process_npy.py`](examples/process_npy.py); for an acquisition-callback
adapter, see
[`examples/acquisition_callback.py`](examples/acquisition_callback.py).

## Configuration guidance

| Parameter | Default | Purpose |
|---|---:|---|
| `window_seconds` | 1.0 s | Longer is usually more stable, but adds latency |
| `step_seconds` | 0.1 s | Sliding-window update period |
| `score_threshold` | 0.18 | Absolute threshold on normalized `rho²` |
| `margin_threshold` | 0.10 | Minimum margin of the first over the second target |
| `dwell_windows` | 3 | Consecutive consistent windows required to activate |
| `release_windows` | 2 | Consecutive inconsistent windows required to release |
| `refractory_seconds` | 0.5 s | Non-responsive period after release |

The FBCCA sub-band fusion weights are normalized, so the output range stays
`[0, 1]`; normalization is a constant shared by all targets and does not change
FBCCA target ordering. The default sub-band low cutoffs are
`(6, 14, 22, 30, 38) Hz` with a high cutoff of `90 Hz`; at low sampling rates
these are automatically limited to below Nyquist and invalid high sub-bands are
removed.

Validate on site in the following order rather than lowering thresholds
directly:

1. Confirm that the stimulator's actual refresh rate, the sampling rate, and
   `frequencies` match.
2. Prefer occipital channels (for example O1/Oz/O2, POz) and check contact
   quality and mains interference.
3. Check idle false triggers first; raising `score_threshold`,
   `margin_threshold`, or `dwell_windows` reduces them.
4. Then check control misses; extend the window when necessary and only lower
   thresholds slightly as a last resort.
5. In real tasks, record per-minute false triggers, activation latency, release
   latency, and classification accuracy.

## Testing

```bash
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m ruff check .
```

The deterministic synthetic signals used in tests only verify the algorithms;
they are not part of the production input interface.

## Algorithm sources

1. Lin, Z. et al. "Frequency Recognition Based on Canonical Correlation
   Analysis for SSVEP-Based BCIs." *IEEE Transactions on Biomedical
   Engineering* 53(12), 2610–2614 (2006).
   [DOI: 10.1109/TBME.2006.886577](https://doi.org/10.1109/TBME.2006.886577)
2. Chen, X. et al. "Filter bank canonical correlation analysis for implementing
   a high-speed SSVEP-based brain-computer interface." *Journal of Neural
   Engineering* 12(4), 046008 (2015).
   [DOI: 10.1088/1741-2560/12/4/046008](https://doi.org/10.1088/1741-2560/12/4/046008)
3. Diez, P. F. et al. "Asynchronous BCI control using high-frequency SSVEP."
   *Journal of NeuroEngineering and Rehabilitation* 8, 39 (2011). This work
   discusses asynchronous thresholds, sustained EEG decision criteria, and
   refractory time.
   [DOI: 10.1186/1743-0003-8-39](https://doi.org/10.1186/1743-0003-8-39)
4. Kim, H. et al. "Comparison of recognition methods for an asynchronous
   (un-cued) BCI system: an investigation with 40-class SSVEP dataset."
   *Biomedical Engineering Letters* 14, 617–630 (2024).
   [DOI: 10.1007/s13534-024-00357-4](https://doi.org/10.1007/s13534-024-00357-4)
5. Wang, J. et al. "An Asynchronous Training-Free SSVEP-BCI Detection Algorithm
   for Non-Equal Prior Probability Scenarios." *IEEE Transactions on Neural
   Systems and Rehabilitation Engineering* 32, 4120–4130 (2024).
   [DOI: 10.1109/TNSRE.2024.3496727](https://doi.org/10.1109/TNSRE.2024.3496727)

This implementation directly adopts the training-free frequency-detection ideas
of items 1 and 2; the asynchronous layer follows the non-control state,
threshold, and temporal-constraint principles emphasized in items 3–5. It does
not claim to reproduce the full experimental protocols of items 4 and 5.
