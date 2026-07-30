"""Process externally acquired EEG stored in a NumPy file.

The input must be a two-dimensional ``.npy`` array shaped [channels, samples].
This example intentionally does not synthesize data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from async_ssvep import AsyncSSVEPProcessor, DecisionEvent, SSVEPConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="path to [channels, samples] .npy EEG")
    parser.add_argument("--sampling-rate", type=float, required=True)
    parser.add_argument("--frequencies", type=float, nargs="+", required=True)
    parser.add_argument("--algorithm", choices=("cca", "fbcca"), default="fbcca")
    parser.add_argument("--chunk-samples", type=int, default=25)
    parser.add_argument("--notch-hz", type=float)
    parser.add_argument(
        "--all-windows",
        action="store_true",
        help="print every state update instead of activated/released events only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eeg = np.load(args.input, allow_pickle=False)
    if eeg.ndim != 2:
        raise ValueError("input must have shape [channels, samples]")
    if args.chunk_samples < 1:
        raise ValueError("--chunk-samples must be >= 1")

    config = SSVEPConfig(
        sampling_rate=args.sampling_rate,
        frequencies=tuple(args.frequencies),
        algorithm=args.algorithm,
        notch_hz=args.notch_hz,
    )
    processor = AsyncSSVEPProcessor(config)

    for start in range(0, eeg.shape[1], args.chunk_samples):
        chunk = eeg[:, start : start + args.chunk_samples]
        for decision in processor.push(chunk):
            if args.all_windows or decision.event is not DecisionEvent.NONE:
                print(json.dumps(decision.as_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
