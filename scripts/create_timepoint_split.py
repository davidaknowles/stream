#!/usr/bin/env python
"""Create an alternate held-out-timepoint split without rebuilding model inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stream_model.data import adjacent_intervals, ordered_days


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heldout-day", action="append", required=True)
    args = parser.parse_args()

    with Path(args.base).open() as handle:
        split = json.load(handle)
    days = ordered_days(split["all_days"])
    heldout = set(args.heldout_day)
    missing = heldout.difference(days)
    if missing:
        raise ValueError(f"Held-out stages are absent from the base split: {sorted(missing)}")
    split["heldout_days"] = [day for day in days if day in heldout]
    split["train_intervals"] = adjacent_intervals(days, heldout)
    split["heldout_touching_intervals"] = [
        (a, b) for a, b in zip(days[:-1], days[1:], strict=True) if a in heldout or b in heldout
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        json.dump(split, handle, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
