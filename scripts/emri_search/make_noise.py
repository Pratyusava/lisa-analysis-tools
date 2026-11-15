#!/usr/bin/env python
"""Make the noise files of a campaign's background.

Writes ``--n`` noise-only streams of ``--T-data`` years, each a recipe
(:class:`lisatools.search.emri.NoiseFile`, a few kB of JSON): the seed, the length, the
cadence, channels and PSD of the campaign file, and a checksum of the noise, which the search
checks when it redraws it:

    python make_noise.py configs/S_svd.toml --out noise --n 100 --T-data 1.0

Stream ``r`` is drawn with seed ``--seed + r``; keep these apart from the injections' noise
seeds (``make_injections.py --noise-seed``), which the search checks. Point the campaign
file's ``inputs.noise`` at ``<out>/*.json``. Use the same PSD and length as the injections.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from lisatools.search.emri import NoiseFile, RunConfig


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", help="campaign file (.toml)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--n", type=int, required=True, help="noise streams")
    parser.add_argument("--T-data", type=float, default=1.0, help="stream length [yr]")
    parser.add_argument("--seed", type=int, default=100_000, help="stream r is drawn with seed + r")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a setting of the campaign file (repeatable)",
    )
    args = parser.parse_args(argv)

    config = RunConfig.from_toml(args.config, overrides=args.set).search
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    for r in range(args.n):
        label = f"noise_{r:04d}"
        NoiseFile.make(config, label, args.seed + r, args.T_data).save(
            os.path.join(args.out, label + ".json")
        )
    print(f"[{time.perf_counter() - t0:6.1f} s] {args.n} noise files in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
