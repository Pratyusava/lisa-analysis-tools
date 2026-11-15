#!/usr/bin/env python
"""Make the injection files a campaign searches.

Draws sources from Speri et al.'s Table 1 prior, each plunging at the end of one of the
search's durations and starting inside ``--f0-range`` (default the paper's 0.5-10 mHz),
generates the FEW signal with LAT's TDI response at the requested SNRs, and writes one
recipe per injection (:class:`lisatools.search.emri.InjectionFile`, a few kB of JSON): the
source parameters, the seed of the noise it is searched on, and checksums of both, which the
search checks when it regenerates them:

    python make_injections.py configs/S_svd.toml --out inj --n 100 --snrs 20 --T-data 1.0

The campaign file supplies only the cadence, channels, PSD, durations and basis grid (to
align the plunges with the search's), never its statistic or basis settings; point its
``inputs.injections`` at ``<out>/*.json``. FEW is required here and wherever the injections are
searched. Injection ``k`` is searched on noise seed ``--noise-seed + k``: keep these seeds
apart from the background's (``make_noise.py --seed``), which the search checks.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from lisatools.search.emri import FEWInjector, InjectionFile, RunConfig, draw_sources
from lisatools.search.emri.injection import F_TDI_ZERO_GUARD, PAPER_F0_RANGE


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", help="campaign file (.toml)")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--n", type=int, required=True, help="sources to draw")
    parser.add_argument("--snrs", type=float, nargs="+", default=[20.0], help="SNR per source")
    parser.add_argument("--T-data", type=float, default=1.0, help="stream length [yr]")
    parser.add_argument(
        "--T-snr", type=float, default=None, help="SNR duration [yr] (default T-data)"
    )
    parser.add_argument(
        "--f0-range",
        type=float,
        nargs=2,
        default=list(PAPER_F0_RANGE),
        metavar=("F_MIN", "F_MAX"),
        help="starting frequencies of the kept sources [Hz] (default: the paper's)",
    )
    parser.add_argument("--source-seed", type=int, default=7, help="seed of the source draws")
    parser.add_argument("--angle-seed", type=int, default=11, help="seed of the angles")
    parser.add_argument(
        "--noise-seed", type=int, default=1000, help="injection k is searched on seed + k"
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a setting of the campaign file (repeatable)",
    )
    args = parser.parse_args(argv)

    run = RunConfig.from_toml(args.config, overrides=args.set)
    config = run.search
    config.prepare_basis()
    os.makedirs(args.out, exist_ok=True)
    T_snr = args.T_snr if args.T_snr is not None else args.T_data
    t0 = time.perf_counter()
    sources = draw_sources(
        config, args.n, args.source_seed, args.T_data, f0_range=tuple(args.f0_range)
    )
    print(f"[{time.perf_counter() - t0:6.1f} s] {len(sources)} sources drawn")
    psd = config.psd()
    injector = FEWInjector(T_snr, dt=config.data.dt, channels=config.data.channels)
    # LAT's analytic PSD has TDI transfer zeros near 0.03 Hz that would dominate the SNR sum
    snr_band = (0.0, F_TDI_ZERO_GUARD) if config.data.psd_file is None else (0.0, None)
    k = 0
    for i, src in enumerate(sources):
        for snr in args.snrs:
            inj = injector(
                src,
                snr,
                psd,
                T_data=args.T_data,
                rng=np.random.default_rng(args.angle_seed + i),
                snr_band=snr_band,
            )
            label = f"inj_{i:03d}_snr{snr:g}"
            recipe = InjectionFile.make(config, inj, injector, label, args.noise_seed + k)
            recipe.save(os.path.join(args.out, label + ".json"))
            k += 1
            print(
                f"[{time.perf_counter() - t0:6.1f} s] {label}: m1 = {src[0]:.3g}, "
                f"m2 = {src[1]:.1f}, a = {src[2]:.2f}, T_pl = {src[3]:.3f} yr, e_f = {src[4]:.3f}"
            )
    print(f"{k} injections in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
