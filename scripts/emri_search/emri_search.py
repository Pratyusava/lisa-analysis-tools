#!/usr/bin/env python
"""Run an EMRI search campaign described by one TOML file.

The file says what is searched (``[inputs]``: injection files, data files, the noise
background), with which statistic (``[statistic] name = "fresnel" | "H"``) and basis
(``[basis] name = "svd" | "chebyshev"``, from a file or built from FEW tracks), and where
the results go (``out``). ``configs/S_svd.toml`` is Speri et al.'s search,
``configs/H_chebyshev.toml`` Cutler & Bustamante-Rosell's, each listing only the tables it uses;
``configs/reference.toml`` has every setting.

    python emri_search.py configs/S_svd.toml                     # search, background, collect
    python emri_search.py configs/S_svd.toml search              # some stages only
    python emri_search.py configs/S_svd.toml --set statistic.name=H --set out=runs/H_svd
    python emri_search.py configs/S_svd.toml --show              # the resolved configuration

Stages:

- ``search``: the injections and data files.
- ``background``: the noise streams the false-alarm rates come from.
- ``collect``: ``summary.csv``, with each stream's detection statistic ``Lambda`` (the maximum
  over durations of its statistic standardised by the background), its FAR, the detections
  at ``p_fa`` and the localisation.

A directory accepts only one configuration, and reruns skip the searches whose result file
exists, so an interrupted campaign continues where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sys

from lisatools.search.emri import RunConfig
from lisatools.search.emri.campaign import STAGES


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("config", help="campaign file (.toml)")
    parser.add_argument(
        "stages",
        nargs="*",
        metavar="stage",
        help=f"stages to run, of {', '.join(STAGES)} (default: all)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a setting, e.g. --set statistic.name=H (repeatable)",
    )
    parser.add_argument("--show", action="store_true", help="print the configuration and exit")
    args = parser.parse_args(argv)

    run = RunConfig.from_toml(args.config, overrides=args.set)
    if args.show:
        print(json.dumps(run.to_dict(), indent=1))
        return 0
    run.run(args.stages or STAGES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
