"""How much of a source population each track basis can represent, with no search.

For every source drawn from the paper's prior (Table 1, plunging at the end of a scan duration,
as make_injections.py draws them) and every basis variant, the basis mismatch
(lisatools.search.emri.mismatch.basis_mismatch): the fraction of the true m = 2 track's
semi-coherent power on the noiseless signal that the best track inside the basis box misses.
A basis cannot find what it cannot represent, so this bounds every search run on it.

    python basis_coverage.py configs/S_svd.toml --n 100 --out ../runs/coverage

The variants (VARIANTS below) share the data settings of the campaign file, and the built
ones share one FEW bank ([basis.bank], drawn once). Writes <out>/coverage.csv (one row per
source, appended as each finishes, so a rerun resumes from it), <out>/coverage.png (the
distribution per variant) and prints a summary.
"""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from lisatools.search.emri import FEWInjector, RunConfig, draw_sources
from lisatools.search.emri.injection import F_TDI_ZERO_GUARD
from lisatools.search.emri.mismatch import basis_mismatch, few_harmonics

# name -> overrides of the campaign file's [basis]; "file" variants load, the others are built
WIDE_BAND = "[5e-4, 2e-2]"
VARIANTS = {
    "svd_lorenzo": [  # Lorenzo's basis file (EMRI-Search), as S_svd.toml
        "basis.name=svd",
        "basis.source=file",
        "basis.file=$EMRI_SEARCH_DIR/examples/new_svd_basis_Tsft50000.0_cut20.npz",
    ],
    "svd_bank": ["basis.name=svd", "basis.source=build"],  # the same recipe, on the shared bank
    "cheb_draft": [  # the draft: 3-8 mHz, 5 coefficients, tracks spanning 95 % of the band
        "basis.name=chebyshev",
        "basis.source=build",
    ],
    "cheb_wide5": [
        "basis.name=chebyshev",
        "basis.source=build",
        f"basis.chebyshev.band={WIDE_BAND}",
        "basis.chebyshev.min_span=0.5",
    ],
    "cheb_wide7": [
        "basis.name=chebyshev",
        "basis.source=build",
        f"basis.chebyshev.band={WIDE_BAND}",
        "basis.chebyshev.n_coeffs=7",
        "basis.chebyshev.min_span=0.5",
    ],
}
THRESHOLDS = (0.1, 0.3, 0.5)  # the summary counts sources with m_basis below each


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="campaign file (.toml): the data settings")
    parser.add_argument("--out", required=True, help="output directory (built bases go here)")
    parser.add_argument("--n", type=int, default=100, help="sources")
    parser.add_argument("--T-data", type=float, default=1.0, help="stream length [yr]")
    parser.add_argument("--source-seed", type=int, default=7, help="seed of the source draws")
    parser.add_argument("--angle-seed", type=int, default=11, help="seed of the angles")
    parser.add_argument("--bank-tracks", type=int, default=None, help="default: [basis.bank]")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument(
        "--n-workers", type=int, default=1, help="processes over sources (~10 GB each)"
    )
    args = parser.parse_args(argv)

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    t0 = time.perf_counter()

    def log(message):
        print(f"[{time.perf_counter() - t0:7.1f} s] {message}", flush=True)

    configs = {name: variant_config(args, name, out) for name in args.variants}
    build_bases(configs, log, args.n_workers)

    reference = next(iter(configs.values()))
    sources = draw_sources(reference, args.n, args.source_seed, args.T_data)
    log(f"{len(sources)} sources drawn")
    csv_path = os.path.join(out, "coverage.csv")
    rows = read_csv(csv_path, list(configs))  # a resumed run keeps the sources already done
    done = {int(row["index"]) for row in rows}
    if done:
        log(f"resuming: {len(done)} sources already in {csv_path}")
    jobs = [
        (i, source, args.angle_seed + i, args.T_data, configs)
        for i, source in enumerate(sources)
        if i not in done
    ]
    with ProcessPoolExecutor(args.n_workers) as pool:
        for row in pool.map(source_coverage, jobs):
            rows.append(row)
            append_csv(csv_path, row)
            losses = "  ".join(f"{name} {row[f'm_basis_{name}']:.2f}" for name in configs)
            band = f"{row['f_start_mHz']:.2f}-{row['f_plunge_mHz']:.2f} mHz"
            log(f"source {row['index']:3d}: f {band} | {losses}")

    rows.sort(key=lambda row: int(row["index"]))
    summarise(rows, list(configs), log)
    plot(os.path.join(out, "coverage.png"), rows, list(configs))
    log(f"wrote {out}/coverage.csv and coverage.png")
    return 0


def variant_config(args, name, out):
    """The campaign file's search settings with the variant's [basis] (built ones saved in
    ``out``)."""
    overrides = [f"out={out}", *VARIANTS[name]]
    if "basis.source=build" in VARIANTS[name]:
        overrides.append(f"basis.file={os.path.join(out, name + '.npz')}")
    if args.bank_tracks is not None:
        overrides.append(f"basis.bank.n_tracks={args.bank_tracks}")
    return RunConfig.from_toml(args.config, overrides).search


def build_bases(configs, log, n_workers=1):
    """Build the variants whose file is missing, from one shared bank (integrated in
    ``n_workers`` processes)."""
    to_build = {
        name: config
        for name, config in configs.items()
        if config.basis.source == "build" and not os.path.exists(config.basis.file)
    }
    if to_build:
        bank = next(iter(to_build.values())).track_bank(log, n_workers=n_workers)
        for name, config in to_build.items():
            log(f"{name}:")
            config.build_basis(log=lambda message: log("  " + message), bank=bank)
    for config in configs.values():
        config.prepare_basis(log=log)


def source_coverage(job):
    """One row: the source, its track's range, and every variant's basis mismatch."""
    index, source, angle_seed, T_data, configs = job
    config = next(iter(configs.values()))
    psd = config.psd()
    injector = FEWInjector(T_data, dt=config.data.dt, channels=config.data.channels)
    snr_band = (0.0, F_TDI_ZERO_GUARD) if config.data.psd_file is None else (0.0, None)
    injection = injector(
        source, 20.0, psd, T_data=T_data, rng=np.random.default_rng(angle_seed), snr_band=snr_band
    )
    n_segments = injection.signal.shape[-1] // int(round(config.data.T_sft / config.data.dt))
    durations = config.durations_of(n_segments)
    times = np.arange(max(durations)) * config.data.T_sft
    track = few_harmonics(injection.params, times, (2,))[2]
    on = track.f > 0.0

    row = dict(index=index)
    row.update(
        {name: float(value) for name, value in zip(("m1", "m2", "a", "T_pl", "e_f"), source)}
    )
    row["f_start_mHz"] = float(track.f[on][0] * 1e3)
    row["f_plunge_mHz"] = float(track.f[on][-1] * 1e3)
    for name, variant in configs.items():
        m_basis, in_box = basis_mismatch(variant, injection.signal, track, durations)
        row[f"m_basis_{name}"] = m_basis
        row[f"in_box_{name}"] = float(in_box)
    return row


def read_csv(path, names):
    """The rows of an earlier run, if it covered the same variants (else start over)."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(fh)]
    if rows and not all(f"m_basis_{name}" in rows[0] for name in names):
        raise SystemExit(f"{path} holds other variants: move it away or pick another --out")
    return rows


def append_csv(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def summarise(rows, names, log):
    f_plunge = np.array([row["f_plunge_mHz"] for row in rows])
    bins = [(0.0, 5.0), (5.0, 10.0), (10.0, np.inf)]
    header = "variant        median  " + "  ".join(f"<{t:g}" for t in THRESHOLDS) + "  unfit"
    header += "  | median by plunge frequency: " + ", ".join(
        f"{lo:g}-{hi:g} mHz (n={np.sum((f_plunge >= lo) & (f_plunge < hi))})" for lo, hi in bins
    )
    log(f"{len(rows)} sources")
    log(header)
    for name in names:
        m = np.array([row[f"m_basis_{name}"] for row in rows])
        fit = np.isfinite(m)
        shares = "  ".join(f"{np.mean(fit & (m < t)):4.2f}" for t in THRESHOLDS)
        by_band = ", ".join(
            (
                f"{np.median(np.where(fit, m, 1.0)[(f_plunge >= lo) & (f_plunge < hi)]):.2f}"
                if np.any((f_plunge >= lo) & (f_plunge < hi))
                else "-"
            )
            for lo, hi in bins
        )
        median = np.median(np.where(fit, m, 1.0))  # an unfit source counts as lost
        log(f"{name:12s}  {median:6.3f}  {shares}  {np.mean(~fit):5.2f}  | {by_band}")


def plot(path, rows, names):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name in names:
        m = np.array([row[f"m_basis_{name}"] for row in rows])
        m = np.sort(np.clip(np.where(np.isfinite(m), m, 1.0), 1e-4, 1.0))
        ax.step(m, np.arange(1, m.size + 1) / m.size, where="post", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("basis mismatch (unfit sources at 1)")
    ax.set_ylabel("fraction of sources at or below")
    ax.set_title(f"Basis coverage of {len(rows)} Table 1 sources")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    sys.exit(main())
