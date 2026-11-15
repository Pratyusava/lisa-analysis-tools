"""The results of a campaign: where each search is saved, and how detections are decided.

- :class:`ResultStore`: the campaign directory. One ``.npz`` per stream and duration, saved
  as each search finishes; a directory holds the results of one configuration only.
- :class:`Background`: the noise-only searches. They give each duration's mean and spread,
  which put every duration on one scale (Speri et al., Sec. III.1).
- :func:`collect`: the summary table. Each stream's detection statistic, its false-alarm
  rate against the background, the detections at the chosen false-alarm probabilities, and
  how well the best track follows the injected one.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import SearchConfig
from .pipeline import SearchResult

SOURCE_COLUMNS = ("m1", "m2", "a", "T_pl", "e_f")
MIN_STREAMS_TO_STANDARDISE = 2  # a standard deviation needs two values
MIN_STREAMS_FOR_GUMBEL = 3


@dataclass
class StreamInfo:
    """What a result records about the stream it searched (small: it travels with each task).

    Attributes:
        label: Name of the stream; its result files start with it.
        kind: ``"injection"``, ``"noise"`` or ``"data"``.
        truth: The injected source's parameters, for the localisation check.
        noise_seed: Seed of the stream's noise realisation (``None`` for a data file).
        snr: The injection's target SNR (0 for noise and data).
    """

    label: str
    kind: str = "injection"
    truth: Optional[dict] = None
    noise_seed: Optional[int] = None
    snr: float = 0.0


# ----------------------------------------------------------------------------- the directory
class ResultStore:
    """A campaign directory: ``config.json`` (the search's configuration), ``fingerprint``
    and ``results/<label>_n<segments>.npz`` per task.

    Opening a directory that holds another configuration's results raises, so two
    campaigns never mix. Reopened without a configuration (``collect``) it takes the saved
    one and its fingerprint, so the basis file need not be at hand.
    """

    def __init__(self, directory, config: Optional[SearchConfig] = None):
        self.directory = str(directory)
        self.results_dir = os.path.join(self.directory, "results")
        config_path = os.path.join(self.directory, "config.json")
        fingerprint_path = os.path.join(self.directory, "fingerprint")

        if os.path.exists(config_path):
            with open(fingerprint_path) as fh:
                saved_fingerprint = fh.read().strip()
            if config is None:
                config = SearchConfig.load_json(config_path)
            elif config.fingerprint() != saved_fingerprint:
                raise ValueError(
                    f"{self.directory} holds results of another configuration "
                    f"({saved_fingerprint} vs {config.fingerprint()}); use a new directory."
                )
            self.fingerprint = saved_fingerprint
        elif config is None:
            raise ValueError(f"{config_path} does not exist and no config was given.")
        else:
            os.makedirs(self.directory, exist_ok=True)
            self.fingerprint = config.fingerprint()
            config.save_json(config_path)
            with open(fingerprint_path, "w") as fh:
                fh.write(self.fingerprint + "\n")

        os.makedirs(self.results_dir, exist_ok=True)
        self.config = config

    def path(self, label, n_seg) -> str:
        return os.path.join(self.results_dir, f"{label}_n{int(n_seg):04d}.npz")

    def done(self, label, n_seg, noise_seed=None) -> bool:
        """Whether the task's result exists.

        With ``noise_seed``, raise if the saved result was searched on another noise
        realisation: the stream's file was remade under the same name.
        """
        path = self.path(label, n_seg)
        if not os.path.exists(path):
            return False
        if noise_seed is not None:
            with np.load(path) as saved:
                saved_seed = int(saved["noise_seed"])
            if saved_seed != int(noise_seed):
                raise ValueError(
                    f"{path} was searched on noise seed {saved_seed}, but {label} now has "
                    f"{noise_seed}: its file was remade; use a new campaign directory."
                )
        return True

    def save(self, info: StreamInfo, n_seg, result: SearchResult):
        """Write one search's result (to a temporary file first, so a crash leaves no
        half-written result that a rerun would take as done)."""
        path = self.path(info.label, n_seg)
        tmp = path + ".tmp.npz"
        history = result.optimization.history
        np.savez(
            tmp,
            label=info.label,
            kind=info.kind,
            n_seg=int(n_seg),
            T_pl=float(result.T_pl),
            shift=int(result.shift),
            best_statistic=float(result.best_statistic),
            best_x=np.asarray(result.best_x),
            best_f=np.asarray(result.best_track.f),  # type: ignore[attr-defined]
            iterations=int(history.get("last_step", self.config.optimizer.n_steps)),
            truth=json.dumps(info.truth or {}),
            noise_seed=-1 if info.noise_seed is None else int(info.noise_seed),
            snr=float(info.snr),
            fingerprint=self.fingerprint,
        )
        os.replace(tmp, path)

    def load(self) -> List[dict]:
        """Every saved result as a dict (arrays as NumPy, ``truth`` as a dict)."""
        records = []
        for path in sorted(glob.glob(os.path.join(self.results_dir, "*.npz"))):
            with np.load(path) as saved:
                record = {
                    key: (value.item() if value.ndim == 0 else value)
                    for key, value in saved.items()
                }
            if record["fingerprint"] != self.fingerprint:
                warnings.warn(f"{path} was made with another configuration; skipped.", stacklevel=2)
                continue
            record["truth"] = json.loads(str(record["truth"]))
            records.append(record)
        return records


# ----------------------------------------------------------------------------- the background
def false_alarm_rate(value, background) -> float:
    """Fraction of ``background`` at or above ``value`` (0 means below ``1 / len(background)``)."""
    background = np.asarray(background, dtype=float)
    return float(np.mean(background >= value)) if background.size else float("nan")


class Background:
    """The noise-only searches: each duration's mean and spread, and the detection statistic.

    As Speri et al. (Sec. III.1; the notebook's Sec. 9). ``X_k`` is the best statistic found
    at duration ``k``. Raw values cannot be compared across durations, because they grow with
    the stream's length, so each is standardised by the noise-only searches at that duration::

        Z_k = (X_k - mu_k) / sigma_k        (sigma_k with ddof = 1)
        Lambda = max_k Z_k                  (a stream's detection statistic)

    The noise streams' own ``Lambda``, standardised the same way, give the false-alarm
    rates: counted (:func:`false_alarm_rate`), and from a Gumbel fit, which is the paper's
    way to reach rates below one over the number of noise streams.

    Args:
        statistics: ``{label: {n_seg: X}}`` of the noise streams. Only the durations every
            noise stream was searched at are used.
    """

    def __init__(self, statistics: Dict[str, Dict[int, float]]):
        self.labels = sorted(statistics)
        self.durations = self._common_durations(statistics)
        # X[i, k]: noise stream i at duration k
        self.X = np.array(
            [[statistics[label][n_seg] for n_seg in self.durations] for label in self.labels]
        ).reshape(len(self.labels), len(self.durations))
        if self.n_streams >= MIN_STREAMS_TO_STANDARDISE:
            self.mu = self.X.mean(axis=0)
            self.sigma = self.X.std(axis=0, ddof=1)
        else:
            self.mu = np.full(len(self.durations), np.nan)
            self.sigma = np.full(len(self.durations), np.nan)
        self._column = {n_seg: k for k, n_seg in enumerate(self.durations)}
        self._noise_lambda_cache: Dict[Tuple[int, ...], Tuple[np.ndarray, float, float]] = {}

    @staticmethod
    def _common_durations(statistics) -> List[int]:
        if not statistics:
            return []
        searched = [set(per_duration) for per_duration in statistics.values()]
        return sorted(set.intersection(*searched))

    @property
    def n_streams(self) -> int:
        return len(self.labels)

    def standardise(self, statistics: Dict[int, float]) -> Dict[int, float]:
        """``{n_seg: Z}`` of a stream's ``{n_seg: X}``, for the durations the background covers."""
        standardised = {}
        for n_seg, x in statistics.items():
            if n_seg in self._column:
                k = self._column[n_seg]
                standardised[n_seg] = (x - self.mu[k]) / self.sigma[k]
        return standardised

    def noise_lambda(self, durations) -> Tuple[np.ndarray, float, float]:
        """The noise streams' ``Lambda`` over ``durations``, and its Gumbel ``(location, scale)``
        (``NaN`` with fewer than three streams)."""
        key = tuple(sorted(durations))
        if key not in self._noise_lambda_cache:
            self._noise_lambda_cache[key] = self._fit_noise_lambda(key)
        return self._noise_lambda_cache[key]

    def _fit_noise_lambda(self, durations) -> Tuple[np.ndarray, float, float]:
        if not durations or self.n_streams == 0:
            return np.zeros(0), float("nan"), float("nan")
        columns = [self._column[n_seg] for n_seg in durations]
        Z = (self.X[:, columns] - self.mu[columns]) / self.sigma[columns]
        noise_lambda = Z.max(axis=1)
        location = scale = float("nan")
        if self.n_streams >= MIN_STREAMS_FOR_GUMBEL and np.all(np.isfinite(noise_lambda)):
            from scipy import stats

            location, scale = (float(v) for v in stats.gumbel_r.fit(noise_lambda))
        return noise_lambda, location, scale


# ----------------------------------------------------------------------------- the summary
def collect(store: ResultStore, tol_bins=2.0, p_fa=(0.5, 1e-2, 1e-4), log=print) -> Dict[str, Any]:
    """Summarise a campaign directory (the detection statistic: :class:`Background`).

    One row per injection or data stream, with the columns:

    =================  ===========================================================
    ``statistic``      ``Lambda``, the detection statistic
    ``duration``       the duration [segments] ``Lambda`` peaks at
    ``raw``            the unstandardised statistic ``X`` at that duration
    ``far``            fraction of noise streams with ``Lambda`` at least as large,
                       over the same durations
    ``far_gumbel``     the same from the Gumbel fit to the noise ``Lambda``
    ``detected_<p>``   ``Lambda`` above the Gumbel threshold at each false-alarm
                       probability in ``p_fa`` (paper: 0.5, 1e-2, 1e-4)
    ``agreement``      with FEW: fraction of the injected track's in-band segments
                       the best track follows within ``tol_bins`` bins
    =================  ===========================================================

    Durations the background does not cover are left out of ``Lambda``.

    Returns the rows and the background, and writes ``summary.csv`` and ``summary.npz``.
    """
    results_by_stream = _group_by_stream(store.load())
    noise_streams = {
        label: results for label, results in results_by_stream.items() if _kind(results) == "noise"
    }
    background = Background(
        {label: _statistics(results) for label, results in noise_streams.items()}
    )

    rows = []
    for label, results in results_by_stream.items():
        if label in noise_streams:
            continue
        row = _summary_row(label, results, background, store.config, tol_bins, p_fa)
        if row is None:
            warnings.warn(
                f"{label}: no duration could be standardised (the background has "
                f"{background.n_streams} streams over {len(background.durations)} durations); "
                "left out of the summary.",
                stacklevel=2,
            )
        else:
            rows.append(row)

    noise_lambda, location, scale = background.noise_lambda(background.durations)
    _write_summary(store.directory, rows, background, noise_lambda, (location, scale))
    _log_summary(log, rows, background, (location, scale), p_fa)
    return dict(
        rows=rows, background=background, noise_lambda=noise_lambda, gumbel=(location, scale)
    )


def _group_by_stream(records) -> Dict[str, Dict[int, dict]]:
    """``{label: {n_seg: record}}``."""
    grouped: Dict[str, Dict[int, dict]] = {}
    for record in records:
        grouped.setdefault(record["label"], {})[int(record["n_seg"])] = record
    return grouped


def _statistics(results: Dict[int, dict]) -> Dict[int, float]:
    """``{n_seg: X}`` of one stream's results."""
    return {n_seg: record["best_statistic"] for n_seg, record in results.items()}


def _kind(results: Dict[int, dict]) -> str:
    return str(next(iter(results.values()))["kind"])


def _summary_row(label, results, background: Background, config, tol_bins, p_fa):
    """The summary row of one stream, or ``None`` when none of its durations standardises."""
    from scipy import stats

    Z = background.standardise(_statistics(results))
    if not Z or not np.all(np.isfinite(list(Z.values()))):
        return None
    best_duration = max(Z, key=lambda n_seg: Z[n_seg])
    best = results[best_duration]
    detection_statistic = Z[best_duration]
    noise_lambda, location, scale = background.noise_lambda(Z)

    row = dict(
        label=label,
        kind=_kind(results),
        snr=best["snr"],
        statistic=detection_statistic,
        raw=best["best_statistic"],
        duration=best_duration,
        n_durations=len(Z),
        far=false_alarm_rate(detection_statistic, noise_lambda),
        far_gumbel=float(stats.gumbel_r.sf(detection_statistic, location, scale)),
        n_background=background.n_streams,
        agreement=_agreement(config, best, tol_bins),
    )
    for p in p_fa:
        threshold = stats.gumbel_r.isf(p, location, scale)
        row[f"detected_{p:g}"] = bool(detection_statistic > threshold)
    for name in SOURCE_COLUMNS:
        row[name] = best["truth"].get(name, float("nan"))
    return row


def _agreement(config: SearchConfig, record, tol_bins) -> float:
    """Fraction of the injected track's in-band segments the best track follows within
    ``tol_bins`` bins; ``NaN`` without FEW or without an injected source."""
    truth = record["truth"]
    if not truth or "m1" not in truth:
        return float("nan")
    n_seg = int(record["n_seg"])
    try:
        from .tracks import FEWHarmonicTrackModel

        segment_times = np.arange(n_seg) * config.data.T_sft
        source = (
            truth["m1"],
            truth["m2"],
            truth["a"],
            truth["T_pl"],
            truth["e_f"],
            truth.get("x_f", 1.0),
        )
        true_f = np.asarray(FEWHarmonicTrackModel(segment_times, m=2, n=0)(source).f)
    except Exception as exc:  # pragma: no cover - FEW missing or a failed integration
        warnings.warn(f"no true track for {record['label']}: {exc}", stacklevel=2)
        return float("nan")

    best_f = np.asarray(record["best_f"])[:n_seg]
    f_min = config.statistic.fresnel.f_min or 0.0  # the in-band cut, whatever the statistic
    in_band = (true_f > f_min) & np.isfinite(best_f)
    if not in_band.any():
        return 0.0
    bin_width = 1.0 / config.data.T_sft
    followed = np.abs(best_f[in_band] - true_f[in_band]) < tol_bins * bin_width
    return float(np.mean(followed))


def _write_summary(directory, rows, background: Background, noise_lambda, gumbel):
    arrays: Dict[str, Any] = dict(
        background_labels=np.array(background.labels),
        background_durations=np.array(background.durations),
        background_X=background.X,
        background_mu=background.mu,
        background_sigma=background.sigma,
        noise_lambda=noise_lambda,
        gumbel=np.array(gumbel),
    )
    if rows:
        columns = list(rows[0])
        with open(os.path.join(directory, "summary.csv"), "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        for column in columns:
            arrays[column] = np.array([row[column] for row in rows])
    np.savez(os.path.join(directory, "summary.npz"), **arrays)


def _log_summary(log, rows, background: Background, gumbel, p_fa: Sequence[float]):
    from scipy import stats

    location, scale = gumbel
    log(
        f"{len(rows)} streams; background: {background.n_streams} noise streams over "
        f"{len(background.durations)} durations, Gumbel fit to their Lambda: location "
        f"{location:.2f}, scale {scale:.2f} (paper: 1.94, 0.56)"
    )
    for p in p_fa:
        threshold = stats.gumbel_r.isf(p, location, scale)
        log(f"  threshold at p_FA = {p:g}: Lambda > {threshold:.2f}")
    for row in rows:
        far = f"{row['far']:.3f}" if np.isfinite(row["far"]) else "n/a"
        agreement = f"{row['agreement']:.2f}" if np.isfinite(row["agreement"]) else "n/a"
        log(
            f"  {row['label']:>24s}  {row['kind']:>9s}  Lambda {row['statistic']:7.2f} at n = "
            f"{row['duration']:4d}  FAR {far} (Gumbel {row['far_gumbel']:.1e})  "
            f"agreement {agreement}"
        )
