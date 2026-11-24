"""Track mismatch: how much of an injected signal's power a recovered track collects.

The search's output is a frequency track, and what matters downstream is how well that track
follows the signal. The ruler is the semi-coherent power a track collects from the
*noiseless* injected signal,

    P(track) = sum_j |int_j s(t) e^{-i phi(t)} dt|^2 / (S(f_j) T_j),

over coherent segments ``j`` of ``k`` SFT segments each. ``phi`` is the track's phase as the
search models it: a linear chirp ``(f_alpha, fdot_alpha)`` from each SFT segment's start,
continuous across segments, and zero where the track is zero. ``S(f_j)`` is the PSD at the
segment's mean track frequency and ``T_j`` the time the track is non-zero in it. The ruler is
the same for every statistic and basis, so the four combinations of a campaign are measured
on one scale.

The mismatch of a track is ``m = 1 - P(track) / P(true)``. ``true`` is a harmonic
``m Phi_phi`` (``m`` = 2, 3, 4) of the injected source from FEW: in TDI the m = 3 and 4
harmonics are often louder than m = 2, and a single-harmonic search may follow any of them,
so the reference is the harmonic the found track is closest to in frequency (column
``harmonic``). Every power is taken over the longest searched stream, so a track found at
too short a duration is charged for the signal it misses. The mismatch splits into the loss
the basis imposes and the loss of the search:

    1 - m_total = (1 - m_basis) (1 - m_search)

- ``m_basis``: the best track the basis can represent (:func:`best_representable`: fits of
  the true track's frequencies inside the basis box, at the scan durations bracketing the
  true plunge) against the true track.
- ``m_search``: the found track against that best representable one.

The fits minimise frequency offsets, not ``-P``, so ``m_basis`` is an upper bound on the
basis loss and ``m_search`` can be negative (the search found a better track than the
fit). Longer coherent segments (``k = 2, 4``) show whether a coherent follow-up could start
from the track. The track models have no LISA orbital Doppler, so for ``k T_sft``
approaching 10^6 s even the true track loses power; keep ``k`` small.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .tracks import ChebyshevTrackModel, SVDTrackModel, Track, _roll_and_zero

DEFAULT_COHERENCE = (1, 2, 4)  # coherent segments of k SFT segments
DEFAULT_HARMONICS = (2, 3, 4)  # the m of the true tracks, m Phi_phi


# ----------------------------------------------------------------------------- the ruler
CHUNK_SEGMENTS = 32  # SFT segments whose template is built at a time (memory, not results)


def track_phase(track: Track, n_segments, n_per_segment, dt) -> Tuple[np.ndarray, np.ndarray]:
    """The phase [rad] of ``track`` at every sample, shape ``(n_segments, n_per_segment)``, and
    which segments the track is non-zero in, shape ``(n_segments,)``.

    In segment ``alpha`` the phase is ``2 pi (f_alpha tau + fdot_alpha tau^2 / 2)`` plus the
    phase the earlier segments accumulated, ``tau`` the time since the segment's start.
    """
    f, fdot, start = _segment_phases(track, n_segments, n_per_segment * dt)
    return _phase_block(f, fdot, start, np.arange(n_per_segment) * dt), f > 0.0


def _segment_phases(track: Track, n_segments, T):
    """``f``, ``fdot`` and the phase [cycles, mod 1] at the start of each segment."""
    f = np.asarray(track.f, dtype=float)[:n_segments]
    fdot = np.asarray(track.fdot, dtype=float)[:n_segments]
    cycles_per_segment = f * T + 0.5 * fdot * T**2
    start = np.concatenate([[0.0], np.cumsum(cycles_per_segment)[:-1]]) % 1.0
    return f, fdot, start


def _phase_block(f, fdot, start, tau):
    """The phase [rad] of the segments ``f, fdot, start`` at the times ``tau`` into each."""
    return 2 * np.pi * (start[:, None] + f[:, None] * tau + 0.5 * fdot[:, None] * tau**2)


def semi_coherent_power(signal, track: Track, dt, T_sft, psd, coherence=1) -> float:
    """``P(track)`` on ``signal`` (shape ``(n_channels, N)``), over coherent segments of
    ``coherence`` SFT segments (see the module docs)."""
    return semi_coherent_powers(signal, track, dt, T_sft, psd, (coherence,))[coherence]


def semi_coherent_powers(signal, track: Track, dt, T_sft, psd, coherence) -> Dict[int, float]:
    """:func:`semi_coherent_power` for each coherence in ``coherence`` (one phase evaluation).

    The track covers the first ``len(track.f)`` segments; the signal beyond them is not
    collected. The template is built :data:`CHUNK_SEGMENTS` segments at a time, so the memory
    used does not grow with the stream.
    """
    signal = np.atleast_2d(signal)
    n_per_segment = int(round(T_sft / dt))
    n_segments = min(signal.shape[-1] // n_per_segment, np.size(track.f))
    f, fdot, start = _segment_phases(track, n_segments, n_per_segment * dt)
    valid = f > 0.0
    tau = np.arange(n_per_segment) * dt

    # the overlap of each channel with the template, per SFT segment
    overlaps = np.zeros((signal.shape[0], n_segments), dtype=complex)
    for first in range(0, n_segments, CHUNK_SEGMENTS):
        block = slice(first, min(first + CHUNK_SEGMENTS, n_segments))
        if not valid[block].any():
            continue
        phase = _phase_block(f[block], fdot[block], start[block], tau)
        template = np.exp(-1j * phase) * valid[block, None]
        samples = slice(block.start * n_per_segment, block.stop * n_per_segment)
        for channel, series in enumerate(signal):
            data = np.asarray(series[samples], dtype=float).reshape(-1, n_per_segment)
            overlaps[channel, block] = np.sum(data * template, axis=-1) * dt

    powers = {}
    for k in coherence:
        n_coherent = n_segments // int(k)
        if n_coherent == 0:
            powers[k] = 0.0
            continue
        used = n_coherent * int(k)
        c = overlaps[:, :used].reshape(-1, n_coherent, int(k)).sum(axis=-1)
        n_valid = valid[:used].reshape(n_coherent, int(k)).sum(axis=-1)
        f_sum = np.where(valid[:used], f[:used], 0.0).reshape(n_coherent, int(k)).sum(axis=-1)
        has_track = n_valid > 0
        f_mean = np.where(has_track, f_sum / np.maximum(n_valid, 1), 1.0)
        S = np.asarray(psd(f_mean), dtype=float)
        norm = np.where(has_track, S * n_valid * n_per_segment * dt, np.inf)
        powers[k] = float(np.sum(np.abs(c) ** 2 / norm[None, :]))
    return powers


# ----------------------------------------------------------------- the best representable track
def best_representable(model, true_track: Track, shift) -> Tuple[List[Track], bool]:
    """Candidate tracks of ``model`` (on the CPU) close to ``true_track`` at ``shift``, and
    whether the unbounded linear fit stayed inside the basis box. The caller keeps the
    candidate that collects the most power.

    1. A linear fit, bounded to the box because the search cannot leave it (an unbounded fit
       on a short stream sees only the end of each basis mode, and they are nearly
       degenerate there):

       - SVD: the least-squares coefficients of the true frequencies, over the samples
         where the true track and the shifted basis are non-zero.
       - Chebyshev: the least-squares coefficients of ``ln(f_mid^2 / fdot)`` (as
         :func:`~lisatools.sources.emri.chebyshevwave.chebyshev_fit`), then the anchor
         ``y_ref`` that aligns the model's ``t(f)`` with the true track's on average.

    2. That fit refined over the unit cube with a robust (Cauchy, one-bin scale) loss on the
       frequency offsets [bins]. Like the power of a segment, which is lost whether the
       track misses it by 2 bins or by 200, the loss saturates, so the refinement matches
       the segments the basis can reach instead of compromising on the ones it cannot
       (a track running out of the box near plunge).

    Returns ``([], False)`` when the true track gives nothing to fit.
    """
    if isinstance(model, SVDTrackModel):
        xi, in_box = _svd_projection(model, true_track, shift)
    elif isinstance(model, ChebyshevTrackModel):
        xi, in_box = _chebyshev_projection(model, true_track, shift)
    else:
        raise TypeError(f"no projection onto a {type(model).__name__}.")
    if xi is None:
        return [], False
    return [
        model(xi, shift),
        model(_robust_refinement(model, true_track, shift, xi), shift),
    ], in_box


def _robust_refinement(model, true_track: Track, shift, xi):
    """``xi`` refined by a Cauchy-loss fit of the frequency offsets [bins] (see
    :func:`best_representable`)."""
    from scipy.optimize import least_squares

    n = model.t_alpha.size
    f_true = np.asarray(true_track.f, dtype=float)[:n]
    on = f_true > 0.0

    def offsets(x):
        return (np.asarray(model(x, shift).f)[on] - f_true[on]) * model.T_sft

    fit = least_squares(
        offsets, xi, bounds=(0.0, 1.0), loss="cauchy", f_scale=1.0, max_nfev=200 * xi.size
    )
    return fit.x


def _bounded_fit(design, target, lb, ub) -> Tuple[np.ndarray, bool]:
    """Least-squares ``design @ x = target`` with ``lb <= x <= ub``, and whether the
    unbounded solution lies inside the bounds."""
    from scipy.optimize import lsq_linear

    free = np.linalg.lstsq(design, target, rcond=None)[0]
    if np.all((free >= lb) & (free <= ub)):
        return free, True
    return lsq_linear(design, target, bounds=(lb, ub)).x, False


def _svd_projection(model: SVDTrackModel, true_track: Track, shift):
    basis = model.basis
    n = model.t_alpha.size
    modes = np.stack([_roll_and_zero(mode, shift, np)[:n] for mode in basis.V])
    f_true = np.asarray(true_track.f, dtype=float)[:n]
    fitted = (f_true > 0.0) & np.any(modes != 0.0, axis=0)
    if fitted.sum() < basis.n_modes:
        return None, False
    coeffs, in_box = _bounded_fit(modes[:, fitted].T, f_true[fitted], basis.lb, basis.ub)
    return np.clip(model.unit(coeffs), 0.0, 1.0), in_box


def _chebyshev_projection(model: ChebyshevTrackModel, true_track: Track, shift):
    from ...sources.emri.chebyshevwave import chebyshev_u

    basis = model.basis
    n = model.t_alpha.size
    f_true = np.asarray(true_track.f, dtype=float)[:n]
    fdot_true = np.asarray(true_track.fdot, dtype=float)[:n]
    in_band = (f_true >= basis.band[0]) & (f_true <= basis.band[1]) & (fdot_true > 0.0)
    if in_band.sum() < basis.n_coeffs + 1:
        return None, False
    y = basis.y(f_true[in_band])
    target = np.log(basis.f_mid**2 / fdot_true[in_band])
    alpha, in_box = _bounded_fit(chebyshev_u(y, basis.n_coeffs).T, target, basis.lb, basis.ub)

    # the model's t(y) is T_anchor + g(y) - g(y_ref), g the time from y = 0: choose g(y_ref)
    # so that the model's times match the true track's segment times on average
    g, f_grid, _ = model.wave.track(alpha, 0.0)
    g, y_grid = np.asarray(g), basis.y(np.asarray(f_grid))
    T_anchor = model.T_ref + shift * model.T_sft
    g_ref = T_anchor + np.mean(np.interp(y, y_grid, g) - model.t_alpha[:n][in_band])
    y_ref = float(np.interp(g_ref, g, y_grid))
    in_box = in_box and model.y_range[0] <= y_ref <= model.y_range[1]
    y_ref = float(np.clip(y_ref, *model.y_range))
    return np.clip(model.unit(alpha, y_ref), 0.0, 1.0), in_box


# ----------------------------------------------------------------------------- the summary columns
def mismatch_columns(
    config,
    results: Dict[int, dict],
    duration,
    signal,
    true_tracks: Optional[Dict[int, Track]] = None,
    coherence: Sequence[int] = DEFAULT_COHERENCE,
    harmonics: Sequence[int] = DEFAULT_HARMONICS,
) -> Dict[str, float]:
    """The mismatch columns of one stream's result at ``duration`` [segments].

    Args:
        config: The campaign's :class:`~.config.SearchConfig`.
        results: The stream's :class:`~.results.ResultStore` records, ``{n_seg: record}``;
            the longest is the stream every power is taken over.
        duration: The duration whose best track is judged (the summary's peak duration).
        signal: The noiseless signal, shape ``(n_channels, N)``.
        true_tracks: ``{m: Track}`` on the longest stream's segments; default FEW's
            ``m Phi_phi`` harmonics of the recorded source, for ``m`` in ``harmonics``.
        coherence: Coherent segments [SFT segments] of the ``mismatch_x<k>`` columns.

    Returns:
        ``mismatch`` (one SFT segment coherent), ``harmonic`` (the ``m`` it is taken
        against: the true track the found one is closest to), ``mismatch_basis``,
        ``mismatch_search``, ``in_basis_box`` (whether the unbounded basis fit stayed in the
        box, 1 / 0) and ``mismatch_x<k>`` for every other ``k``. The split needs the basis
        file; without it its columns are NaN.
    """
    coherence = tuple(sorted({1, *(int(k) for k in coherence)}))
    dt, T_sft = config.data.dt, config.data.T_sft
    psd = config.psd()
    n_total = max(results)
    signal = np.atleast_2d(signal)[:, : n_total * int(round(T_sft / dt))]
    record = results[duration]
    if true_tracks is None:
        true_tracks = few_harmonics(record["truth"], np.arange(n_total) * T_sft, harmonics)

    def powers(track, ks=coherence):
        return semi_coherent_powers(signal, _pad(track, n_total), dt, T_sft, psd, ks)

    found = _pad(_found_track(config, record), n_total)
    harmonic = _followed_harmonic(found, true_tracks)
    P_found = powers(found)
    P_true = {harmonic: powers(true_tracks[harmonic])}

    columns: Dict[str, float] = {"harmonic": float(harmonic)}
    for k in coherence:
        name = "mismatch" if k == 1 else f"mismatch_x{k}"
        columns[name] = _mismatch(P_found[k], P_true[harmonic][k])

    columns["mismatch_basis"] = columns["mismatch_search"] = float("nan")
    columns["in_basis_box"] = float("nan")
    if basis_available(config):
        best = _best_over_durations(
            config, sorted(results), true_tracks[harmonic], powers, duration
        )
        if best is not None:
            P_best, in_box = best
            columns["mismatch_basis"] = _mismatch(P_best, P_true[harmonic][1])
            columns["mismatch_search"] = _mismatch(P_found[1], P_best)
            columns["in_basis_box"] = float(in_box)
    return columns


def basis_mismatch(config, signal, true_track: Track, durations) -> Tuple[float, bool]:
    """``(m_basis, in_box)`` of the campaign's basis for one source, with no search: how much
    of ``true_track``'s power on the noiseless ``signal`` the best representable track
    collects, over the scan ``durations`` bracketing the plunge (see the module docs).
    ``(NaN, False)`` when the basis cannot fit the track at all."""
    dt, T_sft = config.data.dt, config.data.T_sft
    psd = config.psd()
    n_total = max(durations)
    signal = np.atleast_2d(signal)[:, : n_total * int(round(T_sft / dt))]

    def powers(track, ks=(1,)):
        return semi_coherent_powers(signal, _pad(track, n_total), dt, T_sft, psd, ks)

    best = _best_over_durations(config, sorted(durations), true_track, powers)
    if best is None:
        return float("nan"), False
    return _mismatch(best[0], powers(true_track)[1]), best[1]


def _best_over_durations(config, durations, true_track: Track, powers, judged=None):
    """``(P, in_box)`` of the best representable track over the ``durations`` that bracket
    the true plunge (and the ``judged`` one), or ``None`` when none can be fitted."""
    T_sft = config.data.T_sft
    on = np.flatnonzero(np.asarray(true_track.f) > 0.0)
    plunge = on[-1] + 1 if on.size else max(durations)  # the first segment after the plunge
    after = [n for n in durations if n >= plunge]
    before = [n for n in durations if n < plunge]
    candidates = {*after[:1], *before[-1:]} | ({judged} if judged is not None else set())

    best = None
    for n_seg in sorted(candidates):
        model = config.track_model(np.arange(n_seg) * T_sft, force_backend="cpu")
        shift = model.shift_for_plunge(n_seg * T_sft)
        tracks, in_box = best_representable(model, true_track, shift)
        for track in tracks:
            P = powers(track, (1,))[1]
            if best is None or P > best[0]:
                best = (P, in_box)
    return best


def _followed_harmonic(found: Track, true_tracks: Dict[int, Track]) -> int:
    """The ``m`` whose true track the found one is closest to: the smallest median frequency
    offset over the segments where both are non-zero (``m = 2`` when none overlap)."""

    def offset(m):
        f_true = np.asarray(true_tracks[m].f)
        both = (found.f > 0.0) & (f_true > 0.0)
        return float(np.median(np.abs(found.f[both] - f_true[both]))) if both.any() else np.inf

    offsets = {m: offset(m) for m in true_tracks}
    best = min(offsets, key=lambda m: offsets[m])
    return best if np.isfinite(offsets[best]) else (2 if 2 in true_tracks else best)


def _mismatch(power, reference) -> float:
    return 1.0 - power / reference if reference > 0.0 else float("nan")


def _pad(track: Track, n) -> Track:
    """``track`` zero-extended to ``n`` segments."""
    f, fdot = np.asarray(track.f, dtype=float), np.asarray(track.fdot, dtype=float)
    extra = max(n - f.size, 0)
    return Track(np.pad(f, (0, extra))[:n], np.pad(fdot, (0, extra))[:n])


def few_harmonics(truth, segment_times, harmonics) -> Dict[int, Track]:
    """FEW's ``m Phi_phi`` tracks of the recorded source (one inspiral for all ``m``)."""
    from .tracks import FEWHarmonicTrackModel

    source = tuple(truth[name] for name in ("m1", "m2", "a", "T_pl", "e_f"))
    orbital = FEWHarmonicTrackModel(segment_times, m=1, n=0)(source + (truth.get("x_f", 1.0),))
    return {int(m): Track(m * orbital.f, m * orbital.fdot) for m in harmonics}


def _found_track(config, record) -> Track:
    """The best track of a result: as saved, or rebuilt from ``best_x`` (results saved
    before ``best_fdot`` was)."""
    if "best_fdot" in record:
        return Track(np.asarray(record["best_f"]), np.asarray(record["best_fdot"]))
    if not basis_available(config):
        raise FileNotFoundError("the result has no best_fdot and the basis file is missing.")
    model = config.track_model(np.arange(int(record["n_seg"])) * config.data.T_sft, "cpu")
    return model(np.asarray(record["best_x"]), int(record["shift"]))


def basis_available(config) -> bool:
    """Whether the campaign's basis file is at hand (the mismatch split needs it)."""
    return bool(config.basis.file) and os.path.exists(config.basis.file)


# ----------------------------------------------------------------------------- caching
def cached_mismatch_columns(
    directory, record, durations, compute: Callable[[], Dict[str, float]], coherence, with_basis
) -> Dict[str, float]:
    """``compute()``, cached in ``<directory>/mismatch/<label>_n<segments>.json`` (the
    signal is regenerated with FEW, so recomputing on every ``collect`` is slow).

    The cache is reused while the judged result (``record``), the stream's searched
    ``durations`` and ``coherence`` are unchanged, unless it was made without the basis and
    the basis is now at hand (``with_basis``).
    """
    name = f"{record['label']}_n{int(record['n_seg']):04d}.json"
    path = os.path.join(directory, "mismatch", name)
    key = dict(
        best_statistic=float(record["best_statistic"]),
        durations=sorted(int(n) for n in durations),
        coherence=sorted({1, *(int(k) for k in coherence)}),
    )
    if os.path.exists(path):
        with open(path) as fh:
            saved = json.load(fh)
        if saved["key"] == key and (saved["with_basis"] or not with_basis):
            return saved["columns"]
    columns = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(dict(key=key, with_basis=bool(with_basis), columns=columns), fh, indent=1)
    os.replace(tmp, path)
    return columns
