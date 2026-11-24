"""Frequency-track models for the semi-coherent EMRI search.

A track is the frequency evolution of one harmonic sampled at the SFT segment
starts: ``f_alpha``, ``fdot_alpha`` and optionally ``fddot_alpha``. A track
model maps search parameters to a :class:`Track`; the statistic only ever sees
the :class:`Track`, so models (physical FEW trajectories, SVD bases, Cutler &
Bustamante-Rosell's Chebyshev phase model, ...) swap freely.
"""

from typing import Any, NamedTuple, Optional

import numpy as np
from scipy.stats import qmc

from ...utils.parallelbase import LISAToolsParallelModule


class Track(NamedTuple):
    """One harmonic's frequency track on the segment grid (a JAX pytree).

    ``fddot`` may be ``None``; consumers then difference ``fdot`` themselves.
    """

    f: Any
    fdot: Any
    fddot: Optional[Any] = None


class TrackBank(NamedTuple):
    """A bank of harmonic tracks on one grid (:func:`few_track_bank`)."""

    f: np.ndarray  # (n_tracks, n_samples) [Hz]
    fdot: np.ndarray  # (n_tracks, n_samples) [Hz/s]
    params: np.ndarray  # (n_tracks, 4): the draws (log10 m1, m2, a, e_f)


class SVDBasis:
    """Reduced basis for frequency tracks (rows of ``V`` are orthonormal).

    A track is ``coeffs @ V`` with ``coeffs`` inside the box ``[lb, ub]``; the
    search parametrises ``coeffs`` by the unit cube.

    Args:
        V: Basis, shape ``(n_modes, n_samples)``.
        lb, ub: Coefficient box, shape ``(n_modes,)``.
        S: Singular values (diagnostic only).
    """

    def __init__(self, V, lb, ub, S=None):
        self.V = np.asarray(V, dtype=float)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.S = None if S is None else np.asarray(S, dtype=float)

    @property
    def n_modes(self) -> int:
        return self.V.shape[0]

    @property
    def n_samples(self) -> int:
        return self.V.shape[1]

    def frequency_range(self, sample=0):
        """``(min, max)`` of ``(coeffs @ V)[sample]`` over the coefficient box [Hz].

        ``sample=0`` is the range of starting frequencies the basis can
        represent (before any plunge shift); tracks outside it cannot be
        reached from inside the unit cube.
        """
        terms = np.stack([self.lb, self.ub]) * self.V[:, sample]
        return float(terms.min(axis=0).sum()), float(terms.max(axis=0).sum())

    @classmethod
    def build(cls, tracks, n_modes, box_quantile=0.0, first_mode_max=False):
        """SVD of a training bank of tracks, shape ``(n_tracks, n_samples)``
        (e.g. :func:`few_track_bank`), truncated to ``n_modes``.

        The coefficient box is ``[Q_q, Q_(1-q)]`` of the training coefficients
        per mode, ``q = box_quantile`` (``0``: their min / max). With
        ``first_mode_max`` the upper edge of the first mode is the maximum
        instead of ``Q_(1-q)``, so the highest-frequency training tracks stay
        inside the box; ``box_quantile=0.015, first_mode_max=True`` reproduces
        the box of EMRI-Search's basis file.
        """
        tracks = np.asarray(tracks, dtype=float)
        if not 0.0 <= box_quantile < 0.5:
            raise ValueError(f"box_quantile must be in [0, 0.5), not {box_quantile}.")
        U, S, Vt = np.linalg.svd(tracks, full_matrices=False)
        coeffs = U[:, :n_modes] * S[:n_modes]
        lb = np.quantile(coeffs, box_quantile, axis=0)
        ub = np.quantile(coeffs, 1.0 - box_quantile, axis=0)
        if first_mode_max:
            ub[0] = coeffs[:, 0].max()
        return cls(Vt[:n_modes], lb, ub, S=S[:n_modes])

    @classmethod
    def from_npz(cls, path):
        """Load Lorenzo Speri's EMRI-Search basis files (``V_K``, ``lb``, ``ub``)."""
        d = np.load(path, allow_pickle=True)
        S = np.diag(d["S_K"]) if "S_K" in d.files else None
        return cls(d["V_K"], d["lb"], d["ub"], S=S)

    def save_npz(self, path):
        np.savez(
            path,
            V_K=self.V,
            lb=self.lb,
            ub=self.ub,
            S_K=np.diag(self.S) if self.S is not None else np.zeros(0),
        )


class SVDTrackModel(LISAToolsParallelModule):
    """Track = basis expansion, shifted in time to a trial plunge.

    The basis tracks all plunge at ``T_ref``; a trial plunge time ``T_pl`` rolls
    the expansion by ``round_toward_zero((T_pl - T_ref) / T_sft)`` segments,
    zero-filling the vacated samples, and truncates to the data's segments.
    ``fdot`` is the central difference of ``f`` over the segment times.

    Args:
        basis: The :class:`SVDBasis`.
        t_alpha: Segment start times [s], length ``n_segments <= basis.n_samples``.
        T_ref: Plunge time the basis tracks share [s].
        force_backend: ``"cpu"`` (NumPy) or ``"jax"``.
    """

    def __init__(self, basis: SVDBasis, t_alpha, T_ref, force_backend=None):
        super().__init__(force_backend=force_backend)
        self.basis = basis
        self.t_alpha = np.asarray(t_alpha, dtype=float)
        self.T_ref = float(T_ref)
        if self.t_alpha.size > basis.n_samples:
            raise ValueError(
                f"{self.t_alpha.size} segments but the basis only spans {basis.n_samples}."
            )

    @classmethod
    def supported_backends(cls):
        return [cls._BACKEND_PREFIX + "_" + t for t in ("cpu", "jax")]

    @property
    def ndim(self) -> int:
        return self.basis.n_modes

    @property
    def T_sft(self) -> float:
        return float(self.t_alpha[1] - self.t_alpha[0])

    def shift_for_plunge(self, T_pl) -> int:
        """Segment shift that moves the basis plunge ``T_ref`` to ``T_pl`` [s]."""
        return int((T_pl - self.T_ref) / self.T_sft)

    def frequency_range(self):
        """``(min, max)`` frequency [Hz] any track can take at any basis sample
        (any shift brings any sample into the data)."""
        terms = np.stack([self.basis.lb, self.basis.ub])[:, :, None] * self.basis.V[None]
        return float(terms.min(axis=0).sum(axis=0).min()), float(
            terms.max(axis=0).sum(axis=0).max()
        )

    def coeffs(self, xi):
        """Unit-cube point(s) -> basis coefficients."""
        xp = self.xp
        return xp.asarray(self.basis.lb) + xi * xp.asarray(self.basis.ub - self.basis.lb)

    def unit(self, coeffs):
        """Basis coefficients -> unit-cube point(s)."""
        return (coeffs - self.basis.lb) / (self.basis.ub - self.basis.lb)

    def __call__(self, xi, shift=0) -> Track:
        xp = self.xp
        f = self.coeffs(xi) @ xp.asarray(self.basis.V)
        f = _roll_and_zero(f, shift, xp)[: self.t_alpha.size]
        fdot = xp.gradient(f, xp.asarray(self.t_alpha))
        return Track(f, fdot)

    def initial_population(self, n, f0_range=(5e-4, 1e-2), seed=None, out_of_cube="keep"):
        """Space-filling unit-cube points whose tracks start in ``f0_range``.

        Sobol draws in the coefficient box, each track's offset re-set so it
        starts at a uniform ``f0`` in ``f0_range``; draws with any negative
        frequency are dropped, so fewer than ``n`` points may come back.

        Re-setting the offset moves most points out of the unit cube, which the
        optimisers then clip on their first step. ``out_of_cube`` sets what
        happens here:

        * ``"keep"`` (EMRI-Search behaviour): return them unclipped, so the
          screening ranks tracks that clipping later changes.
        * ``"clip"``: clip into the cube before the positivity filter, so the
          screening ranks the tracks that get optimised (their ``f0`` is then
          no longer exactly the drawn one).
        """
        if out_of_cube not in ("keep", "clip"):
            raise ValueError(f"out_of_cube must be 'keep' or 'clip', not {out_of_cube!r}.")
        rng = np.random.default_rng(seed)
        unit = qmc.Sobol(d=self.ndim, rng=rng).random(n)
        V = self.basis.V
        f = (self.basis.lb + unit * (self.basis.ub - self.basis.lb)) @ V
        f = f - f[:, :1] + rng.uniform(*f0_range, size=(n, 1))
        coeffs = f @ V.T
        if out_of_cube == "keep":
            keep = np.all(coeffs @ V >= 0.0, axis=1)
            return self.unit(coeffs[keep])
        unit = np.clip(self.unit(coeffs), 0.0, 1.0)
        keep = np.all((self.basis.lb + unit * (self.basis.ub - self.basis.lb)) @ V >= 0.0, axis=1)
        return unit[keep]


class FEWHarmonicTrackModel(LISAToolsParallelModule):
    """Track of harmonic ``m Phi_phi + n Phi_r`` from a FEW Kerr eccentric
    equatorial inspiral, integrated backwards from plunge (NumPy only).

    Parameters ``(m1, m2, a, T_pl, e_f[, x_f])``: source masses [M_sun] (detector
    frame), spin, time from ``t = 0`` to plunge [yr], eccentricity at the
    separatrix, and ``x_f = +1`` / ``-1`` for a prograde / retrograde orbit
    (default ``+1``; the same tuple :class:`~.injection.FEWInjector` takes).
    The inspiral starts at ``p_sep(a, e_f) + buffer`` and runs back ``T_pl``;
    segments before the backward-integrated start or after plunge get zeros.
    Mirrors EMRI-Search's ``get_f_fdot_fddot_back``.

    Args:
        t_alpha: Segment start times [s].
        m, n: Harmonic numbers (``m = 2, n = 0`` is the dominant one).
        err: Integrator tolerance.
    """

    def __init__(self, t_alpha, m=2, n=0, err=1e-11, force_backend="cpu"):
        super().__init__(force_backend=force_backend)
        self.t_alpha = np.asarray(t_alpha, dtype=float)
        self.m, self.n = int(m), int(n)
        self.err = float(err)

    @classmethod
    def supported_backends(cls):
        return [cls._BACKEND_PREFIX + "_cpu"]

    @property
    def traj(self):
        if getattr(self, "_traj", None) is None:
            self._traj = kerr_ecc_eq_inspiral()
        return self._traj

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_traj", None)
        return state

    def phases_and_derivatives(self, m1, m2, a, T_pl, e_f, x_f=1.0):
        """``(phase, f, fdot, fddot)`` per orbital phase, each ``(2, n_segments)``
        for ``(Phi_phi, Phi_r)``; ``f`` etc. are orbital frequencies (no ``m, n``)."""
        from ...sources.emri.wdm_direct import _phase_columns
        from ...utils.constants import YRSID_SI

        traj = self.traj
        gen = traj.inspiral_generator
        backward_inspiral(traj, m1, m2, a, T_pl, e_f, x_f, err=self.err)

        tau = T_pl * YRSID_SI - self.t_alpha  # time before plunge
        ok = (tau > 0.0) & (tau < traj.integrator_spline_t.max())
        out = np.zeros((4, 2, self.t_alpha.size))
        for order in range(4):
            # _phase_columns -> (Phi_phi, Phi_theta, Phi_r); keep phi and r.
            cols = _phase_columns(gen, tau[ok], order)[:, [0, 2]].T
            # d/dt = -d/dtau: odd derivative orders flip sign.
            out[order][:, ok] = cols if order == 0 else (-1) ** order * cols / (2 * np.pi)
        return out

    def __call__(self, params) -> Track:
        _, f, fdot, fddot = self.phases_and_derivatives(*params)
        w = np.array([self.m, self.n])
        return Track(w @ f, w @ fdot, w @ fddot)


def few_track_bank(
    prior, n_tracks, t_alpha, T_ref, seed=None, m=2, n=0, err=1e-11, max_draws=None, n_workers=1
):
    """A bank of FEW harmonic tracks plunging at ``T_ref``, for :meth:`SVDBasis.build`.

    Draws ``(log10 m1, m2, a, e_f)`` uniformly in ``prior`` (a dict of
    ``(low, high)`` under the keys ``"log10_m1", "m2", "a", "e_f"``; prograde
    orbits) and keeps a draw when its ``m Phi_phi + n Phi_r`` track
    (:class:`FEWHarmonicTrackModel`) spans the whole grid ``t_alpha`` [s], i.e.
    the inspiral integrated back from the plunge at ``T_ref`` [s] reaches
    ``t_alpha[0]``; draws FEW cannot integrate back that far are discarded.
    Segments that do not complete before the plunge (``t + T_sft > T_ref``,
    with ``T_sft`` the grid spacing) are set to zero, as in EMRI-Search's bank.

    Args:
        prior: Uniform ranges of the four parameters.
        n_tracks: Tracks in the bank; drawing continues until there are that many
            (at most ``max_draws`` draws, default ``20 * n_tracks``).
        t_alpha: Segment start times of the basis [s], e.g.
            ``np.arange(0, T_ref, T_sft)``.
        T_ref: Plunge time of every track [s].
        seed: Seed of the draws.
        m, n, err: Harmonic and integrator tolerance (:class:`FEWHarmonicTrackModel`).
        n_workers: Processes the FEW integrations run in (~0.02 s each). The bank is the
            same for every ``n_workers``.

    Returns:
        A :class:`TrackBank`: ``f`` and ``fdot``, shape ``(n_tracks, len(t_alpha))``,
        and the kept draws ``params``, shape ``(n_tracks, 4)``.
    """
    keys = ("log10_m1", "m2", "a", "e_f")
    lo = np.array([prior[k][0] for k in keys], dtype=float)
    hi = np.array([prior[k][1] for k in keys], dtype=float)
    t_alpha = np.asarray(t_alpha, dtype=float)
    T_sft = t_alpha[1] - t_alpha[0]
    incomplete = t_alpha + T_sft > T_ref
    model = FEWHarmonicTrackModel(t_alpha, m=m, n=n, err=err)
    rng = np.random.default_rng(seed)
    max_draws = 20 * n_tracks if max_draws is None else int(max_draws)

    # The draws come from rng one at a time, in the same order whatever n_workers is, and are
    # accepted in that order: the bank does not depend on how the integrations are spread.
    tracks, fdots, params, draws = [], [], [], 0
    with _BankIntegrator(model, T_ref, n_workers) as integrate:
        while len(tracks) < n_tracks:
            if draws >= max_draws:
                raise _bank_exhausted(len(tracks), n_tracks, draws, T_ref)
            batch = min(integrate.batch_size, max_draws - draws)
            candidates = [rng.uniform(lo, hi) for _ in range(batch)]
            draws += batch
            for candidate, track in zip(candidates, integrate(candidates)):
                if track is None or len(tracks) == n_tracks:
                    continue
                f, fdot = track
                f[incomplete] = 0.0
                fdot[incomplete] = 0.0
                tracks.append(f)
                fdots.append(fdot)
                params.append(tuple(candidate))
    return TrackBank(np.array(tracks), np.array(fdots), np.array(params))


def _bank_exhausted(n_kept, n_tracks, draws, T_ref):
    from ...utils.constants import YRSID_SI

    return RuntimeError(
        f"{n_kept} of {n_tracks} tracks after {draws} draws: the prior rarely gives inspirals "
        f"that last {T_ref / YRSID_SI:.3g} yr."
    )


class _BankIntegrator:
    """Integrates bank draws ``(log10 m1, m2, a, e_f)`` into ``(f, fdot)``, or ``None`` when
    FEW cannot integrate the draw back to the grid's start. Serial for ``n_workers = 1``,
    otherwise in that many processes (each builds its own FEW trajectory)."""

    BATCH_PER_WORKER = 16

    def __init__(self, model, T_ref, n_workers=1):
        self.model, self.T_ref, self.n_workers = model, float(T_ref), max(int(n_workers), 1)
        self.batch_size = 1 if self.n_workers == 1 else self.BATCH_PER_WORKER * self.n_workers
        self.pool = None

    def __enter__(self):
        if self.n_workers > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor

            self.pool = ProcessPoolExecutor(
                self.n_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_set_bank_worker,
                initargs=(self.model, self.T_ref),
            )
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self.pool.shutdown()

    def __call__(self, candidates):
        if self.pool is None:
            return [_integrate_draw(self.model, self.T_ref, c) for c in candidates]
        return list(self.pool.map(_bank_worker_draw, candidates, chunksize=4))


_BANK_WORKER: dict = {}


def _set_bank_worker(model, T_ref):
    _BANK_WORKER.update(model=model, T_ref=T_ref)


def _bank_worker_draw(candidate):
    return _integrate_draw(_BANK_WORKER["model"], _BANK_WORKER["T_ref"], candidate)


def _integrate_draw(model, T_ref, candidate):
    """``(f, fdot)`` of one bank draw, or ``None`` when it is unusable."""
    from ...utils.constants import YRSID_SI

    log10_m1, m2, a, e_f = candidate
    try:
        track = model((10**log10_m1, m2, a, T_ref / YRSID_SI, e_f))
    except Exception:  # FEW cannot integrate this draw back to t_alpha[0]
        return None
    f, fdot = np.array(track.f), np.array(track.fdot)
    if f[0] == 0.0:  # the inspiral is shorter than the grid
        return None
    return f, fdot


def kerr_ecc_eq_inspiral():
    """A FEW ``EMRIInspiral`` with the Kerr eccentric equatorial flux ODE."""
    from few.trajectory.inspiral import EMRIInspiral
    from few.trajectory.ode import KerrEccEqFlux

    return EMRIInspiral(func=KerrEccEqFlux)


def backward_inspiral(traj, m1, m2, a, T_pl, e_f, x_f=1.0, err=1e-11):
    """Integrate a FEW inspiral back ``T_pl`` [yr] from just outside the separatrix.

    Starts at ``p_sep(a, e_f, x_f) + separatrix_buffer_dist + 1e-6`` (as
    EMRI-Search does) and leaves the trajectory splines on ``traj``.

    Returns:
        The trajectory tuple ``(t, p, e, x, Phi_phi, Phi_theta, Phi_r)``; its
        last samples are the initial conditions ``T_pl`` before plunge.
    """
    from few.utils.geodesic import get_separatrix

    p_f = traj.inspiral_generator.func.separatrix_buffer_dist + get_separatrix(a, e_f, x_f) + 1e-6
    return traj(m1, m2, a, p_f, e_f, x_f, T=T_pl, integrate_backwards=True, err=err)


class ChebyshevBasis:
    """The coefficient box of Cutler & Bustamante-Rosell's Chebyshev phase model.

    The model itself, ``ln Lambda(y) = sum_i alpha_i U_i(y)`` over a band mapped
    to ``y`` in [-1, 1], is :mod:`lisatools.sources.emri.chebyshevwave`
    (:class:`~lisatools.sources.emri.chebyshevwave.ChebyshevWave` evaluates
    ``t(f)``, :func:`~lisatools.sources.emri.chebyshevwave.chebyshev_fit` fits a
    track). This class holds the search's box ``[lb, ub]`` of coefficients, which
    :meth:`fit` derives from a bank of physical tracks as the draft does (the
    accepted fits' range with a margin, optionally widened about its centre);
    :class:`ChebyshevTrackModel` turns a point of the box into a track.

    Args:
        lb, ub: Coefficient box, shape ``(n_coeffs,)``.
        band: ``(f_min, f_max)`` [Hz] (draft: 3-8 mHz).
        fits: The accepted catalogue fits the box was made from (diagnostic).
    """

    def __init__(self, lb, ub, band=(3e-3, 8e-3), fits=None):
        from ...sources.emri.chebyshevwave import band_scales

        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.band = (float(band[0]), float(band[1]))
        self.f_mid, self.kappa = band_scales(*self.band)
        self.fits = None if fits is None else np.asarray(fits, dtype=float)
        if self.lb.shape != self.ub.shape or self.lb.ndim != 1:
            raise ValueError("lb and ub must be 1-d arrays of the same length.")

    @property
    def n_coeffs(self) -> int:
        return self.lb.size

    def y(self, f):
        """Frequency [Hz] -> ``y``."""
        return np.log(np.asarray(f, dtype=float) / self.f_mid) / self.kappa

    def f(self, y):
        """``y`` -> frequency [Hz]."""
        return self.f_mid * np.exp(self.kappa * np.asarray(y, dtype=float))

    def frequency_range(self):
        """The band: every track is zero outside it."""
        return self.band

    @classmethod
    def fit(
        cls,
        f,
        fdot,
        n_coeffs=5,
        band=(3e-3, 8e-3),
        margin=0.2,
        min_span=1.9,
        rms_max=None,
        scale=1.0,
    ):
        """The box of a bank of tracks (e.g. :func:`few_track_bank`), as the draft's
        catalogue box: each track's coefficients from
        :func:`~lisatools.sources.emri.chebyshevwave.chebyshev_fit`; fits are
        kept when they span at least ``min_span`` of the band's width 2 in
        ``y``, have ``alpha_0 > 0`` and (if ``rms_max``) an rms residual below
        it; the box is the kept coefficients' range widened by ``margin`` of
        that range at each end, then by ``scale`` about its centre (the draft's
        wide box is ``scale=10``).

        Args:
            f, fdot: Tracks, shape ``(n_tracks, n_samples)`` (zeros outside the inspiral).
        """
        from ...sources.emri.chebyshevwave import chebyshev_fit

        fits = []
        for fi, fdi in zip(np.asarray(f, dtype=float), np.asarray(fdot, dtype=float)):
            alpha, span, rms = chebyshev_fit(fi, fdi, band, n_coeffs)
            if np.all(np.isfinite(alpha)) and span >= min_span and alpha[0] > 0.0:
                if rms_max is None or rms <= rms_max:
                    fits.append(alpha)
        if not fits:
            raise ValueError("no track passed the catalogue cuts (band coverage, alpha_0 > 0).")
        fits = np.array(fits)
        lo, hi = fits.min(axis=0), fits.max(axis=0)
        lb, ub = lo - margin * (hi - lo), hi + margin * (hi - lo)
        centre, half = 0.5 * (lb + ub), 0.5 * scale * (ub - lb)
        return cls(centre - half, centre + half, band=band, fits=fits)

    @classmethod
    def from_npz(cls, path):
        d = np.load(path, allow_pickle=True)
        if "cheb_lb" not in d.files:
            raise ValueError(f"{path} is not a Chebyshev basis file (no 'cheb_lb').")
        fits = d["cheb_fits"] if "cheb_fits" in d.files and d["cheb_fits"].size else None
        return cls(d["cheb_lb"], d["cheb_ub"], band=tuple(d["band"]), fits=fits)

    def save_npz(self, path):
        np.savez(
            path,
            cheb_lb=self.lb,
            cheb_ub=self.ub,
            band=np.array(self.band),
            cheb_fits=self.fits if self.fits is not None else np.zeros((0, self.n_coeffs)),
        )


class ChebyshevTrackModel(LISAToolsParallelModule):
    """Track = a :class:`ChebyshevBasis` point, placed by its frequency at the end of the stream.

    The unit cube has ``basis.n_coeffs + 1`` coordinates: the coefficients
    ``alpha`` in the box, and ``y_ref`` in ``y_range``, the track's ``y`` (log
    frequency in the band) at the anchor time ``T_ref``. A trial plunge time
    ``T_pl`` moves the anchor by ``shift_for_plunge(T_pl)`` segments, as the SVD
    model moves its plunge, so Lorenzo's duration scan (anchor at the end of
    each stream) applies unchanged. The track is integrated from the anchor,

        t(y) = T_anchor + (kappa / f_mid) int_{y_ref}^{y} e^{kappa v} Lambda(v) dv

    (draft Eq. A1 with the integration constant at ``y_ref`` instead of 0), by
    :class:`~lisatools.sources.emri.chebyshevwave.ChebyshevWave` on ``n_y``
    points in ``y``, and read at the segment starts; ``fdot = f_mid^2 /
    Lambda``. Samples before the track enters the band or after it leaves are
    zero (the statistics' masks drop them). ``fddot`` is left to the mask's
    differences.

    Args:
        basis: The :class:`ChebyshevBasis` (box and band).
        t_alpha: Segment start times [s].
        T_ref: Anchor time [s] of ``y_ref`` before any shift (the data's end).
        y_range: Range of the anchor coordinate (default the whole band, (-1, 1)).
        n_y: Quadrature points in ``y`` (draft: 5001).
        force_backend: ``"cpu"`` (NumPy) or ``"jax"``.

    Attributes:
        wave: The :class:`~lisatools.sources.emri.chebyshevwave.ChebyshevWave`
            on the same backend.
    """

    def __init__(
        self,
        basis: ChebyshevBasis,
        t_alpha,
        T_ref,
        y_range=(-1.0, 1.0),
        n_y=5001,
        force_backend=None,
    ):
        from ...sources.emri.chebyshevwave import ChebyshevWave

        super().__init__(force_backend=force_backend)
        self.basis = basis
        self.t_alpha = np.asarray(t_alpha, dtype=float)
        self.T_ref = float(T_ref)
        self.y_range = (float(y_range[0]), float(y_range[1]))
        self.n_y = int(n_y)
        self.wave = ChebyshevWave(
            self.n_y,
            basis.band[0],
            basis.band[1],
            n_coeffs=basis.n_coeffs,
            force_backend=self.backend,
        )

    @classmethod
    def supported_backends(cls):
        return [cls._BACKEND_PREFIX + "_" + t for t in ("cpu", "jax")]

    @property
    def ndim(self) -> int:
        return self.basis.n_coeffs + 1

    @property
    def T_sft(self) -> float:
        return float(self.t_alpha[1] - self.t_alpha[0])

    def shift_for_plunge(self, T_pl) -> int:
        """Segment shift that moves the anchor ``T_ref`` to ``T_pl`` [s]."""
        return int((T_pl - self.T_ref) / self.T_sft)

    def frequency_range(self):
        """The band [Hz]: tracks are zero outside it."""
        return self.basis.band

    def coeffs(self, xi):
        """Unit-cube point -> ``(alpha, y_ref)``."""
        xp = self.xp
        b = self.basis
        alpha = xp.asarray(b.lb) + xi[..., : b.n_coeffs] * xp.asarray(b.ub - b.lb)
        y_ref = self.y_range[0] + xi[..., b.n_coeffs] * (self.y_range[1] - self.y_range[0])
        return alpha, y_ref

    def unit(self, alpha, y_ref):
        """``(alpha, y_ref)`` -> unit-cube point."""
        b = self.basis
        xi_y = (np.asarray(y_ref) - self.y_range[0]) / (self.y_range[1] - self.y_range[0])
        return np.concatenate([(np.asarray(alpha) - b.lb) / (b.ub - b.lb), np.atleast_1d(xi_y)])

    def grid(self, alpha, y_ref, T_anchor):
        """``(t, f, fdot)`` of the track on the ``y`` grid, anchored at ``t(y_ref) = T_anchor``."""
        t, f, fdot = self.wave.track(alpha, y_ref)
        return T_anchor + t, f, fdot

    def __call__(self, xi, shift=0) -> Track:
        xp = self.xp
        alpha, y_ref = self.coeffs(xi)
        t, f, fdot = self.grid(alpha, y_ref, self.T_ref + shift * self.T_sft)
        t_alpha = xp.asarray(self.t_alpha)
        inside = (t_alpha >= t[0]) & (t_alpha <= t[-1])
        f_alpha = xp.where(inside, xp.interp(t_alpha, t, f), 0.0)
        fdot_alpha = xp.where(inside, xp.interp(t_alpha, t, fdot), 0.0)
        return Track(f_alpha, fdot_alpha)

    def initial_population(self, n, seed=None):
        """Sobol points of the unit cube (coefficients and anchor frequency)."""
        rng = np.random.default_rng(seed)
        return qmc.Sobol(d=self.ndim, rng=rng).random(n)


def _roll_and_zero(f, shift, xp):
    """Roll along the last axis by ``shift`` and zero the wrapped-in samples."""
    rolled = xp.roll(f, shift, axis=-1)
    idx = xp.arange(f.shape[-1])
    n = f.shape[-1]
    wrapped = xp.where(shift > 0, idx < shift, False) | xp.where(shift < 0, idx >= n + shift, False)
    return xp.where(wrapped, 0.0, rolled)
