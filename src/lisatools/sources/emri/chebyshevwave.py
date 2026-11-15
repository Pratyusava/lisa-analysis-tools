"""Cutler & Bustamante-Rosell's Chebyshev model of a single harmonic's ``t(f)``.

The one implementation of the model in LAT, used by the EMRI search
(:class:`lisatools.search.emri.ChebyshevTrackModel`) and by the FD TDI projection
:class:`ChebyshevXYZ` of the development scripts ``scripts/emri/emri_chebyshev_comparison.py``
and ``scripts/emri/emri_snr_correlation_test.py``.

Over a band ``[f_min, f_max]`` mapped to ``y = ln(f / f_mid) / kappa`` in [-1, 1]
(``f_mid = sqrt(f_min f_max)``, ``kappa = ln(f_max / f_min) / 2``) the inverse chirp rate is
expanded on the Chebyshev polynomials of the second kind,

    ln Lambda(y) = sum_i alpha_i U_i(y),    Lambda = f_mid^2 dt/df   (dimensionless),

so that ``t(y) = (kappa / f_mid) int e^{kappa v} Lambda(v) dv`` and ``fdot = f_mid^2 / Lambda``
(draft "A New Detection Statistic for EMRI searches", Sec. IV and Apps. A, D; five
coefficients). :class:`ChebyshevWave` evaluates ``t(f)`` on a grid in ``y``, in seconds, for one
or many coefficient sets, on NumPy or JAX (differentiable, ``jax.vmap``-able);
:func:`chebyshev_fit` recovers the coefficients of a physical track.

History: Curt Cutler's ``alpha_to_tf`` (hard-wired to five terms, SciPy's cumulative Simpson),
vectorised over sources by Michael Katz as ``ChebyshevWave``, whose time was in units of
``1 / f_mid`` (the integral omitted that factor; the development scripts compensated with an
empirical ``t_scale``). Unified with the search's JAX version in 2026-11: seconds, any number of
terms, backend chosen at construction, the quadrature written for both array modules.
"""

from __future__ import annotations

import numpy as np

from ...utils.parallelbase import LISAToolsParallelModule


def band_scales(f_min, f_max) -> tuple[float, float]:
    """``(f_mid, kappa)`` of a band: ``y = ln(f / f_mid) / kappa`` maps it to [-1, 1]."""
    f_min, f_max = float(f_min), float(f_max)
    if not 0.0 < f_min < f_max:
        raise ValueError(f"band must be 0 < f_min < f_max, not {(f_min, f_max)}.")
    return float(np.sqrt(f_min * f_max)), float(0.5 * np.log(f_max / f_min))


def chebyshev_u(y, n, xp=np):
    """Chebyshev polynomials of the second kind ``U_0 .. U_(n-1)`` at ``y``,
    shape ``(n, len(y))``: ``U_0 = 1``, ``U_1 = 2y``, ``U_(i+1) = 2y U_i - U_(i-1)``."""
    y = xp.asarray(y, dtype=float)
    rows = [xp.ones_like(y)]
    if n > 1:
        rows.append(2.0 * y)
    for _ in range(2, n):
        rows.append(2.0 * y * rows[-1] - rows[-2])
    return xp.stack(rows[:n])


def cumulative_simpson(y, dx, xp=np):
    """Cumulative Simpson integral of equally spaced samples along the last axis.

    Zero at the first sample; each interval is integrated with the quadratic through it and
    its neighbour, alternating sides as :func:`scipy.integrate.cumulative_simpson` does
    (``initial=0``), written with slices so it runs under NumPy and JAX.
    """
    n = y.shape[-1]
    if n < 3:
        raise ValueError("cumulative_simpson needs at least three samples.")

    def forward(v):  # interval [i, i+1] from the quadratic through i, i+1, i+2
        return dx / 3.0 * (1.25 * v[..., :-2] + 2.0 * v[..., 1:-1] - 0.25 * v[..., 2:])

    right = forward(y)  # intervals 0 .. n-3, quadratic to their right
    left = forward(y[..., ::-1])[..., ::-1]  # intervals 1 .. n-2, quadratic to their left
    parts = xp.stack([right[..., ::2], left[..., ::2]], axis=-1)  # even, odd intervals
    parts = parts.reshape(parts.shape[:-2] + (-1,))
    if n % 2 == 0:  # the last (even) interval only has a quadratic to its left
        parts = xp.concatenate([parts, left[..., -1:]], axis=-1)
    zero = xp.zeros(parts.shape[:-1] + (1,), dtype=parts.dtype)
    return xp.concatenate([zero, xp.cumsum(parts, axis=-1)], axis=-1)


def chebyshev_fit(f, fdot, band=(3e-3, 8e-3), n_coeffs=5):
    """Least-squares Chebyshev coefficients of one track.

    Fits ``ln(f_mid^2 / fdot) = sum_i alpha_i U_i(y)`` over the track samples inside ``band``
    with ``fdot > 0``.

    Returns:
        ``(alpha, span, rms)``: the coefficients, the width of the fitted ``y`` interval (at
        most 2, the whole band) and the rms residual of ``ln Lambda``. ``alpha`` is all-NaN
        when fewer than ``n_coeffs + 1`` samples qualify.
    """
    f, fdot = np.asarray(f, dtype=float), np.asarray(fdot, dtype=float)
    f_mid, kappa = band_scales(*band)
    sel = (f >= band[0]) & (f <= band[1]) & (fdot > 0.0)
    if sel.sum() < n_coeffs + 1:
        return np.full(n_coeffs, np.nan), 0.0, np.nan
    y = np.log(f[sel] / f_mid) / kappa
    target = np.log(f_mid**2 / fdot[sel])
    design = chebyshev_u(y, n_coeffs).T
    alpha = np.linalg.lstsq(design, target, rcond=None)[0]
    rms = float(np.sqrt(np.mean((design @ alpha - target) ** 2)))
    return alpha, float(y.max() - y.min()), rms


class ChebyshevWave(LISAToolsParallelModule):
    """The inspiral ``t(f)`` of a Chebyshev coefficient set, on a grid even in ``ln f``.

    The grid, the polynomials and ``e^{kappa y}`` are precomputed once; :meth:`track` and
    :meth:`__call__` then evaluate one coefficient set or a batch. Times are in seconds, with
    ``t = 0`` where the track crosses ``f_mid`` (``y = 0``) unless another ``y_ref`` is given.

    Args:
        npts: Number of grid points in ``y`` (the draft uses 5001; the quadrature is Simpson's).
        fmin: Lower edge of the band [Hz].
        fmax: Upper edge of the band [Hz].
        n_coeffs: Number of Chebyshev terms (the draft's five).
        force_backend: ``"cpu"`` (NumPy, default) or ``"jax"``.

    Attributes:
        fmid, kappa: The band's scales (:func:`band_scales`).
        y, f: The grid, shape ``(npts,)``; ``f = fmid e^{kappa y}``.
        dy: Grid spacing in ``y``.
        exp_kappa_y: ``e^{kappa y}``, the Jacobian ``df/dy`` up to ``kappa fmid``.
        U: The polynomials on the grid, shape ``(n_coeffs, npts)``.
    """

    def __init__(self, npts: int, fmin: float, fmax: float, n_coeffs: int = 5, force_backend=None):
        super().__init__(force_backend=force_backend)
        self.npts, self.fmin, self.fmax = int(npts), float(fmin), float(fmax)
        self.n_coeffs = int(n_coeffs)
        if self.npts < 3 or self.n_coeffs < 1:
            raise ValueError("npts must be at least 3 and n_coeffs at least 1.")
        self.fmid, self.kappa = band_scales(self.fmin, self.fmax)
        self.y = np.linspace(-1.0, 1.0, self.npts)
        self.dy = 2.0 / (self.npts - 1)
        self.exp_kappa_y = np.exp(self.kappa * self.y)
        self.f = self.fmid * self.exp_kappa_y
        self.U = chebyshev_u(self.y, self.n_coeffs)

    @classmethod
    def supported_backends(cls):
        return [cls._BACKEND_PREFIX + "_" + t for t in ("cpu", "jax")]

    def inverse_chirp(self, alpha):
        """``Lambda = exp(sum_i alpha_i U_i(y))`` on the grid; ``alpha``: ``(..., n_coeffs)``."""
        xp = self.xp
        return xp.exp(xp.asarray(alpha) @ xp.asarray(self.U))

    def track(self, alpha, y_ref=0.0):
        """``(t, f, fdot)`` on the grid for ``alpha`` of shape ``(..., n_coeffs)``.

        ``t`` [s] and ``fdot`` [Hz/s] have shape ``(..., npts)``, ``f`` [Hz] shape ``(npts,)``.
        ``t = 0`` at ``y_ref`` (a scalar or shape ``(...,)``; linear interpolation on the grid).
        """
        xp = self.xp
        lam = self.inverse_chirp(alpha)
        integral = cumulative_simpson(xp.asarray(self.exp_kappa_y) * lam, self.dy, xp)
        pos = xp.broadcast_to((xp.asarray(y_ref, dtype=float) + 1.0) / self.dy, integral.shape[:-1])
        i = xp.clip(xp.floor(pos), 0, self.npts - 2).astype(int)
        w = pos - i
        lo = xp.take_along_axis(integral, i[..., None], axis=-1)[..., 0]
        hi = xp.take_along_axis(integral, i[..., None] + 1, axis=-1)[..., 0]
        at_ref = (1.0 - w) * lo + w * hi
        t = (self.kappa / self.fmid) * (integral - at_ref[..., None])
        return t, xp.asarray(self.f), self.fmid**2 / lam

    def __call__(self, *alpha_in, return_phase: bool = False):
        """``(t, f)`` for the coefficients given one per argument (scalars, or arrays over a batch).

        ``t`` is in seconds with ``t = 0`` at ``fmid``. For one coefficient set the arrays are 1-D
        of length ``npts``; for a batch ``t`` has shape ``(nsources, npts)``. With
        ``return_phase`` the first array is instead ``2 pi f t`` (a heuristic phase, not the
        integral of ``2 pi f dt``).
        """
        xp = self.xp
        alpha = xp.asarray(alpha_in, dtype=float).T
        squeeze = alpha.ndim == 1
        alpha = xp.atleast_2d(alpha)
        if alpha.ndim != 2 or alpha.shape[1] != self.n_coeffs:
            raise ValueError(f"expected {self.n_coeffs} coefficients, got shape {alpha.shape}.")
        t, f, _ = self.track(alpha)
        out = 2.0 * np.pi * t * f[None, :] if return_phase else t
        return (out[0], f) if squeeze else (out, f)


class ChebyshevXYZ(ChebyshevWave):
    """Project a :class:`ChebyshevWave` onto the LISA XYZ TDI channels on the fly.

    Combines the Chebyshev ``t(f)`` with :class:`lisatools.response.tdionfly.FDTDIonTheFly`
    and :class:`gpubackendtools.interpolate.CubicSplineInterpolant` to produce TDI output for a
    batch of sources directly in the frequency domain. Development code (NumPy only).

    Args:
        fs: Forwarded to :class:`FDTDIonTheFly`'s ``fs`` slot.
        *args, **kwargs: Forwarded to :class:`ChebyshevWave` (``npts``, ``fmin``, ``fmax``, ...).
    """

    # TODO/DOCS: ``fs`` is passed straight through to FDTDIonTheFly's ``fs`` slot;
    # confirm whether it is a sampling rate (Hz) or a frequency-array (Hz array)
    # for the on-the-fly TDI generator.
    def __init__(self, fs, *args, **kwargs):
        from ...response.tdionfly import FDTDIonTheFly

        super().__init__(*args, **kwargs)
        self.fd_gen = FDTDIonTheFly
        self.fs = fs

    def __call__(  # type: ignore[override]
        self,
        amp0,
        phi0,
        inc,
        psi,
        lam,
        beta,
        t_start,
        *alpha,
        return_spline: bool = False,
        **cheb_kwargs,
    ):
        """Generate XYZ TDI waveforms for one or more EMRI sources.

        Args:
            amp0: Overall amplitude scale applied to the TDI amplitude (broadcast over sources).
            phi0: Constant phase offset added to the reference phase (broadcast over sources).
            inc, psi, lam, beta: Inclination, polarisation, ecliptic longitude and latitude [rad].
            t_start: Start time [s] of each source (1-D array): the track's first grid sample
                is placed there.
            *alpha: Chebyshev coefficients forwarded to :meth:`ChebyshevWave.__call__`.
            return_spline: If ``True``, the TDI generator also returns its spline objects.
            **cheb_kwargs: Extra keyword arguments forwarded to :meth:`ChebyshevWave.__call__`.

        Returns:
            The output of the wrapped :class:`FDTDIonTheFly` call, with the amplitude and
            reference-phase shifts from ``amp0`` / ``phi0`` applied.
        """
        from gpubackendtools.interpolate import CubicSplineInterpolant

        t_intrinsic, _f = ChebyshevWave.__call__(self, *alpha, **cheb_kwargs)
        t_intrinsic = np.atleast_2d(np.asarray(t_intrinsic))

        amp0, phi0 = np.atleast_1d(amp0), np.atleast_1d(phi0)
        inc, psi = np.atleast_1d(inc), np.atleast_1d(psi)
        lam, beta = np.atleast_1d(lam), np.atleast_1d(beta)
        t_start = np.atleast_1d(t_start)

        num_bin = inc.shape[0]
        f = np.tile(np.asarray(_f), (num_bin, 1))

        assert (
            t_start.ndim == 1
            and t_start.shape[0] == inc.shape[0]
            and t_start.shape[0] == t_intrinsic.shape[0]
        )
        t = t_intrinsic - (t_intrinsic[:, 0] - t_start)[:, None]

        t_max = t.max(axis=-1)
        buffer = 500.0
        t_tdi = np.linspace(t_start + buffer, t_max - buffer, self.npts, axis=-1)
        # print("NEED TO FIX t so that it fits orbits window")
        f_of_t = CubicSplineInterpolant(t, f)
        A_of_t = CubicSplineInterpolant(t, np.ones_like(t))

        nsources = t.shape[0]
        CUBIC_SPLINE_GENERAL_SPACING = 3  # 1 linear, 2 log10, 3 general

        phase_ref = (2 * np.pi * f * t).flatten().copy()

        # n_params: 11 is nparams / will not affect spline; needs to be 4? need to check that
        fd_wave_gen = self.fd_gen(
            t_tdi,
            A_of_t,
            f_of_t,
            phase_ref,
            self.fs,
            nsources,
            n_params=4,
            spline_type=CUBIC_SPLINE_GENERAL_SPACING,
        )
        wave_output = fd_wave_gen(inc, psi, lam, beta, return_spline=return_spline)

        # TODO: remove this for speed?
        if np.any(amp0 != 1.0):
            wave_output.tdi_amp = wave_output.tdi_amp * amp0[:, None, None]
        if np.any(phi0 != 0.0):
            wave_output.phase_ref = wave_output.phase_ref + phi0[:, None]
        return wave_output
