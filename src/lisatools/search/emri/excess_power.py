"""Coherent, delay-scanning statistics on a band-limited FFT (:class:`~.data.FDData`).

A track becomes one full-length, unit-norm template; one inverse FFT gives its
complex correlation with the data at every time delay ``tau_j``,

    rho_j = <d | h_{tau_j}> = 4 df sum_k conj(d_k) h_k e^{-2 pi i f_k tau_j} / S_k,

on the grid ``tau_j = j T / M``, ``j < M`` (circular, period ``T``), and a
statistic reduces ``|rho_j|^2`` over the delays. ``M = T / delay_step`` is set
by the delay resolution, not by the number of bins: when the band holds more
bins than ``M``, the weighted spectrum is folded modulo ``M`` (bins ``k`` and
``k + M`` share every ``e^{2 pi i k j / M}``) before an ``M``-point inverse FFT,
which gives ``rho_j`` exactly at a cost set by ``M``.

The statistics:

* :class:`MatchedFilterStatistic` -- the phase- and delay-maximised matched
  filter ``max_j |rho_j|``;
* :class:`ExcessPowerStatistic` -- Cutler & Bustamante-Rosell's ``H``: the
  excess power ``|rho_j|^2 - 2`` summed over a window of consecutive delays,
  normalised to unit noise variance and maximised over the window's start and
  length. A template that only crosses the signal's track in the f-t plane
  overlaps it near the crossing, and so do its time-slides; ``H`` collects that
  power from many delays, which keeps it informative far from the signal where
  ``|rho|`` is buried in noise (C. Cutler & M. J. Bustamante-Rosell, "A New
  Detection Statistic for EMRI searches", draft).

The template (:meth:`DelayCorrelationStatistic.template`) follows from the
track by the stationary-phase approximation: the time ``t(f)`` at which the
track passes each frequency (piecewise-linear inversion of the track), the
phase ``Psi(f) = 2 pi int t(f) df`` (trapezoid rule on the bin grid; constant
and linear terms are maximised over), and the amplitude ``|h(f)| ~ fdot^{-1/2}``
of a constant-amplitude chirp (``amplitude="spa"``) or the flat ``|h(f)| = 1``
of the draft's toy model (``amplitude="flat"``), over the frequencies the track
spans. Neither LISA's response nor the harmonic's amplitude evolution is in
the template.
"""

import copy

import numpy as np
from scipy.fft import next_fast_len

from .statistic import SegmentMask, Statistic
from .tracks import Track


class DelayCorrelationStatistic(Statistic):
    """Base: a track's unit-norm template correlated with the data at every delay.

    Subclasses implement :meth:`reduce` (``|rho_j|^2`` over delays -> statistic).

    Args:
        T: Duration of the data stream [s] (``1 / df``).
        psd: One-sided PSD callable ``psd(f)``.
        t_alpha: Times of the track samples [s] (the track model's grid,
            uniformly spaced).
        first_bin: Absolute bin index (``f / df``) of the data's first bin.
        n_bins: Number of stored bins.
        delay_step: Target delay resolution [s]; the inverse FFT has
            ``T / delay_step`` points (rounded up to a fast size, so the
            actual step :attr:`delay_step` is at most this). Delays finer than
            ``1 / (template bandwidth)`` resolve a matched template's
            correlation peak; a coarser grid samples it (the matched filter
            then loses up to its peak's drop over half a step), while the
            noise normalisation of :class:`ExcessPowerStatistic` stays exact.
        amplitude: ``"spa"`` (``|h(f)| ~ fdot^{-1/2}``) or ``"flat"``.
        mask: Which track samples enter the template (in band, chirping up).
            The default drops the semi-coherent linear-chirp cut, which does
            not apply to a coherent template.
        fdot_floor: Smallest ``fdot`` [Hz/s] in the ``"spa"`` amplitude.
        span: Evaluate on a window of ``span`` [Hz] (plus two bins) starting at
            the track's lowest frequency instead of on the whole band; ``None``
            (default) is the whole band. Exact for a track whose
            :meth:`support` width is at most ``span``: its template is zero
            outside the window, and moving the window only multiplies ``rho_j``
            by a phase. Cost scales with the window, so a population sorted by
            width (:meth:`with_span`, :class:`~.optimize.BucketedLoss`) skips
            the empty bins of the narrow templates.
        force_backend: ``"cpu"`` (NumPy) or ``"jax"``.
    """

    def __init__(
        self,
        T,
        psd,
        t_alpha,
        first_bin=0,
        n_bins=None,
        delay_step=200.0,
        amplitude="spa",
        mask=SegmentMask(dphi_max=None),
        fdot_floor=1e-16,
        span=None,
        force_backend=None,
    ):
        super().__init__(force_backend=force_backend)
        if amplitude not in ("spa", "flat"):
            raise ValueError(f'amplitude must be "spa" or "flat", not {amplitude!r}.')
        if n_bins is None:
            raise ValueError("n_bins is required (for_data sets it from the data).")
        self.T = float(T)
        self.df = 1.0 / self.T
        self.psd = psd
        self.t_alpha = np.asarray(t_alpha, dtype=float)
        if self.t_alpha.size > 2 and not np.allclose(
            np.diff(self.t_alpha), self.t_alpha[1] - self.t_alpha[0], rtol=1e-9, atol=0.0
        ):
            raise ValueError("t_alpha must be uniformly spaced.")
        self.first_bin = int(first_bin)
        self.n_bins = int(n_bins)
        self.amplitude = amplitude
        self.mask = mask
        self.fdot_floor = float(fdot_floor)
        self.freqs = (self.first_bin + np.arange(self.n_bins)) * self.df
        self.psd_bins = np.asarray(psd(self.freqs), dtype=float)
        self.n_fft = next_fast_len(int(np.ceil(self.T / float(delay_step))))
        self._set_span(span)

    def _set_span(self, span):
        self.span = None if span is None else float(span)
        self.n_window = (
            self.n_bins if span is None else min(self.n_bins, int(np.ceil(span / self.df)) + 2)
        )

    def with_span(self, span):
        """A copy evaluated on windows of ``span`` [Hz] (``None``: whole band)."""
        new = copy.copy(self)
        new._set_span(span)
        return new

    @classmethod
    def for_data(cls, data, t_alpha, **kwargs):
        """Statistic on the grid, band and PSD of an :class:`~.data.FDData`."""
        return cls(
            data.T, data.psd, t_alpha, first_bin=data.first_bin, n_bins=data.n_bins, **kwargs
        )

    @property
    def delay_step(self) -> float:
        """Spacing of the delay grid [s]."""
        return self.T / self.n_fft

    def prepare(self, data):
        """The band of an :class:`~.data.FDData`, ``(n_channels, n_bins)``."""
        return self.to_backend(data.fd)

    def fold(self, x):
        """``x`` along its last axis summed modulo ``M`` (= :attr:`n_fft`);
        unchanged when it has at most ``M`` entries."""
        m, n = self.n_fft, x.shape[-1]
        if n <= m:
            return x
        n_fold = -(-n // m)
        pad = [(0, 0)] * (x.ndim - 1) + [(0, n_fold * m - n)]
        return self.xp.pad(x, pad).reshape(x.shape[:-1] + (n_fold, m)).sum(axis=-2)

    def delay_transform(self, x):
        """``M sum_k x_k e^{2 pi i k j / M}`` for every delay ``j < M``, along
        the last axis: :meth:`fold`, then one ``M``-point inverse FFT."""
        return self.xp.fft.ifft(self.fold(x), n=self.n_fft, axis=-1) * self.n_fft

    def _take(self, x, k0):
        """Bins ``k0 .. k0 + n_window`` of ``x`` along its last axis."""
        if self.n_window == self.n_bins:
            return x
        if self.is_jax:
            import jax

            return jax.lax.dynamic_slice_in_dim(x, k0, self.n_window, axis=-1)
        return x[..., int(k0) : int(k0) + self.n_window]

    def _knots(self, track: Track):
        """Masked knots ``(t, f, fdot)`` (last one extended through its segment),
        the running maximum ``f_up`` of ``f``, and the lowest kept frequency."""
        xp = self.xp
        t = xp.asarray(self.t_alpha)
        dt_track = float(self.t_alpha[1] - self.t_alpha[0]) if self.t_alpha.size > 1 else 1.0
        keep = self.mask(track, dt_track, xp) > 0
        f = xp.where(keep, track.f, 0.0)
        fdot = xp.where(keep, track.fdot, 0.0)
        # Samples sit at segment starts: extend the last one through its segment.
        t = xp.concatenate([t, t[-1:] + dt_track])
        f = xp.concatenate([f, f[-1:] + fdot[-1:] * dt_track])
        fdot = xp.concatenate([fdot, fdot[-1:]])
        f_up = self._cummax(f)  # non-decreasing, so t(f) is single-valued
        f_lo = xp.min(xp.where(keep, track.f, xp.inf))
        return t, f, fdot, f_up, f_lo

    def support(self, track: Track):
        """Width [Hz] of the template's frequency range (``-inf`` when no track
        sample passes the mask): the window a :attr:`span` must cover."""
        _, _, _, f_up, f_lo = self._knots(track)
        return f_up[-1] - f_lo

    def _offset(self, f_lo):
        """First bin of the window for a template starting at ``f_lo``."""
        xp = self.xp
        if self.n_window == self.n_bins:
            return 0
        k = xp.floor(xp.where(xp.isfinite(f_lo), f_lo, 0.0) / self.df) - self.first_bin
        if self.is_jax:
            import jax

            k = jax.lax.stop_gradient(k)
        return xp.clip(k, 0, self.n_bins - self.n_window).astype(int)

    def _bracket(self, f_up, k0=0):
        """For every window bin ``f_k``: the knot interval ``i`` with
        ``f_up[i] <= f_k < f_up[i + 1]`` (clipped to the ends) and the linear
        weight in it, i.e. ``np.interp`` on non-decreasing knots. The bins are
        a uniform grid, so each knot's first bin at or above it is arithmetic;
        a cumulative count of those gives ``i`` without searching the bins."""
        xp = self.xp
        n, m = self.n_window, f_up.shape[0]
        pos = f_up / self.df - (self.first_bin + k0)
        if self.is_jax:
            import jax

            pos = jax.lax.stop_gradient(pos)
        first = xp.clip(xp.ceil(pos), 0, n).astype(int)
        if self.is_jax:
            counts = xp.zeros(n + 1, dtype=int).at[first].add(1)
        else:
            counts = np.bincount(first, minlength=n + 1)
        i = xp.clip(xp.cumsum(counts)[:n] - 1, 0, m - 2)
        lo, hi = f_up[i], f_up[i + 1]
        width = hi - lo
        f_k = (self.first_bin + k0 + xp.arange(n)) * self.df
        w = xp.clip((f_k - lo) / xp.where(width > 0, width, 1.0), 0.0, 1.0)
        # a flat last interval: bins above it take the last knot, as np.interp
        return i, xp.where(width > 0, w, (f_k >= hi).astype(float))

    def _cummax(self, x):
        if self.is_jax:
            import jax

            return jax.lax.cummax(x, axis=0)
        return np.maximum.accumulate(x)

    def template(self, track: Track):
        """Unit-norm template on the window bins: ``(A_k, Psi_k, valid, k0)``.

        ``h_k = A_k e^{-i Psi_k}`` on bins ``k0 .. k0 + n_window`` of the data
        (all of them, ``k0 = 0``, without a :attr:`span`), with
        ``4 df sum_k A_k^2 / S_k = 1``; ``valid`` is false (and ``A = 0``) when
        no track sample passes the mask.
        """
        xp = self.xp
        t, _, fdot, f_up, f_lo = self._knots(track)
        k0 = self._offset(f_lo)
        f_k = (self.first_bin + k0 + xp.arange(self.n_window)) * self.df
        inside = (f_k >= f_lo) & (f_k <= f_up[-1])

        i, w = self._bracket(f_up, k0)
        t_k = t[i] + w * (t[i + 1] - t[i])
        psi = 2 * np.pi * self.df * (xp.cumsum(t_k) - 0.5 * t_k)
        if self.amplitude == "spa":
            fdot_k = fdot[i] + w * (fdot[i + 1] - fdot[i])  # t is uniform: same weight
            a = 1.0 / xp.sqrt(xp.maximum(fdot_k, self.fdot_floor))
        else:
            a = xp.ones_like(f_k)
        a = xp.where(inside, a, 0.0)

        S = self._take(xp.asarray(self.psd_bins), k0)
        norm = 4 * self.df * xp.sum(a**2 / S)
        valid = norm > 0
        A = a / xp.sqrt(xp.where(valid, norm, 1.0))
        return A, psi, valid, k0

    def correlation_power(self, arr, track: Track):
        """``sum_channels |rho_j|^2`` on the delay grid, shape ``(n_fft,)``, and the
        template ``(A, Psi, valid, k0)``. Noise mean: 2 per channel."""
        xp = self.xp
        A, psi, valid, k0 = self.template(track)
        S = self._take(xp.asarray(self.psd_bins), k0)
        weight = 4 * self.df * A / S * xp.exp(1j * psi)
        rho = self.delay_transform(self._take(arr, k0) * weight[None, :])
        return xp.sum(xp.abs(rho) ** 2, axis=0), (A, psi, valid, k0)

    def reduce(self, power, template, n_channels):
        raise NotImplementedError

    def evaluate(self, arr, track: Track):
        """The statistic (0 when no track sample passes the mask)."""
        power, template = self.correlation_power(arr, track)
        value = self.reduce(power, template, arr.shape[0])
        return self.xp.where(template[2], value, 0.0)


class MatchedFilterStatistic(DelayCorrelationStatistic):
    """Phase- and delay-maximised matched filter, ``max_j |rho_j|``
    (channels in quadrature; the comparator of Cutler & Bustamante-Rosell)."""

    def reduce(self, power, template, n_channels):
        return self.xp.sqrt(self.xp.max(power))


class ExcessPowerStatistic(DelayCorrelationStatistic):
    """Cutler & Bustamante-Rosell's ``H`` (draft Eqs. 4, A4-A5).

    With ``c_j = sum_channels |rho_j|^2 - 2 n_channels`` (zero noise mean),

        H = max_{N in windows} max_i  sum_{j=i}^{i+N-1} c_j / sqrt(4 N Q(N) n_channels),

    where ``4 N Q(N)`` is the noise variance of one channel's window sum. The
    template's delay autocorrelation ``a_m = <h_0|h_m>`` sets
    ``Q(N) = sum_{|m|<N} (1 - |m|/N) |a_m|^2``. ``normalization="exact"``
    (default) computes it per template (one more FFT); ``"asymptotic"`` uses
    the large-window limit ``Q = sum_m |a_m|^2 = M sum_k' B_k'^2`` with
    ``b_k = 4 df A_k^2 / S_k`` and ``B`` its fold modulo ``M``
    (``n_fft sum_k b_k^2`` when the band has at most ``M`` bins), the draft's
    ``R = L / K`` for flat weights.
    The limit needs windows much longer than the autocorrelation length,
    ~1 / (template bandwidth): true for the draft's templates spanning
    3-8 mHz, not for a track that spans a few bins, whose window sums it
    then under-normalises.
    Windows wrap around the (circular) delay grid. Maximising over windows and
    delays makes the noise distribution non-Gaussian; calibrate thresholds on
    noise-only searches.

    Args:
        windows: Window lengths [s], converted to whole delay cells; default
            the draft's ``{10, 21, 86, 345}`` cells of ~200 s.
        normalization: ``"exact"`` or ``"asymptotic"``.
        **kwargs: As :class:`DelayCorrelationStatistic`.
    """

    def __init__(self, *args, windows=(2e3, 4.2e3, 1.72e4, 6.9e4), normalization="exact", **kwargs):
        super().__init__(*args, **kwargs)
        if normalization not in ("asymptotic", "exact"):
            raise ValueError(
                f'normalization must be "asymptotic" or "exact", not {normalization!r}.'
            )
        self.normalization = normalization
        cells = {min(max(1, int(round(w / self.delay_step))), self.n_fft) for w in windows}
        self.cells = tuple(sorted(cells))

    def window_q(self, A, k0=0):
        """``Q(N)`` for each window in :attr:`cells`, for the template amplitude
        ``A`` on the window at bin ``k0`` (:meth:`template`)."""
        xp = self.xp
        b = 4 * self.df * A**2 / self._take(xp.asarray(self.psd_bins), k0)
        if self.normalization == "asymptotic":
            q = self.n_fft * xp.sum(self.fold(b) ** 2)
            return [q for _ in self.cells]
        a2 = xp.abs(self.delay_transform(b)) ** 2
        s0 = xp.cumsum(a2)
        s1 = xp.cumsum(xp.arange(self.n_fft) * a2)
        return [2 * (s0[n - 1] - s1[n - 1] / n) - a2[0] for n in self.cells]

    def reduce(self, power, template, n_channels):
        xp = self.xp
        c = power - 2.0 * n_channels
        n_max = self.cells[-1]
        cs = xp.concatenate([xp.zeros(1), xp.cumsum(xp.concatenate([c, c[:n_max]]))])
        best = []
        for n, q in zip(self.cells, self.window_q(template[0], template[3])):
            sums = cs[n : n + self.n_fft] - cs[: self.n_fft]
            best.append(xp.max(sums) / xp.sqrt(4.0 * n * n_channels * xp.where(q > 0, q, 1.0)))
        return xp.max(xp.stack(best))
