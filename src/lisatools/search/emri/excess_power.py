"""Coherent, delay-scanning statistics on a band-limited FFT (:class:`~.data.FDData`).

A track becomes one full-length, unit-norm template; one inverse FFT gives its
complex correlation with the data at every time delay ``tau_j``,

    rho_j = <d | h_{tau_j}> = 4 df sum_k conj(d_k) h_k e^{-2 pi i f_k tau_j} / S_k,

and a statistic reduces ``|rho_j|^2`` over the delays:

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
        t_alpha: Times of the track samples [s] (the track model's grid).
        first_bin: Absolute bin index (``f / df``) of the data's first bin.
        n_bins: Number of stored bins.
        delay_step: Target delay resolution [s]; the inverse FFT has
            ``max(n_bins, T / delay_step)`` points (rounded up to a fast size),
            so the actual step :attr:`delay_step` is at most ``1 / bandwidth``.
        amplitude: ``"spa"`` (``|h(f)| ~ fdot^{-1/2}``) or ``"flat"``.
        mask: Which track samples enter the template (in band, chirping up).
            The default drops the semi-coherent linear-chirp cut, which does
            not apply to a coherent template.
        fdot_floor: Smallest ``fdot`` [Hz/s] in the ``"spa"`` amplitude.
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
        self.first_bin = int(first_bin)
        self.n_bins = int(n_bins)
        self.amplitude = amplitude
        self.mask = mask
        self.fdot_floor = float(fdot_floor)
        self.freqs = (self.first_bin + np.arange(self.n_bins)) * self.df
        self.psd_bins = np.asarray(psd(self.freqs), dtype=float)
        self.n_fft = next_fast_len(max(self.n_bins, int(np.ceil(self.T / float(delay_step)))))

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

    def _cummax(self, x):
        if self.is_jax:
            import jax

            return jax.lax.cummax(x, axis=0)
        return np.maximum.accumulate(x)

    def template(self, track: Track):
        """Unit-norm template on the data bins: ``(A_k, Psi_k, valid)``.

        ``h_k = A_k e^{-i Psi_k}`` with ``4 df sum_k A_k^2 / S_k = 1``; ``valid``
        is false (and ``A = 0``) when no track sample passes the mask.
        """
        xp = self.xp
        f_k = xp.asarray(self.freqs)
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
        inside = (f_k >= f_lo) & (f_k <= f_up[-1])

        t_k = xp.interp(f_k, f_up, t)
        psi = 2 * np.pi * self.df * (xp.cumsum(t_k) - 0.5 * t_k)
        if self.amplitude == "spa":
            fdot_k = xp.interp(t_k, t, fdot)
            a = 1.0 / xp.sqrt(xp.maximum(fdot_k, self.fdot_floor))
        else:
            a = xp.ones_like(f_k)
        a = xp.where(inside, a, 0.0)

        S = xp.asarray(self.psd_bins)
        norm = 4 * self.df * xp.sum(a**2 / S)
        valid = norm > 0
        A = a / xp.sqrt(xp.where(valid, norm, 1.0))
        return A, psi, valid

    def correlation_power(self, arr, track: Track):
        """``sum_channels |rho_j|^2`` on the delay grid, shape ``(n_fft,)``, and the
        template ``(A, Psi, valid)``. Noise mean: 2 per channel."""
        xp = self.xp
        A, psi, valid = self.template(track)
        weight = 4 * self.df * A / xp.asarray(self.psd_bins) * xp.exp(1j * psi)
        rho = xp.fft.ifft(arr * weight[None, :], n=self.n_fft, axis=-1) * self.n_fft
        return xp.sum(xp.abs(rho) ** 2, axis=0), (A, psi, valid)

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
    the large-window limit ``Q = n_fft sum_k b_k^2`` with
    ``b_k = 4 df A_k^2 / S_k``, the draft's ``R = L / K`` for flat weights.
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

    def window_q(self, A):
        """``Q(N)`` for each window in :attr:`cells`."""
        xp = self.xp
        b = 4 * self.df * A**2 / xp.asarray(self.psd_bins)
        if self.normalization == "asymptotic":
            q = self.n_fft * xp.sum(b**2)
            return [q for _ in self.cells]
        a2 = xp.abs(xp.fft.ifft(b, n=self.n_fft) * self.n_fft) ** 2
        s0 = xp.cumsum(a2)
        s1 = xp.cumsum(xp.arange(self.n_fft) * a2)
        return [2 * (s0[n - 1] - s1[n - 1] / n) - a2[0] for n in self.cells]

    def reduce(self, power, template, n_channels):
        xp = self.xp
        c = power - 2.0 * n_channels
        n_max = self.cells[-1]
        cs = xp.concatenate([xp.zeros(1), xp.cumsum(xp.concatenate([c, c[:n_max]]))])
        best = []
        for n, q in zip(self.cells, self.window_q(template[0])):
            sums = cs[n : n + self.n_fft] - cs[: self.n_fft]
            best.append(xp.max(sums) / xp.sqrt(4.0 * n * n_channels * xp.where(q > 0, q, 1.0)))
        return xp.max(xp.stack(best))
