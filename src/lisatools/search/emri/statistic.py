"""Semi-coherent detection statistics on SFT data.

A statistic scores a :class:`~.tracks.Track` against SFT data. Subclasses of
:class:`SemiCoherentStatistic` implement :meth:`~SemiCoherentStatistic.segment_terms`
(per-segment matched-filter magnitudes and template norms) and optionally
override :meth:`~SemiCoherentStatistic.evaluate` (how segments combine).
Which segments count is a separate, swappable :class:`SegmentMask`.

The backend (``"cpu"`` = NumPy, ``"jax"``) is fixed at instantiation; the SFT
array is passed in already on that backend (:meth:`~SemiCoherentStatistic.to_backend`)
so a JAX objective can take it as a traced argument instead of a baked constant.
"""

from dataclasses import dataclass
from functools import partial

import numpy as np

from ...utils.parallelbase import LISAToolsParallelModule
from .kernels import chirp_kernel, jax_chirp_kernel, tukey_chirp_kernel
from .tracks import Track


@dataclass(frozen=True)
class SegmentMask:
    """Keep segments where the linear-chirp model holds and the track is in band.

    A segment counts when ``f > f_min``, ``fdot > fdot_min`` and the cubic phase
    error of the linear chirp over a segment, ``|fddot| T^3 / 6``, is below
    ``dphi_max``. ``fddot`` is the track's own, or the central difference of
    ``fdot`` over segments. ``None`` disables a cut.
    """

    f_min: float = 10**-3.5
    fdot_min: float = 1e-13
    dphi_max: float = 1.0

    def __call__(self, track: Track, T_sft, xp):
        keep = xp.ones_like(track.f, dtype=bool)
        if self.f_min is not None:
            keep = keep & (track.f > self.f_min)
        if self.fdot_min is not None:
            keep = keep & (track.fdot > self.fdot_min)
        if self.dphi_max is not None:
            fddot = track.fddot if track.fddot is not None else xp.gradient(track.fdot, T_sft)
            keep = keep & (xp.abs(fddot * T_sft**3 / 6) < self.dphi_max)
        return xp.where(keep, 1.0, 0.0)


class SemiCoherentStatistic(LISAToolsParallelModule):
    """Base: ``0.5 * sum_{channel, segment} w_alpha |<d|h_alpha>|^2 / <h_alpha|h_alpha>``.

    Args:
        T_sft: Segment duration [s].
        psd: One-sided PSD callable ``psd(f)`` (NumPy or JAX arrays).
        mask: Segment selection.
        first_bin: Absolute bin index (``f / df``) of the SFT array's first column
            (non-zero when the SFTs keep only a band).
        force_backend: ``"cpu"`` (NumPy) or ``"jax"``.
    """

    def __init__(self, T_sft, psd, mask=SegmentMask(), first_bin=0, force_backend=None):
        super().__init__(force_backend=force_backend)
        self.T_sft = float(T_sft)
        self.psd = psd
        self.mask = mask
        self.first_bin = int(first_bin)

    @classmethod
    def supported_backends(cls):
        return [cls._BACKEND_PREFIX + "_" + t for t in ("cpu", "jax")]

    @classmethod
    def for_data(cls, data, **kwargs):
        """Statistic on the grid, band and PSD of an :class:`~.data.SFTData`."""
        return cls(data.T_sft, data.psd, first_bin=data.first_bin, **kwargs)

    @property
    def is_jax(self) -> bool:
        return self.backend.name.endswith("jax")

    def to_backend(self, sfts):
        """Move an SFT array ``(n_channels, n_segments, n_bins)`` to this backend."""
        return self.xp.asarray(np.asarray(sfts), dtype=complex)

    def segment_terms(self, sfts, track: Track):
        """Per-segment ``|<d|h>|``, shape ``(n_channels, n_segments)``, and ``<h|h>``,
        shape ``(n_segments,)``, for a unit-amplitude template."""
        raise NotImplementedError

    def evaluate(self, sfts, track: Track):
        """The statistic (0 when the mask removes every segment)."""
        xp = self.xp
        dh_abs, hh = self.segment_terms(sfts, track)
        w = self.mask(track, self.T_sft, xp)
        total = 0.5 * xp.sum(w * dh_abs**2 / hh)
        return xp.where(xp.sum(w) != 0.0, total, 0.0)

    def per_segment(self, sfts, track: Track):
        """Unmasked per-segment contributions ``|<d|h>|^2 / <h|h>``, summed over channels."""
        dh_abs, hh = self.segment_terms(sfts, track)
        return self.xp.sum(dh_abs**2, axis=0) / hh


class FresnelStatistic(SemiCoherentStatistic):
    """Single-harmonic semi-coherent statistic with the linear-chirp kernel.

    Per segment ``alpha``, the template is a unit-amplitude linear chirp with
    the track's ``(f_alpha, fdot_alpha)``; its overlap with the data uses the
    ``2P+1`` SFT bins around ``floor(f_alpha T_sft)``:

        c_alpha = df sum_k conj(d_alpha[k]) / S(k df) K(f_alpha - k df, fdot_alpha, T_sft)
        |<d|h_alpha>| = 2 |c_alpha|,   <h_alpha|h_alpha> = T_sft / (2 S(f_alpha))

    (Speri, Tenorio, Chapman-Bird & Gerosa 2026, arXiv:2510.20891). With the
    defaults this reproduces ``emrisearch.jax_utils.det_stat``.

    Args:
        P: Half-width of the bin window.
        kernel_window_alpha: Tukey taper of the *kernel*; 0 (default) is the
            rectangular kernel of the reference implementation, used even
            though the SFTs are Tukey-windowed. Setting it to the data's taper
            does *not* match the filter better: the noise is windowed too, so
            for white noise the rectangular kernel gives ~3 % more SNR^2 than
            the Tukey one at ``alpha = 0.2`` (it costs 7 kernels, not 1). The
            ``<h|h>`` normalisation stays the rectangular ``T / (2 S)`` either
            way; that is a constant factor (``int w^2 = (1 - 5 alpha / 8) T``
            for the data's window) that rescales the statistic without changing
            any ranking.
        small_drift: Use the double-precision small-``fdot`` kernel series
            (:func:`~.kernels.chirp_kernel`); slower, ~1e-10 more accurate.
        n_bins: Number of stored SFT bins. When given (:meth:`for_data` does),
            ``S`` on those bins is tabulated once instead of being evaluated
            inside every call; the values are identical.
    """

    def __init__(
        self,
        T_sft,
        psd,
        P=100,
        kernel_window_alpha=0.0,
        small_drift=False,
        n_bins=None,
        **kwargs,
    ):
        super().__init__(T_sft, psd, **kwargs)
        self.P = int(P)
        self.kernel_window_alpha = float(kernel_window_alpha)
        self.small_drift = bool(small_drift)
        self.psd_bins = None
        if n_bins is not None:
            k = self.first_bin + np.arange(int(n_bins))
            self.psd_bins = np.asarray(self.psd(k / self.T_sft), dtype=float)

    @classmethod
    def for_data(cls, data, **kwargs):
        kwargs.setdefault("n_bins", data.sfts.shape[-1])
        return super().for_data(data, **kwargs)

    def band(self, n_bins):
        """``(f_lo, f_hi)`` [Hz] a track can occupy with all ``2P+1`` bins stored.

        SFTs starting at bin 0 count as covering down to 0 Hz: the bins the
        window would need below it are negative frequencies, which the
        statistic never uses.
        """
        df = 1.0 / self.T_sft
        f_lo = 0.0 if self.first_bin == 0 else (self.first_bin + self.P) * df
        return f_lo, (self.first_bin + n_bins - 1 - self.P) * df

    def _kernel(self, f0, f1):
        base = partial(
            jax_chirp_kernel if self.is_jax else chirp_kernel, small_drift=self.small_drift
        )
        if self.kernel_window_alpha > 0.0:
            return tukey_chirp_kernel(f0, f1, self.T_sft, self.kernel_window_alpha, kernel=base)
        return base(f0, f1, self.T_sft)

    def segment_terms(self, sfts, track: Track):
        xp = self.xp
        T, df = self.T_sft, 1.0 / self.T_sft
        n_segments, n_bins = sfts.shape[-2], sfts.shape[-1]
        f, fdot = track.f, track.fdot

        k = (f * T).astype(int)[None, :] + xp.arange(-self.P, self.P + 1)[:, None]
        col = k - self.first_bin
        valid = (k >= 0) & (col >= 0) & (col < n_bins)
        col_safe = xp.clip(col, 0, n_bins - 1)
        d = sfts[:, xp.arange(n_segments)[None, :], col_safe]  # (n_channels, 2P+1, n_segments)
        kernel = self._kernel(f[None, :] - k * df, fdot[None, :])
        if self.psd_bins is not None and self.psd_bins.size == n_bins:
            S_k = xp.asarray(self.psd_bins)[col_safe]
        else:
            S_k = self.psd(k * df)
        weight = xp.where(valid, df * kernel / S_k, 0.0)
        c = xp.sum(d.conj() * weight[None], axis=-2)

        hh = 0.5 * T / self.psd(f)
        return 2 * xp.abs(c), hh
