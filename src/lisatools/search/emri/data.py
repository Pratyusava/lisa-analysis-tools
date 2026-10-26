"""SFT data for the semi-coherent EMRI search.

:class:`SFTData` pairs a LAT :class:`~lisatools.domains.STFTSignal` (made by
:meth:`~lisatools.domains.TDSignal.stft`, ``dt * rfft(window * segment)`` per
segment, layout ``(channel, segment, bin)``) with the PSD that whitens it. The
grid (segment times, bin frequencies, active band) is the STFT settings' own.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal.windows import tukey

from ...domains import STFTSignal, TDSettings, TDSignal, get_stft_settings
from ...utils.utility import asnumpy, generate_noise_fd


@dataclass
class SFTData:
    """SFTs of the data plus the PSD that whitens them.

    Attributes:
        signal: The STFT, shape ``(n_channels, n_segments, n_bins)`` over the
            settings' active band; segment duration ``signal.dt``.
        psd: One-sided PSD callable ``psd(f)`` (see :mod:`.psd`), the same for
            every channel.
        dt: Sampling cadence of the underlying time series [s].
        window_alpha: Tukey taper fraction the SFTs were made with.
    """

    signal: STFTSignal
    psd: Callable
    dt: float
    window_alpha: float = 0.2

    @property
    def sfts(self) -> np.ndarray:
        return self.signal.arr

    @property
    def T_sft(self) -> float:
        return float(self.signal.dt)

    @property
    def first_bin(self) -> int:
        """Absolute index (``f / df``) of ``sfts[..., 0]``."""
        return int(self.signal.ind_min)

    @property
    def n_segments(self) -> int:
        return int(self.signal.NT)

    @property
    def freqs(self) -> np.ndarray:
        return asnumpy(self.signal.f_arr)

    @property
    def t_alpha(self) -> np.ndarray:
        """Segment start times [s]."""
        return asnumpy(self.signal.t_arr)

    @classmethod
    def from_timeseries(cls, data, dt, T_sft, psd, window_alpha=0.2, min_freq=0.0, max_freq=None):
        """Segment a time series into Tukey-windowed SFTs.

        Grid from :func:`~lisatools.domains.get_stft_settings`: whole segments
        only, ``T_sft`` snapped down to a multiple of ``dt``;
        ``min_freq``/``max_freq`` keep only that band of bins.

        Args:
            data: Real time series, shape ``(N,)`` or ``(n_channels, N)``.
            dt: Sampling cadence [s].
            T_sft: Segment duration [s].
            psd: One-sided PSD callable.
            window_alpha: Tukey taper fraction per segment.
        """
        data = np.atleast_2d(np.asarray(data, dtype=float))
        td = TDSignal(data, TDSettings(data.shape[-1], dt, force_backend="cpu"))
        settings = get_stft_settings(
            np.arange(data.shape[-1]) * dt,
            T_sft,
            min_freq=min_freq,
            max_freq=max_freq,
            force_backend="cpu",
        )
        nperseg = settings.get_nperseg(dt)
        signal = td.stft(settings=settings, window=tukey(nperseg, window_alpha))
        return cls(signal, psd=psd, dt=dt, window_alpha=window_alpha)

    def with_sfts(self, sfts) -> "SFTData":
        """Same grid and PSD, different SFTs (e.g. a noise-only realisation)."""
        return SFTData(
            STFTSignal(np.asarray(sfts), self.signal.settings), self.psd, self.dt, self.window_alpha
        )


def gaussian_noise(n_samples, dt, psd, n_channels=1, rng=None):
    """Stationary Gaussian noise with one-sided PSD ``psd``, shape ``(n_channels, N)``.

    One :func:`~lisatools.utils.utility.generate_noise_fd` draw per channel,
    inverse-FFT'd. Same normalisation as EMRI-Search's ``generate_noise``.

    Args:
        rng: ``np.random.Generator`` or seed; ``None`` uses NumPy's global RNG
            (seed with ``np.random.seed``).
    """
    if rng is not None:
        rng = np.random.default_rng(rng)
    df = 1.0 / (n_samples * dt)
    fd = [generate_noise_fd(n_samples // 2 + 1, df, func=psd, rng=rng) for _ in range(n_channels)]
    return np.fft.irfft(np.asarray(fd), n=n_samples, axis=-1) / dt
