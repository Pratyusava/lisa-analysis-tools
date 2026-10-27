"""EMRI injections for the semi-coherent search.

:class:`FEWInjector` produces the injections of Speri et al. (2026,
arXiv:2510.20891; EMRI-Search's ``create_signal`` and
``generate_emri_signal_and_sfts``): a FEW ``FastKerrEccentricEquatorialFlux``
waveform through LAT's LISA response
(:func:`~lisatools.sources.emri.response.get_emri_response_wrapper`, TDI 2nd
generation, equal-arm orbits). The inspiral is integrated back from plunge
(:func:`~.tracks.backward_inspiral`), the signal is scaled to a target SNR
(LAT's :func:`~lisatools.diagnostic.snr`) and the distance rescaled to match.

The SNR sums every rFFT bin of the injected duration (EMRI-Search's
``get_snr``). With LAT's analytic TDI2 PSD
(:class:`~.psd.SensitivityPSD` default) the bins near the TDI transfer zeros
(multiples of ~0.03 Hz) have ``S(f)`` near zero and dominate that sum, so pass
``snr_band=(0, F_TDI_ZERO_GUARD)`` (0.025 Hz) there; EMRI-Search's tabulated PSD
(``SensitivityPSD.from_file(..., clip_to_first=True)``) has no such zeros.
"""

import warnings
from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import numpy as np

from ...diagnostic import snr as lat_snr
from ...domains import FDSettings
from ...utils.constants import YRSID_SI
from ...utils.utility import asnumpy
from .data import SFTData, gaussian_noise
from .tracks import backward_inspiral, kerr_ecc_eq_inspiral

ANGLE_NAMES = ("qS", "phiS", "qK", "phiK", "Phi_phi0", "Phi_theta0", "Phi_r0")
DEFAULT_ANGLES = dict(qS=np.pi / 3, phiS=0.0, qK=np.pi / 3, phiK=0.0)
_CHANNEL_INDEX = {"A": 0, "E": 1, "T": 2}
F_TDI_ZERO_GUARD = 0.025
"""[Hz] Below the first TDI2 transfer zero (~0.03 Hz); see the module docs."""


def random_angles(rng=None) -> dict:
    """Isotropic sky and spin, uniform phases, drawn in EMRI-Search's order.

    Args:
        rng: ``np.random.Generator`` or seed; ``None`` uses NumPy's global RNG,
            which reproduces EMRI-Search's ``randomize=True`` after
            ``np.random.seed``.
    """
    rng = np.random if rng is None else np.random.default_rng(rng)
    draws = {}
    for name in ANGLE_NAMES:
        if name in ("qS", "qK"):
            draws[name] = float(np.arccos(rng.uniform(-1, 1)))
        else:
            draws[name] = float(rng.uniform(0, 2 * np.pi))
    return draws


def band_snr(data, dt, psd, band=(0.0, None)):
    """Optimal SNR of real time series ``data`` (``(N,)`` or ``(n_channels, N)``),
    over the rFFT bins with ``band[0] <= f <= band[1]`` (``None``: no upper limit).

    Channels add in quadrature, each whitened by the same ``psd``.
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    f = np.fft.rfftfreq(data.shape[-1], dt)
    keep = (f >= band[0]) & (f <= (np.inf if band[1] is None else band[1]))
    settings = FDSettings(int(keep.sum()), float(f[1] - f[0]), force_backend="cpu")
    psd_f = psd(f[keep])
    snr2 = 0.0
    for channel in data:
        h = np.fft.rfft(channel)[keep] * dt
        snr2 += lat_snr(h, basis_settings=settings, psd=psd_f) ** 2
    return float(np.sqrt(snr2))


@dataclass
class EMRIInjection:
    """A scaled injection, the parameters that produce it, and optionally one
    fixed noise realisation.

    The data streams of a search are built from it: :meth:`timeseries` (signal,
    noise, or both) and :meth:`sft_data` (their SFTs, optionally only the first
    ``n_segments``). :meth:`with_noise` freezes a noise draw, so every stream
    taken from the returned injection (each duration, with or without the
    signal) shares it.

    Attributes:
        signal: TDI time series, shape ``(n_channels, N)``.
        dt: Sampling cadence [s].
        params: Source parameters: ``m1, m2, a, T_pl, e_f, x_f`` (search
            parametrisation), ``p0, e0, x0`` (initial conditions), ``dist``
            [Gpc] at the target SNR, and the angles.
        snr: SNR of the generated signal before trimming (the target).
        channels: Channel names of ``signal``'s rows.
        psd: PSD the SNR was scaled with; the default for noise and whitening.
        noise: Fixed noise realisation, shape of ``signal``; ``None`` until
            :meth:`with_noise`.
    """

    signal: np.ndarray
    dt: float
    params: dict
    snr: float
    channels: str = "A"
    psd: Optional[Callable] = field(default=None, repr=False)
    noise: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        if self.noise is not None:
            self.noise = self._check_noise(self.noise)

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.signal.shape[-1]) * self.dt

    def n_segments(self, T_sft) -> int:
        """Whole ``T_sft`` segments in the signal (the most :meth:`sft_data` keeps)."""
        return self.signal.shape[-1] // int(T_sft / self.dt)

    def draw_noise(self, rng=None, psd=None) -> np.ndarray:
        """A fresh :func:`~.data.gaussian_noise` realisation shaped like ``signal``.

        Args:
            rng: As in :func:`~.data.gaussian_noise` (``None``: NumPy's global RNG).
            psd: Noise PSD; ``None`` uses :attr:`psd`.
        """
        psd = self._psd(psd)
        n_channels, n_samples = self.signal.shape
        return gaussian_noise(n_samples, self.dt, psd, n_channels=n_channels, rng=rng)

    def with_noise(self, rng=None, noise=None, psd=None) -> "EMRIInjection":
        """Copy of this injection carrying one fixed noise realisation.

        Args:
            rng: Generator / seed of the draw (:meth:`draw_noise`).
            noise: Use this realisation instead of drawing one.
            psd: Noise PSD of the draw; ``None`` uses :attr:`psd`.
        """
        if noise is None:
            noise = self.draw_noise(rng, psd)
        return replace(self, noise=noise)

    def timeseries(self, signal=True, noise=True, rng=None, psd=None) -> np.ndarray:
        """Data time series, shape ``(n_channels, N)``.

        Args:
            signal: Include the injected signal.
            noise: ``True`` adds :attr:`noise` when set, else a fresh draw from
                ``rng`` (and ``psd``); ``False`` none; an array adds that
                realisation.
        """
        if not signal and noise is False:
            raise ValueError("Nothing to return: signal=False and noise=False.")
        out = self.signal if signal else np.zeros_like(self.signal)
        if noise is True:
            noise = self.noise if self.noise is not None else self.draw_noise(rng, psd)
        if noise is not False:
            out = out + self._check_noise(noise)
        return out

    def sft_data(
        self,
        T_sft,
        psd=None,
        signal=True,
        noise=True,
        rng=None,
        n_segments=None,
        window_alpha=0.2,
        **kwargs,
    ) -> SFTData:
        """SFTs of :meth:`timeseries` (``signal`` / ``noise`` / ``rng`` as there).

        Args:
            T_sft: Segment duration [s].
            psd: PSD of a fresh noise draw and of the whitening; ``None`` uses
                :attr:`psd`.
            n_segments: Keep the first ``n_segments`` segments only (the data
                stream ending there); ``None`` keeps them all.
            **kwargs: Forwarded to :meth:`~.data.SFTData.from_timeseries`
                (``min_freq``, ``max_freq``).
        """
        psd = self._psd(psd)
        data = self.timeseries(signal=signal, noise=noise, rng=rng, psd=psd)
        return SFTData.from_timeseries(
            data,
            self.dt,
            T_sft,
            psd,
            window_alpha=window_alpha,
            n_segments=n_segments,
            **kwargs,
        )

    def _psd(self, psd):
        psd = self.psd if psd is None else psd
        if psd is None:
            raise ValueError("No PSD: pass psd= (the injection was made without one).")
        return psd

    def _check_noise(self, noise):
        noise = np.atleast_2d(np.asarray(noise, dtype=float))
        if noise.shape != self.signal.shape:
            raise ValueError(f"noise has shape {noise.shape}, signal {self.signal.shape}.")
        return noise


class FEWInjector:
    """FEW Kerr eccentric equatorial injections with LAT's TDI response.

    The response generator is built on first use and cached process-wide by
    :func:`~lisatools.sources.emri.response.get_emri_response_wrapper` (one
    generator is ~2.3 GiB on CPU); it is not pickled.

    Args:
        T: Duration of the generated signal [yr] (EMRI-Search's ``T_snr``).
        dt: Sampling cadence [s].
        channels: TDI channels to keep, any of ``"AET"`` (e.g. ``"A"``, ``"AE"``).
        tdi: TDI generation.
        order: Lagrange interpolation order of the response.
        t_buffer: Response start-up buffer [s] (EMRI-Search's ``t0``).
        eps: FEW mode-selection threshold.
        err: Backward-integration tolerance.
        orbits: LAT orbits; ``None`` uses equal-arm orbits.
        force_backend: FEW / response backend.
    """

    def __init__(
        self,
        T,
        dt=5.0,
        channels="A",
        tdi="2nd generation",
        order=25,
        t_buffer=3e4,
        eps=1e-2,
        err=1e-11,
        orbits=None,
        force_backend="cpu",
    ):
        if not channels or any(c not in _CHANNEL_INDEX for c in channels):
            raise ValueError(f"channels must be drawn from 'AET', not {channels!r}.")
        self.T, self.dt = float(T), float(dt)
        self.channels = channels
        self.tdi, self.order, self.t_buffer = tdi, int(order), float(t_buffer)
        self.eps, self.err = float(eps), float(err)
        self.orbits = orbits
        self.force_backend = force_backend

    @property
    def n_samples(self) -> int:
        return int(self.T * YRSID_SI / self.dt)

    @property
    def generator(self):
        if getattr(self, "_generator", None) is None:
            from ...response.tdiconfig import TDIConfig
            from ...sources.emri.response import get_emri_response_wrapper

            self._generator = get_emri_response_wrapper(
                Tobs=self.n_samples * self.dt,
                dt=self.dt,
                t_start=0.0,
                tdi_config=TDIConfig(self.tdi),
                tdi_chan="AET",
                order=self.order,
                t_buffer=self.t_buffer,
                orbits=self.orbits,
                force_backend=self.force_backend,
                special_frame=False,
            )
        return self._generator

    @property
    def traj(self):
        if getattr(self, "_traj", None) is None:
            self._traj = kerr_ecc_eq_inspiral()
        return self._traj

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_generator", None)
        state.pop("_traj", None)
        return state

    def initial_conditions(self, m1, m2, a, T_pl, e_f, x_f=1.0):
        """``(p0, e0, x0)`` a time ``T_pl`` [yr] before plunge."""
        out = backward_inspiral(self.traj, m1, m2, a, T_pl, e_f, x_f, err=self.err)
        return float(out[1][-1]), float(out[2][-1]), float(out[3][-1])

    def waveform(self, m1, m2, a, p0, e0, x0, dist, angles):
        """TDI channels ``self.channels`` at ``dist`` [Gpc], shape ``(n_channels, N)``."""
        args = [angles[name] for name in ANGLE_NAMES]
        out = self.generator(m1, m2, a, p0, e0, x0, dist, *args, eps=self.eps)
        return np.stack([asnumpy(out[_CHANNEL_INDEX[c]]) for c in self.channels])

    def __call__(
        self,
        params,
        snr,
        psd,
        T_data=None,
        angles=None,
        rng=None,
        dist=1.0,
        snr_band=(0.0, None),
    ) -> EMRIInjection:
        """Injection of ``params = (m1, m2, a, T_pl, e_f[, x_f])`` at SNR ``snr``.

        The SNR is that of the full ``T``-long signal; the scaled signal is then
        trimmed to ``T_data`` [yr] (EMRI-Search's ``T_snr`` / ``T_data``).

        Args:
            psd: One-sided PSD callable for the SNR.
            angles: ``{name: value}`` overriding sky / spin / phases; missing
                ones are drawn by :func:`random_angles` from ``rng`` when
                ``angles`` is ``None``, else take :data:`DEFAULT_ANGLES` (phases 0).
            dist: Distance the unscaled waveform is generated at [Gpc].
            snr_band: ``(f_min, f_max)`` of the SNR sum [Hz] (see module docs).
        """
        m1, m2, a, T_pl, e_f, *rest = params
        x_f = rest[0] if rest else 1.0
        if angles is None:
            angles = random_angles(rng)
        else:
            angles = {**{n: 0.0 for n in ANGLE_NAMES}, **DEFAULT_ANGLES, **angles}
        p0, e0, x0 = self.initial_conditions(m1, m2, a, T_pl, e_f, x_f)
        h = self.waveform(m1, m2, a, p0, e0, x0, dist, angles)

        snr_raw = band_snr(h, self.dt, psd, snr_band)
        if snr_band[1] is None:
            high = band_snr(h, self.dt, psd, (F_TDI_ZERO_GUARD, None))
            if high**2 > 0.1 * snr_raw**2:
                warnings.warn(
                    f"{high**2 / snr_raw**2:.0%} of the SNR^2 comes from "
                    f"f > {F_TDI_ZERO_GUARD * 1e3:g} mHz, "
                    "likely PSD zeros of the TDI transfer function; set snr_band.",
                    stacklevel=2,
                )
        scale = snr / snr_raw
        n_keep = h.shape[-1]
        if T_data is not None:  # samples with t < T_data, as EMRI-Search trims
            n_keep = int(np.count_nonzero(np.arange(n_keep) * self.dt < T_data * YRSID_SI))
        out_params = dict(
            m1=m1,
            m2=m2,
            a=a,
            T_pl=T_pl,
            e_f=e_f,
            x_f=x_f,
            p0=p0,
            e0=e0,
            x0=x0,
            dist=dist / scale,
            **angles,
        )
        return EMRIInjection(
            scale * h[:, :n_keep], self.dt, out_params, float(snr), self.channels, psd
        )
