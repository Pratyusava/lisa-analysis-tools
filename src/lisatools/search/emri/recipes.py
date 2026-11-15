"""Injection and noise files: small JSON recipes that regenerate the data, with a checksum.

A campaign's injections and background are not stored as samples (a year at 5 s is 50 MB per
channel) but as what produces them:

- :class:`NoiseFile`: a seed, the length, cadence, channels and the PSD it was drawn from.
  :meth:`NoiseFile.generate` redraws the noise (:func:`~.data.gaussian_noise`).
- :class:`InjectionFile`: the source parameters (with the distance that gives the target SNR
  and the initial conditions ``p0, e0, x0``, so neither the backward integration nor the SNR
  scaling is redone), the FEW / response settings, and the seed of the noise it is searched
  on. :meth:`InjectionFile.injection` regenerates both.

Regeneration depends on more than the recorded numbers: NumPy's random streams, LAT's noise
and PSD code, FEW and the response can change between versions. Each file therefore carries a
:func:`checksum` of what it produced (each channel's power and a few hundred samples) and the
versions it was made with; regenerated data that does not match raises
:class:`RecipeMismatch` instead of silently searching different data.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

import numpy as np

from .data import gaussian_noise
from .injection import ANGLE_NAMES, EMRIInjection, FEWInjector

FORMAT = 1
CHECKSUM_POINTS = 256
CHECKSUM_RTOL = 1e-10


class RecipeMismatch(ValueError):
    """Regenerated data differ from what the file recorded, or the file does not fit the
    configuration it is searched with."""


# ----------------------------------------------------------------------------- checks
def checksum(samples, n_points=CHECKSUM_POINTS) -> Dict[str, Any]:
    """A small summary of ``samples`` (shape ``(n_channels, N)``) that regenerated data must
    reproduce: the shape, each channel's power, and ``n_points`` evenly spaced samples."""
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    index = np.unique(np.linspace(0, samples.shape[1] - 1, n_points).astype(int))
    return dict(
        shape=list(samples.shape),
        power=[float(np.sum(channel**2)) for channel in samples],
        index=index.tolist(),
        samples=samples[:, index].tolist(),
    )


def verify_checksum(samples, recorded, what, made_with=None, rtol=CHECKSUM_RTOL):
    """Raise :class:`RecipeMismatch` unless ``samples`` match the ``recorded`` checksum.

    Each channel's power must agree to ``rtol`` (relative), and each recorded sample to
    ``rtol`` times the channel's rms. Identical code gives identical data; the tolerance
    only allows for rounding differences between machines.

    Args:
        what: Description of the data, for the error message.
        made_with: The versions the file was made with, for the error message.
    """
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    if list(samples.shape) != list(recorded["shape"]):
        raise RecipeMismatch(f"{what}: shape {list(samples.shape)}, recorded {recorded['shape']}.")
    recorded_power = np.asarray(recorded["power"])
    power = np.sum(samples**2, axis=1)
    rms = np.sqrt(recorded_power / samples.shape[1])
    rms = np.where(rms > 0, rms, 1.0)[:, None]  # a zero channel: compare absolutely
    offsets = np.abs(samples[:, recorded["index"]] - np.asarray(recorded["samples"])) / rms
    largest_offset = float(np.max(offsets, initial=0.0))

    power_matches = np.allclose(power, recorded_power, rtol=rtol, atol=0.0)
    if not power_matches or largest_offset > rtol:
        raise RecipeMismatch(
            f"{what} regenerates differently from when it was made (made with "
            f"{_format_versions(made_with)}; now {_format_versions(versions())}): power "
            f"{power.tolist()} vs {recorded_power.tolist()}, largest sample offset "
            f"{largest_offset:.2e} rms. Remake the file, or rerun with the versions it was "
            "made with."
        )


def versions() -> Dict[str, str]:
    """The versions of the packages regeneration depends on."""
    import lisatools

    found = {"numpy": np.__version__, "lisatools": str(getattr(lisatools, "__version__", "?"))}
    try:
        from importlib.metadata import version

        found["fastemriwaveforms"] = version("fastemriwaveforms")
    except Exception:  # pragma: no cover - FEW not installed
        pass
    return found


def _format_versions(package_versions) -> str:
    if not package_versions:
        return "unknown versions"
    return ", ".join(f"{name} {number}" for name, number in sorted(package_versions.items()))


def psd_identity(config) -> Dict[str, Any]:
    """What identifies a configuration's PSD: the file's name and content, or LAT's
    analytic one; and the channel it is evaluated for."""
    data = config.data
    channel = data.channels[0]
    if not data.psd_file:
        return dict(analytic="LAT TDI2", channel=channel)
    with open(data.psd_file, "rb") as fh:
        content_hash = hashlib.sha1(fh.read()).hexdigest()
    return dict(file=os.path.basename(data.psd_file), sha1=content_hash, channel=channel)


def _check_matches_config(recipe, config, path):
    """The file's cadence, channels and PSD must be the configuration's."""
    data = config.data
    if recipe.dt != data.dt or recipe.channels != data.channels:
        raise RecipeMismatch(
            f"{path}: dt {recipe.dt}, channels {recipe.channels!r}; the configuration has "
            f"dt {data.dt}, channels {data.channels!r}."
        )
    config_psd = psd_identity(config)
    if recipe.psd != config_psd:
        raise RecipeMismatch(
            f"{path} was made with the PSD {recipe.psd}, the configuration has {config_psd}."
        )


def _draw_noise(config, seed, n_samples, n_channels) -> np.ndarray:
    """The one way noise is drawn from a seed, at creation and at every regeneration."""
    rng = np.random.default_rng(seed)
    return gaussian_noise(n_samples, config.data.dt, config.psd(), n_channels, rng)


# ----------------------------------------------------------------------------- files
@dataclass
class NoiseFile:
    """A noise-only stream: :func:`~.data.gaussian_noise` with ``seed`` (see module docs).

    Attributes:
        label: Name of the stream.
        seed: Seed of ``np.random.default_rng``.
        n_samples, dt, channels: Its length, cadence [s] and TDI channels.
        psd: :func:`psd_identity` of the PSD it is drawn from.
        checksum: :func:`checksum` of the noise.
        made_with: :func:`versions` at creation.
    """

    label: str
    seed: int
    n_samples: int
    dt: float
    channels: str
    psd: Dict[str, Any]
    checksum: Dict[str, Any] = field(repr=False)
    made_with: Dict[str, str] = field(default_factory=dict)
    kind: str = "noise"
    format: int = FORMAT

    @classmethod
    def make(cls, config, label, seed, T_data) -> "NoiseFile":
        """The recipe of a ``T_data`` [yr] stream drawn with ``seed`` (draws it once, for
        the checksum)."""
        n_samples = config.n_samples(T_data)
        noise = _draw_noise(config, seed, n_samples, len(config.data.channels))
        return cls(
            label=label,
            seed=int(seed),
            n_samples=n_samples,
            dt=config.data.dt,
            channels=config.data.channels,
            psd=psd_identity(config),
            checksum=checksum(noise),
            made_with=versions(),
        )

    def generate(self, config, path="noise file") -> np.ndarray:
        """The noise, shape ``(n_channels, n_samples)``, checked against the file."""
        _check_matches_config(self, config, path)
        noise = _draw_noise(config, self.seed, self.n_samples, len(self.channels))
        verify_checksum(noise, self.checksum, f"{path} (seed {self.seed})", self.made_with)
        return noise

    def save(self, path):
        _save(self, path)

    @classmethod
    def load(cls, path) -> "NoiseFile":
        return _load(cls, path, "noise")


@dataclass
class InjectionFile:
    """An injection and the noise it is searched on (see module docs).

    Attributes:
        label: Name of the stream.
        params: Source parameters (:attr:`~.injection.EMRIInjection.params`: ``m1, m2, a,
            T_pl, e_f, x_f``, ``p0, e0, x0``, ``dist`` [Gpc] at the target SNR, the angles).
        snr: The target SNR.
        n_samples, dt, channels: The kept length, cadence [s] and TDI channels.
        generator: :class:`~.injection.FEWInjector` settings (``T`` [yr], ``tdi``,
            ``order``, ``t_buffer``, ``eps``, ``force_backend``).
        signal_checksum: :func:`checksum` of the signal.
        noise_seed, noise_checksum: The noise realisation the injection is searched on.
        psd: :func:`psd_identity` of the PSD the SNR was scaled with and the noise drawn from.
        made_with: :func:`versions` at creation.
    """

    label: str
    params: Dict[str, float]
    snr: float
    n_samples: int
    dt: float
    channels: str
    generator: Dict[str, Any]
    signal_checksum: Dict[str, Any] = field(repr=False)
    noise_seed: int = 0
    noise_checksum: Dict[str, Any] = field(default_factory=dict, repr=False)
    psd: Dict[str, Any] = field(default_factory=dict)
    made_with: Dict[str, str] = field(default_factory=dict)
    kind: str = "injection"
    format: int = FORMAT

    @classmethod
    def make(
        cls, config, injection: EMRIInjection, injector: FEWInjector, label, noise_seed
    ) -> "InjectionFile":
        """The recipe of ``injection`` (made by ``injector``), searched on the noise drawn
        with ``noise_seed``."""
        signal = np.atleast_2d(injection.signal)
        n_channels, n_samples = signal.shape
        noise = _draw_noise(config, noise_seed, n_samples, n_channels)
        generator = dict(
            T=injector.T,
            tdi=injector.tdi,
            order=injector.order,
            t_buffer=injector.t_buffer,
            eps=injector.eps,
            force_backend=injector.force_backend,
        )
        return cls(
            label=label,
            params={name: float(value) for name, value in injection.params.items()},
            snr=float(injection.snr),
            n_samples=n_samples,
            dt=float(injection.dt),
            channels=injection.channels,
            generator=generator,
            signal_checksum=checksum(signal),
            noise_seed=int(noise_seed),
            noise_checksum=checksum(noise),
            psd=psd_identity(config),
            made_with=versions(),
        )

    def signal(self, path="injection file") -> np.ndarray:
        """The signal, regenerated with FEW and checked against the file.

        The waveform is generated from the recorded initial conditions ``(p0, e0, x0)`` at
        the recorded distance, which already gives the target SNR, then cut to the kept
        length.
        """
        settings = dict(self.generator)
        duration = settings.pop("T")
        # building an injector is cheap: the response generator is cached process-wide
        injector = FEWInjector(duration, dt=self.dt, channels=self.channels, **settings)
        source = self.params
        angles = {name: source[name] for name in ANGLE_NAMES}
        waveform = injector.waveform(
            source["m1"],
            source["m2"],
            source["a"],
            source["p0"],
            source["e0"],
            source["x0"],
            source["dist"],
            angles,
        )
        signal = np.atleast_2d(waveform)[:, : self.n_samples]
        verify_checksum(signal, self.signal_checksum, f"{path} (signal)", self.made_with)
        return signal

    def noise(self, config, path="injection file") -> np.ndarray:
        """The injection's noise realisation, checked against the file."""
        noise = _draw_noise(config, self.noise_seed, self.n_samples, len(self.channels))
        what = f"{path} (noise seed {self.noise_seed})"
        verify_checksum(noise, self.noise_checksum, what, self.made_with)
        return noise

    def injection(self, config, path="injection file") -> EMRIInjection:
        """The :class:`~.injection.EMRIInjection` with its noise, on ``config``'s PSD."""
        _check_matches_config(self, config, path)
        return EMRIInjection(
            signal=self.signal(path),
            dt=self.dt,
            params=dict(self.params),
            snr=self.snr,
            channels=self.channels,
            psd=config.psd(),
            noise=self.noise(config, path),
        )

    def save(self, path):
        _save(self, path)

    @classmethod
    def load(cls, path) -> "InjectionFile":
        return _load(cls, path, "injection")


def _save(recipe, path):
    """Write to a temporary file first, so a crash leaves no half-written recipe."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(asdict(recipe), fh, indent=1)
    os.replace(tmp, path)


def _load(cls, path, kind):
    with open(path) as fh:
        fields = json.load(fh)
    if fields.get("kind") != kind:
        raise RecipeMismatch(f"{path} is a {fields.get('kind')!r} file, not {kind!r}.")
    if fields.get("format") != FORMAT:
        raise RecipeMismatch(f"{path}: format {fields.get('format')}, this code reads {FORMAT}.")
    return cls(**fields)
