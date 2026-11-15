"""The configuration of one EMRI search: what is searched for, with which statistic and basis.

:class:`SearchConfig` holds one ``*Settings`` block per part of the search, each a
documented dataclass, in the layout of the campaign's TOML file
(``scripts/emri_search/configs/*.toml``; :class:`~.campaign.RunConfig` reads it):

========================  =====================================================
``[data]``                :class:`DataSettings`: cadence, segments, channels, PSD
``[statistic]``           :class:`StatisticSettings`: ``name = "fresnel" | "H"``
                          and one sub-table per statistic
``[basis]``               :class:`BasisSettings`: ``name = "svd" | "chebyshev"``,
                          ``source = "file" | "build"``, one sub-table per basis
                          and the FEW track bank a build draws
``[durations]``           :class:`DurationSettings`: the analysed stream lengths
``[optimizer]``           :class:`OptimizerSettings`: initial screening and Adam + DE
========================  =====================================================

Defaults are Speri et al.'s choices (arXiv:2510.20891 and github.com/lorenzsp/EMRI-Search:
the paper's value where it states one, the repository's otherwise), and Cutler &
Bustamante-Rosell's draft for ``H`` and the Chebyshev model. Construction is cheap and the
object pickles; :meth:`SearchConfig.build_search` does the heavy work, and
:meth:`SearchConfig.fingerprint` identifies the results a configuration produces (only the
chosen statistic's and basis's blocks enter it).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import typing
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...utils.constants import YRSID_SI
from .excess_power import ExcessPowerStatistic
from .injection import PAPER_F0_RANGE, TABLE1_PRIOR
from .optimize import Adam, AdamDE
from .pipeline import EMRISearch
from .psd import SensitivityPSD
from .statistic import FresnelStatistic, SegmentMask
from .tracks import ChebyshevBasis, ChebyshevTrackModel, SVDBasis, SVDTrackModel, few_track_bank

STATISTICS = ("fresnel", "H")
BASES = ("svd", "chebyshev")
BASIS_SOURCES = ("file", "build")


# ----------------------------------------------------------------------------- the blocks
@dataclass
class DataSettings:
    """``[data]``: how a time series becomes the statistic's input.

    Args:
        dt: Sampling cadence [s] (paper: 5).
        T_sft: Segment duration [s] (paper: 5e4).
        window_alpha: Tukey parameter of each segment's window (repo: 0.2).
        channels: TDI channels searched, e.g. ``"A"`` (paper) or ``"AE"``.
        psd_file: Two-column ``(f, S)`` PSD table, e.g. EMRI-Search's
            ``src/emrisearch/data/TDI2_AE_psd.npy``; ``None`` uses LAT's analytic
            TDI2 PSD of ``channels[0]``.
    """

    dt: float = 5.0
    T_sft: float = 5e4
    window_alpha: float = 0.2
    channels: str = "A"
    psd_file: Optional[str] = None


@dataclass
class FresnelSettings:
    """``[statistic.fresnel]``: Speri et al.'s semi-coherent ``S`` (paper Eq. 9).

    Args:
        P: Half-width of the bin window around the track (paper: 100).
        f_min, fdot_min, dphi_max: The segments counted (:class:`~.statistic.SegmentMask`):
            ``f > f_min``, ``fdot > fdot_min``, ``|fddot| T_sft^3 / 6 < dphi_max``
            (paper: 10^-3.5 Hz, 1e-13 Hz/s, 1).
    """

    P: int = 100
    f_min: Optional[float] = 10**-3.5
    fdot_min: Optional[float] = 1e-13
    dphi_max: Optional[float] = 1.0


@dataclass
class HSettings:
    """``[statistic.H]``: Cutler & Bustamante-Rosell's coherent excess power ``H``.

    Args:
        band: Band of the coherent FFT the statistic reads [Hz]; must hold the tracks.
        windows: Delay-window lengths ``H`` maximises over [s] (draft: 10, 21, 86, 345
            cells of ~200 s).
        delay_step: Spacing of the delay grid [s] (draft: ~200).
        normalization: ``"exact"`` (finite-window variance per template) or
            ``"asymptotic"`` (the draft's ``R = L / K``).
        window_alpha: Tukey parameter of the stream before its FFT (0 lets the cut end
            wrap onto the start, a loud transient at delay 0).
        spans: Template widths [Hz] the population is bucketed by (exact, faster);
            ``"none"``: every member is scored on the whole band.
    """

    band: Tuple[float, float] = (3e-4, 2e-2)
    windows: Tuple[float, ...] = (2e3, 4.2e3, 1.72e4, 6.9e4)
    delay_step: float = 200.0
    normalization: str = "exact"
    window_alpha: float = 0.01
    spans: Optional[Tuple[float, ...]] = (2.5e-3, 5e-3, 1e-2)


@dataclass
class StatisticSettings:
    """``[statistic]``: what a track scores.

    Args:
        name: ``"fresnel"`` (Speri et al.'s ``S``, :class:`~.statistic.FresnelStatistic`)
            or ``"H"`` (:class:`~.excess_power.ExcessPowerStatistic`).
        fresnel, H: The settings of each; only the chosen one is used.
    """

    name: str = "fresnel"
    fresnel: FresnelSettings = field(default_factory=FresnelSettings)
    H: HSettings = field(default_factory=HSettings)

    @property
    def active(self):
        return getattr(self, self.name)


@dataclass
class SVDSettings:
    """``[basis.svd]``: Speri et al.'s SVD basis of FEW tracks.

    Args:
        f0_range: Starting frequencies of the initial tracks [Hz] (paper: 0.5-10 mHz).
        n_modes: Singular vectors kept by a build (paper: 20).
        box_quantile: A build's coefficient box ``[Q_q, Q_(1-q)]`` per mode (repo: 0.015,
            which reproduces the box of EMRI-Search's basis file).
        first_mode_max: The first mode's upper edge is the bank's maximum (repo: yes).
    """

    f0_range: Tuple[float, float] = PAPER_F0_RANGE
    n_modes: int = 20
    box_quantile: float = 0.015
    first_mode_max: bool = True


@dataclass
class ChebyshevSettings:
    """``[basis.chebyshev]``: Cutler & Bustamante-Rosell's Chebyshev phase model.

    Args:
        y_range: Range of the anchor coordinate, the track's ``y`` (log frequency in the
            band) at the end of the stream.
        band: The band [Hz] (draft: 3-8 mHz); tracks are zero outside it.
        n_coeffs: Coefficients of a build (draft: 5).
        margin: A build widens the fitted coefficients' range by this at each end (draft: 0.2).
        min_span: A bank track enters the box when its fit covers this much of the band's
            width 2 in ``y`` (draft: 1.9).
        box_scale: Widen the box about its centre (draft: 1, the catalogue box, or 10).
    """

    y_range: Tuple[float, float] = (-1.0, 1.0)
    band: Tuple[float, float] = (3e-3, 8e-3)
    n_coeffs: int = 5
    margin: float = 0.2
    min_span: float = 1.9
    box_scale: float = 1.0


@dataclass
class BankSettings:
    """``[basis.bank]``: the FEW ``m = 2`` track bank a basis build draws.

    Args:
        n_tracks: Tracks in the bank (paper: 10^4; EMRI-Search's file holds 46 371).
        seed: Seed of the draws (repo: 2601).
        prior: Uniform ranges of ``log10_m1``, ``m2``, ``a``, ``e_f`` (paper Table 1).
    """

    n_tracks: int = 10_000
    seed: int = 2601
    prior: Dict[str, Tuple[float, float]] = field(default_factory=lambda: dict(TABLE1_PRIOR))


@dataclass
class BasisSettings:
    """``[basis]``: how a search track is parametrised.

    Args:
        name: ``"svd"`` (:class:`~.tracks.SVDTrackModel`) or ``"chebyshev"``
            (:class:`~.tracks.ChebyshevTrackModel`).
        source: ``"file"``: load :attr:`file` (EMRI-Search's
            ``examples/new_svd_basis_Tsft50000.0_cut20.npz``, or a basis saved by a build);
            ``"build"``: build it from :attr:`bank` with the chosen basis's settings and save
            it to :attr:`file` (default ``<out>/basis.npz``), reused once it exists.
        file: The basis file.
        T_ref: Plunge time of the basis tracks [s] (EMRI-Search's file: 1 yr); the
            Chebyshev anchor time.
        svd, chebyshev, bank: The settings of each basis and of a build's bank.
    """

    name: str = "svd"
    source: str = "file"
    file: str = ""
    T_ref: float = YRSID_SI
    svd: SVDSettings = field(default_factory=SVDSettings)
    chebyshev: ChebyshevSettings = field(default_factory=ChebyshevSettings)
    bank: BankSettings = field(default_factory=BankSettings)

    @property
    def active(self):
        return getattr(self, self.name)


@dataclass
class DurationSettings:
    """``[durations]``: the analysed stream lengths, plunge at each one's end.

    Args:
        T_min: Shortest analysed stream [yr] (paper: 0.1).
        step: Increment of the stream length [segments] (paper: 12, ~7 d).
    """

    T_min: float = 0.1
    step: int = 12


@dataclass
class OptimizerSettings:
    """``[optimizer]``: the initial screening and :class:`~.optimize.AdamDE`.

    Args:
        n_initial: Initial draws screened by the statistic (paper: 1024).
        n_population: Best draws kept and optimised (paper: 512).
        n_steps: Adam + DE iterations (paper: 525).
        patience, min_improvement: Stop once the best statistic has not grown by more
            than ``min_improvement`` (relative) for ``patience`` iterations;
            ``"none"`` (paper) always runs ``n_steps``.
        learning_rate: Adam step size in unit-cube coordinates (paper: 1e-5).
        F_range: DE differential weight range (paper: 0.3-1.5).
        p_elitism: Probability that the DE base vector is the fittest member (repo: 0.5).
        batch_size: Members evaluated at a time (memory only, not in the fingerprint);
            ``"none"``: all at once.
        seed: Base seed of the initial draws and DE (duration ``k`` adds ``k``).
    """

    n_initial: int = 1024
    n_population: int = 512
    n_steps: int = 525
    patience: Optional[int] = None
    min_improvement: float = 1e-3
    learning_rate: float = 1e-5
    F_range: Tuple[float, float] = (0.3, 1.5)
    p_elitism: float = 0.5
    batch_size: Optional[int] = None
    seed: int = 0


# ----------------------------------------------------------------------------- the search
@dataclass
class SearchConfig:
    """Every setting of one search, as the blocks above (see the module docstring)."""

    data: DataSettings = field(default_factory=DataSettings)
    statistic: StatisticSettings = field(default_factory=StatisticSettings)
    basis: BasisSettings = field(default_factory=BasisSettings)
    durations: DurationSettings = field(default_factory=DurationSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)

    def __post_init__(self):
        for name, value, allowed in (
            ("statistic.name", self.statistic.name, STATISTICS),
            ("basis.name", self.basis.name, BASES),
            ("basis.source", self.basis.source, BASIS_SOURCES),
            ("statistic.H.normalization", self.statistic.H.normalization, ("exact", "asymptotic")),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, not {value!r}.")

    # -- serialisation
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, tables: dict) -> "SearchConfig":
        """From nested dicts (a TOML file's tables, a saved ``config.json``); unknown keys
        raise, omitted ones take their defaults, ``"none"`` is ``None`` where allowed."""
        return settings_from_dict(cls, tables)

    def save_json(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1, sort_keys=True)

    @classmethod
    def load_json(cls, path) -> "SearchConfig":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def fingerprint(self) -> str:
        """Hash of the settings that determine a result.

        Every block enters, except:

        - of ``[statistic.*]`` and ``[basis.*]``, only the chosen one's settings;
        - the basis by what it is: the file's content when loaded, the bank's settings when
          built (not the path it is saved to);
        - ``optimizer.batch_size``, which only limits memory.
        """
        settings = self.to_dict()
        settings["optimizer"].pop("batch_size")

        statistic = settings["statistic"]
        settings["statistic"] = {"name": statistic["name"], **statistic[statistic["name"]]}

        basis = settings["basis"]
        identifying = {
            "name": basis["name"],
            "source": basis["source"],
            "T_ref": basis["T_ref"],
            **basis[basis["name"]],
        }
        if basis["source"] == "file":
            identifying["file_content"] = _file_hash(basis["file"])
        else:
            identifying["bank"] = basis["bank"]
        settings["basis"] = identifying

        text = json.dumps(settings, sort_keys=True)
        return hashlib.sha1(text.encode()).hexdigest()[:16]

    def replace(self, **changes) -> "SearchConfig":
        """A copy with ``changes``, keys as dotted paths: ``replace(**{"statistic.name": "H"})``."""
        settings = self.to_dict()
        for key, value in changes.items():
            set_dotted(settings, key, value)
        return SearchConfig.from_dict(settings)

    # -- small pieces
    def psd(self):
        """The PSD: the tabulated ``data.psd_file``, or LAT's analytic TDI2 one."""
        if self.data.psd_file:
            return SensitivityPSD.from_file(self.data.psd_file, channel=self.data.channels[0])
        from .psd import _channel_sens_fn

        return SensitivityPSD(sens_fn=_channel_sens_fn(self.data.channels[0]))

    def mask(self) -> SegmentMask:
        """The segments the Fresnel statistic counts."""
        fresnel = self.statistic.fresnel
        return SegmentMask(fresnel.f_min, fresnel.fdot_min, fresnel.dphi_max)

    def n_samples(self, T_data) -> int:
        """Samples of a ``T_data`` [yr] stream."""
        return int(T_data * YRSID_SI / self.data.dt)

    def durations_of(self, n_segments) -> List[int]:
        """The analysed stream lengths [segments] of a stream with ``n_segments``."""
        shortest = max(1, int(self.durations.T_min * YRSID_SI / self.data.T_sft))
        return list(range(shortest, n_segments + 1, self.durations.step))

    # -- the basis
    def basis_grid(self) -> np.ndarray:
        """Segment start times of the basis tracks [s], up to the plunge at ``T_ref``."""
        return np.arange(0.0, self.basis.T_ref, self.data.T_sft)

    def build_basis(self, log=print):
        """Build the basis from a FEW track bank (``[basis.bank]``) and save it to
        ``basis.file``. Needs FEW; ~0.25 s per bank track."""
        settings = self.basis
        if not settings.file:
            raise ValueError("basis.file is not set (where the built basis is saved).")
        bank = few_track_bank(
            settings.bank.prior,
            settings.bank.n_tracks,
            self.basis_grid(),
            settings.T_ref,
            settings.bank.seed,
        )
        log(f"bank: {len(bank.f)} FEW m = 2 tracks plunging at T_ref")

        basis: Any
        if settings.name == "svd":
            svd = settings.svd
            basis = SVDBasis.build(
                bank.f,
                n_modes=svd.n_modes,
                box_quantile=svd.box_quantile,
                first_mode_max=svd.first_mode_max,
            )
            log(f"SVD basis: {basis.n_modes} modes x {basis.n_samples} samples")
        else:
            chebyshev = settings.chebyshev
            basis = ChebyshevBasis.fit(
                bank.f,
                bank.fdot,
                n_coeffs=chebyshev.n_coeffs,
                band=chebyshev.band,
                margin=chebyshev.margin,
                min_span=chebyshev.min_span,
                scale=chebyshev.box_scale,
            )
            log(f"Chebyshev basis: {len(basis.fits)} of {len(bank.f)} bank tracks enter the box")

        os.makedirs(os.path.dirname(os.path.abspath(settings.file)), exist_ok=True)
        basis.save_npz(settings.file)
        log(f"basis saved to {settings.file}")
        return basis

    def prepare_basis(self, log=print):
        """Make sure the basis file exists (built once for ``source = "build"``) and fits the
        grid: an SVD basis must have one sample per segment before ``T_ref``."""
        if not os.path.exists(self.basis.file):
            if self.basis.source == "file":
                raise FileNotFoundError(f"basis.file {self.basis.file!r} does not exist.")
            self.build_basis(log=log)
        basis = self.load_basis()
        n_grid = self.basis_grid().size
        if self.basis.name == "svd" and basis.n_samples != n_grid:
            raise ValueError(
                f"the SVD basis {self.basis.file} has {basis.n_samples} samples, but "
                f"basis.T_ref / data.T_sft gives {n_grid} segments."
            )

    def load_basis(self):
        """The basis object of ``basis.file`` (cached: every duration's search reuses it)."""
        if not self.basis.file:
            raise ValueError("basis.file is not set.")
        return _load_basis(self.basis.name, os.path.abspath(self.basis.file))

    def track_model(self, t_alpha, force_backend="jax"):
        """The chosen track model on the segment start times ``t_alpha`` [s]."""
        settings = self.basis
        basis = self.load_basis()
        if settings.name == "svd":
            return SVDTrackModel(basis, t_alpha, T_ref=settings.T_ref, force_backend=force_backend)
        return ChebyshevTrackModel(
            basis,
            t_alpha,
            T_ref=settings.T_ref,
            y_range=settings.chebyshev.y_range,
            force_backend=force_backend,
        )

    # -- the search
    def optimizer_object(self) -> AdamDE:
        settings = self.optimizer
        return AdamDE(
            n_steps=settings.n_steps,
            adam=Adam(learning_rate=settings.learning_rate),
            F_range=settings.F_range,
            p_elitism=settings.p_elitism,
            save_every=10,
            batch_size=settings.batch_size,
            patience=settings.patience,
            min_improvement=settings.min_improvement,
        )

    def data_and_statistic(self, stream, n_seg, force_backend="jax"):
        """The data container of the first ``n_seg`` segments of ``stream``
        (:class:`~.campaign.Stream`) and the chosen statistic on it.

        The Fresnel statistic reads the segments' SFTs; ``H`` reads one FFT of the whole
        analysed stream.
        """
        if self.statistic.name == "fresnel":
            data = stream.sft_data(self, n_seg)
            statistic = FresnelStatistic.for_data(
                data, P=self.statistic.fresnel.P, mask=self.mask(), force_backend=force_backend
            )
            return data, statistic

        H = self.statistic.H
        data = stream.fd_data(self, n_seg)
        segment_times = np.arange(n_seg) * self.data.T_sft
        statistic = ExcessPowerStatistic.for_data(
            data,
            segment_times,
            windows=H.windows,
            delay_step=H.delay_step,
            normalization=H.normalization,
            force_backend=force_backend,
        )
        return data, statistic

    def build_search(self, stream, n_seg) -> EMRISearch:
        """The search of the first ``n_seg`` segments of ``stream``, plunge at its end."""
        data, statistic = self.data_and_statistic(stream, n_seg)
        segment_times = np.arange(n_seg) * self.data.T_sft
        track_model = self.track_model(segment_times, force_backend="jax")

        # only the SVD model takes a range of starting frequencies for its initial tracks,
        # and only H can bucket the population by template width
        initial_kwargs = {}
        if self.basis.name == "svd":
            initial_kwargs["f0_range"] = self.basis.svd.f0_range
        spans = self.statistic.H.spans if self.statistic.name == "H" else None

        return EMRISearch(
            data,
            statistic,
            track_model,
            self.optimizer_object(),
            n_initial=self.optimizer.n_initial,
            n_population=self.optimizer.n_population,
            initial_kwargs=initial_kwargs,
            spans=spans,
        )


@lru_cache(maxsize=4)
def _load_basis(kind, path):
    return SVDBasis.from_npz(path) if kind == "svd" else ChebyshevBasis.from_npz(path)


def _file_hash(path) -> str:
    if not path:
        return ""
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()


# ----------------------------------------------------------------------------- dict <-> blocks
# A TOML file gives nested dicts of plain values. The functions below turn them into the
# settings dataclasses above, using each field's type annotation to check and convert its
# value, so a typo or a wrong type is reported with the key it came from.


def settings_from_dict(cls, table: dict, where: str = ""):
    """Build the settings dataclass ``cls`` from a (nested) dict.

    Unknown keys raise, omitted keys take their defaults, and each value is converted to its
    field's type: TOML integers become floats where a float is expected, lists become
    tuples, and ``"none"`` becomes ``None`` where the field is optional.

    Args:
        where: The table's dotted name, for error messages (``""`` for the top level).
    """
    table_name = where or "top level"
    if not isinstance(table, dict):
        raise ValueError(f"[{table_name}] must be a table, not {table!r}.")
    field_types = typing.get_type_hints(cls)
    field_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(table) - field_names
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in [{table_name}]; known: {sorted(field_names)}."
        )
    values = {}
    for key, value in table.items():
        dotted_key = f"{where}.{key}" if where else key
        values[key] = _convert(value, field_types[key], dotted_key)
    return cls(**values)


def _convert(value, field_type, dotted_key):
    """``value`` as ``field_type``: a settings block, an optional, a tuple, list or dict of
    scalars, or a scalar."""
    container, item_types = typing.get_origin(field_type), typing.get_args(field_type)

    is_optional = container is typing.Union and type(None) in item_types
    if is_optional:
        if value is None or (isinstance(value, str) and value.lower() == "none"):
            return None
        field_type = next(t for t in item_types if t is not type(None))
        container, item_types = typing.get_origin(field_type), typing.get_args(field_type)

    if dataclasses.is_dataclass(field_type):
        return settings_from_dict(field_type, value, dotted_key)

    try:
        if container is tuple:
            any_length = len(item_types) == 2 and item_types[1] is Ellipsis  # Tuple[float, ...]
            items = list(value)
            if not any_length and len(items) != len(item_types):
                raise ValueError(f"needs {len(item_types)} values")
            return tuple(_scalar(item, item_types[0]) for item in items)
        if container is list:
            items = [value] if isinstance(value, str) else value  # one path -> a list of one
            return [_scalar(item, item_types[0]) for item in items]
        if container is dict:
            value_type = item_types[1]
            return {
                str(key): _convert(item, value_type, f"{dotted_key}.{key}")
                for key, item in dict(value).items()
            }
        return _scalar(value, field_type)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{dotted_key} = {value!r}: {exc}") from None


def _scalar(value, scalar_type):
    """``value`` as a ``bool``, ``int``, ``float`` or ``str``, refusing a wrong type (a bool
    is never taken for a number, although Python counts it as an integer)."""
    if scalar_type is bool:
        if not isinstance(value, bool):
            raise TypeError("expected true or false")
        return value
    if scalar_type is int:
        if isinstance(value, bool) or not float(value).is_integer():
            raise TypeError("expected an integer")
        return int(value)
    if scalar_type is float:
        if isinstance(value, bool):
            raise TypeError("expected a number")
        return float(value)
    if scalar_type is str:
        if not isinstance(value, str):
            raise TypeError("expected a string")
        return value
    raise TypeError(f"unsupported field type {scalar_type}")


def set_dotted(tables: dict, dotted_key: str, value):
    """``tables["a"]["b"] = value`` for ``dotted_key = "a.b"``, creating tables on the way."""
    *table_names, last = dotted_key.split(".")
    table = tables
    for name in table_names:
        table = table.setdefault(name, {})
        if not isinstance(table, dict):
            raise ValueError(f"{dotted_key}: {name} is not a table.")
    table[last] = value
