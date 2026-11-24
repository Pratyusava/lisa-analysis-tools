"""A search campaign: many data streams searched over many durations, with a background.

A campaign is one TOML file (``scripts/emri_search/configs/*.toml``), read by
:meth:`RunConfig.from_toml` and run by ``scripts/emri_search/emri_search.py``. In order of
use:

1. :class:`RunConfig` reads the file: where the results go (``out``), the processes
   (``n_workers``), what is searched (``[inputs]``, :class:`InputSettings`) and how (every
   other table, the :class:`~.config.SearchConfig`).
2. The inputs become :class:`Stream` objects: an injection with its noise
   (:func:`injection_streams`), noise alone for the background (:func:`noise_streams`), or a
   data file as is (:func:`data_stream`). Injection and noise files are recipes
   (:mod:`.recipes`), so a stream regenerates its data only when a search needs it.
3. :func:`campaign_tasks` makes one search per stream and duration, skipping the ones already
   saved, and :func:`run_campaign` runs them in parallel processes, saving each result as it
   arrives (:class:`~.results.ResultStore`). An interrupted campaign continues where it
   stopped.
4. :func:`~.results.collect` turns the saved results into the summary table.

:meth:`RunConfig.run` does steps 2 to 4.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import glob
import os
import tomllib
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .config import SearchConfig, set_dotted, settings_from_dict
from .injection import EMRIInjection
from .parallel import SearchTask, run_tasks
from .recipes import InjectionFile, NoiseFile
from .results import ResultStore, StreamInfo, collect

STAGES = ("search", "background", "collect")


# ----------------------------------------------------------------------------- the campaign file
@dataclass
class InputSettings:
    """``[inputs]``: what a campaign searches.

    Args:
        injections: Injection files or globs (``make_injections.py``), each searched on the
            noise realisation it records.
        noise: Noise files or globs (``make_noise.py``), the background the false-alarm
            rates come from (paper: 100 streams; a FAR resolves down to ``1 / n``).
        data: Data files searched as is (``.npz`` with ``signal``, shape
            ``(n_channels, N)``, and ``dt``).
    """

    injections: List[str] = field(default_factory=list)
    noise: List[str] = field(default_factory=list)
    data: List[str] = field(default_factory=list)


@dataclass
class RunConfig:
    """One campaign: a TOML file's top-level keys, ``[inputs]`` and the search's tables.

    Args:
        out: The campaign directory (``config.json``, ``results/``, the summary and a
            built basis).
        n_workers: Processes the searches run in.
        tol_bins: Localisation tolerance of :func:`~.results.collect` [bins].
        p_fa: False-alarm probabilities of the detection thresholds (paper: 0.5, 1e-2, 1e-4).
        coherence: Coherent segments [SFT segments] the track mismatch of
            :func:`~.results.collect` is computed over (:mod:`.mismatch`); 1 is always
            included.
        inputs: What is searched.
        search: Everything else in the file (:class:`~.config.SearchConfig`).

    Paths in the file (``out``, the inputs, ``data.psd_file``, ``basis.file``) may use
    ``$VARIABLES`` and ``~`` and are relative to the file's directory.
    """

    out: str = ""
    n_workers: int = 1
    tol_bins: float = 2.0
    p_fa: Tuple[float, ...] = (0.5, 1e-2, 1e-4)
    coherence: Tuple[int, ...] = (1, 2, 4)
    inputs: InputSettings = field(default_factory=InputSettings)
    search: SearchConfig = field(default_factory=SearchConfig)

    # the top-level keys that belong to the run; every other key is a SearchConfig table
    RUN_KEYS = ("out", "n_workers", "tol_bins", "p_fa", "coherence", "inputs")

    # -- reading the file
    @classmethod
    def from_toml(cls, path, overrides: Sequence[str] = ()) -> "RunConfig":
        """Read a campaign file.

        Each override is ``"dotted.key=value"`` with a TOML value (a bare word is a string),
        e.g. ``["statistic.name=H", "optimizer.n_steps=100"]``.
        """
        with open(path, "rb") as fh:
            tables = tomllib.load(fh)
        for override in overrides:
            key, value = parse_override(override)
            set_dotted(tables, key, value)
        return cls.from_dict(tables, base_dir=os.path.dirname(os.path.abspath(path)))

    @classmethod
    def from_dict(cls, tables: dict, base_dir=".") -> "RunConfig":
        """From the file's tables; relative paths are taken from ``base_dir``."""
        search_tables = copy.deepcopy(tables)
        run_tables = {key: search_tables.pop(key) for key in cls.RUN_KEYS if key in search_tables}
        run = settings_from_dict(cls, run_tables)
        run.search = SearchConfig.from_dict(search_tables)
        run._resolve_paths(base_dir)
        return run

    def to_dict(self) -> dict:
        """The resolved configuration, in the campaign file's layout."""
        run_tables = {key: getattr(self, key) for key in self.RUN_KEYS if key != "inputs"}
        run_tables["inputs"] = dataclasses.asdict(self.inputs)
        return {**run_tables, **self.search.to_dict()}

    def _resolve_paths(self, base_dir):
        """Make every path absolute, and give a built basis its default file."""
        if not self.out:
            raise ValueError("the campaign file needs out = '<campaign directory>'.")

        def absolute(path):
            path = os.path.expanduser(os.path.expandvars(path))
            if "$" in path:
                raise ValueError(f"{path!r}: environment variable not set.")
            return os.path.normpath(os.path.join(base_dir, path))

        self.out = absolute(self.out)
        self.inputs.injections = [absolute(path) for path in self.inputs.injections]
        self.inputs.noise = [absolute(path) for path in self.inputs.noise]
        self.inputs.data = [absolute(path) for path in self.inputs.data]

        data, basis = self.search.data, self.search.basis
        if data.psd_file:
            data.psd_file = absolute(data.psd_file)
        if basis.file:
            basis.file = absolute(basis.file)
        elif basis.source == "build":
            basis.file = os.path.join(self.out, "basis.npz")
        else:
            raise ValueError('basis.source = "file" needs basis.file.')

    # -- the inputs
    def injection_paths(self, required=True) -> List[str]:
        return _matching_files("inputs.injections", self.inputs.injections, required)

    def noise_paths(self, required=True) -> List[str]:
        return _matching_files("inputs.noise", self.inputs.noise, required)

    def data_paths(self, required=True) -> List[str]:
        return _matching_files("inputs.data", self.inputs.data, required)

    def check_inputs(self):
        """Check the input files that exist so far.

        - Labels must be unique: results are filed by label.
        - No injection may be searched on a noise realisation of the background: its
          false-alarm rate would then not come from independent noise.
        """
        injections = [(path, InjectionFile.load(path)) for path in self.injection_paths(False)]
        noise = [(path, NoiseFile.load(path)) for path in self.noise_paths(False)]
        labels = (
            [(recipe.label, path) for path, recipe in injections]
            + [(recipe.label, path) for path, recipe in noise]
            + [(_file_stem(path), path) for path in self.data_paths(False)]
        )
        path_of_label: Dict[str, str] = {}
        for label, path in labels:
            if label in path_of_label:
                raise ValueError(
                    f"{path} and {path_of_label[label]} have the same label {label!r}."
                )
            path_of_label[label] = path

        background_seeds = {recipe.seed: path for path, recipe in noise}
        for path, recipe in injections:
            if recipe.noise_seed in background_seeds:
                raise ValueError(
                    f"{path} is searched on noise seed {recipe.noise_seed}, which is also the "
                    f"background stream {background_seeds[recipe.noise_seed]}; remake one of "
                    "them with other seeds."
                )

    def search_streams(self) -> Iterator["Stream"]:
        """The injections, then the data files."""
        yield from injection_streams(self.search, self.injection_paths())
        for path in self.data_paths():
            yield data_stream(self.search, path)

    def background_streams(self) -> Iterator["Stream"]:
        """The noise-only streams."""
        return noise_streams(self.search, self.noise_paths())

    # -- running
    def run(self, stages: Sequence[str] = STAGES, log=print) -> Optional[Dict[str, Any]]:
        """Run the requested stages, always in this order.

        - ``"search"``: the injections and data files.
        - ``"background"``: the noise streams.
        - ``"collect"``: the summary, which is returned.

        The basis is built first when ``basis.source = "build"`` and its file does not exist.
        """
        unknown = set(stages) - set(STAGES)
        if unknown:
            raise ValueError(f"unknown stage(s) {sorted(unknown)}; known: {STAGES}.")

        if set(stages) == {"collect"}:  # needs only the campaign directory
            return self.collect(ResultStore(self.out), log)

        self.check_inputs()
        self.search.prepare_basis(log=log, n_workers=self.n_workers)
        store = ResultStore(self.out, self.search)
        log(
            f"configuration {store.fingerprint} in {self.out}: statistic "
            f"{self.search.statistic.name}, basis {self.search.basis.name}, "
            f"{self.n_workers} workers"
        )
        if "search" in stages:
            n_done = run_campaign(self.search, self.search_streams(), store, self.n_workers, log)
            log(f"search: {n_done} searches done")
        if "background" in stages:
            n_done = run_campaign(
                self.search, self.background_streams(), store, self.n_workers, log
            )
            log(f"background: {n_done} searches done")
        if "collect" in stages:
            return self.collect(store, log)
        return None

    def collect(self, store: ResultStore, log=print) -> Dict[str, Any]:
        """The summary, with the track mismatch of every injection file still at hand."""
        signals = {}
        for path in self.injection_paths(required=False):
            recipe = InjectionFile.load(path)
            signals[recipe.label] = functools.partial(recipe.signal, path)
        return collect(
            store, self.tol_bins, self.p_fa, log=log, signals=signals, coherence=self.coherence
        )


def _matching_files(setting, patterns, required=True) -> List[str]:
    """The files matching ``patterns`` (paths or globs), sorted. A pattern that matches
    nothing raises when ``required``."""
    paths = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if not matches and required:
            raise FileNotFoundError(f"{setting}: nothing matches {pattern!r}.")
        paths.update(matches)
    return sorted(paths)


def parse_override(override: str) -> Tuple[str, Any]:
    """``"a.b=value"`` -> ``("a.b", value)``, the value read as TOML (bare words as strings)."""
    key, equals, text = override.partition("=")
    if not equals or not key.strip():
        raise ValueError(f"override {override!r} is not key=value.")
    try:
        value = tomllib.loads(f"value = {text}")["value"]
    except tomllib.TOMLDecodeError:
        value = text.strip()
    return key.strip(), value


# ----------------------------------------------------------------------------- streams
@dataclass
class Stream:
    """A data stream to search, regenerated only when a search needs it.

    Args:
        info: Label, kind, the injected source's parameters, noise seed and SNR
            (:class:`~.results.StreamInfo`).
        n_samples, dt: Length and cadence, known without regenerating the data.
        load: Zero-argument callable returning the :class:`~.injection.EMRIInjection`
            (the signal, or the data as is, and its fixed noise).
        signal: Whether the signal enters the data (``False`` for the background).
        noise: Whether the noise enters the data (``False`` for a data file).
    """

    info: StreamInfo
    n_samples: int
    dt: float
    load: Callable[[], EMRIInjection] = field(repr=False)
    signal: bool = True
    noise: bool = True
    _loaded: Optional[EMRIInjection] = field(default=None, repr=False)

    @property
    def label(self) -> str:
        return self.info.label

    @property
    def injection(self) -> EMRIInjection:
        """The stream's data, regenerated on first use and kept until :meth:`release`."""
        if self._loaded is None:
            self._loaded = self.load()
        return self._loaded

    def release(self):
        """Drop the regenerated data (once every duration's search is built)."""
        self._loaded = None

    def segments(self, config: SearchConfig) -> int:
        """Whole segments of the stream."""
        samples_per_segment = int(config.data.T_sft / self.dt)
        return self.n_samples // samples_per_segment

    def sft_data(self, config: SearchConfig, n_seg):
        """The SFTs of the first ``n_seg`` segments (what the Fresnel statistic reads)."""
        return self.injection.sft_data(
            config.data.T_sft,
            signal=self.signal,
            noise=self.noise,
            n_segments=n_seg,
            window_alpha=config.data.window_alpha,
        )

    def fd_data(self, config: SearchConfig, n_seg):
        """The band-limited FFT of the first ``n_seg`` segments (what ``H`` reads)."""
        H = config.statistic.H
        return self.injection.fd_data(
            signal=self.signal,
            noise=self.noise,
            duration=n_seg * config.data.T_sft,
            min_freq=H.band[0],
            max_freq=H.band[1],
            window_alpha=H.window_alpha,
        )


def injection_streams(config: SearchConfig, paths: Iterable[str]) -> Iterator[Stream]:
    """Streams of injection files (:class:`~.recipes.InjectionFile`), each on its own noise."""
    for path in sorted(paths):
        recipe = InjectionFile.load(path)
        info = StreamInfo(
            label=recipe.label,
            kind="injection",
            truth=dict(recipe.params),
            noise_seed=recipe.noise_seed,
            snr=recipe.snr,
        )
        load = functools.partial(recipe.injection, config, path)
        yield Stream(info, recipe.n_samples, recipe.dt, load)


def noise_streams(config: SearchConfig, paths: Iterable[str]) -> Iterator[Stream]:
    """Noise-only streams of noise files (:class:`~.recipes.NoiseFile`), the background."""
    for path in sorted(paths):
        recipe = NoiseFile.load(path)
        info = StreamInfo(label=recipe.label, kind="noise", noise_seed=recipe.seed)
        load = functools.partial(_noise_only_injection, recipe, config, path)
        yield Stream(info, recipe.n_samples, recipe.dt, load, signal=False)


def data_stream(config: SearchConfig, path, label=None) -> Stream:
    """A data file searched as is: ``.npz`` with ``signal`` (``(n_channels, N)``) and ``dt``."""
    with np.load(path) as file:
        n_channels, n_samples = np.atleast_2d(file["signal"]).shape
        dt = float(file["dt"]) if "dt" in file.files else config.data.dt
    if dt != config.data.dt or n_channels != len(config.data.channels):
        raise ValueError(f"{path}: dt {dt} / {n_channels} channels do not match the config.")
    info = StreamInfo(label=label or _file_stem(path), kind="data")
    load = functools.partial(_data_file_injection, config, path)
    return Stream(info, n_samples, dt, load, noise=False)


def _noise_only_injection(recipe: NoiseFile, config: SearchConfig, path) -> EMRIInjection:
    """The noise of a noise file, as an injection with a zero signal."""
    noise = recipe.generate(config, path)
    data = config.data
    return EMRIInjection(
        signal=np.zeros_like(noise),
        dt=data.dt,
        params={},
        snr=0.0,
        channels=data.channels,
        psd=config.psd(),
        noise=noise,
    )


def _data_file_injection(config: SearchConfig, path) -> EMRIInjection:
    """A data file's samples, as an injection whose "signal" is the data and with no noise."""
    with np.load(path) as file:
        samples = np.atleast_2d(np.asarray(file["signal"], dtype=float))
    data = config.data
    return EMRIInjection(
        signal=samples, dt=data.dt, params={}, snr=0.0, channels=data.channels, psd=config.psd()
    )


def _file_stem(path) -> str:
    return os.path.splitext(os.path.basename(path))[0]


# ----------------------------------------------------------------------------- running
def campaign_tasks(config: SearchConfig, streams: Iterable[Stream], store: ResultStore):
    """One :class:`~.parallel.SearchTask` per stream and duration not yet in ``store``.

    A stream's data are regenerated only when one of its durations is left to search. The
    optimiser's seed depends on the duration alone (``optimizer.seed + k`` for the ``k``-th
    duration), so every stream is searched with the same seeds.
    """
    for stream in streams:
        durations = config.durations_of(stream.segments(config))
        if not durations:
            warnings.warn(
                f"{stream.label} is shorter than durations.T_min; it is not searched.",
                stacklevel=2,
            )
        for k, n_seg in enumerate(durations):
            if store.done(stream.label, n_seg, stream.info.noise_seed):
                continue
            yield SearchTask(
                search=config.build_search(stream, n_seg),
                T_pl=n_seg * config.data.T_sft,  # plunge at the end of the analysed stream
                seed=config.optimizer.seed + k,
                tag=(stream.info, n_seg),
                keep_history=False,
            )
        stream.release()


def run_campaign(
    config: SearchConfig, streams: Iterable[Stream], store: ResultStore, n_workers=1, log=print
) -> int:
    """Run every task of ``streams`` in ``n_workers`` processes, saving each result as it
    finishes. Returns the number of searches run."""
    n_done = 0
    tasks = campaign_tasks(config, streams, store)
    for (info, n_seg), result in run_tasks(tasks, n_workers=n_workers):
        store.save(info, n_seg, result)
        n_done += 1
        iterations = result.optimization.history.get("last_step", config.optimizer.n_steps)
        log(
            f"[{n_done}] {info.label} n = {n_seg}: {result.best_statistic:.2f} "
            f"({iterations} iterations)"
        )
    return n_done
