"""Modular single-harmonic semi-coherent EMRI search.

Re-implements the search of Speri, Tenorio, Chapman-Bird & Gerosa (2026,
arXiv:2510.20891; github.com/lorenzosp/EMRI-Search) on LAT, split into
swappable parts: data containers (:mod:`.data`), injections
(:mod:`.injection`), PSD (:mod:`.psd`), frequency-track models
(:mod:`.tracks`: SVD bases, Cutler & Bustamante-Rosell's Chebyshev model
from :mod:`lisatools.sources.emri.chebyshevwave`, FEW tracks), statistics
(semi-coherent in :mod:`.statistic` with kernels in :mod:`.kernels`;
coherent delay-scanning, incl. Cutler &
Bustamante-Rosell's ``H``, in :mod:`.excess_power`), optimisers
(:mod:`.optimize`), the driver (:mod:`.pipeline`), parallel tasks
(:mod:`.parallel`), the configuration of a search (:mod:`.config`: statistic, basis
and the rest as ``*Settings`` blocks) and campaigns of many streams with a background for
false-alarm rates: the injection and noise files (:mod:`.recipes`), the run read from one
TOML file (:mod:`.campaign`, behind ``scripts/emri_search/emri_search.py``) and its saved
results and summary (:mod:`.results`).
"""

from .campaign import (  # noqa: F401
    InputSettings,
    RunConfig,
    Stream,
    campaign_tasks,
    data_stream,
    injection_streams,
    noise_streams,
    run_campaign,
)
from .config import (  # noqa: F401
    BankSettings,
    BasisSettings,
    ChebyshevSettings,
    DataSettings,
    DurationSettings,
    FresnelSettings,
    HSettings,
    OptimizerSettings,
    SearchConfig,
    StatisticSettings,
    SVDSettings,
)
from .data import FDData, SFTData, gaussian_noise  # noqa: F401
from .excess_power import (  # noqa: F401
    DelayCorrelationStatistic,
    ExcessPowerStatistic,
    MatchedFilterStatistic,
)
from .injection import (  # noqa: F401
    EMRIInjection,
    FEWInjector,
    band_snr,
    draw_sources,
    random_angles,
)
from .kernels import chirp_kernel, jax_chirp_kernel, tukey_chirp_kernel  # noqa: F401
from .optimize import (  # noqa: F401
    Adam,
    AdamDE,
    BucketedLoss,
    DifferentialEvolution,
    OptimizationResult,
    Optimizer,
    population_evaluator,
    population_map,
)
from .parallel import SearchTask, portable_result, run_tasks  # noqa: F401
from .pipeline import EMRISearch, SearchResult  # noqa: F401
from .psd import SensitivityPSD, TabulatedNoiseModel  # noqa: F401
from .recipes import InjectionFile, NoiseFile, RecipeMismatch  # noqa: F401
from .results import Background, ResultStore, StreamInfo, collect, false_alarm_rate  # noqa: F401
from .statistic import FresnelStatistic, SegmentMask, SemiCoherentStatistic, Statistic  # noqa: F401
from .tracks import (  # noqa: F401
    ChebyshevBasis,
    ChebyshevTrackModel,
    FEWHarmonicTrackModel,
    SVDBasis,
    SVDTrackModel,
    Track,
    TrackBank,
    backward_inspiral,
    few_track_bank,
    kerr_ecc_eq_inspiral,
)
