"""Modular single-harmonic semi-coherent EMRI search.

Re-implements the search of Speri, Tenorio, Chapman-Bird & Gerosa (2026,
arXiv:2510.20891; github.com/lorenzosp/EMRI-Search) on LAT, split into
swappable parts: data containers (:mod:`.data`), injections
(:mod:`.injection`), PSD (:mod:`.psd`), frequency-track models
(:mod:`.tracks`), statistics (semi-coherent in :mod:`.statistic` with kernels
in :mod:`.kernels`; coherent delay-scanning, incl. Cutler &
Bustamante-Rosell's ``H``, in :mod:`.excess_power`), optimisers
(:mod:`.optimize`) and the driver (:mod:`.pipeline`).
"""

from .data import FDData, SFTData, gaussian_noise  # noqa: F401
from .excess_power import (  # noqa: F401
    DelayCorrelationStatistic,
    ExcessPowerStatistic,
    MatchedFilterStatistic,
)
from .injection import EMRIInjection, FEWInjector, band_snr, random_angles  # noqa: F401
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
from .statistic import (  # noqa: F401
    FresnelStatistic,
    SegmentMask,
    SemiCoherentStatistic,
    Statistic,
)
from .tracks import (  # noqa: F401
    FEWHarmonicTrackModel,
    SVDBasis,
    SVDTrackModel,
    Track,
    backward_inspiral,
    kerr_ecc_eq_inspiral,
)
