"""Modular single-harmonic semi-coherent EMRI search.

Re-implements the search of Speri, Tenorio, Chapman-Bird & Gerosa (2026,
arXiv:2510.20891; github.com/lorenzosp/EMRI-Search) on LAT, split into
swappable parts: SFT data (:mod:`.data`), PSD (:mod:`.psd`), frequency-track
models (:mod:`.tracks`), statistics (:mod:`.statistic`, kernels in
:mod:`.kernels`), optimisers (:mod:`.optimize`) and the driver
(:mod:`.pipeline`).
"""

from .data import SFTData, gaussian_noise  # noqa: F401
from .kernels import chirp_kernel, jax_chirp_kernel, tukey_chirp_kernel  # noqa: F401
from .optimize import (  # noqa: F401
    Adam,
    AdamDE,
    DifferentialEvolution,
    OptimizationResult,
    Optimizer,
)
from .pipeline import EMRISearch, SearchResult  # noqa: F401
from .psd import SensitivityPSD, TabulatedNoiseModel  # noqa: F401
from .statistic import FresnelStatistic, SegmentMask, SemiCoherentStatistic  # noqa: F401
from .tracks import FEWHarmonicTrackModel, SVDBasis, SVDTrackModel, Track  # noqa: F401
