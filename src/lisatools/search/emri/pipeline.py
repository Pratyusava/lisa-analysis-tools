"""The modular single-harmonic EMRI search.

:class:`EMRISearch` composes four swappable parts:

* data       -- a data container (:class:`~.data.SFTData`, :class:`~.data.FDData`)
* statistic  -- a :class:`~.statistic.Statistic` on that data (what a track scores)
* tracks     -- a track model, ``model(x, shift) -> Track`` over the unit cube
* optimizer  -- an :class:`~.optimize.Optimizer` (how the cube is explored)

and runs one optimisation per trial plunge time. Any part can be replaced by
another object with the same interface.
"""

import warnings
from typing import NamedTuple

import numpy as np

from .optimize import OptimizationResult, Optimizer, population_map
from .statistic import Statistic


class SearchResult(NamedTuple):
    T_pl: float
    shift: int
    best_x: np.ndarray
    best_statistic: float
    best_track: object
    optimization: OptimizationResult


class EMRISearch:
    """Maximise ``statistic(data, track_model(x, shift))`` over the unit cube.

    Args:
        data: The data container the statistic reads (:meth:`~.statistic.Statistic.prepare`).
        statistic: Detection statistic on the JAX backend (the optimisers
            differentiate and vmap it).
        track_model: Callable ``(x, shift) -> Track`` on the JAX backend, with ``ndim``,
            ``shift_for_plunge(T_pl)`` and ``initial_population(n, seed=...)``.
        optimizer: Population optimiser (JAX).
        n_initial: Initial draws before keeping the best ``n_population`` (scored in
            groups of ``optimizer.batch_size`` when it is set).
        n_population: Population the optimiser works on.
        initial_kwargs: Forwarded to ``track_model.initial_population`` (e.g.
            ``f0_range``).
    """

    def __init__(
        self,
        data,
        statistic: Statistic,
        track_model,
        optimizer: Optimizer,
        n_initial=128,
        n_population=64,
        initial_kwargs=None,
    ):
        for name, part in (("statistic", statistic), ("track model", track_model)):
            if not part.backend.name.endswith("jax"):
                raise ValueError(
                    f"The {name} is on the {part.backend.name} backend; the optimisers "
                    'trace it with JAX, so build it with force_backend="jax".'
                )
        self.data = data
        self.statistic = statistic
        self.track_model = track_model
        self.optimizer = optimizer
        self.n_initial = int(n_initial)
        self.n_population = int(n_population)
        self.initial_kwargs = dict(initial_kwargs or {})
        self._check_band()

    def _check_band(self):
        """Warn when tracks the model can produce would need SFT bins that are not stored.

        Uses ``track_model.frequency_range()`` and ``statistic.band(n_bins)``
        when both exist; segments below the mask's ``f_min`` never count.
        """
        reach = getattr(self.track_model, "frequency_range", None)
        band = getattr(self.statistic, "band", None)
        if reach is None or band is None:
            return
        lo, hi = reach()
        f_min = getattr(self.statistic.mask, "f_min", None)
        lo = max(lo, f_min if f_min is not None else 0.0, 0.0)
        b_lo, b_hi = band(self.data.sfts.shape[-1])
        if lo < b_lo or hi > b_hi:
            warnings.warn(
                f"Tracks reach [{lo:.4g}, {hi:.4g}] Hz but the stored SFT bins only cover the "
                f"statistic's bin window for [{b_lo:.4g}, {b_hi:.4g}] Hz; missing bins count as "
                "zero and lower the statistic near the band edges. Widen min_freq/max_freq by "
                "P / T_sft.",
                stacklevel=3,
            )

    def loss_function(self, arr, shift):
        """``x -> -statistic`` for one trial plunge (the optimiser's objective);
        ``arr`` is the statistic's data array (:meth:`~.statistic.Statistic.prepare`)."""
        stat, model = self.statistic, self.track_model

        def loss(x):
            return -stat.evaluate(arr, model(x, shift))

        return loss

    def run(self, T_pl, seed=0, data=None, x0=None) -> SearchResult:
        """Search one trial plunge time ``T_pl`` [s].

        ``data`` overrides the data container (e.g. a noise-only realisation
        for the background); ``x0`` overrides the initial population.
        """
        import jax

        shift = self.track_model.shift_for_plunge(T_pl)
        arr = self.statistic.prepare(self.data if data is None else data)
        loss = self.loss_function(arr, shift)

        if x0 is None:
            x0 = self.track_model.initial_population(
                self.n_initial, seed=seed, **self.initial_kwargs
            )
            x0 = self.statistic.xp.asarray(x0)
            batch_size = getattr(self.optimizer, "batch_size", None)
            initial_loss = jax.jit(population_map(loss, batch_size))(x0)
            x0 = x0[np.argsort(np.asarray(initial_loss))[: self.n_population]]

        result = self.optimizer.run(loss, x0, jax.random.PRNGKey(seed))
        best_x, best_loss = result.best
        return SearchResult(
            T_pl=T_pl,
            shift=shift,
            best_x=np.asarray(best_x),
            best_statistic=-float(best_loss),
            best_track=self.track_model(best_x, shift),
            optimization=result,
        )

    def scan(self, T_pl_grid, seed=0, data=None):
        """One :meth:`run` per trial plunge time."""
        return [self.run(T_pl, seed=seed + i, data=data) for i, T_pl in enumerate(T_pl_grid)]
