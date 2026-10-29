"""Running independent searches in parallel processes.

A search campaign (injections x seeds x durations, noise backgrounds) is many
independent optimisations, and one JAX process on a CPU keeps only a few cores
busy. :func:`run_tasks` runs picklable zero-argument tasks in ``n_workers``
spawned processes and yields their results as they finish; a
:class:`SearchTask` is one :meth:`~.pipeline.EMRISearch.run`. Tasks are built
lazily (``tasks`` may be a generator) and at most ``max_pending`` are in flight,
so their data need not all be in memory at once.

Each worker is a fresh interpreter (``spawn``): tasks must pickle, and a task
callable must be importable (a module-level function or class, not one defined
in a notebook). Every worker compiles its own JAX functions and holds its own
data, so memory grows with ``n_workers``.
"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from multiprocessing import get_context
from typing import Any, Iterable, Iterator, Optional, Tuple, cast

import numpy as np

from .optimize import OptimizationResult
from .pipeline import EMRISearch, SearchResult
from .tracks import Track


def _to_numpy(tree):
    if isinstance(tree, dict):
        return {k: _to_numpy(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_to_numpy(v) for v in tree)
    if hasattr(tree, "__array__") and not isinstance(tree, np.ndarray):
        return np.asarray(tree)
    return tree


def portable_result(result: SearchResult) -> SearchResult:
    """``result`` with every array as NumPy (cheap to send between processes)."""
    track = cast(Track, result.best_track)
    opt = result.optimization
    return result._replace(
        best_track=Track(*(None if a is None else np.asarray(a) for a in track)),
        optimization=OptimizationResult(
            np.asarray(opt.x), np.asarray(opt.loss), _to_numpy(opt.history)
        ),
    )


@dataclass
class SearchTask:
    """One search: ``search.run(T_pl, seed=seed)``, returned as ``(tag, result)``.

    ``tag`` is the caller's label for the result (e.g. its index in a results
    array); ``keep_history=False`` drops the optimiser history (but its
    ``"last_step"``) before the result is sent back.
    """

    search: EMRISearch
    T_pl: float
    seed: int = 0
    tag: Any = None
    keep_history: bool = True

    def __call__(self) -> Tuple[Any, SearchResult]:
        result = portable_result(self.search.run(self.T_pl, seed=self.seed))
        if not self.keep_history:
            opt = result.optimization
            last = {"last_step": opt.history.get("last_step")}
            result = result._replace(optimization=opt._replace(history=last))
        return self.tag, result


def _call(task):
    return task()


def run_tasks(tasks: Iterable, n_workers: int = 1, max_pending: Optional[int] = None) -> Iterator:
    """Run zero-argument ``tasks``; yield each one's return value as it finishes.

    ``n_workers=1`` runs them in this process, in order (no pickling).
    Otherwise ``n_workers`` spawned processes run them, at most
    ``max_pending`` (default ``2 * n_workers``) submitted at a time, and
    results come in completion order: label them (:attr:`SearchTask.tag`).
    """
    n_workers = int(n_workers)
    if n_workers <= 1:
        for task in tasks:
            yield task()
        return
    max_pending = int(max_pending or 2 * n_workers)
    tasks = iter(tasks)
    with ProcessPoolExecutor(n_workers, mp_context=get_context("spawn")) as pool:
        pending: set = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < max_pending:
                try:
                    pending.add(pool.submit(_call, next(tasks)))
                except StopIteration:
                    exhausted = True
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                yield future.result()
