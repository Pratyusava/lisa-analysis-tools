"""Population optimisers for the EMRI search (JAX).

An optimiser minimises ``loss(x)`` over a population of unit-cube points
``x`` (shape ``(n, ndim)``) via :meth:`Optimizer.run`, which returns an
:class:`OptimizationResult`. ``loss`` takes one point; optimisers map / grad
it over the population themselves (:func:`population_evaluator`). A
:class:`BucketedLoss` lets each member be scored by the cheapest of several
exact implementations of the loss.

* :class:`DifferentialEvolution` -- gradient-free DE with greedy selection.
* :class:`AdamDE` -- Lorenzo Speri's hybrid: one Adam step on every member,
  then one DE proposal per member with greedy selection, per iteration.
"""

from dataclasses import dataclass, field
from typing import Callable, NamedTuple, Optional

import numpy as np


class OptimizationResult(NamedTuple):
    x: object  # final population, (n, ndim)
    loss: object  # its losses, (n,)
    history: dict  # per-saved-step snapshots (best losses / points)

    @property
    def best(self):
        i = int(np.argmin(np.asarray(self.loss)))
        return self.x[i], self.loss[i]


def _jax():
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    return jax, jnp


def population_map(fn, batch_size=None):
    """``fn`` (one point -> pytree) mapped over the population's leading axis.

    ``batch_size=None`` vectorises the whole population in one ``jax.vmap``;
    an integer evaluates it ``batch_size`` members at a time
    (``jax.lax.map``), so peak memory scales with ``batch_size`` instead of the
    population at about the same total cost. Members are independent, so the
    results are the same either way.
    """
    jax, _ = _jax()
    if batch_size is None:
        return jax.vmap(fn)
    batch_size = int(batch_size)
    return lambda x: jax.lax.map(fn, x, batch_size=batch_size)


@dataclass(frozen=True)
class BucketedLoss:
    """One loss with several exact implementations, picked per point.

    ``losses[bucket(x)](x)`` must equal ``losses[-1](x)`` (and so must its
    gradient) for every ``x``; ``losses`` go from cheapest to most general,
    and ``bucket`` (traceable) returns the cheapest one valid at ``x``. A JAX
    population cannot branch per member inside ``vmap``, so
    :func:`population_evaluator` sorts the members by bucket on the host and
    scores each group with its own compiled function, ``chunk_size`` members
    per call (one compiled shape per bucket; the last chunk is padded).
    Calling the object evaluates the general loss.
    """

    losses: tuple
    bucket: Callable
    chunk_size: Optional[int] = None

    def __call__(self, x):
        return self.losses[-1](x)


def population_evaluator(loss, transform=None, batch_size=None):
    """Compiled ``transform(loss)`` over a population, ``(n, ndim) -> (n, ...)``.

    ``transform`` maps a per-point function to another (e.g.
    ``jax.value_and_grad``; ``None``: the loss itself). A plain ``loss`` is
    mapped by :func:`population_map` with ``batch_size``. A
    :class:`BucketedLoss` is dispatched per bucket in chunks of its
    ``chunk_size`` members (default 16, or ``batch_size`` when smaller: small
    chunks waste less on padding, and a chunk bounds the memory as
    ``batch_size`` does).
    """
    jax, jnp = _jax()
    transform = transform or (lambda fn: fn)
    if not isinstance(loss, BucketedLoss):
        return jax.jit(population_map(transform(loss), batch_size))

    tree_map = jax.tree_util.tree_map
    chunk = int(loss.chunk_size or min(16, batch_size or 16))
    fns = [jax.jit(jax.vmap(transform(fn))) for fn in loss.losses]
    which = jax.jit(jax.vmap(loss.bucket))

    def evaluate(x):
        x = jnp.asarray(x)
        b = np.clip(np.asarray(which(x)), 0, len(fns) - 1)
        index, parts = [], []
        for j, fn in enumerate(fns):
            members = np.flatnonzero(b == j)
            for start in range(0, members.size, chunk):
                sel = members[start : start + chunk]
                padded = np.concatenate([sel, np.full(chunk - sel.size, sel[0])])
                parts.append(tree_map(lambda r, m=sel.size: r[:m], fn(x[padded])))
                index.append(sel)
        merged = tree_map(lambda *r: jnp.concatenate(r), *parts)
        inverse = np.argsort(np.concatenate(index))
        return tree_map(lambda r: r[inverse], merged)

    return evaluate


@dataclass
class Adam:
    """Adam (Kingma & Ba), same update as ``optax.adam`` with ``eps_root=0``."""

    learning_rate: float = 1e-5
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8

    def init(self, x):
        _, jnp = _jax()
        return (jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros((), dtype=jnp.int32))

    def update(self, grads, state):
        _, jnp = _jax()
        mu, nu, count = state
        mu = self.b1 * mu + (1 - self.b1) * grads
        nu = self.b2 * nu + (1 - self.b2) * grads**2
        count = count + 1
        mu_hat = mu / (1 - self.b1 ** count.astype(mu.dtype))
        nu_hat = nu / (1 - self.b2 ** count.astype(nu.dtype))
        return -self.learning_rate * mu_hat / (jnp.sqrt(nu_hat) + self.eps), (mu, nu, count)


def de_mutation(key, population, fitness, elitism, differential_weight, num_diff=1):
    """One DE/{best,rand}/num_diff mutant per member, ``a + F (b - c)``.

    ``a`` is the fittest member when ``elitism`` else a random other member;
    ``b, c`` are distinct members different from the member and from ``a``.
    No crossover (every coordinate takes the mutant).
    """
    jax, jnp = _jax()
    n, _ = population.shape
    best = jnp.argmin(fitness)

    def mutate(key, i):
        key_a, _, _, key_ab = jax.random.split(key, 4)
        p = jnp.ones(n).at[i].set(0.0)
        a_idx = jnp.where(elitism, best, jax.random.choice(key_a, n, p=p))
        mutant = population[a_idx]
        p_diff = p.at[a_idx].set(0.0)
        for _ in range(num_diff):
            key_ab, sub = jax.random.split(key_ab)
            bc = jax.random.choice(sub, n, (2,), replace=False, p=p_diff)
            mutant = mutant + differential_weight * (population[bc[0]] - population[bc[1]])
            p_diff = p_diff.at[bc].set(0.0)
        return mutant

    return jax.vmap(mutate)(jax.random.split(key, n), jnp.arange(n))


def de_settings(key, F_range, p_elitism):
    """Per-step DE settings from ``key``: ``(elitism, F)``, ``elitism`` true
    w.p. ``p_elitism`` and ``F ~ U(F_range)``."""
    jax, _ = _jax()
    key_e, key_F = jax.random.split(key)
    elitism = jax.random.uniform(key_e) < p_elitism
    F = jax.random.uniform(key_F, minval=F_range[0], maxval=F_range[1])
    return elitism, F


def greedy_selection(x, loss, x_trial, loss_trial):
    """Keep each trial that lowers its member's loss."""
    _, jnp = _jax()
    improved = loss_trial < loss
    return jnp.where(improved[:, None], x_trial, x), jnp.where(improved, loss_trial, loss), improved


@dataclass
class Optimizer:
    """Base class. ``save_every`` controls how often :attr:`history` is filled;
    ``batch_size`` evaluates the population that many members at a time
    (:func:`population_evaluator`; ``None``: all at once). ``loss`` may be a
    :class:`BucketedLoss`.

    ``patience`` stops early: after ``patience`` steps in which the best
    loss has not dropped by more than ``min_improvement`` (relative) below the
    last best that did; ``None`` (default) always runs ``n_steps``. The last
    step run is ``history["last_step"]``.

    All randomness comes from the ``key`` passed to :meth:`run`.
    """

    n_steps: int = 200
    save_every: int = 10
    batch_size: Optional[int] = None
    patience: Optional[int] = None
    min_improvement: float = 1e-3

    def run(self, loss: Callable, x0, key) -> OptimizationResult:
        raise NotImplementedError

    def _stopper(self):
        """``done(step, fx) -> bool`` for one run (:attr:`patience`)."""
        if self.patience is None:
            return lambda step, fx: False
        state = {"best": np.inf, "since": 0}

        def done(step, fx):
            best = float(np.min(np.asarray(fx)))
            if best < state["best"] - self.min_improvement * abs(state["best"]) or step == 0:
                state["best"], state["since"] = best, 0
            else:
                state["since"] += 1
            return state["since"] >= self.patience

        return done


@dataclass
class _DEOptimizer(Optimizer):
    """Shared DE proposal: per step ``F ~ U(F_range)``, elitism w.p. ``p_elitism``,
    one mutant per member (clipped to the unit cube), greedy selection."""

    F_range: tuple = (0.3, 1.5)
    p_elitism: float = 0.5

    def _de_step(self, keys, x, fx, batch_loss):
        """``keys = (key_mut, key_de)``; returns ``(x, fx, improved)``."""
        _, jnp = _jax()
        key_mut, key_de = keys
        elitism, F = de_settings(key_de, self.F_range, self.p_elitism)
        trial = jnp.clip(de_mutation(key_mut, x, fx, elitism, F), 0.0, 1.0)
        return greedy_selection(x, fx, trial, batch_loss(trial))


@dataclass
class DifferentialEvolution(_DEOptimizer):
    """Gradient-free DE; per step ``F ~ U(F_range)`` and elitism w.p. ``p_elitism``."""

    def run(self, loss, x0, key):
        jax, jnp = _jax()
        batch_loss = population_evaluator(loss, batch_size=self.batch_size)
        x = jnp.clip(jnp.asarray(x0), 0.0, 1.0)
        fx = batch_loss(x)
        history = _new_history()
        done = self._stopper()
        for step in range(self.n_steps + 1):
            key, *keys = jax.random.split(key, 3)
            x, fx, improved = self._de_step(keys, x, fx, batch_loss)
            stop = done(step, fx) or step == self.n_steps
            _record(history, step, self.save_every, x, fx, improved, last=stop)
            if stop:
                break
        return OptimizationResult(x, fx, history)


@dataclass
class AdamDE(_DEOptimizer):
    """Lorenzo Speri's Adam + DE hybrid (EMRI-Search ``QuickStartEMRIsearch``).

    Each iteration: one Adam step on every member (clipped to the unit cube),
    then one DE mutant per member, kept if it beats the member's loss.

    ``stale_selection=True`` reproduces the reference loop, which compares the
    trials against the losses evaluated *before* that iteration's Adam step
    (the Adam-moved point is kept or replaced on a one-step-old score).
    ``False`` re-scores after the Adam step (one more loss evaluation).
    Either way the returned population is re-scored at the end, so
    ``result.loss`` (and :attr:`OptimizationResult.best`) is the loss *at*
    ``result.x``; with stale selection, :attr:`history` keeps the scores the
    selection used.
    """

    adam: Adam = field(default_factory=Adam)
    stale_selection: bool = True

    def run(self, loss, x0, key):
        jax, jnp = _jax()
        value_and_grad = population_evaluator(loss, jax.value_and_grad, self.batch_size)
        batch_loss = population_evaluator(loss, batch_size=self.batch_size)

        @jax.jit
        def update(x, g, state):
            upd, state = jax.vmap(self.adam.update)(g, state)
            return jnp.clip(x + upd, 0.0, 1.0), state

        def adam_step(x, state):
            fx, g = value_and_grad(x)
            x, state = update(x, g, state)
            return x, state, fx

        x = jnp.asarray(x0)
        state = jax.vmap(self.adam.init)(x)
        history = _new_history()
        done = self._stopper()
        for step in range(self.n_steps + 1):
            key, *keys = jax.random.split(key, 3)
            x, state, fx = adam_step(x, state)
            if not self.stale_selection:
                fx = batch_loss(x)
            x, fx, improved = self._de_step(keys, x, fx, batch_loss)
            stop = done(step, fx) or step == self.n_steps
            _record(history, step, self.save_every, x, fx, improved, last=stop)
            if stop:
                break
        if self.stale_selection:
            fx = batch_loss(x)  # rejected members kept their pre-Adam scores
        return OptimizationResult(x, fx, history)


def _new_history():
    return {"step": [], "loss": [], "x": [], "improved_fraction": [], "last_step": None}


def _record(history, step, every, x, fx, improved, last=False):
    if last:
        history["last_step"] = step
    if step % every and not last:
        return
    order = np.argsort(np.asarray(fx))
    history["step"].append(step)
    history["loss"].append(np.asarray(fx)[order])
    history["x"].append(np.asarray(x)[order])
    history["improved_fraction"].append(float(np.mean(np.asarray(improved))))
