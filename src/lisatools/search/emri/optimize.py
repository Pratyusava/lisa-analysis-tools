"""Population optimisers for the EMRI search (JAX).

An optimiser minimises ``loss(x)`` over a population of unit-cube points
``x`` (shape ``(n, ndim)``) via :meth:`Optimizer.run`, which returns an
:class:`OptimizationResult`. ``loss`` takes one point; optimisers vmap / grad
it themselves.

* :class:`DifferentialEvolution` -- gradient-free DE with greedy selection.
* :class:`AdamDE` -- Lorenzo Speri's hybrid: one Adam step on every member,
  then one DE proposal per member with greedy selection, per iteration.
"""

from dataclasses import dataclass, field
from typing import Callable, NamedTuple

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
    """Base class. ``save_every`` controls how often :attr:`history` is filled.

    All randomness comes from the ``key`` passed to :meth:`run`.
    """

    n_steps: int = 200
    save_every: int = 10

    def run(self, loss: Callable, x0, key) -> OptimizationResult:
        raise NotImplementedError


@dataclass
class DifferentialEvolution(Optimizer):
    """Gradient-free DE; per step ``F ~ U(F_range)`` and elitism w.p. ``p_elitism``."""

    F_range: tuple = (0.3, 1.5)
    p_elitism: float = 0.5

    def run(self, loss, x0, key):
        jax, jnp = _jax()
        batch_loss = jax.jit(jax.vmap(loss))
        x = jnp.clip(jnp.asarray(x0), 0.0, 1.0)
        fx = batch_loss(x)
        history = _new_history()
        for step in range(self.n_steps + 1):
            key, key_mut, key_de = jax.random.split(key, 3)
            elitism, F = de_settings(key_de, self.F_range, self.p_elitism)
            trial = jnp.clip(de_mutation(key_mut, x, fx, elitism, F), 0.0, 1.0)
            x, fx, improved = greedy_selection(x, fx, trial, batch_loss(trial))
            _record(history, step, self.save_every, x, fx, improved)
        return OptimizationResult(x, fx, history)


@dataclass
class AdamDE(Optimizer):
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
    F_range: tuple = (0.3, 1.5)
    p_elitism: float = 0.5
    stale_selection: bool = True

    def run(self, loss, x0, key):
        jax, jnp = _jax()
        value_and_grad = jax.vmap(jax.value_and_grad(loss))
        batch_loss = jax.jit(jax.vmap(loss))

        @jax.jit
        def adam_step(x, state):
            fx, g = value_and_grad(x)
            upd, state = jax.vmap(self.adam.update)(g, state)
            return jnp.clip(x + upd, 0.0, 1.0), state, fx

        x = jnp.asarray(x0)
        state = jax.vmap(self.adam.init)(x)
        history = _new_history()
        for step in range(self.n_steps + 1):
            key, key_mut, key_de = jax.random.split(key, 3)
            x, state, fx = adam_step(x, state)
            if not self.stale_selection:
                fx = batch_loss(x)
            elitism, F = de_settings(key_de, self.F_range, self.p_elitism)
            trial = jnp.clip(de_mutation(key_mut, x, fx, elitism, F), 0.0, 1.0)
            x, fx, improved = greedy_selection(x, fx, trial, batch_loss(trial))
            _record(history, step, self.save_every, x, fx, improved)
        if self.stale_selection:
            fx = batch_loss(x)  # rejected members kept their pre-Adam scores
        return OptimizationResult(x, fx, history)


def _new_history():
    return {"step": [], "loss": [], "x": [], "improved_fraction": []}


def _record(history, step, every, x, fx, improved):
    if step % every:
        return
    order = np.argsort(np.asarray(fx))
    history["step"].append(step)
    history["loss"].append(np.asarray(fx)[order])
    history["x"].append(np.asarray(x)[order])
    history["improved_fraction"].append(float(np.mean(np.asarray(improved))))
