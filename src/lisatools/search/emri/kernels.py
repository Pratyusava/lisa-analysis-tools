"""Linear-chirp kernels for semi-coherent SFT statistics.

The kernel of a segment of length ``T`` for a unit-amplitude linear chirp of
start frequency ``f0`` (relative to an SFT bin) and drift ``f1`` is

    K(f0, f1, T) = int_0^T exp[2 pi i (f0 tau + f1 tau^2 / 2)] dtau,

a difference of Fresnel integrals for ``f1 != 0`` and a Dirichlet kernel for
``f1 == 0`` (Tenorio & Gerosa 2025; Speri et al. 2026, arXiv:2510.20891).

The Fresnel form loses ~``v^2 * 1e-16`` relative accuracy at small ``f1``
(``v ~ f0 sqrt(2 / f1)``; ~1e-10 of ``T`` at ``f1 = 1e-13`` a hundred bins off
the carrier). ``small_drift=True`` instead evaluates segments with a small
quadratic phase, ``|pi f1 T^2| <= SMALL_DRIFT_EPS``, by a power series around
the Dirichlet kernel: double precision everywhere and the exact ``d/df1`` at
``f1 = 0`` (the default gives 0 there), at ~3x the cost of a value and ~13x
of a value + gradient in JAX.

Each branch only ever sees inputs that are safe for it, so the NumPy form
(:func:`chirp_kernel`) and the JAX form (:func:`jax_chirp_kernel`) are plain
functions and JAX's autodiff never meets an inf / NaN. The JAX form imports
JAX lazily so this module imports without it.

LAT's C++ ``STFTFresnel`` (``cutils/domains.cu``, ``STFTFresnelWrap``) evaluates
the same kernel (``amp e^{i phase0} K(f0 - f, fdot, T)``). By default it uses a
rational Fresnel approximation (Abramowitz & Stegun 7.3.32-33; up to ~5 % per
bin, ~0.4 % in summed power at ``T = 5e4`` s); ``exact_fresnel=True`` switches
to a Cephes port that matches the Fresnel branch here. It has no small-drift
branch and is not differentiable, so the kernels here are kept for the
statistic.
"""

import numpy as np
from scipy import special as _sp_special

from ...utils.utility import get_array_module

SMALL_DRIFT_EPS = 0.5
"""Largest ``|pi f1 T^2|`` [rad] the ``small_drift`` series evaluates."""
_N_DRIFT_TERMS = 14  # eps^n / n! for n <= 14: truncation < 1e-17 at eps = 0.5
_N_MOMENT_TERMS = 64  # terms of the series for the top moment J_{2N}


def _dirichlet(f0, T, xp):
    return T * xp.exp(1j * xp.pi * f0 * T) * xp.sinc(f0 * T)


def _fresnel_positive(f0, f1_abs, T, xp, fresnel):
    """Kernel for ``f1 > 0`` (``f1_abs`` must be positive)."""
    quot = f0 / f1_abs
    factor = xp.sqrt(2 * f1_abs)
    S_lo, C_lo = fresnel(factor * quot)
    S_hi, C_hi = fresnel(factor * (quot + T))
    diff = (C_hi - C_lo) + 1j * (S_hi - S_lo)
    return xp.exp(-1j * xp.pi * f0 * quot) * diff / factor


def _moments(theta, n_max, xp):
    """``J_m(theta) = int_0^1 x^m e^{i theta x} dx`` for ``m = 0..n_max``.

    Upward recurrence ``J_m = (e^{i theta} - m J_{m-1}) / (i theta)`` where it
    is stable (``m <= |theta|``), downward recurrence
    ``J_{m-1} = (e^{i theta} - i theta J_m) / m`` from a series for
    ``J_{n_max}`` elsewhere. Inputs to the unused direction are replaced by
    safe values so neither ever produces inf / NaN.
    """
    abs_theta = xp.abs(theta)
    e = xp.exp(1j * theta)
    J0 = xp.exp(0.5j * theta) * xp.sinc(theta / (2 * xp.pi))

    theta_up = xp.where(abs_theta >= 1.0, theta, 1.0)
    e_up = xp.exp(1j * theta_up)
    up = [J0]
    for m in range(1, n_max + 1):
        up.append((e_up - m * up[-1]) / (1j * theta_up))

    # J_m = e^{i theta} sum_k (-i theta)^k m! / (m + k + 1)!, convergent without
    # cancellation for |theta| < m + 1.
    theta_dn = xp.where(abs_theta < n_max, theta, 0.0)
    e_dn = xp.exp(1j * theta_dn)
    term = xp.ones_like(e_dn) / (n_max + 1)
    top = term
    for k in range(_N_MOMENT_TERMS):
        term = term * (-1j * theta_dn) / (n_max + k + 2)
        top = top + term
    down = [e_dn * top]
    for m in range(n_max, 0, -1):
        down.append((e_dn - 1j * theta_dn * down[-1]) / m)
    down = down[::-1]

    return [J0] + [xp.where(m <= abs_theta, up[m], down[m]) for m in range(1, n_max + 1)]


def _small_drift_kernel(f0, f1, T, xp):
    """``K = T sum_n (i eps)^n / n! J_{2n}(2 pi f0 T)``, ``eps = pi f1 T^2``."""
    eps = xp.pi * f1 * T**2
    J = _moments(2 * xp.pi * f0 * T, 2 * _N_DRIFT_TERMS, xp)
    total = J[0]
    coeff = xp.ones_like(eps) + 0j
    for n in range(1, _N_DRIFT_TERMS + 1):
        coeff = coeff * (1j * eps) / n
        total = total + coeff * J[2 * n]
    return T * total


def _chirp_kernel(f0, f1, T, xp, fresnel, small_drift):
    if small_drift:
        near = xp.abs(xp.pi * f1 * T**2) <= SMALL_DRIFT_EPS
        f1_safe = 2.0 * SMALL_DRIFT_EPS / (xp.pi * T**2)
    else:
        near = f1 == 0.0
        f1_safe = 1.0 / T**2
    # Each branch gets inputs it can evaluate; the other branch's are swapped
    # for harmless values, so no inf / NaN reaches a value or a gradient.
    f1_big = xp.where(near, f1_safe, f1)
    f1_abs = xp.abs(f1_big)
    big = xp.where(
        f1_big > 0,
        _fresnel_positive(f0, f1_abs, T, xp, fresnel),
        _fresnel_positive(-f0, f1_abs, T, xp, fresnel).conj(),
    )
    if small_drift:
        near_value = _small_drift_kernel(f0, xp.where(near, f1, 0.0), T, xp)
    else:
        near_value = _dirichlet(f0, T, xp)
    return xp.where(near, near_value, big)


def chirp_kernel(f0, f1, T, small_drift=False):
    """NumPy linear-chirp kernel ``K(f0, f1, T)`` (rectangular window).

    ``small_drift=True`` uses the double-precision series for
    ``|pi f1 T^2| <= SMALL_DRIFT_EPS`` (see the module docstring).
    """
    f0, f1 = np.asarray(f0, dtype=float), np.asarray(f1, dtype=float)
    return _chirp_kernel(f0, f1, T, np, _sp_special.fresnel, small_drift)


def tukey_chirp_kernel(f0, f1, T, alpha, kernel=chirp_kernel):
    """Linear-chirp kernel under a Tukey window of taper fraction ``alpha``.

    Exact decomposition of the windowed integral into rectangular kernels:
    a central plateau plus a cosine taper on each side, each taper being a
    sum of three shifted kernels of length ``Delta = alpha T / 2``.
    ``kernel`` selects the rectangular kernel (NumPy or JAX form).
    """
    if alpha <= 0.0:
        return kernel(f0, f1, T)
    xp = get_array_module(f0)
    delta = alpha * T / 2
    nu = 1 / (2 * delta)
    k_central = kernel(f0, f1, T - delta) - kernel(f0, f1, delta)
    k_left = (
        0.5 * kernel(f0, f1, delta)
        - 0.25 * kernel(f0 + nu, f1, delta)
        - 0.25 * kernel(f0 - nu, f1, delta)
    )
    g0 = -f0 - f1 * T
    k_right = xp.exp(2j * xp.pi * (f0 * T + 0.5 * f1 * T**2)) * (
        0.5 * kernel(g0, f1, delta)
        - 0.25 * kernel(g0 + nu, f1, delta)
        - 0.25 * kernel(g0 - nu, f1, delta)
    )
    return k_central + k_left + k_right


def jax_chirp_kernel(f0, f1, T, small_drift=False):
    """JAX linear-chirp kernel, differentiable in ``f0`` and ``f1`` (``T`` a float).

    ``small_drift`` as in :func:`chirp_kernel`.
    """
    import jax.numpy as jnp
    from jax.scipy import special as jsp_special

    f0, f1 = jnp.asarray(f0), jnp.asarray(f1)
    return _chirp_kernel(f0, f1, T, jnp, jsp_special.fresnel, small_drift)
