"""Frequency-track models for the semi-coherent EMRI search.

A track is the frequency evolution of one harmonic sampled at the SFT segment
starts: ``f_alpha``, ``fdot_alpha`` and optionally ``fddot_alpha``. A track
model maps search parameters to a :class:`Track`; the statistic only ever sees
the :class:`Track`, so models (physical FEW trajectories, SVD bases, ...) swap
freely.
"""

from typing import NamedTuple, Optional

import numpy as np
from scipy.stats import qmc

from ...utils.parallelbase import LISAToolsParallelModule


class Track(NamedTuple):
    """One harmonic's frequency track on the segment grid (a JAX pytree).

    ``fddot`` may be ``None``; consumers then difference ``fdot`` themselves.
    """

    f: object
    fdot: object
    fddot: Optional[object] = None


class SVDBasis:
    """Reduced basis for frequency tracks (rows of ``V`` are orthonormal).

    A track is ``coeffs @ V`` with ``coeffs`` inside the box ``[lb, ub]``; the
    search parametrises ``coeffs`` by the unit cube.

    Args:
        V: Basis, shape ``(n_modes, n_samples)``.
        lb, ub: Coefficient box, shape ``(n_modes,)``.
        S: Singular values (diagnostic only).
    """

    def __init__(self, V, lb, ub, S=None):
        self.V = np.asarray(V, dtype=float)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.S = None if S is None else np.asarray(S, dtype=float)

    @property
    def n_modes(self) -> int:
        return self.V.shape[0]

    @property
    def n_samples(self) -> int:
        return self.V.shape[1]

    def frequency_range(self, sample=0):
        """``(min, max)`` of ``(coeffs @ V)[sample]`` over the coefficient box [Hz].

        ``sample=0`` is the range of starting frequencies the basis can
        represent (before any plunge shift); tracks outside it cannot be
        reached from inside the unit cube.
        """
        terms = np.stack([self.lb, self.ub]) * self.V[:, sample]
        return float(terms.min(axis=0).sum()), float(terms.max(axis=0).sum())

    @classmethod
    def build(cls, tracks, n_modes):
        """SVD of a training bank of tracks, shape ``(n_tracks, n_samples)``.

        The coefficient box is the min/max of the training coefficients.
        """
        tracks = np.asarray(tracks, dtype=float)
        U, S, Vt = np.linalg.svd(tracks, full_matrices=False)
        coeffs = U[:, :n_modes] * S[:n_modes]
        return cls(Vt[:n_modes], coeffs.min(axis=0), coeffs.max(axis=0), S=S[:n_modes])

    @classmethod
    def from_npz(cls, path):
        """Load Lorenzo Speri's EMRI-Search basis files (``V_K``, ``lb``, ``ub``)."""
        d = np.load(path, allow_pickle=True)
        S = np.diag(d["S_K"]) if "S_K" in d.files else None
        return cls(d["V_K"], d["lb"], d["ub"], S=S)

    def save_npz(self, path):
        np.savez(
            path,
            V_K=self.V,
            lb=self.lb,
            ub=self.ub,
            S_K=np.diag(self.S) if self.S is not None else np.zeros(0),
        )


class SVDTrackModel(LISAToolsParallelModule):
    """Track = basis expansion, shifted in time to a trial plunge.

    The basis tracks all plunge at ``T_ref``; a trial plunge time ``T_pl`` rolls
    the expansion by ``round_toward_zero((T_pl - T_ref) / T_sft)`` segments,
    zero-filling the vacated samples, and truncates to the data's segments.
    ``fdot`` is the central difference of ``f`` over the segment times.

    Args:
        basis: The :class:`SVDBasis`.
        t_alpha: Segment start times [s], length ``n_segments <= basis.n_samples``.
        T_ref: Plunge time the basis tracks share [s].
        force_backend: ``"cpu"`` (NumPy) or ``"jax"``.
    """

    def __init__(self, basis: SVDBasis, t_alpha, T_ref, force_backend=None):
        super().__init__(force_backend=force_backend)
        self.basis = basis
        self.t_alpha = np.asarray(t_alpha, dtype=float)
        self.T_ref = float(T_ref)
        if self.t_alpha.size > basis.n_samples:
            raise ValueError(
                f"{self.t_alpha.size} segments but the basis only spans {basis.n_samples}."
            )

    @classmethod
    def supported_backends(cls):
        return [cls._BACKEND_PREFIX + "_" + t for t in ("cpu", "jax")]

    @property
    def ndim(self) -> int:
        return self.basis.n_modes

    @property
    def T_sft(self) -> float:
        return float(self.t_alpha[1] - self.t_alpha[0])

    def shift_for_plunge(self, T_pl) -> int:
        """Segment shift that moves the basis plunge ``T_ref`` to ``T_pl`` [s]."""
        return int((T_pl - self.T_ref) / self.T_sft)

    def frequency_range(self):
        """``(min, max)`` frequency [Hz] any track can take at any basis sample
        (any shift brings any sample into the data)."""
        terms = np.stack([self.basis.lb, self.basis.ub])[:, :, None] * self.basis.V[None]
        return float(terms.min(axis=0).sum(axis=0).min()), float(
            terms.max(axis=0).sum(axis=0).max()
        )

    def coeffs(self, xi):
        """Unit-cube point(s) -> basis coefficients."""
        xp = self.xp
        return xp.asarray(self.basis.lb) + xi * xp.asarray(self.basis.ub - self.basis.lb)

    def unit(self, coeffs):
        """Basis coefficients -> unit-cube point(s)."""
        return (coeffs - self.basis.lb) / (self.basis.ub - self.basis.lb)

    def __call__(self, xi, shift=0) -> Track:
        xp = self.xp
        f = self.coeffs(xi) @ xp.asarray(self.basis.V)
        f = _roll_and_zero(f, shift, xp)[: self.t_alpha.size]
        fdot = xp.gradient(f, xp.asarray(self.t_alpha))
        return Track(f, fdot)

    def initial_population(self, n, f0_range=(5e-4, 1e-2), seed=None, out_of_cube="keep"):
        """Space-filling unit-cube points whose tracks start in ``f0_range``.

        Sobol draws in the coefficient box, each track's offset re-set so it
        starts at a uniform ``f0`` in ``f0_range``; draws with any negative
        frequency are dropped, so fewer than ``n`` points may come back.

        Re-setting the offset moves most points out of the unit cube, which the
        optimisers then clip on their first step. ``out_of_cube`` sets what
        happens here:

        * ``"keep"`` (EMRI-Search behaviour): return them unclipped, so the
          screening ranks tracks that clipping later changes.
        * ``"clip"``: clip into the cube before the positivity filter, so the
          screening ranks the tracks that get optimised (their ``f0`` is then
          no longer exactly the drawn one).
        """
        if out_of_cube not in ("keep", "clip"):
            raise ValueError(f"out_of_cube must be 'keep' or 'clip', not {out_of_cube!r}.")
        rng = np.random.default_rng(seed)
        unit = qmc.Sobol(d=self.ndim, rng=rng).random(n)
        V = self.basis.V
        f = (self.basis.lb + unit * (self.basis.ub - self.basis.lb)) @ V
        f = f - f[:, :1] + rng.uniform(*f0_range, size=(n, 1))
        coeffs = f @ V.T
        if out_of_cube == "keep":
            keep = np.all(coeffs @ V >= 0.0, axis=1)
            return self.unit(coeffs[keep])
        unit = np.clip(self.unit(coeffs), 0.0, 1.0)
        keep = np.all((self.basis.lb + unit * (self.basis.ub - self.basis.lb)) @ V >= 0.0, axis=1)
        return unit[keep]


class FEWHarmonicTrackModel(LISAToolsParallelModule):
    """Track of harmonic ``m Phi_phi + n Phi_r`` from a FEW Kerr eccentric
    equatorial inspiral, integrated backwards from plunge (NumPy only).

    Parameters ``(m1, m2, a, T_pl, e_f)``: source masses [M_sun] (detector frame),
    spin, time from ``t = 0`` to plunge [yr], eccentricity at the separatrix.
    The inspiral starts at ``p_sep(a, e_f) + buffer`` and runs back ``T_pl``;
    segments before the backward-integrated start or after plunge get zeros.
    Mirrors EMRI-Search's ``get_f_fdot_fddot_back``.

    Args:
        t_alpha: Segment start times [s].
        m, n: Harmonic numbers (``m = 2, n = 0`` is the dominant one).
        err: Integrator tolerance.
    """

    def __init__(self, t_alpha, m=2, n=0, err=1e-11, force_backend="cpu"):
        super().__init__(force_backend=force_backend)
        self.t_alpha = np.asarray(t_alpha, dtype=float)
        self.m, self.n = int(m), int(n)
        self.err = float(err)

    @classmethod
    def supported_backends(cls):
        return [cls._BACKEND_PREFIX + "_cpu"]

    @property
    def traj(self):
        if getattr(self, "_traj", None) is None:
            from few.trajectory.inspiral import EMRIInspiral
            from few.trajectory.ode import KerrEccEqFlux

            self._traj = EMRIInspiral(func=KerrEccEqFlux)
        return self._traj

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_traj", None)
        return state

    def phases_and_derivatives(self, m1, m2, a, T_pl, e_f):
        """``(phase, f, fdot, fddot)`` per orbital phase, each ``(2, n_segments)``
        for ``(Phi_phi, Phi_r)``; ``f`` etc. are orbital frequencies (no ``m, n``)."""
        from few.utils.geodesic import get_separatrix

        from ...sources.emri.wdm_direct import _phase_columns
        from ...utils.constants import YRSID_SI

        traj = self.traj
        gen = traj.inspiral_generator
        p_f = gen.func.separatrix_buffer_dist + get_separatrix(a, e_f, 1.0) + 1e-6
        traj(m1, m2, a, p_f, e_f, 1.0, T=T_pl, integrate_backwards=True, err=self.err)

        tau = T_pl * YRSID_SI - self.t_alpha  # time before plunge
        ok = (tau > 0.0) & (tau < traj.integrator_spline_t.max())
        out = np.zeros((4, 2, self.t_alpha.size))
        for order in range(4):
            # _phase_columns -> (Phi_phi, Phi_theta, Phi_r); keep phi and r.
            cols = _phase_columns(gen, tau[ok], order)[:, [0, 2]].T
            # d/dt = -d/dtau: odd derivative orders flip sign.
            out[order][:, ok] = cols if order == 0 else (-1) ** order * cols / (2 * np.pi)
        return out

    def __call__(self, params) -> Track:
        m1, m2, a, T_pl, e_f = params
        _, f, fdot, fddot = self.phases_and_derivatives(m1, m2, a, T_pl, e_f)
        w = np.array([self.m, self.n])
        return Track(w @ f, w @ fdot, w @ fddot)


def _roll_and_zero(f, shift, xp):
    """Roll along the last axis by ``shift`` and zero the wrapped-in samples."""
    rolled = xp.roll(f, shift, axis=-1)
    idx = xp.arange(f.shape[-1])
    n = f.shape[-1]
    wrapped = xp.where(shift > 0, idx < shift, False) | xp.where(shift < 0, idx >= n + shift, False)
    return xp.where(wrapped, 0.0, rolled)
