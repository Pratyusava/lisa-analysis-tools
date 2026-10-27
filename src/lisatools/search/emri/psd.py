"""Noise PSDs for the EMRI search, through LAT's sensitivity framework.

A statistic needs ``S(f)`` on the SFT bins and at the continuous track
frequency inside the differentiated objective, so the search sees a PSD as a
callable ``psd(f)`` that accepts NumPy or JAX arrays. :class:`SensitivityPSD`
provides it from any LAT :class:`~lisatools.sensitivity.Sensitivity` (e.g.
:class:`~lisatools.sensitivity.A2TDISens`, analytic TDI2 + optional galactic
foreground via ``stochastic_*`` kwargs); a tabulated curve enters as a LAT noise
model with ``Sn_spl`` (:class:`TabulatedNoiseModel`).
"""

import numpy as np
from scipy.interpolate import CubicSpline

from ...detector import sangria
from ...sensitivity import A2TDISens
from ...utils.utility import get_array_module


class SensitivityPSD:
    """``f -> sens_fn.get_Sn(f, model, **kwargs)``, held constant below ``f_floor``.

    The floor keeps TDI2's ``S(f) -> 0`` at ``f -> 0`` out of the statistic's
    denominators; ``get_Sn`` is evaluated on ``max(|f|, f_floor)`` so the
    discarded branch cannot poison a JAX gradient.

    Args:
        sens_fn: LAT :class:`~lisatools.sensitivity.Sensitivity` class (channel).
        model: Noise model passed to ``get_Sn`` (a ``LISAModel``, a stock name,
            or a :class:`TabulatedNoiseModel`).
        f_floor: Frequency below which the PSD is held constant [Hz].
        floor_value: Held value; ``None`` uses ``get_Sn`` at ``f_floor``.
        **sn_kwargs: Forwarded to ``get_Sn`` (e.g. ``stochastic_params``).
    """

    def __init__(
        self, sens_fn=A2TDISens, model=sangria, f_floor=1e-4, floor_value=None, **sn_kwargs
    ):
        self.sens_fn = sens_fn
        self.model = model
        self.sn_kwargs = sn_kwargs
        self.f_floor = float(f_floor)
        if floor_value is None:
            floor_value = self._get_Sn(np.array([self.f_floor]))[0]
        self.floor_value = float(floor_value)

    def _get_Sn(self, f):
        return self.sens_fn.get_Sn(f, self.model, **self.sn_kwargs)

    def __call__(self, f):
        xp = get_array_module(f)
        f_abs = xp.abs(f)
        S = self._get_Sn(xp.maximum(f_abs, self.f_floor))
        return xp.where(f_abs < self.f_floor, self.floor_value, S)

    @classmethod
    def from_file(cls, path, channel="A", clip_to_first=True, **kwargs):
        """PSD from a two-column ``(f, S)`` table (``.npy`` or text)."""
        model = TabulatedNoiseModel.from_file(
            path, channels=(channel,), clip_to_first=clip_to_first
        )
        return cls(sens_fn=_channel_sens_fn(channel), model=model, **kwargs)


class TabulatedNoiseModel:
    """LAT noise model with tabulated TDI PSDs (``Sensitivity.get_Sn``'s
    ``Sn_spl`` path), interpolated by cubic splines usable from NumPy and JAX.

    Args:
        freqs: Tabulated frequencies [Hz].
        psds: ``{channel: values}``.
    """

    def __init__(self, freqs, psds):
        self.Sn_spl = {ch: _CubicSplineXP(freqs, vals) for ch, vals in psds.items()}

    @classmethod
    def from_file(cls, path, channels=("A", "E"), clip_to_first=True):
        """One table used for every channel in ``channels``.

        ``clip_to_first`` raises values to at least the first tabulated one
        (removes the deep low-frequency dip of TDI2 tables).
        """
        table = np.load(path) if str(path).endswith(".npy") else np.loadtxt(path, skiprows=1)
        freqs, values = np.asarray(table, dtype=float).T
        if clip_to_first:
            values = np.clip(values, values[0], None)
        return cls(freqs, {ch: values for ch in channels})


class _CubicSplineXP:
    """scipy not-a-knot cubic spline, evaluated from its coefficients so the
    same object serves NumPy and (differentiable) JAX callers. Extrapolates
    with the end cubics, as scipy does."""

    def __init__(self, x, y):
        spline = CubicSpline(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        self.knots = spline.x
        self.coeffs = spline.c  # (4, n_intervals), highest power first

    def __call__(self, f):
        xp = get_array_module(f)
        knots, c = xp.asarray(self.knots), xp.asarray(self.coeffs)
        i = xp.clip(xp.searchsorted(knots, f, side="right") - 1, 0, knots.shape[0] - 2)
        dx = f - knots[i]
        return ((c[0, i] * dx + c[1, i]) * dx + c[2, i]) * dx + c[3, i]


def _channel_sens_fn(channel):
    from ... import sensitivity

    return getattr(sensitivity, f"{channel}2TDISens")
