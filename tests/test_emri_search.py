"""lisatools.search.emri: kernels, statistic, track models, optimisers.

Laptop-sized (CPU / JAX-CPU). The reference-parity cases need Lorenzo Speri's
``emrisearch`` package importable (``pip install --no-deps -e ../EMRI-Search``
plus ``interpax``) and skip otherwise.
"""

import copy
import os
import pickle
import types
import unittest

import numpy as np
from scipy.signal.windows import tukey

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    HAVE_JAX = True
except ImportError:  # pragma: no cover
    HAVE_JAX = False

import lisatools
from lisatools.search.emri import (
    Adam,
    AdamDE,
    DifferentialEvolution,
    EMRIInjection,
    EMRISearch,
    ExcessPowerStatistic,
    FDData,
    FresnelStatistic,
    MatchedFilterStatistic,
    SensitivityPSD,
    SFTData,
    SVDBasis,
    SVDTrackModel,
    TabulatedNoiseModel,
    Track,
    chirp_kernel,
    gaussian_noise,
    tukey_chirp_kernel,
)
from lisatools.search.emri.optimize import de_settings
from lisatools.utils.constants import YRSID_SI

T_SFT = 5e4
DT = 5.0
N_SEG = 24
EMRI_SEARCH_DIR = os.environ.get(
    "EMRI_SEARCH_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "EMRI-Search"),
)


def toy_psd():
    """LAT's analytic TDI2 A-channel PSD (sangria noise levels)."""
    return SensitivityPSD()


def import_emrisearch(module):
    """Import an ``emrisearch`` module or skip. Its ``emri_utils`` imports the
    retired ``fastlisaresponse`` at module level only for the response wrapper,
    which these tests replace, so that import is stubbed."""
    import importlib
    import sys

    sys.modules.setdefault("fastlisaresponse", types.SimpleNamespace(ResponseWrapper=None))
    try:
        return importlib.import_module(f"emrisearch.{module}")
    except Exception as exc:  # pragma: no cover
        raise unittest.SkipTest(f"emrisearch not importable: {exc}")


def chirp_track(n_seg=N_SEG, f0=3e-3, fdot=2e-11):
    t = np.arange(n_seg) * T_SFT
    return Track(f0 + fdot * t, np.full(n_seg, fdot))


def chirp_timeseries(track, amp, n_seg=N_SEG):
    """cos(2 pi int f dt) sampled at DT, matching ``track`` at the segment starts."""
    t = np.arange(int(n_seg * T_SFT / DT)) * DT
    f0, fdot = track.f[0], track.fdot[0]
    return amp * np.cos(2 * np.pi * (f0 * t + 0.5 * fdot * t**2))


def brute_kernel(f0, f1, T, window=None, n=200001):
    tau = np.linspace(0.0, T, n)
    w = np.ones_like(tau) if window is None else window(tau)
    return np.trapezoid(w * np.exp(2j * np.pi * (f0 * tau + 0.5 * f1 * tau**2)), tau)


def toy_basis(n_samples=N_SEG, n_modes=3, seed=0):
    """SVD basis of a bank of linear chirps (2-4 mHz, 1-4e-11 Hz/s).

    The bank has rank 2: modes beyond the second have a box of rounding width.
    """
    t = np.arange(n_samples) * T_SFT
    rng = np.random.default_rng(seed)
    f0s, fdots = rng.uniform(2e-3, 4e-3, 200), rng.uniform(1e-11, 4e-11, 200)
    return SVDBasis.build(f0s[:, None] + fdots[:, None] * t, n_modes=n_modes)


def constant_psd(S0=1e-40):
    f = np.logspace(-5, 0, 20)
    return SensitivityPSD(model=TabulatedNoiseModel(f, {"A": np.full(f.size, S0)}))


class KernelTest(unittest.TestCase):
    def test_against_quadrature(self):
        for f0, f1 in [(3e-5, 2e-10), (-1e-5, 5e-11), (2e-5, -3e-10), (1e-5, 0.0)]:
            ref = brute_kernel(f0, f1, T_SFT)
            got = chirp_kernel(np.array(f0), np.array(f1), T_SFT)
            self.assertLess(abs(got - ref) / abs(ref), 1e-6, (f0, f1))

    def test_tukey_against_quadrature(self):
        alpha = 0.2

        def window(tau):
            return tukey(tau.size, alpha)

        for f0, f1 in [(3e-5, 2e-10), (-1e-5, 5e-11), (1e-5, -1e-10)]:
            ref = brute_kernel(f0, f1, T_SFT, window=window)
            got = tukey_chirp_kernel(np.array(f0), np.array(f1), T_SFT, alpha)
            self.assertLess(abs(got - ref) / abs(ref), 1e-4, (f0, f1))

    def test_cpp_stft_fresnel_exact_option(self):
        backend = lisatools.get_backend("cpu")
        df = 1.0 / T_SFT
        offsets = np.arange(-100, 101)
        f0s, fdots = np.array([3.0003e-3, 5.2e-3, 7.7e-3]), np.array([2e-11, -5e-10, 1e-12])
        freqs = (np.round(f0s / df)[:, None] + offsets) * df
        ref = chirp_kernel(f0s[:, None] - freqs, fdots[:, None], T_SFT)
        errs = {}
        for exact in (True, False):
            fresnel = backend.STFTFresnelWrap(
                10, 2501, 1, 0.0, 0.0, 0.05, T_SFT, df, exact_fresnel=exact
            )
            out = np.zeros(freqs.size, dtype=complex)
            n = f0s.size
            fresnel.compute_fourier_values(
                out,
                np.ones(n),
                np.zeros(n),
                f0s,
                fdots,
                np.zeros(n),
                freqs.ravel(),
                1.0,
                n,
                offsets.size,
            )
            errs[exact] = np.max(np.abs(out.reshape(freqs.shape) - ref)) / np.abs(ref).max()
        self.assertLess(errs[True], 1e-10)
        self.assertGreater(errs[False], 1e-4)  # default stays the rational approximation

    @unittest.skipUnless(HAVE_JAX, "jax not installed")
    def test_jax_matches_numpy_and_grad_finite(self):
        from lisatools.search.emri import jax_chirp_kernel

        f0 = np.linspace(-3e-4, 3e-4, 41)
        for f1 in (2e-10, -2e-10, 0.0):
            ref = chirp_kernel(f0, np.full_like(f0, f1), T_SFT)
            got = np.asarray(jax_chirp_kernel(jnp.asarray(f0), jnp.full(f0.shape, f1), T_SFT))
            np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10 * np.abs(ref).max())
        g = jax.grad(lambda a, b: jnp.abs(jax_chirp_kernel(a, b, T_SFT)), argnums=(0, 1))
        for f1 in (2e-10, 0.0):
            self.assertTrue(all(np.isfinite(v) for v in g(1e-5, f1)))

    def test_small_drift_series_is_double_precision(self):
        def simpson_kernel(f0, f1):  # composite Simpson, ~1e-15 of T here
            n = 400_001
            tau = np.linspace(0.0, T_SFT, n)
            y = np.exp(2j * np.pi * (f0 * tau + 0.5 * f1 * tau**2))
            return (tau[1] / 3) * (y[0] + y[-1] + 4 * y[1:-1:2].sum() + 2 * y[2:-1:2].sum())

        df = 1.0 / T_SFT
        for f1 in (0.0, 1e-13, -1e-13, 1e-12, 3e-11, -6e-11):
            for offset in (0.0, 0.3, 2.7, 13.1, 100.4):
                ref = simpson_kernel(offset * df, f1)
                got = chirp_kernel(offset * df, f1, T_SFT, small_drift=True)
                self.assertLess(abs(got - ref) / T_SFT, 1e-14, (f1, offset))
        # Default path loses accuracy there (cancelling Fresnel values), the series does not.
        default = chirp_kernel(100.4 * df, 1e-13, T_SFT)
        self.assertGreater(abs(default - simpson_kernel(100.4 * df, 1e-13)) / T_SFT, 1e-12)

    @unittest.skipUnless(HAVE_JAX, "jax not installed")
    def test_jax_gradients_match_finite_differences(self):
        from lisatools.search.emri import jax_chirp_kernel

        c = 0.3 - 0.7j
        cases = [(3e-5, 2e-10), (-1e-5, 5e-11), (2e-5, -3e-10), (1e-6, 1e-12), (4e-4, 1e-9)]
        for small_drift in (False, True):

            def F(a, b):
                K = jax_chirp_kernel(a, b, T_SFT, small_drift=small_drift)
                return jnp.real(c * K) + jnp.abs(K) ** 2 / T_SFT

            grad = jax.grad(F, argnums=(0, 1))
            points = cases + ([(3e-5, 0.0)] if small_drift else [])
            for a, b in points:
                ga, gb = grad(a, b)
                ha, hb = 1e-12, (abs(b) * 1e-5 if b else 1e-14)
                fa = (F(a + ha, b) - F(a - ha, b)) / (2 * ha)
                fb = (F(a, b + hb) - F(a, b - hb)) / (2 * hb)
                self.assertLess(abs(ga - fa) / abs(fa), 1e-5, (small_drift, a, b))
                self.assertLess(abs(gb - fb) / abs(fb), 1e-5, (small_drift, a, b))
        # The JAX series agrees with the NumPy series.
        f0 = np.linspace(-3e-4, 3e-4, 41)
        for f1 in (0.0, 2e-12, -5e-11):
            ref = chirp_kernel(f0, np.full_like(f0, f1), T_SFT, small_drift=True)
            got = np.asarray(jax_chirp_kernel(f0, np.full_like(f0, f1), T_SFT, small_drift=True))
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-15 * T_SFT)


class PSDTest(unittest.TestCase):
    def test_tabulated_model_matches_scipy(self):
        from scipy.interpolate import CubicSpline

        from lisatools.sensitivity import A2TDISens

        f = np.logspace(-5, 0, 50)
        S = 1e-40 * (1 + (1e-3 / f) ** 2)
        psd = SensitivityPSD(model=TabulatedNoiseModel(f, {"A": S}), f_floor=1e-4)
        x = np.logspace(-4, -0.1, 300)
        np.testing.assert_allclose(psd(x), CubicSpline(f, S)(x), rtol=1e-12)
        np.testing.assert_allclose(psd(x), A2TDISens.get_Sn(x, psd.model), rtol=0)
        self.assertEqual(psd(np.array([5e-5]))[0], psd.floor_value)
        if HAVE_JAX:
            np.testing.assert_allclose(np.asarray(psd(jnp.asarray(x))), psd(x), rtol=1e-13)

    def test_analytic_sensitivity(self):
        from lisatools.sensitivity import A2TDISens

        psd = SensitivityPSD()
        x = np.logspace(-4, -1.5, 50)
        np.testing.assert_array_equal(psd(x), A2TDISens.get_Sn(x))
        self.assertTrue(np.all(psd(np.array([0.0, 1e-6])) == psd.floor_value))
        if HAVE_JAX:
            g = jax.grad(lambda f: jnp.sum(jnp.log(psd(f))))(jnp.asarray([0.0, 1e-3]))
            self.assertTrue(np.all(np.isfinite(np.asarray(g))))


class DataTest(unittest.TestCase):
    def test_sfts_match_reference_convention(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=int(N_SEG * T_SFT / DT) + 123)
        data = SFTData.from_timeseries(x, DT, T_SFT, toy_psd())
        nps = int(T_SFT / DT)
        ref = DT * np.fft.rfft(tukey(nps, 0.2) * x[: N_SEG * nps].reshape(-1, nps), axis=1)
        self.assertEqual(data.sfts.shape, (1, N_SEG, nps // 2 + 1))
        np.testing.assert_allclose(data.sfts[0], ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())

    def test_band_limited_sfts_give_same_statistic(self):
        np.random.seed(5)
        n = int(N_SEG * T_SFT / DT)
        x = gaussian_noise(n, DT, toy_psd())
        full = SFTData.from_timeseries(x, DT, T_SFT, toy_psd())
        band = SFTData.from_timeseries(x, DT, T_SFT, toy_psd(), min_freq=1e-3, max_freq=1e-2)
        self.assertGreater(band.first_bin, 0)
        tr = chirp_track()
        vals = []
        for d in (full, band):
            stat = FresnelStatistic.for_data(d, force_backend="cpu")
            vals.append(stat.evaluate(stat.to_backend(d.sfts), tr))
        np.testing.assert_allclose(vals[1], vals[0], rtol=1e-13)

    def test_noise_matches_reference(self):
        da_utils = import_emrisearch("da_utils")
        n, psd = 4096, toy_psd()
        np.random.seed(11)
        ref = da_utils.generate_noise(n, DT, psd)
        np.random.seed(11)
        got = gaussian_noise(n, DT, psd)[0]
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())

    def test_noise_rng_is_reproducible_and_leaves_global_state(self):
        psd = toy_psd()
        np.random.seed(5)
        before = np.random.get_state()[1].copy()
        a = gaussian_noise(1024, DT, psd, n_channels=2, rng=3)
        b = gaussian_noise(1024, DT, psd, n_channels=2, rng=np.random.default_rng(3))
        c = gaussian_noise(1024, DT, psd, n_channels=2, rng=4)
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))
        self.assertFalse(np.array_equal(a[0], a[1]))
        np.testing.assert_array_equal(np.random.get_state()[1], before)

    def test_first_n_segments(self):
        x = np.random.default_rng(2).normal(size=int(N_SEG * T_SFT / DT) + 123)
        full = SFTData.from_timeseries(x, DT, T_SFT, toy_psd())
        head = SFTData.from_timeseries(x, DT, T_SFT, toy_psd(), n_segments=10)
        self.assertEqual(head.n_segments, 10)
        np.testing.assert_array_equal(head.sfts, full.sfts[:, :10])
        np.testing.assert_array_equal(head.t_alpha, full.t_alpha[:10])
        for bad in (0, N_SEG + 1):
            with self.assertRaisesRegex(ValueError, "n_segments"):
                SFTData.from_timeseries(x, DT, T_SFT, toy_psd(), n_segments=bad)


class InjectionStreamTest(unittest.TestCase):
    """EMRIInjection's data streams on a synthetic chirp (no FEW needed)."""

    def setUp(self):
        signal = chirp_timeseries(chirp_track(), 1e-20)[None]
        self.inj = EMRIInjection(signal, DT, {}, 30.0, "A", toy_psd())

    def test_frozen_noise_is_shared_by_every_stream(self):
        inj = self.inj.with_noise(rng=7)
        self.assertIsNone(self.inj.noise)  # the original is unchanged
        np.testing.assert_array_equal(inj.noise, self.inj.draw_noise(rng=7))
        self.assertEqual(inj.n_segments(T_SFT), N_SEG)
        full = inj.sft_data(T_SFT)
        np.testing.assert_array_equal(inj.sft_data(T_SFT, n_segments=10).sfts, full.sfts[:, :10])
        np.testing.assert_array_equal(inj.sft_data(T_SFT).sfts, full.sfts)  # no new draw
        np.testing.assert_array_equal(
            full.sfts, SFTData.from_timeseries(inj.signal + inj.noise, DT, T_SFT, inj.psd).sfts
        )

    def test_signal_and_noise_parts_add_up(self):
        inj = self.inj.with_noise(rng=7)
        both, sig, noi = (
            inj.sft_data(T_SFT, signal=s, noise=n)
            for s, n in ((True, True), (True, False), (False, True))
        )
        np.testing.assert_allclose(
            both.sfts, sig.sfts + noi.sfts, rtol=0, atol=1e-12 * np.abs(both.sfts).max()
        )
        explicit = self.inj.sft_data(T_SFT, noise=inj.noise)
        np.testing.assert_array_equal(explicit.sfts, both.sfts)

    def test_fresh_draws_without_frozen_noise(self):
        a = self.inj.timeseries(rng=1)
        np.testing.assert_array_equal(a, self.inj.timeseries(rng=1))
        self.assertFalse(np.array_equal(a, self.inj.timeseries(rng=2)))

    def test_bad_requests_raise(self):
        with self.assertRaisesRegex(ValueError, "Nothing"):
            self.inj.timeseries(signal=False, noise=False)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.inj.with_noise(noise=np.zeros(5))
        with self.assertRaisesRegex(ValueError, "PSD"):
            EMRIInjection(self.inj.signal, DT, {}, 30.0).draw_noise()

    def test_pickles_with_noise(self):
        inj = self.inj.with_noise(rng=7)
        clone = pickle.loads(pickle.dumps(copy.deepcopy(inj)))
        np.testing.assert_array_equal(clone.sft_data(T_SFT).sfts, inj.sft_data(T_SFT).sfts)


class StatisticTest(unittest.TestCase):
    def setUp(self):
        self.psd = toy_psd()
        self.track = chirp_track()
        n = int(N_SEG * T_SFT / DT)
        np.random.seed(3)
        noise = gaussian_noise(n, DT, self.psd)
        self.noise = SFTData.from_timeseries(noise, DT, T_SFT, self.psd)
        amp = 30 * np.sqrt(self.psd(self.track.f[:1])[0] / (N_SEG * T_SFT / 2))
        signal = chirp_timeseries(self.track, amp)
        self.signal = SFTData.from_timeseries(signal, DT, T_SFT, self.psd)
        self.data = SFTData.from_timeseries(signal + noise, DT, T_SFT, self.psd)

    def test_signal_recovered_at_true_track(self):
        stat = FresnelStatistic.for_data(self.signal, force_backend="cpu")
        on = stat.evaluate(stat.to_backend(self.signal.sfts), self.track)
        off_track = Track(self.track.f + 3 / T_SFT, self.track.fdot)
        off = stat.evaluate(stat.to_backend(self.signal.sfts), off_track)
        # Rectangular kernel on Tukey(0.2) SFTs recovers ~90 % of SNR^2 / 2 = 450.
        self.assertGreater(on, 350)
        self.assertLess(off, 0.05 * on)

    @unittest.skipUnless(HAVE_JAX, "jax not installed")
    def test_jax_matches_numpy(self):
        for alpha in (0.0, 0.2):
            s_np = FresnelStatistic.for_data(
                self.data, kernel_window_alpha=alpha, force_backend="cpu"
            )
            s_jx = FresnelStatistic.for_data(
                self.data, kernel_window_alpha=alpha, force_backend="jax"
            )
            ref = s_np.evaluate(s_np.to_backend(self.data.sfts), self.track)
            tr = Track(jnp.asarray(self.track.f), jnp.asarray(self.track.fdot))
            got = s_jx.evaluate(s_jx.to_backend(self.data.sfts), tr)
            np.testing.assert_allclose(float(got), ref, rtol=1e-10)

    def test_small_drift_option_changes_statistic_negligibly(self):
        sfts = self.data.sfts
        default = FresnelStatistic.for_data(self.data, force_backend="cpu")
        series = FresnelStatistic.for_data(self.data, small_drift=True, force_backend="cpu")
        np.testing.assert_allclose(
            series.evaluate(sfts, self.track), default.evaluate(sfts, self.track), rtol=1e-8
        )

    def test_tabulated_psd_is_identical(self):
        sfts = self.data.sfts
        tabulated = FresnelStatistic.for_data(self.data, force_backend="cpu")
        self.assertIsNotNone(tabulated.psd_bins)
        direct = FresnelStatistic(
            self.data.T_sft, self.data.psd, first_bin=self.data.first_bin, force_backend="cpu"
        )
        self.assertIsNone(direct.psd_bins)
        self.assertEqual(tabulated.evaluate(sfts, self.track), direct.evaluate(sfts, self.track))

    def test_two_channels_sum(self):
        n = int(N_SEG * T_SFT / DT)
        np.random.seed(8)
        x = gaussian_noise(n, DT, self.psd, n_channels=2)
        x[0] += chirp_timeseries(self.track, 1e-20)
        both = SFTData.from_timeseries(x, DT, T_SFT, self.psd)
        stat = FresnelStatistic.for_data(both, force_backend="cpu")
        total = stat.evaluate(both.sfts, self.track)
        parts = [stat.evaluate(both.sfts[i : i + 1], self.track) for i in range(2)]
        np.testing.assert_allclose(total, sum(parts), rtol=1e-13)
        np.testing.assert_allclose(
            stat.per_segment(both.sfts, self.track),
            sum(stat.per_segment(both.sfts[i : i + 1], self.track) for i in range(2)),
            rtol=1e-13,
        )


class NoiseDistributionTest(unittest.TestCase):
    """Noise-only statistic vs its exact expectation (white noise, Tukey SFTs).

    For SFTs ``d_k = dt sum_j w_j n_j e^{-2 pi i k j / N}`` of white noise of
    variance ``sigma^2 = S / (2 dt)``, the per-segment overlap
    ``c = df sum_k conj(d_k) a_k`` (``a_k = K_k / S``) has
    ``E|c|^2 = df^2 dt^2 sigma^2 sum_j w_j^2 |sum_k a_k e^{2 pi i k j / N}|^2``,
    and each term ``2 |c|^2 / <h|h>`` is exponential with that mean.
    """

    def test_mean_and_variance(self):
        S0, n_draws = 1e-40, 100
        psd = constant_psd(S0)
        track = chirp_track()
        n = int(N_SEG * T_SFT / DT)
        rng = np.random.default_rng(21)
        draws, stat = [], None
        for _ in range(n_draws):
            data = SFTData.from_timeseries(
                gaussian_noise(n, DT, psd, n_channels=2, rng=rng), DT, T_SFT, psd
            )
            stat = stat or FresnelStatistic.for_data(data, force_backend="cpu")
            draws.append(stat.evaluate(data.sfts, track))
        draws = np.array(draws)

        # Exact per-segment expectation.
        N, df = int(T_SFT / DT), 1.0 / T_SFT
        w = tukey(N, 0.2)
        j = np.arange(N)
        expected = []
        for f, fdot in zip(track.f, track.fdot):
            k = int(f * T_SFT) + np.arange(-stat.P, stat.P + 1)
            a = chirp_kernel(f - k * df, np.full(k.size, fdot), T_SFT) / S0
            A = np.exp(2j * np.pi * np.outer(j, k) / N) @ a
            Ec2 = df**2 * DT**2 * (S0 / (2 * DT)) * np.sum(w**2 * np.abs(A) ** 2)
            expected.append(2 * Ec2 / (0.5 * T_SFT / S0))
        m = 2 * np.array(expected)  # two channels
        mean, var = m.sum(), np.sum(m**2)  # sum of independent exponentials
        self.assertLess(abs(draws.mean() - mean), 5 * np.sqrt(var / n_draws))
        self.assertLess(abs(draws.var(ddof=1) / var - 1), 0.4)
        # Per term (segment x channel): 2 int w^2 / T = 2 (1 - 5 alpha / 8). The 2 is
        # the paper's Eq. (6) normalisation: <h|h> = T / (2 S) is half a unit real
        # chirp's T / S, so an unwindowed term has mean 2, not 1, in noise.
        np.testing.assert_allclose(m.mean() / 2, 2 * (1 - 5 * 0.2 / 8), rtol=1e-2)


@unittest.skipUnless(HAVE_JAX, "jax not installed")
class ReferenceParityTest(unittest.TestCase):
    """Agreement (1e-12) with ``emrisearch.jax_utils.det_stat``."""

    @classmethod
    def setUpClass(cls):
        cls.ref = import_emrisearch("jax_utils")
        data_dir = os.path.join(EMRI_SEARCH_DIR, "src", "emrisearch", "data")
        ff, Sn = np.loadtxt(os.path.join(data_dir, "LPA.txt"), skiprows=1).T
        from scipy.interpolate import CubicSpline

        # jax_utils.psd holds f < 1e-4 at the *LPA* spline's value there.
        cls.psd = SensitivityPSD.from_file(
            os.path.join(data_dir, "TDI2_AE_psd.npy"),
            floor_value=float(CubicSpline(ff, Sn**2)(1e-4)),
        )

    def test_det_stat(self):
        rng = np.random.default_rng(7)
        n_bins = int(T_SFT / DT) // 2 + 1
        sfts = rng.normal(size=(N_SEG, n_bins)) + 1j * rng.normal(size=(N_SEG, n_bins))
        sfts *= np.sqrt(self.psd(np.arange(n_bins) / T_SFT))[None, :] * T_SFT
        stat = FresnelStatistic(T_SFT, self.psd, force_backend="jax")
        for f0 in (3e-4, 1.5e-3, 6e-3):  # low f0 exercises bins below 1e-4 Hz
            tr = chirp_track(f0=f0)
            ref = self.ref.det_stat(
                jnp.asarray(sfts.T), jnp.asarray(tr.f), jnp.asarray(tr.fdot), T_sft=T_SFT
            )
            got = stat.evaluate(
                stat.to_backend(sfts[None]), Track(jnp.asarray(tr.f), jnp.asarray(tr.fdot))
            )
            np.testing.assert_allclose(float(got), float(ref), rtol=1e-12, err_msg=f"f0={f0}")


class FEWTrackParityTest(unittest.TestCase):
    """FEWHarmonicTrackModel vs ``emrisearch.emri_utils.get_f_fdot_fddot_back``."""

    def test_tracks_match(self):
        get_f_fdot_fddot_back = import_emrisearch("emri_utils").get_f_fdot_fddot_back

        from lisatools.search.emri import FEWHarmonicTrackModel
        from lisatools.utils.constants import YRSID_SI

        t_alpha = np.arange(0, 0.5 * YRSID_SI, T_SFT)
        params = (1e6, 10.0, 0.9, 0.4, 0.1)  # plunges inside the window
        phi, f, fdot, fddot = get_f_fdot_fddot_back(np.array([*params, 1.0]), t_alpha)
        model = FEWHarmonicTrackModel(t_alpha, m=2, n=1)
        out = model.phases_and_derivatives(*params)
        for ref, got in zip((phi, f, fdot, fddot), out):
            np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)
        track = model(params)
        np.testing.assert_allclose(track.f, 2 * f[0] + f[1], rtol=1e-12)
        self.assertEqual(int(np.sum(track.f == 0.0)), int(np.sum(t_alpha >= 0.4 * YRSID_SI)))


class InitialPopulationTest(unittest.TestCase):
    def setUp(self):
        path = os.path.join(EMRI_SEARCH_DIR, "examples", "new_svd_basis_Tsft50000.0_cut20.npz")
        if not os.path.exists(path):
            self.skipTest("EMRI-Search basis file not found")
        basis = SVDBasis.from_npz(path)
        self.model = SVDTrackModel(
            basis, np.arange(basis.n_samples) * T_SFT, T_ref=YRSID_SI, force_backend="cpu"
        )

    def test_keep_matches_reference_and_leaves_cube(self):
        u = self.model.initial_population(512, seed=0)
        np.testing.assert_array_equal(
            u, self.model.initial_population(512, seed=0, out_of_cube="keep")
        )
        self.assertTrue(np.any((u < 0) | (u > 1)))

    def test_clip_screens_the_optimised_tracks(self):
        u = self.model.initial_population(512, seed=0, out_of_cube="clip")
        self.assertGreater(len(u), 0)
        self.assertTrue(np.all((u >= 0) & (u <= 1)))
        # Clipping is idempotent, so the optimisers' first-step clip leaves these tracks alone.
        V = self.model.basis.V
        f = self.model.coeffs(u) @ V
        np.testing.assert_array_equal(f, self.model.coeffs(np.clip(u, 0, 1)) @ V)
        self.assertTrue(np.all(f >= 0))

    def test_frequency_range(self):
        basis = self.model.basis
        lo, hi = basis.frequency_range()
        self.assertAlmostEqual(lo, -0.69e-3, delta=0.01e-3)  # linprog over the box
        self.assertAlmostEqual(hi, 4.24e-3, delta=0.01e-3)
        r = np.random.default_rng(0).uniform(size=(2000, basis.n_modes))
        f0 = (basis.lb + r * (basis.ub - basis.lb)) @ basis.V[:, 0]
        self.assertTrue(np.all((f0 >= lo) & (f0 <= hi)))
        corner = np.where(basis.V[:, 0] > 0, basis.ub, basis.lb)
        self.assertAlmostEqual(float(corner @ basis.V[:, 0]), hi, places=15)

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            self.model.initial_population(8, seed=0, out_of_cube="reject")


class ShiftTest(unittest.TestCase):
    def setUp(self):
        self.basis = toy_basis(n_samples=N_SEG + 6)
        self.model = SVDTrackModel(
            self.basis, np.arange(N_SEG) * T_SFT, T_ref=(N_SEG + 6) * T_SFT, force_backend="cpu"
        )
        self.x = np.full(self.basis.n_modes, 0.4)
        self.full = self.model.coeffs(self.x) @ self.basis.V  # all N_SEG + 6 samples

    def test_shift_for_plunge_truncates_toward_zero(self):
        T_ref = self.model.T_ref
        for dT, shift in ((2.7, 2), (-2.7, -2), (0.4, 0), (-0.4, 0), (5.0, 5)):
            self.assertEqual(self.model.shift_for_plunge(T_ref + dT * T_SFT), shift, dT)

    def test_roll_and_zero(self):
        f0 = self.model(self.x, 0).f
        np.testing.assert_array_equal(f0, self.full[:N_SEG])
        f = self.model(self.x, 3).f  # plunge later: track starts 3 segments in
        np.testing.assert_array_equal(f[:3], 0.0)
        np.testing.assert_array_equal(f[3:], self.full[: N_SEG - 3])
        f = self.model(self.x, -3).f  # plunge earlier: track advanced 3 segments
        np.testing.assert_array_equal(f, self.full[3 : N_SEG + 3])
        f = self.model(self.x, -(N_SEG + 3)).f  # vacated end reaches into the data
        np.testing.assert_array_equal(f[:3], self.full[N_SEG + 3 :])
        np.testing.assert_array_equal(f[3:], 0.0)
        np.testing.assert_allclose(
            self.model(self.x, 3).fdot[4:-1], np.gradient(f0, T_SFT)[1 : N_SEG - 4], rtol=1e-12
        )


@unittest.skipUnless(HAVE_JAX, "jax not installed")
class OptimizerTest(unittest.TestCase):
    @staticmethod
    def loss(x):
        # Multimodal toy loss: DE accepts some trials, rejects others.
        return jnp.sum((x - 0.3) ** 2) + 0.05 * jnp.sum(jnp.cos(40 * x))

    def x0(self):
        return np.random.default_rng(0).uniform(size=(16, 3))

    def test_reported_loss_is_loss_at_x(self):
        for stale in (True, False):
            result = AdamDE(n_steps=20, adam=Adam(learning_rate=1e-2), stale_selection=stale).run(
                self.loss, self.x0(), jax.random.PRNGKey(0)
            )
            refreshed = jax.vmap(self.loss)(result.x)
            np.testing.assert_allclose(result.loss, refreshed, rtol=0, atol=1e-14)
            best_x, best_loss = result.best
            self.assertAlmostEqual(float(best_loss), float(self.loss(best_x)), places=14)

    def test_de_settings_follow_the_key(self):
        draws = [de_settings(jax.random.PRNGKey(s), (0.3, 1.5), 0.5) for s in range(64)]
        Fs = np.array([float(F) for _, F in draws])
        self.assertEqual(len(np.unique(Fs)), Fs.size)
        self.assertTrue(np.all((Fs >= 0.3) & (Fs <= 1.5)))
        self.assertTrue(0 < sum(bool(e) for e, _ in draws) < len(draws))
        again = de_settings(jax.random.PRNGKey(3), (0.3, 1.5), 0.5)
        self.assertEqual(float(again[1]), Fs[3])

        runs = [
            AdamDE(n_steps=10).run(self.loss, self.x0(), jax.random.PRNGKey(k)) for k in (0, 0, 1)
        ]
        np.testing.assert_array_equal(runs[0].x, runs[1].x)
        self.assertFalse(np.array_equal(runs[0].x, runs[2].x))

    def test_batched_population_matches_vmap(self):
        x0 = np.random.default_rng(1).uniform(size=(13, 3))  # 13 = 3 x 4 + 1: a remainder
        for make in (
            lambda b: AdamDE(n_steps=15, adam=Adam(learning_rate=1e-2), batch_size=b),
            lambda b: DifferentialEvolution(n_steps=15, batch_size=b),
        ):
            full, batched = (make(b).run(self.loss, x0, jax.random.PRNGKey(4)) for b in (None, 4))
            np.testing.assert_allclose(batched.x, full.x, rtol=0, atol=1e-14)
            np.testing.assert_allclose(batched.loss, full.loss, rtol=0, atol=1e-14)

    def test_differential_evolution(self):
        x0 = self.x0()
        start = float(jax.vmap(self.loss)(jnp.clip(jnp.asarray(x0), 0, 1)).min())
        result = DifferentialEvolution(n_steps=60, save_every=1).run(
            self.loss, x0, jax.random.PRNGKey(2)
        )
        best = np.array([losses[0] for losses in result.history["loss"]])
        self.assertTrue(np.all(np.diff(best) <= 0))  # greedy: the best never gets worse
        self.assertLess(best[-1], start)
        np.testing.assert_allclose(result.loss, jax.vmap(self.loss)(result.x), rtol=0, atol=1e-14)
        self.assertTrue(np.all((np.asarray(result.x) >= 0) & (np.asarray(result.x) <= 1)))


@unittest.skipUnless(HAVE_JAX, "jax not installed")
class OptimizerParityTest(unittest.TestCase):
    """AdamDE vs EMRI-Search's QuickStart Adam + DE loop, transcribed.

    The loop below is the notebook's (optax Adam, clip, DE in frequency-track
    space with ``differential_evolution_step`` and ``selection``, back to the
    cube). The only change is where the randomness comes from: each step's key
    and ``(elitism, F)`` follow AdamDE's schedule instead of NumPy's global RNG.
    """

    def test_trajectory_matches_reference(self):
        try:
            import optax
        except ImportError:
            self.skipTest("optax not installed")
        de_ref = import_emrisearch("jax_de_utils")

        # Linear chirps span 2 modes; a 3rd would have a ~1e-17-wide box, and the
        # notebook's round trip through frequency space divides rounding by it.
        basis = toy_basis(n_modes=2)
        V, lower, upper = jnp.asarray(basis.V), basis.lb, basis.ub
        f_target = (lower + 0.37 * (upper - lower)) @ basis.V

        def loss(xi):
            f = (lower + xi * (upper - lower)) @ V
            r = (f - f_target) / 1e-4
            return jnp.sum(r**2) / r.size + 0.3 * jnp.mean(jnp.cos(3 * r))

        lr, n_steps = 1e-2, 15
        x0 = jnp.asarray(np.random.default_rng(4).uniform(size=(12, basis.n_modes)))
        key0 = jax.random.PRNGKey(9)
        ours = AdamDE(n_steps=n_steps, adam=Adam(learning_rate=lr), save_every=1).run(
            loss, x0, key0
        )

        # --- the notebook loop ---
        grad_fn = jax.jit(jax.value_and_grad(loss))
        optimizer = optax.adam(learning_rate=lr)
        batch_xi = x0
        state = jax.vmap(optimizer.init)(batch_xi)

        def update_step(xi, opt_state):
            l, g = grad_fn(xi)
            updates, opt_state = optimizer.update(g, opt_state)
            return jnp.clip(optax.apply_updates(xi, updates), 0.0, 1.0), opt_state, l

        update = jax.jit(jax.vmap(update_step))
        batch_loss_fn = jax.jit(jax.vmap(loss))
        key, improved_ref = key0, []
        for step in range(n_steps + 1):
            key, key_mut, key_de = jax.random.split(key, 3)  # AdamDE's schedule
            elitism, F = de_settings(key_de, (0.3, 1.5), 0.5)
            batch_xi, state, batch_loss = update(batch_xi, state)
            batch_f = (lower + batch_xi * (upper - lower)) @ V
            trial_f = de_ref.differential_evolution_step(
                key_mut,
                batch_f,
                fitness=batch_loss,
                elitism=elitism,
                crossover_rate=1.0,
                differential_weight=F,
            )
            trial_xi = jnp.clip(((trial_f @ V.T) - lower) / (upper - lower), 0.0, 1.0)
            batch_xi, batch_loss, improved = de_ref.selection(
                batch_loss, batch_xi, batch_loss_fn(trial_xi), trial_xi
            )
            improved_ref.append(float(np.mean(np.asarray(improved))))

        np.testing.assert_allclose(ours.x, batch_xi, rtol=0, atol=1e-10)
        np.testing.assert_allclose(ours.history["improved_fraction"], improved_ref, rtol=0)
        self.assertGreater(max(improved_ref), 0)  # DE did accept trials
        # AdamDE re-scores at the end; the notebook keeps the stale scores.
        np.testing.assert_allclose(ours.loss, batch_loss_fn(batch_xi), rtol=0, atol=1e-12)


@unittest.skipUnless(HAVE_JAX, "jax not installed")
class SearchTest(unittest.TestCase):
    def setUp(self):
        self.psd = toy_psd()
        t = np.arange(N_SEG) * T_SFT
        rng = np.random.default_rng(0)
        # Toy bank: linear chirps, plunge-aligned basis of 3 modes.
        bank = np.array(
            [
                f0 + fd * t
                for f0, fd in zip(rng.uniform(2e-3, 4e-3, 200), rng.uniform(1e-11, 4e-11, 200))
            ]
        )
        self.basis = SVDBasis.build(bank, n_modes=3)
        self.track = chirp_track(f0=3e-3, fdot=2.5e-11)
        n = int(N_SEG * T_SFT / DT)
        amp = 40 * np.sqrt(self.psd(np.array([3e-3]))[0] / (N_SEG * T_SFT / 2))
        np.random.seed(1)
        x = chirp_timeseries(self.track, amp) + gaussian_noise(n, DT, self.psd)[0]
        self.data = SFTData.from_timeseries(x, DT, T_SFT, self.psd)

    def make_search(self, n_steps):
        stat = FresnelStatistic.for_data(self.data, force_backend="jax")
        model = SVDTrackModel(
            self.basis, self.data.t_alpha, T_ref=N_SEG * T_SFT, force_backend="jax"
        )
        opt = AdamDE(n_steps=n_steps, save_every=5)
        return EMRISearch(
            self.data,
            stat,
            model,
            opt,
            n_initial=64,
            n_population=16,
            initial_kwargs={"f0_range": (2e-3, 4e-3)},
        )

    def test_band_check(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.make_search(n_steps=1)  # full band: no warning
        stat = FresnelStatistic.for_data(self.data, force_backend="jax")
        model = SVDTrackModel(
            self.basis, self.data.t_alpha, T_ref=N_SEG * T_SFT, force_backend="jax"
        )
        lo, hi = model.frequency_range()
        lo = max(lo, stat.mask.f_min)  # segments below f_min never count
        x = np.zeros(int(N_SEG * T_SFT / DT))
        narrow = SFTData.from_timeseries(x, DT, T_SFT, self.psd, min_freq=lo, max_freq=hi)
        with self.assertWarns(UserWarning):
            EMRISearch(
                narrow,
                FresnelStatistic.for_data(narrow, force_backend="jax"),
                model,
                AdamDE(n_steps=1),
            )
        P_df = stat.P / T_SFT
        wide = SFTData.from_timeseries(
            x,
            DT,
            T_SFT,
            self.psd,
            min_freq=max(0.0, lo - P_df - 1 / T_SFT),
            max_freq=hi + P_df + 1 / T_SFT,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            EMRISearch(
                wide, FresnelStatistic.for_data(wide, force_backend="jax"), model, AdamDE(n_steps=1)
            )

    def test_rejects_cpu_backend(self):
        stat = FresnelStatistic.for_data(self.data, force_backend="cpu")
        model = SVDTrackModel(
            self.basis, self.data.t_alpha, T_ref=N_SEG * T_SFT, force_backend="cpu"
        )
        with self.assertRaisesRegex(ValueError, "force_backend"):
            EMRISearch(self.data, stat, model, AdamDE(n_steps=1))

    def test_recovers_injected_track(self):
        search = self.make_search(n_steps=40)
        res = search.run(T_pl=N_SEG * T_SFT, seed=0)
        stat = search.statistic
        true_val = float(
            stat.evaluate(
                stat.to_backend(self.data.sfts),
                Track(jnp.asarray(self.track.f), jnp.asarray(self.track.fdot)),
            )
        )
        self.assertGreater(res.best_statistic, 0.9 * true_val)
        rel = np.abs(np.asarray(res.best_track.f) - self.track.f) / self.track.f
        self.assertLess(np.median(rel), 1e-3)

    def test_recovers_track_at_nonzero_shift(self):
        n_late = 3  # the injection starts n_late segments in
        t = np.arange(int(N_SEG * T_SFT / DT)) * DT
        tau = t - n_late * T_SFT
        f0, fdot = 3e-3, 2.5e-11
        amp = 40 * np.sqrt(self.psd(np.array([f0]))[0] / ((N_SEG - n_late) * T_SFT / 2))
        signal = np.where(tau >= 0, amp * np.cos(2 * np.pi * (f0 * tau + 0.5 * fdot * tau**2)), 0)
        np.random.seed(2)
        data = SFTData.from_timeseries(
            signal + gaussian_noise(t.size, DT, self.psd)[0], DT, T_SFT, self.psd
        )
        stat = FresnelStatistic.for_data(data, force_backend="jax")
        model = SVDTrackModel(self.basis, data.t_alpha, T_ref=N_SEG * T_SFT, force_backend="jax")
        search = EMRISearch(
            data,
            stat,
            model,
            AdamDE(n_steps=40),
            n_initial=64,
            n_population=16,
            initial_kwargs={"f0_range": (2e-3, 4e-3)},
        )
        T_pl = (N_SEG + n_late) * T_SFT
        self.assertEqual(model.shift_for_plunge(T_pl), n_late)
        res = search.run(T_pl=T_pl, seed=0)
        seg_t = np.arange(N_SEG) * T_SFT - n_late * T_SFT
        true_f = np.where(seg_t >= 0, f0 + fdot * seg_t, 0.0)
        true_track = Track(jnp.asarray(true_f), jnp.gradient(jnp.asarray(true_f), T_SFT))
        true_val = float(stat.evaluate(stat.to_backend(data.sfts), true_track))
        self.assertGreater(res.best_statistic, 0.9 * true_val)
        got = np.asarray(res.best_track.f)
        np.testing.assert_array_equal(got[:n_late], 0.0)
        rel = np.abs(got[n_late:] - true_f[n_late:]) / true_f[n_late:]
        self.assertLess(np.median(rel), 1e-3)

    def test_batched_search_matches(self):
        full = self.make_search(n_steps=5)
        batched = self.make_search(n_steps=5)
        batched.optimizer.batch_size = 5  # 64 screened, 16 optimised: remainders both ways
        a, b = (s.run(T_pl=N_SEG * T_SFT, seed=0) for s in (full, batched))
        np.testing.assert_allclose(b.best_x, a.best_x, rtol=0, atol=1e-12)
        self.assertAlmostEqual(b.best_statistic, a.best_statistic, places=8)

    def test_config_pickles(self):
        search = self.make_search(n_steps=1)
        clone = pickle.loads(pickle.dumps(copy.deepcopy(search)))
        self.assertEqual(clone.statistic.P, search.statistic.P)
        np.testing.assert_array_equal(clone.track_model.basis.V, self.basis.V)


class ExcessPowerTest(unittest.TestCase):
    """FDData and the delay-scanning statistics (matched filter, Cutler &
    Bustamante-Rosell's H) on linear chirps in the analytic PSD."""

    BAND = dict(min_freq=1e-3, max_freq=1e-2)

    def setUp(self):
        self.psd = toy_psd()
        self.n = int(N_SEG * T_SFT / DT)
        self.t_alpha = np.arange(N_SEG) * T_SFT

    def chirp(self, fdot, snr=30.0):
        """A unit-SNR-scaled linear chirp at 3 mHz and its track."""
        from lisatools.search.emri import band_snr

        t = np.arange(self.n) * DT
        h = np.cos(2 * np.pi * (3e-3 * t + 0.5 * fdot * t**2))
        h *= snr / band_snr(h, DT, self.psd, (1e-3, 1e-2))
        return h, Track(3e-3 + fdot * self.t_alpha, np.full(N_SEG, fdot))

    def fd(self, x, **kwargs):
        return FDData.from_timeseries(x, DT, self.psd, **self.BAND, **kwargs)

    def test_fd_data_matches_rfft(self):
        x = np.random.default_rng(3).normal(size=self.n + 17)
        d = self.fd(x, duration=10 * T_SFT)
        n = int(10 * T_SFT / DT)
        ref = DT * np.fft.rfft(x[:n])
        self.assertAlmostEqual(d.T, 10 * T_SFT)
        np.testing.assert_allclose(d.fd[0], ref[d.first_bin : d.first_bin + d.n_bins], rtol=1e-12)
        np.testing.assert_allclose(d.freqs, (d.first_bin + np.arange(d.n_bins)) / d.T)
        with self.assertRaisesRegex(ValueError, "duration"):
            self.fd(x, duration=2 * self.n * DT)

    def test_injection_fd_stream(self):
        signal = chirp_timeseries(chirp_track(), 1e-20)[None]
        inj = EMRIInjection(signal, DT, {}, 30.0, "A", self.psd).with_noise(rng=4)
        d = inj.fd_data(duration=12 * T_SFT, **self.BAND)
        ref = self.fd(inj.signal + inj.noise, duration=12 * T_SFT)
        np.testing.assert_array_equal(d.fd, ref.fd)

    def test_matched_filter_recovers_snr(self):
        h, track = self.chirp(2e-10)
        d = self.fd(h)
        mf = MatchedFilterStatistic.for_data(d, self.t_alpha, force_backend="cpu")
        rho = float(mf.evaluate(mf.prepare(d), track))
        self.assertGreater(rho, 0.98 * 30)  # SPA edge loss ~1% at 288 bins
        self.assertLessEqual(rho, 30 * (1 + 1e-6))

    def test_noise_calibration(self):
        """Mean |rho|^2 = 2 and unit-variance window sums with the exact Q(N)
        (288-bin template); the asymptotic Q over-normalises a 29-bin one."""
        _, wide = self.chirp(2e-10)
        _, narrow = self.chirp(2e-11)
        d0 = self.fd(np.zeros(self.n))
        exact = ExcessPowerStatistic.for_data(d0, self.t_alpha, force_backend="cpu")
        asym = ExcessPowerStatistic.for_data(
            d0, self.t_alpha, normalization="asymptotic", force_backend="cpu"
        )
        q_exact = exact.window_q(exact.template(wide)[0])
        q_asym = asym.window_q(asym.template(narrow)[0])
        rng = np.random.default_rng(0)
        means, wide_sums, narrow_sums = [], [], []
        for _ in range(200):
            arr = exact.prepare(self.fd(gaussian_noise(self.n, DT, self.psd, rng=rng)))
            power, _ = exact.correlation_power(arr, wide)
            means.append(power.mean())
            wide_sums.append([(power[:n] - 2).sum() / np.sqrt(4 * n) for n in exact.cells])
            power, _ = exact.correlation_power(arr, narrow)
            narrow_sums.append((power[: exact.cells[0]] - 2).sum() / np.sqrt(4 * exact.cells[0]))
        wide_sums = np.array(wide_sums)
        # Each draw's delay average has std ~2 / sqrt(288 bins): 3 sigma over 200 draws.
        self.assertAlmostEqual(np.mean(means), 2.0, delta=0.03)
        for k in range(len(exact.cells)):
            std = np.std(wide_sums[:, k] / np.sqrt(q_exact[k]))
            self.assertAlmostEqual(std, 1.0, delta=0.15, msg=exact.cells[k])
        self.assertLess(np.std(np.array(narrow_sums) / np.sqrt(q_asym[0])), 0.5)

    def test_h_prefers_the_signal_track(self):
        h, track = self.chirp(2e-11)
        d = self.fd(h + gaussian_noise(self.n, DT, self.psd, rng=1)[0])
        H = ExcessPowerStatistic.for_data(d, self.t_alpha, force_backend="cpu")
        arr = H.prepare(d)
        on = float(H.evaluate(arr, track))
        off = float(H.evaluate(arr, Track(track.f + 2e-3, track.fdot)))
        crossing = float(H.evaluate(arr, Track(track.f[::-1], track.fdot)))  # crosses once
        print(f"[H] on track {on:.1f}, 2 mHz off {off:.1f}, crossing {crossing:.1f}")
        self.assertGreater(on, 50.0)
        self.assertLess(off, 6.0)
        self.assertGreater(crossing, 4 * off)
        self.assertEqual(float(H.evaluate(arr, Track(np.zeros(N_SEG), np.zeros(N_SEG)))), 0.0)

    def test_jax_matches_numpy(self):
        h, track = self.chirp(2e-11)
        d = self.fd(h + gaussian_noise(self.n, DT, self.psd, rng=2)[0])
        tj = Track(jnp.asarray(track.f), jnp.asarray(track.fdot))
        for cls in (MatchedFilterStatistic, ExcessPowerStatistic):
            s_np = cls.for_data(d, self.t_alpha, force_backend="cpu")
            s_jx = cls.for_data(d, self.t_alpha, force_backend="jax")
            ref = float(s_np.evaluate(s_np.prepare(d), track))
            got = float(s_jx.evaluate(s_jx.prepare(d), tj))
            self.assertAlmostEqual(got / ref, 1.0, places=12, msg=cls.__name__)
            grad = jax.grad(lambda f: s_jx.evaluate(s_jx.prepare(d), Track(f, tj.fdot)))(tj.f)
            self.assertTrue(bool(jnp.all(jnp.isfinite(grad))), cls.__name__)

    def test_search_with_h(self):
        h, track = self.chirp(2.5e-11, snr=40.0)
        d = self.fd(h + gaussian_noise(self.n, DT, self.psd, rng=5)[0])
        stat = ExcessPowerStatistic.for_data(d, self.t_alpha, force_backend="jax")
        model = SVDTrackModel(toy_basis(), self.t_alpha, T_ref=N_SEG * T_SFT, force_backend="jax")
        search = EMRISearch(
            d,
            stat,
            model,
            AdamDE(n_steps=20, save_every=5),
            n_initial=64,
            n_population=16,
            initial_kwargs={"f0_range": (2e-3, 4e-3)},
        )
        res = search.run(T_pl=N_SEG * T_SFT, seed=0)
        noise = search.run(
            T_pl=N_SEG * T_SFT, seed=0, data=self.fd(gaussian_noise(self.n, DT, self.psd, rng=6))
        )
        print(f"[H search] signal {res.best_statistic:.1f}, noise only {noise.best_statistic:.1f}")
        self.assertGreater(res.best_statistic, 3 * noise.best_statistic)

    def test_bad_options_raise(self):
        d = self.fd(np.zeros(self.n))
        with self.assertRaisesRegex(ValueError, "amplitude"):
            MatchedFilterStatistic.for_data(d, self.t_alpha, amplitude="x", force_backend="cpu")
        with self.assertRaisesRegex(ValueError, "normalization"):
            ExcessPowerStatistic.for_data(d, self.t_alpha, normalization="x", force_backend="cpu")

    def test_config_pickles(self):
        d = self.fd(np.zeros(self.n))
        stat = ExcessPowerStatistic.for_data(d, self.t_alpha, force_backend="cpu")
        clone = pickle.loads(pickle.dumps(copy.deepcopy(stat)))
        self.assertEqual(clone.cells, stat.cells)
        self.assertEqual(clone.n_fft, stat.n_fft)


class InjectionTest(unittest.TestCase):
    """FEWInjector: EMRI-Search parity, SNR scaling, signal on the FEW track.

    Builds one FEW + response generator (~2.3 GiB, ~10 s) shared by the class;
    the parity test imports ``emrisearch``, which builds one more.
    """

    T, T_DATA = 0.05, 0.04
    PARAMS = (1e6, 10.0, 0.9, 0.045, 0.1)

    @classmethod
    def setUpClass(cls):
        try:
            import few  # noqa: F401
        except ImportError:  # pragma: no cover
            raise unittest.SkipTest("few not installed")
        from lisatools.search.emri import FEWInjector

        data_dir = os.path.join(EMRI_SEARCH_DIR, "src", "emrisearch", "data")
        path = os.path.join(data_dir, "TDI2_AE_psd.npy")
        cls.psd = SensitivityPSD.from_file(path) if os.path.exists(path) else None
        cls.injector = FEWInjector(cls.T, dt=DT)
        cls.angles = dict(qS=1.0, phiS=0.5, qK=2.0, phiK=0.2, Phi_phi0=0.1, Phi_r0=0.3)
        if cls.psd is not None:
            cls.inj = cls.injector(cls.PARAMS, 30.0, cls.psd, angles=cls.angles)

    def setUp(self):
        if self.psd is None:
            self.skipTest("EMRI-Search PSD file not found")

    def test_snr_and_distance(self):
        from lisatools.search.emri import band_snr

        inj = self.inj
        self.assertEqual(inj.signal.shape, (1, self.injector.n_samples))
        self.assertAlmostEqual(band_snr(inj.signal, DT, self.psd), 30.0, places=10)
        # The waveform is linear in 1 / dist: regenerate at the rescaled distance.
        p = inj.params
        h = self.injector.waveform(
            p["m1"], p["m2"], p["a"], p["p0"], p["e0"], p["x0"], p["dist"], p
        )
        np.testing.assert_allclose(h, inj.signal, rtol=1e-10, atol=1e-10 * np.abs(h).max())

    def test_statistic_on_track(self):
        from lisatools.search.emri import FEWHarmonicTrackModel

        data = self.inj.sft_data(T_SFT, noise=False)
        stat = FresnelStatistic.for_data(data, force_backend="cpu")
        model = FEWHarmonicTrackModel(data.t_alpha, m=2, n=0)
        track = model(self.PARAMS)
        np.testing.assert_array_equal(model((*self.PARAMS, 1.0)).f, track.f)  # x_f = +1
        on = stat.evaluate(data.sfts, track)
        off = stat.evaluate(data.sfts, Track(track.f + 4 / T_SFT, track.fdot, track.fddot))
        noise_mean = 1.75 * float(stat.mask(track, T_SFT, np).sum())
        print(f"[injection] m=2 track {on:.1f}, 4 bins off {off:.1f}, noise mean {noise_mean:.1f}")
        self.assertGreater(on, 3 * noise_mean)
        self.assertLess(off, 0.05 * on)

    def test_analytic_psd_warns_without_band(self):
        with self.assertWarnsRegex(UserWarning, "snr_band"):
            self.injector(self.PARAMS, 30.0, toy_psd(), angles=self.angles)

    def test_pickle_drops_generator(self):
        clone = pickle.loads(pickle.dumps(self.injector))
        self.assertIsNone(getattr(clone, "_generator", None))
        self.assertEqual(clone.n_samples, self.injector.n_samples)

    def test_matches_emrisearch(self):
        """``generate_emri_signal_and_sfts`` with its (retired) fastlisaresponse
        generator replaced by the LAT one: same draws, scaling, trimming, SFTs."""
        emri_utils = import_emrisearch("emri_utils")
        search_utils = import_emrisearch("search_utils")
        da_utils = import_emrisearch("da_utils")
        gen = self.injector.generator

        def response(T, dt):
            return lambda *args, eps, **_: gen(*args, eps=eps)

        # create_signal also calls few.has_backend, gone from the current FEW.
        original = emri_utils.get_response_generator, emri_utils.few
        emri_utils.get_response_generator = response
        emri_utils.few = types.SimpleNamespace(has_backend=lambda name: False)
        try:
            np.random.seed(3)
            ref = search_utils.generate_emri_signal_and_sfts(
                np.array([*self.PARAMS, 1.0]), self.T_DATA, T_SFT, DT, 30.0, self.T
            )
        finally:
            emri_utils.get_response_generator, emri_utils.few = original

        np.random.seed(3)
        inj = self.injector(self.PARAMS, 30.0, da_utils.psd, T_data=self.T_DATA)
        signal = inj.sft_data(T_SFT, noise=False)
        data = inj.sft_data(T_SFT)  # noise from the global RNG, as EMRI-Search draws it

        for name, value in ref["param_dict"].items():
            key = {"Tpl": "T_pl", "ef": "e_f"}.get(name, name)
            if key in inj.params:
                self.assertAlmostEqual(inj.params[key] / value, 1.0, places=10, msg=name)
        for got, want in ((signal.sfts[0], ref["signal_sfts"]), (data.sfts[0], ref["data_sfts"])):
            atol = 1e-9 * np.abs(want).max()
            np.testing.assert_allclose(got, want.T, rtol=1e-9, atol=atol)


if __name__ == "__main__":
    unittest.main()
