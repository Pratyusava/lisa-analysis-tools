"""The catalogue-injection helpers of ``stock.erebor.injections``.

``gb_catalogue_to_sampled9`` (the monitor's and the 09 notebook's GB truth rows)
reproduces the catalogue amplitude from its distance / chirp-mass split and lays
the columns out in the store's ``gb`` key order; ``nearest_row`` picks a catalogue
source by one or several columns. ``load_catalogue`` / ``catalogue_sampling_rows``
are exercised by tests/test_set_store_sources_to_injection.py through the reset
script, which now calls them.
"""

import unittest

import numpy as np

from lisatools.globalfit.stock.erebor.injections import gb_catalogue_to_sampled9, nearest_row


def _entry(n=3, seed=0):
    rng = np.random.default_rng(seed)
    f0 = rng.uniform(1e-3, 1e-2, n)
    mc = rng.uniform(0.2, 1.0, n)
    d_mpc = rng.uniform(1.0, 20.0, n)
    from lisatools.globalfit.stock.erebor.transforms import McDistFdotAstroQuad, gb_amp_from_dist

    amp = gb_amp_from_dist(f0, mc, d_mpc * 1e3)
    _, _, fdot_gr, _ = McDistFdotAstroQuad()(d_mpc * 1e3, f0, mc, np.zeros(n))
    return dict(Amplitude=amp, GW22FrequencySSBFrame=f0,
                GW22FrequencyDerivativeSourceFrame=fdot_gr * 1.5,
                TrueAnomaly=rng.uniform(0, 2 * np.pi, n), InclinationAngle=rng.uniform(0, np.pi, n),
                PolarisationAngle=rng.uniform(0, np.pi, n), RightAscension=rng.uniform(0, 2 * np.pi, n),
                Declination=rng.uniform(-1.5, 1.5, n), LuminosityDistance=d_mpc, ChirpMassSSBFrame=mc)


class GBCatalogueToSampled9Test(unittest.TestCase):
    def test_columns_and_the_amplitude_check(self):
        e = _entry()
        rows, rel = gb_catalogue_to_sampled9(e)
        self.assertEqual(rows.shape, (3, 9))
        np.testing.assert_allclose(rows[:, 0], e["LuminosityDistance"] * 1e3)   # kpc
        np.testing.assert_allclose(rows[:, 1], e["GW22FrequencySSBFrame"] * 1e3)  # mHz
        np.testing.assert_allclose(rows[:, 2], e["ChirpMassSSBFrame"])
        np.testing.assert_allclose(rows[:, 8], 0.5, rtol=1e-6)                  # fdot = 1.5 x GR
        self.assertLess(rel, 1e-6)


class NearestRowTest(unittest.TestCase):
    def test_one_column_and_several(self):
        rows = np.array([[1.0, 10.0, 100.0], [1.1, 20.0, 100.0], [2.0, 10.0, 300.0]])
        self.assertEqual(nearest_row(rows, np.array([1.09, 0.0, 0.0]), [0]), 1)
        self.assertEqual(nearest_row(rows, np.array([1.05, 10.5, 90.0]), [1, 2]), 0)


if __name__ == "__main__":
    unittest.main()
