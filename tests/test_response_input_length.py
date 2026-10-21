"""pyResponseTDI accepts a waveform longer than ``num_pts`` (FEW returns ``n + 1``
samples for ``n = T / dt``): the TDI equals that of the waveform cut to ``num_pts``.

The projection kernel writes each link with stride ``num_inputs_per_source`` while
the TDI kernel read it with stride ``num_pts``, so a longer input used to fail the
``input_links`` length check (and would have mixed links). CPU, ~2 s.
"""

import unittest

import numpy as np

T_SPAN, T_BUF, DT = 86400.0, 10000.0, 5.0


class ResponseInputLengthTest(unittest.TestCase):
    def test_longer_input_matches_trimmed(self):
        from lisatools.detector import EqualArmlengthOrbits
        from lisatools.response.directresponse import pyResponseTDI

        orbits = EqualArmlengthOrbits(force_backend="cpu")
        n = int(T_SPAN / DT)
        t = np.arange(n + 1) * DT
        h = 1e-20 * np.exp(2j * np.pi * (3e-3 * t + 1e-9 * t**2))

        out = []
        for wave in (h, h[:n]):
            resp = pyResponseTDI(
                1.0 / DT,
                n,
                orbits=orbits,
                order=25,
                tdi="2nd generation",
                tdi_chan="AET",
                force_backend="cpu",
            )
            resp.get_projections(wave, 1.0, 0.3, t_buffer=T_BUF)
            out.append(np.asarray(resp.get_tdi_delays()))
        self.assertEqual(out[0].shape, (3, n))
        np.testing.assert_array_equal(out[0], out[1])


if __name__ == "__main__":
    unittest.main()
