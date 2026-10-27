"""Builds emri_search_lorenzo.ipynb (run: python build_emri_search_notebook.py)."""

import nbformat as nbf

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# --------------------------------------------------------------------------- intro
md(r"""
# Semi-coherent EMRI search (Speri, Tenorio, Chapman-Bird & Gerosa 2026) on LISA Analysis Tools

This notebook runs the single-harmonic semi-coherent search of
[arXiv:2510.20891](https://arxiv.org/abs/2510.20891)
(code: [github.com/lorenzosp/EMRI-Search](https://github.com/lorenzosp/EMRI-Search))
with the modular re-implementation in `lisatools.search.emri`.

**Pipeline**

1. **Inject** a FEW Kerr eccentric equatorial EMRI through LISA's TDI response, scaled to a target SNR, into Gaussian noise.
2. **Segment** the data into short Fourier transforms (SFTs).
3. **Score** a frequency track with the semi-coherent linear-chirp (Fresnel) statistic, or with Cutler & Bustamante-Rosell's coherent excess-power statistic $H$ (Sec. 4b, `STATISTIC`).
4. **Parametrise** tracks by an SVD basis of FEW tracks, shifted in time to a trial plunge.
5. **Optimise** the basis coefficients with Lorenzo's Adam + differential-evolution hybrid, once per data duration.
6. **Background**: repeat on noise-only data, normalise the statistic per duration, fit a Gumbel distribution to the noise maxima, set false-alarm thresholds, and decide detection.

**Settings.** The cell under each *Inputs* heading runs the **toy problem** (minutes on a laptop).
The paper's and Lorenzo's repository settings sit next to every input, commented out (`# PAPER:` / `# REPO:`).
Switch them on for production runs.

| Run | Data | Search | Cost on an M3 Pro laptop (measured / projected) |
|---|---|---|---|
| Toy (active) | 0.2 yr, 6 durations | 128 members × 300 iterations | ~30 min incl. a 4-run background |
| One paper search | 1 yr, 48 durations | 512 members × 525 iterations | ~18 h |
| Paper results | 100 noise + 5 SNRs × 100 injections | as above | ~440 days (a cluster job) |

The follow-up stage of the paper (SBI, scipy DE, eryn MCMC) is not part of this notebook.
""")

md(r"""
## 0. Environment

Run in the `lat-emri` conda environment (`lisa-analysis-tools/scripts/emri_search/setup_env_macos.sh`),
with the `dev` branch of `lisa-analysis-tools` (`lisatools.search.emri`) installed and Lorenzo's
EMRI-Search repository checked out next to it (for his PSD table and SVD basis file).
""")

code(r"""
import os
import time
import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

jax.config.update("jax_enable_x64", True)  # the statistic needs float64 / complex128

from lisatools.search.emri import (
    Adam,
    AdamDE,
    EMRISearch,
    ExcessPowerStatistic,
    FEWHarmonicTrackModel,
    FEWInjector,
    FresnelStatistic,
    MatchedFilterStatistic,
    SegmentMask,
    SensitivityPSD,
    SVDBasis,
    SVDTrackModel,
    Track,
    band_snr,
)
from lisatools.utils.constants import YRSID_SI

warnings.filterwarnings("ignore", category=DeprecationWarning)
T_START = time.perf_counter()


def elapsed():
    return f"[{time.perf_counter() - T_START:7.1f} s]"


print("JAX devices:", jax.devices())
""")

# --------------------------------------------------------------------------- inputs
md(r"""
## 1. Inputs

Every knob of the run is defined here; nothing below hard-codes a setting.
Units: seconds [s], years [yr] (1 yr $=$ `YRSID_SI` $= 3.15581\times10^{7}$ s), hertz [Hz], solar masses [$M_\odot$].

### 1a. Paths

| Name | Meaning |
|---|---|
| `EMRI_SEARCH_DIR` | Lorenzo's EMRI-Search checkout (PSD table, SVD basis file) |
| `PSD_FILE` | Two-column table $(f, S_n(f))$ of the TDI2 A/E PSD used by the paper |
| `BASIS_FILE` | Lorenzo's SVD basis (`V_K`, `lb`, `ub`, ...) |
| `RESULTS_DIR` | Where the run's results are saved |
""")

code(r"""
EMRI_SEARCH_DIR = os.environ.get("EMRI_SEARCH_DIR", os.path.expanduser("~/Documents/Research/LISA/EMRI-Search"))
PSD_FILE = os.path.join(EMRI_SEARCH_DIR, "src/emrisearch/data/TDI2_AE_psd.npy")
BASIS_FILE = os.path.join(EMRI_SEARCH_DIR, "examples/new_svd_basis_Tsft50000.0_cut20.npz")
RESULTS_DIR = "results_toy"
# PAPER: RESULTS_DIR = "results_paper"
os.makedirs(RESULTS_DIR, exist_ok=True)
""")

md(r"""
### 1b. Data and SFTs

| Name | Symbol | Meaning |
|---|---|---|
| `DT` | $\Delta t$ | Sampling cadence [s] |
| `T_SFT` | $T_{\rm SFT}$ | Segment (SFT) duration [s] |
| `WINDOW_ALPHA` | $\alpha_w$ | Tukey taper fraction of each segment |
| `T_DATA` | $T_{\rm data}$ | Length of the analysed data [yr] |
| `T_SNR` | $T_{\rm SNR}$ | Length of the signal over which the SNR is defined [yr], $\ge T_{\rm data}$ |
| `CHANNELS` | | TDI channels analysed (the paper: A only) |
""")

code(r"""
DT = 5.0
T_SFT = 5e4
WINDOW_ALPHA = 0.2
T_DATA = 0.2           # PAPER / REPO: 1.0
T_SNR = 0.2            # PAPER / REPO: 1.0
CHANNELS = "A"         # PAPER / REPO: "A"
""")

md(r"""
### 1c. Source (the injection)

| Name | Symbol | Meaning |
|---|---|---|
| `M1`, `M2` | $m_1, m_2$ | Detector-frame masses [$M_\odot$] |
| `SPIN` | $a$ | Dimensionless MBH spin |
| `T_PL` | $T_{\rm pl}$ | Time from the data start to plunge [yr] |
| `E_F` | $e_f$ | Eccentricity at the separatrix |
| `SNR` | $\rho$ | Target optimal SNR over $T_{\rm SNR}$ |
| `SOURCE_SEED` | | Seed for the sky / spin angles and initial phases |
| `NOISE_SEED` | | Seed for the noise of the injection |

The toy places the plunge exactly at the end of one of the analysed durations (see Sec. 6),
which is where the search's plunge-time grid puts it; paper sources plunge anywhere in $[0.1, 1]$ yr.
Paper (Table 1) priors: $\log_{10} m_1 \sim U[\log_{10}5\times10^5, \log_{10}5\times10^6]$,
$m_2 \sim U[1, 100]$, $a \sim U[0, 0.998]$, $T_{\rm pl} \sim U[0.1, 1]$ yr, $e_f \sim U[0, 0.1]$.
""")

code(r"""
M1, M2, SPIN, E_F = 1e6, 10.0, 0.9, 0.05          # REPO (QuickStart): 1e6, 10.0, 0.9, 0.05
T_PL = (YRSID_SI - 520 * T_SFT) / YRSID_SI         # toy: plunge at the end of the 111-segment stream
# REPO (QuickStart): T_PL = 1.0
SNR = 30.0                                         # REPO: 30.0; PAPER: 20, 25, 30, 35, 40
SOURCE_SEED = 1                                    # REPO: 2601 (np.random.seed)
NOISE_SEED = 2

# PAPER: one draw from the Table 1 priors per injection
# rng_src = np.random.default_rng(SOURCE_SEED)
# M1 = 10 ** rng_src.uniform(np.log10(5e5), np.log10(5e6))
# M2 = rng_src.uniform(1.0, 100.0)
# SPIN = rng_src.uniform(0.0, 0.998)
# T_PL = rng_src.uniform(0.1, 1.0)
# E_F = rng_src.uniform(0.0, 0.1)

SOURCE = (M1, M2, SPIN, T_PL, E_F)
""")

md(r"""
### 1d. SVD track basis

| Name | Symbol | Meaning |
|---|---|---|
| `BASIS_SOURCE` | | `"lorenzo"`: search with `BASIS_FILE`; `"build"`: search with the basis built in Sec. 5 |
| `N_BANK` | $N$ | Training tracks drawn from the prior (paper: $10^4$; Lorenzo's file holds 46 371) |
| `BANK_PRIOR` | | Uniform ranges of $(\log_{10} m_1, m_2, a, e_f)$; training tracks plunge at $T_{\rm ref} = 1$ yr |
| `N_MODES` | $K$ | Retained singular vectors |
| `BOX_QUANTILE` | $q$ | Coefficient box $[Q_q, Q_{1-q}]$ per mode (mode 0's upper edge: the maximum), as in Lorenzo's file |
| `BANK_SEED` | | Seed of the training draws |

The toy searches with Lorenzo's basis: a few hundred training tracks give too narrow a box to reach the source (Sec. 5 shows the check).
""")

code(r"""
BASIS_SOURCE = "lorenzo"     # PAPER / REPO: "lorenzo" (or "build" with N_BANK = 10_000 .. 46_371)
N_BANK = 200                 # PAPER: 10_000;  Lorenzo's file: 46_371 tracks
BANK_PRIOR = {               # PAPER (Table 1)
    "log10_m1": (np.log10(5e5), np.log10(5e6)),
    "m2": (1.0, 100.0),
    "a": (0.0, 0.998),
    "e_f": (0.0, 0.1),
}
N_MODES = 20                 # PAPER / REPO: 20
BOX_QUANTILE = 0.015         # REPO (exact match to lb / ub in BASIS_FILE); paper text: "symmetric 3 sigma"
BANK_SEED = 2601
T_REF = YRSID_SI             # basis tracks plunge at 1 yr
""")

md(r"""
### 1e. Statistic

| Name | Symbol | Meaning |
|---|---|---|
| `P` | $P$ | Half-width of the bin window around the track ($2P+1$ bins per segment) |
| `MASK` | $f_{\min}, \dot f_{\min}, \delta\phi_{\max}$ | Segments counted: $f > f_{\min}$, $\dot f > \dot f_{\min}$, $\lvert\ddot f\rvert T_{\rm SFT}^3/6 < \delta\phi_{\max}$ |
| `STATISTIC` | | What the search maximises: `"fresnel"` (Lorenzo's $\mathcal S$, Sec. 4) or `"H"` (Cutler & Bustamante-Rosell, Sec. 4b) |
| `H_BAND` | | Frequency band of the coherent FFT $\tilde d(f)$ that $H$ reads [Hz]; must contain the tracks |
| `H_WINDOWS` | $W$ | Delay-window lengths $N\,\delta\tau$ that $H$ maximises over [s] |
| `H_DELAY_STEP` | $\delta\tau$ | Target spacing of the delay grid [s] |
| `H_NORMALIZATION` | $Q(N)$ | `"exact"`: finite-window variance per template; `"asymptotic"`: the draft's large-window $R = L/K$ |
""")

code(r"""
P = 100                                                   # PAPER / REPO: 100
MASK = SegmentMask(f_min=10**-3.5, fdot_min=1e-13, dphi_max=1.0)   # PAPER / REPO
STATISTIC = "fresnel"                                     # PAPER / REPO;  "H": Cutler & Bustamante-Rosell
H_BAND = (3e-4, 2e-2)
H_WINDOWS = (2e3, 4.2e3, 1.72e4, 6.9e4)                   # CB-R draft: {10, 21, 86, 345} cells of ~200 s
H_DELAY_STEP = 200.0                                      # CB-R draft: ~200 s
H_NORMALIZATION = "exact"                                 # CB-R draft: "asymptotic"
""")

md(r"""
### 1f. Search

| Name | Symbol | Meaning |
|---|---|---|
| `T_MIN` | $T_{\min}$ | Shortest analysed data stream [yr] |
| `DURATION_STEP` | | Increment of the stream length [segments] (12 SFTs $\approx$ 7 d) |
| `N_INITIAL` | | Space-filling initial draws, screened by the statistic |
| `N_POPULATION` | $N_p$ | Best initial draws kept and optimised |
| `N_STEPS` | | Adam + DE iterations (the loop runs `N_STEPS + 1` times, as Lorenzo's) |
| `LEARNING_RATE` | $\eta$ | Adam step size in unit-cube coordinates |
| `F_RANGE` | $[F_{\min}, F_{\max}]$ | DE differential weight $F \sim U(F_{\min}, F_{\max})$ per iteration |
| `P_ELITISM` | $p_e$ | Probability that the DE base vector is the fittest member |
| `F0_RANGE` | | Starting frequencies of the initial tracks [Hz] (the toy uses the paper's range) |
| `BATCH_SIZE` | | Members evaluated at a time (memory); `None` = all at once |
| `SEARCH_SEED` | | Seed of initial draws and DE randomness |

The toy keeps the paper's iteration logic but a smaller population and fewer iterations. On the toy's 111-segment stream,
64 members x 200 iterations found the source in 1 of 2 seeds, 128 x 300 in 2 of 2 (with the paper's $f_0$ range).
""")

code(r"""
T_MIN = 0.1                 # PAPER: 0.1 yr
DURATION_STEP = 12          # PAPER: 12 segments  (48 durations for 1 yr)
N_INITIAL = __N_INITIAL__   # PAPER: 1024; REPO (QuickStart): 128
N_POPULATION = __N_POP__    # PAPER: 512;  REPO (QuickStart): 64
N_STEPS = __N_STEPS__       # PAPER: 525;  REPO (QuickStart): 200
LEARNING_RATE = 1e-5        # PAPER / REPO: 1e-5
F_RANGE = (0.3, 1.5)        # PAPER / REPO
P_ELITISM = 0.5             # REPO (np.random.choice([True, False])); not stated in the paper
F0_RANGE = (5e-4, 1e-2)     # PAPER
# REPO (QuickStart, which starts near the true track): F0_RANGE = (1e-3, 5e-3)
BATCH_SIZE = None           # 18 GB laptop, 1 yr: 128;  64 GB: 256 or None
SEARCH_SEED = 0
""")

md(r"""
### 1g. Background and detection

| Name | Meaning |
|---|---|
| `N_NOISE` | Noise-only realisations searched (paper: 100) |
| `P_FA` | False-alarm probabilities of the detection thresholds |
| `PAPER_GUMBEL` | The paper's noise fit (location, scale), for reference |
""")

code(r"""
N_NOISE = 4                        # PAPER: 100
P_FA = (0.5, 1e-2, 1e-4)           # PAPER
PAPER_GUMBEL = (1.94, 0.56)        # PAPER Sec. III.1
""")

# --------------------------------------------------------------------------- injection
md(r"""
## 2. Injection

FEW's `FastKerrEccentricEquatorialFlux` waveform is integrated **backwards** from just outside the separatrix,
$p_f = p_{\rm sep}(a, e_f, x_f) + \Delta p_{\rm buffer}$, for a time $T_{\rm pl}$, giving the initial conditions $(p_0, e_0, x_0)$.
LAT's response turns it into the second-generation TDI channels $A, E, T$ (equal-arm orbits).
The signal is scaled to the target SNR over its first $T_{\rm SNR}$, with the one-sided PSD $S_n(f)$:

$$
\rho^2 = 4\,\Delta f \sum_{k} \frac{\lvert \tilde h(f_k) \rvert^2}{S_n(f_k)}, \qquad
\tilde h(f_k) = \Delta t\, \mathrm{rFFT}[h](f_k), \qquad \Delta f = \frac{1}{N\Delta t},
$$

$h \to h\,\rho_{\rm target}/\rho$ and $d_L \to d_L\,\rho/\rho_{\rm target}$; the scaled signal is then cut to $T_{\rm data}$.
The PSD is the paper's tabulated TDI2 A/E curve, raised to its first value below its minimum and held constant below $10^{-4}$ Hz.
(LAT's analytic `SensitivityPSD()` also works, with `snr_band=(0, 0.025)`: its transfer-function zeros near 0.03 Hz would dominate the full-band sum.)

The injector returns an `EMRIInjection` (signal, parameters, PSD). `with_noise(rng)` returns a copy carrying one fixed
Gaussian realisation $n(t)$ with PSD $S_n$, so the data are $d(t) = h(t) + n(t)$, and every stream taken from that
object below (each duration, with or without the signal) shares the same $n(t)$, as in the paper.
""")

code(r"""
psd = SensitivityPSD.from_file(PSD_FILE)
injector = FEWInjector(T_SNR, dt=DT, channels=CHANNELS)
injection = injector(SOURCE, SNR, psd, T_data=T_DATA, rng=np.random.default_rng(SOURCE_SEED))
injection = injection.with_noise(rng=np.random.default_rng(NOISE_SEED))   # d = h + n, n fixed from here on
print(elapsed(), "injection:", {k: round(v, 4) if isinstance(v, float) else v for k, v in injection.params.items()})
print(f"SNR of the analysed {T_DATA} yr: {band_snr(injection.signal, DT, psd):.2f}  (target {SNR} over {T_SNR} yr)")
""")

# --------------------------------------------------------------------------- SFTs
md(r"""
## 3. Short Fourier transforms

The data are cut into $N_{\rm seg} = \lfloor T_{\rm data} / T_{\rm SFT} \rfloor$ segments starting at $t_\alpha = \alpha\,T_{\rm SFT}$.
Each is Tukey-windowed ($w_j$) and Fourier transformed:

$$
\tilde d_\alpha[k] = \Delta t \sum_{j=0}^{N_s - 1} w_j\, d(t_\alpha + j\Delta t)\, e^{-2\pi i jk/N_s},
\qquad N_s = T_{\rm SFT}/\Delta t, \qquad f_k = k/T_{\rm SFT}.
$$

`injection.sft_data(T_SFT, signal=..., noise=..., n_segments=n)` returns these as an `SFTData` for the stream of the
first $n$ segments: signal + noise (default), signal only (`noise=False`) or noise only (`signal=False`).
""")

code(r"""
full = injection.sft_data(T_SFT, window_alpha=WINDOW_ALPHA)                    # h + n
signal_only = injection.sft_data(T_SFT, noise=False, window_alpha=WINDOW_ALPHA)  # h
N_SEG_TOTAL = full.n_segments
print(f"{full.n_segments} segments x {full.sfts.shape[-1]} bins (df = {1 / T_SFT:.1e} Hz)")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
white = np.abs(full.sfts[0]) / np.sqrt(psd(full.freqs) * T_SFT / 2)
for a_, arr, title in ((ax[0], np.abs(signal_only.sfts[0]) / np.sqrt(psd(full.freqs) * T_SFT / 2), "signal"),
                       (ax[1], white, "signal + noise")):
    a_.pcolormesh(full.t_alpha / 86400, full.freqs * 1e3, arr.T, shading="auto")
    a_.set_title(f"whitened |SFT|, {title}")
    a_.set_xlabel("t [d]")
ax[0].set_ylabel("f [mHz]")
ax[0].set_ylim(1, 12)
plt.tight_layout()
""")

# --------------------------------------------------------------------------- statistic
md(r"""
## 4. The semi-coherent statistic

In segment $\alpha$ the dominant ($m = 2$) harmonic is modelled as a linear chirp with local frequency $f_\alpha$ and drift $\dot f_\alpha$.
Its Fourier transform is the **Fresnel kernel**

$$
K(f_0, \dot f, T) = \int_0^{T} e^{2\pi i\,(f_0 \tau + \dot f \tau^2/2)}\, d\tau
= \frac{e^{-i\pi f_0^2/\dot f}}{\sqrt{2\dot f}}\Big[\,C(u) + i S(u)\,\Big]_{u_0}^{u_1},
\quad u_0 = \frac{f_0\sqrt{2\dot f}}{\dot f},\ u_1 = u_0 + T\sqrt{2\dot f},
$$

with $C, S$ the Fresnel integrals (a Dirichlet kernel for $\dot f = 0$; $\dot f < 0$ by conjugation).
The phase-maximised matched filter over the $2P+1$ bins $k \in \lfloor f_\alpha T_{\rm SFT}\rfloor + [-P, P]$ is (paper Eq. 7)

$$
c_\alpha = \Delta f \sum_{k} \frac{\tilde d_\alpha^*[k]\; K(f_\alpha - f_k, \dot f_\alpha, T_{\rm SFT})}{S_n(f_k)},
\qquad \lvert\langle d \mid h_\alpha\rangle\rvert = 2\lvert c_\alpha \rvert,
\qquad \langle h_\alpha \mid h_\alpha\rangle = \frac{T_{\rm SFT}}{2 S_n(f_\alpha)},
$$

and the statistic (paper Eq. 6, Lorenzo's normalisation) sums the masked segments:

$$
\mathcal S = \frac12 \sum_{\alpha} w_\alpha\, \frac{\lvert\langle d \mid h_\alpha\rangle\rvert^2}{\langle h_\alpha \mid h_\alpha\rangle},
\qquad
w_\alpha = \Theta\!\left(f_\alpha - f_{\min}\right)\Theta\!\left(\dot f_\alpha - \dot f_{\min}\right)\Theta\!\left(\delta\phi_{\max} - \tfrac16\lvert\ddot f_\alpha\rvert T_{\rm SFT}^3\right).
$$

The last cut keeps segments where the neglected cubic phase is below $\delta\phi_{\max}$ (the linear chirp holds).
In noise each term is exponentially distributed with mean $2\!\int w^2/T = 2(1 - 5\alpha_w/8) = 1.75$ for $\alpha_w = 0.2$.

Below: the statistic on the injection's true frequency tracks (FEW, integrated back from plunge) for a few harmonics $m\Phi_\phi + n\Phi_r$.
""")

code(r"""
stat_full = FresnelStatistic.for_data(full, P=P, mask=MASK, force_backend="cpu")
for m, n in ((2, 0), (2, 1), (2, -1), (3, 0), (1, 0)):
    tr = FEWHarmonicTrackModel(full.t_alpha, m=m, n=n)(SOURCE)
    print(f"harmonic (m={m}, n={n:+d}): S(data) = {stat_full.evaluate(full.sfts, tr):7.1f}   "
          f"S(signal only) = {stat_full.evaluate(signal_only.sfts, tr):7.1f}   "
          f"segments kept {int(MASK(tr, T_SFT, np).sum())}")
true_track = FEWHarmonicTrackModel(full.t_alpha, m=2, n=0)(SOURCE)
noise_mean = 1.75 * MASK(true_track, T_SFT, np).sum()
print(f"noise-only expectation on the m=2 track: ~{noise_mean:.0f}")
print(f"true track starts at f = {true_track.f[0]:.3e} Hz; F0_RANGE = {F0_RANGE}")
""")

# --------------------------------------------------------------------------- H statistic
md(r"""
## 4b. Alternative statistic: Cutler & Bustamante-Rosell's $H$

Instead of cutting the data into segments, $H$ correlates **one full-length template** $h$ with the whole stream at every
time delay $\tau_j = j\,\delta\tau$ (one inverse FFT):

$$
\rho_j = \langle d \mid h_{\tau_j}\rangle = 4\,\Delta f \sum_k \frac{\tilde d_k^*\, \tilde h_k\, e^{-2\pi i f_k \tau_j}}{S_n(f_k)},
\qquad \langle h\mid h\rangle = 1, \qquad \mathbb E_{\rm noise}\lvert\rho_j\rvert^2 = 2 .
$$

A template far from the signal overlaps it only where their f–t tracks **cross**, coherently for
$t_{\rm coh} \approx \lvert\Delta\dot f\rvert^{-1/2}$; time-slides of the template still cross, so the overlap spreads over
many consecutive delays. $H$ collects that excess power over a window of $N$ delays, normalised to unit noise variance, and maximises
over the window's start $i$ and length $N \in W$ (draft Eqs. 4, A4–A5):

$$
H = \max_{N \in W}\, \max_i \frac{\sum_{j=i}^{i+N-1}\big(\lvert\rho_j\rvert^2 - 2\big)}{\sqrt{4 N\, Q(N)}},
\qquad
Q(N) = \sum_{\lvert m\rvert < N}\Big(1 - \frac{\lvert m\rvert}{N}\Big)\lvert\langle h_0 \mid h_m\rangle\rvert^2 .
$$

$Q(N)$ corrects for the correlation of neighbouring delays (`"exact"`, per template). The draft uses its large-window limit
$R = L/K$, which only holds when the template spans a wide band (`"asymptotic"`). The matched filter
$M = \max_j \lvert\rho_j\rvert$ is the coherent comparator.

**Templates from our tracks.** The track model is unchanged (Sec. 5–6); the template follows by stationary phase:
$t(f)$ by inverting the track, $\Psi(f) = 2\pi\!\int t\,df$, amplitude $\lvert\tilde h\rvert \propto \dot f^{-1/2}$
(constant-amplitude chirp), with $\tilde h = \lvert\tilde h\rvert e^{-i\Psi}$. LISA's response is not in the template.

**Below:** how $\mathcal S$, $M$ and $H$ respond as a template is rotated away from the true track about the middle of the
stream, $f(t) \to f(t) + \epsilon\,(t - t_c)$: every such template still **crosses** the truth at $t_c$.
Each response is $(\text{value} - \mu_0)/\sigma_0$, with $\mu_0, \sigma_0$ from noise-only draws at the true template (draft Fig. 4).
""")

code(r"""
fd_full = injection.fd_data(min_freq=H_BAND[0], max_freq=H_BAND[1])
t_segments = full.t_alpha
H_stat = ExcessPowerStatistic.for_data(fd_full, t_segments, windows=H_WINDOWS, delay_step=H_DELAY_STEP,
                                       normalization=H_NORMALIZATION, force_backend="cpu")
M_stat = MatchedFilterStatistic.for_data(fd_full, t_segments, delay_step=H_DELAY_STEP, force_backend="cpu")
print(f"FFT band {H_BAND} Hz: {fd_full.n_bins} bins; delay step {H_stat.delay_step:.0f} s; windows {H_stat.cells} cells")

t_c = 0.5 * N_SEG_TOTAL * T_SFT
EPS = np.array([0.0, 1e-13, 3e-13, 1e-12, 3e-12, 1e-11, 3e-11])   # fdot offsets [Hz/s]


def rotated(eps):
    return Track(true_track.f + eps * (t_segments - t_c), true_track.fdot + eps, true_track.fddot)


def all_statistics(sfts_arr, fd_arr, track):
    return (float(stat_full.evaluate(sfts_arr, track)), float(M_stat.evaluate(fd_arr, track)),
            float(H_stat.evaluate(fd_arr, track)))


N_CAL = 20
cal = []
for r in range(N_CAL):
    noise_r = injection.with_noise(rng=np.random.default_rng(20_000 + r))
    cal.append(all_statistics(stat_full.prepare(noise_r.sft_data(T_SFT, signal=False, window_alpha=WINDOW_ALPHA)),
                              H_stat.prepare(noise_r.fd_data(signal=False, min_freq=H_BAND[0], max_freq=H_BAND[1])),
                              true_track))
mu0, sigma0 = np.mean(cal, axis=0), np.std(cal, axis=0, ddof=1)
sfts_arr, fd_arr = stat_full.prepare(full), H_stat.prepare(fd_full)
response = np.array([(np.array(all_statistics(sfts_arr, fd_arr, rotated(e))) - mu0) / sigma0 for e in EPS])
for e, (zs, zm, zh) in zip(EPS, response):
    print(f"eps = {e:7.1e} Hz/s:  S {zs:8.1f}   M {zm:8.1f}   H {zh:8.1f}   (noise sigma units)")
print(elapsed(), "response computed")

plt.figure(figsize=(6, 3.6))
x = np.where(EPS > 0, EPS, EPS[1] / 3)
for k, name in enumerate((r"$\mathcal{S}$ (Fresnel)", r"$M = \max|\rho|$", r"$H$")):
    plt.loglog(x, np.clip(response[:, k], 0.1, None), "o-", label=name)
plt.axhline(3, color="grey", ls=":", lw=0.8)
plt.xlabel(r"$\epsilon$ [Hz/s] (leftmost: true track)"); plt.ylabel("(value - noise mean) / noise std")
plt.legend(); plt.tight_layout()
""")

# --------------------------------------------------------------------------- basis
md(r"""
## 5. SVD basis of frequency tracks

A bank of $N$ FEW $m = 2$ tracks $f_{i,\alpha} = f(t_\alpha;\theta_i)$, all plunging at $T_{\rm ref} = 1$ yr, on the 1-yr segment grid
$t_\alpha = \alpha T_{\rm SFT}$, $\alpha = 0, \dots, 631$ (the 632nd sample starts a partial segment past the 631 full ones and is set to zero),
is decomposed and truncated (paper Eqs. 9–10):

$$
f_{i,\alpha} = \sum_{p} U_{ip}\, S_p\, V_{p\alpha} \;\approx\; \sum_{p=1}^{K} W_{ip}\, V_{p\alpha}, \qquad W_{ip} = U_{ip} S_p .
$$

A search track is any $f_\alpha(c) = \sum_p c_p V_{p\alpha}$ with coefficients in the box $c_p \in [\ell_p, u_p]$,
$\ell_p = Q_{q}(W_{\cdot p})$, $u_p = Q_{1-q}(W_{\cdot p})$ ($u_1 = \max_i W_{i1}$), parametrised by the unit cube
$\xi \in [0,1]^K$: $c_p = \ell_p + \xi_p (u_p - \ell_p)$.

The cell builds a small bank (`N_BANK`) with the paper's priors, then checks two things against Lorenzo's file:
that his 20 modes span the new tracks, and whether each basis's box contains the injected source.
""")

code(r"""
t_basis = np.arange(0.0, YRSID_SI, T_SFT)              # 632 samples, as V_K
bank_model = FEWHarmonicTrackModel(t_basis, m=2, n=0)
rng_bank = np.random.default_rng(BANK_SEED)
lo = np.array([BANK_PRIOR[k][0] for k in ("log10_m1", "m2", "a", "e_f")])
hi = np.array([BANK_PRIOR[k][1] for k in ("log10_m1", "m2", "a", "e_f")])
bank = []
for log10_m1, m2, a, e_f in rng_bank.uniform(lo, hi, size=(N_BANK, 4)):
    try:
        f = bank_model((10**log10_m1, m2, a, T_REF / YRSID_SI, e_f)).f
    except Exception:          # FEW cannot integrate back a full year for this draw
        continue
    if f[0] != 0.0:            # keep tracks that span the whole year
        f[-1] = 0.0            # partial 632nd segment
        bank.append(f)
bank = np.array(bank)
print(elapsed(), f"bank: {len(bank)} of {N_BANK} draws kept")

U, S_vals, Vt = np.linalg.svd(bank, full_matrices=False)
W = U[:, :N_MODES] * S_vals[:N_MODES]
lb, ub = np.quantile(W, BOX_QUANTILE, axis=0), np.quantile(W, 1 - BOX_QUANTILE, axis=0)
ub[0] = W[:, 0].max()
built_basis = SVDBasis(Vt[:N_MODES], lb, ub, S=S_vals[:N_MODES])
lorenzo_basis = SVDBasis.from_npz(BASIS_FILE)


def projection_error(tracks, basis):
    rec = (tracks @ basis.V.T) @ basis.V
    return np.linalg.norm(tracks - rec, axis=1) / np.linalg.norm(tracks, axis=1)


print(f"new tracks on Lorenzo's {lorenzo_basis.n_modes} modes: median rel. error {np.median(projection_error(bank, lorenzo_basis)):.1e}")
f_src = bank_model((M1, M2, SPIN, 1.0, E_F)).f
f_src[-1] = 0.0                                      # the source as a 1-yr basis track
for name, b in (("built", built_basis), ("Lorenzo", lorenzo_basis)):
    xi = (f_src @ b.V.T - b.lb) / (b.ub - b.lb)
    print(f"{name:8s} basis: source xi in [{xi.min():.2f}, {xi.max():.2f}]; modes outside the box: {list(np.where((xi < 0) | (xi > 1))[0])}")

basis = lorenzo_basis if BASIS_SOURCE == "lorenzo" else built_basis

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].semilogy(S_vals[:40] / S_vals[0], "o-", label="built bank")
if lorenzo_basis.S is not None:
    ax[0].semilogy(lorenzo_basis.S / lorenzo_basis.S[0], "s", mfc="none", label="Lorenzo's file")
ax[0].set_xlabel("mode p"); ax[0].set_ylabel(r"$S_p / S_1$"); ax[0].legend()
for i in np.linspace(0, len(bank) - 1, 15, dtype=int):
    ax[1].plot(t_basis / YRSID_SI, bank[i] * 1e3, lw=0.8)
ax[1].set_xlabel("t [yr]"); ax[1].set_ylabel("f [mHz]"); ax[1].set_title("training tracks (plunge at 1 yr)")
plt.tight_layout()
""")

# --------------------------------------------------------------------------- track model
md(r"""
## 6. Plunge-time shift and data durations

The basis tracks plunge at $T_{\rm ref}$. A trial plunge time $T_{\rm pl}$ rolls the expansion by

$$
s = \operatorname{trunc}\!\left(\frac{T_{\rm pl} - T_{\rm ref}}{T_{\rm SFT}}\right),
\qquad f_\alpha \to f_{\alpha - s} \quad (\text{vacated samples} = 0),
$$

then keeps the data's segments; $\dot f_\alpha$ is the central difference of $f_\alpha$ over $t_\alpha$ and
$\ddot f_\alpha$ (for the mask) that of $\dot f_\alpha$.
The paper analyses progressively longer streams of $n_k = n_{\min} + 12k$ segments, from $T_{\min} = 0.1$ yr,
and assumes the plunge at the end of each stream: $T_{\rm pl} = n_k T_{\rm SFT}$.
""")

code(r"""
N_MIN = int(T_MIN * YRSID_SI / T_SFT)
DURATIONS = list(range(N_MIN, N_SEG_TOTAL + 1, DURATION_STEP))
probe = SVDTrackModel(basis, full.t_alpha, T_ref=T_REF, force_backend="cpu")
print(f"{len(DURATIONS)} durations [segments]: {DURATIONS}")
print("plunge shifts:", [probe.shift_for_plunge(n * T_SFT) for n in DURATIONS])
print(f"source plunges at segment {T_PL * YRSID_SI / T_SFT:.2f}")
""")

# --------------------------------------------------------------------------- optimiser
md(r"""
## 7. Optimiser: Adam + differential evolution

**Initial population.** `N_INITIAL` Sobol points in the coefficient box; each track's offset is reset so it starts at
$f_0 \sim U(\texttt{F0\_RANGE})$, draws with a negative frequency are dropped, and the `N_POPULATION` best by $\mathcal S$ are kept.

**Loss.** $L(\xi) = -\mathcal S\big(f(\xi;\,s)\big)$, differentiable in $\xi$ through the Fresnel kernel (JAX).

**One iteration**, for every member $i$:

1. Adam step (Kingma & Ba) on $\nabla_\xi L$, learning rate $\eta$, clipped to $[0,1]^K$.
2. DE proposal without crossover (paper Eq. 11), with $F \sim U(F_{\min}, F_{\max})$ and the base vector the fittest member with probability $p_e$, else a random one:
$$
v_i = x_{a} + F\,(x_{b} - x_{c}), \qquad a, b, c \text{ distinct}, \ne i, \qquad v_i \to \operatorname{clip}(v_i, 0, 1).
$$
3. Greedy selection: $x_i \leftarrow v_i$ if $L(v_i) < L(x_i)$.

As in Lorenzo's loop, the selection compares the trial with the loss *before* the Adam step (`stale_selection=True`); the final population is re-scored.
""")

code(r"""
def make_optimizer():
    return AdamDE(
        n_steps=N_STEPS,
        adam=Adam(learning_rate=LEARNING_RATE),
        F_range=F_RANGE,
        p_elitism=P_ELITISM,
        save_every=10,
        batch_size=BATCH_SIZE,
    )


def make_data_and_statistic(stream, n_seg, signal=True, force_backend="jax"):
    # The data the chosen statistic reads, for the first n_seg segments, and the statistic on it.
    t_alpha = np.arange(n_seg) * T_SFT
    if STATISTIC == "fresnel":
        data = stream.sft_data(T_SFT, signal=signal, n_segments=n_seg, window_alpha=WINDOW_ALPHA)
        return data, FresnelStatistic.for_data(data, P=P, mask=MASK, force_backend=force_backend), t_alpha
    data = stream.fd_data(signal=signal, duration=n_seg * T_SFT, min_freq=H_BAND[0], max_freq=H_BAND[1])
    stat = ExcessPowerStatistic.for_data(data, t_alpha, windows=H_WINDOWS, delay_step=H_DELAY_STEP,
                                         normalization=H_NORMALIZATION, force_backend=force_backend)
    return data, stat, t_alpha


def search_stream(stream, n_seg, seed, signal=True):
    # Best track for the first n_seg segments of `stream` (an EMRIInjection with fixed noise),
    # plunge at the end of the stream; signal=False searches its noise alone.
    data, stat, t_alpha = make_data_and_statistic(stream, n_seg, signal)
    model = SVDTrackModel(basis, t_alpha, T_ref=T_REF, force_backend="jax")
    search = EMRISearch(data, stat, model, make_optimizer(), n_initial=N_INITIAL,
                        n_population=N_POPULATION, initial_kwargs={"f0_range": F0_RANGE})
    return search.run(T_pl=n_seg * T_SFT, seed=seed), data, stat
""")

# --------------------------------------------------------------------------- injection search
md(r"""
## 8. Search the injection

One optimisation per duration $n_k$; for each we record the best statistic $\mathcal S_k$ and compare it with the
statistic of the true track on the same stream.
""")

code(r"""
injection_results = {}
for k, n_seg in enumerate(DURATIONS):
    res, data_k, stat_k = search_stream(injection, n_seg, seed=SEARCH_SEED + k)
    tt = Track(true_track.f[:n_seg], true_track.fdot[:n_seg], true_track.fddot[:n_seg])
    s_true = float(stat_k.evaluate(stat_k.prepare(data_k), tt))
    injection_results[n_seg] = dict(result=res, true_statistic=s_true)
    print(f"{elapsed()} n = {n_seg:4d} segments: best S = {res.best_statistic:8.1f}   true track S = {s_true:8.1f}")
np.savez(os.path.join(RESULTS_DIR, "injection.npz"),
         durations=DURATIONS,
         best_statistic=[injection_results[n]["result"].best_statistic for n in DURATIONS],
         true_statistic=[injection_results[n]["true_statistic"] for n in DURATIONS],
         best_x=[injection_results[n]["result"].best_x for n in DURATIONS],
         params=str(injection.params))
""")

code(r"""
best_n = max(DURATIONS, key=lambda n: injection_results[n]["result"].best_statistic)
best = injection_results[best_n]["result"]
f_best = np.asarray(best.best_track.f)
t_k = full.t_alpha[:best_n]

fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
ax[0].plot(DURATIONS, [injection_results[n]["result"].best_statistic for n in DURATIONS], "o-", label="search best")
ax[0].plot(DURATIONS, [injection_results[n]["true_statistic"] for n in DURATIONS], "k--", label="true track")
ax[0].axvline(T_PL * YRSID_SI / T_SFT, color="grey", ls=":", label="true plunge")
ax[0].set_xlabel("stream length [segments]"); ax[0].set_ylabel(r"$\mathcal{S}$"); ax[0].legend()
ax[1].plot(t_k / 86400, true_track.f[:best_n] * 1e3, "k", lw=3, alpha=0.3, label="true m=2 track")
ax[1].plot(t_k / 86400, f_best * 1e3, "C1", label=f"best fit (n = {best_n})")
ax[1].set_xlabel("t [d]"); ax[1].set_ylabel("f [mHz]"); ax[1].legend()
ok = true_track.f[:best_n] > 0
ax[2].semilogy(t_k[ok] / 86400, np.abs(f_best[ok] / true_track.f[:best_n][ok] - 1))
ax[2].set_xlabel("t [d]"); ax[2].set_ylabel("|relative frequency error|")
plt.tight_layout()
hist = best.optimization.history
plt.figure(figsize=(5, 3.2))
plt.plot(hist["step"], [-l[0] for l in hist["loss"]])
plt.xlabel("iteration"); plt.ylabel(r"best $\mathcal{S}$"); plt.title(f"convergence, n = {best_n}")
plt.tight_layout()
""")

# --------------------------------------------------------------------------- background
md(r"""
## 9. Noise background, normalisation and detection

The same search runs on `N_NOISE` noise-only realisations. Because $\mathcal S$ grows with the number of segments,
the paper normalises it **per duration** with the noise-only mean and standard deviation, and the detection statistic
of a dataset is the maximum over durations:

$$
\bar{\mathcal S}_k = \frac{\mathcal S_k - \mu_k}{\sigma_k},
\qquad \Lambda = \max_k \bar{\mathcal S}_k .
$$

The noise-only $\Lambda$ follow a Gumbel distribution (extreme-value theory),
$P(\Lambda < x) = \exp\!\big[-e^{-(x - \mu_G)/\beta_G}\big]$ (paper: $\mu_G = 1.94$, $\beta_G = 0.56$).
A false-alarm probability $p_{\rm FA}$ sets the threshold

$$
\Lambda_{\rm th}(p_{\rm FA}) = \mu_G - \beta_G \ln\!\big[-\ln(1 - p_{\rm FA})\big],
$$

and a dataset is a detection when $\Lambda > \Lambda_{\rm th}$. The detection probability at a given SNR is the fraction
of injections detected (paper Eqs. 12–13). With the toy's `N_NOISE = 4` the fit only illustrates the procedure.
""")

code(r"""
noise_statistics = np.zeros((N_NOISE, len(DURATIONS)))
for r in range(N_NOISE):
    noise_r = injection.with_noise(rng=np.random.default_rng(10_000 + r))   # a new n(t), same grid
    for k, n_seg in enumerate(DURATIONS):
        res, _, _ = search_stream(noise_r, n_seg, seed=SEARCH_SEED + 1000 * (r + 1) + k, signal=False)
        noise_statistics[r, k] = res.best_statistic
    print(f"{elapsed()} noise run {r + 1}/{N_NOISE}: max S = {noise_statistics[r].max():.1f}")
np.savez(os.path.join(RESULTS_DIR, "background.npz"), durations=DURATIONS, best_statistic=noise_statistics)
""")

code(r"""
mu_k = noise_statistics.mean(axis=0)
sigma_k = noise_statistics.std(axis=0, ddof=1)
Lambda_noise = ((noise_statistics - mu_k) / sigma_k).max(axis=1)
S_inj = np.array([injection_results[n]["result"].best_statistic for n in DURATIONS])
Lambda_inj = ((S_inj - mu_k) / sigma_k).max()

loc, scale = stats.gumbel_r.fit(Lambda_noise)
print(f"Gumbel fit to {N_NOISE} noise maxima: location {loc:.2f}, scale {scale:.2f}   (paper: {PAPER_GUMBEL})")
for p in P_FA:
    th = stats.gumbel_r.isf(p, loc, scale)
    print(f"p_FA = {p:g}: threshold Lambda_th = {th:6.2f}   ->  injection Lambda = {Lambda_inj:.1f}: "
          f"{'DETECTED' if Lambda_inj > th else 'not detected'}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for r in range(N_NOISE):
    ax[0].plot(DURATIONS, noise_statistics[r], color="grey", alpha=0.6)
ax[0].plot(DURATIONS, S_inj, "o-C1", label="injection")
ax[0].set_xlabel("stream length [segments]"); ax[0].set_ylabel(r"best $\mathcal{S}_k$"); ax[0].legend()
x = np.linspace(min(Lambda_noise.min(), 0) - 1, max(Lambda_noise.max(), 5) + 1, 300)
ax[1].hist(Lambda_noise, bins=max(N_NOISE // 5, 3), density=True, alpha=0.5, label="noise maxima")
ax[1].plot(x, stats.gumbel_r.pdf(x, loc, scale), label="Gumbel fit")
ax[1].plot(x, stats.gumbel_r.pdf(x, *PAPER_GUMBEL), "--", label="paper fit")
for p in P_FA:
    ax[1].axvline(stats.gumbel_r.isf(p, loc, scale), color="k", ls=":", lw=0.8)
ax[1].set_xlabel(r"$\Lambda$"); ax[1].legend()
ax[1].set_title(f"injection: Lambda = {Lambda_inj:.1f}")
plt.tight_layout()
print(elapsed(), "done")
""")

md(r"""
## 10. Producing the paper's results

1. Switch every input above to its `# PAPER:` value (1 yr data, 48 durations, 1024 → 512 members, 525 iterations,
   $f_0 \in [5\times10^{-4}, 10^{-2}]$ Hz, `N_NOISE = 100`); on a 64 GB laptop `BATCH_SIZE = 256` or `None`.
2. Background: run Sec. 9 once (100 noise realisations). Keep `mu_k`, `sigma_k` and the Gumbel fit.
3. Injections: for each SNR in 20, 25, 30, 35, 40 and 100 draws of the Table 1 priors (Sec. 1c, `SOURCE_SEED` = draw index),
   run Secs. 2–8 and record $\Lambda$; the detection probability is the fraction above $\Lambda_{\rm th}(p_{\rm FA})$.
4. Budget: one full search $\approx$ 18 h on an M3 Pro (measured per-iteration cost, `BATCH_SIZE = 128`); scale by your machine's
   measured iteration time. The full set (600 searches) is a cluster job.

Not included: the paper's follow-up (simulation-based inference, scipy DE on the physical parameters, eryn MCMC).
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (lat-emri)", "language": "python"}
OUT = "emri_search_lorenzo.ipynb"
if __name__ == "__main__":
    import sys

    settings = dict(arg.split("=") for arg in sys.argv[1:])
    text = nbf.writes(nb)
    for key, default in (("N_INITIAL", "1024"), ("N_POP", "128"), ("N_STEPS", "300")):
        text = text.replace(f"__{key}__", settings.get(key, default))
    open(OUT, "w").write(text)
    print("wrote", OUT)
