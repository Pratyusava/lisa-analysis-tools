# EMRI search on a Mac (Apple Silicon)

The modular semi-coherent EMRI search (`lisatools.search.emri`) and its walkthrough notebook
`emri_search_lorenzo.ipynb` (Lorenzo Speri's search, arXiv:2510.20891, plus Cutler &
Bustamante-Rosell's `H` statistic and Chebyshev track model). Everything runs on the CPU with JAX.

| File | What it is |
|---|---|
| `setup_env_macos.sh` | Builds the `lat-emri` conda environment (dev stack + search extras) |
| `emri_search_lorenzo.ipynb` | The notebook, committed with the toy run's outputs |
| `build_emri_search_notebook.py` | The notebook's source: regenerates it, keeping the outputs of unchanged cells (see *Editing the notebook*) |
| `emri_search.py` | Runs a campaign described by one TOML file: search injection / data files, the noise background, FARs (see *Running a campaign*) |
| `configs/` | Campaign files: `S_svd.toml` (Speri et al.'s search), `H_chebyshev.toml` (Cutler & Bustamante-Rosell's `H` on a built Chebyshev basis), `reference.toml` (every setting, both statistics and bases) |
| `make_injections.py` | Injection files for a campaign (FEW sources from the paper's prior), as recipes with checksums |
| `make_noise.py` | Noise files for a campaign's background, as seeds with checksums |
| `basis_coverage.py` | How much of the source population each track basis can represent, with no search (basis mismatch per source and variant; ~10 GB per worker, FEW's generator) |

## Setup

**1. Prerequisites**

```sh
xcode-select --install                     # Apple clang + git

curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh    # accept the defaults, "yes" to initialise conda
exec zsh                                   # reload the shell so `conda` is on PATH
conda --version

ssh -T git@github.com                      # GitHub SSH access
```

(`brew install --cask miniforge` and then `conda init zsh` also works. An Intel/Rosetta
Anaconda works too: the script forces a native arm64 environment.)

**2. Clone this repo and EMRI-Search side by side** (the notebook reads Lorenzo's PSD and SVD
basis from `../EMRI-Search`):

```sh
mkdir -p ~/Documents/Research/LISA && cd ~/Documents/Research/LISA
git clone git@github.com:Pratyusava/lisa-analysis-tools.git
git clone git@github.com:lorenzsp/EMRI-Search.git
cd lisa-analysis-tools && git checkout dev
```

**3. Build the environment** (~15–30 min, mostly compiling):

```sh
bash scripts/emri_search/setup_env_macos.sh        # env name lat-emri; ENV_NAME=... to change
```

It creates an arm64 Python 3.12 env with the compilers (cmake, ninja, LAPACKE, gfortran), runs
`install.sh` (clones and builds GPUBackendTools, Eryn and FastEMRIWaveforms next to this repo),
installs JAX, EMRI-Search and the `make check` tools, registers the env as the Jupyter kernel
*Python (lat-emri)*, and re-signs any `.so` whose code signature is invalid (otherwise Python is
killed with "Killed: 9" / exit 137 and no traceback on Apple Silicon). It ends by printing
`backends: ['cpu', 'jax']`.

**4. Verify**

```sh
conda activate lat-emri
make check          # ~1.5 min; ends with "All checks passed!"
```

The nanobind "leaked ..." lines at exit and the lisaconstants/astropy version warning are
harmless. `Killed: 9` means a module with an invalid signature; re-sign it (step 6 of the
setup script does this; re-run the script or its `codesign` loop).

## Running the notebook

```sh
cd scripts/emri_search
jupyter lab emri_search_lorenzo.ipynb
```

- Pick the kernel **Python (lat-emri)** (Kernel → Change Kernel). The setup script registers it;
  by hand: `conda activate lat-emri && python -m ipykernel install --user --name lat-emri --display-name "Python (lat-emri)"`.
- VS Code works too: install the Python and Jupyter extensions, open the repo folder, open the
  notebook and choose *Select Kernel → Jupyter Kernel → Python (lat-emri)* (and *Python: Select
  Interpreter → lat-emri* for scripts). Use the headless command below for long runs: closing the
  editor or sleeping the Mac stops a notebook running in it.
- EMRI-Search elsewhere: `EMRI_SEARCH_DIR=/path/to/EMRI-Search jupyter lab ...` (or `... code .`).
- Results go to `results_toy/` (or `results_paper/`) next to the notebook; git ignores them.
- Inputs are in Sec. 1: the toy values are active, the paper's (`# PAPER:`), Lorenzo's repo's
  (`# REPO:`) and the Cutler & Bustamante-Rosell draft's (`# CB-R draft:`) are commented next to
  each one. `STATISTIC` picks `"fresnel"` (Lorenzo) or `"H"`; `BASIS` picks `"svd"` (Lorenzo's SVD basis of FEW
  tracks) or `"chebyshev"` (Cutler & Bustamante-Rosell's phase model); `BASIS_SOURCE` picks `"file"` (Lorenzo's
  basis, or an `.npz` saved by a build) or `"build"` (from a FEW track bank with the Sec. 1d inputs, saved to
  `BASIS_SAVE`).

Long runs (e.g. the Sec. 10 statistic comparison), headless and with the Mac kept awake:

```sh
caffeinate -i jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 emri_search_lorenzo.ipynb > run.log 2>&1 &
```

Sec. 10 saves each statistic's results to `results_*/compare_<scale>_<statistic>.npz`; with
`COMPARE_RESUME = True` a rerun skips the finished ones and continues a partial one.

## Running a campaign

`emri_search.py` runs the notebook's search on many streams, in parallel processes, from one
campaign file (TOML); the library behind it is `lisatools.search.emri` (`RunConfig`,
`SearchConfig`). The file lists:

| Table | What it sets |
|---|---|
| top level | `out` (the campaign directory), `n_workers` |
| `[inputs]` | `injections` and `noise` (recipe files or globs, see below), `data` (files searched as is) |
| `[data]` | cadence, segments, channels, `psd_file` |
| `[statistic]` | `name = "fresnel"` (Speri et al.'s S) or `"H"` (Cutler & Bustamante-Rosell); `[statistic.fresnel]`, `[statistic.H]` |
| `[basis]` | `name = "svd"` or `"chebyshev"`; `source = "file"` (load `file`) or `"build"` (from the FEW bank in `[basis.bank]`, saved to `<out>/basis.npz` and reused); `[basis.svd]`, `[basis.chebyshev]` |
| `[durations]`, `[optimizer]` | the analysed stream lengths, the initial screening and Adam + DE |

`configs/reference.toml` lists every setting, of both statistics and both bases, with its value
from the paper (`PAPER`), Lorenzo's repository (`REPO`) or the Cutler & Bustamante-Rosell draft
(`draft`); these are also the defaults. The run configs list only the tables they use:
`S_svd.toml` has `[statistic.fresnel]` and `[basis.svd]`, `H_chebyshev.toml` has
`[statistic.H]`, `[basis.chebyshev]` and the `[basis.bank]` its build draws; data, durations
and optimiser are the same in both, so the two campaigns differ only in the statistic and the
basis. An omitted key takes its default. Paths may use `$EMRI_SEARCH_DIR` and `~` and are
relative to the file; `"none"` switches off an optional setting. Only the chosen statistic's
and basis's settings identify a campaign.

**Injection and noise files are recipes, not samples** (a few kB of JSON each, instead of 50 MB
per channel for a year):

- `make_injections.py` writes one file per injection: the source parameters (with the distance
  that gives the target SNR and the initial conditions, so the signal is regenerated directly
  with FEW), the seed of the noise it is searched on (`--noise-seed + k`), and checksums of
  both. It draws from the paper's Table 1 prior and keeps sources starting in `--f0-range`
  (default the paper's 0.5-10 mHz); from the campaign file it takes only the cadence, channels,
  PSD, durations and basis grid, so the plunges fall where the search looks.
- `make_noise.py` writes the background: one file per noise-only stream, its seed (`--seed + r`)
  and a checksum. The paper used 100 streams; a FAR resolves down to `1 / n`.

Each search regenerates the data and checks it against the file's checksum (each channel's power
and a few hundred samples, to 1e-10) and against the campaign's cadence, channels and PSD. A
NumPy, LAT or FEW version that produces different data therefore stops the search with an error
naming the versions the file was made with, instead of searching different data. The search also
refuses an injection whose noise seed is one of the background's, and a result whose stream file
was remade with another noise seed. Searching injections needs FEW.

```sh
conda activate lat-emri
cd scripts/emri_search
export EMRI_SEARCH_DIR=~/Documents/Research/LISA/EMRI-Search   # Lorenzo's basis and PSD files

# 1. the inputs: 100 injections at SNR 20 (FEW) and 100 noise-only streams, one year each
python make_injections.py configs/S_svd.toml --out inj --n 100 --snrs 20 --T-data 1.0
python make_noise.py configs/S_svd.toml --out noise --n 100 --T-data 1.0

# 2. the campaign: search the injections, then the background, then the summary
python emri_search.py configs/S_svd.toml
python emri_search.py configs/H_chebyshev.toml         # the other statistic and basis, same injections

# stages one at a time, a setting changed on the command line, the resolved configuration
python emri_search.py configs/S_svd.toml search background collect
python emri_search.py configs/S_svd.toml --set statistic.name=H --set out=../runs/H_svd
python emri_search.py configs/S_svd.toml --show
```

The campaign directory holds `config.json` (the resolved search configuration), `fingerprint`,
`results/` (one `.npz` per stream and duration) and `summary.csv` / `summary.npz`. The summary
follows Speri et al. (Sec. III.1, the notebook's Sec. 9): the best statistic `X_k` of each
duration is standardised by the noise-only searches at that duration, `Z_k = (X_k - mu_k) /
sigma_k`, and a stream's detection statistic is `Lambda = max_k Z_k` (raw values are not compared
across durations: they grow with the stream's length). Each row has `Lambda` (`statistic`), the
duration it peaks at and the raw `X` there; `far`, the fraction of noise streams whose `Lambda`
is at least as large (it resolves down to one over the number of noise files); `far_gumbel`,
from a Gumbel fit to the noise `Lambda` (the paper's: location 1.94, scale 0.56);
`detected_<p>` at each false-alarm probability of `p_fa` (paper: 0.5, 1e-2, 1e-4); and how well
the best track follows the injected one. The standardisation needs at least 2 noise streams and
the Gumbel fit 3.

The track's quality is its **mismatch** (`lisatools.search.emri.mismatch`): the semi-coherent
power the best track collects from the noiseless injected signal, against the power of the true
track, `mismatch = 1 - P(found) / P(true)`, on one ruler for every statistic and basis. The true
track is FEW's `m Phi_phi` harmonic closest to the found track (`harmonic`, m = 2, 3, 4: in TDI the
m = 3 harmonic often carries a third of the power, and a single-harmonic search may lock onto it),
and every power is taken over the longest searched stream, so a track found at too short a
duration is charged for what it misses. It splits as
`1 - mismatch = (1 - mismatch_basis) (1 - mismatch_search)`: `mismatch_basis` is the loss of the
best track the basis can represent (fits of the true track inside the basis box at the scan
durations bracketing the plunge; `in_basis_box` says whether the unbounded fit stayed in the
box), and `mismatch_search` the loss of the found track against that one (negative when the search
beat the fit, which is an upper bound on the basis loss). `mismatch_x<k>` uses coherent segments of
`k` SFT segments (`coherence`, default 1, 2, 4), the question a coherent follow-up asks.
`collect` regenerates each injection's signal with FEW once and caches the columns in
`<out>/mismatch/`; `agreement` (the fraction of segments within `tol_bins` bins) is kept as a
quick check. A directory accepts only one configuration, and reruns skip the searches whose
result file exists, so an interrupted campaign continues where it stopped. A data file is an
`.npz` with `signal` of shape `(n_channels, N)` and `dt`, listed in `inputs.data`.

## Editing the notebook

`build_emri_search_notebook.py` is the source of truth: edit it, then run
`python build_emri_search_notebook.py` in this folder. The existing notebook's executed outputs are
carried over to every cell whose code is unchanged; the cells whose code changed are listed and must be
re-executed: open the notebook, run them, save; or `python build_emri_search_notebook.py --run=2,4-18,30`
executes the listed cells (indices as printed) in one fresh kernel right after regenerating and saves their
outputs (`--kernel=python3` runs them in the current environment's kernel instead of the notebook's).
`--fresh` writes an unexecuted notebook. Edits made to the `.ipynb` itself are lost at the next regeneration,
so put them in the generator; running cells in Jupyter and saving is fine, the outputs are picked up.

**Memory (`BATCH_SIZE`, 64 GB machine):** toy runs `None`; paper settings with `"fresnel"`
`256` (or `None`); paper settings with `"H"` start at `32` (its FFT has ~6×10⁵ bins for 1 yr).
JAX's Apple GPU backend has no complex128, so the search runs on the CPU cores.

## Reproducibility

`install.sh` pulls the latest heads of the sibling repos. The notebook's outputs were produced
with:

| Repo | Branch @ commit |
|---|---|
| GPUBackendTools | `spline@a63f363` |
| FastEMRIWaveforms | `gpu_backend@e7c550d4` |
| Eryn | `dev@d46b5aa` |
| EMRI-Search | `main@0f4e834` |

If results differ, check these out (`git -C ../<repo> checkout <commit>`) and reinstall.

**Why not plain `pip install`?** The PyPI releases predate this code (LAT 1.2.8 has no
`lisatools.search.emri`), and every package compiles C++ locally, which needs compilers and
LAPACKE that pip cannot install; conda only supplies those, and `install.sh` itself is a chain of
`pip install -e`.
