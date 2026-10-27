# EMRI search on a Mac (Apple Silicon)

The modular semi-coherent EMRI search (`lisatools.search.emri`) and its walkthrough notebook
`emri_search_lorenzo.ipynb` (Lorenzo Speri's search, arXiv:2510.20891, plus Cutler &
Bustamante-Rosell's `H` statistic). Everything runs on the CPU with JAX.

| File | What it is |
|---|---|
| `setup_env_macos.sh` | Builds the `lat-emri` conda environment (dev stack + search extras) |
| `emri_search_lorenzo.ipynb` | The notebook, committed with the toy run's outputs |
| `build_emri_search_notebook.py` | Regenerates the notebook (unexecuted): `python build_emri_search_notebook.py` |

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
  each one. `STATISTIC` picks `"fresnel"` (Lorenzo) or `"H"`.

Long runs (e.g. the Sec. 10 statistic comparison), headless and with the Mac kept awake:

```sh
caffeinate -i jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 emri_search_lorenzo.ipynb > run.log 2>&1 &
```

Sec. 10 saves each statistic's results to `results_*/compare_<statistic>.npz`; with
`COMPARE_RESUME = True` a rerun skips the finished ones.

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
