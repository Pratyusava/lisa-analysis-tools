#!/usr/bin/env bash
# Laptop (macOS arm64, CPU-only) environment for the modular EMRI search
# (lisatools.search.emri). Reproduces the env the search was developed in.
#
#   bash scripts/emri_search/setup_env_macos.sh
#
# Clones the dev-stack siblings (GPUBackendTools, Eryn, FastEMRIWaveforms, ...)
# next to this repo, exactly like ./install.sh. BBHx / GBGPU / phentax are not
# needed by the EMRI search and are skipped.
# No `set -u`: conda-forge's compiler (cctools) activate/deactivate scripts read
# unset variables (CONDA_BACKUP_AR, ...), which nounset turns into a fatal error.
set -eo pipefail

ENV_NAME="${ENV_NAME:-lat-emri}"
LAT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. Python 3.12 + the native build toolchain LAT/GBT/FEW need on macOS
#    (cmake/ninja for scikit-build-core, LAPACKE found through pkg-config,
#    gfortran because GBT's CMake project() enables Fortran).
#    CONDA_SUBDIR forces a native arm64 env even from an Intel (Rosetta)
#    Anaconda install: x86_64 has no numba / recent jaxlib wheels for macOS.
#    Re-runnable: an existing env is updated in place instead of recreated.
TOOLCHAIN=(python=3.12 cmake ninja pkg-config liblapacke lapack openblas gfortran)
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    CONDA_SUBDIR=osx-arm64 conda install -n "$ENV_NAME" -c conda-forge -y "${TOOLCHAIN[@]}"
else
    CONDA_SUBDIR=osx-arm64 conda create -n "$ENV_NAME" -c conda-forge -y "${TOOLCHAIN[@]}"
fi
conda activate "$ENV_NAME"
conda config --env --set subdir osx-arm64
python -c "import platform; assert platform.machine() == 'arm64', platform.machine()"

# 2. The LAT dev stack via the canonical installer, with Apple clang.
export CC=/usr/bin/clang CXX=/usr/bin/clang++ FC="$CONDA_PREFIX/bin/gfortran"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig"
SKIP_BBHX=1 SKIP_GBGPU=1 SKIP_PHENTAX=1 "$LAT_DIR/install.sh"

# 3. Search extras: JAX, plus Lorenzo Speri's reference package for the parity
#    tests (no deps: it pins lisaanalysistools==1.2.0 and the retired
#    fastlisaresponse; only its jax_utils, which needs interpax, is imported).
#    optax: the optimiser-parity test runs his notebook's Adam + DE loop.
pip install jax interpax optax
pip install --no-deps -e "$(dirname "$LAT_DIR")/EMRI-Search"

# 4. Validation tools for `make check` (format -> lint -> typecheck -> test).
pip install black isort flake8 mypy

# 5. Register the env as a Jupyter kernel, so JupyterLab / VS Code started
#    from any env (e.g. Miniforge's base) can run the notebook on it.
python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME)"

# 6. Apple Silicon kills (SIGKILL, "Killed: 9" / exit 137, no traceback) any
#    process that loads a binary whose code signature was invalidated after
#    linking (GBT's interp.so comes out that way). Ad-hoc re-sign every .so the
#    env can load whose signature is invalid: the env itself and the sibling
#    source trees (editable builds can leave modules there). Universal wheels
#    whose x86_64 slice is "not signed at all" load fine and are left alone.
find "$CONDA_PREFIX" "$(dirname "$LAT_DIR")" -name '*.so' -print0 2>/dev/null |
    while IFS= read -r -d '' so; do
        if codesign -v "$so" 2>&1 | grep -q "invalid"; then
            echo "re-signing $so"
            codesign --force --sign - "$so"
        fi
    done

python -W ignore -c "import lisatools, few, jax; print('backends:', [b for b in ('cpu', 'jax') if lisatools.has_backend(b)])"
