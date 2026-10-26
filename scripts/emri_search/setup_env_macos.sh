#!/usr/bin/env bash
# Laptop (macOS arm64, CPU-only) environment for the modular EMRI search
# (lisatools.search.emri). Reproduces the env the search was developed in.
#
#   bash scripts/emri_search/setup_env_macos.sh
#
# Clones the dev-stack siblings (GPUBackendTools, Eryn, FastEMRIWaveforms, ...)
# next to this repo, exactly like ./install.sh. BBHx / GBGPU / phentax are not
# needed by the EMRI search and are skipped.
set -euo pipefail

ENV_NAME="${ENV_NAME:-lat-emri}"
LAT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

source "$(conda info --base)/etc/profile.d/conda.sh"

# 1. Python 3.12 + the native build toolchain LAT/GBT/FEW need on macOS
#    (cmake/ninja for scikit-build-core, LAPACKE found through pkg-config,
#    gfortran because GBT's CMake project() enables Fortran).
#    CONDA_SUBDIR forces a native arm64 env even from an Intel (Rosetta)
#    Anaconda install: x86_64 has no numba / recent jaxlib wheels for macOS.
CONDA_SUBDIR=osx-arm64 conda create -n "$ENV_NAME" -c conda-forge -y \
    python=3.12 cmake ninja pkg-config liblapacke lapack openblas gfortran
conda activate "$ENV_NAME"
conda config --env --set subdir osx-arm64
python -c "import platform; assert platform.machine() == 'arm64', platform.machine()"

# 2. The LAT dev stack via the canonical installer, with Apple clang.
export CC=/usr/bin/clang CXX=/usr/bin/clang++ FC="$CONDA_PREFIX/bin/gfortran"
export PKG_CONFIG_PATH="$CONDA_PREFIX/lib/pkgconfig"
SKIP_BBHX=1 SKIP_GBGPU=1 SKIP_PHENTAX=1 "$LAT_DIR/install.sh"

# 3. Apple Silicon kills (SIGKILL, exit 137, no traceback) any process that
#    loads a binary whose signature was invalidated after linking. GBT's
#    installed interp.so comes out that way; ad-hoc re-sign whatever is broken.
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
find "$SITE" -name '*.so' -print0 | while IFS= read -r -d '' so; do
    if codesign -v "$so" 2>&1 | grep -q "invalid signature"; then
        echo "re-signing $so"
        codesign --force --sign - "$so"
    fi
done

# 4. Search extras: JAX, plus Lorenzo Speri's reference package for the parity
#    tests (no deps: it pins lisaanalysistools==1.2.0 and the retired
#    fastlisaresponse; only its jax_utils, which needs interpax, is imported).
#    optax: the optimiser-parity test runs his notebook's Adam + DE loop.
pip install jax interpax optax
pip install --no-deps -e "$(dirname "$LAT_DIR")/EMRI-Search"

# 5. Validation tools for `make check` (format -> lint -> typecheck -> test).
pip install black isort flake8 mypy

python -W ignore -c "import lisatools, few, jax; print('backends:', [b for b in ('cpu', 'jax') if lisatools.has_backend(b)])"
