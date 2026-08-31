#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_PY=/home/esrf/matteo1996a/.conda/envs/jax-gpu/bin/python
ENV_NAME=jax-pynx-env
ENV_PREFIX="/data/projects/id01ml/Virtual_envs/${ENV_NAME}"

# ── 1. Create a plain venv, no conda involved at all ──
"$BOOTSTRAP_PY" -m venv "$ENV_PREFIX"

PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"
"$PIP" install --upgrade pip

# ── 2. Base scientific stack (PyPI wheels, no conda-forge needed) ──
"$PIP" install --no-cache-dir \
    scipy pooch matplotlib scikit-image scikit-learn \
    ipython ipykernel h5py hdf5plugin silx fabio \
    psutil mako numexpr cython pytest "pytools<=2024.1.3"

# ── 3. JAX stack, pinned exactly to what's validated in jax-gpu ──
"$PIP" install --no-cache-dir \
    "jax==0.6.2" "jaxlib==0.6.2" \
    "jax-cuda12-pjrt==0.6.2" "jax-cuda12-plugin==0.6.2" \
    "optax==0.2.6" "numpy==2.2.6" "chex==0.1.90" \
    equinox jaxopt jaxtyping lineax optimistix

# ── 4. PyNX GPU backend — after jax, --no-cache-dir to avoid stale wheels ──
"$PIP" install --no-cache-dir pyopencl pycuda pyvkfft

# ── 5. PyNX itself — devel branch (numpy>2 compatible) ──
"$PIP" install --no-cache-dir \
    https://gitlab.esrf.fr/favre/PyNX/-/archive/devel/PyNX-devel.tar.gz

# ── 6. cdiutils ──
"$PIP" install --no-cache-dir cdiutils

# ── 7. convex_ad_jax from GitHub (editable) ──
git clone https://github.com/matteomasto/convex-ad.git "${ENV_PREFIX}/src/convex-ad"
"$PIP" install --no-cache-dir -e "${ENV_PREFIX}/src/convex-ad"   # adjust path once you confirm the subfolder

# ── 8. Register as a Jupyter kernel ──
"$PY" -m ipykernel install --user \
    --name="${ENV_NAME}" --display-name "Python (JAX + PyNX)"

# ── 9. Sanity checks ──
echo "--- jax ---";      "$PY" -c "import jax; print(jax.__version__); print(jax.devices())"
echo "--- pynx ---";     "$ENV_PREFIX/bin/pynx-info"
echo "--- pyvkfft ---";  "$ENV_PREFIX/bin/pyvkfft-info"
echo "--- cdiutils ---"; "$PY" -c "import cdiutils; print(cdiutils.__version__)"
echo "--- numpy ---";    "$PY" -c "import numpy; print(numpy.__version__)"