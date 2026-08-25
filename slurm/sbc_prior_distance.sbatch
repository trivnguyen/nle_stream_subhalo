#!/bin/bash
# sbc_prior_distance.sbatch — posterior-to-prior distances for the
# tempered SBC runs, cached for nle_sbc.ipynb to plot.
#
#   sbatch slurm/sbc_prior_distance.sbatch [base_dir] [out_npz]
#   sbatch slurm/sbc_prior_distance.sbatch /scratch/tvnguyen/nle_sbc
#
# A pure post-processing pass over npz files that already exist -- no
# simulator, no model, no sampling. At the current 7 temperatures x 500
# datasets it is ~100 s wall and ~400 CPU-s, which a login node runs
# fine inside its 3600 CPU-s limit, so this wrapper is a convenience for
# when the dataset count grows, not a requirement.
#
# CPUS is deliberately small. The heavy call is one batched k-NN query
# per temperature; the rest is 999-row queries where a 192-thread pool
# costs more than it saves. Trillium hands out whole CPU nodes anyway,
# so the extra 184 cores sit idle either way -- asking for them would
# only make the job slower. See --workers in sbc_prior_distance.py.

#SBATCH --job-name=nle_sbc_prior_dist
#SBATCH --account=def-tingli
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=tri.vt.nguyen@gmail.com
#SBATCH --chdir=/scratch/tvnguyen
#SBATCH --output=/scratch/tvnguyen/slurm_logs/nle/sbc/%x_%j.out
#SBATCH --error=/scratch/tvnguyen/slurm_logs/nle/sbc/%x_%j.err

set -uo pipefail

NLE_DIR="${NLE_DIR:-/home/tvnguyen/projects/sbi_stream/nle_stream_subhalo}"
BASE_DIR="${1:-/scratch/tvnguyen/nle_sbc}"
OUT_NPZ="${2:-${BASE_DIR}/prior_distance.npz}"
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

export OMP_NUM_THREADS="${WORKERS}"
export MKL_NUM_THREADS="${WORKERS}"

# $HOME is read-only on compute nodes. `stream_sims/__init__` pulls in
# sims -> Nbody_streams, whose numba kernels are @njit(cache=True) and
# will refuse to import if they cannot write a cache; numba's user-wide
# cache locator follows XDG_CACHE_HOME.
export XDG_CACHE_HOME="/scratch/tvnguyen/cache"
export MPLCONFIGDIR="/scratch/tvnguyen/cache/matplotlib"
export TORCH_HOME="/scratch/tvnguyen/cache/torch"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TORCH_HOME"

mkdir -p /scratch/tvnguyen/slurm_logs/nle/sbc

echo "host      : $(hostname)"
echo "base dir  : ${BASE_DIR}"
echo "out       : ${OUT_NPZ}"
echo "workers   : ${WORKERS}"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/torch}"
[[ -s "$VENV_DIR/.modules" ]] && module load $(cat "$VENV_DIR/.modules") 2>/dev/null
source "$VENV_DIR/bin/activate"

python "${NLE_DIR}/sbc_prior_distance.py" \
    --base-dir "${BASE_DIR}" \
    --out "${OUT_NPZ}" \
    --workers "${WORKERS}"
