#!/bin/bash
# sbc_calibration.sbatch — job array for sbc_calibration.py.
#
#   sbatch --array=0-24 slurm/sbc_calibration.sbatch \
#       8 250 /scratch/tvnguyen/nle_sbc/aau_250stars
#
# Each array task samples PER_TASK datasets in lockstep on one node and
# writes one npz to $OUT_DIR. When the whole array is done, on a login
# node:
#
#   python sbc_calibration.py combine --output-dir <OUT_DIR>
#
# CPU job, for Trillium's CPU cluster (tri-login01): partitions there
# report GRES=(null) and 192 cores per node. `sbc_calibration.py` picks
# its device from `torch.cuda.is_available()`, so an allocation without a
# GPU lands on CPU by itself -- nothing else needs changing.
#
# Trillium is two clusters -- CPU (tri-login01) and GPU (trig-login01) --
# sharing $HOME and /scratch but not schedulers, and one shell session can
# be on either. CHECK `hostname` BEFORE SUBMITTING: job IDs, partitions and
# the GPU requirement all differ, and `squeue` on one cluster says nothing
# about jobs on the other.
#
# On the GPU cluster every node carries h100s and the scheduler REJECTS a
# submission with no --gpus-per-node, so to run there instead, override at
# submit time:
#
#   sbatch --gpus-per-node=1 --cpus-per-task=24 --time=01:00:00 ...
#
# Measured throughput, one dataset-task of 7.7e8 rows at 250 stars:
#   H100                 942,998 rows/s   ~14 min
#   CPU, 4 threads        46,912 rows/s
#   CPU, 16 threads       80,159 rows/s   (scaling is strongly sublinear)
# A full 192-core node should land near 150-250k rows/s, i.e. 1-2 h per
# task; --time below is set for the pessimistic 80k rows/s case.
#
# Sizing: measure it rather than trusting an extrapolation -- one probe on
# the debug partition (--n-datasets 2 --n-steps 500) prints a rows/s, and
# a task's total is PER_TASK * n_walkers * n_steps * n_stars rows. At the
# defaults that is 8 * 64 * 6000 * 250 = 7.7e8 rows per task. Cost is
# linear in --n-stars; raise --time to match if you raise it.
#
# $HOME is read-only on compute nodes, so everything is written under
# /scratch/tvnguyen. No --mem requested: it's a no-op on this cluster's
# partitions (DefMemPerNode == MaxMemPerNode).

#SBATCH --job-name=nle_sbc
#SBATCH --account=def-tingli
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tri.vt.nguyen@gmail.com
#SBATCH --chdir=/scratch/tvnguyen
#SBATCH --output=/scratch/tvnguyen/slurm_logs/nle/sbc/%x_%A_%a.out
#SBATCH --error=/scratch/tvnguyen/slurm_logs/nle/sbc/%x_%A_%a.err

set -uo pipefail

# Config comes from POSITIONAL arguments, not environment variables. This
# site's sbatch is a wrapper that injects `--export=NONE --get-user-env`,
# so `TAG=x sbatch ...` silently loses TAG and the job runs with the
# defaults below -- which is exactly what a probe on 2026-08-06 did:
# asked for 2 datasets, ran 8, wrote to the wrong directory. Arguments
# after the third are forwarded to python, and that path does work.
#
#   sbatch --array=0-24 slurm/sbc_calibration.sbatch \
#       <per_task> <n_stars> <out_dir> [--n-steps N ...]
NLE_DIR="${NLE_DIR:-/home/tvnguyen/projects/sbi_stream/nle_stream_subhalo}"
PER_TASK="${1:-8}"
N_STARS="${2:-250}"
OUT_DIR="${3:-/scratch/tvnguyen/nle_sbc/default}"
[[ $# -gt 0 ]] && shift $(( $# < 3 ? $# : 3 ))
DELTA_VMIN="${DELTA_VMIN:-0.03}"
CONFIG="${CONFIG:-$NLE_DIR/configs/nsf_9params_200ds_track.py}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
START=$(( TASK_ID * PER_TASK ))

mkdir -p "$OUT_DIR" /scratch/tvnguyen/slurm_logs/nle/sbc

# Caches that default to $HOME, which is read-only on compute nodes.
export XDG_CACHE_HOME="/scratch/tvnguyen/cache"
export MPLCONFIGDIR="/scratch/tvnguyen/cache/matplotlib"
export TORCH_HOME="/scratch/tvnguyen/cache/torch"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TORCH_HOME"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "===== task $TASK_ID start ====="
log "node:       $(hostname)"
log "cpus:       ${SLURM_CPUS_PER_TASK:-unknown}"
log "gpus:       $(nvidia-smi -L 2>/dev/null | tr '\n' ';' || echo none)"
log "config:     $CONFIG"
log "datasets:   [$START, $(( START + PER_TASK )))"
log "n_stars:    $N_STARS"
log "delta_vmin: $DELTA_VMIN"
log "out_dir:    $OUT_DIR"

# The `torch` venv, as in run_inference_nb.sbatch -- NOT the `torch-jax`
# one train_nle.sbatch uses. torch-jax is the training environment and has
# no pocomc, nautilus, sklearn or pqm; `torch` carries the full inference
# stack. Activated directly: ~/.bashrc's non-interactive guard skips its
# own function definitions under sbatch, so `venv-activate` would not
# exist.
VENV_DIR="${VENV_DIR:-$HOME/.venvs/torch}"
set +u  # Lmod's `module load` internals aren't -u-safe.
if [[ -s "$VENV_DIR/.modules" ]]; then
    module load $(cat "$VENV_DIR/.modules") 2>/dev/null
fi
source "$VENV_DIR/bin/activate"
set -u
log "python:     $(which python3) ($(python3 --version 2>&1))"

cd "$NLE_DIR"
python3 sbc_calibration.py run \
    --config "$CONFIG" \
    --output-dir "$OUT_DIR" \
    --start "$START" \
    --n-datasets "$PER_TASK" \
    --n-stars "$N_STARS" \
    --delta-vmin "$DELTA_VMIN" \
    --threads "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"
RC=$?

log "===== task $TASK_ID end (exit $RC) ====="
exit "$RC"
