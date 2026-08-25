#!/bin/bash
# train_nle.sbatch — SLURM job script for train_nle.py.
#
# Normally launched via ./submit.sh <config.py>; can also run directly:
#   sbatch train_nle.sbatch configs/nsf_9params.py
#
# $HOME is read-only on compute nodes here, so all writes (logs, manifest,
# caches) go under /scratch/tvnguyen. No --mem requested: it's a no-op on
# this cluster's GPU partitions (DefMemPerNode == MaxMemPerNode). Single
# GPU only — train_nle.py's Trainer has no devices/strategy set, so more
# than one GPU would trigger unintended DDP.

#SBATCH --job-name=nle_train
#SBATCH --account=def-tingli
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tri.vt.nguyen@gmail.com
#SBATCH --chdir=/scratch/tvnguyen
#SBATCH --output=/scratch/tvnguyen/slurm_logs/nle/manual/%x_%j.out
#SBATCH --error=/scratch/tvnguyen/slurm_logs/nle/manual/%x_%j.err

set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch train_nle.sbatch <path/to/config.py> [--config.field=value ...]" >&2
    exit 1
fi

CONFIG="$1"
shift
EXTRA_ARGS=("$@")

# Not self-located via $BASH_SOURCE: SLURM runs a copy of this script from
# a spool dir, so that would resolve to the wrong directory. submit.sh
# exports NLE_DIR correctly; this fallback is only for direct `sbatch` use.
NLE_DIR="${NLE_DIR:-/home/tvnguyen/projects/sbi_stream/nle_stream_subhalo}"
RUN_DIR="${RUN_DIR:-/scratch/tvnguyen/slurm_logs/nle/manual/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
MANIFEST="$RUN_DIR/manifest.txt"
mkdir -p "$RUN_DIR"

# Resolve config relative to nle_stream/ if it isn't already absolute/relative-found.
# A trailing ":variant" is ml_collections' argument to get_config(), not part
# of the path, so it is split off for every filesystem test and reattached.
CONFIG_PATH="${CONFIG%%:*}"
CONFIG_VARIANT=""
[[ "$CONFIG" == *:* ]] && CONFIG_VARIANT=":${CONFIG#*:}"
if [[ ! -f "$CONFIG_PATH" && -f "$NLE_DIR/$CONFIG_PATH" ]]; then
    CONFIG_PATH="$NLE_DIR/$CONFIG_PATH"
fi
CONFIG="${CONFIG_PATH}${CONFIG_VARIANT}"

# Redirect caches that default to $HOME (read-only on compute nodes).
export XDG_CACHE_HOME="/scratch/tvnguyen/cache"
export MPLCONFIGDIR="/scratch/tvnguyen/cache/matplotlib"
export WANDB_CONFIG_DIR="/scratch/tvnguyen/cache/wandb_config"
export TORCH_HOME="/scratch/tvnguyen/cache/torch"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$WANDB_CONFIG_DIR" "$TORCH_HOME"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MANIFEST"; }

log "===== job start ====="
log "job_id:       $SLURM_JOB_ID"
log "job_name:     $SLURM_JOB_NAME"
log "node:         $(hostname)"
log "partition:    ${SLURM_JOB_PARTITION:-unknown}"
log "gpus:         $(nvidia-smi -L 2>/dev/null | tr '\n' ';' )"
log "cpus:         ${SLURM_CPUS_PER_TASK:-unknown}"
log "mem:          ${SLURM_MEM_PER_NODE:-unknown} MB"
log "config:       $CONFIG"
log "extra_args:   ${EXTRA_ARGS[*]:-<none>}"
log "sims commit:  $(git -C "$NLE_DIR/../sbi_stream_pipeline" rev-parse --short HEAD 2>/dev/null || echo unknown)"

cp -- "$CONFIG_PATH" "$RUN_DIR/config_snapshot.py" 2>/dev/null || true

# The stream project's venv (torch-jax), not `torch` — see
# sbi_stream_pipeline/README.md. Activated directly rather than via
# ~/.bashrc's venv-activate: ~/.bashrc's non-interactive guard skips its
# own function definitions under a non-interactive sbatch shell, so
# venv-activate wouldn't exist.
VENV_DIR="${VENV_DIR:-$HOME/.venvs/torch-jax}"
set +u  # Lmod's `module load` internals aren't -u-safe.
if [[ -s "$VENV_DIR/.modules" ]]; then
    module load $(cat "$VENV_DIR/.modules") 2>/dev/null
fi
source "$VENV_DIR/bin/activate"
set -u
log "python:       $(which python3) ($(python3 --version 2>&1))"
log "torch:        $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"

on_exit() {
    local code=$?
    log "exit_code:    $code"
    log "===== job end ====="
    exit "$code"
}
trap on_exit EXIT
trap 'log "received TERM (likely time limit approaching)"' TERM

cd "$NLE_DIR"
python3 train_nle.py --config "$CONFIG" "${EXTRA_ARGS[@]}" --config.enable_progress_bar=False 2>&1 | tee -a "$RUN_DIR/stdout.log"
exit "${PIPESTATUS[0]}"
