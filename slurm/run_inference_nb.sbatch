#!/bin/bash
# run_inference_nb.sbatch — execute the inference notebooks on a GPU node.
#
#   sbatch run_inference_nb.sbatch [notebook.ipynb ...]
#
# With no argument it runs both the emcee and the pocomc notebook, in order.
# The login node caps a process at 3600 CPU-seconds and 8 GB RSS, which the
# sampling blows through, hence a batch job. $HOME is read-only on compute
# nodes, so each notebook is copied to $SCRATCH, executed there in place,
# and left for the caller to copy back.

#SBATCH --job-name=nle_inference_nb
#SBATCH --account=def-tingli
#SBATCH --partition=debug
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch/tvnguyen
#SBATCH --output=/scratch/tvnguyen/slurm_logs/nle/inference/%x_%j.out
#SBATCH --error=/scratch/tvnguyen/slurm_logs/nle/inference/%x_%j.err

set -uo pipefail

NLE_DIR="${NLE_DIR:-/home/tvnguyen/projects/sbi_stream/nle_stream_subhalo}"
NOTEBOOKS=("$@")
if [[ ${#NOTEBOOKS[@]} -eq 0 ]]; then
    NOTEBOOKS=("$NLE_DIR/nle_inference.ipynb"
               "$NLE_DIR/nle_inference_pocomc.ipynb")
fi
OUT_DIR="/scratch/tvnguyen/nle_inference/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

export XDG_CACHE_HOME="/scratch/tvnguyen/cache"
export MPLCONFIGDIR="/scratch/tvnguyen/cache/matplotlib"
export JUPYTER_DATA_DIR="/scratch/tvnguyen/.jupyter-data"
export JUPYTER_RUNTIME_DIR="/scratch/tvnguyen/.jupyter-runtime"
export TORCH_HOME="/scratch/tvnguyen/cache/torch"
export PYTHONPATH="$NLE_DIR:/home/tvnguyen/my_modules${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$JUPYTER_RUNTIME_DIR" "$TORCH_HOME"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "===== job start ====="
log "node:      $(hostname)"
log "gpus:      $(nvidia-smi -L 2>/dev/null | tr '\n' ';')"
log "notebooks: ${NOTEBOOKS[*]}"
log "out_dir:   $OUT_DIR"

# The `torch` venv, which owns the notebook's `python3` kernel.
VENV_DIR="${VENV_DIR:-$HOME/.venvs/torch}"
set +u  # Lmod's `module load` internals aren't -u-safe.
if [[ -s "$VENV_DIR/.modules" ]]; then
    module load $(cat "$VENV_DIR/.modules") 2>/dev/null
fi
source "$VENV_DIR/bin/activate"
set -u
log "python:    $(which python3) ($(python3 --version 2>&1))"

STATUS=0
for NOTEBOOK in "${NOTEBOOKS[@]}"; do
    cp -- "$NOTEBOOK" "$OUT_DIR/"
    EXECUTED="$OUT_DIR/$(basename "$NOTEBOOK")"

    log "executing: $EXECUTED"
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=7200 "$EXECUTED"
    RC=$?
    log "nbconvert exit: $RC  ($EXECUTED)"
    [[ $RC -ne 0 ]] && STATUS=$RC

    # Rendered copy for reading the results without a kernel.
    jupyter nbconvert --to html "$EXECUTED" --output-dir "$OUT_DIR" || true
done

log "===== job end (status $STATUS) ====="
exit "$STATUS"
