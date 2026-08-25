#!/bin/bash
# sbc_temper.sbatch — tempered SBC, all datasets inside ONE whole node.
#
#   sbatch slurm/sbc_temper.sbatch <T> <n_datasets> <n_stars> <out_dir>
#   sbatch slurm/sbc_temper.sbatch 2.2 200 250 /scratch/tvnguyen/nle_sbc/T2p2
#
# Deliberately NOT a job array. Trillium's CPU nodes are allocated whole
# (192 cores), so a 25-task array would hold 25 nodes to run one serial
# pocomc chain each and leave ~184 cores idle per node. Instead this takes
# one node and runs N_WORKERS independent `sbc_calibration.py run`
# processes on it, each owning a slice of the datasets and each pinned to
# a few threads.
#
# Why many small workers beat one big one: torch's CPU scaling is strongly
# sublinear -- measured 46,912 rows/s at 4 threads and only 80,159 at 16,
# so 24 x 8 threads delivers far more aggregate throughput than 1 x 192.
# pocomc is serial per dataset anyway, so there is nothing to gain from
# threading a single run harder.
#
# Sizing at the defaults: 200 datasets / 24 workers = ~9 datasets each,
# ~390 s per dataset at n_total=2000 and 250 stars, so ~1 h wall. The 4 h
# limit is 4x headroom.
#
# `--n-total 2000` matches the untempered aau_250stars run. Do not lower
# it for speed: the tempered ranks are only comparable to that baseline if
# the sampler settings are identical.

#SBATCH --job-name=nle_sbc_temper
#SBATCH --account=def-tingli
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=04:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=tri.vt.nguyen@gmail.com
#SBATCH --chdir=/scratch/tvnguyen
#SBATCH --output=/scratch/tvnguyen/slurm_logs/nle/sbc/%x_%j.out
#SBATCH --error=/scratch/tvnguyen/slurm_logs/nle/sbc/%x_%j.err

set -uo pipefail

NLE_DIR="${NLE_DIR:-/home/tvnguyen/projects/sbi_stream/nle_stream_subhalo}"
TEMPERATURE="${1:-1.0}"
N_DATASETS="${2:-200}"
N_STARS="${3:-250}"
OUT_DIR="${4:-/scratch/tvnguyen/nle_sbc/temper_T${TEMPERATURE}}"
# Dataset index to start from. Non-zero extends an existing run: the
# index seeds each dataset (`SeedSequence([seed, i])`), so [200, 500)
# are new draws, nothing already computed is repeated, and the npz names
# (`sbc_<start>.npz`) cannot collide with the earlier files.
START_OFFSET="${5:-0}"
N_WORKERS="${N_WORKERS:-24}"
THREADS="${THREADS:-8}"
N_TOTAL="${N_TOTAL:-2000}"
DELTA_VMIN="${DELTA_VMIN:-0.03}"
CONFIG="${CONFIG:-$NLE_DIR/configs/nsf_9params_200ds_track.py}"

mkdir -p "$OUT_DIR" /scratch/tvnguyen/slurm_logs/nle/sbc
export XDG_CACHE_HOME="/scratch/tvnguyen/cache"
export MPLCONFIGDIR="/scratch/tvnguyen/cache/matplotlib"
export TORCH_HOME="/scratch/tvnguyen/cache/torch"
# Each worker must keep to its own thread budget, or 24 of them will each
# try to use all 192 cores and thrash.
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$TORCH_HOME"

log() { echo "[$(date '+%F %T')] $*"; }

log "===== start ====="
log "node:        $(hostname)   cpus ${SLURM_CPUS_PER_TASK:-?}"
log "temperature: $TEMPERATURE"
log "datasets:    $N_DATASETS at $N_STARS stars, n_total $N_TOTAL"
log "index range: [$START_OFFSET, $(( START_OFFSET + N_DATASETS )))"
log "workers:     $N_WORKERS x $THREADS threads"
log "out_dir:     $OUT_DIR"

VENV_DIR="${VENV_DIR:-$HOME/.venvs/torch}"
set +u
[[ -s "$VENV_DIR/.modules" ]] && module load $(cat "$VENV_DIR/.modules") 2>/dev/null
source "$VENV_DIR/bin/activate"
set -u

# Ceiling division, so the last worker takes the short slice rather than
# some datasets being dropped.
PER=$(( (N_DATASETS + N_WORKERS - 1) / N_WORKERS ))
log "per worker:  $PER datasets"

cd "$NLE_DIR"
pids=()
for (( w = 0; w < N_WORKERS; w++ )); do
    LOCAL=$(( w * PER ))
    [[ $LOCAL -ge $N_DATASETS ]] && break
    COUNT=$PER
    [[ $(( LOCAL + COUNT )) -gt $N_DATASETS ]] && COUNT=$(( N_DATASETS - LOCAL ))
    START=$(( START_OFFSET + LOCAL ))
    python3 sbc_calibration.py run \
        --config "$CONFIG" \
        --output-dir "$OUT_DIR" \
        --start "$START" \
        --n-datasets "$COUNT" \
        --n-stars "$N_STARS" \
        --n-total "$N_TOTAL" \
        --temperature "$TEMPERATURE" \
        --delta-vmin "$DELTA_VMIN" \
        --threads "$THREADS" \
        > "$OUT_DIR/worker_${START_OFFSET}_$(printf "%03d" "$w").log" 2>&1 &
    pids+=($!)
done
log "launched ${#pids[@]} workers"

RC=0
for pid in "${pids[@]}"; do
    wait "$pid" || RC=$?
done

log "workers done (exit $RC); $(ls "$OUT_DIR"/sbc_*.npz 2>/dev/null | wc -l) npz written"
log "===== end ====="
exit "$RC"
