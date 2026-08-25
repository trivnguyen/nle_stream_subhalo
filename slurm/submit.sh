#!/usr/bin/env bash
# submit.sh — submit train_nle.py to SLURM with per-run log bookkeeping.
#
# Usage:
#   ./submit.sh <config.py> [--time=D-HH:MM:SS] [--partition=name] [...] [--config.field=value ...]
#
# Examples:
#   ./submit.sh ../configs/nsf_9params.py
#   ./submit.sh ../configs/nsf_9params.py --config.train_batch_size=2048
#   ./submit.sh ../configs/nsf_9params.py --partition=debug --time=00:15:00
#
# Resource overrides — trailing --flag=value, --flag value, or env var, all
# optional (--config.<field>=<value> overrides go straight through to python):
#   --account=   / ACCOUNT   (default: def-tingli)
#   --partition= / PARTITION (default: compute)   # debug | compute | compute_h200 | *_full_node
#   --gpus=      / GPUS      (default: 1)         # keep at 1 — train_nle.py isn't DDP-ready
#   --cpus=      / CPUS      (default: 24)        # full quarter-node cores; no mem flag, it's a no-op here
#   --time=      / TIME      (default: 12:00:00)  # compute partition max is 1 day
#
# $HOME is read-only on compute nodes, so all run artifacts (SLURM out/err,
# manifest, config snapshot, stdout log) live under $SCRATCH instead, one
# directory per run: $SCRATCH/slurm_logs/nle/<config_name>/<timestamp>/.
# A one-line-per-run index is also kept at $SCRATCH/slurm_logs/nle/runs.tsv.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.py> [--config.field=value ...]" >&2
    exit 1
fi

CONFIG_ARG="$1"
shift
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NLE_DIR="$(dirname "$SCRIPT_DIR")"

# Resolve config to an absolute path (accept paths relative to cwd or to nle_stream/).
# A trailing ":variant" is ml_collections' argument to get_config(); split it
# off before testing for the file, then carry it through to train_nle.py.
CONFIG_PATH="${CONFIG_ARG%%:*}"
CONFIG_VARIANT=""
[[ "$CONFIG_ARG" == *:* ]] && CONFIG_VARIANT=":${CONFIG_ARG#*:}"

if [[ -f "$CONFIG_PATH" ]]; then
    CONFIG_ABS="$(cd "$(dirname "$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
elif [[ -f "$NLE_DIR/$CONFIG_PATH" ]]; then
    CONFIG_ABS="$(cd "$(dirname "$NLE_DIR/$CONFIG_PATH")" && pwd)/$(basename "$CONFIG_PATH")"
else
    echo "Error: config not found: $CONFIG_PATH" >&2
    exit 1
fi

ACCOUNT="${ACCOUNT:-def-tingli}"
PARTITION="${PARTITION:-compute}"
GPUS="${GPUS:-1}"
CPUS="${CPUS:-24}"
TIME="${TIME:-12:00:00}"

# Trailing overrides win over the env vars above; --config.field=value
# passes through to train_nle.py.
#
# Both `--flag value` and `--flag=value` are accepted. Taking only the
# `=` form is what made `--time 1-00:00:00` look like it did nothing: it
# fell through to the passthrough branch, so the job queued at the default
# time limit and then died on absl's unknown-flag error.
#
# An unrecognized --flag is a hard error for the same reason. Forwarding
# it silently only defers the failure to the compute node.
need_value() {
    if [[ $# -lt 2 ]]; then
        echo "Error: $1 needs a value (e.g. $1=VALUE or $1 VALUE)." >&2
        exit 1
    fi
}

REMAINING_ARGS=()
set -- "${EXTRA_ARGS[@]}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --account=*)   ACCOUNT="${1#*=}";   shift   ;;
        --account)     need_value "$@"; ACCOUNT="$2";   shift 2 ;;
        --partition=*) PARTITION="${1#*=}"; shift   ;;
        --partition)   need_value "$@"; PARTITION="$2"; shift 2 ;;
        --gpus=*)      GPUS="${1#*=}";      shift   ;;
        --gpus)        need_value "$@"; GPUS="$2";      shift 2 ;;
        --cpus=*)      CPUS="${1#*=}";      shift   ;;
        --cpus)        need_value "$@"; CPUS="$2";      shift 2 ;;
        --time=*)      TIME="${1#*=}";      shift   ;;
        --time)        need_value "$@"; TIME="$2";      shift 2 ;;
        # absl takes --config.x=y and --config.x y; a bare value after the
        # latter lands in the catch-all below, which is what we want.
        --config.*)    REMAINING_ARGS+=("$1"); shift ;;
        --*)
            echo "Error: unknown option '$1'." >&2
            echo "Resource flags: --account --partition --gpus --cpus --time" >&2
            echo "Everything else must be a --config.<field>=<value> " \
                 "override for python." >&2
            exit 1
            ;;
        *)             REMAINING_ARGS+=("$1"); shift ;;
    esac
done
EXTRA_ARGS=("${REMAINING_ARGS[@]}")

LOG_ROOT="${LOG_ROOT:-${SCRATCH:-/scratch/$(whoami)}/slurm_logs/nle}"
# The variant joins the name, so sweep arms get their own log directory
# and job name instead of all colliding on the shared config's basename.
CONFIG_NAME="$(basename "$CONFIG_ABS" .py)${CONFIG_VARIANT:+_${CONFIG_VARIANT#:}}"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
RUN_DIR="$LOG_ROOT/$CONFIG_NAME/$RUN_ID"
mkdir -p "$RUN_DIR"
cp -- "$CONFIG_ABS" "$RUN_DIR/config_snapshot.py"

# nle_stream isn't versioned yet; 'unknown' until it is.
NLE_COMMIT="$(git -C "$NLE_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
# The simulator that produced the training datasets.
SIMS_COMMIT="$(git -C "$NLE_DIR/../sbi_stream_pipeline" rev-parse --short HEAD 2>/dev/null || echo unknown)"

{
    echo "submitted_at:    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "submitted_by:    $(whoami)@$(hostname)"
    echo "config:          $CONFIG_ABS"
    echo "extra_args:      ${EXTRA_ARGS[*]:-<none>}"
    echo "account:         $ACCOUNT"
    echo "partition:       $PARTITION"
    echo "gpus_per_node:   $GPUS"
    echo "cpus_per_task:   $CPUS"
    echo "time_limit:      $TIME"
    echo "nle_commit:      $NLE_COMMIT"
    echo "sims_commit:     $SIMS_COMMIT"
} > "$RUN_DIR/manifest.txt"

JOB_NAME="nle_${CONFIG_NAME}"

JOBID="$(sbatch --parsable \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --gpus-per-node="$GPUS" \
    --cpus-per-task="$CPUS" \
    --time="$TIME" \
    --job-name="$JOB_NAME" \
    --output="$RUN_DIR/slurm-%j.out" \
    --error="$RUN_DIR/slurm-%j.err" \
    --export=ALL,RUN_DIR="$RUN_DIR",NLE_DIR="$NLE_DIR" \
    "$SCRIPT_DIR/train_nle.sbatch" "$CONFIG_ABS$CONFIG_VARIANT" \
    "${EXTRA_ARGS[@]}")"

echo "job_id:          $JOBID" >> "$RUN_DIR/manifest.txt"

RUNS_INDEX="$LOG_ROOT/runs.tsv"
if [[ ! -f "$RUNS_INDEX" ]]; then
    printf 'job_id\tsubmitted_at\tconfig_name\tpartition\trun_dir\textra_args\n' > "$RUNS_INDEX"
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOBID" "$(date '+%Y-%m-%d %H:%M:%S')" "$CONFIG_NAME" "$PARTITION" "$RUN_DIR" "${EXTRA_ARGS[*]:-}" \
    >> "$RUNS_INDEX"

echo "Submitted job $JOBID ($JOB_NAME, partition=$PARTITION, gpu:$GPUS)"
echo "Logs: $RUN_DIR"
echo "Track: squeue -j $JOBID   |   tail -f $RUN_DIR/slurm-$JOBID.out"
