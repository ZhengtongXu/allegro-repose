#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
# One training chunk.   args: <run_name> <priority|standby>   (submitted by submit.sh)
# 1. queues its successor first (afterany), so a hard kill never breaks the chain;
# 2. trains; train.py resumes from the newest loadable checkpoint and checkpoints and exits on
#    SIGUSR1 (sent 5 min before the wall limit) or after <MODE>_CHUNK_MIN minutes if that is set;
# 3. exit 0 = finished -> cancel the successor; 3 = more to do -> successor runs;
#    other = crash -> successor retries; 3 crashes within 10 min of start in a row -> FAILED.
set -uo pipefail
RUN=$1 MODE=${2:-priority}
source "${AREPOSE_CL:?set by submit.sh}/common.sh"  # Slurm runs a spooled copy of this script
source "$CL/site.env"
ST=$RUNS/$RUN
read -r _ SEED EXTRA < <(job_line "$RUN")
echo "$(date '+%F %T') start $SLURM_JOB_ID $MODE on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" >> "$ST/history.log"
[[ -f $ST/DONE || -f $ST/FAILED ]] && exit 0
bash "$CL/submit.sh" "$RUN" --after "$SLURM_JOB_ID" || true

cd "$REPO"
export MUJOCO_GL=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_PROJECT
chunk_var=${MODE^^}_CHUNK_MIN
BUDGET=(); [[ -n ${!chunk_var:-} ]] && BUDGET=(--max-chunk-min "${!chunk_var}")
t0=$(date +%s)
.venv/bin/python scripts/train.py --run-name "$RUN" --seed "$SEED" "${BUDGET[@]}" $EXTRA &
pid=$!
trap 'kill -USR1 $pid 2>/dev/null' USR1 TERM
while kill -0 $pid 2>/dev/null; do wait $pid; code=$?; done
wait $pid 2>/dev/null; code=${code:-$?}
echo "$(date '+%F %T') end $SLURM_JOB_ID exit=$code after $(( $(date +%s) - t0 ))s" >> "$ST/history.log"

cancel_successor() { squeue -h -u "$USER" -n "$(job_name "$RUN")" -t PD -o %i | xargs -r scancel; }
if [[ $code == 0 ]]; then
  cancel_successor
elif [[ $code == 3 ]]; then
  rm -f "$ST/.crashes"
else
  if (( $(date +%s) - t0 < 600 )); then echo x >> "$ST/.crashes"; else rm -f "$ST/.crashes"; fi
  if [[ -f $ST/.crashes ]] && (( $(wc -l < "$ST/.crashes") >= 3 )); then
    echo "$(date '+%F %T') FAILED: 3 quick crashes in a row" | tee -a "$ST/history.log" > "$ST/FAILED"
    cancel_successor
  fi
fi
exit 0
