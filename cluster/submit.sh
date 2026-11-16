#!/bin/bash
# Queue the next chunk of a run listed in cluster/jobs.txt.
#   usage: submit.sh <run_name> [--mode priority|standby] [--after <jobid>]
# The mode is remembered in runs/<run>/MODE (default priority); later chunks reuse it, so pass
# --mode once to start or to switch a run. At most one pending chunk per run (job name arp_<run>).
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
source "$CL/site.env"
RUN=$1; shift
AFTER="" MODE=""
while [[ $# -gt 0 ]]; do case $1 in
  --after) AFTER=$2; shift 2;;
  --mode) MODE=$2; shift 2;;
  *) echo "bad arg $1"; exit 2;; esac; done
[[ -n $(job_line "$RUN") ]] || { echo "run $RUN not in jobs.txt"; exit 2; }
ST=$RUNS/$RUN
[[ -f $ST/DONE ]] && { echo "$RUN: DONE"; exit 0; }
[[ -f $ST/FAILED ]] && { echo "$RUN: FAILED (rm $ST/FAILED to retry)"; exit 0; }
mkdir -p "$CL/logs" "$ST"
[[ -n $MODE ]] && echo "$MODE" > "$ST/MODE"
MODE=$(cat "$ST/MODE" 2>/dev/null || echo priority)
NAME=$(job_name "$RUN")
pending=$(squeue -h -u "$USER" -n "$NAME" -t PD,CF -o %i | wc -l)
running=$(squeue -h -u "$USER" -n "$NAME" -t R -o %i | { grep -vx "${AFTER:-none}" || true; } | wc -l)
if (( pending > 0 )) || { [[ -z $AFTER ]] && (( running > 0 )); }; then echo "$RUN: already queued/running"; exit 0; fi
SB=(-A "$ACCOUNT" --gres=gpu:1 --mem="$MEM" -J "$NAME" -o "$CL/logs/${RUN}_%j.out" --signal=B:USR1@300)
case $MODE in
  priority) SB+=(-p "$PRIORITY_PARTITION" -q "$PRIORITY_QOS" -c "$PRIORITY_CPUS" -t "$PRIORITY_WALL")
            [[ -n ${PRIORITY_CONSTRAINT:-} ]] && SB+=(--constraint="$PRIORITY_CONSTRAINT");;
  standby)  SB+=(-p "$STANDBY_PARTITIONS" -q "$STANDBY_QOS" -c "$STANDBY_CPUS" -t "$STANDBY_WALL");;
  *) echo "unknown mode $MODE"; exit 2;;
esac
[[ -n $AFTER ]] && SB+=(--dependency=afterany:$AFTER)
jid=$(sbatch --parsable "${SB[@]}" "$CL/job.sbatch" "$RUN" "$MODE")
echo "$(date '+%F %T') submitted $jid $MODE after=${AFTER:-none}" | tee -a "$ST/history.log"
