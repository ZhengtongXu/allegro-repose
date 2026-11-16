#!/bin/bash
# One line per run in jobs.txt: queue state and mode, last training line, last evaluation.
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
for run in $(awk '!/^#/ && NF {print $1}' "$CL/jobs.txt"); do
  st=$(squeue -h -u "$USER" -n "$(job_name "$run")" -o %T | sort | uniq -c | tr -s ' ' | tr '\n' ' ')
  [[ -f $RUNS/$run/DONE ]] && st="DONE"
  [[ -f $RUNS/$run/FAILED ]] && st="FAILED"
  mode=$(cat "$RUNS/$run/MODE" 2>/dev/null || echo priority)
  log=$(ls -t "$CL"/logs/${run}_[0-9]*.out 2>/dev/null | head -1)
  last=$([[ -n $log ]] && grep -E "^\[ppo\] it " "$log" | tail -1 | cut -c1-110)
  ev=$([[ -n $log ]] && grep -oE "successes_per_episode': [0-9.]+" "$log" | tail -1)
  printf '%-20s %-8s [%s] %s | %s\n' "$run" "$mode" "${st% }" "$last" "$ev"
done
