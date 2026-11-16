# Shared by submit.sh, job.sbatch and status.sh (sourced): paths, the jobs.txt lookup, job names.
CL=${AREPOSE_CL:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export AREPOSE_CL=$CL                       # job.sbatch runs from a Slurm spool copy and needs the real path
REPO=$(cd "$CL/.." && pwd)
RUNS=${AREPOSE_RUNS:-$REPO/runs}
job_line() { awk -v r="$1" '!/^#/ && $1==r' "$CL/jobs.txt"; }   # "<run> <seed> <args...>" or empty
job_name() { echo "arp_$1"; }
