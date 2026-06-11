#!/bin/bash
# Run selected sphere-plate cases at a given node count N for a given number of steps.
# Usage: ./run_cases.sh <N> <steps> <case...>
#   e.g. ./run_cases.sh 100 5000 2 3 4 5
set -e
cd "$(dirname "$0")"
LMP=../../build_serial/lmp
N=${1:-100}
STEPS=${2:-5000}
shift 2 || true
CASES=("$@")
if [ ${#CASES[@]} -eq 0 ]; then CASES=(1 2 3 4 5); fi
for c in "${CASES[@]}"; do
  tmp="/tmp/sp_case_${c}_N${N}.in"
  sed -e "s/variable N equal 3200/variable N equal ${N}/" \
      -e "s/^run 50000/run ${STEPS}/" \
      "input_case_${c}" > "$tmp"
  echo "=== case ${c}  (N=${N}, steps=${STEPS}) ==="
  "$LMP" -in "$tmp" 2>&1 | grep -iE "ERROR|WARNING|Loop time" || true
done
echo "done"
