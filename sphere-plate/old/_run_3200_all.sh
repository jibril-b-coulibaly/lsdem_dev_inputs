#!/bin/bash
# N=3200 runs for all five cases. Prescribed-kinematics cases (2-5, 5000 steps)
# first, the slow free bounce (case 1, 30000 steps) last. Progress is appended
# to a PERSISTENT log in the project dir so it survives a WSL reset (which clears
# /tmp). Dump names at N=3200 are already 4-digit, so no rename is needed; each
# run overwrites the stale file in place.
cd /home/dvdh/LAMMPS/lammps-dev/lsdem_dev_inputs/sphere-plate || exit 9
LMP=/home/dvdh/LAMMPS/lammps-dev/build_serial/lmp
LOG=./_3200_progress.log
declare -A BASE=( [1]=bounce [2]=tangent [3]=shear [4]=twist [5]=roll )
echo "START $(date +%F_%T)" > "$LOG"
for c in 2 3 4 5 1; do
  b=${BASE[$c]}
  if [ "$c" -eq 1 ]; then STEPS=30000; else STEPS=5000; fi
  sed -e "s/^run 50000/run ${STEPS}/" "input_case_${c}" > "/tmp/c${c}_3200.in"
  echo "RUN $b (case $c) N=3200 steps=$STEPS  $(date +%T)" >> "$LOG"
  "$LMP" -in "/tmp/c${c}_3200.in" > "/tmp/c${c}_3200.log" 2>&1
  rc=$?
  echo "  DONE $b rc=$rc  $(grep -i 'Loop time' "/tmp/c${c}_3200.log" | head -1)  $(date +%T)" >> "$LOG"
done
echo "ALL_DONE $(date +%F_%T)" >> "$LOG"
