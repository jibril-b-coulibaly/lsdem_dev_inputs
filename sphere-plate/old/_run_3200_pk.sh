#!/bin/bash
# N=3200 runs for the prescribed-kinematics cases (2-5), 5000 steps each.
# Progress is logged to a PERSISTENT file in the project dir so it survives a
# WSL reset (which clears /tmp). Dumps are already 4-digit (padded) at N=3200.
cd /home/dvdh/LAMMPS/lammps-dev/lsdem_dev_inputs/sphere-plate || exit 9
LMP=/home/dvdh/LAMMPS/lammps-dev/build_serial/lmp
LOG=./_3200_progress.log
declare -A BASE=( [2]=tangent [3]=shear [4]=twist [5]=roll )
echo "START $(date +%F_%T)" > "$LOG"
for c in 2 3 4 5; do
  b=${BASE[$c]}
  sed -e "s/^run 50000/run 5000/" "input_case_${c}" > "/tmp/c${c}_3200.in"
  echo "RUN $b (case $c) N=3200 steps=5000  $(date +%T)" >> "$LOG"
  "$LMP" -in "/tmp/c${c}_3200.in" > "/tmp/c${c}_3200.log" 2>&1
  rc=$?
  echo "  DONE $b rc=$rc  $(grep -i 'Loop time' "/tmp/c${c}_3200.log" | head -1)  $(date +%T)" >> "$LOG"
done
echo "ALL_DONE $(date +%F_%T)" >> "$LOG"
