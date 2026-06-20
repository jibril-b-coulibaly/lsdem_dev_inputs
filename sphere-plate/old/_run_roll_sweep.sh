#!/bin/bash
# Roll (case 5) N-sweep with the true-rolling setup, PADDED dump names so they
# replace the stale Jun-11 sliding-setup files. Args: list of N values.
cd /home/dvdh/LAMMPS/lammps-dev/lsdem_dev_inputs/sphere-plate || exit 9
LMP=/home/dvdh/LAMMPS/lammps-dev/build_serial/lmp
for N in "$@"; do
  printf -v NP "%04d" "$N"
  tmp=/tmp/roll_N${N}.in
  sed -e "s/variable N equal 3200/variable N equal ${N}/" \
      -e "s/^run 50000/run 5000/" input_case_5 > "$tmp"
  echo "=== roll N=$N  start $(date +%H:%M:%S) ==="
  "$LMP" -in "$tmp" > "/tmp/roll_N${N}.log" 2>&1
  rc=$?
  grep -iE 'Loop time|ERROR' "/tmp/roll_N${N}.log" | head -1
  if [ "$rc" != "0" ]; then echo "  *** exit=$rc ***"; fi
  if [ -f "dump_grain_roll_${N}.txt" ] && [ "$N" != "$NP" ]; then
    mv -f "dump_grain_roll_${N}.txt" "dump_grain_roll_${NP}.txt"
  fi
done
# remove the stray unpadded N=400 dump if a padded one now exists
[ -f dump_grain_roll_0400.txt ] && rm -f dump_grain_roll_400.txt
echo "ROLL_SWEEP_DONE $(date +%H:%M:%S)"
