#!/bin/bash
# Full N-sweep for cases 1-4 with the current (force-clip) binary.
# Writes PADDED dump names (dump_grain_<base>_NNNN.txt) so each N overwrites the
# stale file and there is exactly one dump per (case, N).
cd /home/dvdh/LAMMPS/lammps-dev/lsdem_dev_inputs/sphere-plate || exit 9
LMP=/home/dvdh/LAMMPS/lammps-dev/build_serial/lmp
declare -A BASE=( [1]=bounce [2]=tangent [3]=shear [4]=twist )
for N in 100 200 400 800 1600 3200; do
  printf -v NP "%04d" "$N"
  for c in 1 2 3 4; do
    if [ "$c" -eq 1 ]; then STEPS=30000; else STEPS=5000; fi
    b=${BASE[$c]}
    tmp=/tmp/sweep_c${c}_N${N}.in
    sed -e "s/variable N equal 3200/variable N equal ${N}/" \
        -e "s/^run 50000/run ${STEPS}/" "input_case_${c}" > "$tmp"
    echo "=== case $c ($b) N=$N steps=$STEPS  start $(date +%H:%M:%S) ==="
    "$LMP" -in "$tmp" > "/tmp/sweep_c${c}_N${N}.log" 2>&1
    rc=$?
    grep -iE 'Loop time|ERROR' "/tmp/sweep_c${c}_N${N}.log" | head -1
    if [ "$rc" != "0" ]; then echo "  *** case $c N=$N exit=$rc ***"; fi
    # rename unpadded -> padded (overwrites stale), no-op when N already 4 digits
    if [ -f "dump_grain_${b}_${N}.txt" ] && [ "$N" != "$NP" ]; then
      mv -f "dump_grain_${b}_${N}.txt" "dump_grain_${b}_${NP}.txt"
    fi
  done
done
echo "SWEEP_DONE $(date +%H:%M:%S)"
