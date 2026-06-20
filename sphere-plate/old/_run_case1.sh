#!/bin/bash
# Bounded case-1 runner. Usage: _run_case1.sh <N> <STEPS> <TIMEOUT_S>
# Always wrapped in `timeout` so it can never hang the session.
cd /home/dvdh/LAMMPS/lammps-dev/lsdem_dev_inputs/sphere-plate || { echo "cd FAILED"; exit 9; }
LMP=/home/dvdh/LAMMPS/lammps-dev/build_serial/lmp
N=${1:-100}
STEPS=${2:-1000}
TMO=${3:-120}

if [ ! -x "$LMP" ]; then echo "NO BINARY at $LMP"; exit 8; fi
echo "binary mtime: $(stat -c '%y' "$LMP")"

IN=/tmp/c1_N${N}.in
OUT=/tmp/c1_N${N}_${STEPS}.log
# Match ANY "variable N equal <num>" so the template default (3200) is handled.
sed -e "s/variable N equal [0-9]\+/variable N equal ${N}/" \
    -e "s/^run 50000/run ${STEPS}/" input_case_1 > "$IN"
echo "N line -> $(grep 'variable N equal' "$IN")"

echo "running N=${N} steps=${STEPS} (timeout ${TMO}s)..."
start=$(date +%s.%N)
timeout "$TMO" "$LMP" -in "$IN" > "$OUT" 2>&1
rc=$?
end=$(date +%s.%N)
printf "lmp exit=%s  elapsed=%.1fs\n" "$rc" "$(echo "$end - $start" | bc)"
if [ "$rc" = "124" ]; then echo "*** TIMED OUT — binary is too slow / hung ***"; fi
grep -iE 'Loop time|ERROR|exception' "$OUT" | head
echo -n "last dump step: "
tail -1 "dump_grain_bounce_${N}.txt" 2>/dev/null | awk '{print $1}'
