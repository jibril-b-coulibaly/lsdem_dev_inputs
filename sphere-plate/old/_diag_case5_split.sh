#!/bin/bash
# Isolate the case-5 rolling-resistance tangential force into elastic vs viscous.
cd /home/dvdh/LAMMPS/lammps-dev/lsdem_dev_inputs/sphere-plate || exit 9
LMP=/home/dvdh/LAMMPS/lammps-dev/build_serial/lmp

report() {  # $1 = label
python3 - "$1" <<'PY'
import sys, numpy as np
lab = sys.argv[1]
rows = [l.split() for l in open('dump_grain_roll_400.txt')
        if l.strip() and not l.startswith('#') and len(l.split()) >= 13]
a = np.array([[float(x) for x in r[:13]] for r in rows])
st = a[:, 0]; m = st >= 0.7 * st.max()
fx, fy = a[m, 7].mean(), a[m, 8].mean()
ft = np.hypot(fx, fy)
print(f"  {lab:10s} |Ft|={ft:8.0f}  dir=({fx/ft:+.2f},{fy/ft:+.2f})")
PY
}

run() {  # $1 = label, $2 = "kn kt mu etan etat knp cut"
  sed -e "s/variable N equal 3200/variable N equal 400/" \
      -e "s/^run 50000/run 5000/" \
      -e "s/^pair_coeff.*/pair_coeff * * $2/" \
      input_case_5 > "/tmp/c5_$1.in"
  timeout 120 "$LMP" -in "/tmp/c5_$1.in" > "/tmp/c5_$1.log" 2>&1
  report "$1"
}

echo "case 5 rolling resistance, tangential-force split (N=400):"
echo "  viscous-only analytical prediction |Ft| = 19635"
run full   "1.0e8 0.5e8 0.5 1.0e6 1.0e6 0 1.0"
run noVisc "1.0e8 0.5e8 0.5 1.0e6 0.0   0 1.0"
run noElas "1.0e8 0.0   0.5 1.0e6 1.0e6 0 1.0"
echo "done"
