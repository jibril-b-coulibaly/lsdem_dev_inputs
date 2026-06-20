#!/usr/bin/env python3
"""Automated validation of the five sphere-plate verification cases against
analytical_solutions.py. For each case it picks the dump for a target node
count (default 400, override with argv[1]; falls back to the largest available)
and reports, over the steady-state window:
  - cases 2-5: per-component sim/analytical ratio and sign agreement;
  - case 1   : bounce apex heights and coefficient of restitution.
sim dump cols: 0=step 1..3=xyz 4..6=v 7..9=F 10..12=T
analytical    : 0..2=X 3..5=V 6..8=F 9..11=T
"""
import sys, glob, re, numpy as np
from analytical_solutions import analytical_solution, normal_force_linear

DT = 1.0e-4
R = 1.0; DENSITY = 2500; MASS = 4.0/3.0*np.pi*R**3*DENSITY
KN = 1.0e8; ETAN = 1.0e6; G = -9.81
COMP = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']
CASES = {1: 'bounce', 2: 'tangent', 3: 'shear', 4: 'twist', 5: 'roll'}

def load(path):
    rows = []
    for line in open(path):
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        c = s.split()
        if len(c) < 13:
            continue
        try:
            rows.append([float(x) for x in c[:13]])
        except ValueError:
            continue
    return np.array(rows)

def pick_dump(base, target_N):
    """Return (path, N) for the dump closest to target_N among available files."""
    found = {}
    for p in glob.glob(f'dump_grain_{base}_*.txt'):
        m = re.search(r'_0*(\d+)\.txt$', p)
        if m:
            found[int(m.group(1))] = p
    if not found:
        return None, None
    N = target_N if target_N in found else max(found)
    return found[N], N

def validate_steady(cf, path, N):
    sim = load(path)
    steps = sim[:, 0].astype(int)
    tmax = steps.max()
    ana = analytical_solution(np.arange(tmax + 1) * DT, cf)
    mask = (steps > 0) & (steps >= 0.7 * tmax)
    idx = np.where(mask)[0]
    si = steps[idx]
    print(f"\n=== case {cf} ({CASES[cf]})  N={N}, steady window {si.min()}-{si.max()} ===")
    print(f"  {'comp':4} {'sim_mean':>12} {'ana_mean':>12} {'ratio':>8}  sign")
    for k, cname in enumerate(COMP):
        sm = np.mean(sim[idx, 7 + k]); am = np.mean(ana[si, 6 + k])
        if abs(am) < 1e-6 and abs(sm) < 1e-6:
            continue
        ratio = sm / am if abs(am) > 1e-9 else float('nan')
        signok = (np.sign(sm) == np.sign(am)) or abs(sm) < 1e-6
        note = '' if abs(am) > 1e-6 else '  (ana=0; sim noise)'
        print(f"  {cname:4} {sm:12.1f} {am:12.1f} {ratio:8.2f}  {'OK' if signok else 'FLIP'}{note}")

def apexes(z, st):
    return [(st[i] * DT, z[i]) for i in range(1, len(z) - 1)
            if z[i] > z[i - 1] and z[i] >= z[i + 1] and z[i] > R]

def bounce(n, dt=DT):
    z = 1.5; v = 0.0; Z = np.zeros(n); Z[0] = z
    for i in range(1, n):
        vn = -v
        fz = MASS * G if z > R else normal_force_linear(R - z, vn, R, KN, ETAN) + MASS * G
        a = fz / MASS; v += a * dt; z += v * dt + 0.5 * a * dt * dt; Z[i] = z
    return Z

def restitution(pk):
    if len(pk) < 2:
        return float('nan')
    h1, h2 = pk[0][1] - R, pk[1][1] - R
    return (h2 / h1) ** 0.5 if h1 > 0 else float('nan')

def validate_bounce(path, N):
    sim = load(path); z = sim[:, 3]; st = sim[:, 0].astype(int)
    ap_s = apexes(z, st)
    ana = bounce(int(st.max()) + 1)
    ap_a = apexes(ana, np.arange(len(ana)))
    print(f"\n=== case 1 (bounce)  N={N} ===")
    print("  apex heights@time   sim:", [f"{zz:.4f}@{tt:.3f}" for tt, zz in ap_s][:4])
    print("                      ana:", [f"{zz:.4f}@{tt:.3f}" for tt, zz in ap_a][:4])
    print(f"  restitution  sim={restitution(ap_s):.3f}  ana={restitution(ap_a):.3f}")

if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    for cf, base in CASES.items():
        path, N = pick_dump(base, target)
        if path is None:
            print(f"\ncase {cf} ({base}): no dump found"); continue
        if cf == 1:
            validate_bounce(path, N)
        else:
            validate_steady(cf, path, N)
