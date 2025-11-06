#!/usr/bin/env python3
"""
plot_grain_bounce.py — plots LAMMPS grain bounce data

Now uses actual time instead of simulation steps.
"""

import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from analytical_solutions import *

# Fallback: use mathtext (Computer Modern for math) and a sans-serif font for text
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
    "font.size": 10.0,
})

# ---- Parsing ----
def parse_file(path):
    steps = []
    rows = []
    with open(path, 'r') as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            cols = s.split()
            if len(cols) < 13:
                continue
            try:
                step = float(cols[0])
                vals = [float(x) for x in cols[1:13]]
            except ValueError:
                continue
            steps.append(step)
            rows.append(vals)
    if len(steps) == 0:
        return np.array([]), np.empty((0,12))
    return np.array(steps), np.vstack(rows)

# ---- Analytical ----
# def analytical_solution(t):
#     freq = 200.0
#     phase = comp_index * (np.pi / 6.0)
#     return 0.1 * np.sin(2.0 * np.pi * freq * t + phase)

# ---- Plotting ----
def make_plots(all_data, all_times, labels, dt, caseflag, outdir="figures", outprefix="grain_bounce"):
    os.makedirs(outdir, exist_ok=True)

    groups = [
        ("Coordinates", ["x","y","z"], [0,1,2]),
        ("Velocity",    ["x","y","z"], [3,4,5]),
        ("Force",       ["x","y","z"], [6,7,8]),
        ("Torque",      ["x","y","z"], [9,10,11]),
    ]

    # Determine global maximum time
    global_max_t = 0.0
    for t in all_times:
        if t.size > 0:
            global_max_t = max(global_max_t, float(np.max(t)))
    if global_max_t <= 0.0:
        global_max_t = dt  # fallback

    # Compute analytical solution
    t_ana = np.arange(0.0, global_max_t + dt, dt)
    y_ana = analytical_solution(t_ana,caseflag)

    saved_files = []
    for gname, comps, indices in groups:
        for ci, comp in enumerate(comps):
            idx = indices[ci]
            fig, ax = plt.subplots(figsize=(8,4))

            # Analytical (plotted first, solid black line, thicker)
            ax.plot(t_ana, y_ana[:,idx], linestyle='-', linewidth=2.8,
                    color='k', label='analytical (dummy)', zorder=0)

            # Numerical datasets
            for data, times, lab in zip(all_data, all_times, labels):
                if data.size == 0 or times.size == 0:
                    continue
                ax.plot(times, data[:, idx], linewidth=1.2, label=lab, zorder=1)

            # Boxed axes with ticks
            for spine in ax.spines.values():
                spine.set_visible(True)
            ax.tick_params(which='both', direction='in', top=True, right=True)

            ax.set_xlim(left=0.0, right=global_max_t)
            ax.set_xlabel('time [s]')
            ax.set_ylabel(f'{gname} {comp}')
            ax.set_title(f'{gname} component {comp}')
            ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)
            ax.legend(loc='best', fontsize='small', frameon=True)

            fig.tight_layout(pad=1.0)
            fname = os.path.join(outdir, f"{outprefix}_{gname.lower()}_{comp}.png".replace(" ", "_"))
            fig.savefig(fname, dpi=200, bbox_inches='tight')
            saved_files.append(fname)
            plt.close(fig)

    return saved_files

# ---- Main ----
def main(dt=1.0,caseflag=0):
    pattern = "dump_grain_bounce_*.txt"
    files = sorted(glob.glob(pattern))

    if len(files) == 0:
        print("No matching files found. Creating two demo files for demonstration.")
        files = ["dump_grain_bounce_demo1.txt", "dump_grain_bounce_demo2.txt"]
        t = np.arange(0,15000,10)
        for fn, amp in zip(files, (0.5, 0.6)):
            with open(fn, 'w') as fh:
                for ti in t:
                    vals = np.concatenate([
                        amp*np.sin(0.005*ti + 0.1*np.arange(3)),       # coords
                        0.02*np.cos(0.01*ti + 0.2*np.arange(3)),       # vel
                        0.1*np.sin(0.007*ti + 0.3*np.arange(3)),       # force
                        0.05*np.cos(0.004*ti + 0.4*np.arange(3)),      # torque
                    ])
                    fh.write(f"{ti} " + " ".join(f"{v:.6e}" for v in vals) + "\n")

    all_data = []
    all_times = []
    labels = []
    for f in files:
        steps, d = parse_file(f)
        times = steps * dt
        all_times.append(times)
        all_data.append(d)
        labels.append(os.path.basename(f))

    print(f"Found {len(files)} file(s): {files}")
    saved = make_plots(all_data, all_times, labels, dt, caseflag, outdir="figures", outprefix="grain_bounce")
    print("Saved plots:")
    for s in saved:
        print(" -", s)

if __name__ == "__main__":
    # Example usage: main(dt=1e-5)
    main(dt=1.0e-4,caseflag=0)

# End of file