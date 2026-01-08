#!/usr/bin/env python3
"""
Plots LAMMPS grain bounce data with Nature journal styling
"""

import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from analytical_solutions import *

# Nature journal style with professional appearance
# Using Computer Modern for math and a clean serif font for text
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
    "font.size": 10.0,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.5,
    "grid.linewidth": 0.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})

# Nature-inspired colorblind-friendly palette
NATURE_COLORS = [
    '#0173B2',  # Blue
    '#DE8F05',  # Orange
    '#029E73',  # Green
    '#CC78BC',  # Purple
    '#CA9161',  # Tan
    '#949494',  # Gray
    '#ECE133',  # Yellow
    '#56B4E9',  # Sky blue
]

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

# ---- Extract number from filename for case 1 ----
def extract_number_from_filename(filename, caseflag):
    """
    For case 1, extract zero-padded numbers from filename
    and format as N_n = value
    """
    if caseflag != 1:
        return os.path.basename(filename)
    
    # Look for patterns like _0400, _400, etc.
    match = re.search(r'_0*(\d+)', filename)
    if match:
        number = int(match.group(1))
        return f'$N_n = {number}$'
    else:
        return os.path.basename(filename)

# ---- Compute MAE ----
def compute_mae(times, data, t_ana, y_ana, idx, t_min=None, t_max=None):
    """
    Compute Mean Absolute Error between numerical and analytical solutions
    within the specified time range.
    """
    # Apply time range filter
    if t_min is None:
        t_min = -np.inf
    if t_max is None:
        t_max = np.inf
    
    # Filter numerical data
    mask = (times >= t_min) & (times <= t_max)
    t_filt = times[mask]
    d_filt = data[mask, idx]
    
    if len(t_filt) == 0:
        return np.nan
    
    # Interpolate analytical solution at numerical time points
    y_ana_interp = np.interp(t_filt, t_ana, y_ana[:, idx])
    
    # Compute MAE
    mae = np.mean(np.abs(d_filt - y_ana_interp))
    return mae

# ---- Plotting ----
def make_plots(all_data, all_times, labels, dt, caseflag, outdir="figures", 
               outprefix="grain_bounce", t_min=None, t_max=None):
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

    # Apply user-specified time limits if provided
    plot_t_min = t_min if t_min is not None else 0.0
    plot_t_max = t_max if t_max is not None else global_max_t
    
    # Compute analytical solution
    t_ana = np.arange(0.0, global_max_t + dt, dt)
    y_ana = analytical_solution(t_ana, caseflag)

    saved_files = []
    for gname, comps, indices in groups:
        for ci, comp in enumerate(comps):
            idx = indices[ci]
            fig, ax = plt.subplots(figsize=(8, 4.5))

            # Analytical (plotted first, solid black line, thicker)
            ax.plot(t_ana, y_ana[:,idx], linestyle='-', linewidth=2.2,
                    color='#2d2d2d', label='Analytical solution', zorder=0)

            # Compute MAE for each dataset
            mae_values = []
            
            # Professional boxed axes with ticks
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
            ax.tick_params(which='both', direction='in', top=True, right=True,
                          length=4, width=1.0)

            ax.set_xlim(left=plot_t_min, right=plot_t_max)
            ax.set_xlabel('Time (s)', fontweight='normal')
            ax.set_ylabel(f'{gname} {comp}', fontweight='normal')
            ax.set_title(f'{gname} component {comp}', fontweight='bold', pad=10)
            ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.4, color='#cccccc')
            
            # Numerical datasets with Nature colors
            for i, (data, times, lab) in enumerate(zip(all_data, all_times, labels)):
                if data.size == 0 or times.size == 0:
                    mae_values.append(np.nan)
                    continue
                
                color = NATURE_COLORS[i % len(NATURE_COLORS)]
                ax.plot(times, data[:, idx], linewidth=1.0, label=lab, 
                       color=color, zorder=1, alpha=0.9)
                
                # Compute MAE for this dataset
                mae = compute_mae(times, data, t_ana, y_ana, idx, plot_t_min, plot_t_max)
                mae_values.append(mae)
            ax.legend(loc='best', fontsize=8, frameon=True, 
                     fancybox=False, shadow=False, framealpha=0.95)
            
            # Add MAE inset
            # Position inset in lower right corner
            inset_ax = fig.add_axes([0.65, 0.18, 0.25, 0.25])
            inset_ax.axis('off')
            
            # Create MAE text
            mae_text = "Mean Absolute Error:\n"
            mae_text += "─" * 24 + "\n"
            for i, (lab, mae) in enumerate(zip(labels, mae_values)):
                if not np.isnan(mae):
                    color = NATURE_COLORS[i % len(NATURE_COLORS)]
                    # Use a colored marker to match the plot
                    if len(lab) < 12:
                        mae_text += f"▪ {lab}:  {mae:.3e}\n"
                    else:
                        mae_text += f"▪ {lab}: {mae:.3e}\n"
            
            # Add white background box
            bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='#666666', linewidth=1.0, alpha=0.95)
            inset_ax.text(0.05, 0.95, mae_text, transform=inset_ax.transAxes,
                         fontsize=7, verticalalignment='top', horizontalalignment='left',
                         bbox=bbox_props, family='monospace')

            fig.tight_layout(pad=1.2)
            fname = os.path.join(outdir, f"{outprefix}_{gname.lower()}_{comp}.png".replace(" ", "_"))
            fig.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
            saved_files.append(fname)
            plt.close(fig)

    return saved_files

# ---- Main ----
def main(dt=1.0, caseflag=0, t_min=None, t_max=None):
    """
    Main plotting function
    
    Parameters:
    -----------
    dt : float
        Time step for converting steps to time
    caseflag : int
        Case identifier for analytical solution
    t_min : float, optional
        Minimum time for x-axis (default: 0)
    t_max : float, optional
        Maximum time for x-axis (default: maximum time in data)
    """
    pattern = "dump_*.txt"
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
        # Extract formatted label based on caseflag
        label = extract_number_from_filename(f, caseflag)
        labels.append(label)

    print(f"Found {len(files)} file(s): {files}")
    if t_min is not None or t_max is not None:
        print(f"Time axis limits: [{t_min if t_min is not None else 0.0}, {t_max if t_max is not None else 'max'}]")
    
    saved = make_plots(all_data, all_times, labels, dt, caseflag, 
                      outdir="figures", outprefix="case_"+str(caseflag),
                      t_min=t_min, t_max=t_max)
    print("Saved plots:")
    for s in saved:
        print(" -", s)

if __name__ == "__main__":
    # Example usage with time limits:
    # main(dt=1e-4, caseflag=1, t_min=0.0, t_max=0.01)
    
    # Standard usage:
    main(dt=1.0e-4, caseflag=1, t_max=3.0)

# End of file