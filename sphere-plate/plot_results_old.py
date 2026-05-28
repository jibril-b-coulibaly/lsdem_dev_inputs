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

# SI Units for each group
SI_UNITS = {
    "Coordinate": "m",
    "Velocity": "m/s",
    "Force": "N",
    "Torque": "N·m",
}

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

# ---- Extract number from label ----
def extract_number_from_label(label):
    """
    Extract the numerical value from labels like '$N_n = 400$'
    Returns None if no number found
    """
    match = re.search(r'N_n = (\d+)', label)
    if match:
        return int(match.group(1))
    return None

# ---- Compute MAE ----
def compute_mae(steps, data, y_ana_full, idx, t_min=None, t_max=None, dt=1.0):
    """
    Compute Mean Absolute Error between numerical and analytical solutions
    within the specified time range, skipping the first data point.
    
    Parameters:
    -----------
    steps : array
        Step numbers from simulation file
    data : array
        Numerical data array
    y_ana_full : array
        Full analytical solution at every time step
    idx : int
        Component index to compare
    t_min : float, optional
        Minimum time for filtering
    t_max : float, optional
        Maximum time for filtering
    dt : float
        Time step
    """
    # Convert steps to time
    times = steps * dt
    
    # Apply time range filter
    if t_min is None:
        t_min = -np.inf
    if t_max is None:
        t_max = np.inf
    
    # Filter numerical data (skip first point)
    mask = (times >= t_min) & (times <= t_max)
    if np.sum(mask) > 1:
        # Skip the first valid point
        valid_indices = np.where(mask)[0][1:]  # Skip index 0 of valid points
        if len(valid_indices) == 0:
            return np.nan
    else:
        return np.nan
    
    # Get filtered data
    steps_filt = steps[valid_indices]
    d_filt = data[valid_indices, idx]
    
    # Sample analytical solution at the exact step indices
    # Since simulation outputs every N steps, we need to sample at those exact steps
    step_indices = steps_filt.astype(int)
    y_ana_sampled = y_ana_full[step_indices, idx]
    
    # Compute MAE
    mae = np.mean(np.abs(d_filt - y_ana_sampled))
    return mae

# ---- Plotting ----
def make_plots(all_data, all_times, all_steps, labels, dt, caseflag, outdir="figures", 
               outprefix="grain_bounce", t_min=None, t_max=None, fig_width=8, fig_height=4.5,
               show_title=True):
    os.makedirs(outdir, exist_ok=True)

    groups = [
        ("Coordinate",  ["x","y","z"], [0,1,2]),
        ("Velocity",    ["x","y","z"], [3,4,5]),
        ("Force",       ["x","y","z"], [6,7,8]),
        ("Torque",      ["x","y","z"], [9,10,11]),
    ]

    # Determine global maximum time
    global_max_t = 0.0
    global_max_step = 0
    for t, steps in zip(all_times, all_steps):
        if t.size > 0:
            global_max_t = max(global_max_t, float(np.max(t)))
            global_max_step = max(global_max_step, int(np.max(steps)))
    if global_max_t <= 0.0:
        global_max_t = dt  # fallback
        global_max_step = 1

    # Apply user-specified time limits if provided
    plot_t_min = t_min if t_min is not None else 0.0
    plot_t_max = t_max if t_max is not None else global_max_t
    
    # Compute analytical solution at full resolution (every step)
    steps_full = np.arange(0, global_max_step + 1)
    t_ana_full = steps_full * dt
    y_ana_full = analytical_solution(t_ana_full, caseflag)

    # Dictionary to store all MAE values for convergence plot
    # Structure: mae_data[label][group_name][component] = mae_value
    mae_data = {label: {} for label in labels}

    saved_files = []
    for gname, comps, indices in groups:
        # Initialize MAE storage for this group
        for label in labels:
            mae_data[label][gname] = {}
        
        for ci, comp in enumerate(comps):
            idx = indices[ci]
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # Set up grid first (so it's behind everything)
            ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6, color='#999999', zorder=0)

            # Analytical (plotted next, solid black line, thicker)
            ax.plot(t_ana_full, y_ana_full[:,idx], linestyle='-', linewidth=2.2,
                    color='#2d2d2d', label='Analytical', zorder=1) # 'Analytical solution'

            # Compute MAE for each dataset to determine padding
            mae_values = []
            for data, times, steps_arr, lab in zip(all_data, all_times, all_steps, labels):
                if data.size == 0 or times.size == 0:
                    mae_values.append(np.nan)
                else:
                    mae = compute_mae(steps_arr, data, y_ana_full, idx, plot_t_min, plot_t_max, dt)
                    mae_values.append(mae)
                    # Store MAE for convergence plot
                    mae_data[lab][gname][comp] = mae
            
            # Find maximum number of digits before decimal for alignment
            max_exp = -999
            for mae in mae_values:
                if not np.isnan(mae):
                    exp = int(np.floor(np.log10(mae))) if mae > 0 else 0
                    max_exp = max(max_exp, exp)
            
            # Determine padding needed
            if max_exp >= -2:  # MAE >= 0.01
                # Need more padding for larger numbers
                spacing = "   "
            else:
                spacing = "  "
            
            # Get viridis colormap
            cmap = plt.cm.viridis
            n_datasets = len([d for d in all_data if d.size > 0])
            colors = [cmap(i / max(1, n_datasets - 1)) for i in range(n_datasets)]
            
            # Numerical datasets with viridis colors
            color_idx = 0
            for data, times, steps_arr, lab, mae in zip(all_data, all_times, all_steps, labels, mae_values):
                if data.size == 0 or times.size == 0:
                    continue
                
                color = colors[color_idx]
                color_idx += 1
                
                # if len(lab) < 12:
                #     spac = " "
                # else:
                #     spac = ""

                # # Create legend label with aligned MAE
                # if not np.isnan(mae):
                #     label_with_mae = f'{lab}, {spac}MAE = {mae:.3e}'
                # else:
                #     label_with_mae = f'{lab}, {spac}MAE = N/A'

                label_with_mae = lab
                
                ax.plot(times, data[:, idx], linewidth=1.0, label=label_with_mae, 
                       color=color, zorder=2, alpha=0.9)

            # Professional boxed axes with ticks
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
            ax.tick_params(which='both', direction='in', top=True, right=True,
                          length=4, width=1.0)

            ax.set_xlim(left=plot_t_min, right=plot_t_max)
            ax.set_xlabel('Time (s)', fontweight='normal')
            ax.set_ylabel(f'{gname} {comp} ({SI_UNITS[gname]})', fontweight='normal')
            if show_title:
                ax.set_title(f'{gname} component {comp}', fontweight='bold', pad=10)
            ax.legend(loc='best', fontsize=8, frameon=True, 
                     fancybox=False, shadow=False, framealpha=0.95)

            fig.tight_layout(pad=1.2)
            fname = os.path.join(outdir, f"{outprefix}_{gname.lower()}_{comp}.png".replace(" ", "_"))
            fig.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
            saved_files.append(fname)
            plt.close(fig)

    return saved_files, mae_data

# ---- Plot MAE convergence ----
def plot_mae_convergence(mae_data, labels, outdir="figures", outprefix="grain_bounce",
                        fig_width=8, fig_height=5):
    """
    Plot MAE vs number of nodes for all components
    """
    # Extract node numbers from labels
    node_numbers = []
    for label in labels:
        num = extract_number_from_label(label)
        if num is not None:
            node_numbers.append(num)
        else:
            # Cannot create convergence plot without node numbers
            print("Warning: Could not extract node numbers from labels. Skipping convergence plot.")
            return None
    
    if len(node_numbers) == 0:
        return None
    
    # Organize data for plotting
    groups = [
        ("Coordinate",  ["x","y","z"]),
        ("Velocity",    ["x","y","z"]),
        ("Force",       ["x","y","z"]),
        ("Torque",      ["x","y","z"]),
    ]
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Set up grid
    ax.grid(True, linestyle=':', linewidth=0.7, alpha=0.6, color='#999999', zorder=0, which='both')
    
    # Get viridis colormap for 12 components
    cmap = plt.cm.viridis
    colors = [cmap(i / 11) for i in range(12)]
    
    # Pre-determined marker cycle
    markers = ['o', 's', '^']   # circle, square, triangle

    # Mapping from group name to compact label prefix
    label_prefix = {
        "Coordinate": "",
        "Velocity": "v_",
        "Force": "F_",
        "Torque": "T_",
    }

    # Plot each component
    color_idx = 0
    for gname, comps in groups:
        prefix = label_prefix.get(gname, "")
        for comp in comps:
            mae_values = []
            for label in labels:
                if gname in mae_data[label] and comp in mae_data[label][gname]:
                    mae = mae_data[label][gname][comp]
                    if not np.isnan(mae):
                        mae_values.append(mae)
                    else:
                        mae_values.append(None)
                else:
                    mae_values.append(None)
            
            # Filter out None values
            valid_nodes = []
            valid_mae = []
            for n, m in zip(node_numbers, mae_values):
                if m is not None:
                    valid_nodes.append(n)
                    valid_mae.append(m)
            
            if len(valid_nodes) > 0:
                # build compact label (avoid leading underscore when prefix == "")
                compact_label = f"{prefix}{comp}" if prefix else comp

                ax.plot(
                    valid_nodes,
                    valid_mae,
                    marker=markers[color_idx % len(markers)],
                    markersize=5,
                    linewidth=1.5,
                    label=compact_label,
                    color=colors[color_idx],
                    alpha=0.9,
                    zorder=2
                )

            color_idx += 1
    
    # Professional boxed axes with ticks
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax.tick_params(which='both', direction='in', top=True, right=True,
                  length=4, width=1.0)
    
    ax.set_xlabel('Number of nodes ($N_n$)', fontweight='normal')
    ax.set_ylabel('Mean Absolute Error', fontweight='normal')
    #ax.set_title('MAE Convergence vs Resolution', fontweight='bold', pad=10)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(loc='best', fontsize=7, frameon=True, 
             fancybox=False, shadow=False, framealpha=0.95, ncol=2)
    
    fig.tight_layout(pad=1.2)
    fname = os.path.join(outdir, f"{outprefix}_mae_convergence.png")
    fig.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return fname

# ---- Main ----
def main(dt=1.0, caseflag=0, t_min=None, t_max=None, fig_width=8, fig_height=4.5, show_title=True):
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
    fig_width : float, optional
        Figure width in inches (default: 8)
    fig_height : float, optional
        Figure height in inches (default: 4.5)
    show_title : bool, optional
        Whether to show plot titles (default: True)
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
    all_steps = []
    labels = []
    for f in files:
        steps, d = parse_file(f)
        times = steps * dt
        all_times.append(times)
        all_steps.append(steps)
        all_data.append(d)
        # Extract formatted label based on caseflag
        label = extract_number_from_filename(f, caseflag)
        labels.append(label)

    print(f"Found {len(files)} file(s): {files}")
    if t_min is not None or t_max is not None:
        print(f"Time axis limits: [{t_min if t_min is not None else 0.0}, {t_max if t_max is not None else 'max'}]")
    
    saved, mae_data = make_plots(all_data, all_times, all_steps, labels, dt, caseflag, 
                                 outdir="figures", outprefix="case_"+str(caseflag),
                                 t_min=t_min, t_max=t_max, fig_width=fig_width, 
                                 fig_height=fig_height, show_title=show_title)
    print("Saved plots:")
    for s in saved:
        print(" -", s)
    
    # Create MAE convergence plot
    convergence_plot = plot_mae_convergence(mae_data, labels, 
                                           outdir="figures", 
                                           outprefix="case_"+str(caseflag),
                                           fig_width=fig_width, fig_height=fig_height)
    if convergence_plot:
        print(f"Saved convergence plot:\n - {convergence_plot}")

if __name__ == "__main__":
    # Can provide dt, caseflag, t_min, t_max, fig_width, fig_height, show_title.
    main(dt=1.0e-4, caseflag=1, t_max=3.0, fig_width=4.0, fig_height=3.0, show_title=False)

# End of file