# !/usr/bin/env python3

# Python script that computes the nodes for a plate with a normal in the z-direction
import numpy as np

# Size of the plate
L = 5.0*1.0 # 5 times the radius of the sphere
T = 1.0*1.0 # 1 time the radius of the sphere
# Number of points that we had for the sphere (to compute spacing)
equivNumNodes = 100       
# Spacing between nodes (edge length in HCP lattice). Last number is R.
dist = np.sqrt( 8.0 * np.pi / (np.sqrt(3) * equivNumNodes) ) * 1.0 # Set such that area per node for plate and sphere are equal

# Generate (near-)uniformly spaced points on a plate and write them
# to a LAMMPS-like data file "nodes_plate_numNodes".
outfile = f"nodes_plate_{equivNumNodes}"  # output filename

def hex_on_rect_xy(z, L, dist):
    """Hex lattice on rectangle in the x-y plane at height z (centre at (0,0,z))."""
    a1 = np.array([dist, 0.0, 0.0])
    a2 = np.array([dist/2.0, dist*np.sqrt(3)/2.0, 0.0])
    n = int(np.ceil(L / dist)) + 2
    xmin, xmax = -0.5*L, 0.5*L
    ymin, ymax = -0.5*L, 0.5*L
    pts = []
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            r = i*a1 + j*a2
            x, y = r[0], r[1]
            if (xmin <= x <= xmax) and (ymin <= y <= ymax):
                pts.append((x, y, z))
    return pts

def hex_on_rect_yz(x, L, T, dist):
    """Hex lattice on rectangle spanned by y (length L) and z (length T) at plane x (centre at (x,0,0))."""
    a1 = np.array([0.0, dist, 0.0])                     # step along +y
    a2 = np.array([0.0, dist/2.0, dist*np.sqrt(3)/2.0]) # y/ z combination
    n = int(np.ceil(max(L, T) / dist)) + 2
    ymin, ymax = -0.5*L, 0.5*L
    zmin, zmax = -0.5*T, 0.5*T
    pts = []
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            r = i*a1 + j*a2
            y, z = r[1], r[2]
            if (ymin <= y <= ymax) and (zmin <= z <= zmax):
                pts.append((x, y, z))
    return pts

def hex_on_rect_xz(y, L, T, dist):
    """Hex lattice on rectangle spanned by x (length L) and z (length T) at plane y (centre at (0,y,0))."""
    a1 = np.array([dist, 0.0, 0.0])                     # step along +x
    a2 = np.array([dist/2.0, 0.0, dist*np.sqrt(3)/2.0]) # x/ z combination
    n = int(np.ceil(max(L, T) / dist)) + 2
    xmin, xmax = -0.5*L, 0.5*L
    zmin, zmax = -0.5*T, 0.5*T
    pts = []
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            r = i*a1 + j*a2
            x, z = r[0], r[2]
            if (xmin <= x <= xmax) and (zmin <= z <= zmax):
                pts.append((x, y, z))
    return pts

# Collect points from faces
all_pts = []

# top and bottom: z = ±T/2, span L x L
all_pts += hex_on_rect_xy( 0.5*T, L, dist)
all_pts += hex_on_rect_xy(-0.5*T, L, dist)

# side faces at x = ±L/2 (span y x z = L x T)
all_pts += hex_on_rect_yz( 0.5*L, L, T, dist)
all_pts += hex_on_rect_yz(-0.5*L, L, T, dist)

# side faces at y = ±L/2 (span x x z = L x T)
all_pts += hex_on_rect_xz( 0.5*L, L, T, dist)
all_pts += hex_on_rect_xz(-0.5*L, L, T, dist)

# Deduplicate points with a rounding tolerance to avoid floating point near-duplicates
if len(all_pts) == 0:
    pts = np.empty((0,3))
else:
    arr = np.array(all_pts)
    # round to avoid tiny fp differences (12 decimals is usually safe)
    rounded = np.round(arr, decimals=12)
    uniq = np.unique(rounded, axis=0)
    # sort deterministically by x, y, z
    order = np.lexsort((uniq[:,2], uniq[:,1], uniq[:,0]))
    pts = uniq[order]

# -------------------------
# Write output file in same format as original script
# -------------------------
with open(outfile, "w") as f:
    f.write("# Generated using make_nodes_plate.py (hexagonal lattice sampling)\n\n")
    f.write(f"{len(pts)} atoms\n")
    f.write(f" 1 atom types\n\n")
    f.write(f"{-0.5*L} {0.5*L} xlo xhi\n")
    f.write(f"{-0.5*L} {0.5*L} ylo yhi\n")
    f.write(f"{-0.5*T} {0.5*T} zlo zhi\n\n")
    f.write("Atoms\n\n")
    for i, (xi, yi, zi) in enumerate(pts):
        f.write(f"{i+1} 1 1 1.0 {xi} {yi} {zi}\n")

print(f"Wrote {len(pts)} nodes for a plate with side length L = {L} and thickness T = {T} to {outfile} using a hexagonal lattice pattern.")

# End of file

"""
# --- tiny 3D visual check (append this to your script) ---
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
import numpy as np

def set_axes_equal(ax):
    #Make 3D axes have equal scale.
    #Source idea: compute max range across axes and set limits accordingly.

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    max_range = 0.5 * max(x_range, y_range, z_range)

    ax.set_xlim3d(x_middle - max_range, x_middle + max_range)
    ax.set_ylim3d(y_middle - max_range, y_middle + max_range)
    ax.set_zlim3d(z_middle - max_range, z_middle + max_range)

def plot_pts_3d(pts, show=True, save_path=None, marker='o', ms=4):
    #Quick 3D scatter of pts (N,3). Colours by z to help identify faces.
    #- pts: numpy array (N,3)
    #- save_path: optional filename (PNG) to save the figure
    
    if pts.size == 0:
        raise ValueError("pts is empty")

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    xs, ys, zs = pts[:,0], pts[:,1], pts[:,2]

    sc = ax.scatter(xs, ys, zs, c=zs, cmap='viridis', marker=marker, s=ms, depthshade=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(f'Hex lattice nodes (N={len(pts)})')

    set_axes_equal(ax)
    fig.colorbar(sc, ax=ax, label='z coordinate')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

# Example usage (uncomment preferred line):
plot_pts_3d(pts)                     # show interactive window
# plot_pts_3d(pts, save_path='nodes_plate.png', show=False)  # save to file without showing
"""