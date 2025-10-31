# !/usr/bin/env python3

# Python script that computes the nodes for a 3D sphere
import numpy as np

# Radius sphere
rad = 1.0 

# Generate (near-)uniformly spaced points on a sphere and write them
# to a LAMMPS-like data file "nodes_sphere_numNodes".
numNodes = 100       # Desired number of points
outfile = f"nodes_sphere_{numNodes}"  # output filename

def fibonacci_sphere(n, radius=1.0):
    """Return n points on a sphere of given radius using the Fibonacci (equal-area) lattice."""
    if n <= 0:
        return np.zeros((0, 3))
    if n == 1:
        return np.array([[0.0, 0.0, radius]])
    i = np.arange(0, n)
    # z evenly spaced in [-1,1]
    z = 1.0 - 2.0 * i / (n - 1)
    theta = np.arccos(z)            # polar angle
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    phi = golden_angle * i          # azimuth
    sin_theta = np.sin(theta)
    x = np.cos(phi) * sin_theta
    y = np.sin(phi) * sin_theta
    coords = np.column_stack((x, y, z)) * radius
    return coords

pts = fibonacci_sphere(numNodes, radius=rad)

# -------------------------
# Write output file in same format as original script
# -------------------------
with open(outfile, "w") as f:
    f.write("# Generated using make_nodes_sphere.py (uniform sphere sampling)\n\n")
    f.write(f"{len(pts)} atoms\n")
    f.write(f" 1 atom types\n\n")
    f.write(f"{-rad} {rad} xlo xhi\n")
    f.write(f"{-rad} {rad} ylo yhi\n")
    f.write(f"{-rad} {rad} zlo zhi\n\n")
    f.write("Atoms\n\n")
    for i, (xi, yi, zi) in enumerate(pts):
        f.write(f"{i+1} 1 1 1.0 {xi} {yi} {zi}\n")

print(f"Wrote {len(pts)} nodes for a sphere with R = {rad} to {outfile} using a Fibonacci sequence.")

# End of file