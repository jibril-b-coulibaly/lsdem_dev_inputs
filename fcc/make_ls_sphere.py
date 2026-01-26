#!/usr/bin/env python3
# # Python script that computes level set for a 3D sphere
import numpy as np
from math import sqrt, pi

# Copy settings from FCC test in van der Haven et al. (2023)
rad = 0.12

# Set this such that nx, ny, nz = 25
nintervals = 22
nbuff = 1
nx = nintervals + 1 + 2*nbuff # Number of grid points in x-direction, i.e., # intervals + 1
ny = nintervals + 1 + 2*nbuff # y-direction
nz = nintervals + 1 + 2*nbuff
stride = 2*rad/nintervals # Grid stride 0.5

# Assume the grid starts at [0,0,0] and place COM at center of grid
xcom = 0.5 * (nx - 1) * stride # I don't get the 5.25 from Joel
ycom = 0.5 * (ny - 1) * stride # not sure it matters because we have room
zcom = 0.5 * (nz - 1) * stride

ls_dem_grid = np.zeros(nx * ny * nz)
# Traverse grid along x, then y, then z
for iz in range(nz):
    for iy in range(ny):
        for ix in range(nx):
            ndx = ix + iy * nx + iz * nx * ny
            delx = ix * stride - xcom
            dely = iy * stride - ycom
            delz = iz * stride - zcom
            ls_dem_grid[ndx] = np.sqrt(delx*delx + dely*dely + delz*delz) - rad

grid_min = [-xcom, -ycom, -zcom] # Relative to COM

with open("grid_sphere.txt", "w") as f:
    f.write(f"# Grid file for a disk r={rad} \n")
    f.write(f"{nx} {ny} {nz}\n")
    f.write(f"{stride}\n")
    f.write(f"{grid_min[0]} {grid_min[1]} {grid_min[2]}\n")
    for ls in ls_dem_grid:
        f.write(f"{ls}\n")



# Associated script to make a data file for the sphere

"""
Generate (near-)uniformly spaced points on a sphere and write them
to a LAMMPS-like data file "data_sphere".

Two methods:
 - "fibonacci": Fibonacci / equal-area lattice (works for any N)
 - "icosa": subdivided icosahedron (very uniform, constrained N)
"""

# -------------------------
# User parameters
# -------------------------
num_points = 512       # desired number of points (used directly for fibonacci)
outfile = "data_sphere_512"  # output filename
# -------------------------

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
    golden_angle = pi * (3.0 - sqrt(5.0))
    phi = golden_angle * i          # azimuth
    sin_theta = np.sin(theta)
    x = np.cos(phi) * sin_theta
    y = np.sin(phi) * sin_theta
    coords = np.column_stack((x, y, z)) * radius
    return coords

pts = fibonacci_sphere(num_points, radius=rad)

# -------------------------
# Write output file in same format as original script
# -------------------------
with open(outfile, "w") as f:
    f.write("# Generated using make_ls_sphere.py (uniform sphere sampling)\n\n")
    f.write(f"{len(pts)} atoms\n")
    f.write(f" 1 atom types\n\n")
    f.write(f"{-rad} {rad} xlo xhi\n")
    f.write(f"{-rad} {rad} ylo yhi\n")
    f.write(f"{-rad} {rad} zlo zhi\n\n")
    f.write("Atoms\n\n")
    for i, (xi, yi, zi) in enumerate(pts):
        f.write(f"{i+1} 0 1 {xi} {yi} {zi}\n")

print(f"Wrote {len(pts)} points to {outfile} using method='{method}'.")
