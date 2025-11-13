# !/usr/bin/env python3

# Python script that computes the level set for a 3D sphere
import numpy as np

# Radius sphere
rad = 1.0 

# Grid settings
nintervals = 50
nbuff = 1 # Buffer cells outside zero surface
nx = nintervals + 1 + 2*nbuff # Number of grid points in x-direction, i.e., # intervals + 1
ny = nintervals + 1 + 2*nbuff # y-direction
nz = nintervals + 1 + 2*nbuff # z-direction
stride = 2*rad/nintervals # Spacing between grid points (cell edge length)

# Assume the grid starts at [0,0,0] and place COM at center of grid
xcom = 0.5 * (nx - 1) * stride
ycom = 0.5 * (ny - 1) * stride
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

print(f"Wrote sphere grid file with R = {rad} and grid spacing {stride}.")

# End of file