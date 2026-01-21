# Python script that computes level set for a 2D disk
import numpy as np

# Use the same values Joel hardcoded pair_ls_dem
rad = 2.5

nx = 21 # Number of grid points in x-direction, i.e., # intervals + 1
ny = 21 # y-direction
stride = 0.5 # Grid stride

# Assume the grid starts at [0,0,0] and place COM at center of grid
xcom = 0.5 * (nx - 1) * stride # I don't get the 5.25 from Joel
ycom = 0.5 * (ny - 1) * stride # not sure it matters because we have room

ls_dem_grid = np.zeros(nx * ny)
# Traverse grid along x, then y
for iy in range(ny):
    for ix in range(nx):
        ndx = ix + iy * nx
        delx = ix * stride - xcom
        dely = iy * stride - ycom
        ls_dem_grid[ndx] = np.sqrt(delx*delx + dely*dely) - rad

grid_min = [-xcom, -ycom] # Relative to COM

with open("grid_disk.txt", "w") as f:
    f.write(f"# Grid file for a disk r={rad} \n")
    f.write(f"{nx} {ny}\n")
    f.write(f"{stride}\n")
    f.write(f"{grid_min[0]} {grid_min[1]}\n")
    for ls in ls_dem_grid:
        f.write(f"{ls}\n")
