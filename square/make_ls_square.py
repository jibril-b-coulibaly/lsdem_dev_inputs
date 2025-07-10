# Python script that computes level set for a 2D square
import numpy as np

# Use the same values for squares to fit inside Joel hardcoded spheres
rad = 2.5
# Half square side
halfside = rad * np.sqrt(0.5)

nx = 21 # Number of grid points in x-direction, i.e., # intervals + 1
ny = 21 # y-direction
nz = 1 # z-direction
stride = 0.5 # Grid stride

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
            ls_dem_grid[ndx] = np.max([np.abs(delx), np.abs(dely)]) - halfside
            print(ix, iy, iz, delx, dely, delz, xcom, ycom, zcom, halfside, ls_dem_grid[ndx])

grid_min = [-xcom, -ycom, -zcom] # Relative to COM

with open("grid_square.txt", "w") as f:
    f.write(f"# Grid file for a square of side s={2*halfside} \n")
    f.write(f"{nx} {ny} {nz}\n")
    f.write(f"{stride}\n")
    f.write(f"{grid_min[0]} {grid_min[1]} {grid_min[2]}\n")
    for ls in ls_dem_grid:
        f.write(f"{ls}\n")




# Associated script to make a data file for the square

num_atoms_per_side = 10
# We want atoms in the corners, i.e. 4*(n-1) atoms total
num_atoms = 4*(num_atoms_per_side-1)
x = np.zeros(num_atoms)
y = np.zeros(num_atoms)

# We fill in the first n-1 first atoms of the edge and proceed to next edge by rotating
# (There must be a smarter way than building larger arrays at take [:-1], e.g. using np.arange())
xedge = np.linspace(-halfside, halfside, num_atoms_per_side)[:-1]
yedge = -halfside
for i, angle in enumerate(np.linspace(0.0, 2*np.pi, 5)[:-1]):    
    x[i*(num_atoms_per_side-1):(i+1)*(num_atoms_per_side-1)] = np.cos(angle) * xedge - np.sin(angle) * yedge
    y[i*(num_atoms_per_side-1):(i+1)*(num_atoms_per_side-1)] = np.sin(angle) * xedge + np.cos(angle) * yedge


with open("data_square", "w") as f:
    f.write(f"# Generated using make_ls_square.py \n\n")
    f.write(f"{num_atoms} atoms\n")
    f.write(f" 1 atom types\n\n")
    f.write(f"{-halfside} {halfside} xlo xhi\n")
    f.write(f"{-halfside} {halfside} ylo yhi\n")
    f.write(f"{-stride} {stride} zlo zhi\n\n")
    f.write(f"Atoms\n\n")
    for i, (xi, yi) in enumerate(zip(x, y)):
        f.write(f"{i+1} 1 1 1.0 {xi} {yi} 0.0\n")
