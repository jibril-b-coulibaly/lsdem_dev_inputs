# Python script that computes level set for a 3D cube
import numpy as np

# Use the same values for squares to fit inside Joel hardcoded spheres
rad = 2.5
# Half square side
halfside = rad / np.sqrt(3)

nx = 20 # Number of grid points in x-direction, i.e., # intervals + 1
ny = 20 # y-direction
nz = 20 # z-direction
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
            ls_dem_grid[ndx] = np.max([np.abs(delx), np.abs(dely), np.abs(delz)]) - halfside

grid_min = [-xcom, -ycom, -zcom] # Relative to COM

with open("grid_cube.txt", "w") as f:
    f.write(f"# Grid file for a cube of side s={2*halfside} \n")
    f.write(f"{nx} {ny} {nz}\n")
    f.write(f"{stride}\n")
    f.write(f"{grid_min[0]} {grid_min[1]} {grid_min[2]}\n")
    for ls in ls_dem_grid:
        f.write(f"{ls}\n")




# Associated script to make a data file for the cube

num_atoms_per_side = 10
# We want atoms in the corners, i.e.
# 8 atoms in the corners
# n-2 atoms per edge (corner excluded)
# (n-2)*(n-2) atoms on faces (edges and corner excluded)
num_atoms = 8 + 12*(num_atoms_per_side-2) + 6 *(num_atoms_per_side-2)*(num_atoms_per_side-2)
x = np.zeros(num_atoms)
y = np.zeros(num_atoms)
z = np.zeros(num_atoms)

# Corners
x[0:8] = halfside * np.array([-1, 1, 1, -1, -1, 1, 1, -1])
y[0:8] = halfside * np.array([-1, -1, 1, 1, -1, -1, 1, 1])
z[0:8] = halfside * np.array([-1, -1, -1, -1, 1, 1, 1, 1])

# Edges
edge1 = np.linspace(-halfside, halfside, num_atoms_per_side)[1:-1]
edge2 = [-halfside, -halfside, halfside, halfside]
edge3 = [-halfside, halfside, halfside, -halfside]
for i in range(3):
    for ie in range(4):
        ndx = ie + 4*i
        x[8 + (num_atoms_per_side-2)*ndx: 8 + (num_atoms_per_side-2)*(ndx+1)] = edge1 * (i==0) + edge3[ie] * (i==1) + edge2[ie] * (i==2)
        y[8 + (num_atoms_per_side-2)*ndx: 8 + (num_atoms_per_side-2)*(ndx+1)] = edge2[ie] * (i==0) + edge1 * (i==1) + edge3[ie] * (i==2)
        z[8 + (num_atoms_per_side-2)*ndx: 8 + (num_atoms_per_side-2)*(ndx+1)] = edge3[ie] * (i==0) + edge2[ie] * (i==1) + edge1 * (i==2)

# Faces
for i in range(3):
    for i3, x3 in enumerate([-halfside, halfside]):
        for i2, x2 in enumerate(edge1):
            for i1, x1 in enumerate(edge1):
                ndx = i1 + (num_atoms_per_side-2)*i2 + (num_atoms_per_side-2)*(num_atoms_per_side-2)*i3 + 2*(num_atoms_per_side-2)*(num_atoms_per_side-2)*i
                x[8 + (num_atoms_per_side-2)*12 + ndx] = x1 * (i==0) + x2 * (i==1) + x3 * (i==2)
                y[8 + (num_atoms_per_side-2)*12 + ndx] = x2 * (i==0) + x3 * (i==1) + x1 * (i==2)
                z[8 + (num_atoms_per_side-2)*12 + ndx] = x3 * (i==0) + x1 * (i==1) + x2 * (i==2)



with open("data_cube", "w") as f:
    f.write(f"# Generated using make_ls_cube.py \n\n")
    f.write(f"{num_atoms} atoms\n")
    f.write(f" 1 atom types\n\n")
    f.write(f"{-halfside} {halfside} xlo xhi\n")
    f.write(f"{-halfside} {halfside} ylo yhi\n")
    f.write(f"{-halfside} {halfside} zlo zhi\n\n")
    f.write(f"Atoms\n\n")
    for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
        f.write(f"{i+1} 1 1 1.0 {xi} {yi} {zi}\n")
