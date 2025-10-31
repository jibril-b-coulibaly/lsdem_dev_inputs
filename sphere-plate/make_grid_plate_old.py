# !/usr/bin/env python3

# Python script that computes the level set for a plate
import numpy as np

# Size of the plate LxLxD
L = 5.0*1.0 # 5 times the radius of the sphere
T = 1.0*1.0 # 1 time the radius of the sphere
# Spacing between grid points (cell edge length)
stride = 0.04 # Same as the sphere with R = 1.0 and nx,nz,ny = 53.

# Grid settings
nbuff = 2 # Buffer cells outside zero surface
nx = int(round(L/stride)) + 1 + 2*nbuff # Number of grid points in x-direction, i.e., # intervals + 1
ny = int(round(L/stride)) + 1 + 2*nbuff # y-direction
nz = int(round(T/stride)) + 1 + 2*nbuff # z-direction

# Assume the grid starts at [0,0,0] and place COM at center of grid
xcom = 0.5 * (nx - 1) * stride
ycom = 0.5 * (ny - 1) * stride
zcom = 0.5 * (nz - 1) * stride

ls_dem_grid = np.zeros(nx * ny * nz)
# Traverse grid along x, then y, then z
for iz, delz in enumerate(np.linspace(-0.5*T-nbuff*stride, 0.5*T+nbuff*stride, nz)):
    for iy, dely in enumerate(np.linspace(-0.5*L-nbuff*stride, 0.5*L+nbuff*stride, ny)):
        for ix, delx in enumerate(np.linspace(-0.5*L-nbuff*stride, 0.5*L+nbuff*stride, nx)):
            ndx = ix + iy * nx + iz * nx * ny
            #delx = ix * stride - xcom
            #dely = iy * stride - ycom
            #delz = iz * stride - zcom
            lsx = np.abs(delx) - 0.5*L
            lsy = np.abs(dely) - 0.5*L
            lsz = np.abs(delz) - 0.5*T
            ls_out = np.sqrt(np.maximum(lsx,0.0)**2 + np.maximum(lsy,0.0)**2 + np.maximum(lsz,0.0)**2)
            if (ls_out > 0.0):
                ls_dem_grid[ndx] = ls_out
            else:
                ls_dem_grid[ndx] = np.max([lsx,lsy,lsz])

grid_min = [-xcom, -ycom, -zcom] # Relative to COM

with open("grid_plate.txt", "w") as f:
    f.write(f"# Grid file for a plate L={L} and T={T} \n")
    f.write(f"{nx} {ny} {nz}\n")
    f.write(f"{stride}\n")
    f.write(f"{grid_min[0]} {grid_min[1]} {grid_min[2]}\n")
    for ls in ls_dem_grid:
        f.write(f"{ls}\n")

print(f"Wrote plate grid file with edge lengths L = {L}, thickness T = {T}, and grid spacing {stride}.")

# End of file



# ---- robust construction of coordinates matching ndx ordering ----
# Create integer index arrays with the same ordering used to fill ls_dem_grid
ixs, iys, izs = np.indices((nx, ny, nz))  # shapes (nx,ny,nz)
# ravel in C-order to match ix + iy*nx + iz*nx*ny
ixf = ixs.ravel(order='C')   # length N = nx*ny*nz
iyf = iys.ravel(order='C')
izf = izs.ravel(order='C')

# node positions (node-centred) used in your delx formula:
X_nodes = (ixf - 0.5*(nx-1)) * stride
Y_nodes = (iyf - 0.5*(ny-1)) * stride
Z_nodes = (izf - 0.5*(nz-1)) * stride

# cell-centred positions (recommended for volume integration)
X_cells = (ixf + 0.5 - 0.5*nx) * stride   # (ix + 0.5) - nx/2  => symmetrical
Y_cells = (iyf + 0.5 - 0.5*ny) * stride
Z_cells = (izf + 0.5 - 0.5*nz) * stride

# ---- decide inside-mask (sign convention: inside where SDF <= 0) ----
inside_mask = ls_dem_grid <= 0.0   # boolean array of length N

# diagnostic counts and left/right balance
def balance_stats(coords, mask, axis='x'):
    coord = coords
    pos = np.sum(mask & (coord > 0.0))
    neg = np.sum(mask & (coord < 0.0))
    zero = np.sum(mask & (np.isclose(coord, 0.0, atol=1e-12)))
    return pos, neg, zero

print("Counts of inside voxels / nodes (total):", np.sum(inside_mask))
print("Node-based balance (x pos/neg/zero):", balance_stats(X_nodes, inside_mask))
print("Node-based balance (y pos/neg/zero):", balance_stats(Y_nodes, inside_mask))
print("Node-based balance (z pos/neg/zero):", balance_stats(Z_nodes, inside_mask))
print("Cell-centre balance (x pos/neg/zero):", balance_stats(X_cells, inside_mask))
print("Cell-centre balance (y pos/neg/zero):", balance_stats(Y_cells, inside_mask))
print("Cell-centre balance (z pos/neg/zero):", balance_stats(Z_cells, inside_mask))

# ---- compute inertia tensor using either node positions or cell-centres ----
rho = 2500.0      # optional density; if you only need dimensionless inertia set rho=1
dV = stride**3    # voxel volume

def inertia_from_points(X, Y, Z, mask, density=1.0, vol=dV):
    m = density * vol
    xm = X[mask]
    ym = Y[mask]
    zm = Z[mask]
    # diagonal terms
    Ixx = np.sum(m * (ym**2 + zm**2))
    Iyy = np.sum(m * (xm**2 + zm**2))
    Izz = np.sum(m * (xm**2 + ym**2))
    # products of inertia (note sign convention: Ixy = -∑ m x y)
    Ixy = -np.sum(m * xm * ym)
    Ixz = -np.sum(m * xm * zm)
    Iyz = -np.sum(m * ym * zm)
    I = np.array([[Ixx, Ixy, Ixz],
                  [Ixy, Iyy, Iyz],
                  [Ixz, Iyz, Izz]])
    return I

I_nodes = inertia_from_points(X_nodes, Y_nodes, Z_nodes, inside_mask, density=1.0)
I_cells = inertia_from_points(X_cells, Y_cells, Z_cells, inside_mask, density=1.0)

print("Inertia (node-sampling):\n", I_nodes)
print("Inertia (cell-centre sampling):\n", I_cells)
print("Off-diagonal ratios: nodes {:.3e}, cells {:.3e}".format(
    np.linalg.norm(I_nodes - np.diag(np.diag(I_nodes))) / np.linalg.norm(np.diag(I_nodes)),
    np.linalg.norm(I_cells - np.diag(np.diag(I_cells))) / np.linalg.norm(np.diag(I_cells)))
)