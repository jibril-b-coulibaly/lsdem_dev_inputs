import numpy as np

# -------------------------
# Parameters
# -------------------------
# Size of the plate LxLxD
L = 5.0*1.0 # 5 times the radius of the sphere
T = 1.0*1.0 # 1 time the radius of the sphere
# Spacing between grid points (cell edge length)
stride = 0.04 # Same as the sphere with R = 1.0 and nx,nz,ny = 53.
nbuff = 1

nx = int(round(L/stride)) + 1 + 2*nbuff
ny = int(round(L/stride)) + 1 + 2*nbuff
nz = int(round(T/stride)) + 1 + 2*nbuff

# -------------------------
# Cell-centre coordinates (x-fast ordering)
# -------------------------
grid_min_x = - 0.5 * (nx - 1) * stride
grid_min_y = - 0.5 * (ny - 1) * stride
grid_min_z = - 0.5 * (nz - 1) * stride
#grid_min_x = -0.5 * L - nbuff * stride #+ 0.5 * stride
#grid_min_y = -0.5 * L - nbuff * stride #+ 0.5 * stride
#grid_min_z = -0.5 * T - nbuff * stride #+ 0.5 * stride

xs = grid_min_x + np.arange(nx) * stride
ys = grid_min_y + np.arange(ny) * stride
zs = grid_min_z + np.arange(nz) * stride

N = nx * ny * nz
Xc = np.empty(N, dtype=float)
Yc = np.empty(N, dtype=float)
Zc = np.empty(N, dtype=float)

#ls_dem_grid = np.zeros(nx*ny*nz)

idx = 0
for iz in range(nz):
    z = zs[iz]
    for iy in range(ny):
        y = ys[iy]
        for ix in range(nx):
            x = xs[ix]
            Xc[idx] = x
            Yc[idx] = y
            Zc[idx] = z

            # halfL = 0.5 * L
            # halfT = 0.5 * T

            # dx = np.abs(x) - halfL
            # dy = np.abs(y) - halfL
            # dz = np.abs(z) - halfT

            # dxp = np.maximum(dx, 0.0)
            # dyp = np.maximum(dy, 0.0)
            # dzp = np.maximum(dz, 0.0)

            # outside_dist = np.sqrt(dxp*dxp + dyp*dyp + dzp*dzp)
            # inside_val = np.maximum(np.maximum(dx, dy), dz)

            # if outside_dist > 0.0:
            #     ls_dem_grid[idx] = outside_dist
            # else:
            #     ls_dem_grid[idx] = inside_val

            idx += 1

# -------------------------
# SDF for axis-aligned box (cell-centres)
# -------------------------
halfL = 0.5 * L
halfT = 0.5 * T

dx = np.abs(Xc) - halfL
dy = np.abs(Yc) - halfL
dz = np.abs(Zc) - halfT

dxp = np.maximum(dx, 0.0)
dyp = np.maximum(dy, 0.0)
dzp = np.maximum(dz, 0.0)

outside_dist = np.sqrt(dxp*dxp + dyp*dyp + dzp*dzp)
inside_val = np.maximum(np.maximum(dx, dy), dz)

ls_dem_grid = np.where(outside_dist > 0.0, outside_dist, inside_val)   # φ flattened (x-fast)

#test = ls_dem_grid - ls_dem_grid2
#print(np.sum(abs(test)))

# -------------------------
# Write grid file
# -------------------------
with open("grid_plate.txt", "w") as f:
    f.write(f"# Grid file for a plate L={L} and T={T} \n")
    f.write(f"{nx} {ny} {nz}\n")
    f.write(f"{stride}\n")
    f.write(f"{grid_min_x} {grid_min_y} {grid_min_z}\n")
    for ls in ls_dem_grid:
        f.write(f"{ls}\n")

print(f"Wrote plate grid file with edge lengths L = {L}, thickness T = {T}, and grid spacing {stride}.")

# End of file

# # -------------------------
# # Smeared Heaviside H(phi) and occupancy w = 1 - H
# # -------------------------
# def smeared_heaviside(phi, eps):
#     phi = np.asarray(phi, dtype=float)
#     H = np.empty_like(phi)
#     neg = phi <= -eps
#     pos = phi >= eps
#     trans = ~(neg | pos)
#     H[neg] = 0.0
#     H[pos] = 1.0
#     if np.any(trans):
#         p = phi[trans] / eps
#         H[trans] = 0.5 * (1.0 + p + np.sin(np.pi * p) / np.pi)
#     return H

# eps = np.sqrt(0.75) * stride / 1.5   # smoothing half-width (tune if you need)
# H = smeared_heaviside(ls_dem_grid, eps)
# w = 1.0 - H           # occupancy in [0,1] (1 = fully inside)

# density = 2500.0
# voxel_vol = stride**3
# masses = density * voxel_vol * w

# total_mass = float(masses.sum())
# if total_mass == 0.0:
#     raise RuntimeError("Total mass is zero — check sdf or eps.")

# # -------------------------
# # Centre of mass (cell-centre convention)
# # -------------------------
# com_x = float(np.sum(masses * Xc) / total_mass)
# com_y = float(np.sum(masses * Yc) / total_mass)
# com_z = float(np.sum(masses * Zc) / total_mass)
# com = np.array([com_x, com_y, com_z], dtype=float)

# # -------------------------
# # Raw inertia tensor about COM (NO symmetrisation)
# # Ixx = sum m (y^2 + z^2), etc.
# # products Ixy = - sum m x y  (no sign change here)
# # -------------------------
# xr = Xc - com_x
# yr = Yc - com_y
# zr = Zc - com_z

# Ixx = np.sum(masses * (yr**2 + zr**2))
# Iyy = np.sum(masses * (xr**2 + zr**2))
# Izz = np.sum(masses * (xr**2 + yr**2))
# Ixy = -np.sum(masses * xr * yr)
# Ixz = -np.sum(masses * xr * zr)
# Iyz = -np.sum(masses * yr * zr)

# I_raw = np.array([[Ixx, Ixy, Ixz],
#                   [Ixy, Iyy, Iyz],
#                   [Ixz, Iyz, Izz]], dtype=float)

# # -------------------------
# # Diagnostics: raw values and symmetry checks
# # -------------------------
# offdiag_fro = np.linalg.norm(I_raw - np.diag(np.diag(I_raw)))
# sym_diff_max = np.max(np.abs(I_raw - I_raw.T))

# diag = np.diag(I_raw)
# rel_offdiag = np.zeros_like(diag)
# for k in range(3):
#     if diag[k] != 0:
#         rel_offdiag[k] = offdiag_fro / diag[k]
#     else:
#         rel_offdiag[k] = np.nan

# print("\n=== RAW INERTIA & COM DIAGNOSTICS (NO SYMMETRISING) ===")
# print(f"Total mass (kg): {total_mass:.6f}")
# print(f"COM (m)         : {com}")
# print("\nRaw inertia tensor about COM (kg·m^2):")
# print(I_raw)
# print(f"\nFrobenius norm of off-diagonals: {offdiag_fro:.6e}")
# print(f"Max |I - I^T| elementwise: {sym_diff_max:.6e}")
# print(f"Relative offdiag / diag (approx): {rel_offdiag}")

# # -------------------------
# # Mirror imbalance diagnostics (weights)
# # Compare w(x,y,z) vs w(-x,y,z), etc. Report sum |diff| and normalized by sum(w)
# # -------------------------
# W = w.reshape((nx, ny, nz))   # x-fast index ordering
# sum_w = float(W.sum())

# def mirror_diff_axis(W, axis):
#     # axis: 0=x, 1=y, 2=z
#     if axis == 0:
#         W_pos = W
#         W_neg = W[::-1, :, :]   # flip x index
#     elif axis == 1:
#         W_pos = W
#         W_neg = W[:, ::-1, :]   # flip y index
#     else:
#         W_pos = W
#         W_neg = W[:, :, ::-1]   # flip z index
#     diff = np.abs(W_pos - W_neg)
#     return float(diff.sum()), float(np.count_nonzero(diff > 0.0))

# dx_sum, dx_count = mirror_diff_axis(W, 0)
# dy_sum, dy_count = mirror_diff_axis(W, 1)
# dz_sum, dz_count = mirror_diff_axis(W, 2)

# print("\nMirror imbalance (sum |w - w_mirror|, #cells different) per axis:")
# print(f" X-axis: sum|Δw| = {dx_sum:.6e}   differing cells = {dx_count}")
# print(f" Y-axis: sum|Δw| = {dy_sum:.6e}   differing cells = {dy_count}")
# print(f" Z-axis: sum|Δw| = {dz_sum:.6e}   differing cells = {dz_count}")
# print(f" Normalized (sum|Δw| / total_w): X {dx_sum/sum_w:.6e}, Y {dy_sum/sum_w:.6e}, Z {dz_sum/sum_w:.6e}")

# # -------------------------
# # Binary inside-mask counts (for cross-check)
# # -------------------------
# inside_mask = ls_dem_grid <= 0.0
# count_inside = int(np.count_nonzero(inside_mask))
# count_xpos = int(np.count_nonzero(inside_mask & (Xc > 0.0)))
# count_xneg = int(np.count_nonzero(inside_mask & (Xc < 0.0)))
# count_ypos = int(np.count_nonzero(inside_mask & (Yc > 0.0)))
# count_yneg = int(np.count_nonzero(inside_mask & (Yc < 0.0)))
# count_zpos = int(np.count_nonzero(inside_mask & (Zc > 0.0)))
# count_zneg = int(np.count_nonzero(inside_mask & (Zc < 0.0)))

# print("\nBinary inside-mask counts (sdf <= 0):")
# print(" Total inside voxels:", count_inside)
# print(" X pos / neg:", (count_xpos, count_xneg))
# print(" Y pos / neg:", (count_ypos, count_yneg))
# print(" Z pos / neg:", (count_zpos, count_zneg))

# # Final quick guidance printed numerically
# if dx_sum/sum_w > 1e-12 or dy_sum/sum_w > 1e-12 or dz_sum/sum_w > 1e-12:
#     print("\nNote: mirror imbalance > 1e-12 detected (see normalized values). This indicates the occupancy field is not perfectly symmetric and will produce non-zero off-diagonals.")
# else:
#     print("\nMirror imbalance negligible → occupancy symmetric to machine precision; any remaining off-diagonals are round-off noise.")
