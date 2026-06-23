#!/usr/bin/env python3
# Elongated rectangular slab ("fibre-like") for the LS-DEM item-2c normal-filter
# demo. It is the cube (make_ls_cube.py) stretched along z by `ratio`, keeping the
# SAME number of surface nodes per axis (na). That makes the z node spacing
# ratio-times the x/y spacing, so the auto pair cutoff (factor x worst spacing =
# the z spacing) is over-generous for the dense x/y faces -> many back-facing
# false candidates there -> the normal filter (pair_style ls/dem auto nfilter)
# removes them. Outputs grid_slab.txt + data_slab and prints the infile inertia.
import numpy as np

hs    = 1.4433756729740645   # cube half-side (x,y); reuse so packing scales match
ratio = 3.0
h     = np.array([hs, hs, ratio*hs])   # half-extents
stride = 0.5
na    = 10                              # surface nodes per axis (-> 488 shell nodes)

# ---- LS grid (exact box signed-distance) -------------------------------------
def box_sdf(p):
    q = np.abs(p) - h
    outside = np.sqrt(np.sum(np.maximum(q, 0.0)**2))
    inside  = min(max(q[0], max(q[1], q[2])), 0.0)
    return outside + inside

# grid dims: cover +-(h + margin), centred on the COM (like the cube: gmin = -(n-1)*stride/2)
margin = 3.3
def ndim(ext):
    n = int(np.ceil(2.0*ext/stride)) + 1
    return n + (1 - n % 2)            # force odd so the COM sits on a grid node
nx, ny, nz = ndim(h[0]+margin), ndim(h[1]+margin), ndim(h[2]+margin)
gmin = -0.5*(np.array([nx, ny, nz]) - 1)*stride

with open("grid_slab.txt", "w") as f:
    f.write(f"# Grid file for a slab half-extents {h.tolist()} (ratio {ratio})\n")
    f.write(f"{nx} {ny} {nz}\n")
    f.write(f"{stride}\n")
    f.write(f"{gmin[0]} {gmin[1]} {gmin[2]}\n")
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                p = gmin + np.array([ix, iy, iz])*stride
                f.write(f"{box_sdf(p)}\n")

# ---- surface nodes (box shell): all grid corners on >=1 face, deduped ---------
xs = np.linspace(-h[0], h[0], na)
ys = np.linspace(-h[1], h[1], na)
zs = np.linspace(-h[2], h[2], na)
nodes = []
for i in range(na):
    for j in range(na):
        for k in range(na):
            if i in (0, na-1) or j in (0, na-1) or k in (0, na-1):
                nodes.append((xs[i], ys[j], zs[k]))
num_atoms = len(nodes)

with open("data_slab", "w") as f:
    f.write("# Generated using make_ls_slab.py\n\n")
    f.write(f"{num_atoms} atoms\n 1 atom types\n\n")
    f.write(f"{-h[0]} {h[0]} xlo xhi\n")
    f.write(f"{-h[1]} {h[1]} ylo yhi\n")
    f.write(f"{-h[2]} {h[2]} zlo zhi\n\n")
    f.write("Atoms\n\n")
    for i, (xi, yi, zi) in enumerate(nodes):
        f.write(f"{i+1} 1 1 {xi} {yi} {zi}\n")

# ---- molecule file for the SMALL fix (global storage; inertia auto-computed) -
# format mirrors fcc2/mol_sphere_512: "<gridfile> <grid_index> <scale> lsdem".
with open("mol_slab", "w") as f:
    f.write("# Generated using make_ls_slab.py (elongated slab, small fix + global)\n")
    f.write(f"{num_atoms} atoms\n")
    f.write("1.0 mass\n")
    f.write("grid_slab.txt 0 1.0 lsdem\n\n")
    f.write("Coords\n\n")
    for i, (xi, yi, zi) in enumerate(nodes):
        f.write(f"{i+1} {xi} {yi} {zi}\n")
    f.write("\nTypes\n\n")
    for i in range(num_atoms):
        f.write(f"{i+1} 1\n")

# ---- solid-box inertia (mass 1) for an (optional) large-fix infile -----------
m = 1.0
L = 2.0*h
Ixx = m/12.0*(L[1]**2 + L[2]**2)
Iyy = m/12.0*(L[0]**2 + L[2]**2)
Izz = m/12.0*(L[0]**2 + L[1]**2)
print(f"num_atoms = {num_atoms}")
print(f"grid = {nx} x {ny} x {nz}, gmin = {gmin.tolist()}")
print(f"node spacing  x/y = {2*hs/(na-1):.4f}   z = {2*ratio*hs/(na-1):.4f}  (ratio {ratio})")
print(f"circumscribed radius = {np.sqrt(np.sum(h**2)):.4f}")
print(f"inertia Ixx Iyy Izz = {Ixx:.4f} {Iyy:.4f} {Izz:.4f}")
