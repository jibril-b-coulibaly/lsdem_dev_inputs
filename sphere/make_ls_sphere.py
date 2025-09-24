#!/usr/bin/env python3
# # Python script that computes level set for a 3D sphere
import numpy as np
from math import sqrt, pi

# Use the same values Joel hardcoded pair_ls_dem
rad = 2.5

nintervals = 20
nbuff = 6 # Must be pretty high in some cases.
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
method = "fibonacci"   # "fibonacci" or "icosa"
num_points = 1600       # desired number of points (used directly for fibonacci)
subdiv = 3             # only used for "icosa": number of subdivisions (integer >=0)
outfile = "data_sphere"  # output filename
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

def icosahedron():
    """Return vertices and faces for an icosahedron (unit sphere)."""
    t = (1.0 + sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],

        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],

        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1],
    ], dtype=float)
    # normalize to unit sphere
    verts /= np.linalg.norm(verts, axis=1)[:, None]

    faces = np.array([
        [0, 11, 5], [0,5,1], [0,1,7], [0,7,10], [0,10,11],
        [1,5,9], [5,11,4], [11,10,2], [10,7,6], [7,1,8],
        [3,9,4], [3,4,2], [3,2,6], [3,6,8], [3,8,9],
        [4,9,5], [2,4,11], [6,2,10], [8,6,7], [9,8,1],
    ], dtype=int)
    return verts, faces

def subdivide_icosa(verts, faces, n_subdiv):
    """Subdivide each triangular face n_subdiv times (Loop subdivision style for geodesic sphere).
       Returns unique vertices projected to unit sphere."""
    if n_subdiv <= 0:
        return verts.copy()

    # cache for midpoint of edge -> index
    midpoint_cache = {}

    def midpoint_index(i1, i2):
        """Return index of midpoint vertex between i1 and i2, creating if necessary."""
        key = tuple(sorted((i1, i2)))
        if key in midpoint_cache:
            return midpoint_cache[key]
        v1 = vertices[i1]
        v2 = vertices[i2]
        mid = (v1 + v2) / 2.0
        # project to unit sphere
        mid /= np.linalg.norm(mid)
        vertices.append(mid)
        idx = len(vertices) - 1
        midpoint_cache[key] = idx
        return idx

    # start with lists (so we can append)
    vertices = [v.copy() for v in verts]
    faces_list = [tuple(f) for f in faces]

    for _ in range(n_subdiv):
        new_faces = []
        midpoint_cache.clear()
        for tri in faces_list:
            v0, v1, v2 = tri
            a = midpoint_index(v0, v1)
            b = midpoint_index(v1, v2)
            c = midpoint_index(v2, v0)
            # create four new triangles
            new_faces.append((v0, a, c))
            new_faces.append((v1, b, a))
            new_faces.append((v2, c, b))
            new_faces.append((a, b, c))
        faces_list = new_faces

    vertices = np.array(vertices)
    # final projection to unit sphere (numerical safety)
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    return vertices

# -------------------------
# Choose method and generate points
# -------------------------
if method.lower() == "fibonacci":
    pts = fibonacci_sphere(num_points, radius=rad)
elif method.lower() == "icosa":
    # generate base icosahedron and subdivide
    base_verts, base_faces = icosahedron()
    pts = subdivide_icosa(base_verts, base_faces, subdiv)
    # scale to radius
    pts *= rad
    num_points = pts.shape[0]  # update actual number
else:
    raise ValueError("Unknown method. Use 'fibonacci' or 'icosa'.")

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
        f.write(f"{i+1} 1 1 1.0 {xi} {yi} {zi}\n")

print(f"Wrote {len(pts)} points to {outfile} using method='{method}'.")
