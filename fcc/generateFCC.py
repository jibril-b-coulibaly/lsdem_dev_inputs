import math

# === FCC lattice parameters ===
nUnitCells = 4  # per dimension
r = 0.12
latticeLen = 2.0 * math.sqrt(2.0) * r
nNodes = 512

# === Collect FCC positions ===
coords = []

# Corners
for i in range(nUnitCells):
    for j in range(nUnitCells):
        for k in range(nUnitCells):
            coords.append((i*latticeLen, j*latticeLen, k*latticeLen))

# Faces
for i in range(nUnitCells):
    for j in range(nUnitCells):
        for k in range(nUnitCells):
            coords.append((i*latticeLen, (j+0.5)*latticeLen, (k+0.5)*latticeLen))

for i in range(nUnitCells):
    for j in range(nUnitCells):
        for k in range(nUnitCells):
            coords.append(((i+0.5)*latticeLen, j*latticeLen, (k+0.5)*latticeLen))

for i in range(nUnitCells):
    for j in range(nUnitCells):
        for k in range(nUnitCells):
            coords.append(((i+0.5)*latticeLen, (j+0.5)*latticeLen, k*latticeLen))

# Sanity check
nParticles = len(coords)
print(f"Total particles: {nParticles}")

# === Write to file ===
output_file = "fcc_read_data_and_groups.txt"

with open(output_file, "w") as f:

    # --- Section 1: read_data commands ---
    f.write("# Command filename add number_atoms mol_id shift posx posy posz\n")
    number_atoms = 0
    for mol_id, (x, y, z) in enumerate(coords, start=1):
        line = (f"read_data data_sphere_{nNodes} add {number_atoms} {mol_id} "
                f"shift {x:.6f} {y:.6f} {z:.6f}\n")
        f.write(line)
        number_atoms += 512

    # --- Section 2: group definitions ---
    f.write("\n# Group definitions\n")
    for mol_id in range(1, nParticles+1):
        f.write(f"group           grain{mol_id} molecule {mol_id}\n")

    # --- Section 3: body info block ---
    f.write("\n# ID1 masstotal xcm ycm zcm ixx iyy izz ixy ixz iyz vxcm vycm vzcm lx ly lz ixcm iycm izcm memoryflag scale gridfile\n")
    f.write(f"{nParticles}\n")
    # Default constants for all bodies
    masstotal = 2500.0*4.0/3.0*math.pi*(r**3.0)
    ixx = iyy = izz = 2/5*masstotal*(r**2.0)
    ixy = ixz = iyz = 0.0
    vxcm = vycm = vzcm = 0.0
    lx = ly = lz = 0.0
    ixcm = iycm = izcm = 0
    memoryflag = 0
    scale = 1.0
    gridfile = "grid_sphere.txt"

    for ID1, (x, y, z) in enumerate(coords, start=1):
        f.write(f"{ID1} {masstotal:.10f} {x:.6f} {y:.6f} {z:.6f} "
                f"{ixx:.11f} {iyy:.11f} {izz:.11f} {ixy:.1f} {ixz:.1f} {iyz:.1f} "
                f"{vxcm:.1f} {vycm:.1f} {vzcm:.1f} "
                f"{lx:.1f} {ly:.1f} {lz:.1f} "
                f"{ixcm} {iycm} {izcm} {memoryflag} {scale:.1f} {gridfile}\n")  

print(f"File written: {output_file}")
