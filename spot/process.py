#Original spot geometry courtesy of https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/

import numpy as np

fin = open("ls_spot_392.csv", "r")

line = next(fin)
x, y, z = line.strip().split(",")

line = next(fin)
q1, q2, q3, q4 = line.strip().split(",") # is this needed? Quaternion relative to ...?

line = next(fin)
nGPx, nGPy, nGPz = [int(x) for x in line.strip().split(",")]

line = next(fin)
grid_spacing = line.strip().split(",")[0]

line = next(fin)
grid_minx, grid_miny, grid_minz = line.strip().split(",")

line = next(fin)
grid_maxx, grid_maxy, grid_maxz = line.strip().split(",")

line = next(fin)
#skip aabe line

nodes_x = []
nodes_y = []
nodes_z = []

for line in fin:
    line = line.strip()
    if line == "":
        break
    nx, ny, nz, nnx, nny, nnz, area = line.split(",")
    nodes_x.append(float(nx))
    nodes_y.append(float(ny))
    nodes_z.append(float(nz))

grid_scale = 1.0 # is this defined in the csv?

fout = open("spot.mol", "w")
fout.write("#Spot molecule\n")
fout.write("{} atoms\n".format(len(nodes_x)))
fout.write("1.0 mass\n")
fout.write("grid_spot.txt 0 {} lsdem\n\n".format(grid_scale))
fout.write("Coords\n\n")

for i in range(len(nodes_x)):
    fout.write("{} {:.6f} {:.6f} {:.6f}\n".format(i+1, nodes_x[i], nodes_y[i], nodes_z[i]))

fout.write("Types\n\n")
for i in range(len(nodes_x)):
    fout.write("{} 1\n".format(i+1))
fout.write("\n")
fout.close()

line = next(fin)  # skip blank line

values = np.zeros((nGPx, nGPy, nGPz), dtype=float)

for ix in range(nGPx):
    for iy in range(nGPy):
        line = next(fin)
        row = line.strip().split(",")
        if (len(row) != nGPz):
            print("Error reading grid at ix={}, iy={}".format(ix, iy))
            print(row)
            exit(1)
        for j, val in enumerate(row):
            values[ix, iy, j] = float(val)

    if ix != nGPx - 1:
        line1 = next(fin)  # skip blank lines between yz planes
        line2 = next(fin)

print(nGPx, nGPy, nGPz, grid_spacing)

fout = open("grid_spot.txt", "w")
fout.write("# Grid file for spot\n")
fout.write("{} {} {}\n".format(nGPx, nGPy, nGPz))
fout.write("{}\n".format(grid_spacing))
fout.write("{} {} {}\n".format(grid_minx, grid_miny, grid_minz))
for iz in range(nGPz):
    for iy in range(nGPy):
        for ix in range(nGPx):
            if values[ix, iy, iz] == 0.0:
                print("Warning: zero value at ix={}, iy={}, iz={}".format(ix, iy, iz))
            val = values[ix, iy, iz]
            fout.write("{}\n".format(val))

fout.close()
fin.close()
