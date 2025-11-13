#!/usr/bin/env python3
# convert_dump.py
# Python 3 only, no external dependencies required.

from typing import List, Dict, Tuple
import sys

def convert_lammps_dump(infile: str, outfile: str = "dump_grain_bounce_100.txt") -> None:
    """
    Read a LAMMPS dump file and write per-timestep averages:
    step x y z vx vy vz fx fy fz torque_x torque_y torque_z

    If a column (vx, fx, ...) is missing in the ATOMS header it will be
    treated as zero (one warning printed).
    """
    required = ["x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz"]
    saw_missing_warning = False

    def parse_atoms_header(line: str) -> List[str]:
        # line like: "ITEM: ATOMS id type x y z vx vy vz fx fy fz"
        parts = line.strip().split()
        # remove the leading 'ITEM:' and 'ATOMS'
        if len(parts) >= 3 and parts[0] == "ITEM:" and parts[1] == "ATOMS":
            return parts[2:]
        # fallback: try to find 'ATOMS' then return following tokens
        try:
            idx = parts.index("ATOMS")
            return parts[idx+1:]
        except ValueError:
            return parts[2:] if len(parts) > 2 else []

    with open(infile, "r") as inf, open(outfile, "w") as outf:
        outf.write("# step x y z vx vy vz fx fy fz torque_x torque_y torque_z\n")
        line = inf.readline()
        while line:
            if line.startswith("ITEM: TIMESTEP"):
                # read timestep
                ts_line = inf.readline()
                if not ts_line:
                    break
                timestep = int(ts_line.strip())

                # read number of atoms
                line = inf.readline()  # should be "ITEM: NUMBER OF ATOMS"
                if not line:
                    break
                if not line.startswith("ITEM: NUMBER OF ATOMS"):
                    # attempt to recover: skip until we find it
                    while line and not line.startswith("ITEM: NUMBER OF ATOMS"):
                        line = inf.readline()
                    if not line:
                        break
                natoms_line = inf.readline()
                if not natoms_line:
                    break
                try:
                    natoms = int(natoms_line.strip())
                except ValueError:
                    # if malformed, skip this block
                    natoms = 0

                # skip box bounds (commonly 3 lines, but may vary if triclinic; we assume 3)
                # read next line; expect "ITEM: BOX BOUNDS ..." then 3 lines
                line = inf.readline()
                if not line:
                    break
                if line.startswith("ITEM: BOX BOUNDS"):
                    # skip 3 lines (typical)
                    for _ in range(3):
                        line = inf.readline()
                # now expect ITEM: ATOMS ...
                while line and not line.startswith("ITEM: ATOMS"):
                    line = inf.readline()
                if not line:
                    break
                atom_cols = parse_atoms_header(line)
                # map column names to indices
                col_index: Dict[str, int] = {}
                for i, name in enumerate(atom_cols):
                    # LAMMPS may use x y z or xu yu zu; accept anything starting with x,y,z etc.
                    col_index[name] = i

                # find indices for needed fields; if not present mark missing
                indices = {}
                missing = []
                for key in required:
                    if key in col_index:
                        indices[key] = col_index[key]
                    else:
                        # try common aliases: x->x,y->y,z->z; sometimes 'xu' or 'x' etc.
                        alt_found = False
                        for cname in atom_cols:
                            if cname.startswith(key):  # e.g. 'x' matches 'x', 'xu'
                                indices[key] = col_index[cname]
                                alt_found = True
                                break
                        if not alt_found:
                            indices[key] = None
                            missing.append(key)

                if missing and not saw_missing_warning:
                    print("Warning: missing columns in dump header, filling missing fields with zeros:", missing, file=sys.stderr)
                    saw_missing_warning = True

                # accumulate sums
                sum_vals = {k: 0.0 for k in required}
                count = 0

                # read natoms lines (safe against short files)
                for _ in range(natoms):
                    atom_line = inf.readline()
                    if not atom_line:
                        break
                    toks = atom_line.strip().split()
                    if len(toks) < 1:
                        continue
                    # convert numeric tokens to floats lazily for requested columns
                    for k in required:
                        idx = indices[k]
                        if idx is None or idx >= len(toks):
                            # missing -> treat as zero
                            continue
                        try:
                            sum_vals[k] += float(toks[idx])
                        except ValueError:
                            # cannot parse -> treat as zero
                            continue
                    count += 1

                if count == 0:
                    # nothing to average: write zeros
                    xm = ym = zm = 0.0
                    vxm = vym = vzm = 0.0
                    fxm = fym = fzm = 0.0
                else:
                    xm = sum_vals["x"] / count
                    ym = sum_vals["y"] / count
                    zm = sum_vals["z"] / count
                    vxm = sum_vals["vx"] / count
                    vym = sum_vals["vy"] / count
                    vzm = sum_vals["vz"] / count
                    fxm = sum_vals["fx"] / count
                    fym = sum_vals["fy"] / count
                    fzm = sum_vals["fz"] / count

                # torques are zero per your instruction
                tx = ty = tz = 0.0

                outf.write(f"{timestep} {xm:.6e} {ym:.6e} {zm:.6e} "
                           f"{vxm:.6e} {vym:.6e} {vzm:.6e} "
                           f"{fxm:.6e} {fym:.6e} {fzm:.6e} "
                           f"{tx:.6e} {ty:.6e} {tz:.6e}\n")

                # advance
                line = inf.readline()
                continue

            # otherwise advance line-by-line until next block
            line = inf.readline()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 convert_dump.py <dump-file>")
        sys.exit(1)
    convert_lammps_dump(sys.argv[1])
    print("Wrote dump_grain_bounce_100.txt")
