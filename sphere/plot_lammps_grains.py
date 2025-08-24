#!/usr/bin/env python3
"""plot_lammps_grains.py

Parse a LAMMPS "custom" dump file created with a command like:
  dump mydump all custom 100 atomDump_sphere id type mol x y z vx vy vz fx fy fz

Compute the mean x coordinate of each molecule (molecule ID field 'mol')
for every dumped timestep and plot the x coordinate vs timestep for each molecule.

Usage:
  python3 plot_lammps_grains.py -i atomDump_sphere -o grains_x.png

Outputs:
  - PNG plot (default grains_x.png)
  - Optional CSV with the time series for each molecule (if --csv given)

Notes:
  - Requires Python packages: numpy, pandas, matplotlib
  - The script is conservative when reading the dump: it detects the column order
    from the "ITEM: ATOMS ..." heading and uses the fields present.
"""

import argparse
import sys
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def parse_lammps_custom_dump(path, require_fields=('id','mol','x')):
    """Parse a LAMMPS custom dump file and return an OrderedDict mapping timestep->DataFrame of atom data."""
    frames = OrderedDict()  # timestep -> dataframe for atoms in that frame
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dump file not found: {path}")
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            if line.startswith("ITEM: TIMESTEP"):
                # read timestep
                ts_line = f.readline()
                if not ts_line:
                    break
                timestep = int(ts_line.strip())
                # read number of atoms header
                hdr = f.readline()  # ITEM: NUMBER OF ATOMS
                natoms_line = f.readline()
                natoms = int(natoms_line.strip())
                # read box bounds header and three lines of box bounds (usually)
                box_hdr = f.readline()  # ITEM: BOX BOUNDS ...
                # box may have 3 lines (3 dims); read until next "ITEM: ATOMS" appears
                # but standard LAMMPS format has exactly 3 lines for 3D
                box_lines = [f.readline() for _ in range(3)]
                # now read atoms header
                atoms_hdr_line = f.readline()
                if not atoms_hdr_line or not atoms_hdr_line.startswith('ITEM: ATOMS'):
                    # if unexpected, skip
                    raise RuntimeError(f"Expected 'ITEM: ATOMS' but got: {atoms_hdr_line}")
                # parse column names and their order
                tokens = atoms_hdr_line.strip().split()
                # tokens[0] = ITEM:, tokens[1] = ATOMS, rest are column names
                columns = tokens[2:]
                # read natoms lines and parse floats/ints
                data = {col: [] for col in columns}
                for _ in range(natoms):
                    parts = f.readline().strip().split()
                    if len(parts) < len(columns):
                        raise RuntimeError("Atom line has fewer columns than expected")
                    for col, val in zip(columns, parts):
                        data[col].append(val)
                # convert to DataFrame, with numeric conversion where possible
                df = pd.DataFrame(data)
                # convert numeric columns to float or int where appropriate
                for col in df.columns:
                    # try int first if all integer strings, otherwise float
                    try:
                        if all('.' not in x and 'e' not in x.lower() for x in df[col]):
                            df[col] = df[col].astype(int)
                        else:
                            df[col] = df[col].astype(float)
                    except Exception:
                        # fallback to float conversion where possible
                        try:
                            df[col] = df[col].astype(float)
                        except Exception:
                            pass
                frames[timestep] = df
                line = f.readline()
            else:
                line = f.readline()
    return frames

def compute_mean_x_per_molecule(frames, mol_field='mol', x_field='x'):
    """Return a DataFrame indexed by timestep with columns for each molecule id containing
    the mean x coordinate for that molecule at that timestep."""
    records = []
    timesteps = []
    mol_ids_set = set()
    for ts, df in frames.items():
        if mol_field not in df.columns or x_field not in df.columns:
            raise KeyError(f"Required columns '{mol_field}' and '{x_field}' not found in dump frame.")
        grouped = df.groupby(mol_field)[x_field].mean()
        rec = grouped.to_dict()
        records.append(rec)
        timesteps.append(ts)
        mol_ids_set.update(rec.keys())
    mol_ids = sorted(mol_ids_set)
    # build DataFrame
    out_df = pd.DataFrame(records, index=timesteps, columns=mol_ids)
    out_df.index.name = 'timestep'
    # sort by timestep
    out_df = out_df.sort_index()
    return out_df

def plot_molecules_x(df, out_png='grains_x.png', xlabel='Timestep', ylabel='Mean x coordinate', title=None, save_csv=None):
    plt.figure(figsize=(8,5))
    for col in df.columns:
        plt.plot(df.index.values, df[col].values, label=f'mol {col}')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"Saved plot to: {out_png}")
    if save_csv:
        df.to_csv(save_csv)
        print(f"Saved CSV time series to: {save_csv}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot mean x coordinate of molecules from a LAMMPS dump file.")
    parser.add_argument('-i', '--input', required=True, help='Path to LAMMPS dump file (custom format)')
    parser.add_argument('-o', '--output', default='grains_x.png', help='Output PNG file name')
    parser.add_argument('--csv', default=None, help='Optional CSV filename to save time series (per-molecule)')
    parser.add_argument('--mol-field', default='mol', help="Column name for molecule ID in dump (default: 'mol')")
    parser.add_argument('--x-field', default='x', help="Column name for x coordinate in dump (default: 'x')")
    args = parser.parse_args()

    frames = parse_lammps_custom_dump(args.input)
    df = compute_mean_x_per_molecule(frames, mol_field=args.mol_field, x_field=args.x_field)
    if df.empty:
        print('No data parsed or no molecules found.')
        sys.exit(1)
    plot_molecules_x(df, out_png=args.output, save_csv=args.csv, title=f"Mean x per molecule from {os.path.basename(args.input)}")

if __name__ == '__main__':
    main()
