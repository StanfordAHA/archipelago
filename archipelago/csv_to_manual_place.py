"Script to extract manual place from CSV layout."

import csv
import re

def extract_layout_positions(csv_path, output_path):
    # --- Load CSV file ---
    grid = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            grid.append(row)

    # --- Find all IO tiles (case-insensitive) ---
    io_positions = []
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if isinstance(cell, str) and cell.strip().lower() == "io":
                io_positions.append((r, c))

    if not io_positions:
        raise ValueError("No IO tiles found!")

    # Origin = top-left IO tile
    origin_r, origin_c = min(io_positions)

    # --- Generalized regex patterns ---
    # PEs: p[num], case-insensitive
    pat_pe  = re.compile(r"[pP](\d+)")

    # Registers: r[num], case-insensitive
    pat_reg = re.compile(r"[rR](\d+)")

    # MEMs: m[num], *case-sensitive*
    pat_mem = re.compile(r"m(\d+)")

    results = []

    # --- Scan grid and extract ALL matching elements ---
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if not isinstance(cell, str):
                continue

            text = cell.strip()

            # Find *all* occurrences in the cell
            pes  = pat_pe.findall(text)
            regs = pat_reg.findall(text)
            mems = pat_mem.findall(text)

            if not (pes or regs or mems):
                continue  # no elements in this cell

            x = c - origin_c
            y = r - origin_r

            # Append all PEs
            for pe_num in pes:
                results.append((f"p{pe_num}", x, y))

            # Append all Registers
            for reg_num in regs:
                results.append((f"r{reg_num}", x, y))

            # Append all MEMs
            for mem_num in mems:
                results.append((f"m{mem_num}", x, y))

    # --- Write output file ---
    with open(output_path, "w") as f:
        for elem, x, y in results:
            f.write(f"{elem} {x} {y}\n")

    print(f"Done. Wrote {len(results)} elements to {output_path}")


if __name__ == "__main__":
    # Set up csv and output path as arguments using argparse
    import argparse
    parser = argparse.ArgumentParser(description="Extract manual place from CSV layout.")
    parser.add_argument("csv_path", type=str, help="Path to the input CSV layout file.")
    parser.add_argument("output_path", type=str, help="Path to the output manual place file.")
    args = parser.parse_args()

    extract_layout_positions(args.csv_path, args.output_path)
