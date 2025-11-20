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

    # Regexes
    pat_pe_only = re.compile(r"^\s*[pP](\d+)\s*$")
    pat_pe_rf   = re.compile(r"^\s*[pP](\d+)\s*,\s*[rR](\d+)\s*$")
    pat_mem     = re.compile(r"^\s*m(\d+)\s*$")  # <-- case-sensitive

    results = []

    # --- Scan each cell ---
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if not isinstance(cell, str):
                continue
            text = cell.strip()

            # Case 1: P[num] (case-insensitive)
            m = pat_pe_only.match(text)
            if m:
                x = c - origin_c
                y = r - origin_r
                results.append((f"p{m.group(1)}", x, y))
                continue

            # Case 2: P[num], r[num] (case-insensitive)
            m = pat_pe_rf.match(text)
            if m:
                pe = f"p{m.group(1)}"
                rf = f"r{m.group(2)}"
                x = c - origin_c
                y = r - origin_r
                results.append((pe, x, y))
                results.append((rf, x, y))
                continue

            # Case 3: m[num] (case-sensitive)
            m = pat_mem.match(text)
            if m:
                x = c - origin_c
                y = r - origin_r
                results.append((f"m{m.group(1)}", x, y))
                continue

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