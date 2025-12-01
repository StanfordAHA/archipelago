"Script to duplicate manual place for low unrolling duplication"

from collections import defaultdict

def low_unrolling_duplication(input_manual_place_path, output_manual_place_path, pe_increment=16, reg_increment=16, mem_increment=16):
    output_placement = open(output_manual_place_path, "w")

    node_count = defaultdict(int)

    with open(input_manual_place_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                node, x, y = line.split()
                node_type = node[0]
                node_num = int(node[1:])

                # PE
                if node_type == 'p':
                    new_node_num = node_num + pe_increment * node_count[node]
                # Reg
                elif node_type == 'r':
                    new_node_num = node_num + reg_increment * node_count[node]
                # MEM
                elif node_type == 'm':
                    new_node_num = node_num + mem_increment * node_count[node]


                output_placement.write(f"{node_type}{new_node_num} {x} {y}\n")
                node_count[node] += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract manual place from CSV layout.")
    parser.add_argument("input_manual_place_path", type=str, help="Path to the input manual place file.")
    parser.add_argument("output_manual_place_path", type=str, help="Path to the output manual place file.")
    args = parser.parse_args()

    low_unrolling_duplication(args.input_manual_place_path, args.output_manual_place_path, pe_increment=36, reg_increment=24, mem_increment=4)
