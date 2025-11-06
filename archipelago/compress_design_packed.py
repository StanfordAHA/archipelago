import re
import pydot
import argparse
from collections import defaultdict, deque

def build_and_collapse_graph(filename: str) -> pydot.Dot:
    """
    Read the edge file, build a pydot digraph, and collapse chains of nodes
    whose names start with 'r'. Collapsed regs are replaced by nodes named
    (rX, N regs), where X is a unique ID and N is the number of regs collapsed.
    """

    edges = []           # (src_name, src_type, dst_name, dst_type, edge_label)
    node_types = {}      # name -> type

    def parse_node(inner: str):
        parts = [p.strip() for p in inner.split(",", 1)]
        if len(parts) == 2:
            return parts[0], parts[1]
        return inner.strip(), "unknown"

    # --- Parse file into edges and node types ---
    with open(filename, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            edge_name = line.split(":", 1)[0].strip()
            groups = re.findall(r"\(([^)]+)\)", line)
            if not groups:
                continue
            src_name, src_type = parse_node(groups[0])
            node_types[src_name] = src_type
            for g in groups[1:]:
                dst_name, dst_type = parse_node(g)
                node_types[dst_name] = dst_type
                edges.append((src_name, src_type, dst_name, dst_type, edge_name))

    # --- Build adjacency ---
    succs = defaultdict(list)
    for sname, stype, dname, dtype, lbl in edges:
        succs[sname].append((dname, lbl))

    def is_reg(name: str) -> bool:
        return name.startswith("r")

    # --- Build simplified graph ---
    new_graph = pydot.Dot(graph_type="digraph")
    added_nodes = set()
    added_edges = set()
    collapsed_counter = 0  # for unique reg node names

    node_types_new = {}  # track new node types

    def add_node(name: str, ntype: str):
        if name not in added_nodes:
            new_graph.add_node(pydot.Node(name, label=f"({name}, {ntype})"))
            added_nodes.add(name)
            node_types_new[name] = ntype

    # Helper to compute minimal reg counts from a starting reg to first non-reg sinks
    def reg_first_nonreg_min_counts(start_reg: str) -> dict:
        queue = deque([start_reg])
        reg_distance = {start_reg: 1}
        sink_min = {}
        while queue:
            current_reg = queue.popleft()
            current_reg_dist = reg_distance[current_reg]
            for next_reg, _ in succs.get(current_reg, []):
                if is_reg(next_reg):
                    if next_reg not in reg_distance:
                        reg_distance[next_reg] = current_reg_dist + 1
                        queue.append(next_reg)
                else:
                    prev = sink_min.get(next_reg)
                    if prev is None or current_reg_dist < prev:
                        sink_min[next_reg] = current_reg_dist
        return sink_min

    for src in list(succs.keys()):
        if is_reg(src):
            continue  # skip registers as starting points

        add_node(src, node_types.get(src, "unknown"))

        for dst, edge_label in succs[src]:
            if not is_reg(dst):
                add_node(dst, node_types.get(dst, "unknown"))
                key = (src, dst, edge_label)
                if key not in added_edges:
                    new_graph.add_edge(pydot.Edge(src, dst, label=edge_label))
                    added_edges.add(key)
                continue

            # collapse chain of regs starting at dst
            sink_min_counts = reg_first_nonreg_min_counts(dst)
            for end, reg_count in sink_min_counts.items():
                if is_reg(end):
                    continue

                collapsed_name = f"r{collapsed_counter}"
                collapsed_type = f"{reg_count} fifos"
                collapsed_counter += 1

                add_node(collapsed_name, collapsed_type)
                add_node(end, node_types.get(end, "unknown"))
                key_in = (src, collapsed_name, edge_label)

                # edge src -> collapsed
                if key_in not in added_edges:
                    new_graph.add_edge(pydot.Edge(src, collapsed_name, label=edge_label))
                    added_edges.add(key_in)

                # edges collapsed -> endpoints
                key_out = (collapsed_name, end, None)
                if key_out not in added_edges:
                    new_graph.add_edge(pydot.Edge(collapsed_name, end))
                    added_edges.add(key_out)

    # attach node_types_new to graph object for later use
    new_graph.node_types_new = node_types_new
    return new_graph


def export_graph_to_file(graph: pydot.Dot, outfile: str):
    """
    Export the simplified graph to a text file in the same format as input.
    """
    node_types = graph.node_types_new
    lines = []
    edge_id = 1

    for edge in graph.get_edges():
        src = edge.get_source()
        dst = edge.get_destination()
        src_type = node_types.get(src, "unknown")
        dst_type = node_types.get(dst, "unknown")
        lbl = edge.get_label()
        if lbl:
            line = f"e{edge_id}: ({src}, {src_type})\t({dst}, {dst_type})"
        else:
            line = f"e{edge_id}: ({src}, {src_type})\t({dst}, {dst_type})"
        lines.append(line)
        edge_id += 1

    with open(outfile, "w") as fh:
        fh.write("Netlists:\n")
        fh.write("\n".join(lines))
        fh.write("\n")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress design packed file by collapsing register chains.")
    parser.add_argument(
        "-i", "--input_design_packed",
        type=str,
        default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/zircon_residual_relu_fp/bin/design_post_pipe.packed",
        help="Input design packed file"
    )
    parser.add_argument(
        "-o", "--output_design_packed",
        type=str,
        default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/zircon_residual_relu_fp/bin/design_post_pipe_compressed.packed",
        help="Output compressed design packed file"
    )
    args = parser.parse_args()

    g = build_and_collapse_graph(args.input_design_packed)
    export_graph_to_file(g, args.output_design_packed)
    print(f"Compressed graph exported to {args.output_design_packed}")
    # print("\033[93mNOTE: The compression script currently assumes there are no branches from regs in the compute graph. This assumption may not hold in all cases.\033[0m")