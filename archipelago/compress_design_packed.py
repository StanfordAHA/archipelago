import re
import pydot
from collections import defaultdict

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
            regs_seen = set()
            endpoints = set()
            stack = [dst]
            while stack:
                cur = stack.pop()
                if cur in regs_seen:
                    continue
                if not is_reg(cur):
                    continue
                regs_seen.add(cur)
                for succ, _ in succs.get(cur, []):
                    if is_reg(succ):
                        stack.append(succ)
                    else:
                        endpoints.add(succ)

            reg_count = len(regs_seen)
            collapsed_name = f"r{collapsed_counter}"
            collapsed_type = f"{reg_count} regs"
            collapsed_counter += 1

            add_node(collapsed_name, collapsed_type)

            # edge src -> collapsed
            key_in = (src, collapsed_name, edge_label)
            if key_in not in added_edges:
                new_graph.add_edge(pydot.Edge(src, collapsed_name, label=edge_label))
                added_edges.add(key_in)

            # edges collapsed -> endpoints
            for end in endpoints:
                if is_reg(end):
                    continue
                add_node(end, node_types.get(end, "unknown"))
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
    packed_filename = "/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/zircon_residual_relu_fp/bin/design_post_pipe.packed"
    g = build_and_collapse_graph(packed_filename)
    output_filename = "/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/zircon_residual_relu_fp/bin/design_post_pipe_compressed.packed"
    export_graph_to_file(g, output_filename)
    print(f"Compressed graph exported to {output_filename}")
    print("\033[93mNOTE: The compression script currently assumes there are no branches from regs in the compute graph. This assumption may not hold in all cases.\033[0m")