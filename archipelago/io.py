import os, re
import shutil
import argparse
from graphviz import Digraph

# Handle both relative import (when used as module) and absolute import (when run as script)
try:
    from .compress_design_packed import build_and_collapse_graph, export_graph_to_file
except ImportError:
    # Import when running as a standalone script
    from compress_design_packed import build_and_collapse_graph, export_graph_to_file


def dump_packing_result(netlist, bus, filename, id_to_name):
    def tuple_to_str(t_val):
        return "(" + ", ".join([str(val) for val in t_val]) + ")"
    # netlists
    with open(filename, "w+") as f:
        f.write("Netlists:\n")
        net_ids = list(netlist.keys())
        net_ids.sort(key=lambda x: int(x[1:]))
        for net_id in net_ids:
            f.write("{}: ".format(net_id))
            f.write("\t".join([tuple_to_str(entry)
                               for entry in netlist[net_id]]))
            f.write("\n")
        f.write("\n")

        f.write("ID to Names:\n")
        ids = set(id_to_name.keys())
        for _, net in netlist.items():
            for blk_id in net:
                if isinstance(blk_id, (list, tuple)):
                    blk_id = blk_id[0]
                assert isinstance(blk_id, str)
                ids.add(blk_id)
        ids = list(ids)
        ids.sort(key=lambda x: int(x[1:]))
        for blk_id in ids:
            blk_name = str(id_to_name[blk_id]) if blk_id in id_to_name \
                else str(blk_id)
            f.write(str(blk_id) + ": " + blk_name + "\n")

        f.write("\n")
        # registers that have been changed to PE
        f.write("Netlist Bus:\n")
        for net_id in bus:
            f.write(str(net_id) + ": " + str(bus[net_id]) + "\n")


def dump_placement_result(board_pos, filename, id_to_name=None):
    # copied from cgra_pnr
    if id_to_name is None:
        id_to_name = {}
        for blk_id in board_pos:
            id_to_name[blk_id] = blk_id
    blk_keys = list(board_pos.keys())
    blk_keys.sort(key=lambda b: int(b[1:]))
    with open(filename, "w+") as f:
        header = "{0}\t\t\t{1}\t{2}\t\t#{3}\n".format("Block Name",
                                                      "X",
                                                      "Y",
                                                      "Block ID")
        f.write(header)
        f.write("-" * len(header) + "\n")
        for blk_id in blk_keys:
            x, y = board_pos[blk_id]
            f.write("{0}\t\t{1}\t{2}\t\t#{3}\n".format(id_to_name[blk_id],
                                                       x,
                                                       y,
                                                       blk_id))


def load_routing_result(filename):
    # copied from pnr python implementation
    with open(filename) as f:
        lines = f.readlines()

    routes = {}
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index].strip()
        line_index += 1
        if line[:3] == "Net":
            tokens = line.split(" ")
            net_id = tokens[2]
            routes[net_id] = []
            num_seg = int(tokens[-1])
            for seg_index in range(num_seg):
                segment = []
                line = lines[line_index].strip()
                line_index += 1
                assert line[:len("Segment")] == "Segment"
                tokens = line.split()
                seg_size = int(tokens[-1])
                for i in range(seg_size):
                    line = lines[line_index].strip()
                    line_index += 1
                    line = "".join([x for x in line if x not in ",()"])
                    tokens = line.split()
                    tokens = [int(x) if x.isdigit() else x for x in tokens]
                    segment.append(tokens)
                routes[net_id].append(segment)
    return routes


def load_packing_result(filename):
    import pythunder
    netlist, bus_mode = pythunder.io.load_netlist(filename)
    id_to_name = pythunder.io.load_id_to_name(filename)
    return (netlist, bus_mode), id_to_name

def _generate_visualization_from_packed(packed_file, output_basename, label_edges=False):
    """
    Parse the .packed file and create a Graphviz diagram named 'design_packed'
    (by default, 'design_packed.pdf').
    """
    colors = {
        "p": "blue",
        "m": "orange",
        "M": "purple",
        "I": "green",
        "i": "green",
        "r": "red",
    }

    with open(packed_file, 'r') as f:
        lines = f.readlines()

    graph = Digraph()
    read_netlist = False

    for line in lines:
        if line.strip() == "":
            # If there's a blank line, you can decide whether it signals end-of-netlist
            break

        if read_netlist:
            # Example netlist lines:
            # "Netlists:\n"
            # "netA:(m1, in)\t(m2, out)\n"
            edge_id = line.split(":")[0]  # e.g., "netA"
            # The line portion after ":" might have multiple (blk_id, port) segments
            remainder = line.split(":")[1].strip()

            # The first one is the source:
            src_part = remainder.split("\t")[0].strip()
            # src_part might look like "(m1, in)"
            src_full = src_part.strip("()")
            source, source_port = [x.strip() for x in src_full.split(",")]

            # Ensure the source node is drawn
            if 'fifos' in source_port:
                source_label = source_port
                graph.node(source, color=colors.get(source[0], "black"), label=source_label)
            else:
                graph.node(source, color=colors.get(source[0], "black"))

            # The rest are destinations, if any
            dest_parts = remainder.split("\t")[1:]
            for dest_part in dest_parts:
                dest_full = dest_part.strip("()\n")
                dest, dest_port = [x.strip() for x in dest_full.split(",")]
                # Ensure the destination node is drawn
                if 'fifos' in dest_port:
                    dest_label = dest_port
                    graph.node(dest, color=colors.get(dest[0], "black"), label=dest_label)
                else:
                    graph.node(dest, color=colors.get(dest[0], "black"))

                if label_edges:
                    graph.edge(source, dest, label=f"{source_port}->{dest_port}")
                else:
                    graph.edge(source, dest)

        if line.startswith("Netlists:"):
            read_netlist = True

    # Render the diagram. By default, this generates a PDF at output_basename.pdf
    graph.render(filename=output_basename, cleanup=True)

def dump_packed_result(app_name, cwd, inputs, id_to_name, copy_to_dir=None, visualize=True):
    assert inputs is not None
    if id_to_name is None:
        id_to_name = {}
    input_netlist, input_bus = inputs
    assert isinstance(input_netlist, dict)
    netlist = {}
    for net_id, net in input_netlist.items():
        assert isinstance(net, list)
        for entry in net:
            assert len(entry) == 2, "entry in the net has to be " \
                                    "(blk_id, port)"
        netlist[net_id] = net
    # dump the packed file
    packed_file = os.path.join(cwd, app_name + ".packed")
    dump_packing_result(netlist, input_bus, packed_file, id_to_name)

    # copy file over
    if copy_to_dir is not None:
        shutil.copy2(packed_file, copy_to_dir)

    # visualize the packed file
    if visualize:
        graph_output = os.path.join(cwd, "design_packed")
        _generate_visualization_from_packed(packed_file, graph_output)

    return packed_file


def dump_meta_file(halide_src, app_name, cwd):
    bn = os.path.basename
    dn = os.path.dirname
    halide_name = bn(dn(dn(halide_src)))
    with open(os.path.join(cwd, "{0}.meta".format(app_name)), "w+") as f:
        f.write("placement={0}.place\n".format(app_name))
        f.write("bitstream={0}.bs\n".format(halide_name))
        if os.path.exists(os.path.join(cwd, 'bin/input.raw')):
            ext = '.raw'
        else:
            ext = '.pgm'
        f.write(f"input=input{ext}\n")
        if os.path.exists(os.path.join(cwd, 'bin/gold.raw')):
            ext = '.raw'
        else:
            ext = '.pgm'
        f.write(f"output=gold{ext}\n")


def add_unique_path(path, new_segments):
    def is_sublist(a, b):
        n, m = len(a), len(b)
        return any(a == b[i:i+n] for i in range(m - n + 1))

    for existing_path in new_segments:
        if is_sublist(path, existing_path):
            return

        if is_sublist(existing_path, path):
            new_segments.remove(existing_path)
    new_segments.append(path)

def generate_packed_from_place_and_route(cwd, place_file, route_file, new_packed_file, new_compressed_packed_file, visualize=True):
    """
    Generate design packed file from design.place and design.route.

    1. For complete segments (those starting with REG or PORT) the connection is built directly
    2. If a segment starts with SB, it is considered a branch
       We search within the same net for a complete segment that contains the same SB node
       If found, we prepend the source node from that complete segment so that the branch
       now has a proper starting point.
    """
    ## Parse design.place => place_map and id_to_names
    place_map = {}
    id_to_names = {}

    with open(place_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith('-'):
            continue

        # Expecting format: BlockName, X, Y, "#BlockID"
        tokens = re.split(r"\t+", line)
        if len(tokens) < 4:
            continue

        block_name = tokens[0]
        x_str = tokens[1]
        y_str = tokens[2]
        block_id_str = tokens[3]

        if not block_id_str.startswith('#'):
            continue

        block_id = block_id_str[1:]  # remove '#'
        try:
            x = int(x_str)
            y = int(y_str)
        except ValueError:
            continue

        # Parse track_side from block_name by looking for "@T"
        track_side = None
        at_idx = block_name.find('@T')
        if at_idx != -1:
            track_side = block_name[at_idx+1:]  # e.g. "T4_NORTH"

        # New mapping: key = (block_id, block_name) ; value = (x, y, track_side)
        place_map[(block_id, block_name)] = (x, y, track_side)
        id_to_names[block_id] = block_name

    ## Parse design.route => route_nets with segments and branch (SB) nodes.
    # route_nets: dictionary mapping net_id to a list of segments.
    # Each segment is a list of nodes. A node is a tuple:
    #    - For PORT lines: ('port', (block_id, route_name))
    #    - For REG lines: ('reg', (block_id, block_name))
    #    - For SB lines:  ('SB', sb_coords)
    route_nets = {}
    current_net = None
    current_segment = None

    with open(route_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("Net ID:"):
            # e.g., "Net ID: e127 Segment Size: 2"
            m = re.match(r'^Net ID:\s+(\S+)', line)
            if m:
                current_net = m.group(1)
                route_nets[current_net] = []  # list of segments
                current_segment = None
            continue

        if line.startswith("Segment:"):
            # e.g., "Segment: 0 Size: 6"
            current_segment = []
            route_nets[current_net].append(current_segment)
            continue

        if current_net is None:
            continue

        # Process PORT lines
        if line.startswith("PORT "):
            # e.g., "PORT PE_output_width_17_num_1 (26,16,17)"
            pm = re.match(r'^PORT\s+(\S+)\s*\((\d+),\s*(\d+),\s*(\d+)\)', line)
            if pm:
                route_name = pm.group(1)
                x = int(pm.group(2))
                y = int(pm.group(3))
                # For IO/PE/MEM blocks, we expect track_side == None.
                candidates = [
                    (bid, bname)
                    for (bid, bname), (xx, yy, ts) in place_map.items()
                    if xx == x and yy == y and ts is None
                ]
                if candidates:
                    if len(candidates) == 1:
                        candidate_key = candidates[0]
                    else:
                        # When multiple candidates exist, use the port name to decide:
                        # if route_name contains "_17_", choose candidate with block_id starting with "I"
                        # else choose candidate with block_id starting with "i".
                        # This should only happen to output IOs.
                        if "_17_" in route_name:
                            candidate_key = next(((bid, bname) for (bid, bname) in candidates if bid.startswith("I")), None)
                        else:
                            candidate_key = next(((bid, bname) for (bid, bname) in candidates if bid.startswith("i")), None)
                        if candidate_key is None:
                            candidate_key = candidates[0]
                    # Ensure we are in a segment.
                    if current_segment is None:
                        current_segment = []
                        route_nets[current_net].append(current_segment)
                    current_segment.append(('port', (candidate_key[0], route_name)))
            continue

        # Process REG lines
        elif line.startswith("REG "):
            # e.g., "REG T0_EAST (0, 12, 5, 17)"
            rm = re.match(r'^REG\s+(\S+)\s*\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)', line)
            if rm:
                track_side = rm.group(1)
                x = int(rm.group(3))
                y = int(rm.group(4))
                # Lookup using x, y, and track_side.
                candidates = [
                    (bid, bname)
                    for (bid, bname), (xx, yy, ts) in place_map.items()
                    if xx == x and yy == y and ts == track_side
                ]
                if candidates:
                    candidate_key = candidates[0]
                    if current_segment is None:
                        current_segment = []
                        route_nets[current_net].append(current_segment)
                    current_segment.append(('reg', (candidate_key[0], candidate_key[1])))
                else:
                    print(f"Warning: Could not find place for REG at ({x},{y},{track_side})")
                    if current_segment is None:
                        current_segment = []
                        route_nets[current_net].append(current_segment)
                    current_segment.append(('reg', (f"unknown_reg_{x}_{y}_{track_side}", f"reg_{x}_{y}_{track_side}")))
            continue

        # Process SB lines (branch indicator lines)
        elif line.startswith("SB "):
            # e.g., "SB (0, 18, 5, 2, 0, 17)"
            m = re.match(r'^SB\s*\(([^)]+)\)', line)
            if m:
                coords_str = m.group(1)  # e.g., "0, 18, 5, 2, 0, 17"
                try:
                    coords = tuple(int(x.strip()) for x in coords_str.split(','))
                except ValueError:
                    coords = None
                if coords is not None:
                    if current_segment is None:
                        current_segment = []
                        route_nets[current_net].append(current_segment)
                    current_segment.append(('SB', coords))
            continue

        # Ignore RMUX and other lines
        else:
            continue

    ## Post-processing: Backtrace branch segments
    # For each net, determine the complete segments whose first node is REG or PORT
    # Then for each branch segment (one starting with SB), resolve its source as follows:
    #   If there is exactly one complete source in the net, prepend that source to every branch segment.
    #   If multiple complete sources exist, try to match the branch's first SB value with any SB node
    #   in a complete segment; if found, use that complete segment's first node.
    for net_id, segments in route_nets.items():
        new_segments = segments.copy()
        complete_sources = []
        complete_source_indices = []

        for i in range(len(segments)):
            seg = segments[i]
            if seg and seg[0][0] in ('reg','port'):
                complete_sources.append(seg)
                complete_source_indices.append(i)


        if len(complete_sources) == 1:
            for seg in segments:
                if seg and seg[0][0] == 'SB':
                    branch_key = seg[0][1]

                    # Source path is complete_sources[0] up to branch_key
                    source_path = complete_sources[0]
                    paths_to_try_matching = [source_path]
                    # Add all other segments that are not "seg" to paths_to_try_matching
                    for i in range(len(segments)):
                        if segments[i] is not seg:
                            paths_to_try_matching.append(segments[i])

                    # Try to find the matching SB node in any of the paths_to_try_matching
                    matched = False
                    for path in paths_to_try_matching:
                        try:
                            source_path = path
                            for i, node in enumerate(source_path):
                                if node[0] == 'SB' and node[1] == branch_key:
                                    if i == 0:
                                        # This means the source_path itself is a branch; skip it.
                                        continue
                                    new_child_path = source_path[i-1:]
                                    source_path = source_path[:i]

                                    # Remove old paths before adding new ones
                                    if seg in new_segments:
                                        new_segments.remove(seg)
                                    if source_path in new_segments:
                                        new_segments.remove(source_path)

                                    raise StopIteration
                        except StopIteration:
                            matched = True
                            break

                    assert matched, "Could not find matching SB node in any source"

                    seg.insert(0, source_path[-1])

                    # We have modified the original parent segment to stop at the branching point. We also add a new segment to the rest of original parent segment
                    # add_unique_path(seg, new_segments)
                    add_unique_path(new_child_path, new_segments)
                    add_unique_path(source_path, new_segments)
                add_unique_path(seg, new_segments)

            route_nets[net_id] = new_segments

        elif len(complete_sources) > 1:
            for seg in segments:
                if seg and seg[0][0] == 'SB':
                    branch_key = seg[0][1]
                    matched_source_index = None

                    for i in range(len(complete_sources)):
                        comp_seg = complete_sources[i]
                        # Check if any SB node in the complete segment matches branch_key.
                        if any(node[0]=='SB' and node[1]==branch_key for node in comp_seg):
                            matched_source_index = i
                            break

                    if matched_source_index is None:
                        matched_source_index = 0 # Fall back to first complete source.

                    source_path = complete_sources[matched_source_index]

                    paths_to_try_matching = [source_path]
                    # Add all other segments that are not "seg" to paths_to_try_matching
                    for i in range(len(segments)):
                        if segments[i] is not seg:
                            paths_to_try_matching.append(segments[i])

                    # Try to find the matching SB node in any of the paths_to_try_matching
                    matched = False
                    for path in paths_to_try_matching:
                        try:
                            source_path = path
                            for i, node in enumerate(source_path):
                                if node[0] == 'SB' and node[1] == branch_key:
                                    if i == 0:
                                        # This means the source_path itself is a branch; skip it.
                                        continue
                                    new_child_path = source_path[i-1:]
                                    source_path = source_path[:i]

                                    # Remove old paths before adding new ones
                                    if seg in new_segments:
                                        new_segments.remove(seg)
                                    if source_path in new_segments:
                                        new_segments.remove(source_path)

                                    raise StopIteration
                        except StopIteration:
                            matched = True
                            break

                    assert matched, "Could not find matching SB node in any source"


                    # Prepend the branching point to child paths
                    seg.insert(0, source_path[-1])

                    # add_unique_path(seg, new_segments)
                    add_unique_path(new_child_path, new_segments)
                    add_unique_path(source_path, new_segments)
                add_unique_path(seg, new_segments)

            route_nets[net_id] = new_segments

    ## Convert each segment's chain into adjacency pairs
    # Only include valid nodes from REG or PORT; skip SB nodes.
    adjacency_netlists = {}
    for net_id, segments in route_nets.items():
        net_pairs = set()
        for segment in segments:
            valid_nodes = [node for node in segment if node[0] in ('reg','port')]
            if len(valid_nodes) < 2:
                continue
            for i in range(len(valid_nodes)-1):
                # Now valid_nodes[i] is e.g. ('reg', (block_id, block_name))
                _, (left_id, left_name) = valid_nodes[i]
                _, (right_id, right_name) = valid_nodes[i+1]
                net_pairs.add(((left_id, left_name), (right_id, right_name)))
        adjacency_netlists[net_id] = list(net_pairs)

    ## Write out design packed file
    with open(new_packed_file, 'w') as f:
        f.write("Netlists:\n")
        for net_id, adj_pairs in adjacency_netlists.items():
            if not adj_pairs:
                f.write(f"{net_id}:\n")
                continue
            for (L, R) in adj_pairs:
                (Lid, Lname) = L
                (Rid, Rname) = R
                f.write(f"{net_id}: ({Lid}, {Lname})\t({Rid}, {Rname})\n")
        f.write("\n")
        f.write("ID to Names:\n")
        all_ids = sorted(id_to_names.keys())
        for bid in all_ids:
            f.write(f"{bid}: {id_to_names[bid]}\n")
        f.write("\n")

    print(f"Wrote post-pipelining design_packed to {new_packed_file}.")

    g = build_and_collapse_graph(new_packed_file)
    export_graph_to_file(g, new_compressed_packed_file)
    print(f"Wrote post-pipelining compressed design_packed to {new_compressed_packed_file}.")

    if visualize:
        _generate_visualization_from_packed(new_packed_file, cwd + "/design_packed_post_pipe")
        _generate_visualization_from_packed(new_compressed_packed_file, cwd + "/design_packed_post_pipe_compressed")



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Generate visualization from design packed file.")
    # parser.add_argument(
    #     "-i", "--input_design_packed",
    #     type=str,
    #     default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/zircon_deq_ResReLU_quant_fp/bin/design_post_pipe_compressed.packed",
    #     help="Input design packed file"
    # )
    # parser.add_argument(
    #     "-o", "--output_pdf",
    #     type=str,
    #     default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/zircon_deq_ResReLU_quant_fp/bin/design_post_pipe_compressed",
    #     help="Output PDF file base name (without .pdf extension)"
    # )
    # args = parser.parse_args()

    # print(f"Generating visualization from {args.input_design_packed}. The result is placed at {args.output_pdf}.pdf")
    # _generate_visualization_from_packed(args.input_design_packed, args.output_pdf)


    cwd = "/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/get_apply_e8m0_scale_fp/"
    place_file = "/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/get_apply_e8m0_scale_fp/bin/design.place"
    route_file = "/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/get_apply_e8m0_scale_fp/bin/design.route"
    new_packed_file = "/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/get_apply_e8m0_scale_fp/bin/design_post_pipe_new.packed"
    new_compressed_packed_file = "/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/get_apply_e8m0_scale_fp/bin/design_post_pipe_compressed_new.packed"
    generate_packed_from_place_and_route(cwd, place_file, route_file, new_packed_file, new_compressed_packed_file, visualize=True)

