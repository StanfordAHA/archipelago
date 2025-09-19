import os, re
import shutil
from graphviz import Digraph


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
            if 'regs' in source_port:
                source_label = source_port
                graph.node(source, color=colors.get(source[0], "black"), label=source_label)
            else:
                graph.node(source, color=colors.get(source[0], "black"))
            # breakpoint()

            # The rest are destinations, if any
            dest_parts = remainder.split("\t")[1:]
            for dest_part in dest_parts:
                dest_full = dest_part.strip("()\n")
                dest, dest_port = [x.strip() for x in dest_full.split(",")]
                # Ensure the destination node is drawn
                if 'regs' in dest_port:
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

def generate_packed_from_place_and_route(cwd, place_file, route_file, new_packed_file, visualize=True):
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
        complete_sources = []
        for seg in segments:
            if seg and seg[0][0] in ('reg','port'):
                complete_sources.append(seg[0])
        if len(complete_sources) == 1:
            source_node = complete_sources[0]
            for seg in segments:
                if seg and seg[0][0] == 'SB':
                    seg.insert(0, source_node)
        elif len(complete_sources) > 1:
            for seg in segments:
                if seg and seg[0][0] == 'SB':
                    branch_key = seg[0][1]
                    matched_source = None
                    for comp_seg in segments:
                        if comp_seg and comp_seg[0][0] in ('reg','port'):
                            # Check if any SB node in the complete segment matches branch_key.
                            if any(node[0]=='SB' and node[1]==branch_key for node in comp_seg):
                                matched_source = comp_seg[0]
                                break
                    if matched_source:
                        seg.insert(0, matched_source)
                    else:
                        # Fall back to first complete source.
                        seg.insert(0, complete_sources[0])

    ## Convert each segment's chain into adjacency pairs
    # Only include valid nodes from REG or PORT; skip SB nodes.
    adjacency_netlists = {}
    for net_id, segments in route_nets.items():
        net_pairs = []
        for segment in segments:
            valid_nodes = [node for node in segment if node[0] in ('reg','port')]
            if len(valid_nodes) < 2:
                continue
            for i in range(len(valid_nodes)-1):
                # Now valid_nodes[i] is e.g. ('reg', (block_id, block_name))
                _, (left_id, left_name) = valid_nodes[i]
                _, (right_id, right_name) = valid_nodes[i+1]
                net_pairs.append(((left_id, left_name), (right_id, right_name)))
        adjacency_netlists[net_id] = net_pairs

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
    if visualize:
        _generate_visualization_from_packed(new_packed_file, cwd + "/design_packed_post_pipe")



if __name__ == "__main__":
    packed_filename = "/aha/design_post_pipe_compressed.packed"
    output_base_name = "/aha/design_packed_compressed"
    print(f"Generating visualization from {packed_filename}. The result is placed at {output_base_name}.pdf")
    _generate_visualization_from_packed(packed_filename, output_base_name)