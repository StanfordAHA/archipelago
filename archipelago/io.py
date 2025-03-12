import os
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
            graph.node(source, color=colors.get(source[0], "black"))

            # The rest are destinations, if any
            dest_parts = remainder.split("\t")[1:]
            for dest_part in dest_parts:
                dest_full = dest_part.strip("()\n")
                dest, dest_port = [x.strip() for x in dest_full.split(",")]
                # Ensure the destination node is drawn
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
