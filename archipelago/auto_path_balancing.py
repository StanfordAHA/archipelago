import re
import pydot
import argparse
from collections import defaultdict, deque
import itertools
import json
import re


class Path:
    def __init__(self, nodes_edges=[]):
        self.nodes_edges = nodes_edges  # list of (node, edge_name) tuples
        self.nodes = [node for node, edge in nodes_edges]
        self.interconnect_fifo_count = 0
        self.pond_behavioral_fifo_count = 0
        self.pe_fifo_count = 0
        self.total_fifo_count = 0
        self.pe_count = 0

    def add_node_edge(self, node, edge_name):
        self.nodes_edges.append((node, edge_name))
        self.nodes.append(node)

    def get_nodes_edges(self):
        return self.nodes_edges

    def get_nodes(self):
        return self.nodes

    def print_path(self):
        print(" -> ".join([f"{node}({edge})" if edge else f"{node}" for node, edge in self.nodes_edges]))

    def get_source(self):
        if self.nodes_edges:
            return self.nodes_edges[0][0]
        return None

    def get_destination(self):
        if self.nodes_edges:
            return self.nodes_edges[-1][0]
        return None

    def get_total_fifo_count(self):
        return self.interconnect_fifo_count + self.pond_behavioral_fifo_count + self.pe_fifo_count

    def get_pond_behavioral_fifo_count(self):
        return self.pond_behavioral_fifo_count

    def set_pond_behavioral_fifo_count(self, count):
        self.pond_behavioral_fifo_count = count

    def update_pond_behavioral_fifo_count(self, path_balance_metadata):
        total_pond_fifos = 0
        for node, edge in self.nodes_edges:
            if node.startswith("p"):
                if node in path_balance_metadata["balance_lengths"]:
                    total_pond_fifos += path_balance_metadata["balance_lengths"][node]
        self.pond_behavioral_fifo_count = total_pond_fifos

    def update_interconnect_fifo_count(self):
        interconnect_fifo_count = sum(1 for node, edge in self.nodes_edges if node.startswith("r"))
        self.interconnect_fifo_count = interconnect_fifo_count


    # Do not count PE output FIFOs if PE is destination
    # Do not count PE input FIFOs if PE is source
    def update_pe_fifo_count(self, pe_bypass_config, edge_dict):
        pe_fifo_count = 0
        for node, edge in self.nodes_edges:
            if node.startswith("p"):
                pe_num_active_fifos = 3
                # Handle this based on which specific input FIFOs are bypassed
                if node in pe_bypass_config["input_fifo_bypass"] or node == self.get_source():
                    pe_input_num = int(edge_dict[edge][1][1].split("PE_input_width_17_num_")[1])
                    if pe_bypass_config["input_fifo_bypass"][node][pe_input_num] == 1 or node == self.get_source():
                        pe_num_active_fifos -= 1
                if node in pe_bypass_config["output_fifo_bypass"] or node == self.get_destination():
                    pe_num_active_fifos -= 1
                if node in pe_bypass_config["prim_outfifo_bypass"] or node == self.get_destination():
                    pe_num_active_fifos -= 1
                pe_fifo_count += pe_num_active_fifos
        self.pe_fifo_count = pe_fifo_count

    def get_pe_count(self):
        return self.pe_count

    def update_pe_count(self):
        pe_count = sum(1 for node, edge in self.nodes_edges if node.startswith("p"))
        self.pe_count = pe_count


class ReconvergenceGroup:
    def __init__(self, source, destination, paths=[]):
        self.source = source
        self.destination = destination
        self.paths = paths
        self.max_fifo_count = 0  # to be computed later
        self.broadcast_edges = set()
        self.join_edges = set()

    def add_path(self, path):
        self.paths.append(path)

    def get_paths(self):
        return self.paths

    def get_source(self):
        return self.source

    def get_destination(self):
        return self.destination

    def set_max_fifo_count(self, count):
        self.max_fifo_count = count

    def add_broadcast_edge(self, edge):
        self.broadcast_edges.add(edge)

    def add_join_edge(self, edge):
        self.join_edges.add(edge)

    def get_broadcast_edges(self):
        return self.broadcast_edges

    def get_join_edges(self):
        return self.join_edges

class Node:
    def __init__(self, name):
        self.name = name
        self.parents = []    # list of Node objects
        self.children = []   # list of Node objects

    def __repr__(self):
        return f"Node({self.name})"


def extract_id_to_name(filename):
    result = {}

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue  # skip empty lines or malformed lines

            key, value = line.split(':', 1)  # split only on first colon
            key = key.strip()
            value = value.strip()
            result[key] = value

    return result

def build_edge_dict(packed_filename):
    netlists = {}
    in_netlist_section = False

    with open(packed_filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Detect start of the Netlists section
            if line.startswith("Netlists:"):
                in_netlist_section = True
                continue

            # Detect the end of the Netlists section
            if in_netlist_section and (line.startswith("ID to Names:") or line.startswith("Netlist Bus:")):
                break

            # Parse netlist lines like:
            # e1: (r1, reg)    (I0, f2io_17_0)
            if in_netlist_section and ":" in line:
                match = re.match(r"(\w+):\s*(\(.*\))\s*\(.*\)", line)
                if not match:
                    # Split manually if regex fails
                    edge, rest = line.split(":", 1)
                    tuples = re.findall(r"\(([^)]+)\)", rest)
                else:
                    edge = match.group(1)
                    tuples = re.findall(r"\(([^)]+)\)", line)

                # Each tuple string looks like "r1, reg" → split by comma
                parsed_tuples = [tuple(s.strip().split(", ")) for s in tuples]
                netlists[edge] = parsed_tuples

    return netlists

def build_adjacency(filename: str):
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

    return succs


def build_parent_child_node_info(adjacency):
    # Step 1: Create Node objects for all unique node names
    nodes = {}

    # Ensure all nodes exist (both parents and children)
    for parent, edges in adjacency.items():
        if parent not in nodes:
            nodes[parent] = Node(parent)
        for child, _ in edges:
            if child not in nodes:
                nodes[child] = Node(child)

    # Step 2: Populate parents and children lists
    for parent, edges in adjacency.items():
        parent_node = nodes[parent]
        for child, _ in edges:
            child_node = nodes[child]
            parent_node.children.append(child_node)
            child_node.parents.append(parent_node)

    return nodes  # dictionary: {name: Node object}

def find_start_and_end_nodes(graph):
    """
    Finds start nodes (no incoming edges) and end nodes (no outgoing edges)
    for a graph where adjacency values are lists of (next_node, edge_name) tuples.
    """
    all_nodes = set(graph.keys())
    all_successors = {dst for edges in graph.values() for (dst, _) in edges}

    # Include successors in the total set of nodes
    all_nodes |= all_successors

    start_nodes = all_nodes - all_successors          # no incoming edges
    end_nodes = all_nodes - set(graph.keys())         # no outgoing edges

    return start_nodes, end_nodes




def find_start_and_end_nodes_intra_graph(graph):
    """
    Finds:
      - Start nodes: nodes with no incoming edges OR multiple outgoing edges
      - End nodes: nodes with no outgoing edges OR multiple incoming edges

    Graph format: dict[node] = [(next_node, edge_name), ...]
    """

    # --- Step 1: Gather node sets ---
    all_nodes = set(graph.keys())
    all_successors = {dst for edges in graph.values() for (dst, _) in edges}
    all_nodes |= all_successors  # include all nodes that appear as targets

    # --- Step 2: Count incoming and outgoing edges ---
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    for src, edges in graph.items():
        out_degree[src] += len(edges)
        for (dst, _) in edges:
            in_degree[dst] += 1

    # Ensure every node appears in both degree dicts
    for n in all_nodes:
        in_degree.setdefault(n, 0)
        out_degree.setdefault(n, 0)

    # --- Step 3: Identify node sets ---
    start_nodes = {n for n in all_nodes if in_degree[n] == 0 or out_degree[n] > 1}
    end_nodes   = {n for n in all_nodes if out_degree[n] == 0 or in_degree[n] > 1}

    return start_nodes, end_nodes

# def find_start_and_end_nodes_intra_graph(graph):
#     """
#     Finds:
#       - Start nodes:
#           * Nodes with no incoming edges
#           * Nodes with multiple outgoing edges
#           * Nodes with multiple incoming edges AND at least one outgoing edge
#       - End nodes:
#           * Nodes with no outgoing edges
#           * Nodes with multiple incoming edges

#     The graph is an adjacency dictionary where values are lists of (next_node, edge_name) tuples.
#     """
#     # Gather all nodes (both keys and successors)
#     all_nodes = set(graph.keys())
#     all_successors = {dst for edges in graph.values() for (dst, _) in edges}
#     all_nodes |= all_successors

#     # Compute in-degree and out-degree
#     in_degree = {node: 0 for node in all_nodes}
#     out_degree = {node: 0 for node in all_nodes}

#     for src, edges in graph.items():
#         out_degree[src] += len(edges)
#         for (dst, _) in edges:
#             in_degree[dst] += 1

#     # Define start and end sets
#     start_nodes = {
#         n for n in all_nodes
#         if in_degree[n] == 0
#         or out_degree[n] > 1
#         or (in_degree[n] > 1 and out_degree[n] > 0)
#     }

#     end_nodes = {
#         n for n in all_nodes
#         if out_degree[n] == 0 or in_degree[n] > 1
#     }

#     return start_nodes, end_nodes


def find_all_paths(graph, intra_graph_effort=0):
    """
    Finds all possible paths from every start node to every end node.
    Each path is returned as a Path instance containing (node, edge_name) tuples,
    with the final node having edge_name == None.
    """
    if intra_graph_effort == 0:
        start_nodes, end_nodes = find_start_and_end_nodes(graph)
    elif intra_graph_effort == 1:
        start_nodes, end_nodes = find_start_and_end_nodes_intra_graph(graph)
    else:
        raise ValueError("intra_graph_effort must be 0 or 1.")

    # Print start and end nodes
    print("Start nodes found:", start_nodes)
    print("End nodes found:", end_nodes)
    all_paths = []

    def dfs(current_node, this_end, path_obj: Path):
        # If this node has no outgoing edges (end node)
        if current_node == this_end:
            path_obj.add_node_edge(current_node, None)
            path_obj.update_interconnect_fifo_count()
            all_paths.append(path_obj)
            return

        for (neighbor, edge_name) in graph[current_node]:
            # Copy current path and extend with this hop
            new_nodes_edges = list(path_obj.get_nodes_edges())
            new_path = Path(new_nodes_edges)
            new_path.add_node_edge(current_node, edge_name)
            dfs(neighbor, this_end, new_path)

    for start in start_nodes:
        for this_end in end_nodes:
            if start == this_end:
                continue
            print(f"Finding paths from {start} to {this_end}...")
            dfs(start, this_end, Path([]))


    return all_paths


def find_all_paths_saved(graph, intra_graph_effort=0):
    """
    Finds all possible paths from every start node to every end node.
    Each path is returned as a Path instance containing (node, edge_name) tuples,
    with the final node having edge_name == None.
    """
    if intra_graph_effort == 0:
        start_nodes, end_nodes = find_start_and_end_nodes(graph)
    elif intra_graph_effort == 1:
        start_nodes, end_nodes = find_start_and_end_nodes_intra_graph(graph)
    else:
        raise ValueError("intra_graph_effort must be 0 or 1.")

    # Print start and end nodes
    print("Start nodes found:", start_nodes)
    print("End nodes found:", end_nodes)
    all_paths = []

    def dfs(current_node, path_obj: Path):
        # If this node has no outgoing edges (end node)
        if current_node in end_nodes or current_node not in graph:
            path_obj.add_node_edge(current_node, None)
            path_obj.update_interconnect_fifo_count()
            all_paths.append(path_obj)
            return

        for (neighbor, edge_name) in graph[current_node]:
            # Copy current path and extend with this hop
            new_nodes_edges = list(path_obj.get_nodes_edges())
            new_path = Path(new_nodes_edges)
            new_path.add_node_edge(current_node, edge_name)
            dfs(neighbor, new_path)

    for start in start_nodes:
        # Remove this node from the end nodes
        node_removed = False
        if start in end_nodes:
            node_removed = True
            end_nodes.remove(start)
        dfs(start, Path([]))
        # Re-add the node to end nodes if it was removed
        if node_removed:
            end_nodes.add(start)

    return all_paths


def closest_sum(target_sum, choices, effort_level):
    best_combo = None
    best_diff = float('inf')
    best_sum = None

    # Explore using 1 to effort_level operands (allowing reuse)
    for k in range(1, effort_level + 1):
        for combo in itertools.combinations_with_replacement(choices, k):
            s = sum(combo)
            diff = abs(target_sum - s)
            if diff < best_diff:
                best_diff = diff
                best_combo = combo
                best_sum = s
            # Stop early if exact match
            if diff == 0:
                return list(best_combo)

    return list(best_combo)



def get_io_reconvergence_groups(all_paths, E64_mode=False, Multi_bank_mode=False, id_to_name=None, pe_bypass_config=None, edge_dict=None):
    """
    Finds io reconvergence groups from the provided paths.
    """

    if Multi_bank_mode:
        assert E64_mode, "E64_mode must be True if Multi_bank_mode is True."

    if E64_mode or Multi_bank_mode:
        assert id_to_name is not None, "id_to_name mapping must be provided in E64 or Multi-bank mode."


    # Create empty list of ReconvergenceGroups
    reconvergence_groups = []

    for path in all_paths:
        path.update_interconnect_fifo_count()
        path.update_pe_count()
        path.update_pe_fifo_count(pe_bypass_config, edge_dict)
        path_source = path.get_source()
        path_destination = path.get_destination()

        # Treat all MU I/Os as the same source
        if path_source.startswith("U") or path_source.startswith("V"):
            path_source = "MU"

        # # FIXME: Temporary HACK to skip GLB inputs
        # else:
        #     continue

        # Group I/Os using same GLB tile in E64 or Multi-bank mode
        # Handle input I/Os
        if E64_mode and path_source.startswith("I"):
            source_node_name = id_to_name.get(path_source)
            source_node_name_parse_list = source_node_name.split("stencil_")[2].split("_read")
            source_lane_idx = int(source_node_name_parse_list[0]) if len(source_node_name_parse_list) > 1 else 0

            if Multi_bank_mode:
                path_source = f"GLB_input_group_{source_lane_idx // 8}"
            elif E64_mode:
                path_source = f"GLB_input_group_{source_lane_idx // 4}"

        # Handle output I/Os
        if E64_mode and path_destination.startswith("I"):
            destination_node_name = id_to_name.get(path_destination)
            destination_node_name_parse_list = destination_node_name.split("stencil_")[2].split("_write")
            destination_lane_idx = int(destination_node_name_parse_list[0]) if len(destination_node_name_parse_list) > 1 else 0

            if Multi_bank_mode:
                path_destination = f"GLB_output_group_{destination_lane_idx // 8}"
            elif E64_mode:
                path_destination = f"GLB_output_group_{destination_lane_idx // 4}"

        # Check if there is an existing reconvergence group with this source, destination pair
        # TODO: Also check that there is no edge overlap for outputs (i.e. going into destinatiion)
        # TODO: And there MUST be edge overlap for inputs (if it's not an I/O tile or MU I/O) (i.e. coming from source)
        source_is_IO_tile = path_source.startswith("GLB_input_group_") or path_source == "MU" or path_source.startswith("I")
        for group in reconvergence_groups:
            enforced_input_edge_overlap = path.nodes_edges[0][1] in group.get_broadcast_edges()
            if group.get_source() == path_source and group.get_destination() == path_destination and (source_is_IO_tile or enforced_input_edge_overlap):
                if not(source_is_IO_tile):
                    # if len(group.get_broadcast_edges()) != 1:
                    #     breakpoint()
                    assert len(group.get_broadcast_edges()) == 1, "There should only be one broadcast edge in the group."

                # if not(path.nodes_edges[-2][1] in group.get_join_edges()):    # Ensure no edge overlap for outputs (i.e. going into destination)
                #     group.get_paths().append(path)
                #     # Add join edge
                #     group.add_join_edge(path.nodes_edges[-2][1])
                #     # Add broadcast edge
                #     group.add_broadcast_edge(path.nodes_edges[0][1])
                #     break


                group.get_paths().append(path)
                # Add join edge
                group.add_join_edge(path.nodes_edges[-2][1])
                # Add broadcast edge
                group.add_broadcast_edge(path.nodes_edges[0][1])
                break
        else:
            # If not, create a new group
            new_reconvergence_group = ReconvergenceGroup(path_source, path_destination, [path])
            # Add broadcast edge
            new_reconvergence_group.add_broadcast_edge(path.nodes_edges[0][1])
            # Add join edge
            new_reconvergence_group.add_join_edge(path.nodes_edges[-2][1])
            reconvergence_groups.append(new_reconvergence_group)

    # Trim reconvergence groups with only one path
    reconvergence_groups = [group for group in reconvergence_groups if len(group.get_paths()) > 1]
    return reconvergence_groups


def reconvergence_group_greater_than(rg1: ReconvergenceGroup, rg2: ReconvergenceGroup):
    """
    Returns true if rg1 is "greater than" rg2, meaning that rg2 is contained within rg1.
    """
    rg2_source = rg2.get_source()
    rg2_dest = rg2.get_destination()

    for path in rg1.get_paths():
        if (rg2_source in path.get_nodes() and rg2_source != path.get_source()) \
        or (rg2_dest in path.get_nodes() and rg2_dest != path.get_destination()):
                print(f"  Reconvergence Group from {rg2_source} to {rg2_dest} is inside Reconvergence Group from {rg1.get_source()} to {rg1.get_destination()}.")
                return True

    return False


def key_contained_in_rg_to_left(key, arr, j):
    for i in range(j, -1, -1):
        if reconvergence_group_greater_than(arr[i], key):
            return True
    return False


def insertion_sort_reconvergence_groups(arr):
    # Traverse from the second element to the end
    for i in range(1, len(arr)):
        key = arr[i]          # Element to be inserted
        j = i - 1

        # Move elements of arr[0..i-1], that are greater than key,
        # one position ahead to make space for the key

        # Greater than key means that the key reconvergence group is inside the arr[j] reconvergence group
        # while j >= 0 and arr[j] > key:
        # while j >= 0 and reconvergence_group_greater_than(arr[j], key):
        while j >= 0 and key_contained_in_rg_to_left(key, arr, j):
            arr[j + 1] = arr[j]
            j -= 1

        # Place the key in its correct position
        arr[j + 1] = key

def merge_sort_reconvergence_groups(arr):
    # Base case: a list of 0 or 1 elements is already sorted
    if len(arr) <= 1:
        return arr

    # Split the list into two halves
    mid = len(arr) // 2
    left_half = merge_sort_reconvergence_groups(arr[:mid])
    right_half = merge_sort_reconvergence_groups(arr[mid:])

    # Merge the sorted halves
    return merge_rg(left_half, right_half)


def merge_rg(left, right):
    """Merge two sorted lists into one sorted list."""
    merged = []
    i = j = 0
    # breakpoint()

    # Compare elements from both halves and pick the smaller one
    while i < len(left) and j < len(right):
        left_rg_inside_right_rg = False
        left_source = left[i].get_source()
        left_dest = left[i].get_destination()

        # Debugging
        # if left[i].get_destination() == "GLB_output_group_1":
        #     breakpoint()


        # Check if left_source or dest is in right[j]'s paths nodes
        for rpath in right[j].get_paths():
            if (left_source in rpath.get_nodes() and left_source != rpath.get_source()) \
            or (left_dest in rpath.get_nodes() and left_dest != rpath.get_destination()):
                    print(f"  Reconvergence Group from {left_source} to {left_dest} is inside Reconvergence Group from {right[j].get_source()} to {right[j].get_destination()}.")
                    # breakpoint()
                    left_rg_inside_right_rg = True
                    break
        # left_rg_inside_right_rg = left[i] <= right[j]

        if left_rg_inside_right_rg: # TODO Need to change this condition
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    # Add remaining elements from both halves
    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged


def append_reconvergence_group_position_metadata(reconvergence_group, parent_child_node_info):
    # Compute source distance_to_graph_source
    source_node = reconvergence_group.get_source()
    dist_to_source = 0
    dest_node = reconvergence_group.get_destination()
    dist_to_dest = 0




def get_intra_graph_reconvergence_groups(graph, E64_mode=False, Multi_bank_mode=False, id_to_name=None):
    raise NotImplementedError("This function is not yet implemented.")


def update_path_balance_metadata(path_balance_metadata, path, parent_child_node_info, fifo_deficit, total_stream_length, id_to_name, effort_level):
    balance_length_choices = []
    MAX_BALANCE_LENGTH = 16
    for i in range(2, MAX_BALANCE_LENGTH + 1):  # check only factors between 2 and 16
        if total_stream_length % i == 0:
            balance_length_choices.append(i)

    chosen_balance_lengths = closest_sum(fifo_deficit, balance_length_choices, effort_level)

    # Update path metadata with info about added ponds
    # curr_path_behavioral_fifo_count = path.get_pond_behavioral_fifo_count()
    # path.set_pond_behavioral_fifo_count(curr_path_behavioral_fifo_count + sum(chosen_balance_lengths))

    # add the ponds with chosen balance lengths to the path, starting from the destination
    num_ponds_added = 0
    for node, edge in reversed(path.get_nodes_edges()):
        if node.startswith("p"):



            if node in path_balance_metadata["balance_lengths"]:
                continue  # Already added pond here. Can't add multiple ponds to same PE

            # TODO: Also need to check if the PE has multiple inputs. If so, just skip it for now
            # The condition is really if multiple lanes converge into the same PE
            if len(parent_child_node_info[node].parents) > 1:
                continue  # skip this PE

            path_balance_metadata["balance_lengths"][node] = chosen_balance_lengths[num_ponds_added]
            path_balance_metadata["total_stream_lengths"][node] = total_stream_length
            node_full_name = id_to_name[node]
            path_balance_metadata["name_to_id"][node_full_name] = node
            path_balance_metadata["pe_to_pond"][node] = True # assume true for now
            num_ponds_added += 1

            print(f"    Added pond {node} with balance length {chosen_balance_lengths[num_ponds_added-1]}")

        if num_ponds_added == len(chosen_balance_lengths):
            break

    # Update path metadata with info about added ponds
    path.update_pond_behavioral_fifo_count(path_balance_metadata)


def balance_io_reconvergence_groups(reconvergence_groups, parent_child_node_info, id_to_name, total_stream_length=1568, effort_level=1, mu_source_only=False):
    path_balance_metadata = {
        "balance_lengths": {},
        "total_stream_lengths": {},
        "name_to_id": {},
        "pe_to_pond": {},
    }

    # Find max fifo count across all paths in all reconvergence groups
    for group in reconvergence_groups:
        max_fifo_count = 0
        all_paths = group.get_paths()
        for path in all_paths:
            reg_count = path.get_total_fifo_count()
            if reg_count > max_fifo_count:
                max_fifo_count = reg_count
        group.set_max_fifo_count(max_fifo_count)
        print(f"Reconvergence Group from {group.get_source()} to {group.get_destination()} has max fifo count: {max_fifo_count}")


    # Balance each reconvergence group
    for group in reconvergence_groups:
        if mu_source_only:
            # Only balance MU to GLB groups
            if not (group.get_source() == "MU"):
                print(f"  Skipping reconvergence group from {group.get_source()} to {group.get_destination()} (not MU source).")
                continue
        print(f"Balancing Reconvergence Group from {group.get_source()} to {group.get_destination()}:")
        all_paths = group.get_paths()
        for path in all_paths:
            # Update path metadata with info about added ponds
            path.update_pond_behavioral_fifo_count(path_balance_metadata)
            path_fifo_count = path.get_total_fifo_count()
            fifo_deficit = group.max_fifo_count - path_fifo_count
            if fifo_deficit > 0:
                print(f"  Path from {path.get_source()} to {path.get_destination()} has fifo count {path_fifo_count}, needs {fifo_deficit} more FIFOs.")
                # Here, implement logic to insert FIFOs into the path as needed
                # This is a placeholder for actual balancing logic
                actual_effort_level = min(effort_level, path.get_pe_count())
                update_path_balance_metadata(path_balance_metadata, path, parent_child_node_info, fifo_deficit, total_stream_length, id_to_name, actual_effort_level)

                # TODO: Need to update fifo count here based on ponds that were just added
                # Need some way of conveying info about ponds that have already been added
            else:
                print(f"  Path from {path.get_source()} to {path.get_destination()} is already balanced with fifo count {path_fifo_count}.")


    return path_balance_metadata


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress design packed file by collapsing register chains.")
    parser.add_argument(
        "-i", "--input_design_packed",
        type=str,
        default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/zircon_deq_ResReLU_quant_fp/bin_saved/design.packed",
        help="Input design packed file"
    )
    parser.add_argument(
        "-p", "--input_design_post_pipe_packed",
        type=str,
        default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/zircon_deq_ResReLU_quant_fp/bin_saved/design_post_pipe.packed",
        help="Input design packed file"
    )
    parser.add_argument(
        "-d", "--id_to_name",
        type=str,
        default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/zircon_deq_ResReLU_quant_fp/bin_saved/design.id_to_name",
        help="Input id_to_name mapping file"
    )
    parser.add_argument(
        "-b", "--pe_bypass_config",
        type=str,
        default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/zircon_deq_ResReLU_quant_fp/bin_saved/pe_id_to_fifo_bypass_config.json",
        help="Input PE bypass configuration file"
    )
    parser.add_argument(
        "-o", "--output_design_packed",
        type=str,
        default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/zircon_deq_ResReLU_quant_fp/bin_saved/design_post_pipe_compressed.packed",
        help="Output compressed design packed file"
    )
    parser.add_argument("-e", "--intra_graph_effort", type=int, default=1, help="Effort level for intra-graph path balancing")
    args = parser.parse_args()

    adjacency = build_adjacency(args.input_design_post_pipe_packed)
    edge_dict = build_edge_dict(args.input_design_packed)
    parent_child_node_info = build_parent_child_node_info(adjacency)

    paths = find_all_paths(adjacency, args.intra_graph_effort)
    for p in paths:
        p.print_path()

    print(f"Total paths found: {len(paths)}")
    print("-----")


    id_to_name = extract_id_to_name(args.id_to_name)
    # Read from json
    pe_bypass_config = json.load(open(args.pe_bypass_config, 'r'))

    # Get I/O reconvergence groups
    io_reconvergence_groups = get_io_reconvergence_groups(paths, E64_mode=True, Multi_bank_mode=True, id_to_name=id_to_name, pe_bypass_config=pe_bypass_config, edge_dict=edge_dict)

    for n, group in enumerate(io_reconvergence_groups, start=1):
        print(f"I/O Reconvergence Group {n}:")
        print(f"Source: {group.get_source()}, Destination: {group.get_destination()}, Number of paths: {len(group.get_paths())}")
        print("-----")
        # Print paths in the group
        for path in group.get_paths():
            path.print_path()
        print("=====")


    # TODO: Need to reorder the reconverge groups so we balance the innermost ones first
    # io_reconvergence_groups = merge_sort_reconvergence_groups(io_reconvergence_groups)
    insertion_sort_reconvergence_groups(io_reconvergence_groups)

    print("-----")
    print("After sorting reconvergence groups:")
    print("-----")

    for n, group in enumerate(io_reconvergence_groups, start=1):
        print(f"I/O Reconvergence Group {n}:")
        print(f"Source: {group.get_source()}, Destination: {group.get_destination()}, Number of paths: {len(group.get_paths())}")
        print("-----")
        # Print paths in the group
        for path in group.get_paths():
            path.print_path()
        print("=====")


    # Balance the reconvergence groups (returns path balancing metadata). Also prints out the balancing info.
    balancing_metadata = balance_io_reconvergence_groups(io_reconvergence_groups, parent_child_node_info, id_to_name, total_stream_length=1568, effort_level=2, mu_source_only=True)

    with open(f"path_balancing.json", "w") as f:
        import json

        json.dump(balancing_metadata, f, indent=4)
