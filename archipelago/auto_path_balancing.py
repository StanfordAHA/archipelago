import re
import pydot
import argparse
from collections import defaultdict, deque
import itertools
import json
import os



def find_pe_input_num_from_driving_edge(pe_node, edge_dict, driving_edge):
    for node in edge_dict[driving_edge]:
        if node[0] == pe_node:
            pe_input_num = int(node[1].split("PE_input_width_17_num_")[1])
            return pe_input_num

    raise ValueError(f"Could not find PE input num for PE node {pe_node} from driving edge {driving_edge}.")


class Path:
    def __init__(self, nodes_edges=[]):
        self.nodes_edges = nodes_edges  # list of (node, edge_name) tuples
        self.nodes = [node for node, edge in nodes_edges]
        self.interconnect_fifo_count = 0
        self.pond_behavioral_fifo_count = 0
        self.pe_fifo_count = 0
        self.total_fifo_count = 0
        self.open_pe_count = 0
        self.initial_deficit = 0
        self.final_deficit = 0

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

    def update_pond_behavioral_fifo_count(self, path_balance_metadata, edge_dict):
        total_pond_fifos = 0
        for i in range(len(self.nodes_edges)):
            node, edge = self.nodes_edges[i]
            if node.startswith("p"):
                if node in path_balance_metadata["balance_lengths"]:

                    # PE-to-pond: simply add the balance length
                    if path_balance_metadata["pe_to_pond"][node][0] == True:
                        total_pond_fifos += path_balance_metadata["balance_lengths"][node]
                    # Pond-to-PE: only add the balance length is this path is using the corresponding PE input port
                    else:
                        if i > 0:
                            driving_edge = self.nodes_edges[i-1][1]
                        else:
                            raise ValueError("PE node is at the start of the path. Pond-to-PE is not possible. Please fix.")

                        pe_input_num = find_pe_input_num_from_driving_edge(node, edge_dict, driving_edge)
                        path_balance_metadata_pe_input_num = int(path_balance_metadata["pe_to_pond"][node][1].split("data")[1])
                        if pe_input_num == path_balance_metadata_pe_input_num:
                            total_pond_fifos += path_balance_metadata["balance_lengths"][node]

        self.pond_behavioral_fifo_count = total_pond_fifos

    def update_interconnect_fifo_count(self):
        interconnect_fifo_count = sum(1 for node, edge in self.nodes_edges if node.startswith("r"))
        self.interconnect_fifo_count = interconnect_fifo_count


    # Do not count PE output FIFOs if PE is destination
    # Do not count PE input FIFOs if PE is source
    def update_pe_fifo_count(self, pe_bypass_config, edge_dict):
        pe_fifo_count = 0
        for i in range(len(self.nodes_edges)):
            node, edge = self.nodes_edges[i]
            if node.startswith("p"):
                pe_num_active_fifos = 3
                # Handle this based on which specific input FIFOs are bypassed
                if node in pe_bypass_config["input_fifo_bypass"] or node == self.get_source():
                    if node == self.get_source():
                        pe_num_active_fifos -= 1
                    else:
                        driving_edge = self.nodes_edges[i-1][1]
                        pe_input_num = find_pe_input_num_from_driving_edge(node, edge_dict, driving_edge)
                        if pe_bypass_config["input_fifo_bypass"][node][pe_input_num] == 1:
                            pe_num_active_fifos -= 1
                if node in pe_bypass_config["output_fifo_bypass"] or node == self.get_destination():
                    pe_num_active_fifos -= 1
                if node in pe_bypass_config["prim_outfifo_bypass"] or node == self.get_destination():
                    pe_num_active_fifos -= 1
                pe_fifo_count += pe_num_active_fifos
        self.pe_fifo_count = pe_fifo_count

    def get_open_pe_count(self):
        return self.open_pe_count

    def update_open_pe_count(self, path_balance_metadata=None, seen_pes=None, disallow_use_of_seen_pes=False, legal_pond_pes=None):
        open_pe_count = 0
        if path_balance_metadata is None:
            for node, edge in self.nodes_edges:
                if node.startswith("p"):
                    if disallow_use_of_seen_pes and seen_pes is not None and node in seen_pes:
                        continue
                    if legal_pond_pes is not None and node not in legal_pond_pes:
                        continue
                    open_pe_count += 1
        else:
            for node, edge in self.nodes_edges:
                if node.startswith("p") and node not in path_balance_metadata["balance_lengths"]:
                    if disallow_use_of_seen_pes and seen_pes is not None and node in seen_pes:
                        continue
                    if legal_pond_pes is not None and node not in legal_pond_pes:
                        continue
                    open_pe_count += 1
        self.open_pe_count = open_pe_count


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

    def update_max_fifo_count(self, path_balance_metadata, edge_dict):
        max_fifo_count = 0
        all_paths = self.get_paths()
        for path in all_paths:
            path.update_interconnect_fifo_count()
            path.update_pond_behavioral_fifo_count(path_balance_metadata, edge_dict)
            reg_count = path.get_total_fifo_count()
            if reg_count > max_fifo_count:
                max_fifo_count = reg_count
        self.set_max_fifo_count(max_fifo_count)

    def add_broadcast_edge(self, edge):
        self.broadcast_edges.add(edge)

    def add_join_edge(self, edge):
        self.join_edges.add(edge)

    def get_broadcast_edges(self):
        return self.broadcast_edges

    def get_join_edges(self):
        return self.join_edges

    def get_total_available_pes(self):
        total_available_pes = 0
        for path in self.get_paths():
            total_available_pes += path.get_open_pe_count()
        return total_available_pes

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


def closest_sum(target_sum, choices, balance_length_effort_level):
    best_combo = None
    best_diff = float('inf')
    best_sum = None

    # Explore using 1 to balance_length_effort_level operands (allowing reuse)
    for k in range(1, balance_length_effort_level + 1):
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

def filter_reconvergence_groups_saved(reconvergence_groups):
    def any_lists_share_element(*lists):
        seen = set()
        for lst in lists:
            for item in lst:
                if item in seen:
                    return True
                seen.add(item)
        return False

    def find_overlapping_lists(lists):
        seen = {}               # maps element → index of list where it first appeared
        overlaps = set()        # stores list indices that overlap with another list

        for idx, lst in enumerate(lists):
            for item in lst:
                if item in seen:
                    # record both the previously seen index and current index
                    overlaps.add(seen[item])
                    overlaps.add(idx)
                else:
                    seen[item] = idx

        return overlaps


    filtered_reconvergence_groups = reconvergence_groups.copy()
    for rg in reconvergence_groups:
        # RGs that terminate at GLB are exceptions to this rule (due to E64/Multi-bank grouping)
        if rg.get_destination().startswith("GLB"):
            continue

        # Gather all paths in the reconvergence group and exclude source and destination nodes
        path_nodes_lists = [path.get_nodes()[1:-1] for path in rg.get_paths()]  # exclude source and destination nodes
        if any_lists_share_element(*path_nodes_lists):
            print(f"  Filtering out reconvergence group from {rg.get_source()} to {rg.get_destination()} because its paths directly pass through the same node(s).")
            filtered_reconvergence_groups.remove(rg)

    return filtered_reconvergence_groups


def filter_reconvergence_groups(reconvergence_groups, filter_overlapping_GLB_output_paths=False):
    def build_conflict_graph(lists):
        N = len(lists)
        conflicts = {i: set() for i in range(N)}
        for i in range(N):
            for j in range(i+1, N):
                if set(lists[i]) & set(lists[j]):
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        return conflicts

    def find_maximal_collision_free_groups(lists):
        N = len(lists)
        conflicts = build_conflict_graph(lists)
        maximal_groups = []

        def backtrack(start, current_group):
            extended = False
            for next_idx in range(start, N):
                if all(next_idx not in conflicts[i] for i in current_group):
                    backtrack(next_idx + 1, current_group + [next_idx])
                    extended = True
            if not extended and len(current_group) >= 2:
                maximal_groups.append(set(current_group))

        backtrack(0, [])
        return maximal_groups

    filtered_reconvergence_groups = reconvergence_groups.copy()

    for rg in reconvergence_groups:
        # RGs that terminate at GLB are exceptions to this rule (due to E64/Multi-bank grouping)
        if not(filter_overlapping_GLB_output_paths) and rg.get_destination().startswith("GLB"):
            continue

        orig_paths = rg.get_paths()
        filtered_paths = []

        orig_paths_without_src_dest = []
        for i in range(len(orig_paths)):
            path = orig_paths[i]
            nodes_without_src_dest = path.get_nodes()[1:-1]  # exclude source and destination nodes
            orig_paths_without_src_dest.append(nodes_without_src_dest)

        # overlapping_path_indices = find_overlapping_lists(orig_paths_without_src_dest)
        non_overlapping_path_groups = find_maximal_collision_free_groups(orig_paths_without_src_dest)


        if len(non_overlapping_path_groups) == 1:
            # No overlaps; keep all paths
            if len(non_overlapping_path_groups[0]) == len(orig_paths):
                continue
            # Keep only paths without overlaps
            else:
                for i in range(len(orig_paths)):
                    path = orig_paths[i]
                    if i in non_overlapping_path_groups[0]:
                        filtered_paths.append(path)
                    else:
                        print(f"  Filtering out path {path.print_path()} in reconvergence group from {rg.get_source()} to {rg.get_destination()}.")
                rg.paths = filtered_paths

        elif (len(non_overlapping_path_groups) == 0):
            print(f"  Filtering out entire reconvergence group from {rg.get_source()} to {rg.get_destination()} because all its paths directly pass through the same node(s).")
            filtered_reconvergence_groups.remove(rg)


        else:
            # Multiple groups of non-overlapping paths found; need to create new RGs
            filtered_reconvergence_groups.remove(rg)
            print(f"  Splitting reconvergence group from {rg.get_source()} to {rg.get_destination()} into {len(non_overlapping_path_groups)} groups due to path overlaps.")
            for new_rg in non_overlapping_path_groups:
                new_paths = []
                new_broadcast_edges = set()
                new_join_edges = set()
                for i in new_rg:
                    path = orig_paths[i]
                    new_paths.append(path)
                    new_broadcast_edges.add(path.nodes_edges[0][1])
                    new_join_edges.add(path.nodes_edges[-2][1])

                new_rg_instance = ReconvergenceGroup(rg.get_source(), rg.get_destination(), new_paths)
                new_rg_instance.broadcast_edges = new_broadcast_edges
                new_rg_instance.join_edges = new_join_edges
                filtered_reconvergence_groups.append(new_rg_instance)

    return filtered_reconvergence_groups

def get_io_reconvergence_groups(all_paths, E64_multi_bank_config, id_to_name=None, pe_bypass_config=None, edge_dict=None, filter_overlapping_GLB_output_paths=False):
    """
    Finds io reconvergence groups from the provided paths.
    """
    input_E64_mode = E64_multi_bank_config.get("input_E64", False)
    input_Multi_bank_mode = E64_multi_bank_config.get("input_Multi_bank", False)
    output_E64_mode = E64_multi_bank_config.get("output_E64", False)
    output_Multi_bank_mode = E64_multi_bank_config.get("output_Multi_bank", False)

    if input_Multi_bank_mode:
        assert input_E64_mode, "input_E64_mode must be True if input_Multi_bank_mode is True."

    if output_Multi_bank_mode:
        assert output_E64_mode, "output_E64_mode must be True if output_Multi_bank_mode is True."

    if input_E64_mode or input_Multi_bank_mode or output_E64_mode or output_Multi_bank_mode:
        assert id_to_name is not None, "id_to_name mapping must be provided in E64 or multi_bank_mode."

    # Create empty list of ReconvergenceGroups
    reconvergence_groups = []

    for path in all_paths:
        path.update_interconnect_fifo_count()
        path.update_open_pe_count()
        path.update_pe_fifo_count(pe_bypass_config, edge_dict)
        path_source = path.get_source()
        path_destination = path.get_destination()

        # Ignore paths with MEM tiles in the middle
        if any(node.startswith("M") or node.startswith("m") for node in path.get_nodes()[1:-1]):
            continue

        # Treat all MU I/Os as the same source
        if path_source.startswith("U") or path_source.startswith("V"):
            path_source = "MU"

        # Group I/Os using same GLB tile in E64 or Multi-bank mode
        # Handle input I/Os
        if input_E64_mode and path_source.startswith("I"):
            source_node_name = id_to_name.get(path_source)
            source_node_name_parse_list = source_node_name.split("stencil_")[2].split("_read")
            source_lane_idx = int(source_node_name_parse_list[0]) if len(source_node_name_parse_list) > 1 else 0

            if input_Multi_bank_mode:
                path_source = f"GLB_input_group_{source_lane_idx // 8}"
            elif input_E64_mode:
                path_source = f"GLB_input_group_{source_lane_idx // 4}"

        # Handle output I/Os
        if output_E64_mode and path_destination.startswith("I"):
            destination_node_name = id_to_name.get(path_destination)
            destination_node_name_parse_list = destination_node_name.split("stencil_")[2].split("_write")
            destination_lane_idx = int(destination_node_name_parse_list[0]) if len(destination_node_name_parse_list) > 1 else 0

            if output_Multi_bank_mode:
                path_destination = f"GLB_output_group_{destination_lane_idx // 8}"
            elif output_E64_mode:
                path_destination = f"GLB_output_group_{destination_lane_idx // 4}"

        # Check if there is an existing reconvergence group with this source, destination pair
        # TODO: Also check that there is no edge overlap for outputs (i.e. going into destinatiion)
        # TODO: And there MUST be edge overlap for inputs (if it's not an I/O tile or MU I/O) (i.e. coming from source)
        source_is_IO_tile = path_source.startswith("GLB_input_group_") or path_source == "MU" or path_source.startswith("I")
        for group in reconvergence_groups:
            enforced_input_edge_overlap = path.nodes_edges[0][1] in group.get_broadcast_edges()
            if group.get_source() == path_source and group.get_destination() == path_destination and (source_is_IO_tile or enforced_input_edge_overlap):
                if not(source_is_IO_tile):
                    assert len(group.get_broadcast_edges()) == 1, "There should only be one broadcast edge in the group."

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

    # Filter paths reconvergence groups that aren't real (directly pass through another reconvergence group)
    reconvergence_groups = filter_reconvergence_groups(reconvergence_groups, filter_overlapping_GLB_output_paths=filter_overlapping_GLB_output_paths)

    # Trim reconvergence groups with only one path
    reconvergence_groups = [group for group in reconvergence_groups if len(group.get_paths()) > 1]

    return reconvergence_groups

def is_ordered_subsequence(sub, full):
    it = iter(full)
    return all(x in it for x in sub)


def reconvergence_group_greater_than_search_paths(rg1: ReconvergenceGroup, rg2: ReconvergenceGroup):
    """
    Returns true if rg1 is "greater than" rg2, meaning that rg2 is contained within rg1.
    Containement is defined as all paths in rg2 being fully contained within at least one path in rg1.
    """
    rg2_paths = rg2.get_paths()
    rg1_paths = rg1.get_paths()


    for rg2_path in rg2_paths:
        # Check if rg2_path is contained within any of the rg1 paths. If there is any rg2path that is not fully contained within rg1's paths, just return false
        this_rg2_path_contained_in_rg1 = False
        for rg1_path in rg1_paths:
            rg2_without_dest_nodes_edges = rg2_path.get_nodes_edges()[:-1]  # exclude destination node because the outgoing edge is None for rg2_path
            if rg2_path.get_destination() in rg1_path.get_nodes() and is_ordered_subsequence(rg2_without_dest_nodes_edges, rg1_path.get_nodes_edges()):
                this_rg2_path_contained_in_rg1 = True
                break
        if not this_rg2_path_contained_in_rg1:
            return False

    print(f"  Reconvergence Group from {rg2.get_source()} to {rg2.get_destination()} is inside Reconvergence Group from {rg1.get_source()} to {rg1.get_destination()}.")
    return True

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
        if reconvergence_group_greater_than_search_paths(arr[i], key):
            return True
    return False

def rg_to_left_has_more_available_pes(key, arr, j):
    for i in range(j, -1, -1):
        if arr[i].get_total_available_pes() > key.get_total_available_pes():
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

        # Containment met. Now keep moving so groups with fewer total available PEs are to the left of groups with more available PEs
        while j >= 0 and rg_to_left_has_more_available_pes(key, arr, j):
            arr[j + 1] = arr[j]
            j -= 1

        # Check if key destination is p5
        if key.get_destination() == "p5":
            print(f"P5 initially placed at position {j+1} during insertion sort. Current left RGs:")
            for z in range(j+1):
                print(f"  Left RG {arr[z].get_source()} -> {arr[z].get_destination()}")

        # Place the key in its correct position
        arr[j + 1] = key


def update_path_balance_metadata(path_balance_metadata, path, parent_child_node_info, edge_dict, fifo_deficit, total_stream_length, id_to_name, balance_length_effort_level, seen_pes, disallow_use_of_seen_pes, legal_pond_pes=None, pe_to_reconvergence_group_count=None):
    balance_length_choices = []
    MAX_BALANCE_LENGTH = 32
    POND_EXTENT_COUNTER_WIDTH = 11
    MAX_EXTENT = 2**(POND_EXTENT_COUNTER_WIDTH - 1)  # Counter is signed so max extent is 2^10

    min_balance_length = 1 if total_stream_length <= MAX_EXTENT else 2
    for i in range(min_balance_length, MAX_BALANCE_LENGTH + 1):  # check only factors between (1 or 2) and 32
        if total_stream_length % i == 0:
            balance_length_choices.append(i)

    chosen_balance_lengths = closest_sum(fifo_deficit, balance_length_choices, balance_length_effort_level)

    if fifo_deficit < min_balance_length:
        print(f"\033[94mINFO: Skipping balancing from {path.get_source()} to {path.get_destination()}. FIFO deficit {fifo_deficit} is less than minimum balance length {min_balance_length}.\033[0m")
        return

    # add the ponds with chosen balance lengths to the path, starting from the destination
    num_ponds_added = 0
    reversed_path = list(reversed(path.get_nodes_edges()))

    pes_to_consider = []
    for i in range(len(reversed_path)):
        node, edge = reversed_path[i]
        if node.startswith("p"):
            if node in path_balance_metadata["balance_lengths"]:
                continue  # Already added pond here. No revisiting of PEs that have already been assigned ponds.

            if disallow_use_of_seen_pes and node in seen_pes:
                print(f"    Skipping pond {node} because it is in the seen_pes set.")
                continue

            if legal_pond_pes is not None and node not in legal_pond_pes:
                print(f"    Skipping pond {node} because it is not in the legal_pond_pes set.")
                continue

            pes_to_consider.append(node)


    # Print the PEs being considered before sorting
    print(f"    Considering PEs for pond placement (before sorting): {pes_to_consider}")

    # Now sort the pes_to_consider based on their reconvergence group participation (less participation is better)
    if pe_to_reconvergence_group_count is not None:
        pes_to_consider.sort(key=lambda pe: pe_to_reconvergence_group_count.get(pe, 0))

    print(f"    Considering PEs for pond placement (after sorting): {pes_to_consider}")

    # Building a smarter list of PEs to traverse
    for node in pes_to_consider:
        if path.get_destination() == node:
            driving_edge = reversed_path[i+1][1]
            pe_input_num = find_pe_input_num_from_driving_edge(node, edge_dict, driving_edge)
            pe_data_port_name = f"data{pe_input_num}"
            path_balance_metadata["pe_to_pond"][node] = (False, pe_data_port_name)
        else:
            path_balance_metadata["pe_to_pond"][node] = (True, "")

        path_balance_metadata["balance_lengths"][node] = chosen_balance_lengths[num_ponds_added]
        path_balance_metadata["total_stream_lengths"][node] = total_stream_length
        node_full_name = id_to_name[node]
        path_balance_metadata["name_to_id"][node_full_name] = node
        num_ponds_added += 1

        print(f"    Added pond {node} with balance length {chosen_balance_lengths[num_ponds_added-1]}")

        if num_ponds_added == len(chosen_balance_lengths):
            break

    # Update path metadata with info about added ponds
    path.update_pond_behavioral_fifo_count(path_balance_metadata, edge_dict)
    path.update_open_pe_count(path_balance_metadata, seen_pes, disallow_use_of_seen_pes, legal_pond_pes)


def print_imbalances(reconvergence_groups, path_balance_metadata, edge_dict, initial=False):
    print("--------------------------------")
    if initial:
        print("Initial Reconvergence Group Imbalances:")
    else:
        print("Final Reconvergence Group Imbalances:")
    print("--------------------------------")

    # Update FIFO counts
    for group in reconvergence_groups:
        all_paths = group.get_paths()
        for path in all_paths:
            path.update_interconnect_fifo_count()
            path.update_pond_behavioral_fifo_count(path_balance_metadata, edge_dict)
        group.update_max_fifo_count(path_balance_metadata, edge_dict)


    # Print output
    for group in reconvergence_groups:
        all_paths = group.get_paths()
        for path in all_paths:
            path_fifo_count = path.get_total_fifo_count()
            fifo_deficit = group.max_fifo_count - path_fifo_count
            if initial:
                path.initial_deficit = fifo_deficit
            else:
                path.final_deficit = fifo_deficit

            if initial:
                print(f"  RG from {group.get_source()} to {group.get_destination()}: Path from {path.get_source()} to {path.get_destination()} has fifo count {path_fifo_count}, deficit {fifo_deficit}.")
            else:
                print(f"  RG from {group.get_source()} to {group.get_destination()}: Path from {path.get_source()} to {path.get_destination()} has fifo count {path_fifo_count}, final deficit {fifo_deficit} (initial deficit was {path.initial_deficit}).")


def analyze_pe_reconvergence_group_participation(reconvergence_groups, skip_glb_output_reconvergence_groups=False):
    # For every PE, keep track of number of reconvergence groups it appears in. Only count each PE once per reconvergence group.
    pe_to_reconvergence_group_count = defaultdict(int)
    for rg in reconvergence_groups:
        if skip_glb_output_reconvergence_groups and rg.get_destination().startswith("GLB_output_group_"):
            continue
        pes_in_this_rg = set()
        for path in rg.get_paths():
            for node, edge in path.get_nodes_edges():
                if node.startswith("p"):
                    pes_in_this_rg.add(node)
        for pe in pes_in_this_rg:
            pe_to_reconvergence_group_count[pe] += 1
    return pe_to_reconvergence_group_count


def balance_io_reconvergence_groups(path_balance_metadata, reconvergence_groups, parent_child_node_info, edge_dict, id_to_name, seen_pes, total_stream_length=1568, balance_length_effort_level=1,
                                    mu_source_only=False, disallow_use_of_seen_pes=False, legal_pond_pes=None, pe_to_reconvergence_group_count=None, skip_glb_output_reconvergence_groups=False):
    # Find max fifo count across all paths in all reconvergence groups
    for group in reconvergence_groups:
        group.update_max_fifo_count(path_balance_metadata, edge_dict)

    # Balance each reconvergence group
    for group in reconvergence_groups:
        if skip_glb_output_reconvergence_groups and group.get_destination().startswith("GLB_output_group_"):
            print("  Skipping GLB output group.")
            continue
        if mu_source_only:
            # Only balance MU to GLB groups
            if not (group.get_source() == "MU"):
                print(f"  Skipping reconvergence group from {group.get_source()} to {group.get_destination()} (not MU source).")
                continue
        print(f"Balancing Reconvergence Group from {group.get_source()} to {group.get_destination()}:")
        group.update_max_fifo_count(path_balance_metadata, edge_dict)
        all_paths = group.get_paths()
        for path in all_paths:
            # Update path metadata with info about added ponds
            path.update_pond_behavioral_fifo_count(path_balance_metadata, edge_dict)
            path.update_open_pe_count(path_balance_metadata, seen_pes, disallow_use_of_seen_pes, legal_pond_pes)
            path_fifo_count = path.get_total_fifo_count()

            fifo_deficit = group.max_fifo_count - path_fifo_count
            if fifo_deficit > 0:
                print(f"  Path from {path.get_source()} to {path.get_destination()} has fifo count {path_fifo_count}, needs {fifo_deficit} more FIFOs.")
                if path.get_open_pe_count() == 0:
                    print(f"\033[93mWARNING: Path from {path.get_source()} to {path.get_destination()} in RG from {group.get_source()} to {group.get_destination()} has no open PEs to insert ponds into. Skipping balancing for this path, though it needs {fifo_deficit} more FIFOs.\033[0m")
                    continue
                actual_balance_length_effort_level = min(balance_length_effort_level, path.get_open_pe_count())
                update_path_balance_metadata(path_balance_metadata, path, parent_child_node_info, edge_dict, fifo_deficit, total_stream_length, id_to_name, actual_balance_length_effort_level, seen_pes, disallow_use_of_seen_pes, legal_pond_pes, pe_to_reconvergence_group_count)
            else:
                print(f"  Path from {path.get_source()} to {path.get_destination()} is already balanced with fifo count {path_fifo_count}.")

            # Add all PEs used in this path to seen_pes
            for node, edge in path.get_nodes_edges():
                if node.startswith("p"):
                    seen_pes.add(node)

            group.update_max_fifo_count(path_balance_metadata, edge_dict)
            print(f"  RG from {group.get_source()} to {group.get_destination()} has new max fifo count {group.max_fifo_count}.")


    return path_balance_metadata


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress design packed file by collapsing register chains.")
    parser.add_argument(
        "-i", "--input_design_packed",
        type=str,
        default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/maxpooling_dense_rv_fp/bin_e64/design.packed",
        help="Input design packed file"
    )
    parser.add_argument(
        "-p", "--input_design_post_pipe_packed",
        type=str,
        default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/maxpooling_dense_rv_fp/bin_e64/design_post_pipe.packed",
        help="Input design packed file"
    )
    parser.add_argument(
        "-d", "--id_to_name",
        type=str,
        default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/maxpooling_dense_rv_fp/bin_e64/design.id_to_name",
        help="Input id_to_name mapping file"
    )
    parser.add_argument(
        "-b", "--pe_bypass_config",
        type=str,
        default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/maxpooling_dense_rv_fp/bin_e64/pe_id_to_fifo_bypass_config.json",
        help="Input PE bypass configuration file"
    )
    parser.add_argument("-e", "--intra_graph_effort", type=int, default=1, help="Effort level for intra-graph path balancing")
    parser.add_argument("-l", "--balance_length_effort_level", type=int, default=2, help="Effort level for balance length selection")
    parser.add_argument(
        "-s", "--pe_bogus_stream_length",
        type=str,
        default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/maxpooling_dense_rv_fp/pe_bogus_stream_length.json",
        help="Input PE bogus stream length file"
    )
    parser.add_argument(
        "-o", "--output_path_balancing_config",
        type=str,
        default="/aha/Halide-to-Hardware/apps/hardware_benchmarks/apps/maxpooling_dense_rv_fp/path_balancing_configs/new_path_balancing.json",
        help="Output path balancing configuration file"
    )
    parser.add_argument("-g", "--skip_glb_output_reconvergence_groups", action="store_true", help="Skip GLB output reconvergence groups during balancing")

    args = parser.parse_args()

    adjacency = build_adjacency(args.input_design_post_pipe_packed)
    edge_dict = build_edge_dict(args.input_design_packed)
    parent_child_node_info = build_parent_child_node_info(adjacency)
    skip_glb_output_reconvergence_groups = args.skip_glb_output_reconvergence_groups

    seen_pes = set()

    paths = find_all_paths(adjacency, args.intra_graph_effort)
    for p in paths:
        p.print_path()

    print(f"Total paths found: {len(paths)}")
    print("-----")


    id_to_name = extract_id_to_name(args.id_to_name)
    # Read from json
    pe_bypass_config = json.load(open(args.pe_bypass_config, 'r'))

    E64_Multi_bank_config = {
        "input_E64": True,
        "input_Multi_bank": False,
        "output_E64": True,
        "output_Multi_bank": True,
    }
    io_reconvergence_groups = get_io_reconvergence_groups(paths, E64_Multi_bank_config, id_to_name=id_to_name, pe_bypass_config=pe_bypass_config, edge_dict=edge_dict, filter_overlapping_GLB_output_paths=False)

    # Build pe_to_recongenvergence_group count dict
    pe_to_reconvergence_group_count = analyze_pe_reconvergence_group_participation(io_reconvergence_groups, skip_glb_output_reconvergence_groups=skip_glb_output_reconvergence_groups)
    print("PE to Reconvergence Group Participation Counts:")
    for pe, count in pe_to_reconvergence_group_count.items():
        print(f"  PE {pe} appears in {count} reconvergence groups")


    for n, group in enumerate(io_reconvergence_groups, start=1):
        print(f"I/O Reconvergence Group {n}:")
        print(f"Source: {group.get_source()}, Destination: {group.get_destination()}, Number of paths: {len(group.get_paths())}")
        print("-----")
        # Print paths in the group
        for path in group.get_paths():
            path.print_path()
        print("=====")


    insertion_sort_reconvergence_groups(io_reconvergence_groups)

    print("-----")
    print("After sorting reconvergence groups:")
    print("-----")

    i = 1
    for n, group in enumerate(io_reconvergence_groups, start=1):
        if skip_glb_output_reconvergence_groups and group.get_destination().startswith("GLB_output_group_"):
            print("  Skipping GLB output group.")
            continue
        print(f"I/O Reconvergence Group {i}:")
        i += 1
        print(f"Source: {group.get_source()}, Destination: {group.get_destination()}, Number of paths: {len(group.get_paths())}")
        print("-----")
        # Print paths in the group
        for path in group.get_paths():
            path.print_path()
        print("=====")


    balancing_metadata = {
        "balance_lengths": {},
        "total_stream_lengths": {},
        "name_to_id": {},
        "pe_to_pond": {},
    }

    print_imbalances(io_reconvergence_groups, balancing_metadata, edge_dict, initial=True)

    # For maxpooling, output stream length = 58x57x32. For unroll by 16, stream length per lane = 6612
    print("----- 1st Pass Balancing -----")
    balancing_metadata = balance_io_reconvergence_groups(balancing_metadata, io_reconvergence_groups, parent_child_node_info, edge_dict, id_to_name, seen_pes, total_stream_length=6612,
                                                         balance_length_effort_level=args.balance_length_effort_level, mu_source_only=False,
                                                         disallow_use_of_seen_pes=False, legal_pond_pes=None, pe_to_reconvergence_group_count=pe_to_reconvergence_group_count, skip_glb_output_reconvergence_groups=skip_glb_output_reconvergence_groups)

    # print("----- 2nd Pass Balancing -----")
    # seen_pes = set()
    # # 2nd pass to catch any remaining imbalances
    # balancing_metadata = balance_io_reconvergence_groups(balancing_metadata, io_reconvergence_groups, parent_child_node_info, edge_dict, id_to_name, seen_pes, total_stream_length=6612,
    #                                                      balance_length_effort_level=args.balance_length_effort_level, mu_source_only=False,
    #                                                      disallow_use_of_seen_pes=False, legal_pond_pes=None, pe_to_reconvergence_group_count=pe_to_reconvergence_group_count, skip_glb_output_reconvergence_groups=skip_glb_output_reconvergence_groups)
    print_imbalances(io_reconvergence_groups, balancing_metadata, edge_dict)

    output_path = args.output_path_balancing_config
    print(f"Writing path balancing metadata to {output_path}...")
    with open(output_path, "w") as f:
        import json

        json.dump(balancing_metadata, f, indent=4)
