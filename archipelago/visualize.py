import sys
import os
import random
import math
import argparse
import glob
from typing import Dict, List, Tuple, Set
import pythunder
import pycyclone
from pycyclone.io import load_placement
from archipelago.io import load_routing_result
from archipelago.pnr_graph import (
    RoutingResultGraph,
    construct_graph,
    TileType,
    RouteType,
    TileNode,
    RouteNode,
)

try:
    from PIL import Image, ImageDraw
except:
    Image = None
    ImageDraw = None

import time

# Draw parameters
GLOBAL_TILE_WIDTH = 200
GLOBAL_TILE_MARGIN = 40  # each side is 40 pixs
GLOBAL_TILE_WIDTH_INNER = GLOBAL_TILE_WIDTH - 2 * GLOBAL_TILE_MARGIN
GLOBAL_OFFSET_X = 20  # outer margin
GLOBAL_OFFSET_Y = 20
GLOBAL_NUM_TRACK = 5
GLOBAL_ARROW_DISTANCE = GLOBAL_TILE_WIDTH_INNER // (GLOBAL_NUM_TRACK * 2 + 1)
ARROW_WIDTH = 10
# TODO: hardcode MU16 y coord
MU16_Y_COORD = 17
MAX_PACKED_GLB_IO_NUM = 4
MAX_PACKED_MU_IO_NUM = 2

side_map = ["Right", "Bottom", "Left", "Top"]
io_map = ["IN", "OUT"]


def draw_arrow(
    draw,
    x,
    y,
    dir="UP",
    len=GLOBAL_TILE_MARGIN,
    color="Black",
    width=1,
    source_port=False,
    sink_port=False,
):
    arr_w = max(min(width, 7), 3)
    if dir == "UP":
        dx = 0
        dy = -1
        rx = arr_w
        ry = 0.7 * len
    elif dir == "DOWN":
        dx = 0
        dy = 1
        rx = arr_w
        ry = 0.7 * len
    elif dir == "LEFT":
        dx = -1
        dy = 0
        rx = 0.7 * len
        ry = arr_w
    elif dir == "RIGHT":
        dx = 1
        dy = 0
        rx = 0.7 * len
        ry = arr_w
    else:
        print("[Error] unsupported arrow direction")
        exit()
    xy = [(x, y), (x + dx * len * 0.8, y + dy * len * 0.8)]
    if dir == "UP" or dir == "DOWN":
        lxy = (x + dy * rx, y + dy * ry)
        rxy = (x + -dy * rx, y + dy * ry)
    else:
        lxy = (x + dx * rx, y + -dx * ry)
        rxy = (x + dx * rx, y + dx * ry)
    draw.line(xy=xy, fill=color, width=width)
    draw.polygon([(x + dx * len, y + dy * len), lxy, rxy], fill=color)

    pw = (GLOBAL_TILE_WIDTH - 2 * GLOBAL_TILE_MARGIN) / 30
    if source_port:
        xy = [(x - pw, y - pw), (x + pw, y - pw), (x + pw, y + pw), (x - pw, y + pw)]
        draw.polygon(xy=xy, fill="Green", outline="Black", width=1)

    if sink_port:
        x += dx * len
        y += dy * len
        xy = [(x - pw, y - pw), (x + pw, y - pw), (x + pw, y + pw), (x - pw, y + pw)]
        draw.polygon(xy=xy, fill="Green", outline="Black", width=1)  # TODO same color


def draw_diagonal_arrow(
    draw, x, y, dir, x2, y2, dir2="UP", len=GLOBAL_TILE_MARGIN, color="Black", width=1
):
    # color = "Blue"
    if dir == "UP":
        dx = 0
        dy = -len
        rx = 0.09
        ry = 0.8
    elif dir == "DOWN":
        dx = 0
        dy = len
        rx = 0.09
        ry = 0.8
    elif dir == "LEFT":
        dx = -len
        dy = 0
        rx = 0.8
        ry = 0.09
    elif dir == "RIGHT":
        dx = len
        dy = 0
        rx = 0.8
        ry = 0.09
    else:
        print("[Error] unsupported arrow direction")
        exit()
    xy = [(x, y), (x + dx, y + dy)]

    if dir2 == "UP":
        dx = 0
        dy = -len
        rx = 0.09
        ry = 0.8
    elif dir2 == "DOWN":
        dx = 0
        dy = len
        rx = 0.09
        ry = 0.8
    elif dir2 == "LEFT":
        dx = -len
        dy = 0
        rx = 0.8
        ry = 0.09
    elif dir2 == "RIGHT":
        dx = len
        dy = 0
        rx = 0.8
        ry = 0.09
    else:
        print("[Error] unsupported arrow direction")
        exit()
    xy2 = [(x2, y2), (x2 + dx, y2 + dy)]
    new_xy = [xy[0], xy2[1]]
    draw.line(xy=new_xy, fill=color, width=width)


def draw_arrow_between_sb(draw, node, node2, graph, color="Black", width=1):
    tile_x = node.x
    tile_y = node.y
    side = side_map[node.side]
    io = io_map[node.io]
    track_id = node.track
    nlanes1 = lane_count_for_side(graph[node.x, node.y].switchbox, side)

    tile_x2 = node2.x
    tile_y2 = node2.y
    side2 = side_map[node2.side]
    io2 = io_map[node2.io]
    track_id2 = node2.track
    nlanes2 = lane_count_for_side(graph[node2.x, node2.y].switchbox, side2)

    if tile_x != tile_x2 or tile_y != tile_y2:
        return

    if side == "Top":
        if io == "IN":
            dir = "DOWN"
            x = (
                GLOBAL_OFFSET_X
                + GLOBAL_TILE_MARGIN
                + tile_x * GLOBAL_TILE_WIDTH
                + (track_id + 1) * arrow_distance_for_count(nlanes1)
            )
            y = GLOBAL_OFFSET_Y + tile_y * GLOBAL_TILE_WIDTH
        elif io == "OUT":
            dir = "UP"
            x = (
                GLOBAL_OFFSET_X
                + GLOBAL_TILE_MARGIN
                + tile_x * GLOBAL_TILE_WIDTH
                + (track_id + 1 + GLOBAL_NUM_TRACK) * arrow_distance_for_count(nlanes1)
            )
            y = GLOBAL_OFFSET_Y + GLOBAL_TILE_MARGIN + tile_y * GLOBAL_TILE_WIDTH
    elif side == "Right":
        if io == "IN":
            dir = "LEFT"
            x = GLOBAL_OFFSET_X + tile_x * GLOBAL_TILE_WIDTH + GLOBAL_TILE_WIDTH
            y = (
                GLOBAL_OFFSET_Y
                + GLOBAL_TILE_MARGIN
                + tile_y * GLOBAL_TILE_WIDTH
                + (track_id + 1) * arrow_distance_for_count(nlanes1)
            )
        elif io == "OUT":
            dir = "RIGHT"
            x = (
                GLOBAL_OFFSET_X
                + tile_x * GLOBAL_TILE_WIDTH
                + GLOBAL_TILE_WIDTH
                - GLOBAL_TILE_MARGIN
            )
            y = (
                GLOBAL_OFFSET_Y
                + GLOBAL_TILE_MARGIN
                + tile_y * GLOBAL_TILE_WIDTH
                + (track_id + 1 + GLOBAL_NUM_TRACK) * arrow_distance_for_count(nlanes1)
            )
    elif side == "Bottom":
        if io == "IN":
            dir = "UP"
            x = (
                GLOBAL_OFFSET_X
                + GLOBAL_TILE_MARGIN
                + tile_x * GLOBAL_TILE_WIDTH
                + (track_id + 1 + GLOBAL_NUM_TRACK) * arrow_distance_for_count(nlanes1)
            )
            y = GLOBAL_OFFSET_Y + tile_y * GLOBAL_TILE_WIDTH + GLOBAL_TILE_WIDTH
        elif io == "OUT":
            dir = "DOWN"
            x = (
                GLOBAL_OFFSET_X
                + GLOBAL_TILE_MARGIN
                + tile_x * GLOBAL_TILE_WIDTH
                + (track_id + 1) * arrow_distance_for_count(nlanes1)
            )
            y = (
                GLOBAL_OFFSET_Y
                + tile_y * GLOBAL_TILE_WIDTH
                + GLOBAL_TILE_WIDTH
                - GLOBAL_TILE_MARGIN
            )
    elif side == "Left":
        if io == "IN":
            dir = "RIGHT"
            x = GLOBAL_OFFSET_X + tile_x * GLOBAL_TILE_WIDTH
            y = (
                GLOBAL_OFFSET_Y
                + GLOBAL_TILE_MARGIN
                + tile_y * GLOBAL_TILE_WIDTH
                + (track_id + 1 + GLOBAL_NUM_TRACK) * arrow_distance_for_count(nlanes1)
            )
        elif io == "OUT":
            dir = "LEFT"
            x = GLOBAL_OFFSET_X + tile_x * GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
            y = (
                GLOBAL_OFFSET_Y
                + GLOBAL_TILE_MARGIN
                + tile_y * GLOBAL_TILE_WIDTH
                + (track_id + 1) * arrow_distance_for_count(nlanes1)
            )

    if side2 == "Top":
        if io2 == "IN":
            dir2 = "DOWN"
            x2 = (
                GLOBAL_OFFSET_X
                + GLOBAL_TILE_MARGIN
                + tile_x2 * GLOBAL_TILE_WIDTH
                + (track_id2 + 1) * arrow_distance_for_count(nlanes2)
            )
            y2 = GLOBAL_OFFSET_Y + tile_y2 * GLOBAL_TILE_WIDTH
        elif io2 == "OUT":
            dir2 = "UP"
            x2 = (
                GLOBAL_OFFSET_X
                + GLOBAL_TILE_MARGIN
                + tile_x2 * GLOBAL_TILE_WIDTH
                + (track_id2 + 1 + GLOBAL_NUM_TRACK) * arrow_distance_for_count(nlanes2)
            )
            y2 = GLOBAL_OFFSET_Y + GLOBAL_TILE_MARGIN + tile_y2 * GLOBAL_TILE_WIDTH
    elif side2 == "Right":
        if io2 == "IN":
            dir2 = "LEFT"
            x2 = GLOBAL_OFFSET_X + tile_x2 * GLOBAL_TILE_WIDTH + GLOBAL_TILE_WIDTH
            y2 = (
                GLOBAL_OFFSET_Y
                + GLOBAL_TILE_MARGIN
                + tile_y2 * GLOBAL_TILE_WIDTH
                + (track_id2 + 1) * arrow_distance_for_count(nlanes2)
            )
        elif io2 == "OUT":
            dir2 = "RIGHT"
            x2 = (
                GLOBAL_OFFSET_X
                + tile_x2 * GLOBAL_TILE_WIDTH
                + GLOBAL_TILE_WIDTH
                - GLOBAL_TILE_MARGIN
            )
            y2 = (
                GLOBAL_OFFSET_Y
                + GLOBAL_TILE_MARGIN
                + tile_y2 * GLOBAL_TILE_WIDTH
                + (track_id2 + 1 + GLOBAL_NUM_TRACK) * arrow_distance_for_count(nlanes2)
            )
    elif side2 == "Bottom":
        if io2 == "IN":
            dir2 = "UP"
            x2 = (
                GLOBAL_OFFSET_X
                + GLOBAL_TILE_MARGIN
                + tile_x2 * GLOBAL_TILE_WIDTH
                + (track_id2 + 1 + GLOBAL_NUM_TRACK) * arrow_distance_for_count(nlanes2)
            )
            y2 = GLOBAL_OFFSET_Y + tile_y2 * GLOBAL_TILE_WIDTH + GLOBAL_TILE_WIDTH
        elif io2 == "OUT":
            dir2 = "DOWN"
            x2 = (
                GLOBAL_OFFSET_X
                + GLOBAL_TILE_MARGIN
                + tile_x2 * GLOBAL_TILE_WIDTH
                + (track_id2 + 1) * arrow_distance_for_count(nlanes2)
            )
            y2 = (
                GLOBAL_OFFSET_Y
                + tile_y2 * GLOBAL_TILE_WIDTH
                + GLOBAL_TILE_WIDTH
                - GLOBAL_TILE_MARGIN
            )
    elif side2 == "Left":
        if io2 == "IN":
            dir2 = "RIGHT"
            x2 = GLOBAL_OFFSET_X + tile_x2 * GLOBAL_TILE_WIDTH
            y2 = (
                GLOBAL_OFFSET_Y
                + GLOBAL_TILE_MARGIN
                + tile_y2 * GLOBAL_TILE_WIDTH
                + (track_id2 + 1 + GLOBAL_NUM_TRACK) * arrow_distance_for_count(nlanes2)
            )
        elif io2 == "OUT":
            dir2 = "LEFT"
            x2 = GLOBAL_OFFSET_X + tile_x2 * GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
            y2 = (
                GLOBAL_OFFSET_Y
                + GLOBAL_TILE_MARGIN
                + tile_y2 * GLOBAL_TILE_WIDTH
                + (track_id2 + 1) * arrow_distance_for_count(nlanes2)
            )
    draw_diagonal_arrow(
        draw=draw, x=x, y=y, dir=dir, x2=x2, y2=y2, dir2=dir2, color=color, width=width
    )


def draw_arrow_on_tile(
    draw,
    tile_x,
    tile_y,
    side,
    io,
    track_id,
    color="Black",
    width=1,
    source_port=False,
    sink_port=False,
    num_tracks=GLOBAL_NUM_TRACK,
):
    if side == "Top":
        if io == "IN":
            dir = "DOWN"
            x = (
                GLOBAL_OFFSET_X
                + GLOBAL_TILE_MARGIN
                + tile_x * GLOBAL_TILE_WIDTH
                + (track_id + 1) * arrow_distance_for_count(num_tracks)
            )
            y = GLOBAL_OFFSET_Y + tile_y * GLOBAL_TILE_WIDTH
        elif io == "OUT":
            dir = "UP"
            x = (
                GLOBAL_OFFSET_X
                + GLOBAL_TILE_MARGIN
                + tile_x * GLOBAL_TILE_WIDTH
                + (track_id + 1 + num_tracks) * arrow_distance_for_count(num_tracks)
            )
            y = GLOBAL_OFFSET_Y + GLOBAL_TILE_MARGIN + tile_y * GLOBAL_TILE_WIDTH
    elif side == "Right":
        if io == "IN":
            dir = "LEFT"
            x = GLOBAL_OFFSET_X + tile_x * GLOBAL_TILE_WIDTH + GLOBAL_TILE_WIDTH
            y = (
                GLOBAL_OFFSET_Y
                + GLOBAL_TILE_MARGIN
                + tile_y * GLOBAL_TILE_WIDTH
                + (track_id + 1) * arrow_distance_for_count(num_tracks)
            )
        elif io == "OUT":
            dir = "RIGHT"
            x = (
                GLOBAL_OFFSET_X
                + tile_x * GLOBAL_TILE_WIDTH
                + GLOBAL_TILE_WIDTH
                - GLOBAL_TILE_MARGIN
            )
            y = (
                GLOBAL_OFFSET_Y
                + GLOBAL_TILE_MARGIN
                + tile_y * GLOBAL_TILE_WIDTH
                + (track_id + 1 + num_tracks) * arrow_distance_for_count(num_tracks)
            )
    elif side == "Bottom":
        if io == "IN":
            dir = "UP"
            x = (
                GLOBAL_OFFSET_X
                + GLOBAL_TILE_MARGIN
                + tile_x * GLOBAL_TILE_WIDTH
                + (track_id + 1 + num_tracks) * arrow_distance_for_count(num_tracks)
            )
            y = GLOBAL_OFFSET_Y + tile_y * GLOBAL_TILE_WIDTH + GLOBAL_TILE_WIDTH
        elif io == "OUT":
            dir = "DOWN"
            x = (
                GLOBAL_OFFSET_X
                + GLOBAL_TILE_MARGIN
                + tile_x * GLOBAL_TILE_WIDTH
                + (track_id + 1) * arrow_distance_for_count(num_tracks)
            )
            y = (
                GLOBAL_OFFSET_Y
                + tile_y * GLOBAL_TILE_WIDTH
                + GLOBAL_TILE_WIDTH
                - GLOBAL_TILE_MARGIN
            )
    elif side == "Left":
        if io == "IN":
            dir = "RIGHT"
            x = GLOBAL_OFFSET_X + tile_x * GLOBAL_TILE_WIDTH
            y = (
                GLOBAL_OFFSET_Y
                + GLOBAL_TILE_MARGIN
                + tile_y * GLOBAL_TILE_WIDTH
                + (track_id + 1 + num_tracks) * arrow_distance_for_count(num_tracks)
            )
        elif io == "OUT":
            dir = "LEFT"
            x = GLOBAL_OFFSET_X + tile_x * GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
            y = (
                GLOBAL_OFFSET_Y
                + GLOBAL_TILE_MARGIN
                + tile_y * GLOBAL_TILE_WIDTH
                + (track_id + 1) * arrow_distance_for_count(num_tracks)
            )
    draw_arrow(
        draw=draw,
        x=x,
        y=y,
        dir=dir,
        color=color,
        width=width,
        source_port=source_port,
        sink_port=sink_port,
    )


def draw_reg_on_tile(draw, tile_x, tile_y, reg_name, track_id, graph):
    sb = graph[tile_x, tile_y].switchbox
    # Per-side lane count + spacing (handles tall SBs)
    def lanes_and_step(side_str):
        n = lane_count_for_side(sb, side_str)
        return n, arrow_distance_for_count(n)

    if "NORTH" in reg_name:
        n, step = lanes_and_step("Top")
        x = (
            GLOBAL_OFFSET_X
            + GLOBAL_TILE_MARGIN
            + tile_x * GLOBAL_TILE_WIDTH
            + (track_id % n + 1 + n) * step
        )
        y = GLOBAL_OFFSET_Y + tile_y * GLOBAL_TILE_WIDTH + (GLOBAL_TILE_WIDTH / 7)
    elif "EAST" in reg_name:
        n, step = lanes_and_step("Right")
        x = (
            GLOBAL_OFFSET_X
            + tile_x * GLOBAL_TILE_WIDTH
            + GLOBAL_TILE_WIDTH
            - (GLOBAL_TILE_WIDTH / 7)
        )
        y = (
            GLOBAL_OFFSET_Y
            + GLOBAL_TILE_MARGIN
            + tile_y * GLOBAL_TILE_WIDTH
            + (track_id % n + 1 + n) * step
        )
    elif "SOUTH" in reg_name:
        n, step = lanes_and_step("Bottom")
        x = (
            GLOBAL_OFFSET_X
            + GLOBAL_TILE_MARGIN
            + tile_x * GLOBAL_TILE_WIDTH
            + (track_id % n + 1) * step
        )
        y = (
            GLOBAL_OFFSET_Y
            + tile_y * GLOBAL_TILE_WIDTH
            + GLOBAL_TILE_WIDTH
            - (GLOBAL_TILE_WIDTH / 7)
        )
    elif "WEST" in reg_name:
        n, step = lanes_and_step("Left")
        x = GLOBAL_OFFSET_X + tile_x * GLOBAL_TILE_WIDTH + (GLOBAL_TILE_WIDTH / 7)
        y = (
            GLOBAL_OFFSET_Y
            + GLOBAL_TILE_MARGIN
            + tile_y * GLOBAL_TILE_WIDTH
            + (track_id % n + 1) * step
        )

    pw = (GLOBAL_TILE_WIDTH - 2 * GLOBAL_TILE_MARGIN) / 25
    xy = [(x - pw, y - pw), (x + pw, y - pw), (x + pw, y + pw), (x - pw, y + pw)]
    draw.polygon(xy=xy, fill="Red", outline="Black", width=1)


def find_last_sb(routing_result_graph, node):
    found_sb = False
    found_port = False

    curr_node = node
    while not found_sb and not found_port:
        # assert len(routing_result_graph.sources[curr_node]) == 1, (
        #     curr_node,
        #     routing_result_graph.sources[curr_node],
        # )

        source = routing_result_graph.sources[curr_node][0]

        if isinstance(source, TileNode) or source.route_type == RouteType.PORT:
            found_port = True
        elif source.route_type == RouteType.SB:
            found_sb = True

        curr_node = source

    if found_sb:
        return curr_node
    else:
        return None


def draw_used_routes(draw, routing_result_graph, width, graph):
    color = lambda: (
        random.randint(64, 128),
        random.randint(64, 255),
        random.randint(64, 255),
        255,
    )
    net_colors = {}

    for node in routing_result_graph.get_routes():
        if node.route_type == RouteType.SB and node.bit_width == width:
            if node.net_id not in net_colors:
                net_colors[node.net_id] = color()

            source_port = False
            sink_port = False
            for source in routing_result_graph.sources[node]:
                if (
                    isinstance(source, RouteNode)
                    and source.route_type == RouteType.PORT
                ):
                    source_port = True
            for sink in routing_result_graph.sinks[node]:
                if isinstance(sink, RouteNode) and sink.route_type == RouteType.PORT:
                    sink_port = True
            nlanes = lane_count_for_side(graph[node.x, node.y].switchbox, side_map[node.side])

            draw_arrow_on_tile(
                draw,
                node.x,
                node.y,
                side_map[node.side],
                io_map[node.io],
                node.track % nlanes,
                color=net_colors[node.net_id],
                width=ARROW_WIDTH,
                source_port=source_port,
                sink_port=sink_port,
                num_tracks=nlanes,
            )

            last_sb = find_last_sb(routing_result_graph, node)

            if last_sb:
                draw_arrow_between_sb(
                    draw,
                    node,
                    last_sb,
                    graph,
                    color=net_colors[node.net_id],
                    width=ARROW_WIDTH,
                )
        elif node.route_type == RouteType.REG and node.bit_width == width:
            draw_reg_on_tile(draw, node.x, node.y, node.reg_name, node.track, graph)


def draw_crit_routes(draw, routing_result_graph, width, crit_nodes, graph):
    color = lambda: (255, 0, 0, 255)
    net_colors = {}

    for node in routing_result_graph.get_routes():
        if (
            node.route_type == RouteType.SB
            and node.bit_width == width
            and node in crit_nodes
        ):
            if node.net_id not in net_colors:
                net_colors[node.net_id] = color()

            source_port = False
            sink_port = False
            for source in routing_result_graph.sources[node]:
                if (
                    isinstance(source, RouteNode)
                    and source.route_type == RouteType.PORT
                ):
                    source_port = True
            for sink in routing_result_graph.sinks[node]:
                if isinstance(sink, RouteNode) and sink.route_type == RouteType.PORT:
                    sink_port = True
            nlanes = lane_count_for_side(graph[node.x, node.y].switchbox, side_map[node.side])

            draw_arrow_on_tile(
                draw,
                node.x,
                node.y,
                side_map[node.side],
                io_map[node.io],
                node.track % nlanes,
                color=net_colors[node.net_id],
                width=ARROW_WIDTH,
                source_port=source_port,
                sink_port=sink_port,
                num_tracks=nlanes,
            )

            last_sb = find_last_sb(routing_result_graph, node)

            if last_sb:
                draw_arrow_between_sb(
                    draw,
                    node,
                    last_sb,
                    graph,
                    color=net_colors[node.net_id],
                    width=ARROW_WIDTH,
                )
        elif node.route_type == RouteType.REG and node.bit_width == width:
            draw_reg_on_tile(draw, node.x, node.y, node.reg_name, node.track, graph)


def add_loc(draw, x, y):
    px = GLOBAL_OFFSET_X + x * GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
    py = GLOBAL_OFFSET_Y + y * GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
    pw = GLOBAL_TILE_WIDTH - 2 * GLOBAL_TILE_MARGIN
    cxy = (px + int(pw * 0.05), py + int(pw * 0.05))
    draw.text(xy=cxy, text=f"({x},{y})", fill="Black")


def create_tile(
    draw,
    x,
    y,
    w=GLOBAL_TILE_WIDTH,
    tile_type=None,
    tile_id=None,
    width=2,
    Content=False,
):
    color_tile = "lightgrey"
    color_line = "Black"
    pr = 0.4
    px = GLOBAL_OFFSET_X + x * GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
    py = GLOBAL_OFFSET_Y + y * GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
    pw = GLOBAL_TILE_WIDTH - 2 * GLOBAL_TILE_MARGIN
    xy = [(px, py), (px + pw, py), (px + pw, py + pw), (px, py + pw)]
    txy = (px + int(pw * pr), py + int(pw * 0.4))
    t2xy = (px + int(pw * pr), py + int(pw * 0.6))
    draw.polygon(xy=xy, fill=color_tile, outline=color_line, width=width)

def lane_count_for_side(sb, side_str):
    try:
        if side_str in ("Left", "Right") and (sb.num_horizontal_track > sb.num_track):
            return sb.num_horizontal_track
        else:
            return GLOBAL_NUM_TRACK
    except Exception:
        return GLOBAL_NUM_TRACK

def arrow_distance_for_count(n):
    inner = GLOBAL_TILE_WIDTH - 2 * GLOBAL_TILE_MARGIN
    return max(1, inner // (n * 2 + 1))

def draw_all_tiles(draw, img, graph, starting_x_coord):
    tmp = Image.new(
        "RGB",
        (
            GLOBAL_TILE_WIDTH + 2 * GLOBAL_OFFSET_X,
            GLOBAL_TILE_WIDTH * 2 + 2 * GLOBAL_OFFSET_Y,
        ),
        "White",
    )
    draw1 = ImageDraw.Draw(tmp)

    # draw sample IO tile
    switchbox = graph[0, 0].switchbox
    sides = switchbox.SIDES

    create_tile(draw=draw1, x=0, y=0)
    for side_str in ["Top", "Right", "Bottom", "Left"]:
        nlanes = lane_count_for_side(switchbox, side_str)
        for io in ["IN", "OUT"]:
            for t in range(nlanes):
                draw_arrow_on_tile(
                    draw1, tile_x=0, tile_y=0,
                    side=side_str, io=io, track_id=t,
                    width=1,
                    num_tracks=nlanes
                )
    box1 = (
        GLOBAL_OFFSET_X,
        GLOBAL_OFFSET_Y,
        GLOBAL_TILE_WIDTH + GLOBAL_OFFSET_X,
        GLOBAL_TILE_WIDTH + GLOBAL_OFFSET_Y,
    )
    region1 = tmp.crop(box1)

    # draw sample tile
    switchbox = graph[0, 1].switchbox
    sides = switchbox.SIDES

    create_tile(draw=draw1, x=0, y=1)

    # draw the arrows
    for side_str in ["Top", "Right", "Bottom", "Left"]:
        nlanes = lane_count_for_side(switchbox, side_str)
        for io in ["IN", "OUT"]:
            for t in range(nlanes):
                draw_arrow_on_tile(
                    draw1, tile_x=0, tile_y=1,
                    side=side_str, io=io, track_id=t,
                    width=1,
                    num_tracks=nlanes
                )
    box2 = (
        GLOBAL_OFFSET_X,
        GLOBAL_TILE_WIDTH + GLOBAL_OFFSET_Y,
        GLOBAL_TILE_WIDTH + GLOBAL_OFFSET_X,
        GLOBAL_TILE_WIDTH * 2 + GLOBAL_OFFSET_Y,
    )
    region2 = tmp.crop(box2)

    for x, y in graph:
        if x < starting_x_coord and y > 0:
            continue
        # MU16: draw only TOP tracks
        if y == MU16_Y_COORD:
            create_tile(draw=draw, x=x, y=y)
            sb = graph[x, y].switchbox
            nlanes = lane_count_for_side(sb, "Top")
            for t in range(nlanes):
                draw_arrow_on_tile(
                    draw, tile_x=x, tile_y=y,
                    side="Top", io="OUT", track_id=t,
                    width=1, num_tracks=nlanes
                )
            add_loc(draw, x, y)
            continue
        box = (
            GLOBAL_OFFSET_X + x * GLOBAL_TILE_WIDTH,
            GLOBAL_OFFSET_Y + y * GLOBAL_TILE_WIDTH,
            GLOBAL_OFFSET_X + (x + 1) * GLOBAL_TILE_WIDTH,
            GLOBAL_OFFSET_Y + (y + 1) * GLOBAL_TILE_WIDTH,
        )
        if y == 0:
            img.paste(region1, box)
        else:
            img.paste(region2, box)
        add_loc(draw, x, y)


def create_tile_types(count, width=2):
    temp = dict()
    w = GLOBAL_TILE_WIDTH - 2 * GLOBAL_TILE_MARGIN
    sx = 0
    ex = w
    dy = w // count
    w = GLOBAL_TILE_WIDTH - 2 * GLOBAL_TILE_MARGIN
    ex = w
    dy = w // count
    tmp = Image.new("RGB", (ex, dy * 10), "White")

    draw = ImageDraw.Draw(tmp)
    ind = 0
    for tile_type in TileType:
        if tile_type == TileType.PE:
            color_tile = "dodgerblue"
            color_line = "Black"
            pr = 0.4
        elif tile_type == TileType.MEM:
            color_tile = "gold"
            color_line = "Black"
            pr = 0.4
        elif tile_type == TileType.POND:
            color_tile = "Khaki"
            color_line = "Black"
            pr = 0.4
        elif tile_type == TileType.IO1 or tile_type == TileType.IO16:
            color_tile = "palegreen"
            color_line = "Black"
            pr = 0.4
        elif tile_type == TileType.REG:
            color_tile = "salmon"
            color_line = "Black"
            pr = 0.4
        else:
            color_tile = "lightgrey"
            color_line = "Black"
            pr = 0.4
        sy = ind * dy
        ey = (ind + 1) * dy
        xy = ((sx, sy), (sx, ey), (ex, ey), (ex, sy))
        draw.polygon(xy=xy, fill=color_tile, outline=color_line, width=width)
        box = (sx, sy, ex, ey)
        temp[tile_type] = tmp.crop(box)
    return temp


def draw_used_tiles(draw, img, tile_history, count, tmp, width=2):
    for loc in tile_history:
        (x, y) = loc
        cont = tile_history[loc]
        w = GLOBAL_TILE_WIDTH - 2 * GLOBAL_TILE_MARGIN
        sx = GLOBAL_OFFSET_X + x * GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
        ex = sx + w
        by = GLOBAL_OFFSET_Y + y * GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
        ey_box = by + w

        local_slots = max(1, len(cont[0]))
        band_h = w // local_slots
        for i in range(len(cont[0])):
            tile_type = cont[0][i]
            sy = by + i * band_h
            ey = sy + band_h if i < local_slots - 1 else ey_box
            # Sample fill color from the template stripe
            try:
                fill_color = tmp[tile_type].getpixel((4, 4))
            except Exception:
                fill_color = "lightgrey"
            draw.rectangle([sx, sy, ex, ey], fill=fill_color, outline="Black", width=width)

def label_used_tiles(draw, img, tile_history, count, width=2):
    for loc in tile_history:
        (x, y) = loc
        cont = tile_history[loc]
        w = GLOBAL_TILE_WIDTH - 2 * GLOBAL_TILE_MARGIN
        sx = GLOBAL_OFFSET_X + x * GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
        by = GLOBAL_OFFSET_Y + y * GLOBAL_TILE_WIDTH + GLOBAL_TILE_MARGIN
        row_slots = max(1, len(cont[0]))
        dy_row = w // row_slots
        ey_box = by + w

        n = min(row_slots, len(cont[0]))
        for i in range(n):
            tile_type = cont[0][i]
            tile_id = cont[1][i]
            sy = by + i * dy_row
            ey = sy + dy_row if i < row_slots - 1 else ey_box
            slot_h = ey - sy

            txy1 = (sx + int(w * 0.3), sy + int(slot_h * 0.4))
            txy2 = (sx + int(w * 0.6), sy + int(slot_h * 0.4))
            draw.text(xy=txy1, text=str(tile_type).split("TileType.")[1], fill="Black")
            draw.text(xy=txy2, text=tile_id, fill="Black")

        cxy = (sx + int(w * 0.05), by + int(w * 0.05))
        draw.text(xy=cxy, text=f"({x},{y})", fill="Black")


def load_graph(graph_files):
    graph_result = {}
    for graph_file in graph_files:
        bit_width = os.path.splitext(graph_file)[0]
        bit_width = int(os.path.basename(bit_width))
        graph = pycyclone.io.load_routing_graph(graph_file)
        graph_result[bit_width] = graph
    return graph_result


def visualize_pnr(routing_graphs, routing_result_graph, crit_nodes, app_dir, starting_x_coord=12):
    if not Image or not ImageDraw:
        print("Please install python package Pillow to generate visualization")
        return

    array_width = 0
    array_height = 0
    for node in routing_result_graph.nodes:
        array_width = max(array_width, node.x)
        array_height = max(array_height, node.y)
    array_width += 1
    array_height += 1

    # process tile history
    tiles = routing_result_graph.get_tiles()
    blk_id_list = {tile.tile_id: tile for tile in tiles}
    tile_history = dict()
    count = 1

    for blk_id, node in blk_id_list.items():
        if (node.x, node.y) not in tile_history:
            tile_history[(node.x, node.y)] = [[node.tile_type], [blk_id]]
        else:
            tile_history[(node.x, node.y)][0].append(node.tile_type)
            tile_history[(node.x, node.y)][1].append(blk_id)
            count = max(len(tile_history[(node.x, node.y)][0]), count)

    # create template for tiles

    template = create_tile_types(count)

    for width, graph in routing_graphs.items():
        # initialize image
        img_width = array_width * GLOBAL_TILE_WIDTH + 3 * GLOBAL_OFFSET_X
        img_height = array_height * GLOBAL_TILE_WIDTH + 3 * GLOBAL_OFFSET_X
        img = Image.new("RGB", (img_width, img_height), "White")
        draw = ImageDraw.Draw(img)

        # draw all the tiles
        draw_all_tiles(draw, img, graph, starting_x_coord)

        draw_used_tiles(draw, img, tile_history, count, template)

        draw_used_routes(draw, routing_result_graph, width, graph)

        if crit_nodes is not None:
            draw_crit_routes(draw, routing_result_graph, width, crit_nodes, graph)

        label_used_tiles(draw, img, tile_history, count, template)

        img.save(f"{app_dir}/pnr_result_{width}.png", format="PNG")
