"""Track widths other than 1/16/17.

On a fabric built from a lake spec with data_width W, the data track is
W + 1 bits, and port and routing-graph names carry that width
(PE_input_width_33_num_0,
PondTop_output_width_33_num_0, f2io_33, 33.graph). Nothing here may assume 17.
"""
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent / "data"


@pytest.mark.parametrize("width", (9, 16, 17, 33, 65))
def test_pond_output_0_at_any_width(width):
    from archipelago.pipeline import POND_OUTPUT_0
    assert POND_OUTPUT_0.fullmatch(f"PondTop_output_width_{width}_num_0")
    assert not POND_OUTPUT_0.fullmatch(f"PondTop_output_width_{width}_num_1")
    assert not POND_OUTPUT_0.fullmatch(f"PondTop_output_width_{width}_num_01")
    assert not POND_OUTPUT_0.fullmatch(f"PondTop_input_width_{width}_num_0")
    assert not POND_OUTPUT_0.fullmatch("")


def test_sta_delays_fall_back_to_17_then_16():
    from archipelago.sta import _TrackWidthFallback
    d = _TrackWidthFallback({"PE_B17_NORTH_SOUTH": 185,
                             "PE_B16_EAST_WEST": 170,
                             "PE_B1_NORTH_SOUTH": 208,
                             "pe": 700})
    assert d["PE_B33_NORTH_SOUTH"] == 185  # dw32
    assert d["PE_B65_NORTH_SOUTH"] == 185  # dw64
    assert d["PE_B9_NORTH_SOUTH"] == 185   # dw8
    assert d["PE_B33_EAST_WEST"] == 170    # no B17 entry: the B16 one
    assert d["PE_B1_NORTH_SOUTH"] == 208   # present keys are untouched
    with pytest.raises(KeyError):
        d["PE_B33_WEST_EAST"]              # in neither width
    with pytest.raises(KeyError):
        d["mem"]                           # not a track-width key


def test_shipped_sta_delays_cover_33_bit_tracks():
    import json
    import archipelago.sta as sta
    from archipelago.sta import _TrackWidthFallback
    with open(Path(sta.__file__).parent / "sta_delays.json") as f:
        delays = _TrackWidthFallback(json.load(f))
    b17 = [k for k in delays if "_B17_" in k]
    assert b17
    for k in b17:
        assert delays[k.replace("_B17_", "_B33_")] == delays[k]


@pytest.mark.parametrize("width", (9, 17, 33, 65))
def test_pe_input_number_at_any_width(width):
    from archipelago.auto_path_balancing import (
        find_pe_input_num_from_driving_edge)
    edges = {"e1": [("p2", f"PE_output_width_{width}_num_0"),
                    ("p3", f"PE_input_width_{width}_num_2")]}
    assert find_pe_input_num_from_driving_edge("p3", edges, "e1") == 2


def test_post_pipe_packed_puts_wide_outputs_on_the_data_io(tmp_path):
    """dw32 scalar_reduction: output IO tile (1, 0) holds both I0 (data) and
    i0 (valid). f2io_33 must go to I0, f2io_1 to i0 (the old "_17_" name test
    sent both to i0)."""
    from archipelago.io import generate_packed_from_place_and_route
    src = DATA / "dw32_scalar_reduction"
    out = tmp_path / "post.packed"
    generate_packed_from_place_and_route(
        str(tmp_path), str(src / "design.place"), str(src / "design.route"),
        str(out), str(tmp_path / "post_compressed.packed"), visualize=False)
    text = out.read_text()
    assert "(I0, f2io_33)" in text
    assert "(i0, f2io_1)" in text
    assert "(i0, f2io_33)" not in text


def test_congestion_counts_every_track_width():
    from archipelago.congestion import collect_track_usage
    # one SB segment on a 33-bit track and one on a 1-bit track
    routing = {"e1": [[("SB", 0, 1, 1, 0, 1, 33)],
                      [("SB", 2, 1, 1, 3, 0, 1)]]}
    usage = collect_track_usage(routing, 2, 2, (1, 33))
    assert set(usage) == {1, 33}
    tile = usage[33][(1, 1)]
    assert sum(tile[s][io][t] for s in tile for io in tile[s]
               for t in tile[s][io]) == 1
