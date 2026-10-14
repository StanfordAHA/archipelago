"""pipeline.reg_into_route puts a pipelining register into every segment of the net
that passes the broken hop (a segment restarting at the net's source repeats the
shared prefix); it used to patch the first segment only, leaving the RMUX with two
different inputs in the route file (2026-10-14)."""
from archipelago.pipeline import reg_into_route


class Node:
    def __init__(self, seg):
        self.seg = seg

    def to_route(self):
        return self.seg


SRC = ["REG", "T3_SOUTH", 3, 6, 4, 17]
SB = ["SB", 2, 6, 5, 2, 1, 17]
RMUX = ["RMUX", "2_2", 6, 5, 17]
REG = ["REG", "T2_WEST", 2, 6, 5, 17]


def test_register_goes_into_every_segment_with_the_hop():
    routes = {"e22": [[SRC, SB, RMUX, ["SB", 1, 5, 5, 1, 1, 17]],
                      [SRC, SB, RMUX, ["SB", 1, 5, 5, 3, 1, 17]]],
              "e9": [[["PORT", "io2f_17", 0, 0, 17], ["SB", 0, 0, 1, 1, 1, 17]]]}
    reg_into_route(routes, Node(SB), Node(REG))
    for seg in routes["e22"]:
        assert seg[:4] == [SRC, SB, REG, RMUX]
    assert len(routes["e9"][0]) == 2


def test_single_segment_unchanged_behaviour():
    routes = {"e1": [[SRC, SB, RMUX]]}
    reg_into_route(routes, Node(SB), Node(REG))
    assert routes["e1"] == [[SRC, SB, REG, RMUX]]
