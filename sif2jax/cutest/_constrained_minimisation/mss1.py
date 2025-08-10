from ._mss_base import MSSBase


class MSS1(MSSBase):
    """A rank-two relaxation of a maximum stable set problem.

    This is a quadratic optimization problem arising from graph theory,
    specifically from the maximum stable set problem. The problem has
    45 vertices and 72 edges.

    Source: N. Gould, March 2002

    Classification: QQR2-AN-90-73

    TODO: Human review needed - constraint Jacobian has minor discrepancies with
    pycutest.
    Most tests pass (668/673) but Hessian and Jacobian tests show differences.
    The implementation follows the SIF file structure correctly but may need
    adjustment in edge constraint ordering or formulation.
    """

    def _get_n_vertices(self) -> int:
        """Return the number of vertices in the graph."""
        return 45

    def _get_n_edges(self) -> int:
        """Return the number of edges in the graph."""
        return 72

    def _get_edges(self) -> tuple:
        """Define the edges based on the SIF file ELEMENT USES section."""
        # These edges are derived from the ELEMENT USES section in the SIF file
        # Each edge connects two vertices (0-indexed)
        edges = [
            (9, 0),  # Edge 1: X10-X1
            (10, 1),  # Edge 2: X11-X2
            (10, 9),  # Edge 3: X11-X10
            (11, 2),  # Edge 4: X12-X3
            (11, 9),  # Edge 5: X12-X10
            (11, 10),  # Edge 6: X12-X11
            (12, 3),  # Edge 7: X13-X4
            (13, 4),  # Edge 8: X14-X5
            (13, 12),  # Edge 9: X14-X13
            (14, 5),  # Edge 10: X15-X6
            (14, 12),  # Edge 11: X15-X13
            (14, 13),  # Edge 12: X15-X14
            (15, 6),  # Edge 13: X16-X7
            (16, 7),  # Edge 14: X17-X8
            (16, 15),  # Edge 15: X17-X16
            (17, 8),  # Edge 16: X18-X9
            (17, 15),  # Edge 17: X18-X16
            (17, 16),  # Edge 18: X18-X17
            (18, 0),  # Edge 19: X19-X1
            (19, 4),  # Edge 20: X20-X5
            (19, 18),  # Edge 21: X20-X19
            (20, 8),  # Edge 22: X21-X9
            (20, 18),  # Edge 23: X21-X19
            (20, 19),  # Edge 24: X21-X20
            (21, 0),  # Edge 25: X22-X1
            (22, 3),  # Edge 26: X23-X4
            (22, 21),  # Edge 27: X23-X22
            (23, 7),  # Edge 28: X24-X8
            (23, 21),  # Edge 29: X24-X22
            (23, 22),  # Edge 30: X24-X23
            (24, 0),  # Edge 31: X25-X1
            (25, 5),  # Edge 32: X26-X6
            (25, 24),  # Edge 33: X26-X25
            (26, 6),  # Edge 34: X27-X7
            (26, 24),  # Edge 35: X27-X25
            (26, 25),  # Edge 36: X27-X26
            (27, 1),  # Edge 37: X28-X2
            (28, 3),  # Edge 38: X29-X4
            (28, 27),  # Edge 39: X29-X28
            (29, 8),  # Edge 40: X30-X9
            (29, 27),  # Edge 41: X30-X28
            (29, 28),  # Edge 42: X30-X29
            (30, 1),  # Edge 43: X31-X2
            (31, 4),  # Edge 44: X32-X5
            (31, 30),  # Edge 45: X32-X31
            (32, 6),  # Edge 46: X33-X7
            (32, 30),  # Edge 47: X33-X31
            (32, 31),  # Edge 48: X33-X32
            (33, 1),  # Edge 49: X34-X2
            (34, 5),  # Edge 50: X35-X6
            (34, 33),  # Edge 51: X35-X34
            (35, 7),  # Edge 52: X36-X8
            (35, 33),  # Edge 53: X36-X34
            (35, 34),  # Edge 54: X36-X35
            (36, 2),  # Edge 55: X37-X3
            (37, 5),  # Edge 56: X38-X6
            (37, 36),  # Edge 57: X38-X37
            (38, 8),  # Edge 58: X39-X9
            (38, 36),  # Edge 59: X39-X37
            (38, 37),  # Edge 60: X39-X38
            (39, 2),  # Edge 61: X40-X3
            (40, 4),  # Edge 62: X41-X5
            (40, 39),  # Edge 63: X41-X40
            (41, 7),  # Edge 64: X42-X8
            (41, 39),  # Edge 65: X42-X40
            (41, 40),  # Edge 66: X42-X41
            (42, 2),  # Edge 67: X43-X3
            (43, 3),  # Edge 68: X44-X4
            (43, 42),  # Edge 69: X44-X43
            (44, 6),  # Edge 70: X45-X7
            (44, 42),  # Edge 71: X45-X43
            (44, 43),  # Edge 72: X45-X44
        ]
        return tuple(edges)
