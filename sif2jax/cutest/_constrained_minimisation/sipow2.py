import jax.numpy as jnp

from ..._misc import inexact_asarray
from ..._problem import AbstractConstrainedMinimisation


class SIPOW2(AbstractConstrainedMinimisation):
    """
    SIPOW2 problem.

    This is a discretization of a semi-infinite programming problem, of
    minimizing the variable x_2 within a circle of radius 1. The circle
    is replaced by a discrete set of equally-spaced supporting tangents.
    The symmetry in SIPOW1.SIF is imposed by replacing those constraints
    by an alternative set.

    Source: problem 2 in
    M. J. D. Powell,
    "Log barrier methods for semi-infinite programming calculations"
    Numerical Analysis Report DAMTP 1992/NA11, U. of Cambridge, UK.

    SIF input: A. R. Conn and Nick Gould, August 1993

    classification LLR2-AN-2-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})
    m: int = 2000

    def __check_init__(self):
        if self.m % 2 != 0:
            raise ValueError("m must be even for SIPOW2")

    def objective(self, y, args):
        del args
        x1, x2 = y
        # Linear objective: minimize x2
        return x2

    def y0(self):
        # Starting point
        return inexact_asarray(jnp.array([0.8, 0.5]))

    def args(self):
        return None

    def expected_result(self):
        # Solution not provided in detail in SIF file
        return None

    def expected_objective_value(self):
        # Solution value provided as -1.0
        return jnp.array(-1.0)

    def num_variables(self):
        return 2

    def bounds(self):
        # All variables are free (unbounded)
        return jnp.array([-jnp.inf, -jnp.inf]), jnp.array([jnp.inf, jnp.inf])

    def constraint(self, y):
        x1, x2 = y
        m = self.m
        m_half = m // 2

        # First M/2 constraints: tangent lines
        # Generate angles for the constraints (note: 4*pi instead of 2*pi)
        j_vals = inexact_asarray(jnp.arange(1, m_half + 1))
        angles = 4.0 * jnp.pi * j_vals / inexact_asarray(m)
        cos_vals = jnp.cos(angles)
        sin_vals = jnp.sin(angles)

        # C(j): x1 * cos(4*pi*j/m) + x2 * sin(4*pi*j/m) >= -1
        # In pycutest form: x1 * cos(4*pi*j/m) + x2 * sin(4*pi*j/m) + 1
        first_half = x1 * cos_vals + x2 * sin_vals + 1.0

        # Second M/2 constraints: simply x1 >= -1
        # In pycutest form: x1 + 1
        second_half = jnp.full(m_half, x1 + 1.0)

        # Combine all inequality constraints
        ineq_constraint = jnp.concatenate([first_half, second_half])

        # Equality constraints: none
        eq_constraint = None

        return eq_constraint, ineq_constraint
