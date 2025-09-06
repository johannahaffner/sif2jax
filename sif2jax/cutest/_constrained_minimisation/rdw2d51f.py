# TODO: Human review needed
# Complex finite-element PDE-constrained optimization (132k vars, 65k constraints)
# Implementation is computationally intensive and needs optimization
# Current implementation may not scale efficiently to the required problem size

import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


class RDW2D51F(AbstractConstrainedMinimisation):
    """RDW2D51F problem - A finite-element approximation to distributed optimal control.

    A finite-element approximation to the distributed optimal control problem

        min 1/2||u-v||_L2^2 + beta ||f||_L2^2

    subject to - nabla^2 u = f

    where v is given on and within the boundary of a unit [0,1] box in
    2 dimensions, and u = v on its boundary. The discretization uses
    quadrilateral elements. There are simple bounds on the controls f.

    The problem is stated as a quadratic program.

    Source: example 5.1 in
    T. Rees, H. S. Dollar and A. J. Wathen
    "Optimal solvers for PDE-constrained optimization"
    SIAM J. Sci. Comp. (to appear) 2009

    with the control bounds as specified in

    M. Stoll and A. J. Wathen
    "Preconditioning for PDE constrained optimization with
     control constraints"
    OUCL Technical Report 2009

    SIF input: Nick Gould, May 2009
               correction by S. Gratton & Ph. Toint, May 2024

    Classification: QLR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Grid parameters
    N: int = 256
    beta: float = 0.01

    def _compute_target_data(self):
        """Compute target data V(i,j)."""
        N = self.N
        h = 1.0 / N
        V = jnp.zeros((N + 1, N + 1))

        # For i in [0, N/2], j in [0, N/2]: V = (2*i*h - 1)^2 * (2*j*h - 1)^2
        N_half = N // 2
        for i in range(N_half + 1):
            for j in range(N_half + 1):
                val_i = 2.0 * i * h - 1.0
                val_j = 2.0 * j * h - 1.0
                V = V.at[i, j].set(val_i**2 * val_j**2)

        # Other regions are zero (already initialized)
        return V

    @property
    def n(self):
        """Number of variables."""
        return 2 * (self.N + 1) * (self.N + 1)

    def objective(self, y, args):
        """Compute the objective function."""
        del args
        N = self.N
        h = 1.0 / N
        h2 = h * h
        h2_36 = h2 / 36.0
        beta_h2_36 = 2.0 * self.beta * h2_36

        # Precompute target data
        V = self._compute_target_data()

        # Variables: [F(0,0), F(0,1), ..., F(N,N), U(0,0), U(0,1), ..., U(N,N)]
        n_grid = (N + 1) * (N + 1)
        F = y[:n_grid].reshape(N + 1, N + 1)
        U = y[n_grid:].reshape(N + 1, N + 1)

        # Objective: sum over elements (i,j) from 0 to N-1 of
        # h^2/36 * M_element(U-V) + 2*beta*h^2/36 * M0_element(F)

        obj = jnp.array(0.0, dtype=y.dtype)
        for i in range(N):
            for j in range(N):
                # Element mass matrix contributions for (u-v)^T M (u-v)
                u1, u2, u3, u4 = U[i, j], U[i, j + 1], U[i + 1, j], U[i + 1, j + 1]
                v1, v2, v3, v4 = V[i, j], V[i, j + 1], V[i + 1, j], V[i + 1, j + 1]

                uv1, uv2, uv3, uv4 = u1 - v1, u2 - v2, u3 - v3, u4 - v4

                # M element: 2*(uv1^2 + uv2^2 + uv3^2 + uv4^2) + 2*(...) + (...)
                M_val = (
                    2.0 * (uv1**2 + uv2**2 + uv3**2 + uv4**2)
                    + 2.0 * (uv1 * uv2 + uv1 * uv3 + uv2 * uv4 + uv3 * uv4)
                    + (uv1 * uv4 + uv2 * uv3)
                )

                obj += h2_36 * M_val

                # Element mass matrix contributions for f^T M0 f
                f1, f2, f3, f4 = F[i, j], F[i, j + 1], F[i + 1, j], F[i + 1, j + 1]

                # M0 element: same structure as M but for F
                M0_val = (
                    2.0 * (f1**2 + f2**2 + f3**2 + f4**2)
                    + 2.0 * (f1 * f2 + f1 * f3 + f2 * f4 + f3 * f4)
                    + (f1 * f4 + f2 * f3)
                )

                obj += beta_h2_36 * M0_val

        return obj

    def constraint(self, y):
        """Compute the constraints."""
        N = self.N
        h = 1.0 / N
        h2 = h * h
        minus_h2_36 = -h2 / 36.0
        one_sixth = 1.0 / 6.0

        n_grid = (N + 1) * (N + 1)
        F = y[:n_grid].reshape(N + 1, N + 1)
        U = y[n_grid:].reshape(N + 1, N + 1)

        # Equality constraints L(i,j) = 0 for i,j in [1, N-1]
        constraints = []

        for i in range(1, N):
            for j in range(1, N):
                # Constraint L(i,j) involves linear combinations of A,B,C,D with U
                # and P,Q,R,S elements with F
                constraint_val = 0.0

                # A(i-1,j-1), B(i-1,j), C(i,j-1), D(i,j) contributions with 1/6 coeff
                # These are linear element functions
                # A element: 4*u1 - u2 - u3 - 2*u4
                u1, u2, u3, u4 = U[i - 1, j - 1], U[i - 1, j], U[i, j - 1], U[i, j]
                constraint_val += one_sixth * (4.0 * u1 - u2 - u3 - 2.0 * u4)

                # B element: -u1 + 4*u2 - 2*u3 - u4
                u1, u2, u3, u4 = U[i - 1, j], U[i - 1, j + 1], U[i, j], U[i, j + 1]
                constraint_val += one_sixth * (-u1 + 4.0 * u2 - 2.0 * u3 - u4)

                # C element: -u1 - 2*u2 + 4*u3 - u4
                u1, u2, u3, u4 = U[i, j - 1], U[i, j], U[i + 1, j - 1], U[i + 1, j]
                constraint_val += one_sixth * (-u1 - 2.0 * u2 + 4.0 * u3 - u4)

                # D element: -2*u1 - u2 - u3 + 4*u4
                u1, u2, u3, u4 = U[i, j], U[i, j + 1], U[i + 1, j], U[i + 1, j + 1]
                constraint_val += one_sixth * (-2.0 * u1 - u2 - u3 + 4.0 * u4)

                # P,Q,R,S elements for F with -h^2/36 coefficient
                # P element: 4*f1 + 2*f2 + 2*f3 + f4
                f1, f2, f3, f4 = F[i - 1, j - 1], F[i - 1, j], F[i, j - 1], F[i, j]
                constraint_val += minus_h2_36 * (4.0 * f1 + 2.0 * f2 + 2.0 * f3 + f4)

                # Q element: 2*f1 + 4*f2 + f3 + 2*f4
                f1, f2, f3, f4 = F[i - 1, j], F[i - 1, j + 1], F[i, j], F[i, j + 1]
                constraint_val += minus_h2_36 * (2.0 * f1 + 4.0 * f2 + f3 + 2.0 * f4)

                # R element: 2*f1 + f2 + 4*f3 + 2*f4
                f1, f2, f3, f4 = F[i, j - 1], F[i, j], F[i + 1, j - 1], F[i + 1, j]
                constraint_val += minus_h2_36 * (2.0 * f1 + f2 + 4.0 * f3 + 2.0 * f4)

                # S element: f1 + 2*f2 + 2*f3 + 4*f4
                f1, f2, f3, f4 = F[i, j], F[i, j + 1], F[i + 1, j], F[i + 1, j + 1]
                constraint_val += minus_h2_36 * (f1 + 2.0 * f2 + 2.0 * f3 + 4.0 * f4)

                constraints.append(constraint_val)

        equality_constraints = jnp.array(constraints)
        inequality_constraints = None

        return equality_constraints, inequality_constraints

    @property
    def y0(self):
        """Initial guess."""
        N = self.N
        n_grid = (N + 1) * (N + 1)

        # Initialize both F and U to zero
        return jnp.zeros(2 * n_grid)

    @property
    def args(self):
        """Additional arguments."""
        return None

    @property
    def bounds(self):
        """Bounds on variables."""
        N = self.N
        h = 1.0 / N
        n_grid = (N + 1) * (N + 1)

        # Precompute target data
        V = self._compute_target_data()

        lower = jnp.full(2 * n_grid, -jnp.inf)
        upper = jnp.full(2 * n_grid, jnp.inf)

        # Set bounds for boundary U variables (fixed to target values)
        # and F variables (bounds according to SIF)

        # F boundary variables are fixed at 0
        for i in [0, N]:
            for j in range(N + 1):
                idx = i * (N + 1) + j
                lower = lower.at[idx].set(0.0)
                upper = upper.at[idx].set(0.0)

        for j in [0, N]:
            for i in range(1, N):
                idx = i * (N + 1) + j
                lower = lower.at[idx].set(0.0)
                upper = upper.at[idx].set(0.0)

        # Interior F variables have bounds based on exponential function
        for i in range(1, N):
            x1 = i * h
            x1_2 = x1 * x1
            factor = 0.1 * (2.0 - x1)

            for j in range(1, N):
                x2 = j * h
                x2_2 = x2 * x2
                lower_bound = factor * jnp.exp(-x1_2 - x2_2)

                idx = i * (N + 1) + j
                lower = lower.at[idx].set(lower_bound)

                # Upper bounds depend on j
                if j <= N // 2:
                    upper = upper.at[idx].set(0.6)
                else:
                    upper = upper.at[idx].set(0.9)

        # U boundary variables are fixed to target values V
        for i in [0, N]:
            for j in range(N + 1):
                idx = n_grid + i * (N + 1) + j
                val = V[i, j]
                lower = lower.at[idx].set(val)
                upper = upper.at[idx].set(val)

        for j in [0, N]:
            for i in range(1, N):
                idx = n_grid + i * (N + 1) + j
                val = V[i, j]
                lower = lower.at[idx].set(val)
                upper = upper.at[idx].set(val)

        return lower, upper

    @property
    def expected_result(self):
        """Expected optimal solution."""
        return None

    @property
    def expected_objective_value(self):
        """Expected optimal objective value."""
        return None
