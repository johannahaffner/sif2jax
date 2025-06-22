import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree, Scalar

from ..._problem import AbstractUnconstrainedMinimisation


class MANCINO(AbstractUnconstrainedMinimisation):
    """Mancino's function with variable dimension.

    Source:
        E. Spedicato,
        "Computational experience with quasi-Newton algorithms for
        minimization problems of moderate size",
        Report N-175, CISE, Milano, 1975.

    See also:
        Buckley #51 (p. 72), Schittkowski #391 (for N = 30)

    SIF input:
        Ph. Toint, Dec 1989.
        correction by Ph. Shott, January, 1995.
        correction by S. Gratton & Ph. Toint, May 2024

    Classification: SUR2-AN-V-0
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    n: int = 100  # Can also be tested with 60 as in FOSSEE implementation

    def objective(self, y: Float[Array, "..."], args: PyTree) -> Scalar:
        """Compute the objective function for the Mancino problem based on AMPL."""
        n = self.n

        # AMPL objective: sum {i in 1..N} alpha[i]^2
        # where alpha[i] = 1400*x[i] + (i-50)^3 +
        # sum {j in 1..N} v[i,j]*((sin(log(v[i,j])))^5 + (cos(log(v[i,j])))^5)
        # and v[i,j] = sqrt(x[i]^2 + i/j)

        def compute_alpha_i(i):
            i_float = i.astype(jnp.float64)
            i_int = i.astype(jnp.int32)
            x_i = y[i_int - 1]  # Convert to 0-indexed

            # First term: 1400*x[i] (directly from AMPL)
            linear_term = 1400.0 * x_i

            # Second term: (i-50)^3
            cubic_term = (i_float - 50.0) ** 3

            # Third term: sum over all j from 1 to N
            all_j = jnp.arange(1, n + 1, dtype=jnp.float64)

            def compute_v_term(j):
                # v[i,j] = sqrt(x[i]^2 + i/j) - this includes ALL j, including j=i
                v_ij = jnp.sqrt(x_i**2 + i_float / j)

                # Compute sin(log(v[i,j]))^5 + cos(log(v[i,j]))^5
                log_v = jnp.log(v_ij)
                sin_val = jnp.sin(log_v)
                cos_val = jnp.cos(log_v)
                sin_pow5 = jnp.power(sin_val, 5)
                cos_pow5 = jnp.power(cos_val, 5)

                return v_ij * (sin_pow5 + cos_pow5)

            sum_term = jnp.sum(jax.vmap(compute_v_term)(all_j))

            # alpha[i] = 1400*x[i] + (i-50)^3 + sum{...}
            alpha_i = linear_term + cubic_term + sum_term
            return alpha_i

        # Compute alpha for all i and return sum of squares
        indices = jnp.arange(1, n + 1)
        alphas = jax.vmap(compute_alpha_i)(indices)

        # Return sum of squares without any scaling
        return jnp.sum(alphas**2)

    def y0(self) -> Float[Array, "n"]:
        """Initial guess for the Mancino problem based on AMPL file."""
        n = self.n

        # AMPL initialization: x[i] := -8.710996D-4*((i-50)^3 + sum {...})
        coefficient = -8.710996e-4

        # Function to compute the sum term for each i
        def compute_sum_for_i(i):
            i_float = i.astype(jnp.float64)

            # Sum over all j from 1 to N (including j=i, as per AMPL)
            all_j = jnp.arange(1, n + 1, dtype=jnp.float64)

            def term_i_j(j):
                # AMPL: sqrt(i/j)*((sin(log(sqrt(i/j))))^5 + (cos(log(sqrt(i/j))))^5)
                i_over_j = i_float / j
                sqrt_i_over_j = jnp.sqrt(i_over_j)
                log_val = jnp.log(sqrt_i_over_j)
                sin_val = jnp.sin(log_val)
                cos_val = jnp.cos(log_val)

                # AMPL: sin^5 + cos^5
                sin_pow5 = jnp.power(sin_val, 5)
                cos_pow5 = jnp.power(cos_val, 5)

                return sqrt_i_over_j * (sin_pow5 + cos_pow5)

            return jnp.sum(jax.vmap(term_i_j)(all_j))

        # Compute initial values for all indices
        indices = jnp.arange(1, n + 1)

        def compute_x_i(i):
            i_float = i.astype(jnp.float64)
            # AMPL: (i-50)^3 + sum{...}
            cubic_term = (i_float - 50.0) ** 3
            sum_term = compute_sum_for_i(i)
            return coefficient * (cubic_term + sum_term)

        return jax.vmap(compute_x_i)(indices)

    def args(self) -> None:
        return None

    def expected_result(self):
        return None

    def expected_objective_value(self):
        return None
