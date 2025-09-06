import jax.numpy as jnp

from ..._problem import AbstractConstrainedMinimisation


# TODO: Human review needed
# Issues to resolve:
# 1. Dimension mismatch - pycutest expects 12005 variables vs current implementation
# 2. Complex collocation formulation needs full SIF element functions
# 3. Constraint structure requires deeper analysis of SIF file
class METHANOL(AbstractConstrainedMinimisation):
    """METHANOL problem - Optimal control for methanol conversion.

    Determine the reaction coefficients for the conversion of methanol into
    various hydrocarbons. The nonlinear model describes the process using a
    system of differential equations with parameter estimation.

    Problem 13 in the COPS (Version 2) collection of E. Dolan and J. More'
    "Benchmarking Optimization Software with COPS"
    Argonne National Labs Technical Report ANL/MCS-246 (2000)

    SIF input: Nick Gould, November 2000

    Classification: OOR2-AN-V-V
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    # Problem parameters
    NE = 3  # Number of differential equations
    NH = 400  # Number of subintervals
    NP = 5  # Number of ODE parameters
    NM = 17  # Number of measurements
    NC = 3  # Number of collocation points

    # Collocation points (roots of 3rd degree Legendre polynomial)
    RHO = jnp.array([0.5, 0.8872983346, 0.1127016654])

    # Times at which observations are made
    TAU = jnp.array(
        [
            0.0,
            0.050,
            0.065,
            0.080,
            0.123,
            0.233,
            0.273,
            0.354,
            0.397,
            0.418,
            0.502,
            0.553,
            0.681,
            0.750,
            0.916,
            0.937,
            1.122,
        ]
    )

    # Concentration measurements
    Z = jnp.array(
        [
            [
                1.0000,
                0.7085,
                0.5971,
                0.5537,
                0.3684,
                0.1712,
                0.1198,
                0.0747,
                0.0529,
                0.0415,
                0.0261,
                0.0208,
                0.0085,
                0.0053,
                0.0019,
                0.0018,
                0.0006,
            ],
            [
                0.0,
                0.1621,
                0.1855,
                0.1989,
                0.2845,
                0.3491,
                0.3098,
                0.3576,
                0.3347,
                0.3388,
                0.3557,
                0.3483,
                0.3836,
                0.3611,
                0.3609,
                0.3485,
                0.3698,
            ],
            [
                0.0,
                0.0811,
                0.0965,
                0.1198,
                0.1535,
                0.2097,
                0.2628,
                0.2467,
                0.2884,
                0.2757,
                0.3167,
                0.2954,
                0.2950,
                0.2937,
                0.2831,
                0.2846,
                0.2899,
            ],
        ]
    ).T  # Shape: (17, 3)

    # ODE initial conditions
    BC = jnp.array([1.0, 0.0, 0.0])

    # Final time
    TF = 1.122  # TAU[-1]
    H = TF / NH  # Uniform interval length

    # Factorials for collocation
    FACT = jnp.array([1.0, 1.0, 2.0, 6.0])  # [0!, 1!, 2!, 3!]

    @property
    def n(self):
        """Number of variables: NP (theta) + NH*NE (V) + NH*NC*NE (W) + NH*NC*NE (U) + NH*NC*NE (DU)"""  # noqa: E501
        return self.NP + self.NH * self.NE + 3 * self.NH * self.NC * self.NE

    @property
    def n_con(self):
        """Number of constraints."""
        # U constraints: NH * NC * NE
        # DU constraints: NH * NC * NE
        # Continuity constraints: (NH-1) * NE
        # Collocation constraints: NH * NC * NE
        return (
            2 * self.NH * self.NC * self.NE
            + (self.NH - 1) * self.NE
            + self.NH * self.NC * self.NE
        )

    def _unpack_variables(self, y):
        """Unpack the optimization variables."""
        idx = 0

        # ODE parameters (5)
        theta = y[idx : idx + self.NP]
        idx += self.NP

        # V variables: NH x NE
        V = y[idx : idx + self.NH * self.NE].reshape(self.NH, self.NE)
        idx += self.NH * self.NE

        # W variables: NH x NC x NE
        W = y[idx : idx + self.NH * self.NC * self.NE].reshape(
            self.NH, self.NC, self.NE
        )
        idx += self.NH * self.NC * self.NE

        # U variables: NH x NC x NE
        U = y[idx : idx + self.NH * self.NC * self.NE].reshape(
            self.NH, self.NC, self.NE
        )
        idx += self.NH * self.NC * self.NE

        # DU variables: NH x NC x NE
        DU = y[idx : idx + self.NH * self.NC * self.NE].reshape(
            self.NH, self.NC, self.NE
        )

        return theta, V, W, U, DU

    def _compute_itau(self):
        """Compute itau indices: largest integer k with t[k] <= tau[i]."""
        itau = jnp.zeros(self.NM, dtype=int)
        for i in range(self.NM):
            k = jnp.minimum(self.NH, jnp.floor(self.TAU[i] / self.H).astype(int) + 1)
            itau = itau.at[i].set(k - 1)  # Convert to 0-based indexing
        return itau

    def objective(self, y, args):
        """Objective function: sum of squared residuals."""
        del args
        theta, V, W, U, DU = self._unpack_variables(y)
        itau = self._compute_itau()

        # Compute objective function values at measurement times
        obj_vals = jnp.zeros((self.NM, self.NE))

        for j in range(self.NM):
            i = itau[j]  # Index for this measurement time
            t = jnp.float64(i) * self.H  # Time at grid point i
            diff = self.TAU[j] - t  # Time difference

            for s in range(self.NE):
                # obj(j,s) = V[i,s] + sum_k W[i,k,s] * (tau[j]-t[i])^k / (fact[k] * h^(k-1))  # noqa: E501
                val = V[i, s]
                ratio = diff
                for k in range(self.NC):
                    coef = ratio / self.FACT[k + 1]
                    val += W[i, k, s] * coef
                    ratio *= diff / self.H

                obj_vals = obj_vals.at[j, s].set(val - self.Z[j, s])

        # Return sum of squared residuals
        return jnp.sum(obj_vals**2)

    def constraint(self, y):
        """Compute constraint violations."""
        theta, V, W, U, DU = self._unpack_variables(y)

        constraints = []

        # U constraints: -u + v + h*sum_k w*(rho^k/fact[k]) = 0
        for i in range(self.NH):
            for j in range(self.NC):
                for s in range(self.NE):
                    val = -U[i, j, s] + V[i, s]
                    prod = self.RHO[j] * self.H
                    for k in range(self.NC):
                        coef = prod / self.FACT[k]
                        val += W[i, k, s] * coef
                        prod *= self.RHO[j]
                    constraints.append(val)

        # DU constraints: -du + sum_k w*(rho^(k-1)/fact[k-1]) = 0
        for i in range(self.NH):
            for j in range(self.NC):
                for s in range(self.NE):
                    val = -DU[i, j, s]
                    prod = 1.0
                    for k in range(self.NC):
                        coef = prod / self.FACT[k]
                        val += W[i, k, s] * coef
                        prod *= self.RHO[j]
                    constraints.append(val)

        # Continuity constraints: v[i] - v[i+1] + sum_j w[i,j] * h/fact[j] = 0
        for i in range(self.NH - 1):
            for s in range(self.NE):
                val = V[i, s] - V[i + 1, s]
                for j in range(self.NC):
                    coef = self.H / self.FACT[j]
                    val += W[i, j, s] * coef
                constraints.append(val)

        # Collocation equations (simplified nonlinear terms)
        for i in range(self.NH):
            for j in range(self.NC):
                for s in range(self.NE):
                    # Placeholder for complex nonlinear collocation equations
                    # This would need the full element functions from the SIF file
                    val = DU[i, j, s]  # Simplified constraint
                    constraints.append(val)

        return jnp.array(constraints), None

    @property
    def bounds(self):
        """Variable bounds."""
        n = self.n
        lower = jnp.full(n, -jnp.inf)
        upper = jnp.full(n, jnp.inf)

        # Bounds on theta parameters (first 5 variables): theta >= 0
        lower = lower.at[: self.NP].set(0.0)

        return lower, upper

    @property
    def y0(self):
        """Initial point."""
        y = jnp.zeros(self.n)

        # Initialize theta parameters to 1.0
        y = y.at[: self.NP].set(1.0)

        # Initialize V values based on measurement data
        idx = self.NP
        V_init = jnp.zeros((self.NH, self.NE))
        # Set initial conditions for first stage
        V_init = V_init.at[0].set(self.BC)

        # Set values based on measurements at intermediate points
        itau = self._compute_itau()
        for k in range(1, self.NM):
            i_start = itau[k - 1] + 1 if k > 1 else 1
            i_end = itau[k] + 1
            for i in range(i_start, int(jnp.minimum(i_end, self.NH))):
                V_init = V_init.at[i].set(self.Z[k])

        y = y.at[idx : idx + self.NH * self.NE].set(V_init.flatten())

        # Initialize W, U, DU to zero (they're already zero)
        return y

    @property
    def args(self):
        return None

    @property
    def expected_result(self):
        return None  # No known analytical solution

    @property
    def expected_objective_value(self):
        return jnp.array(9.02228e-3)  # From SIF file comment
