import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: human review required
class DIXMAANA1(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version A1).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It's a variant of DIXMAANA with beta=0, so elements/groups
    of type 2 are removed.

    The objective function includes quadratic terms, quartic terms, and bilinear terms.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.
    update Nick Gould, August 2022, to remove beta=0 terms.

    Classification: OUR2-AN-V-0
    """

    n: int = 3000  # Default dimension

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters
        alpha = 1.0
        # Note: Beta is zero for DIXMAANA1, so not used
        gamma = 0.125
        delta = 0.125

        # Note: We don't use the i_vals directly in this implementation

        # Compute the first term (type 1): sum(alpha * (x_i)^2)
        term1 = alpha * jnp.sum(y**2)

        # Compute the third term (type 3): sum(gamma * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        term3 = gamma * jnp.sum((y[indices1] ** 2) * (y[indices2] ** 4))

        # Compute the fourth term (type 4): sum(delta * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        term4 = delta * jnp.sum(y[indices1] * y[indices2])

        return term1 + term3 + term4

    def y0(self):
        # Initial value is 2.0 for all variables
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)


# TODO: human review required
class DIXMAANE1(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version E1).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It's a variant of DIXMAANE with beta=0, so elements/groups
    of type 2 are removed.

    The objective function includes quadratic terms, quartic terms, and bilinear terms
    with non-trivial powers of (i/n) in the weights.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.

    Classification: OUR2-AN-V-0
    """

    n: int = 3000  # Default dimension

    def __init__(self, n=None):
        if n is not None:
            self.n = n

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters
        alpha = 1.0
        # Beta is zero for DIXMAANE1
        gamma = 0.125
        delta = 0.125

        # Powers for each group
        k1 = 1  # Power for group 1
        # k2 not used since beta=0
        k3 = 0  # Power for group 3
        k4 = 1  # Power for group 4

        # Indices for each variable
        # i_vals not used directly
        # i_over_n not used directly

        # Compute the first term (type 1): sum(alpha * (i/n)^k1 * (x_i)^2)
        term1 = alpha * jnp.sum(((jnp.arange(1, n + 1) / n) ** k1) * (y**2))

        # Compute the third term (type 3): sum(gamma * (i/n)^k3 * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term3 = gamma * jnp.sum(
            ((valid_i1 + 1) / n) ** k3 * (y[valid_i1] ** 2) * (y[valid_i2] ** 4)
        )

        # Compute the fourth term (type 4): sum(delta * (i/n)^k4 * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term4 = delta * jnp.sum(((valid_i1 + 1) / n) ** k4 * y[valid_i1] * y[valid_i2])

        return term1 + term3 + term4

    def y0(self):
        # Initial value is 2.0 for all variables
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)


# TODO: human review required
class DIXMAANF(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version F).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It includes quadratic, sin, quartic, and bilinear terms
    with non-trivial powers of (i/n) in the weights.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.

    Classification: OUR2-AN-V-0
    """

    n: int = 3000  # Default dimension

    def __init__(self, n=None):
        if n is not None:
            self.n = n

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters
        alpha = 1.0
        beta = 0.0625
        gamma = 0.0625
        delta = 0.0625

        # Powers for each group
        k1 = 1  # Power for group 1
        k2 = 0  # Power for group 2
        k3 = 0  # Power for group 3
        k4 = 1  # Power for group 4

        # Indices for each variable
        # i_vals not used directly
        # i_over_n not used directly

        # Compute the 1st term (type 1): sum(alpha * (i/n)^k1 * (x_i)^2)
        term1 = alpha * jnp.sum(((jnp.arange(1, n + 1) / n) ** k1) * (y**2))

        # Compute the 2nd term (type 2): sum(beta * (i/n)^k2 * sin(x_i) * sin(x_{i+1})
        # for i from 1 to n-1
        indices1 = jnp.arange(n - 1)
        indices2 = indices1 + 1
        term2 = beta * jnp.sum(
            ((indices1 + 1) / n) ** k2 * jnp.sin(y[indices1]) * jnp.sin(y[indices2])
        )

        # Compute the 3rd term (type 3): sum(gamma * (i/n)^k3 * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term3 = gamma * jnp.sum(
            ((valid_i1 + 1) / n) ** k3 * (y[valid_i1] ** 2) * (y[valid_i2] ** 4)
        )

        # Compute the 4th term (type 4): sum(delta * (i/n)^k4 * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term4 = delta * jnp.sum(((valid_i1 + 1) / n) ** k4 * y[valid_i1] * y[valid_i2])

        return term1 + term2 + term3 + term4

    def y0(self):
        # Initial value is 2.0 for all variables
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)


# TODO: human review required
class DIXMAANG(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version G).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It includes quadratic, sin, quartic, and bilinear terms
    with non-trivial powers of (i/n) in the weights and increased parameter values.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.

    Classification: OUR2-AN-V-0
    """

    n: int = 3000  # Default dimension

    def __init__(self, n=None):
        if n is not None:
            self.n = n

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters
        alpha = 1.0
        beta = 0.125
        gamma = 0.125
        delta = 0.125

        # Powers for each group
        k1 = 1  # Power for group 1
        k2 = 0  # Power for group 2
        k3 = 0  # Power for group 3
        k4 = 1  # Power for group 4

        # Indices for each variable
        # i_vals not used directly
        # i_over_n not used directly

        # Compute the 1st term (type 1): sum(alpha * (i/n)^k1 * (x_i)^2)
        term1 = alpha * jnp.sum(((jnp.arange(1, n + 1) / n) ** k1) * (y**2))

        # Compute the 2nd term (type 2): sum(beta * (i/n)^k2 * sin(x_i) * sin(x_{i+1})
        # for i from 1 to n-1
        indices1 = jnp.arange(n - 1)
        indices2 = indices1 + 1
        term2 = beta * jnp.sum(
            ((indices1 + 1) / n) ** k2 * jnp.sin(y[indices1]) * jnp.sin(y[indices2])
        )

        # Compute the 3rd term (type 3): sum(gamma * (i/n)^k3 * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term3 = gamma * jnp.sum(
            ((valid_i1 + 1) / n) ** k3 * (y[valid_i1] ** 2) * (y[valid_i2] ** 4)
        )

        # Compute the 4th term (type 4): sum(delta * (i/n)^k4 * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term4 = delta * jnp.sum(((valid_i1 + 1) / n) ** k4 * y[valid_i1] * y[valid_i2])

        return term1 + term2 + term3 + term4

    def y0(self):
        # Initial value is 2.0 for all variables
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)


# TODO: human review required
class DIXMAANH(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version H).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It includes quadratic, sin, quartic, and bilinear terms
    with non-trivial powers of (i/n) in the weights and significantly increased
    parameter values.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.

    Classification: OUR2-AN-V-0
    """

    n: int = 3000  # Default dimension

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters
        alpha = 1.0
        beta = 0.26
        gamma = 0.26
        delta = 0.26

        # Powers for each group
        k1 = 1  # Power for group 1
        k2 = 0  # Power for group 2
        k3 = 0  # Power for group 3
        k4 = 1  # Power for group 4

        # Indices for each variable
        # i_vals not used directly
        # i_over_n not used directly

        # Compute the 1st term (type 1): sum(alpha * (i/n)^k1 * (x_i)^2)
        term1 = alpha * jnp.sum(((jnp.arange(1, n + 1) / n) ** k1) * (y**2))

        # Compute the 2nd term (type 2): sum(beta * (i/n)^k2 * sin(x_i) * sin(x_{i+1})
        # for i from 1 to n-1
        indices1 = jnp.arange(n - 1)
        indices2 = indices1 + 1
        term2 = beta * jnp.sum(
            ((indices1 + 1) / n) ** k2 * jnp.sin(y[indices1]) * jnp.sin(y[indices2])
        )

        # Compute the 3rd term (type 3): sum(gamma * (i/n)^k3 * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term3 = gamma * jnp.sum(
            ((valid_i1 + 1) / n) ** k3 * (y[valid_i1] ** 2) * (y[valid_i2] ** 4)
        )

        # Compute the 4th term (type 4): sum(delta * (i/n)^k4 * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term4 = delta * jnp.sum(((valid_i1 + 1) / n) ** k4 * y[valid_i1] * y[valid_i2])

        return term1 + term2 + term3 + term4

    def y0(self):
        # Initial value is 2.0 for all variables
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)


# TODO: human review required
class DIXMAANB(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version B).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It includes all four term types.

    The objective function includes quadratic terms, quartic terms,
    bilinear terms, and sin terms.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.

    Classification: OUR2-AN-V-0
    """

    n: int = 3000  # Default dimension

    def __init__(self, n=None):
        if n is not None:
            self.n = n

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters
        alpha = 1.0
        beta = 0.0625
        gamma = 0.0625
        delta = 0.0625

        # Indices for each variable
        # i_vals not used directly
        # i_over_n not used directly

        # Compute the first term (type 1): sum(alpha * (x_i)^2)
        term1 = alpha * jnp.sum(y**2)

        # Compute the second term (type 2): sum(beta * sin(x_i) * sin(x_{i+1})
        # for i from 1 to n-1
        indices1 = jnp.arange(n - 1)
        indices2 = indices1 + 1
        term2 = beta * jnp.sum(jnp.sin(y[indices1]) * jnp.sin(y[indices2]))

        # Compute the third term (type 3): sum(gamma * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        term3 = gamma * jnp.sum(
            (y[indices1[valid_indices]] ** 2) * (y[indices2[valid_indices]] ** 4)
        )

        # Compute the fourth term (type 4): sum(delta * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        term4 = delta * jnp.sum(y[indices1[valid_indices]] * y[indices2[valid_indices]])

        return term1 + term2 + term3 + term4

    def y0(self):
        # Initial value is 2.0 for all variables
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)


# TODO: human review required
class DIXMAANC(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version C).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It includes all four term types with non-zero powers
    of (i/n) in the weights.

    The objective function includes quadratic terms, quartic terms,
    bilinear terms, and sin terms.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.

    Classification: OUR2-AN-V-0
    """

    n: int = 3000  # Default dimension

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters
        alpha = 1.0
        beta = 0.0625
        gamma = 0.0625
        delta = 0.0625

        # Powers for each group
        k1 = 1  # Power for group 1
        k2 = 1  # Power for group 2
        k3 = 1  # Power for group 3
        k4 = 1  # Power for group 4

        # Indices for each variable
        # i_vals not used directly
        # i_over_n not used directly

        # Compute the first term (type 1): sum(alpha * (i/n)^k1 * (x_i)^2)
        term1 = alpha * jnp.sum(((jnp.arange(1, n + 1) / n) ** k1) * (y**2))

        # Compute the 2nd term (type 2): sum(beta * (i/n)^k2 * sin(x_i) * sin(x_{i+1})
        # for i from 1 to n-1
        indices1 = jnp.arange(n - 1)
        indices2 = indices1 + 1
        term2 = beta * jnp.sum(
            ((indices1 + 1) / n) ** k2 * jnp.sin(y[indices1]) * jnp.sin(y[indices2])
        )

        # Compute the 3rd term (type 3): sum(gamma * (i/n)^k3 * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term3 = gamma * jnp.sum(
            ((valid_i1 + 1) / n) ** k3 * (y[valid_i1] ** 2) * (y[valid_i2] ** 4)
        )

        # Compute the fourth term (type 4): sum(delta * (i/n)^k4 * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term4 = delta * jnp.sum(((valid_i1 + 1) / n) ** k4 * y[valid_i1] * y[valid_i2])

        return term1 + term2 + term3 + term4

    def y0(self):
        # Initial value is 2.0 for all variables
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)


# TODO: human review required
class DIXMAAND(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version D).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It includes all four term types with higher powers
    of (i/n) in the weights.

    The objective function includes quadratic terms, quartic terms,
    bilinear terms, and sin terms.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.

    Classification: OUR2-AN-V-0
    """

    n: int = 3000  # Default dimension

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters
        alpha = 1.0
        beta = 0.0625
        gamma = 0.0625
        delta = 0.0625

        # Powers for each group
        k1 = 2  # Power for group 1
        k2 = 2  # Power for group 2
        k3 = 2  # Power for group 3
        k4 = 2  # Power for group 4

        # Indices for each variable
        # i_vals not used directly
        # i_over_n not used directly

        # Compute the first term (type 1): sum(alpha * (i/n)^k1 * (x_i)^2)
        term1 = alpha * jnp.sum(((jnp.arange(1, n + 1) / n) ** k1) * (y**2))

        # Compute the 2nd term (type 2): sum(beta * (i/n)^k2 * sin(x_i) * sin(x_{i+1})
        # for i from 1 to n-1
        indices1 = jnp.arange(n - 1)
        indices2 = indices1 + 1
        term2 = beta * jnp.sum(
            ((indices1 + 1) / n) ** k2 * jnp.sin(y[indices1]) * jnp.sin(y[indices2])
        )

        # Compute the 3rd term (type 3): sum(gamma * (i/n)^k3 * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term3 = gamma * jnp.sum(
            ((valid_i1 + 1) / n) ** k3 * (y[valid_i1] ** 2) * (y[valid_i2] ** 4)
        )

        # Compute the fourth term (type 4): sum(delta * (i/n)^k4 * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term4 = delta * jnp.sum(((valid_i1 + 1) / n) ** k4 * y[valid_i1] * y[valid_i2])

        return term1 + term2 + term3 + term4

    def y0(self):
        # Initial value is 2.0 for all variables
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)


# TODO: human review required
class DIXMAANI1(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version I1).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It's a variant of DIXMAANI with beta=0, so elements/groups
    of type 2 are removed.

    The objective function includes quadratic terms, quartic terms, and bilinear terms
    with higher powers of (i/n) in the weights.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.

    Classification: OUR2-AN-V-0
    """

    n: int = 3000  # Default dimension

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters
        alpha = 1.0
        # Beta is zero for DIXMAANI1
        gamma = 0.125
        delta = 0.125

        # Powers for each group
        k1 = 2  # Power for group 1
        # k2 not used since beta=0
        k3 = 0  # Power for group 3
        k4 = 2  # Power for group 4

        # Indices for each variable
        # i_vals not used directly
        # i_over_n not used directly

        # Compute the first term (type 1): sum(alpha * (i/n)^k1 * (x_i)^2)
        term1 = alpha * jnp.sum(((jnp.arange(1, n + 1) / n) ** k1) * (y**2))

        # Compute the third term (type 3): sum(gamma * (i/n)^k3 * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term3 = gamma * jnp.sum(
            ((valid_i1 + 1) / n) ** k3 * (y[valid_i1] ** 2) * (y[valid_i2] ** 4)
        )

        # Compute the fourth term (type 4): sum(delta * (i/n)^k4 * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term4 = delta * jnp.sum(((valid_i1 + 1) / n) ** k4 * y[valid_i1] * y[valid_i2])

        return term1 + term3 + term4

    def y0(self):
        # Initial value is 2.0 for all variables
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)


# TODO: human review required
class DIXMAANJ(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version J).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It includes quadratic, sin, quartic, and bilinear terms
    with higher powers of (i/n) in the weights.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.

    Classification: OUR2-AN-V-0
    """

    n: int = 3000  # Default dimension

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters
        alpha = 1.0
        beta = 0.0625
        gamma = 0.0625
        delta = 0.0625

        # Powers for each group
        k1 = 2  # Power for group 1
        k2 = 0  # Power for group 2
        k3 = 0  # Power for group 3
        k4 = 2  # Power for group 4

        # Indices for each variable
        # i_vals not used directly
        # i_over_n not used directly

        # Compute the first term (type 1): sum(alpha * (i/n)^k1 * (x_i)^2)
        term1 = alpha * jnp.sum(((jnp.arange(1, n + 1) / n) ** k1) * (y**2))

        # Compute the 2nd term (type 2): sum(beta * (i/n)^k2 * sin(x_i) * sin(x_{i+1})
        # for i from 1 to n-1
        indices1 = jnp.arange(n - 1)
        indices2 = indices1 + 1
        term2 = beta * jnp.sum(
            ((indices1 + 1) / n) ** k2 * jnp.sin(y[indices1]) * jnp.sin(y[indices2])
        )

        # Compute the third term (type 3): sum(gamma * (i/n)^k3 * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term3 = gamma * jnp.sum(
            ((valid_i1 + 1) / n) ** k3 * (y[valid_i1] ** 2) * (y[valid_i2] ** 4)
        )

        # Compute the fourth term (type 4): sum(delta * (i/n)^k4 * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term4 = delta * jnp.sum(((valid_i1 + 1) / n) ** k4 * y[valid_i1] * y[valid_i2])

        return term1 + term2 + term3 + term4

    def y0(self):
        # Initial value is 2.0 for all variables
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)


# TODO: human review required
class DIXMAANK(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version K).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It includes quadratic, sin, quartic, and bilinear terms
    with higher powers of (i/n) in the weights and increased parameter values.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.

    Classification: OUR2-AN-V-0
    """

    n: int = 3000  # Default dimension

    def __init__(self, n=None):
        if n is not None:
            self.n = n

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters
        alpha = 1.0
        beta = 0.125
        gamma = 0.125
        delta = 0.125

        # Powers for each group
        k1 = 2  # Power for group 1
        k2 = 0  # Power for group 2
        k3 = 0  # Power for group 3
        k4 = 2  # Power for group 4

        # Indices for each variable
        # i_vals not used directly
        # i_over_n not used directly

        # Compute the first term (type 1): sum(alpha * (i/n)^k1 * (x_i)^2)
        term1 = alpha * jnp.sum(((jnp.arange(1, n + 1) / n) ** k1) * (y**2))

        # Compute the 2nd term (type 2): sum(beta * (i/n)^k2 * sin(x_i) * sin(x_{i+1})
        # for i from 1 to n-1
        indices1 = jnp.arange(n - 1)
        indices2 = indices1 + 1
        term2 = beta * jnp.sum(
            ((indices1 + 1) / n) ** k2 * jnp.sin(y[indices1]) * jnp.sin(y[indices2])
        )

        # Compute the third term (type 3): sum(gamma * (i/n)^k3 * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term3 = gamma * jnp.sum(
            ((valid_i1 + 1) / n) ** k3 * (y[valid_i1] ** 2) * (y[valid_i2] ** 4)
        )

        # Compute the fourth term (type 4): sum(delta * (i/n)^k4 * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term4 = delta * jnp.sum(((valid_i1 + 1) / n) ** k4 * y[valid_i1] * y[valid_i2])

        return term1 + term2 + term3 + term4

    def y0(self):
        # Initial value is 2.0 for all variables
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)


# TODO: human review required
class DIXMAANL(AbstractUnconstrainedMinimisation):
    """Dixon-Maany test problem (version L).

    This is a variable-dimension unconstrained optimization problem from the
    Dixon-Maany family. It includes quadratic, sin, quartic, and bilinear terms
    with higher powers of (i/n) in the weights and significantly increased
    parameter values.

    Source:
    L.C.W. Dixon and Z. Maany,
    "A family of test problems with sparse Hessians for unconstrained optimization",
    TR 206, Numerical Optimization Centre, Hatfield Polytechnic, 1988.

    SIF input: Ph. Toint, Dec 1989.
    correction by Ph. Shott, January 1995.

    Classification: OUR2-AN-V-0
    """

    n: int = 3000  # Default dimension

    def objective(self, y, args):
        del args
        n = y.shape[0]
        m = n // 3

        # Problem parameters
        alpha = 1.0
        beta = 0.26
        gamma = 0.26
        delta = 0.26

        # Powers for each group
        k1 = 2  # Power for group 1
        k2 = 0  # Power for group 2
        k3 = 0  # Power for group 3
        k4 = 2  # Power for group 4

        # Indices for each variable
        # i_vals not used directly
        # i_over_n not used directly

        # Compute the first term (type 1): sum(alpha * (i/n)^k1 * (x_i)^2)
        term1 = alpha * jnp.sum(((jnp.arange(1, n + 1) / n) ** k1) * (y**2))

        # Compute the 2nd term (type 2): sum(beta * (i/n)^k2 * sin(x_i) * sin(x_{i+1})
        # for i from 1 to n-1
        indices1 = jnp.arange(n - 1)
        indices2 = indices1 + 1
        term2 = beta * jnp.sum(
            ((indices1 + 1) / n) ** k2 * jnp.sin(y[indices1]) * jnp.sin(y[indices2])
        )

        # Compute the third term (type 3): sum(gamma * (i/n)^k3 * (x_i)^2 * (x_{i+m})^4)
        # for i from 1 to 2m
        indices1 = jnp.arange(2 * m)
        indices2 = indices1 + m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term3 = gamma * jnp.sum(
            ((valid_i1 + 1) / n) ** k3 * (y[valid_i1] ** 2) * (y[valid_i2] ** 4)
        )

        # Compute the fourth term (type 4): sum(delta * (i/n)^k4 * x_i * x_{i+2m})
        # for i from 1 to m
        indices1 = jnp.arange(m)
        indices2 = indices1 + 2 * m
        # Ensure we don't go out of bounds
        valid_indices = indices2 < n
        valid_i1 = indices1[valid_indices]
        valid_i2 = indices2[valid_indices]
        term4 = delta * jnp.sum(((valid_i1 + 1) / n) ** k4 * y[valid_i1] * y[valid_i2])

        return term1 + term2 + term3 + term4

    def y0(self):
        # Initial value is 2.0 for all variables
        return jnp.full(self.n, 2.0)

    def args(self):
        return None

    def expected_result(self):
        # The minimum is at the origin
        return jnp.zeros(self.n)

    def expected_objective_value(self):
        # At the origin, all terms are zero
        return jnp.array(0.0)
