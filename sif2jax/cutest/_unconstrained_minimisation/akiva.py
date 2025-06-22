import jax.numpy as jnp

from ..._problem import AbstractUnconstrainedMinimisation


# TODO: needs human review
class AKIVA(AbstractUnconstrainedMinimisation):
    """The AKIVA function.

    Find the set of elemental and structural coefficients of a hierarchical
    logit model that maximizes the likelihood of a particular sample of
    observations.

    Source: A simple example of binary logit
    Ben-Akiva and Lerman, "Discrete Choice Analysis", MIT Press, 1985.

    SIF input: automatically produced by HieLoW
             Hierarchical Logit for Windows, written by Michel Bierlaire
             Mon Jan 30 12:12:18 1995
             SAVEs removed December 3rd 2014

    Classification: OUR2-AN-2-0

    Note: This implementation is based on the AKIVA SIF file, but the objective
    function has been reversed from a maximization to a minimization problem
    to match the framework's requirements.
    """

    y0_iD: int = 0
    provided_y0s: frozenset = frozenset({0})

    def objective(self, y, args):
        del args
        (
            beta,
            cte_auto,
        ) = y  # beta is the transit coefficient, cte_auto is the auto constant

        # Data from the SIF file (21 observations with 3 characteristics each)
        # Each row: [auto_value, transit_value, constant]
        data = jnp.array(
            [
                [52.9, 4.4, 1.0],  # Observation 1 - chose Transit (1)
                [4.1, 28.5, 1.0],  # Observation 2 - chose Transit (1)
                [4.1, 86.9, 1.0],  # Observation 3 - chose Auto (2)
                [56.2, 31.6, 1.0],  # Observation 4 - chose Transit (1)
                [51.8, 20.2, 1.0],  # Observation 5 - chose Transit (1)
                [0.2, 91.2, 1.0],  # Observation 6 - chose Auto (2)
                [27.6, 79.7, 1.0],  # Observation 7 - chose Auto (2)
                [89.9, 2.2, 1.0],  # Observation 8 - chose Transit (1)
                [41.5, 24.5, 1.0],  # Observation 9 - chose Transit (1)
                [95.0, 43.5, 1.0],  # Observation 10 - chose Transit (1)
                [99.1, 8.4, 1.0],  # Observation 11 - chose Transit (1)
                [18.5, 84.0, 1.0],  # Observation 12 - chose Auto (2)
                [82.0, 38.0, 1.0],  # Observation 13 - chose Auto (2)
                [8.6, 1.6, 1.0],  # Observation 14 - chose Transit (1)
                [22.5, 74.1, 1.0],  # Observation 15 - chose Auto (2)
                [51.4, 83.8, 1.0],  # Observation 16 - chose Auto (2)
                [81.0, 19.2, 1.0],  # Observation 17 - chose Transit (1)
                [51.0, 85.0, 1.0],  # Observation 18 - chose Auto (2)
                [62.2, 90.1, 1.0],  # Observation 19 - chose Auto (2)
                [95.1, 22.2, 1.0],  # Observation 20 - chose Transit (1)
                [41.6, 91.5, 1.0],  # Observation 21 - chose Auto (2)
            ]
        )

        # Choices: 1 = Transit, 2 = Auto
        choices = jnp.array(
            [1, 1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2]
        )

        # Vectorized computation of utilities for all observations
        utility_transit = beta * data[:, 1]  # beta * transit_value
        utility_auto = (
            beta * data[:, 0] + cte_auto * data[:, 2]
        )  # beta * auto_value + cte_auto * constant

        # Compute exponentials of utilities
        exp_utility_transit = jnp.exp(utility_transit)
        exp_utility_auto = jnp.exp(utility_auto)

        # Compute probabilities for both alternatives
        prob_transit = exp_utility_transit / (exp_utility_transit + exp_utility_auto)
        prob_auto = exp_utility_auto / (exp_utility_transit + exp_utility_auto)

        # Select probability based on choice using jnp.where
        # choices == 1 means transit was chosen, choices == 2 means auto was chosen
        chosen_probs = jnp.where(choices == 1, prob_transit, prob_auto)

        # Compute log likelihood
        log_likelihood = jnp.sum(jnp.log(chosen_probs))

        # Return negative log likelihood for minimization
        return jnp.array(-log_likelihood)

    def y0(self):
        # Initial values from SIF file (both start at 0.0)
        return jnp.array([0.0, 0.0])

    def args(self):
        return None

    def expected_result(self):
        # The optimal values are not explicitly given in the SIF file
        return None

    def expected_objective_value(self):
        # The SIF file has a commented out line: "*LO SOLUTION 6.1660422124"
        # Since we're minimizing negative log likelihood, we negate this value
        # return jnp.array(-6.1660422124)
        return None
