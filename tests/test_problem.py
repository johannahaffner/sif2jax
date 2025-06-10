import jax
import jax.flatten_util as jfu
import jax.numpy as jnp
import numpy as np
import pycutest  # pyright: ignore[reportMissingImports]  - test runs in container
import pytest  # pyright: ignore[reportMissingImports]  - test runs in container
import sif2jax


def get_test_cases(requested):
    if requested is not None:
        # Split by comma and strip whitespace
        test_case_names = [name.strip() for name in requested.split(",")]
        test_cases = []

        for name in test_case_names:
            try:
                test_case = sif2jax.cutest.get_problem(name)
                assert (
                    test_case is not None
                ), f"Test case '{name}' not found in sif2jax.cutest problems."
                test_cases.append(test_case)
            except Exception as e:
                raise RuntimeError(
                    f"Test case '{name}' not found in sif2jax.cutest problems."
                ) from e
        return tuple(test_cases)
    else:
        return sif2jax.problems


def pytest_generate_tests(metafunc):
    if "problem" in metafunc.fixturenames:
        requested = metafunc.config.getoption("--test-case")
        test_cases = get_test_cases(requested)
        metafunc.parametrize("problem", test_cases, scope="class")


class TestProblem:
    @pytest.fixture(scope="class")
    def pycutest_problem(self, problem):
        """Load pycutest problem once per problem per class."""
        return pycutest.import_problem(problem.name())

    def test_correct_name(self, pycutest_problem):
        assert pycutest_problem is not None

    def test_correct_dimension(self, problem, pycutest_problem):
        dimensions = problem.y0().size
        assert dimensions == pycutest_problem.n

    def test_correct_starting_value(self, problem, pycutest_problem):
        assert np.allclose(pycutest_problem.x0, problem.y0())

    def test_correct_objective_at_start(self, problem, pycutest_problem):
        pycutest_value = pycutest_problem.obj(pycutest_problem.x0)
        sif2jax_value = problem.objective(problem.y0(), problem.args())
        assert np.allclose(pycutest_value, sif2jax_value)

    def test_correct_gradient_at_start(self, problem, pycutest_problem):
        pycutest_gradient = pycutest_problem.grad(pycutest_problem.x0)
        sif2jax_gradient = jax.grad(problem.objective)(problem.y0(), problem.args())
        assert np.allclose(pycutest_gradient, sif2jax_gradient)

    def test_correct_hessian_at_start(self, problem, pycutest_problem):
        if problem.num_variables() < 1000:
            pycutest_hessian = pycutest_problem.ihess(pycutest_problem.x0)
            sif2jax_hessian = jax.hessian(problem.objective)(
                problem.y0(), problem.args()
            )
            assert np.allclose(pycutest_hessian, sif2jax_hessian)
        else:
            pytest.skip("Skip Hessian test for large problems to save time and memory")

    def test_correct_constraint_dimensions(self, problem, pycutest_problem):
        num_equalities, num_inequalities, _ = problem.num_constraints()

        if pycutest_problem.m == 0:
            assert num_equalities == 0
            assert num_inequalities == 0
        else:
            pycutest_constraints = pycutest_problem.cons(pycutest_problem.x0)
            assert pycutest_constraints is not None

            pycutest_equalities = pycutest_constraints[pycutest_problem.is_eq_cons]  # pyright: ignore
            pycutest_equalities = jnp.array(pycutest_equalities).squeeze()
            assert pycutest_equalities.size == num_equalities

            pycutest_inequalities = pycutest_constraints[~pycutest_problem.is_eq_cons]  # pyright: ignore
            pycutest_inequalities = jnp.array(pycutest_inequalities).squeeze()
            assert pycutest_inequalities.size == num_inequalities

    def test_correct_number_of_finite_bounds(self, problem, pycutest_problem):
        _, _, num_finite_bounds = problem.num_constraints()

        # Pycutest defaults unconstrained variables to -1e20 and 1e20
        pycutest_finite_lower = jnp.sum(jnp.asarray(pycutest_problem.bl > -1e20))
        pycutest_finite_upper = jnp.sum(jnp.asarray(pycutest_problem.bu < 1e20))

        assert num_finite_bounds == pycutest_finite_lower + pycutest_finite_upper

    def test_correct_constraints_at_start(self, problem, pycutest_problem):
        if isinstance(problem, sif2jax.AbstractConstrainedMinimisation):
            assert pycutest_problem.m > 0, "Problem should have constraints."

            pycutest_constraints = pycutest_problem.cons(pycutest_problem.x0)
            pycutest_equalities = pycutest_constraints[pycutest_problem.is_eq_cons]  # pyright: ignore
            pycutest_equalities = jnp.array(pycutest_equalities).squeeze()
            pycutest_inequalities = pycutest_constraints[~pycutest_problem.is_eq_cons]  # pyright: ignore
            pycutest_inequalities = jnp.array(pycutest_inequalities).squeeze()

            sif2jax_constraints = problem.constraint(problem.y0())
            sif2jax_equalities, sif2jax_inequalities = sif2jax_constraints
            sif2jax_equalities, _ = jfu.ravel_pytree(sif2jax_equalities)
            sif2jax_inequalities, _ = jfu.ravel_pytree(sif2jax_inequalities)

            # Check that the constraints match
            assert np.allclose(pycutest_equalities, sif2jax_equalities)
            assert np.allclose(pycutest_inequalities, sif2jax_inequalities)
        else:
            pytest.skip("Problem has no constraints")

    @pytest.mark.skip(
        reason="Seems to be a likely culprint in memory failure in CI. FIX"
    )
    def test_compilation(self, problem):
        try:
            compiled = jax.jit(problem.objective)
            _ = compiled(problem.y0(), problem.args())
        except Exception as e:
            raise RuntimeError(f"Compilation failed for {problem.name()}") from e
        jax.clear_caches()

    def test_vmap(self, problem):
        try:
            vmapped = jax.vmap(problem.objective, in_axes=(0, None))
            y0 = problem.y0()
            _ = vmapped(jnp.array([y0, y0, y0]), problem.args())
        except Exception as e:
            raise RuntimeError(f"Vmap failed for {problem.name()}") from e
