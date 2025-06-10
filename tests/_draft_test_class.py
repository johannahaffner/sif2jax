import jax
import numpy as np
import pycutest
import pytest


def pytest_generate_tests(metafunc):
    if "problem" in metafunc.fixturenames:
        # requested = metafunc.config.getoption("--test-case")
        test_cases = []  # TODO: implement get_test_cases function
        metafunc.parametrize("problem", test_cases, scope="class")


class TestProblem:
    @pytest.fixture(scope="class", autouse=True)
    def setup_pycutest_problem(self, problem):
        """Load pycutest problem once per problem per class."""
        self.pycutest_problem = pycutest.import_problem(problem.name())
        self.sif_problem = problem

    def test_correct_name(self):
        assert self.pycutest_problem is not None

    def test_correct_dimension(self):
        dimensions = self.sif_problem.y0().size
        assert dimensions == self.pycutest_problem.n

    def test_correct_starting_value(self):
        assert np.allclose(self.pycutest_problem.x0, self.sif_problem.y0())

    def test_correct_objective_at_start(self):
        pycutest_value = self.pycutest_problem.obj(self.pycutest_problem.x0)
        sif2jax_value = self.sif_problem.objective(
            self.sif_problem.y0(), self.sif_problem.args()
        )
        assert np.allclose(pycutest_value, sif2jax_value)

    def test_correct_gradient_at_start(self):
        pycutest_gradient = self.pycutest_problem.grad(self.pycutest_problem.x0)
        sif2jax_gradient = jax.grad(self.sif_problem.objective)(
            self.sif_problem.y0(), self.sif_problem.args()
        )
        assert np.allclose(pycutest_gradient, sif2jax_gradient)

    def test_correct_hessian_at_start(self):
        if self.sif_problem.num_variables() < 1000:
            pycutest_hessian = self.pycutest_problem.ihess(self.pycutest_problem.x0)
            sif2jax_hessian = jax.hessian(self.sif_problem.objective)(
                self.sif_problem.y0(), self.sif_problem.args()
            )
            assert np.allclose(pycutest_hessian, sif2jax_hessian)
        else:
            pytest.skip("Skipping Hessian test for large problems")

    def test_compilation(self):
        try:
            compiled = jax.jit(self.sif_problem.objective)
            _ = compiled(self.sif_problem.y0(), self.sif_problem.args())
        except Exception as e:
            raise RuntimeError(
                f"Compilation failed for {self.sif_problem.name()}"
            ) from e
        jax.clear_caches()

    # ... rest of your tests
