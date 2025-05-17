import pycutest


# Dummy test to get started
def test_rosenbrock():
    problem = pycutest.import_problem("ROSENBR")
    assert problem is not None
