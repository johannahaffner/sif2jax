import pycutest as pc  # pyright: ignore[reportMissingImports]


if __name__ == "__main__":
    # Avoid caching real-world problems and modelling problems, some of them are large
    # ...while that makes them great candidates for caching, it can also break the build
    # of the container image.
    problems = pc.find_problems(n=[1, 10], origin="academic")
    print(problems)
    for name in problems:
        problem = pc.import_problem(name)
        print(problem.x0, problem.n)  # Do something to trigger caching
