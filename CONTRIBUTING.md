# Contributing

Contributions (pull requests) are very welcome! Here's how to get started.

---

**Getting started**

First fork the library on GitHub.

Then clone and install the library in development mode:

```bash
git clone https://github.com/your-username-here/sif2jax.git
cd sif2jax
pip install -e .
```

Then install the pre-commit hook:

```bash
pip install pre-commit
pre-commit install
```

These hooks use ruff to format and lint the code, and pyright to make sure types align.

---

**If you're making changes to the code:**

Now make your changes. Make sure to include additional tests if necessary.

To run tests locally, you can either use the container available on DockerHub at johannahaffner/pycutest:latest or install pycutest locally, which will require an installation of CUTEst libraries written in Fortran. 
Reliable instructions for installation on Linux and Mac are available on the landing page of the pycutest repository on Github ([here](https://github.com/jfowkes/pycutest)).
You can also build a container from the Dockerfile and run the tests from in there.

The following instructions are for the first approach, which just requires loading the pre-existing container from Docker Hub. 
This happens inside `run_tests.sh`, which then creates an editable install of your current version of `sif2jax` and runs the tests on that.

```bash
chmod +x run_tests.sh
./run_tests.sh
```

Then push your changes back to your fork of the repository:

```bash
git push
```

Finally, open a pull request on GitHub!

---

**Running benchmarks:**

We use `pytest-benchmark` to measure the performance of sif2jax implementations against pycutest (Fortran). Benchmarks are located in the `benchmarks/` folder and are skipped by default during normal test runs. These benchmarks run on ALL problems and are intended for release documentation.

If you need runtime information for specific problems, use the runtime tests in the main test suite instead.

Since benchmarks require pycutest with Fortran libraries, use the provided script:

```bash
bash benchmarks/run_benchmarks.sh  # Run all benchmarks (sif2jax vs pycutest)

bash benchmarks/run_benchmarks.sh --benchmark-save=release_v1  # Save benchmark results
```

The script will run the benchmarks in a containerised environment in which pycutest and the Fortran backend are available.
If you have pycutest installed locally, you can also run them directly without the script.

To compare benchmark results:

```bash
# Compare runs by their ID numbers
pytest-benchmark compare 0001 0002

# Compare all saved runs
pytest-benchmark compare
```
---

**If you're making changes to the documentation:**

Make your changes. You can then build the documentation by doing

```bash
pip install -e '.[docs]'
mkdocs build
mkdocs serve
```

You can then see your local copy of the documentation by navigating to `localhost:8000` in a web browser.

**A word on using generative AI to port benchmark problems** 

We've used Claude Code extensively to make it possible to port this large collection of benchmark problems.
The one thing that has worked for us is to use test-driven development throughout. So if you'd like to add a feature, we recommend thinking about the tests this feature should pass for you to be convinced that it has been implemented correctly, and then verifying this test yourself before "commissioning" any work. 
Output improves with the granularity of the tests provided, so it is preferable to define a number of very small tests, rather than one large tests. Tests should ideally provide a natural ordering of increasing complexity (e.g. verify the initial point is correct before verifying the objective at the initial point).
Think of this as automating the feedback-giving portion of human supervision to the greatest extent possible.
Then ask the model to work in batches of problems, testing and fixing them one by one works best.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License Agreement (CLA). 
You (or your employer) retain the copyright to your contribution; this simply gives us permission to use and redistribute your contributions as part of the project. Head over to <https://cla.developers.google.com/> 
to see your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one (even if it was for a different project), you probably don't need to do it again.