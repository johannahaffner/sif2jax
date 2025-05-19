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

To run tests locally, you can either use the container available on DockerHub at
johannahaffner/pycutest:latest or install pycutest locally, which will require an
installation of CUTEst libraries written in Fortran. Reliable instructions for 
installation on Linux and Mac are available on the landing page of the pycutest
repository on Github ([here](https://github.com/jfowkes/pycutest)).
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

**If you're making changes to the documentation:**

Make your changes. You can then build the documentation by doing

```bash
pip install -e '.[docs]'
mkdocs build
mkdocs serve
```

You can then see your local copy of the documentation by navigating to `localhost:8000` in a web browser.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement (CLA). You (or your employer) retain the copyright to your
contribution; this simply gives us permission to use and redistribute your
contributions as part of the project. Head over to <https://cla.developers.google.com/> 
to see your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.