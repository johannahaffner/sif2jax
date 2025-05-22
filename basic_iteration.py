import inspect
import os
import re
import subprocess
import tempfile

import ollama  # pyright: ignore
import pycutest


system_prompt = """
You are a helpful assistant, very conscientious, careful and precise. You value clean
and readable code and take care to factorise it well. You are a little bit of a stickler
for PEP8 and like to use type hints.
You have expertise working in the JAX ecosystem and are familiar with Equinox, and you
are passionate about the speed and ease of use of JAX. You are helping us benchmark
solvers on nonlinear optimisation problems from the CUTEst library - we can only do that
if we get precise and correct Python implementations of the problems in the SIF format.
If we succeed, then we can use our solvers on a much larger class of problems and 
blend data-driven methods with traditional optimisation, for instance in robotics. That
is exciting, isn't it? You're glad to be on board.

You are given a SIF file, which contains instructions to generate Fortran code. You are
an expert in parsing these and carefully examine it, before outputting a Python class 
definition that implements the problem in the SIF file. It is *very important* that you 
return only the content of the Python file, no other text. 

A few other criteria:
- The problem should implement the abstract  base class 
    `AbstractUnconstrainedMinimisation` from a file you get as input. Implementing an
    abstract base class means that you need to write a concrete implementation for all
    methods declared as abstract in the base class. There should be no abstract methods 
    left in the class you write.
- You do not need to re-implement methods that are not abstract in the base class.
- You do not add any methods to the class.
- You only return the implemented problem class, please don't add functions for parsing
    SIF files.
- The name of the class should be the same as the name of the problem you are given in 
    the SIF file. You should write it in all caps for consistency with the SIF file.
- All the code you write should be JAX-transformable - please only use JAX-compatible
    control flow and data structures.
- You should use the `jax.numpy` module for all numerical operations.
- You should use the `jax.vmap` function to vectorise your code where appropriate, in
    particular to avoid for loops that only have index dependence.
- Under no circumstances can you make changes to the abstract base classes. What you
    need to fix is the code you generated, not the code you are provided.

You will get feedback on your implementation, for example from a ruff check of the code
you wrote. You implement this feedback diligently and return an updated implementation.
"""


def run_ruff_check(generated_code: str) -> str:
    def maybe_strip(s: str) -> str:
        match = re.search(r"```(?:\w*\n)?(.*?)```", s, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return ""  # No code in output  # TODO handle this error

    generated_code = maybe_strip(generated_code)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
        temp_file.write(generated_code)
        temp_file_path = temp_file.name
    try:
        result = subprocess.run(
            ["ruff", "check", temp_file_path], capture_output=True, text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Ruff check error: {e}"
    finally:
        os.unlink(temp_file_path)


if __name__ == "__main__":
    # TODO: more user-friendly input here: Needs to be an exact ID match
    problem_iD = input("Enter a SIF iD: ")

    # Assume problem is in top-level directory -> TODO better search
    sif_path = "./archive/mastsif/" + problem_iD + ".SIF"
    with open(sif_path, encoding="utf-8") as file:
        sif_file = file.read()

    base_path = "./sif2jax/_problem.py"
    with open(base_path, encoding="utf-8") as file:
        base = file.read()

    first_draft = ollama.generate(
        model="llama3.1",
        prompt="Write a python function to solve the problem in the SIF file: \n "
        + sif_file
        + "\n\n"
        + "The function should be a class that implements the abstract base class "
        + "AbstractUnconstrainedMinimisation defined here: \n"
        + base,
        system=system_prompt,
    )
    print(first_draft.response)
    print(inspect.signature(ollama.generate))

    ruff_feedback = run_ruff_check(first_draft.response)
    print(ruff_feedback)

    second_draft = ollama.generate(
        model="llama3.1",
        prompt="Please fix the code you wrote according to the feedback from ruff: \n"
        + ruff_feedback
        + "\n\n"
        + base,
        system=system_prompt,
    )
    print(second_draft.response)

    # Now verify against pycutest
    pycutest_problem = pycutest.import_problem(problem_iD)
