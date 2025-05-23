import inspect
import os
import re
import subprocess
import tempfile
# from collections.abc import Callable

import ollama  # pyright: ignore
from ollama._types import GenerateResponse
import pycutest
from typing import Tuple, List, Callable, Optional
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxtyping import ArrayLike, Scalar

base_prompt = """
We are porting the cutest optimization benchmark into Python/JAX. This consists of
a series of "SIF" files, which are a variant of XML files. These SIF files contain
all information defining the optimization problem. There exists a Fortran decoder 
for these files that is accessible from a Python library "pycutest". This implementation
is limited for us as it only provides black box C callables for the objective function,
constraint functions, jacobians, hessians and so on. This makes it fundamentally
incompatible with our JAX approach that wants to be able to use autodiff and
vmap on these problems. Therefore we are asking you to perform this translation
to Python/JAX for us. 
"""

# prompt for asking for the objective
objective_prompt = """
You will now be provided with a SIF file for a single problem. Your task is to 
provide us an objective function written in Python/JAX which matches that of the
decoded one from pycutest, which is parsing the same SIF file. Here is an example
of a desired output for a trivial quadratic cost function:

def objective(y: jnp.array) -> jnp.array:
    return y[0]**2 + y[1]**2

Make sure that after you are done thinking that you only output this function with the 
correct function signature! Furthermore ensure that the objective has only one input
argument "y" as shown above, and assume that any constants are available in the 
scope.
"""

# helper for segmenting up deepseek thinking and output
THINK_RE = re.compile(
    r"<think\b[^>]*>(.*?)</think\s*>",  # capture tag body, ignore attributes
    flags=re.IGNORECASE | re.DOTALL,    # case‑insensitive, match new‑lines
)

def split_thought_and_output(raw: str) -> Tuple[str, str]:
    # Collect every <think>…</think>
    thoughts = [m.strip() for m in THINK_RE.findall(raw)]
    thinking = "\n\n".join(thoughts)

    # Strip the thinking sections from the message
    output = THINK_RE.sub("", raw).strip()

    return thinking, output

def _indent_len(s: str) -> int:
    """Count leading spaces or tabs (tabs count as 1 for our purpose)."""
    return len(s) - len(s.lstrip(" \t"))

def extract_objective_def(code: str, problem_name: Optional[str] = None) -> str:

    # Build a single regex covering both prefixes, ignore case
    prefixes = ["objective"]
    if problem_name:
        prefixes.append(re.escape(problem_name))   # escape just in case
    prefix_pat = "|".join(prefixes)

    func_re = re.compile(
        rf"^(\s*)def\s+(?:{prefix_pat})\w*\s*\(",   # def <prefix…>(
        flags=re.IGNORECASE                         # ← case‑insensitive
    )

    lines: List[str] = code.splitlines()

    for i, line in enumerate(lines):
        m = func_re.match(line)
        if m:
            base_indent = _indent_len(line)
            body: List[str] = [line]

            # collect indented body
            for nxt in lines[i + 1 :]:
                if nxt.strip() == "":                    # keep blank lines
                    body.append(nxt)
                    continue
                if _indent_len(nxt) > base_indent:       # still inside body
                    body.append(nxt)
                else:
                    break                                # dedented ⇒ done
            return "\n".join(body)

    raise ValueError("No objective‑like function found.")

def load_function(src: str, fn_name: str, extra_globals: dict | None = None):
    """Compile `src` and return the callable named `fn_name`."""
    # A fresh *module‑like* namespace to keep things contained
    ns: dict = {}
    if extra_globals:
        ns.update(extra_globals)         # make jnp, np, etc. visible inside exec
    exec(src, ns)                        # compile & run the code string
    return ns[fn_name]                   # pull out the requested symbol

def test_objective(objective: Callable[[ArrayLike], Scalar], problem_iD: str):
    pycutest_problem = pycutest.import_problem(problem_iD)
    y0 = jnp.asarray(pycutest_problem.x0)

    pycutest_f0, pycutest_grad0 = pycutest_problem.obj(y0, gradient=True)
    f0, grad0 = jax.value_and_grad(objective)(y0)

    pycutest_hess0 = pycutest_problem.hess(y0)
    hess0 = jax.hessian(objective)(y0)

    assert np.allclose(f0, pycutest_f0), f"Objective mismatch for {problem_iD}"
    assert np.allclose(grad0, pycutest_grad0), f"Gradient mismatch for {problem_iD}"
    assert np.allclose(hess0, pycutest_hess0), f"Hessian mismatch for {problem_iD}"

    grad0_signs = np.sign(grad0)
    pycutest_grad0_signs = np.sign(pycutest_grad0)
    gradient_signs_match = np.all(grad0_signs == pycutest_grad0_signs)
    assert gradient_signs_match, f"Gradient sign mismatch for {problem_iD}"

    hess0_signs = np.sign(hess0)
    pycutest_hess0_signs = np.sign(pycutest_hess0)
    hessian_signs_match = np.all(hess0_signs == pycutest_hess0_signs)
    assert hessian_signs_match, f"Hessian sign mismatch for {problem_iD}"

    return True

def init(sif_file: str, model: str) -> Tuple[str, str]:

    prompt = base_prompt \
        + "\n\n\n" \
        + objective_prompt \
        + "Here is the SIF file: \n" \
        + sif_file
    
    candidate_response = ollama.generate(
        model=model,
        prompt=prompt
    )["response"]

    return prompt, candidate_response

def inner_loop(prompt: str, candidate_response: str, model: str, name: str) -> Tuple[Callable, str, str]:

    thoughts, candidate_output_raw = split_thought_and_output(candidate_response)
    concat_prompt = prompt + "\n" + candidate_output_raw # IGNORING THOUGHTS IN THE CONCATENATED PROMPT

    while True:
        try:
            print("attempting to load function...")
            candidate_objective_str = extract_objective_def(candidate_output_raw, name) # extract_python_blocks(first_output)[0]
            candidate_objective = load_function(
                candidate_objective_str,
                "objective",
                extra_globals={"jnp": jnp},          # expose jnp so the body finds it
            )
            candidate_objective(y0) # test an inference of the function
            print(f"successfully loaded function: \n{candidate_objective_str}")
            break

        except Exception as err:
            print(f"failed to load provided function, refining...")
            print(f"failed function was: \n{candidate_objective_str}")
            concat_prompt = concat_prompt + "\n" + """
            Your provided output was not a valid Python function. 

            You are to only provide a valid Python callable function after your thinking period
            with the signature "objective". Furthermore ensure that the objective has only one input
            argument "y" as shown previously.

            This is the error recieved when attempting to parse and then run your provided output:
            """ + str(err)

            new_candidate = ollama.generate(
                model=model,
                prompt=concat_prompt
            )

            thoughts, candidate_output_raw = split_thought_and_output(new_candidate["response"])

    return candidate_objective, candidate_objective_str, concat_prompt

def outer_loop(prompt: str, candidate_response: str, model: str, name: str) -> Callable:

    # test if the init function is valid, if so iteratively correct it
    candidate_objective, candidate_objective_str, concat_prompt = inner_loop(prompt, candidate_response, model, name)

    while True:
        try:
            print("testing candidate objective correctness...")
            tests_passed = test_objective(candidate_objective, problem_iD)
            print("candidate objective is validated, breaking")
            objective = candidate_objective
            break
        except Exception as err:
            print("candidate objective failed tests, refining...")
            print(f"failed function was: \n {candidate_objective_str}")
            concat_prompt += "\n" + f"""
            Your provided objective function was incorrect, although it was a valid
            Python function. The error was as follows: 

            {err}

            I will now provide more information on the results of the tests to help 
            you interate.
            """

            pycutest_problem = pycutest.import_problem(problem_iD)
            y0 = jnp.asarray(pycutest_problem.x0)

            pycutest_f0, pycutest_grad0 = pycutest_problem.obj(y0, gradient=True)
            f0, grad0 = jax.value_and_grad(candidate_objective)(y0)

            pycutest_hess0 = pycutest_problem.ihess(y0)
            hess0 = jax.hessian(candidate_objective)(y0)

            objective_diff = f"f0 - pycutest_f0: \n{f0 - pycutest_f0}"
            gradient_diff = f"grad0 - pycutest_grad0: \n{grad0 - pycutest_grad0}"
            hessian_diff = f"hess0 - pycutest_hess0: \n{hess0 - pycutest_hess0}"

            full_match  = lambda x, y: jnp.isclose(x, y, rtol=1e-05, atol=1e-08).astype(jnp.int32)
            objective_match = f"matching objective: \n{full_match(f0, pycutest_f0)}"
            gradient_match = f"matching jacobian elements: \n{full_match(grad0, pycutest_grad0)}"
            hessian_match = f"matching hessian elements: \n{full_match(hess0, pycutest_hess0)}"

            sign_match = lambda x, y: (jnp.sign(x) == jnp.sign(y)).astype(jnp.int32)
            objective_sign_match = f"matching objective sign: \n{sign_match(f0, pycutest_f0)}"
            gradient_sign_match = f"matching jacobian element signs: \n{sign_match(grad0, pycutest_grad0)}"
            hessian_sign_match = f"matching hessian element signs: \n{sign_match(hess0, pycutest_hess0)}"

            abs_value_match = lambda x, y: jnp.isclose(jnp.abs(x), jnp.abs(y), rtol=1e-05, atol=1e-08).astype(jnp.int32)
            objective_abs_match = f"matching objective absolute value: \n{abs_value_match(f0, pycutest_f0)}"
            gradient_abs_match = f"matching jacobian elements absolute values: \n{abs_value_match(grad0, pycutest_grad0)}"
            hessian_abs_match = f"matching hessian elements absolute values: \n{abs_value_match(hess0, pycutest_hess0)}"

            concat_prompt += "\n" + f"""
            The difference from pycutest in the values for the objective, its jacobian, and its hessian are:

            {objective_diff}
            {gradient_diff}
            {hessian_diff}

            The elements that match the pycutest for the objective, jacobian, and hessian are showwn below (1's for a match, 0's for a mismatch)

            {objective_match}
            {gradient_match}
            {hessian_match}

            The elements that match the sign of the pycutest in the objective, jacobian, and hessian are shown below (1's for a match, 0's for a mismatch)

            {objective_sign_match}
            {gradient_sign_match}
            {hessian_sign_match}

            The elements that have the same absolute values as those of the pycutest in the objective, jacobian, and hessian are shown below (1's for a match, 0's for a mismatch)

            {objective_abs_match}
            {gradient_abs_match}
            {hessian_abs_match}
            """

            new_candidate = ollama.generate(
                model=model,
                prompt=concat_prompt
            )

            candidate_objective, candidate_objective_str, concat_prompt = inner_loop(concat_prompt, new_candidate["response"], model, name)

    return objective

if __name__ == "__main__":

    known_correct_translations = [
        "AKIVA", 
        "BIGGS6", 
        "CHWIRUT1LS",
        "CHWIRUT2LS",
        "CLIFF",
        "CLUSTERLS",
        "COSINE",
        "CURLY10",
        "CURLY20",
        "CURLY30",
        "DANIWOODLS",
        "DQRTIC",
        "EGGCRATE",
        "ELATVIDU",
        "GENHUMPS",
        "GROWTHLS",
        "HAIRY",
        "HIMMELBG",
        "HIMMELBH",
        "HUMPS",
        "JENSMP",
        "KIRBY2LS",
        "KOWOSB",
        "LSC1LS",
        "LSC2LS",
        "ROSENBR"
    ]

    model = "deepseek-r1:70b"
    # problem_iD = input("Enter a SIF iD: ")
    problem_iD = "BIGGS6"
    pycutest_problem = pycutest.import_problem(problem_iD)
    y0 = pycutest_problem.x0
    sif_path = str(os.getenv('MASTSIF')) + problem_iD + ".SIF"
    with open(sif_path, "r", encoding="utf-8") as file:
        sif_file = file.read()
    print(sif_file)

    # test inner loop
    # candidate_objective = inner_loop(sif_file, model)
    prompt, candidate_response = init(sif_file, model)
    objective = outer_loop(prompt, candidate_response, model, problem_iD)



    print("fin")
