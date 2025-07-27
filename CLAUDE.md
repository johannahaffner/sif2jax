# System prompt for Claude

You're helping us translate problems from the CUTEST collection to clean and efficient JAX implementations.
The problems currently exist as instructions to build Fortran code, which does not work in a JAX context and prevents us from properly validating optimisation algorithms. 
This is a crucial bottleneck that needs fixing!

Since these problem definitions are to be used to validate optimisation algorithms and compare the results to algorithms running the original SIF definitions, it is very important that our JAX implementations are as close as possible to the original - up to numerical precision.

## ⚠️ CRITICAL: Testing Requirements

**ALWAYS use the provided bash script for testing.** Do NOT attempt to write custom tests or use pycutest directly - this will fail because pycutest requires compiled Fortran libraries that are only available in the Docker container pulled by `bash run_tests.sh`.

The testing workflow is:
1. Make any change to a file
2. Run `bash run_tests.sh --test-case "PROBLEM"`. Use `sudo bash ...` if you run into permission issues.
3. Fix issues based on test results
4. Repeat until tests pass

Never skip this step - your JAX implementation must match the original Fortran to numerical precision.

## Tipps and Tricks

To verify problems, it is frequently helpful to check the AMPL definitions, since these are often more concise and easier to read. You can find them here: https://github.com/ampl/global-optimization/tree/master/cute - files will have all lowercase names, and a suffix .mod.
Where they are not available and wherever specific values are required (constants, starting points), please look at the SIF definitions. 
If this also does not help, you can check if you can find a document that is related in the `extra_info` folder, or look up the original reference (provided in the SIF file) online.
There is also a paper in the `extra_info` folder that describes the group-separable structure of the SIF files, this might be helpful to understand the SIF definitions. You can find screenshots of the relevant sections in `extra_info/s2mpj/`.

## Porting problems

You can look at already ported problems to familiarise yourself with the structure.
When porting problems, please consider the following: 

1. Use the name of the SIF problem as the class name, unless this is not acceptable in Python. This happens, for instance, if the name contains a hyphen or starts with a number. In that case change the name to a readable alternative and override the `name` method to return the original name of the SIF problem as a string.
2. Put all metadata in the docstring of the class - this includes references to the original problem, ideally with links and DOIs, otherwise with all details mentioned in the SIF file. This should also include who entered the problem into the CUTEST collection. 
3. Add the classification number at the bottom of the docstring.
4. Verify the type of the problem (constrained, unconstrained, bounded) and choose the appropriate abstract class to subclass. 
5. Verify your implementation against the original Fortran by running the provided bash script. Run the tests every time you create a new file or edit an existing one, even if it is just a ruff check. (More below.)
6. Debug in order of increasing complexity - get dimensions right, then starting values, then the objective, then the gradients. 
7. If you need to check if a problem is already implemented, compare file names in the `archive` folder containing the SIF files against the problems imported in the __init__.py files of the `sif2jax` folder.
8. **Work continuously without stopping for summaries.** Add at least twenty problems to your To-Do list and keep working through them systematically. Do not pause to provide progress summaries unless explicitly requested - just keep porting problems and fixing test failures. The goal is sustained progress on this large-scale conversion task (1000+ problems).
9. Run ruff format and ruff check on your work. Ruff is installed in the working directory.
10. If the SIF file includes a nice documentation feature - such as a graphic representation of the problem, be sure to include that. Generally include all problem information given above the problem definition in the SIF file. 
11. Classification numbers should match the numbers given in the SIF file. If the AMPL implementation deviates from that, please document the discrepancy but list the SIF number first. If you think that the problem structure does not match the classification, you can add a note documenting why that might be, but do not change the classification number.
12. Do not ever hard-code a data type.
13. Each problem should get its own file, avoid putting several problems into the same file. If you want to factor out common functionality, then prefer importing the base class into the file in which the concrete problem is implemented.
14. Declare all dataclass fields - we're inheriting from Equinox.Module through the abstract base class `Problem`. 

## Testing

You can check any work against the original Fortran implementations. For this purpose, please run `bash run_tests.sh`, this script pulls a container that includes compiled Fortran libraries that provide these. 
Without this container, you cannot run any tests. 
Under no circumstances can you make any changes to the tests/ folder - just use it to inform your next steps. Never make edits to run_tests.sh.

When you don't know what to do, find the next problem to work on by examining the missing_problems.md for problems that are not ticked off ([] SOMEPROBLEM) and checking `sif2jax/cutest/__init__.py` to see if the prblem is already being imported.
When you make any change to a file, please run the tests again. You can run problem-specific tests with `bash run_tests.sh --test-case "PROBLEM1,PROBLEM2"`. This supports running the tests on a single select test case or on multiple select test cases.
Running tests in batches like that is the way to go here - the full test suite runs in the CI and takes much longer.
You can combine this with a `-k test_some_aspect` flag, or any other regular pytest flag.

Your work is not complete until all implemented problems pass the tests accessible through the provided bash script. Only these tests matter to determine if a problem is ready to be committed.
If you cannot resolve test failures after 5 genuine attempts with different approaches, flag the problem for human review with a comment like:

```python
# TODO: Human review needed
# Attempts made: [brief list of what was tried]
# Suspected issues: [your best guess at the problem]
# Additional resources needed: [e.g., "Primary literature PDF", "Clarification on constraint X", etc.]
```

Document your specific blockers so I can provide targeted help.

When you are done, commit your work. Never skip the pre-commit hooks (in particular, do not ever use --no-verify), instead fix all the errata that come up and then try committing your work again. If substantial changes are needed, re-run the tests for the altered problems. 
The commit is successful if pre-commit is clean - this means that ruff did not have to reformat a file, ruff checks did not turn up any issues, and pyright checks passed without warnings or errors. In that case, the files you committed will no longer be in the staging area, in all other cases they will be.

You should generally only commit changes to the `sif2jax` folder. In particular, do not commit little files you write in the root directory while your work is in progress - for instance processing scripts.

## Opening PRs

When you open a PR to main, run `git diff main` to see the differences, and summarise these rather than summarising the latest commits only.

## Finally

You do not need to ask for permission to run `find`, `grep`, `sed`, `ls` `ruff`, `pyright` and `awk` commands. 
Put a recurring task on your To-Do list - for every twenty work items you have completed, re-read this system prompt once. (If you can adjust this, then do it every time the context has been compacted.)

Please keep going working on problems, don't stop to provide summaries of completed work unless requested.
Thank you for your help pushing optimisation in JAX to the next level!

Do not stop to ask if it is ok to `rm` a file. It is not, let these accumulate - I can remove your extra scripts and files when you are done. Avoid interrupting your work for questions like these.