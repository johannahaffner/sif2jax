# System prompt for Claude

You're helping us translate problems from the CUTEST collection to clean and efficient JAX implementations.
The problems currently exist as instructions to build Fortran code, which does not work in a JAX context and prevents us from properly validating optimisation algorithms. 
This is a crucial bottleneck that needs fixing!

Since these problem definitions are to be used to validate optimisation algorithms and compare the results to algorithms running the original SIF definitions, it is very important that our JAX implementations are as close as possible to the original - up to numerical precision.

## Tipps and Tricks

To verify problems, it is frequently helpful to check the AMPL definitions, since these are often more concise and easier to read. 
Where they are not available and wherever specific values are required (constants, starting points), please look at the SIF definitions. 
If this also does not help, you can check if you can find a document that is related in the `extra_info` folder, or look up the original reference (provided in the SIF file) online.
There is also a paper in the `extra_info` folder that describes the group-separable structure of the SIF files, this might be helpful to understand the SIF definitions and I suggest you read it in before working on this.
There are also six screenshots, if you cannot open the PDF.

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
8. Add at least twenty problems to your To-Do list. Don't stop after five! We have lots to do.
9. Run ruff format and ruff check on your work. Ruff is installed in the working directory.
10. If the SIF file includes a nice documentation feature - such as a graphic representation of the problem, be sure to include that. Generally include all problem information given above the problem definition in the SIF file. 

## Testing

You can check any work against the original Fortran implementations. For this purpose, please run `bash run_tests.sh`, this script pulls a container that includes compiled Fortran libraries that provide these. 
Without this container, you cannot run any tests. 
Under no circumstances can you make any changes to the tests/ folder - just use it to inform your next steps. 

When you don't know what to do, find the next problem to work on by using `bash run_tests.sh --exit-first` and start fixing the first problem for which the tests fail.
When you make any change to a file, please run the tests again. You can run problem-specific tests with `bash run_tests.sh --test-case "PROBLEM"`. 

Your work is not complete until all implemented problems pass the tests. If you can't fix the issues in a problem after several attempts, it is ok to flag the problem class as requiring human review, but please give it a serious attempt before that.

## Finally

You do not need to ask for permission to run `grep`, `sed`, `ls` and `awk` commands. 

Please keep going working on problems, don't stop to provide summaries of completed work unless requested.
Thank you for your help pushing optimisation in JAX to the next level!