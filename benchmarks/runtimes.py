# This file should run sif2jax problems (objective and constraint functions) and their
# pycutest counterparts, time them as appropriate (see runtime tests for suggestions)
# and export the results to a .csv file to be saved in the benchmarks folder.
# This file should have the following format:
#
# - problem name
# - sif2jax objective runtime
# - pycutest objective runtime
# - sif2jax constraint runtime
# - pycutest constraint runtime
# - dimensionality
#
# Problems that do not implement a constraint method should have a NaN as the value for
# the constraint runtime. (I think all problems implement an objective function?)
# Prefer to avoid using pandas, work directly with (ordered) dictionaries or lists if
# possible.
# This file needs to be run within the container, so you will need to create a script
# that runs it and mirrors the setup in the run_tests.sh script - the local path and
# image details should be copied.
