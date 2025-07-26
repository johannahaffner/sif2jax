# This file should contain tests that verify that our implementations do not trigger
# recompilation.
# This should use the following equinox debugging utility:
# https://docs.kidger.site/equinox/api/debug/#equinox.debug.assert_max_traces
# The structure of the tests can be simple - we just need to call the objective and
# constraint functions repeatedly, after wrapping them as needed (and described in the
# the documentation linked above).
# This assumes that the gradients and Hessians do not trigger recompilation if the
# objective function does not, which is probably safe to assume.
