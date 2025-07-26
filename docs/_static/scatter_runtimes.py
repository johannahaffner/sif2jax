# In this file, we create a scatter plot of the runtimes of sif2jax problems, plotted
# as a function of the runtime of the Fortran implementation available through pycutest.
# The runtimes should be available as a .csv file, stored in the benchmarks folder.
# The runtimes of pycutest problems should be plotted on the x axis, and the runtimes
# of sif2jax problems on the y axis. The plot should be symmetric and square, and it
# should have a 1:1 line on the diagonal (dashed or dotted).
# The dots should be coloured according to the dimensionality of the problem. The title
# is runtimes of objective functions. We need a colorbar on the right. Use a rainbow
# colormap, such as jet or Spectral.
