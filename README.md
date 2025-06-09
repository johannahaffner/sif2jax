# sif2jax
Functionally pure definitions of optimisation problems extracted from Standard Input Format (SIF).

This is for you if you write optimisation software in JAX or Python and want to stress-test it on the CUTEst set of benchmark problems. 
We're porting these to JAX to be able to use them without a Fortran backend, enabling realistic measurements of runtimes, leveraging all of JAX capabilities: autodifferentiation, autocompilation, and autoparallelism.

## cyipopt installation

I recommend using the conda install as it packages all necessary binaries. I can add this to the dockerimage later.

```bash
conda install -c conda-forge cyipopt
```

## CUTEst problems in SIF format

To make the CUTEst problems in SIF format accessible to the LLM:

```bash
mkdir archive  # This name is in the .gitignore
cd archive
git clone https://bitbucket.org/optrove/sif ./mastsif
```
Suggestion: change the permissions to that folder to remove write access.

