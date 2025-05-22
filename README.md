# sif2jax
Functionally pure definitions of optimisation problems extracted from Standard Input Format (SIF).

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

