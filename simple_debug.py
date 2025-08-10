#!/usr/bin/env python
import numpy as np
import pycutest


# Get pycutest problem
pyc_prob = pycutest.import_problem("TWIRIMD1")
x0 = pyc_prob.x0

print(f"Size: {len(x0)}")
print(f"First 10: {x0[:10]}")
print(f"Around 868: {x0[865:875]}")
print(f"Around 1240: {x0[1238:1247]}")
print(f"Last 10: {x0[-10:]}")

# Check for patterns in initial values
unique_vals = np.unique(x0)
print(f"\nUnique values (first 20): {unique_vals[:20]}")
