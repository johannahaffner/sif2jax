import numpy as np


# Parse the SIF file for initial values
initial_values = {}

with open("/workspace/twir-series/archive/mastsif/TWIRIMD1.SIF") as f:
    for line in f:
        if line.startswith(" X"):
            parts = line.split()
            if len(parts) >= 3:
                var_name = parts[1]
                try:
                    value = float(parts[2])
                    initial_values[var_name] = value
                except ValueError:
                    # Skip lines that don't have numeric values
                    pass

# Problem dimensions
Nnod = 31
Nred = 28
Ndia = 3
Ntim = 6
Nage = 4
Ntra = 7

# Build the initial vector in the correct order
# Order: x variables, then k/phi interleaved, then keff, then epsilon

# Initialize arrays
x_values = np.zeros((Nnod, Nage, Ntra))
k_values = np.zeros((Nnod, Ntim))
phi_values = np.zeros((Nnod, Ntim))
keff_values = np.zeros(Ntim)
epsilon_value = 0.01  # Default

# Parse x variables
for i in range(1, Nnod + 1):
    for l in range(1, Nage + 1):
        for m in range(1, Ntra + 1):
            key = f"x{i},{l},{m}"
            if key in initial_values:
                x_values[i - 1, l - 1, m - 1] = initial_values[key]

# Parse k variables
for i in range(1, Nnod + 1):
    for t in range(1, Ntim + 1):
        key = f"k{i},{t}"
        if key in initial_values:
            k_values[i - 1, t - 1] = initial_values[key]

# Parse phi variables
for i in range(1, Nnod + 1):
    for t in range(1, Ntim + 1):
        key = f"phi{i},{t}"
        if key in initial_values:
            phi_values[i - 1, t - 1] = initial_values[key]

# Parse keff variables
for t in range(1, Ntim + 1):
    key = f"keff{t}"
    if key in initial_values:
        keff_values[t - 1] = initial_values[key]

# Parse epsilon
if "epsilon" in initial_values:
    epsilon_value = initial_values["epsilon"]

# Flatten x in the right order
x_flat = x_values.ravel()

# Interleave k and phi
kphi_interleaved = np.zeros(2 * Nnod * Ntim)
for i in range(Nnod):
    for t in range(Ntim):
        idx = 2 * (i * Ntim + t)
        kphi_interleaved[idx] = k_values[i, t]
        kphi_interleaved[idx + 1] = phi_values[i, t]

# Combine all
y0 = np.concatenate([x_flat, kphi_interleaved, keff_values, [epsilon_value]])

print(f"Total size: {len(y0)}")
print(f"First 10: {y0[:10]}")
print(f"Around 868: {y0[865:875]}")
print(f"Last 10: {y0[-10:]}")

# Save for use in implementation
np.save("twirimd1_y0.npy", y0)
