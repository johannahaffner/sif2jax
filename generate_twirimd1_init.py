import numpy as np


# Load the correct values
y0 = np.load("twirimd1_y0.npy")

print("# Replace the initial_guess method with this:")
print("    @property")
print("    def initial_guess(self) -> jnp.ndarray:")
print('        """Initial guess for TWIRIMD1 - exact values from SIF file."""')
print("        return jnp.array([")

# Print values in rows of 10
for i in range(0, len(y0), 10):
    row = y0[i : min(i + 10, len(y0))]
    values_str = ", ".join(f"{v:.6g}" for v in row)
    if i + 10 < len(y0):
        print(f"            {values_str},")
    else:
        print(f"            {values_str}")

print("        ])")
