import numpy as np
import matplotlib.pyplot as plt

# Parameters
grid_size = 100       # Number of 1D lattice points
time_steps = 30       # Total time evolution steps
k = 0.1               # Coupling strength

# Initialize 1D lattice with 3-vector fields at each point
A = np.random.rand(grid_size, 3) * 0.1
energy_history = []

# Local evolution rule: sum of cross-products with neighbors
def cross_neighbor_field_1D(A, x):
    neighbors = []
    if x > 0:
        neighbors.append(np.cross(A[x - 1], A[x]))
    if x < grid_size - 1:
        neighbors.append(np.cross(A[x + 1], A[x]))
    return np.sum(neighbors, axis=0)

# Time evolution
for t in range(time_steps):
    A_new = A.copy()
    for x in range(1, grid_size - 1):
        A_new[x] += k * cross_neighbor_field_1D(A, x)
    A = A_new
    energy = np.linalg.norm(A, axis=1)
    energy_history.append(np.sum(energy))

# Plot total energy vs. time
plt.plot(range(time_steps), energy_history, marker='o')
plt.title("Total Energy Over Time (1D)")
plt.xlabel("Timestep")
plt.ylabel("Total Energy")
plt.grid(True)
plt.tight_layout()
plt.show()
