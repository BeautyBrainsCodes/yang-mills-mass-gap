
import numpy as np
import matplotlib.pyplot as plt

# Define grid parameters
grid_size = 10  # smaller due to computational constraints
time_steps = 30
k = 0.1

# Initialize 4D lattice: [x, y, z, vector(3)]
A = np.random.rand(grid_size, grid_size, grid_size, 3) * 0.1
energy_history = []

def cross_neighbor_field_4D(A, x, y, z):
    neighbors = []
    if x > 0: neighbors.append(np.cross(A[x-1, y, z], A[x, y, z]))
    if x < grid_size - 1: neighbors.append(np.cross(A[x+1, y, z], A[x, y, z]))
    if y > 0: neighbors.append(np.cross(A[x, y-1, z], A[x, y, z]))
    if y < grid_size - 1: neighbors.append(np.cross(A[x, y+1, z], A[x, y, z]))
    if z > 0: neighbors.append(np.cross(A[x, y, z-1], A[x, y, z]))
    if z < grid_size - 1: neighbors.append(np.cross(A[x, y, z+1], A[x, y, z]))
    return np.sum(neighbors, axis=0)

# Time evolution of 3D space = 4D simulation
for t in range(time_steps):
    A_new = np.copy(A)
    for x in range(1, grid_size-1):
        for y in range(1, grid_size-1):
            for z in range(1, grid_size-1):
                A_new[x, y, z] += k * cross_neighbor_field_4D(A, x, y, z)
    A = A_new
    energy = np.linalg.norm(A, axis=3)
    energy_history.append(np.sum(energy))  # total energy snapshot

# Plot total energy over time
plt.figure(figsize=(10, 5))
plt.plot(energy_history)
plt.title("4D Simulation: Total Energy Over Time")
plt.xlabel("Time Step (t)")
plt.ylabel("Total Energy (E)")
plt.grid(True)
plt.show()
