import numpy as np
import matplotlib.pyplot as plt

# Define problem parameters
T = 1.0  # Final time
N = 100  # Number of time steps
dt = T / N  # Time step
gamma = 0.9  # Discount factor
alpha = 0.1  # Learning rate

# Define state and control grids
x_min, x_max = 0.0, 1.0  # State space limits
u_min, u_max = -1.0, 1.0  # Control input limits
num_x = 101  # Number of grid points for state
num_u = 101  # Number of grid points for control
x_grid = np.linspace(x_min, x_max, num_x)  # State grid
u_grid = np.linspace(u_min, u_max, num_u)  # Control grid

# Initialize value function and control
V = np.zeros(num_x)  # Value function
u_star = np.zeros(num_x)  # Optimal control

# Define the dynamics function
def dynamics(x, u):
    return x + u

# Define the cost function
def cost(x, u):
    return 0.5 * (x ** 2 + u ** 2)

# Perform value iteration using finite differences
for t in reversed(range(N)):
    V_new = np.zeros_like(V)
    for i in range(num_x):
        x = x_grid[i]
        J_vals = np.zeros(num_u)
        for j in range(num_u):
            u = u_grid[j]
            x_next = dynamics(x, u)
            if x_next < x_min or x_next > x_max:
                J_vals[j] = float('inf')  # Penalize out-of-bounds states
            else:
                J_vals[j] = cost(x, u) + gamma * V[i+1]  # Bellman update
        V_new[i] = np.min(J_vals)  # Update value function
        u_star[i] = u_grid[np.argmin(J_vals)]  # Update optimal control
    V = V_new  # Update value function for the next time step

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(x_grid, V, label='Value Function')
plt.xlabel('State (x)')
plt.ylabel('Value')
plt.title('Value Function vs State')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(x_grid, u_star, label='Optimal Control')
plt.xlabel('State (x)')
plt.ylabel('Control (u)')
plt.title('Optimal Control vs State')
plt.legend()
plt.grid(True)
plt.show()
