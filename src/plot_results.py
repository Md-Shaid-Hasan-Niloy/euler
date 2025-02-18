import matplotlib.pyplot as plt
import numpy as np
from src.ode_model import ODEModel
from src.euler_solver import EulerSolver

def exact_solution(t, y0=1.0):
    """Computes the exact solution y(t) = y0 * exp(-t)"""
    return y0 + (-1000)*(t-np.sin(-t))

# Set up the ODE model
y0 = 1.0
t0 = 0.0
tf = 3.0
dt = 0.001
from numba import njit

@njit
def decay(y, t):
    return -1000*(y-np.cos(t))

model = ODEModel(f=decay, y_0=y0, t_0=t0, t_f=tf, dt=dt)


# Solve using Euler's method
solver = EulerSolver(model, t0, tf, dt)
t_values, y_numerical = solver.solve()

# Compute exact solution
y_exact = exact_solution(t_values, y0)

# Compute relative difference
relative_diff = np.abs(y_exact - y_numerical) / np.abs(y_exact)

# Plot results
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# Upper panel: Numerical vs. Exact solution
ax[0].plot(t_values, y_numerical, label="Euler's Method", linestyle='--', marker='o')
ax[0].plot(t_values, y_exact, label="Exact Solution", linestyle='-')
ax[0].set_ylabel("y(t)")
ax[0].legend()
ax[0].set_title("Numerical vs. Exact Solution")

# Lower panel: Relative Difference
ax[1].plot(t_values, relative_diff, label="Relative Difference", color="red")
ax[1].set_xlabel("Time t")
ax[1].set_ylabel("Relative Difference")
ax[1].set_yscale("log")  # Log scale for better visualization
ax[1].legend()

plt.tight_layout()
plt.show()
