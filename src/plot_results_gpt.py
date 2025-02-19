import numpy as np
import matplotlib.pyplot as plt

# Define the ODE: y' = -1000(y - cos(t))
def f(y, t):
    -y

# Euler's method implementation
def euler_solver(f, y0, t0, T, dt):
    t_values = np.arange(t0, T, dt)
    y_values = np.zeros_like(t_values)
    y_values[0] = y0

    for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1] + dt * f(y_values[i-1], t_values[i-1])

    return t_values, y_values

# Exact solution for the ODE: y(t) = (1000000*cos(t) + 1000*sin(t) + e^(-1000*t)) / 1000001
def exact_solution(t):
    return (

# Initial conditions
y0 = 1
t0 = 0
T = 0.1  # Time range to observe behavior
dt = 0.001  # Step size

# Solve using Euler's method
t_values, y_values = euler_solver(f, y0, t0, T, dt)

# Exact solution
t_exact = np.linspace(t0, T, 1000)  # High resolution for exact solution
y_exact = exact_solution(t_exact)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_values, y_values, label=f"Euler Method (dt = {dt})", linestyle="-", color="blue")
plt.plot(t_exact, y_exact, label="Exact Solution", linestyle="--", color="black")
plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.title(f"Euler Method vs Exact Solution for y' = -1000(y - cos(t)) (dt = {dt})")
plt.legend()
plt.grid()
plt.show()
