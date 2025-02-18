import numpy as np
import matplotlib.pyplot as plt

# Define the stiff ODE dy/dt = 1000 * y * cos(t)
def f(y, t):
    return 1000 * y * np.cos(t)

# Euler's method implementation
def euler_solver(f, y0, t0, T, dt):
    t_values = np.arange(t0, T, dt)
    y_values = np.zeros_like(t_values)
    y_values[0] = y0
    
    for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1] + dt * f(y_values[i-1], t_values[i-1])
    
    return t_values, y_values

# Initial conditions
y0 = 1
t0 = 0
T = 0.01  # Short time to observe instability

# Solve for different step sizes
dt_values = [0.001, 0.0005, 0.0001]  # Large to small step sizes

plt.figure(figsize=(10, 6))
for dt in dt_values:
    t_values, y_values = euler_solver(f, y0, t0, T, dt)
    plt.plot(t_values, y_values, label=f"dt = {dt}")

plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.title("Euler Method on a Stiff ODE")
plt.legend()
plt.grid()
plt.show()
