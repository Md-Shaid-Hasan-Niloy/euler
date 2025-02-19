import numpy as np
import matplotlib.pyplot as plt

# Exponential Decay ODE
def exponential_decay_ode(y, t, k):
    return -k * y

def exponential_decay_exact(t, y0, k):
    return y0 * np.exp(-k * t)

# Logistic Growth ODE
def logistic_growth_ode(y, t, r, K):
    return r * y * (1 - y / K)

def logistic_growth_exact(t, y0, r, K):
    return K / (1 + (K / y0 - 1) * np.exp(-r * t))

# Euler's Method for Numerical Solution
def euler_solver(f, y0, t0, T, dt, *args):
    t_values = np.arange(t0, T, dt)
    y_values = np.zeros_like(t_values)
    y_values[0] = y0

    for i in range(1, len(t_values)):
        y_values[i] = y_values[i-1] + dt * f(y_values[i-1], t_values[i-1], *args)

    return t_values, y_values

# Parameters
y0 = 1
t0 = 0
T = 10
dt = 0.01
k = 0.5  # Decay rate for exponential decay
r = 0.2  # Growth rate for logistic growth
K = 10   # Carrying capacity for logistic growth

# Solve Exponential Decay ODE
t_exp, y_exp_euler = euler_solver(exponential_decay_ode, y0, t0, T, dt, k)
y_exp_exact = exponential_decay_exact(t_exp, y0, k)

# Solve Logistic Growth ODE
t_log, y_log_euler = euler_solver(logistic_growth_ode, y0, t0, T, dt, r, K)
y_log_exact = logistic_growth_exact(t_log, y0, r, K)

# Plot Results
plt.figure(figsize=(12, 6))

# Exponential Decay
plt.subplot(1, 2, 1)
plt.plot(t_exp, y_exp_euler, label="Euler Method", linestyle="-", color="blue")
plt.plot(t_exp, y_exp_exact, label="Exact Solution", linestyle="--", color="black")
plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.title("Exponential Decay ODE")
plt.legend()
plt.grid()

# Logistic Growth
plt.subplot(1, 2, 2)
plt.plot(t_log, y_log_euler, label="Euler Method", linestyle="-", color="red")
plt.plot(t_log, y_log_exact, label="Exact Solution", linestyle="--", color="black")
plt.xlabel("Time t")
plt.ylabel("y(t)")
plt.title("Logistic Growth ODE")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
