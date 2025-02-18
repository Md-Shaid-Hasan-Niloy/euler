import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor

def euler_solver(f, y0, t):
    """ Implements Euler's method for solving ODEs. """
    y = np.zeros(len(t))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + (t[i] - t[i-1]) * f(y[i-1], t[i-1])
    return y

def stiff_ode(y, t):
    """ Defines the stiff ODE: dy/dt = 1000(y cos(t)) """
    return 1000 * y * np.cos(t)

def run_one_case(value):
    """ Runs Euler's method for a given step size and measures error & runtime. """
    i, dt = value  # Unpack tuple
    t = np.arange(0, 1 + dt, dt)  # Time steps from 0 to 1
    y_exact = np.exp(1000 * np.sin(t))  # Exact solution
    y0 = 1  # Initial condition

    start_time = time.time()
    y_numerical = euler_solver(stiff_ode, y0, t)
    end_time = time.time()

    final_error = np.abs(y_numerical[-1] - y_exact[-1])  # Error at final step
    runtime = end_time - start_time  # Measure execution time

    return (i, dt, final_error, runtime)

# Define step sizes
m = 4  # Number of decreasing step sizes
dt_values = [0.5**i for i in range(m+1)]  # Step sizes: 1, 0.5, 0.25, 0.125, 0.0625
values = [(i, dt_values[i]) for i in range(m+1)]  # Store index-step size pairs

# Run cases in parallel
if __name__ == "__main__":
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_one_case, values))

    # Print results
    print("Index | Step Size | Final Error | Runtime (s)")
    for res in results:
        print(f"{res[0]:5d} | {res[1]:8.5f} | {res[2]:12.5e} | {res[3]:10.5f}")
