import numpy as np
from numba import njit
from src.ode_model import ODEModel
from src.euler_solver import EulerSolver

@njit
def decay(y, t):
    """
    Exponential decay ODE: y'(t) = -y.
    """
    return -y

def test_euler_exponential():
    # Analytical solution for exponential decay
    def exact_solution(t, y0):
        return y0 * np.exp(-t)

    # Set up the ODE model
    y0 = 1.0
    t0 = 0.0
    tf = 1.0
    dt = 0.01
    model = ODEModel(f=decay, y_0=y0, t_0=t0, t_f=tf, dt=dt)

    # Solve using Euler's method
    solver = EulerSolver(model, t0, tf, dt)
    t_values, y_values = solver.solve()

    # Compare with the exact solution
    y_exact = exact_solution(t_values, y0)
    error = np.abs(y_values[-1] - y_exact[-1])
    assert error < 1e-2, "Euler method error is too large"

if __name__ == "__main__":
    test_euler_exponential()
    print("All tests passed!")
