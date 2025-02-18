import numpy as np
from numba import njit

@njit
def euler_step(f, y, t, dt):
    """
    Perform one step of Euler's method: y_{n+1} = y_n + dt * f(y_n, t_n).
    """
    return y + dt * f(y, t)

class EulerSolver:
    """
    Solves an ODE using the Euler method.
    """
    def __init__(self, model, t_0: float, t_f: float, dt: float):
        self.model = model
        self.t_0 = t_0
        self.t_f = t_f
        self.dt = dt

    def solve(self):
        """
        Solve the ODE using the Euler method and return the time series and solution.
        """
        t_values = np.arange(self.t_0, self.t_f, self.dt)
        y_values = np.zeros_like(t_values)
        y_values[0] = self.model.y_0

        for i in range(1, len(t_values)):
            y_values[i] = euler_step(self.model.f, y_values[i-1], t_values[i-1], self.dt)

        return t_values, y_values
