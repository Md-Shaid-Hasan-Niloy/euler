from typing import Callable

class ODEModel:
    """
    Encapsulates the ODE y'(t) = f(y,t) and parameters.
    """
    def __init__(self, f: Callable[[float, float], float], y_0: float = 1.0, t_0: float = 0.0, t_f: float = 1.0, dt: float = 0.1):
        """
        Initialize the ODE model with the ODE function f, initial condition y_0, start time t_0, end time t_f, and time step dt.
        """
        self.f = f  # Do not decorate with @njit here
        self.y_0 = y_0  # Initial condition
        self.t_0 = t_0  # Start time
        self.t_f = t_f  # End time
        self.dt = dt  # Time step

    def call(self, y: float, t: float) -> float:
        """
        Call the ODE function f with the current state y and time t.
        """
        return self.f(y, t)

# Example usage
def decay(y: float, t: float = None) -> float:
    """
    Exponential decay ODE: y'(t) = -y.
    """
    return -y

ode = ODEModel(f=decay, y_0=1.0, t_0=0.0, t_f=1.0, dt=0.1)
print(ode.call(1.0, 0.0))  # Outputs: -1.0
