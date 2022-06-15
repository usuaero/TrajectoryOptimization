import numpy as np
import scipy.optimize as sopt
from trajectories import TrapezoidalTrajectory
from airplane import Airplane


def optimize_trajectory(airplane, x, h0, h1, N_steps_per_leg=100):
    """Optimizes a trapezoidal trajectory for the given airplane.
    
    Parameters
    ----------
    airplane : Airplane
        Airplane object for which the trajectory should be optimized.
    
    x : ndarray
        Distance waypoints at which the altitude may be set by the optimize, including start and finish.

    h0 : float
        Initial altitude.

    h1 : float
        Desination altitude.

    N_steps_per_leg : int
        Number of integration steps to use over each leg of the flight.
    
    Returns
    -------
    TrajectoryParameters 
        Data for the optimal trajectory.

    W_fuel
        Total weight of fuel burned.

    t_flight
        Total flight time.
    """

    # Get number of trajectory legs
    N_legs = len(x) - 1
    N_steps = np.ones(N_legs, dtype=int)*N_steps_per_leg

    # Define objective function
    def f(h):

        # Calculate trajectory
        trajectory = TrapezoidalTrajectory(airplane, x, h, N_steps)

        # Calculate
        data = trajectory.calculate()
        W_fuel = data.W[0] - data.W[-1]
        t_flight = np.trapz(1.0/data.V, x=data.x)
        print(W_fuel, t_flight/3600)

        return W_fuel

    # Initial guess
    h = np.zeros_like(x)
    h[:-1] = h0
    h[-1] = h1

    # Optimize
    result = sopt.minimize(f, h)
    return result.fun