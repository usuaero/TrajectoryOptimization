import numpy as np
import matplotlib.pyplot as plt
from airplane import Airplane


def RK4(f, x0, t0, t1, N_steps):
    """Integrates f(t,x) from t0 to t1 in steps of dt.
    
    Parameters
    ----------
    f : callable
        Function to integrate.

    x0 : ndarray
        Initial state for dependent variables.
    
    t0 : float
        Initial state for independent variable.
    
    t1 : float
        Final state for independent variable.

    N_steps : float
        Number of steps to take in the independent variable.
    """

    # Initialize storage
    t = np.linspace(t0, t1, N_steps+1)
    dt = t[1]-t[0]
    N = len(t)
    x = np.zeros((N,len(x0)))
    x[0,:] = x0

    # Loop
    for i, (ti, xi) in enumerate(zip(t[:-1],x[:-1])):

        k1 = f(ti, xi)
        k2 = f(ti+0.5*dt, xi+0.5*dt*k1)
        k3 = f(ti+0.5*dt, xi+0.5*dt*k2)
        k4 = f(ti+dt, xi+dt*k3)

        x[i+1,:] = xi + 0.1666666666666666666667*dt*(k1 + 2.0*k2 + 2.0*k3 + k4)

    return t, x


class TrapezoidalTrajectory:
    """Defines a trajectory with fixed climb angles between altitude and distance waypoints.
    
    Parameters
    ----------
    airplane : Airplane
        The airplane performing the trajectory.

    x : ndarray
        Distance waypoints. Must be the same length as h.

    h : ndarray
        Altitude waypoints. Must be the same length as x.

    N_steps : ndarray
        Number of integration steps to take for each leg.
    """

    def __init__(self, airplane, x, h, N_steps):

        # Store
        self._airplane = airplane
        self._x = x
        self._h = h
        self._N_steps = N_steps
        self._N_legs = len(x) - 1

        # Calculate climb angle for each leg
        self._gammas = np.zeros(self._N_legs)
        for i in range(self._N_legs):
            self._gammas[i] = np.arctan2(self._h[i+1]-self._h[i], self._x[i+1]-self._x[i])

        # Total number of integration steps
        self._N_total_steps = np.sum(N_steps).item() + 1


    def calculate(self):
        """Calculates the dependent parameters of the trajectory assuming the aircraft always flies at the
        maximum range airspeed.

        Returns
        -------
        ndarray
            Distance

        ndarray
            Climb angle

        ndarray
            Altitude

        ndarray
            Weight
        """

        # Allocate storage
        x = np.zeros(self._N_total_steps)
        h = np.zeros(self._N_total_steps)
        V = np.zeros(self._N_total_steps)
        TA = np.zeros(self._N_total_steps)
        K = np.zeros(self._N_total_steps)
        CL = np.zeros(self._N_total_steps)
        fuel_burn = np.zeros(self._N_total_steps)
        gamma = np.zeros(self._N_total_steps)
        W = np.zeros(self._N_total_steps)
        V_stall = np.zeros(self._N_total_steps)

        # Set final state (we'll be doing a backwards integration here)
        x[-1] = self._x[-1]
        h[-1] = self._h[-1]
        W[-1] = self._airplane.get_min_weight()

        # Loop through legs of the trajectory
        j_end = self._N_total_steps - 1
        j_start = j_end - self._N_steps[-1]
        for i in range(self._N_legs)[::-1]:

            # Integrate
            y0 = np.array([self._gammas[i], h[j_end], W[j_end]])
            xi, yi = RK4(self._airplane.state_equation_wrt_x, y0, self._x[i+1], self._x[i], self._N_steps[i])

            # Parse out state variables
            x[j_start:j_end+1] = xi[::-1]
            gamma[j_start:j_end+1] = yi[::-1,0]
            h[j_start:j_end+1] = yi[::-1,1]
            W[j_start:j_end+1] = yi[::-1,2]

            # Set up for next iteration
            j_end = j_start
            j_start = j_end - self._N_steps[i-1]

        # Calculate other parameters
        for i in range(self._N_total_steps):
            V[i] = self._airplane.get_max_range_airspeed(gamma[i], W[i], h[i])
            CL[i] = self._airplane.get_CL(W[i], gamma[i], V[i], h[i])
            K[i], TA[i], fuel_burn[i] = self._airplane.get_engine_performance(V[i], W[i], h[i], gamma[i])
            V_stall[i] = self._airplane.get_stall_speed(W[i], h[i], gamma[i])

        return x, gamma, h, W, V, CL, K, TA, fuel_burn, V_stall