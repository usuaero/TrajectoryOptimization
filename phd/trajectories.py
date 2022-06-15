import numpy as np
from airplane import Airplane


def RK4(f, x0, t0, t1, dt):
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

    dt : float
        Step size for independent variable.
    """

    # Initialize storage
    t = np.arange(t0, t1, dt)
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
        self._N_total_steps = np.sum(N_steps).item()


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
        gamma = np.zeros(self._N_total_steps)
        W = np.zeros(self._N_total_steps)

        # Set final state (we'll be doing a backwards integration here)
        x[-1] = self._x[-1]
        h[-1] = self._h[-1]
        W[-1] = self._airplane.get_min_weight()

        # Loop through legs of the trajectory
        j_end = self._N_total_steps - 1
        j_start = j_end - self._N_steps[-1]
        for i in range(self._N_legs)[::-1]:

            # Get dx
            dx = -(self._x[i+1] - self._x[i])/self._N_steps[i]

            # Integrate
            y0 = np.array([self._gammas[i], h[j_end], W[j_end]])
            x, y = RK4(self._airplane.state_equation_wrt_x, y0, self._x[i+1], self._x[i], dx)

            # Parse out state variables
            x[j_start:j_end] = x
            gamma[j_start:j_end] = y[:,0]
            h[j_start:j_end] = y[:,1]
            W[j_start:j_end] = y[:,2]

            # Set up for next iteration
            j_end = j_start
            j_start = j_end - self._N_steps[i-1]

        return x, gamma, h, W