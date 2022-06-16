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
        self.data = TrajectoryParameters(self._N_total_steps)

        # Set final state (we'll be doing a backwards integration here)
        self.data.x[-1] = self._x[-1]
        self.data.h[-1] = self._h[-1]
        self.data.W[-1] = self._airplane.get_min_weight()

        # Loop through legs of the trajectory
        j_end = self._N_total_steps - 1
        j_start = j_end - self._N_steps[-1]
        for i in range(self._N_legs)[::-1]:

            # Integrate
            y0 = np.array([self._gammas[i], self.data.h[j_end], self.data.W[j_end]])
            xi, yi = RK4(self._airplane.state_equation_wrt_x, y0, self._x[i+1], self._x[i], self._N_steps[i])

            # Parse out state variables
            self.data.x[j_start:j_end+1] = xi[::-1]
            self.data.gamma[j_start:j_end+1] = yi[::-1,0]
            self.data.h[j_start:j_end+1] = yi[::-1,1]
            self.data.W[j_start:j_end+1] = yi[::-1,2]

            # Set up for next iteration
            j_end = j_start
            j_start = j_end - self._N_steps[i-1]

        # Calculate other parameters
        for i in range(self._N_total_steps):
            self.data.V[i] = self._airplane.get_max_range_airspeed(self.data.gamma[i], self.data.W[i], self.data.h[i])
            self.data.CL[i] = self._airplane.get_CL(self.data.W[i], self.data.gamma[i], self.data.V[i], self.data.h[i])
            self.data.K[i], self.data.TA[i], self.data.fuel_burn[i] = self._airplane.get_engine_performance(self.data.V[i], self.data.W[i], self.data.h[i], self.data.gamma[i])
            self.data.V_stall[i] = self._airplane.get_stall_speed(self.data.W[i], self.data.h[i], self.data.gamma[i])

        return self.data


    def write_data(self, filename):
        """Writes the trajectory data to a txt file.
        
        Parameters
        ----------
        filename : str
            File to write the data to.
            
        """

        # Open file
        with open(filename, 'w') as data_file:

            # Write header
            print("x[ft] Altitude[ft] Velocity[ft/s] gamma W V_stall CL K T_A fuel_burn", file=data_file)

            for i in range(self._N_total_steps):
                print("{0:>20.12f} {1:>20.12f} {2:>20.12f} {3:>20.12f} {4:>20.12f} {5:>20.12f} {6:>20.12f} {7:>20.12f} {8:>20.12f} {9:>20.12f}".format(self.data.x[i],
                                                                                                                                                       self.data.h[i],
                                                                                                                                                       self.data.V[i],
                                                                                                                                                       self.data.gamma[i],
                                                                                                                                                       self.data.W[i],
                                                                                                                                                       self.data.V_stall[i],
                                                                                                                                                       self.data.CL[i],
                                                                                                                                                       self.data.K[i],
                                                                                                                                                       self.data.TA[i],
                                                                                                                                                       self.data.fuel_burn[i]), file=data_file)



    def plot(self):
        """Plots the data for this trajectory."""

        # Plot
        fig, ax = plt.subplots(nrows=2,ncols=3)

        ax[0,0].plot(self.data.x, self.data.h, 'b-')
        ax[0,0].set_xlabel('$x$ [ft]')
        ax[0,0].set_ylabel('$h$ [ft]')
        ax[0,0].set_title('Altitude')

        ax[0,1].plot(self.data.x, self.data.V, 'b-', label='Actual')
        ax[0,1].plot(self.data.x, self.data.V_stall, 'r--', label='Stall')
        ax[0,1].plot(self.data.x, self.data.V_stall*1.1, 'y--', label='10% Margin')
        ax[0,1].plot(self.data.x, self.data.V_stall*1.2, 'g--', label='20% Margin')
        ax[0,1].set_xlabel('$x$ [ft]')
        ax[0,1].set_ylabel('$V$ [ft/s]')
        ax[0,1].set_title('Airspeed')
        ax[0,1].legend()

        ax[0,2].plot(self.data.x, self.data.W, 'b-')
        ax[0,2].set_xlabel('$x$ [ft]')
        ax[0,2].set_ylabel('$W$ [lbf]')
        ax[0,2].set_title('Weight')

        ax[1,0].plot(self.data.x, self.data.CL, 'b-')
        ax[1,0].set_xlabel('$x$ [ft]')
        ax[1,0].set_ylabel('$CL$')
        ax[1,0].set_title('Lift Coefficient')

        ax[1,1].plot(self.data.x, self.data.K, 'b-')
        ax[1,1].set_xlabel('$x$ [ft]')
        ax[1,1].set_ylabel('$\\kappa$')
        ax[1,1].set_title('Throttle Setting')

        ax[1,2].plot(self.data.x, self.data.TA, 'r--', label='Available')
        ax[1,2].plot(self.data.x, self.data.TA*self.data.K, 'b-', label='Used')
        ax[1,2].set_xlabel('$x$ [ft]')
        ax[1,2].set_ylabel('$T$ [lbf]')
        ax[1,2].set_title('Thrust')
        ax[1,2].legend()

        plt.show()


    def get_total_flight_time(self):
        """Gives the total flight time for the trajectory."""
        return np.trapz(1.0/self.data.V, x=self.data.x)


    def get_total_fuel_burn(self):
        """Calculates the total fuel burn."""
        return self.data.W[0] - self.data.W[-1]


class TrajectoryParameters:
    """Container class for trajectory data.
    
    Parameters
    ----------
    N_steps : int
        Total number of steps in the trajectory.
    """

    def __init__(self, N_steps):
        
        # Initialize
        self.N_steps = N_steps

        # Allocate storage
        self.x = np.zeros(self.N_steps)
        self.h = np.zeros(self.N_steps)
        self.V = np.zeros(self.N_steps)
        self.TA = np.zeros(self.N_steps)
        self.K = np.zeros(self.N_steps)
        self.CL = np.zeros(self.N_steps)
        self.fuel_burn = np.zeros(self.N_steps)
        self.gamma = np.zeros(self.N_steps)
        self.W = np.zeros(self.N_steps)
        self.V_stall = np.zeros(self.N_steps)