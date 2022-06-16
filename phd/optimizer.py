import numpy as np
import scipy.optimize as sopt
from trajectories import TrapezoidalTrajectory
from airplane import Airplane


large = 1.0e9


def optimize_trajectory(airplane, x0, x1, h0, h1, x_guess, h_guess, N_steps_per_leg=100):
    """Optimizes a trapezoidal trajectory for the given airplane.
    
    Parameters
    ----------
    airplane : Airplane
        Airplane object for which the trajectory should be optimized.
    
    x0 : float
        Initial distance.

    x1 : float
        Desination distance.
    
    h0 : float
        Initial altitude.

    h1 : float
        Desination altitude.

    x_guess : float
        Initial guess for the intermediate distance waypoints.

    h_guess : float
        Initial guess for the intermediate altitude waypoints.

    N_steps_per_leg : int
        Number of integration steps to use over each leg of the flight.
    
    Returns
    -------
    TrapezoidalTrajectory
        The optimal trajectory
    """

    # Get number of trajectory legs
    N_legs = len(x_guess) + 1
    N_steps = np.ones(N_legs, dtype=int)*N_steps_per_leg

    # Define objective function
    def f(waypoints):

        # Calculate
        try:
            trajectory = run_trajectory(airplane, x0, x1, h0, h1, waypoints, N_steps)
            data = trajectory.data
        except:
            return large
        else:

            # Get fuel burn
            W_fuel = trajectory.get_total_fuel_burn()

            # Check throttle constraint
            K_max = np.max(trajectory.data.K).item()
            margin = 1.0 - K_max
            if margin < 0.0:
                W_fuel -= large*margin

            # Check stall constraint
            margin = data.V - data.V_stall
            min_margin = np.min(margin).item()
            if min_margin < 0.0:
                W_fuel -= large*min_margin

            # Check takeoff and landing stall constraint
            landing_margin = data.V[0] - data.V_stall[0]*1.1
            if landing_margin < 0.0:
                W_fuel -= large*(landing_margin)
            takeoff_margin = data.V[-1] - data.V_stall[-1]*1.2
            if takeoff_margin < 0.0:
                W_fuel -= large*(takeoff_margin)

            # Print out data
            t_flight = np.trapz(1.0/data.V, x=data.x)
            print()
            print(waypoints)
            print(W_fuel, t_flight/3600)
            return W_fuel

    # Set up constraints
    
    # Throttle constraint
    def g_throttle(waypoints):

        # Run trajectory
        try:
            traj = run_trajectory(airplane, x0, x1, h0, h1, waypoints, N_steps)
        except:
            return -1.0

        # Get throttle furthest outside [0, 1]
        K_max = np.max(traj.data.K).item()
        margin = 1.0 - K_max
        print()
        print(margin)
        return margin

    throttle_constraint = {
        "type" : "ineq",
        "fun" : g_throttle
    }

    # Optimize
    guess = np.zeros(2*(N_legs-1))
    guess[:N_legs-1] = x_guess
    guess[N_legs-1:] = h_guess
    options = {
        "finite_diff_rel_step" : np.ones_like(guess)*1e-4,
        "disp" : True
    }
    result = sopt.minimize(f, guess, method='Nelder-Mead', options=options)#, constraints=[throttle_constraint])

    # Get optimal trajectory
    return run_trajectory(airplane, x0, x1, h0, h1, result.x, N_steps)


def run_trajectory(airplane, x0, x1, h0, h1, waypoints, N_steps):
    """Runs the given trajectory.
    
    Parameters
    ----------
    airplane : Airplane
        Airplane object for which the trajectory should be optimized.
    
    x0 : float
        Initial distance.

    x1 : float
        Desination distance.

    h0 : float
        Initial altitude.

    h1 : float
        Desination altitude.

    waypoints : float
        Intermediate distance and altitude waypoints.

    N_steps_per_leg : int
        Number of integration steps to use over each leg of the flight.
    
    Returns
    -------
    TrapezoidalTrajectory
        The trajectory with data calculated.
    """

    # Parse out inputs
    N_legs = len(N_steps)
    x = waypoints[:N_legs-1]
    h = waypoints[N_legs-1:]

    # Get distances
    x_traj = np.zeros(N_legs + 1)
    x_traj[0] = x0
    x_traj[-1] = x1
    x_traj[1:-1] = x

    # Get altitudes
    h_traj = np.zeros(N_legs + 1)
    h_traj[0] = h0
    h_traj[-1] = h1
    h_traj[1:-1] = h

    # Run
    trajectory = TrapezoidalTrajectory(airplane, x_traj, h_traj, N_steps)
    trajectory.calculate()
    return trajectory