import numpy as np
import matplotlib.pyplot as plt
from trajectories import TrapezoidalTrajectory
from airplane import Airplane


if __name__=="__main__":

    # Declare input
    airplane_info = {
        "wingspan" : 180,
        "wing_area" : 3600,
        "empty_weight" : 330000,
        "fuel_capacity" : 300000,
        "min_fuel" : 45000,
        "CL_max" : 1.4,
        "CD0" : 0.0194,
        "CD1" : -0.0159,
        "CD2" : 0.09,
        "CM1" : 3.0,
        "CM2" : 30.0
    }
    propulsion_info = {
        "N_engines" : 2,
        "q" : 0.6,
        "CT" : 6.0706e-6,
        "N1" : 0.9,
        "T0" : 93000,
        "m" : 0.75,
        "a1" : -9.5e-4,
        "a2" : 5e-7
    }

    # Initialize airplane
    AMG12 = Airplane(airplane_info, propulsion_info)
    
    # Initialize trajectory
    x = np.array([0.0, 400000.0, 2500000.0, 3108188.29708082])
    h = np.array([128.0, 10000.0, 10000.0, 4226.0])
    N_steps = np.array([100, 100, 100])
    trajectory = TrapezoidalTrajectory(AMG12, x, h, N_steps)

    # Calculate
    x, gamma, h, W = trajectory.calculate()

    # Plot
    fig, ax = plt.subplots(ncols=2)
    ax[0].plot(x, h)
    ax[1].plot(x, W)
    ax[0].set_xlabel('$x$ [ft]')
    ax[1].set_xlabel('$x$ [ft]')
    ax[0].set_ylabel('$h$ [ft]')
    ax[1].set_ylabel('$W$ [lbf]')
    plt.show()