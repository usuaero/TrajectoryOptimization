import numpy as np
import matplotlib.pyplot as plt
from airplane import Airplane
from optimizer import optimize_trajectory

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

    # Optimize
    x0 = 0.0
    x1 = 3108188.29708082
    h0 = 128.0
    h1 = 4226.0
    x_guess = np.array([122932.81785758, 198363.21760193, 281635.97198683, 299099.31164569, 305000.0, 312895.38519193, 2627692.28068185, 2894465.27724405])
    h_guess = np.array([17935.60606775,  27946.08293001,  38366.08934554,  37241.89657242,  37000.0,  36405.21445405,  36741.28573272,   18688.70170814])
    opt_traj = optimize_trajectory(AMG12, x0, x1, h0, h1, x_guess, h_guess, N_steps_per_leg=30)

    # Get total flight time and fuel burn
    t_flight = opt_traj.get_total_flight_time()
    print()
    print("Total flight time: {0} hr".format(round(t_flight/3600, 5)))
    print("Total fuel burn: {0} lbf".format(opt_traj.data.W[0] - opt_traj.data.W[-1]))

    # Show optimal trajectory
    opt_traj.write_data('optimal_trajectory.txt')
    opt_traj.plot()