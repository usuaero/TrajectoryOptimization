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
    
    # Inputs
    x = np.array([0.0,   123051.00468246, 198463.35848024, 280666.54096332, 298424.79372597, 298642.49806828, 312794.39436446, 2627028.27011332, 2893802.4644102, 3108188.29708082])
    h = np.array([128.0, 17958.06269174,  27965.54308721,  38257.01437954,  37041.19613503,  37028.51073348,  36087.87455237,  36784.85023584,   18733.55060457,  4226.0])
    N_total_steps = 10000
    N_steps = np.ones(len(x)-1, dtype=int)*int(N_total_steps/(len(x)-1))
    total_steps = np.sum(N_steps).item()
    if total_steps != N_total_steps:
        N_steps[0] += N_total_steps - total_steps
    print()
    print("Total steps used: {0}".format(np.sum(N_steps).item()))

    # Initialize trajectory
    trajectory = TrapezoidalTrajectory(AMG12, x, h, N_steps)

    # Calculate
    data = trajectory.calculate()
    print("Total fuel burn: {0} lbf".format(data.W[0] - data.W[-1]))

    # Get total flight time
    t_flight = trajectory.get_total_flight_time()
    print("Total flight time: {0} hr".format(round(t_flight/3600, 5)))

    # Plot
    trajectory.write_data("optimal_trajectory.txt")
    trajectory.plot()