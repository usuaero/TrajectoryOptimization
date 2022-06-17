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
    x = np.array([0.0,   123049.45075844, 198463.59488465, 280660.49765378, 298443.22584921, 298626.44225277, 312467.93493334, 2627054.78067103, 2893786.32187765, 3108188.29708082])
    h = np.array([128.0, 17956.45288514,  27965.03435763,  38255.69646853,  37041.14134656,  37027.59762872,  36107.99574718,  36784.09702616,   18734.63290521,   4226.0])
    N_steps = np.ones(len(x)-1, dtype=int)*int(1000/6)
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