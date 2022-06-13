from airplane import Airplane


if __name__=="__main__":

    # Declare input
    airplane_info = {
        "wingspan" : 180,
        "wing_area" : 3600,
        "empty_weight" : 330000,
        "fuel_capacity" : 300000,
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

    # Test state calculation
    L, D, gamma, TA, V, M = AMG12.calc_state(0.5, 10000, 530000, 0.0)
    print(V)
    print(gamma)