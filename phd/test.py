import enum
import numpy as np
import matplotlib.pyplot as plt
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

    ## Plot fuel consumption as a function of airspeed
    #h = 10000
    #W = 530000
    #Vs = np.linspace(100, 1000, 200)
    #FC = np.zeros_like(Vs)
    #for i, V in enumerate(Vs):
    #    FC[i] = AMG12.get_fuel_consumption_slf(V, W, h)

    #plt.figure()
    #plt.plot(Vs, FC)
    #plt.xlabel('$V$ [ft/s]')
    #plt.ylabel('Fuel Consumption')
    #plt.show()

    # Plot minimum fuel consumption airspeed as a function of altitude and weight
    N_h = 10
    N_W = 10
    hs = np.linspace(0.0, 20000.0, N_h)
    Ws = np.linspace(345000.0, 630000.0, N_W)
    V_MFC = np.zeros((N_h, N_W))
    for i, h in enumerate(hs):
        for j, W in enumerate(Ws):
            V_MFC[i,j] = AMG12.get_min_fuel_consumption_airspeed_slf(W, h)

    plt.figure()
    for i, h in enumerate(hs):
        plt.plot(Ws, V_MFC[i,:], label=str(round(h)))
    plt.xlabel('$W$ [lbf]')
    plt.ylabel('$V_{MFC}$')
    plt.legend(title='Altitude [ft]')
    plt.show()