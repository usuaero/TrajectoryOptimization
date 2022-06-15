from airplane import Airplane

class TrapezoidalTrajectory:
    """Defines a trajectory with fixed climb angle, fixed cruise altitude, and fixed descent angle.
    Over the cruise portion, the aircraft will fly at the minimum fuel consumption airspeed.
    
    Parameters
    ----------
    airplane : Airplane
        The airplane performing the trajectory.
        
    h0 : float
        Initial altitude.
        
    hc : float
        Cruise altitude.
        
    h1 : float
        Destination altitude.
        
    xc : float
        Distance at which the airplane ends climb and begins cruise.
        
    xd : float
        Distance at which the airplane ends cruise and beings descent.
        
    x1 : float
        Distance to destination.
    """

    def __init__(self, airplane, h0, hc, h1, xc, xd, x1):
        
        # Store
        self._airplane = airplane
        self._h0 = h0
        self._hc = hc
        self._h1 = h1
        self._xc = xc
        self._xd = xd
        self._x1 = x1

        # Calculate climb angles
        self._gamma_c = np.arctan2(hc-h0, xc)
        self._gamma_d = np.arctan2(h1-hc, x1-xd)