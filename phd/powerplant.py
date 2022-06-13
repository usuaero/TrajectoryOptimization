import numpy as np
from standard_atmosphere import StandardAtmosphere

class Powerplant:
    """A class defining a jet powerplant."""

    def __init__(self, propulsion_info) -> None:
        
        # Store info
        self._Ne = propulsion_info["N_engines"]
        self._q = propulsion_info["q"]
        self._CT = propulsion_info["CT"]
        self._N1 = propulsion_info["N1"]
        self._T0 = propulsion_info["T0"]
        self._m = propulsion_info["m"]
        self._a1 = propulsion_info["a1"]
        self._a2 = propulsion_info["a2"]

        # Initialize standard atmosphere model
        self._std_atmos = StandardAtmosphere("English")


    def get_available_thrust(self, h, V):
        """Calculates the available thrust based on the altitude and velocity.
        
        Parameters
        ----------
        h : float
            Altitude in ft.
            
        V : float
            Velocity in ft/s.
        
        Returns
        -------
        float
            Available thrust in lbf.
        """

        # Calculate thrust
        return self.get_max_thrust(h)*(1 + self._a1*V + self._a2*V*V)


    def get_thrust_specific_fuel_consumption(self, h, M):
        """Calculates the thrust-specific fuel consumption based on the altitude and velocity.
        
        Parameters
        ----------
        h : float
            Altitude in ft.
            
        V : float
            Velocity in ft/s.
        
        Returns
        -------
        float
            Thrust-specific fuel consumption.
        """

        # Get temperatures
        Th = self._std_atmos.T(h)
        T0 = self._std_atmos.T(0)

        # Calculate Mach number
        a = self._std_atmos.a(h)
        M = V/a

        # Calculate consumption
        return self._CT*(Th/T0)**0.5*M**self._q


    def get_max_thrust(self, h):
        """Returns the maximum possible thrust at the given altitude.
        
        Parameters
        ----------
        h : float
            Altitude in ft.
            
        Returns
        -------
        float
            Max thrust in lbf.
        """

        # Get temperatures
        Th = self._std_atmos.T(h)
        T0 = self._std_atmos.T(0)

        # Calculate thrust
        return self._Ne*self._N1*self._T0*(Th/T0)**self._m