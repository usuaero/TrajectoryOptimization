import numpy as np
import scipy.optimize as sopt
from powerplant import Powerplant
from standard_atmosphere import StandardAtmosphere


class Airplane:
    """A class defining an airplane. Modeled using a linear drag model, stteady-state dynamics, etc."""

    def __init__(self, info, propulsion_info):
        
        # Store parameters
        self._b = info["wingspan"]
        self._Sw = info["wing_area"]
        self._We = info["empty_weight"]
        self._Wf_max = info["fuel_capacity"]
        self._CL_max = info["CL_max"]
        self._CD0 = info["CD0"]
        self._CD1 = info["CD1"]
        self._CD2 = info["CD2"]
        self._CM1 = info["CM1"]
        self._CM2 = info["CM2"]

        # Store powerplant info
        self._engines = Powerplant(propulsion_info)

        # Initialize standard atmosphere model
        self._std_atmos = StandardAtmosphere("English")


    def calc_state(self, K, h, W, x):
        """Calculates the state of the aircraft based off the throttle setting, position, and weight.
        
        Parameters
        ----------
        K : float
            Throttle setting between 0 and 1.

        h : float
            Altitude in ft.

        W : float
            Weight in lbf.

        x : float
            Position in ft.
        """

        # Get atmospheric parameters
        rho = self._std_atmos.rho(h)
        a = self._std_atmos.a(h)
        
        # Define function to find the root of
        def f(state):

            # Parse out state
            L = state[0]
            D = state[1]
            gamma = state[2]
            TA = state[3]
            V = state[4]

            # Initialize root
            result = np.zeros(5)

            # Calculate some preliminaries
            M = V/a
            nondim = 1.0/(0.5*rho*V*V*self._Sw)
            CL = L*nondim
            CD = D*nondim
            PA = K*TA*V
            PR = D*V
            
            # Lift equation
            result[0] = L - W*np.sin(gamma)

            # Drag equation
            result[1] = D - TA*K + W*np.cos(gamma)

            # Climb rate equation
            result[2] = (PA-PR)/W - V*np.sin(gamma)

            # Drag coefficient equation
            result[3] = CD - self.get_CD(CL, M)

            # Available thrust equation
            result[4] = TA - self._engines.get_available_thrust(h, V)

            return result

        # Solve
        V_guess = np.sqrt(2.0*W/(rho*self._Sw*0.5)) # Guess a lift coefficient of 0.5
        state_guess = np.array([W, 0.0, 0.0, self._engines.get_max_thrust(h), V_guess])
        state = sopt.fsolve(f, state_guess)

        # Parse out state
        L = state[0]
        D = state[1]
        gamma = state[2]
        TA = state[3]
        V = state[4]

        # Get extras
        M = V/a

        return L, D, gamma, TA, V, M


    def get_CD(self, CL, M):
        """Calculates the drag coefficient.
        
        Parameters
        ----------
        CL : float
            Lift coefficient.
            
        M : float
            Mach number.
            
        Returns
        -------
        float
        
            Drag coefficient.
        """

        return (self._CD0 + self._CD1*CL + self._CD2*CL*CL)*(1.0 + self._CM1*M**self._CM2)


    def get_lift(self, W, gamma):
        """Calculates the lift.
        
        Parameters
        ----------
        W : float
            Weight.
            
        gamma : float
            Climb angle.
            
        Returns
        -------
        float
            Lift force in lbf.
        """

        return W*np.cos(gamma)


    def get_CL(self, W, gamma, V, h):
        """Calculates the lift coefficient.
        
        Parameters
        ----------
        W : float
            Weight.
            
        gamma : float
            Climb angle.

        V : float
            Velocity.

        h : float
            Altitude.
            
        Returns
        -------
        float
            Lift coefficient
        """

        # Get density and lift
        rho = self._std_atmos.rho(h)
        L = self.get_lift(W, gamma)

        # Calculate lift coefficient
        return 2.0*L/(rho*V*V*self._Sw)


    def get_drag(self, W, gamma, V, h):
        """Calculates the drag force.
        
        Parameters
        ----------
        W : float
            Weight.
            
        gamma : float
            Climb angle.

        V : float
            Velocity.

        h : float
            Altitude.
            
        Returns
        -------
        float
            Drag force in lbf.
        """

        # Get CL and rho
        CL = self.get_CL(W, gamma, V, h)
        rho = self._std_atmos.rho(h)

        # Get CD
        M = V/self._std_atmos.a(h)
        CD = self.get_CD(CL, M)

        return 0.5*rho*V*V*self._Sw*CD


    def get_fuel_consumption_slf(self, V, W, h):
        """Calculates the fuel consumption rate in steady-level flight.
        
        Parameters
        ----------
        V : float
            Velocity.

        W : float
            Weight.

        h : float
            Altitude.
        
        Returns
        -------
        float
            Fuel consumption rate.
        """

        # Get drag
        D = self.get_drag(W, 0.0, V, h)

        # Get thrust-specific fuel consumption
        qT = self._engines.get_thrust_specific_fuel_consumption(h, V)

        return qT*D


    def get_min_fuel_consumption_airspeed_slf(self, W, h):
        """Calculates the velocity which minimizes fuel consumption rate in steady-level flight.
        
        Parameters
        ----------
        W : float
            Weight.

        h : float
            Altitude.
        
        Returns
        -------
        float
            Minimum fuel consumption airspeed.
        """

        # Optimize
        V_min = self.get_stall_speed(W, h)
        V_max = 0.95*self._std_atmos.a(h) # No transonic here
        result = sopt.minimize_scalar(self.get_fuel_consumption_slf, bounds=(V_min, V_max), method='bounded', args=(W, h))
        return result.x


    def get_stall_speed(self, W, h):
        """Calculates the stall speed.
        
        Parameters
        ----------
        W : float
            Weight.

        h : float
            Altitude.
        
        Returns
        -------
        float
            Stall speed.
        """

        # Get density
        rho = self._std_atmos.rho(h)

        return np.sqrt(2.0*W/(rho*self._Sw*self._CL_max))