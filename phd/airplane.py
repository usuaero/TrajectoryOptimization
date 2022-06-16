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
        self._Wf_min = info["min_fuel"]
        self._CL_max = info["CL_max"]
        self._CD0 = info["CD0"]
        self._CD1 = info["CD1"]
        self._CD2 = info["CD2"]
        self._CM1 = info["CM1"]
        self._CM2 = info["CM2"]

        # Store powerplant info
        self.engines = Powerplant(propulsion_info)

        # Initialize standard atmosphere model
        self._std_atmos = StandardAtmosphere("English")


    def get_min_weight(self):
        return self._We + self._Wf_min


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


    def get_fuel_consumption(self, V, gamma, W, h):
        """Calculates the fuel consumption rate.
        
        Parameters
        ----------
        V : float
            Velocity.

        gamma : float
            Climb angle.

        W : float
            Weight.

        h : float
            Altitude.
        
        Returns
        -------
        float
            Fuel consumption rate per s.
        """

        # Get drag
        D = self.get_drag(W, gamma, V, h)

        # Calculate thrust
        T = D + W*np.sin(gamma)

        # Get thrust-specific fuel consumption
        qT = self.engines.get_thrust_specific_fuel_consumption(h, V)

        return qT*T


    def get_min_fuel_consumption_airspeed(self, gamma, W, h):
        """Calculates the velocity which minimizes fuel consumption rate per s.
        
        Parameters
        ----------
        gamma : float
            Climb angle.

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
        V_min = self.get_stall_speed(W, h, gamma)
        V_max = 0.95*self._std_atmos.a(h) # No transonic here
        result = sopt.minimize_scalar(self.get_fuel_consumption, bounds=(V_min, V_max), method='bounded', args=(gamma, W, h))
        return result.x


    def get_max_range_airspeed(self, gamma, W, h):
        """Calculates the velocity which minimizes fuel consumption rate per distance travelled.
        
        Parameters
        ----------
        gamma : float
            Climb angle.

        W : float
            Weight.

        h : float
            Altitude.
        
        Returns
        -------
        float
            Maximum range airspeed.
        """

        # Declare objective function
        def f(V):
            return self.get_fuel_consumption(V, gamma, W, h)/V

        # Optimize
        V_min = self.get_stall_speed(W, h, gamma)
        V_max = 0.9*self._std_atmos.a(h) # No transonic here
        result = sopt.minimize_scalar(f, bounds=(V_min, V_max), method='bounded')
        return result.x


    def get_stall_speed(self, W, h, gamma):
        """Calculates the stall speed.
        
        Parameters
        ----------
        W : float
            Weight.

        h : float
            Altitude.

        gamma : float
            Climb angle.
        
        Returns
        -------
        float
            Stall speed.
        """

        # Get density
        rho = self._std_atmos.rho(h)

        return np.sqrt(2.0*W*np.cos(gamma)/(rho*self._Sw*self._CL_max))


    def get_max_speed(self, W, h, gamma):
        """Calculates the max speed based on how much thrust the engines can provide.
        
        Parameters
        ----------
        W : float
            Weight.

        h : float
            Altitude.

        gamma : float
            Climb angle.
        
        Returns
        -------
        float
            Stall speed.
        """

        # Define root function
        def f(V):
            TA = self.engines.get_available_thrust(h, V)
            D = self.get_drag(W, gamma, V, h)
            return D - TA + W*np.sin(gamma)

        # Find root
        V_stall = self.get_stall_speed(W, h, gamma)
        result = sopt.root_scalar(f, x0=V_stall*1.5, x1=V_stall*1.6)

        return result.root


    def get_min_speed(self, W, h, gamma):
        """Calculates the min speed based on zero thrust.
        
        Parameters
        ----------
        W : float
            Weight.

        h : float
            Altitude.

        gamma : float
            Climb angle.
        
        Returns
        -------
        float
            Stall speed.
        """

        # Define root function
        def f(V):
            D = self.get_drag(W, gamma, V, h)
            return D + W*np.sin(gamma)

        # Find root
        V_stall = self.get_stall_speed(W, h, gamma)
        result = sopt.root_scalar(f, x0=V_stall*1.5, x1=V_stall*1.6)

        return result.root


    def get_engine_performance(self, V, W, h, gamma):
        """Calculates the throttle setting, thrust available, and fuel consumption rate at the given state.
        
        Parameters
        ----------
        V : float
            Velocity.

        gamma : float
            Climb angle.

        W : float
            Weight.

        h : float
            Altitude.
        
        Returns
        -------
        float
            Throttle setting.

        float
            Thrust available in lbf.
        """

        # Calculate thrust available
        TA = self.engines.get_available_thrust(h, V)

        # Calculate drag
        D = self.get_drag(W, gamma, V, h)

        # Calculate throttle
        K = (D + W*np.sin(gamma)) / TA

        # Get thrust-specific fuel consumption
        qT = self.engines.get_thrust_specific_fuel_consumption(h, V)

        # Check for minimum fuel burn
        fuel_burn = max(0.0, qT*K*TA)

        return K, TA, fuel_burn


    def get_dynamics(self, V, W, h, gamma):
        """Calculates the dynamic equations for the aircraft.
        
        Parameters
        ----------
        V : float
            Velocity.

        gamma : float
            Climb angle.

        W : float
            Weight.

        h : float
            Altitude.
        
        Returns
        -------
        float
            Derivative of distance.

        float
            Climb rate.

        float
            Rate of weight loss.
        """

        # Calculate kinematics
        x_dot = V*np.cos(gamma)
        h_dot = V*np.sin(gamma)

        # Calculate mass rate
        _,_,fuel_burn = self.get_engine_performance(V, W, h, gamma)

        return x_dot, h_dot, -fuel_burn


    def state_equation_wrt_x(self, x, y):
        """Returns a state space derivative for the aircraft with x as the independent variable
        
        Parameters
        ----------
        y : ndarray
            An array containing gamma, height, and weight.
            
        Returns
        -------
        ndarray
            State space derivative of the state.
        """

        # Parse out state
        gamma = y[0]
        h = y[1]
        W = y[2]

        # Calculate best airspeed
        V = self.get_max_range_airspeed(gamma, W, h)

        # Get derivatives
        x_dot, h_dot, W_dot = self.get_dynamics(V, W, h, gamma)

        return np.array([0.0, h_dot/x_dot, W_dot/x_dot])