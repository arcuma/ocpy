import sympy as sym
import numpy as np
import numba
import abc
import time
from os.path import join, abspath, dirname

from ocpy import symutils
from ocpy.ocp import OCP
from ocpy.logger import Logger
from ocpy.plotter import Plotter


class SolverBase(abc.ABC):
    """ Abstract solver class.
    """
    def __init__(self, ocp: OCP):
        """ Init, supposing unconstrained Newton-type method.
        """
        self._ocp = ocp
        self._sim_name = ocp.get_ocp_name()
        self._log_dir = join(dirname(dirname(abspath(__file__))), 'log',
                             self._sim_name)
        self._solver_name = ''
        # dimensions of state and input.
        self._n_x = ocp.get_n_x()
        self._n_u = ocp.get_n_u()
        # Horizon length, num of discretization, time step.
        self._T = ocp.get_T()
        self._N = ocp.get_N()
        self._dt = self._T / self._N
        # initial time, state and guess.
        self._t0 = ocp.get_t0()
        self._x0 = ocp.get_x0()
        self._xs_guess = ocp.get_xs_guess()
        self._us_guess = ocp.get_us_guess()
        # stepsize of line search.
        self._alphas = np.array([0.5**i for i in range(8)])
        # damping value.
        self._gamma_ini = 1e-3
        self._rho_gamma = 10.0
        self._gamma_min = 1e-8
        self._gamma_max = 1e+6
        # solver parameters
        self._max_iters = 500
        self._stop_threshold = 1e-3
        # flag
        self._is_single_shooting = True
        self._initialized = False
        # derivatives of functions
        self._df = ocp.get_df()
        self._dl = ocp.get_dl()
        self._f, self._fx, self._fu, self._fxx, self._fux, self._fuu = self._df
        self._l, self._lx, self._lu, self._lxx, self._lux, self._luu, \
            self._lf, self._lfx, self._lfxx = self._dl

    def ocp(self):
        """ Return OCP.

        Returns:
            OCP: ocp instance.
        """
        return self._ocp

    def set_log_directory(self, log_dir: str):
        """ Set directory path where data are logged.

        Args:
            log_dir (str): Log directory.
        """
        self._log_dir = log_dir

    def get_log_directory(self) -> str:
        """ Get directory path where data are logged.

        Args:
            log_dir (str): Log directory.
        """
        return self._log_dir
    
    def set_damping_coefficient(self, gamma_ini: float=None, rho_gamma: float=None,
                                gamma_min: float=None, gamma_max: float=None):
        """ Set gammaing coefficient of Newton method.
        
        Args: 
            gamma_ini (float): Initial value of damping coefficient.
            rho_gamma (float): Increasing/decreasing factor of gamma. (>=1)
            gamma_min (float): Minimum value of damp.
            gamma_max (float): Maximum value of damp.
        """
        if gamma_ini is not None:
            self._gamma_ini = gamma_ini
        if rho_gamma is not None:
            self._rho_gamma = rho_gamma
        if gamma_min is not None:
            self._gamma_min = gamma_min
        if gamma_max is not None:
            self._gamma_max = gamma_max

    def set_alphas(self, alphas: np.ndarray=None):
        """ Set alphas: candidates of step size of line search.
        
        Args: 
            alphas (np.ndarray): Array of alpha.  0 <= alpha_i <= 1.0.
        """
        if alphas is not None:
            self._alphas = np.array(alphas, dtype=float)

    def set_max_iters(self, max_iters: int=None):
        """ Set number of maximum iteration.
        
        Args: 
            max_iters (int): Number of maximum iteration.
        """
        if max_iters is not None:
            self._max_iters = max_iters    

    def reset_initial_condition(self, t0: float=None, x0: np.ndarray=None):
        """ Reset t0 and x0.

        Args:
            t0 (float): Initial time of horizon.
            x0 (float): Initial state of horizon.
        """
        self._ocp.reset_initial_condition(t0, x0)
        self._t0 = self._ocp.get_t0()
        self._x0 = self._ocp.get_x0()
    
    def reset_horizon(self, T: float=None, N: float=None):
        """ Reset T and N.

        Args:
            T (float): Horizon length.
            N (int): Discretization grid number.
        """
        self._ocp.reset_horizon(T, N)
        self._T = self._ocp.get_T()
        self._N = self._ocp.get_N()
        self._dt = self._T / self._N

    def reset_guess(self, xs_guess: np.ndarray=None, us_guess: np.ndarray=None):
        """ Reset guess of state and input trajectory.

        Args:
            xs_guess (np.ndarray): Guess of state trajectory.
            us_guess (np.ndarray): Guess of input trajectory.
        """
        self._ocp.reset_guess(xs_guess, us_guess)
        self._xs_guess = self._ocp.get_xs_guess()
        self._us_guess = self._ocp.get_us_guess()

    @abc.abstractmethod
    def set_solver_parameters(self):
        """ Set solver parameters.
        """
        pass

    @abc.abstractmethod
    def init_solver(self):
        """ Initialize solver. Call once before you first call solve().
        """
        pass

    @abc.abstractmethod
    def solve(
            self,
            t0: float=None, x0: np.ndarray=None, N: int=None, T: float=None,
            xs_guess: np.ndarray=None ,us_guess: np.ndarray=None,
            gamma_fixed: float=None, enable_line_search: bool=True,
            result: bool=False, log: bool=False, plot: bool=False
        ):
        """ Solve ocp.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def print_result():
        pass

    @staticmethod
    @abc.abstractmethod
    def log_data():
        pass

    @staticmethod
    @abc.abstractmethod
    def plot_data():
        pass
