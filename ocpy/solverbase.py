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
        """ Constructor. Supposing unconstrained single-shooting Newton-type method.
        """
        self._ocp = ocp
        self._sim_name = ocp.get_ocp_name()
        self._log_dir = join(dirname(dirname(abspath(__file__))), 'log',
                             self._sim_name)
        self._solver_name = ''

        # dimensions of state and input.
        self._n_x = ocp.get_n_x()
        self._n_u = ocp.get_n_u()

        # initial time and state
        self._t0 = ocp.get_t0()
        self._x0 = ocp.get_x0()

        # Horizon length, num of discretization, time step.
        self._T = ocp.get_T()
        self._N = ocp.get_N()
        self._dt = self._T / self._N

        # derivatives of functions
        self._df = ocp.get_df()
        self._dl = ocp.get_dl()
        self._f, self._fx, self._fu, self._fxx, self._fux, self._fuu = self._df
        self._l, self._lx, self._lu, self._lxx, self._lux, self._luu, \
            self._lf, self._lfx, self._lfxx = self._dl

        # stepsize parameter
        self._alpha_min = 1e-4
        self._r_alpha = 0.5

        # regularization value.
        self._gamma_init = 0.0
        self._r_gamma = 5.0
        self._gamma_min = 0.0
        self._gamma_max = 0.0

        # solver parameters
        self._max_iters = 1000

        # flag
        self._is_single_shooting = True
        self._initialized = False

        # optimal trajectory
        self._xs_opt = np.zeros((self._N + 1, self._n_x))
        self._us_opt = np.zeros((self._N, self._n_u))

        # time grids
        ts = np.array([self._t0 + i*self._dt for i in range(self._N + 1)])

        # result (success flag, NoI, ...)
        self._result = {}
        self._result['is_success'] = None
        self._result['noi'] = None
        self._result['computation_time'] = None

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
        assert isinstance(log_dir, str)
        self._log_dir = log_dir

    def get_log_directory(self) -> str:
        """ Get directory path where data are logged.

        Returns:
            log_dir (str): Log directory.
        """
        return self._log_dir
    
    def set_regularization_param(self, gamma_init: float=None, r_gamma: float=None,
                                 gamma_min: float=None, gamma_max: float=None):
        """ Set regularization parameters of Newton method.
        
        Args: 
            gamma_init (float): Initial value of regularization coefficient.
            r_gamma (float >= 1): Increasing/decreasing factor of gamma.
            gamma_min (float): Minimum value of gamma.
            gamma_max (float): Maximum value of gamma.
        """
        if gamma_init is not None:
            self._gamma_init = gamma_init
        if r_gamma is not None:
            self._r_gamma = r_gamma
        if gamma_min is not None:
            self._gamma_min = gamma_min
        if gamma_max is not None:
            self._gamma_max = gamma_max

    def set_line_search_param(self, alpha_min: float=None, r_alpha: float=None):
        """ Set parameters related to line search.

        Args:
            alpha_min (float): Minimum stepsize.
            r_alpha (float): Update ratio of alpha. Must be (0, 1).
        """
        if alpha_min is not None:
            self._alpha_min = alpha_min
        if r_alpha is not None:
            self._r_alpha = r_alpha

    def set_max_iters(self, max_iters: int=None):
        """ Set number of maximum iteration.
        
        Args: 
            max_iters (int): Number of maximum iteration.
        """
        if max_iters is not None:
            assert max_iters > 0
            self._max_iters = max_iters

    def set_initial_condition(self, t0: float=None, x0: np.ndarray=None):
        """ Reset t0 and x0.

        Args:
            t0 (float): Initial time.
            x0 (float): Initial state.
        """
        if t0 is not None:
            self._t0 = float(t0)
        if x0 is not None:
            x0 = np.array(x0, dtype=float).reshape(-1)
            assert x0.shape == (self._n_x,)
            self._x0 = x0
    
    def set_horizon(self, T: float=None, N: float=None):
        """ Reset T and N.

        Args:
            T (float > 0): Horizon length.
            N (int > 0): Discretization grid number.

        Note:
            If horison is changed, guess must be changed. \
            Use reset_guess().
        """
        if T is not None:
            assert T > 0
            self._T = float(T)
        if N is not None:
            assert N > 0
            self._N = N
        self._dt = self._T / self._N
        print("Method set_horizon(T, N) was called.")
        print("If you changed horizon parameters, do not forget to call "
              "set_guess() or reset_guess() to change initial guess.")

    def get_xs_opt(self):
        """ Get optimal state trajectory.

        Returns:
            xs_opt (np.ndarray): Optimal state trajectory.
        """
        return self._xs_opt
    
    def get_us_opt(self):
        """ Get optimal input trajectory.

        Returns:
            us_opt (np.ndarray): Optimal input trajectory.
        """
        return self._us_opt

    def get_result(self):
        """ Get result.

        Returns:
            result (dict): result.
        """
        return self._result

    @abc.abstractmethod
    def set_guess(self, us_guess: np.ndarray=None):
        """ Set initial guess.
        """
        pass

    @abc.abstractmethod
    def reset_guess(self):
        """ Reset guess.
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
            update_gamma: bool=True, enable_line_search: bool=True,
            max_iters: int=None, warm_start: bool=False,
            result: bool=False, log: bool=False, plot: bool=False
        ):
        """ Solve ocp.
        """
        pass

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
