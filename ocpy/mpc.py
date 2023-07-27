### WIP

import sympy as sym
import numpy as np
import time
from os.path import join, abspath, dirname

from ocpy.ocp import OCP
from ocpy.ddp import SolverBase
from ocpy.logger import Logger
from ocpy.plotter import Plotter


class MPC:
    """ Model Predictive Control(MPC) class.
    """
    def __init__(self, solver: SolverBase):
        """ Constructior.

        Args:
            solver (SolberBase): Solver instance.
        """
        self._solver = solver
        self._ocp = solver.ocp()
        self._f = self._ocp.get_df()[0]
        self._t0 = self._ocp.get_t0()
        self._x0 = self._ocp.get_x0()
        self._us_guess = self._ocp.get_us_guess()
        self._xs_guess = self._ocp.get_xs_guess()
        self._initialized = False
    
    def init_mpc(
            self, t0: float=None, x0: np.ndarray=None, T: float=None, N: int=None,
            xs_guess: np.ndarray=None ,us_guess: np.ndarray=None,
        ):
        """ Solve ocp once for getting solution guess.

        Args:
            t0 (float): Initial time.
            x0 (numpy.array): Initial state. Size must be n_x.
            T (float): Horizon length.
            N (int): Discretization grid number.
            xs_guess (numpy.array): Guess of state trajectory. \
                Size must be (N+1)*n_x. Only used in multiple-shooting.
            us_guess (numpy.array): Guess of input trajectory. \
                Size must be N*n_u.
        
        Note:
            Do not change initial condition after this.
        """
        xs, us, _, _, _ = self._solver.solve(t0, x0, N, T, xs_guess, us_guess)
        self._t0 = t0
        self._x0 = x0
        self._T = T
        self._N = N
        self._xs_guess = xs
        self._us_guess = us

    def run(self, T_sim: float=20, sampling_time: float=0.005,
            max_iters_mpc: int=3):
        """ Run MPC.

        Args:
            T_sim (float): Simulation time.
            sampling_time (float): Sampling time. OCP must be solved within \
                sampling time.
            max_iter (int): Maximum iteration number of each OCP.
        """
        pass            


