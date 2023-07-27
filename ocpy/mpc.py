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
        
        Returns:
            xs_real (numpy.ndarray): optimal state trajectory. (N * n_x)
            us_real (numpy.ndarray): optimal control trajectory. (N * n_u)
            ts_real (numpy.ndarray): time history.

        """
        t = self._t0
        x = self._x0.copy()
        T = self._T
        N = self._N
        xs_guess = self._xs_guess.copy()
        us_guess = self._us_guess.copy()
        f = self._f
        assert T_sim > 0
        assert sampling_time > 0
        ts_real = np.arange(t, t + T_sim + sampling_time, sampling_time)
        N_sim = len(ts_real) - 1
        # xs_real = np.zeros((N_sim + 1, x.shape[0]))
        # us_real = np.zeros((N_sim, us_guess.shape[1]))
        xs_real = []
        us_real = []
        # MPC
        for t in ts_real:
            xs_opt, us_opt, *_ = self._solver.solve(
                t, x, N, T, xs_guess, us_guess, 1e-3
            )
            u = us_opt[0]
            xs_real.append(x)
            us_real.append(u)
            # simulation
            x_next =  x + f(x, u, t) * sampling_time
            x = x_next
        # convert into numpy
        xs_real = np.array(xs_real, dtype=float)
        us_real = np.array(us_real, dtype=float)
        return xs_real, us_real, ts_real
