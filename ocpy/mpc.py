### WIP

import sympy as sym
import numpy as np
import numba
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
        """ Constructor.

        Args:
            solver (SolberBase): Solver instance.
        """
        self._solver = solver
        self._ocp = solver.ocp()
        self._sim_name = self._ocp.get_ocp_name()
        self._log_dir = join(dirname(dirname(abspath(__file__))),
                             'log_mpc', self._sim_name)
        self._f = self._ocp.get_df()[0]
        self._t0 = self._ocp.get_t0()
        self._x0 = self._ocp.get_x0()
        self._us_guess = self._ocp.get_us_guess()
        self._xs_guess = self._ocp.get_xs_guess()
        self._initialized = False
    
    def get_log_directory(self):
        return self._log_dir

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
        xs, us, *_ = self._solver.solve(t0, x0, T, N, xs_guess, us_guess)
        self._t0 = t0
        self._x0 = x0
        self._T = T
        self._N = N
        self._xs_guess = xs
        self._us_guess = us

    def run(self, T_sim: float=20, sampling_time: float=0.005,
            max_iters_mpc: int=5, result=True, log=True, plot=True):
        """ Run MPC.

        Args:
            T_sim (float): Simulation time.
            sampling_time (float): Sampling time. OCP must be solved within \
                sampling time.
            mpc_max_iters (int): Maximum iteration number of OCP at each samping.
        
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
        self._solver.set_max_iters(max_iters_mpc)
        ts_real = np.arange(t, t + T_sim + sampling_time*1e-6, sampling_time)
        # record real trajectory of state and control.
        xs_real = []
        us_real = []
        total_ctime = 0.0
        # MPC
        for t in ts_real:
            xs_opt, us_opt, *_, ctime, _ = self._solver.solve(
                t, x, T, N, xs_guess, us_guess
            )
            # In MPC, it uses initial value of optimal input trajectory.
            u = us_opt[0]
            x_next = self.update_state(f, x, u, t, sampling_time)
            # save
            xs_real.append(x)
            us_real.append(u)
            total_ctime += ctime
            # for the next sampling time
            x = x_next
            xs_guess = xs_opt
            us_guess = us_opt
        # convert into numpy
        xs_real = np.array(xs_real, dtype=float)
        us_real = np.array(us_real, dtype=float)
        # average computational time
        ave_ctime = total_ctime / len(ts_real)
        if result:
           self.print_result(self._solver._solver_name, x, ave_ctime) 
        if log:
            self.log_data(self._log_dir, xs_real, us_real, ts_real)
        if plot:
            self.plot_data(self._log_dir, xs_real, us_real, ts_real)
        return xs_real, us_real, ts_real
    
    @staticmethod
    @numba.njit
    def update_state(
            f, x: np.ndarray, u: np.ndarray, t: float,
            sampling_time: float, precision: float=None
        ):
        """ Simulation.

        f (function): RHS of state equation. 
        x (np.ndarray): State. 
        u (np.ndarray): Control input. 
        t (float): Time.
        sampling_time (float): Sampling time.
        precision (float=None):
        """
        if precision is None:
            x_next = x + f(x, u, t) * sampling_time
        else:
            x_next = x.copy()
            q = int(sampling_time // precision)
            r = sampling_time % precision
            for i in range(q):
                x_next = x + f(x, u, t + i*precision) * precision
                x = x_next
            x_next = x + f(x, u, t + q*precision) * r
        return x_next
    
    @staticmethod
    def print_result(solver_name: str, x: np.ndarray, ave_ctime: float):
        """ Print result.
        
        Args:
            is_success (bool): Flag of success or failure.
            iters (int): Number of iterations.
            computational_time (float): total computational time.
        """
        print('------------------- RESULT -------------------')
        print(f'Final state: {x}')
        print(f'Average computational time: {ave_ctime} [s]')
        print('----------------------------------------------')

    @staticmethod
    def log_data(log_dir: str, xs: np.ndarray, us: np.ndarray, ts: np.ndarray):
        """ Log data.
        
        Args:
            log_dir (str): Directory where data are saved.
            xs (numpy.ndarray): optimal state trajectory. (N + 1) * n_x
            us (numpy.ndarray): optimal control trajectory. N * n_u
            ts (numpy.ndarray): time history.
            Js (numpy.ndarray): costs at each iteration.
        """
        logger = Logger(log_dir)
        logger.save(xs, us, ts)
    
    @staticmethod
    def plot_data(log_dir: str, xs: np.ndarray, us: np.ndarray, ts: np.ndarray):
        """ Plot data and save it.
        
        Args:
            log_dir (str): Directory where data are saved.
            xs (numpy.ndarray): optimal state trajectory. (N + 1) * n_x
            us (numpy.ndarray): optimal control trajectory. N * n_u
            ts (numpy.ndarray): time history.
            Js (numpy.ndarray): costs at each iteration.
        """
        plotter = Plotter(log_dir, xs, us, ts)
        plotter.plot(save=True)
