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
        self._initialized = False
    
    def set_log_directory(self, log_dir: str):
        """ Set directory path where data are logged.

        Args:
            log_dir (str): Log directory.
        """
        assert isinstance(log_dir, str)
        self._log_dir = log_dir

    def get_log_directory(self):
        """ Get directory path where data are logged.

        Returns:
            log_dir (str): Log directory.
        """
        return self._log_dir

    def init_mpc(self):
        """ Solve ocp once for getting solution guess.

        Args:

        Note:
            Do not change initial condition after this method is called.
        """
        self._solver.solve()

    def run(self, T_sim: float=20, sampling_time: float=0.005,
            max_iters_mpc: int=5, result=True, log=True, plot=True):
        """ Run MPC.

        Args:
            T_sim (float): Simulation time.
            sampling_time (float): Sampling time. OCP must be solved within \
                sampling time.
            mpc_max_iters (int): Maximum iteration number of OCP at each samping.
            result (bool): If true, summary of result is printed.
            log (bool): If true, results are logged to log_dir.
            plot (bool): If true, graphs are generated and saved.
        
        Returns:
            xs_real (numpy.ndarray): State History.
            us_real (numpy.ndarray): Control History.
            ts_real (numpy.ndarray): Time at each stage.
        """
        assert T_sim > 0
        assert sampling_time > 0

        t = self._t0
        x = self._x0.copy()
        f = self._f
        solver = self._solver


        self._solver.set_max_iters(max_iters_mpc)

        ts_real = np.arange(t, t + T_sim + sampling_time*1e-6, sampling_time)

        # record real trajectory of state and control.
        xs_real = []
        us_real = []

        total_time = 0.0
        total_noi = 0

        # MPC
        for t in ts_real:
            solver.set_initial_condition(t, x)
            solver.solve(warm_start=True, gamma_fixed=0.0)

            us_opt = solver.get_us_opt()
            result = solver.get_result()
            computation_time = result['computation_time']
            noi = result['noi']
            total_time += computation_time
            total_noi += noi

            # In MPC, we use initial value of optimal input trajectory.
            u = us_opt[0]
            x_next = self.update_state(f, x, u, t, sampling_time, 1e-3)

            # save
            xs_real.append(x)
            us_real.append(u)

            # for the next sampling time
            x = x_next

        # convert into numpy
        xs_real = np.array(xs_real, dtype=float)
        us_real = np.array(us_real, dtype=float)

        # average computation time
        ave_time = total_time / len(ts_real)
        ave_noi = total_noi / len(ts_real)
        
        if result:
           self.print_result(x, ave_time, ave_noi) 
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
    
    def print_result(self, x_final: np.ndarray,ave_noi: float, ave_time: float):
        """ Print result.
        
        Args:
            x_final (np.ndarray): Final state
            ave_time (float): average computation time.
            ave_noi (float): average number of iteration.
        """
        print('------------------- RESULT -------------------')
        print(f'solver: {self._solver._solver_name}')
        print(f'Final state: {x_final}')
        print(f'Average number of iterations: {ave_noi:.6f}')
        print(f'Average computation time: {ave_time:.6f} [s]')
        print('----------------------------------------------')

    @staticmethod
    def log_data(log_dir: str, xs: np.ndarray, us: np.ndarray, ts: np.ndarray):
        """ Log data.
        
        Args:
            log_dir (str): Directory where data are saved.
            xs (numpy.ndarray): optimal state trajectory. (N + 1) * n_x
            us (numpy.ndarray): optimal control trajectory. N * n_u
            ts (numpy.ndarray): Time at each stage.
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
            ts (numpy.ndarray): Time at each stage.
        """
        plotter = Plotter(log_dir, xs, us, ts)
        plotter.plot(save=True)
