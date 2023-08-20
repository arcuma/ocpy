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

        # dict containing result
        self._result_mpc = {}
        self._result_mpc['xs'] = np.ndarray(0)
        self._result_mpc['us'] = np.ndarray(0)
        self._result_mpc['ts'] = np.ndarray(0)
        self._result_mpc['noi_ave'] = None
        self._result_mpc['computation_time_ave'] = None

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
        
        Note:
            Do not change initial condition after this method is called.
        """
        self._solver.solve()
        self._initialized = True

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

            # In MPC, we use initial value of optimal input trajectory.
            us_opt = solver.get_us_opt()
            u = us_opt[0]
            x_next = self.update_state(f, x, u, t, sampling_time, 1e-3)

            # save
            xs_real.append(x)
            us_real.append(u)

            result = solver.get_result()
            computation_time = result['computation_time']
            noi = result['noi']
            total_time += computation_time
            total_noi += noi            

            # for the next sampling time
            x = x_next

        # convert into numpy
        xs_real = np.array(xs_real, dtype=float)
        us_real = np.array(us_real, dtype=float)

        # average computation time
        noi_ave = total_noi / len(ts_real)
        computation_time_ave = total_time / len(ts_real)

        self._result_mpc['xs'] = xs_real
        self._result_mpc['us'] = us_real
        self._result_mpc['ts'] = ts_real
        self._result_mpc['noi_ave'] = noi_ave
        self._result_mpc['computation_time_ave'] = computation_time_ave

        if result:
           self.print_result() 
        if log:
            self.log_data()
        if plot:
            self.plot_data(save=log)
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
    
    def print_result(self):
        """ Print result.
        """
        res = self._result_mpc
        noi_ave = res['noi_ave']
        computation_time_ave = res['computation_time_ave']

        print('------------------- RESULT -------------------')
        print(f'solver: {self._solver._solver_name}')
        print(f'Average number of iterations: {noi_ave:.6f}')
        print(f'Average computation time: {computation_time_ave:.6f} [s]')
        print('----------------------------------------------')

    def log_data(self,):
        """ Log data.
        """
        res = self._result_mpc
        xs = res['xs']
        us = res['us']
        ts = res['ts']

        logger = Logger(self._log_dir)
        logger.save(xs, us, ts)
    
    def plot_data(self, save=True):
        """ Plot data and save it.
        """
        res = self._result_mpc
        xs = res['xs']
        us = res['us']
        ts = res['ts']

        plotter = Plotter(self._log_dir, xs, us, ts)
        plotter.plot(save=save)

    def get_result(self):
        """ Get result.
        """
        return self._result_mpc
