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
        self._result_mpc['noi_hist'] = np.ndarray(0)
        self._result_mpc['noi_ave'] = None
        self._result_mpc['computation_time_hist'] = np.ndarray(0)
        self._result_mpc['computation_time_ave'] = None

        self._enable_line_search = None
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

    def init_mpc(self, max_iters_init=10, save_result=True):
        """ Solve ocp once for getting solution guess.

        Args:
            enable_line_search (bool): If True, enable line search in the solver.
        
        Note:
            Do not change initial condition after this method is called.
        """
        max_iters_eva = self._solver._max_iters
        self._solver._max_iters = max_iters_init
        self._solver.solve(save_result=save_result)
        self._solver._max_iters = max_iters_eva
        self._initialized = True

    def run(self, T_sim: float=20, sampling_time: float=0.005,
            feedback_delay=False,
            result=True, log=True, plot=True):
        """ Run MPC.

        Args:
            T_sim (float): Simulation time.
            sampling_time (float): Sampling time. OCP must be solved within \
                sampling time.
            mpc_max_iters (int): Maximum iteration number of OCP at each samping.
            feedback_delay (bool=False): If True, input is delayed for the \
                sampling time.
            result (bool): If true, summary of result is printed.
            log (bool): If true, results are logged to log_dir.
            plot (bool): If true, graphs are generated and saved.
        
        Returns:
            xs_real (np.ndarray): State History.
            us_real (np.ndarray): Control History.
            ts_real (np.ndarray): Time at each stage.
        """
        assert T_sim > 0
        assert sampling_time > 0

        t = self._t0
        x = self._x0.copy()
        f = self._f
        solver = self._solver

        ts_real = np.arange(t, t + T_sim + sampling_time*1e-6, sampling_time)

        # record real trajectory of state and control.
        xs_real = []
        us_real = []

        # history of NOI
        noi_hist = []
        computation_time_hist = []

        # for feedback delay
        u = solver._us_guess[0, :]

        # MPC
        for i, t in enumerate(ts_real):

            if feedback_delay:
                x_next = self.RK4(f, x, u, t, sampling_time)
                # save
                xs_real.append(x)
                us_real.append(u)

            solver.set_initial_condition(t, x)

            solver.solve(from_opt=True)

            # In MPC, we use initial value of optimal input trajectory.
            us_opt = solver.get_us_opt()
            u = us_opt[0]

            if not feedback_delay:
                x_next = self.RK4(f, x, u, t, sampling_time)
                # save
                xs_real.append(x)
                us_real.append(u)

            # get result
            result = solver.get_result()
            noi = result['noi']
            computation_time = result['computation_time']
            noi_hist.append(noi)
            computation_time_hist.append(computation_time)

            # update current state
            x = x_next

        # convert into numpy
        xs_real = np.array(xs_real, dtype=float)
        us_real = np.array(us_real, dtype=float)

        noi_hist = np.array(noi_hist, dtype=int)
        computation_time_hist = np.array(computation_time_hist, dtype=float)

        self._result_mpc['xs'] = xs_real
        self._result_mpc['us'] = us_real
        self._result_mpc['ts'] = ts_real
        self._result_mpc['noi_hist'] = noi_hist
        self._result_mpc['noi_ave'] = np.mean(noi_hist)
        self._result_mpc['computation_time_hist'] = computation_time_hist
        self._result_mpc['computation_time_ave'] = np.mean(computation_time_hist)

        if result:
           self.print_result() 
        if log:
            self.log_data()
        if plot:
            self.plot_data(save=log)
        return xs_real, us_real, ts_real
    
    @staticmethod
    def forward_euler(f, x: np.ndarray, u: np.ndarray, t:float, dt:float):
        return x + f(x, u, t) * dt

    @staticmethod
    def RK4(f, x: np.ndarray, u: np.ndarray, t:float, dt:float):
        k1 = f(x, u, t)
        k2 = f(x + 0.5*dt*k1, u, t + 0.5*dt)
        k3 = f(x + 0.5*dt*k2, u, t + 0.5*dt)
        k4 = f(x + dt*k3, u, t + dt)
        x_next = x + (k1 + 2*k2 + 2*k3 + k4)*dt/6
        
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
