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
        self._solver = solver
        self._ocp = solver.ocp()
        self._f = self._ocp.get_derivatives()[0][0]
    
    def init_mpc(self, t_start: float, x0: np.ndarray=None, us_guess=None,
                 max_iter=200, alphas=0.5**np.arange(8), damp_init=1e-3,
                 damp_min=1e-3, damp_max=1e3):
        """ Initialize mpc, it is, solve initial ocp at t_start.

        Args:
            t_start (float): Start time of simulation.
            x0 (np.ndarray): initial state.
            us_guess (np.ndarray): initial guess of control trajectory.
        
        Note:
            If x0 or us_guess are None, the value when ocp is set are used.
        """
        if x0 is None:
            x0 = self._ocp.get_x0()
        if us_guess is None:
            us_guess = self._ocp.get_us_guess()
        # solve ocp at initial time.
        ts, xs, us, Js = self._solver.solve(
            t_start, x0, us_guess, max_iter=200, alphas=0.5**np.arange(8), 
            damp_init=damp_init, damp_min=damp_min, damp_max=damp_max, log=False)
        self._t_start = t_start
        self._x0 = x0
        self._us_guess = us
        self._damp_init = damp_init
        self._damp_min = damp_min
        self._damp_max = damp_max
        self._init_done = True
        return ts, xs, us, Js

    def run(self, T_sim: float, sampling_time: float,
            max_iter: float=3):
        """ Run MPC

        Args:
            t_start (float): Start time.
            T_sim (float): Simulation time.
            sampling_time (float): Sampling time. OCP are to solved within \
                sampling time.
            max_iter (int): Maximum iteration number of each OCP.
        """
        assert self._init_done
        t_start = self._t_start
        t = t_start
        N_sim = int(T_sim / sampling_time)

        x0 = self._x0
        us_guess = self._us_guess
        # real state trajectory.
        x_hist = np.empty((N_sim + 1, len(x0))) 
        x_hist[0] = x0
        # real input trajectory.
        u_hist = np.empty((N_sim, us_guess.shape[1]))
        # sample time
        t_hist = np.array([t_start + i*sampling_time for i in range(N_sim + 1)])
        # MPC simulation
        for i in range(N_sim):
            t = t_start + i*sampling_time
            print('time: ', t)
            # solve OCP at time t.
            _, xs, us, Js = self._solver.solve(
                t, x_hist[i], us_guess, max_iter=max_iter, alphas=np.array([1.0, 0.5]),
                damp_init=self._damp_init, damp_min=self._damp_min,
                damp_max=self._damp_max, log=False)
            # update control input
            u_hist[i] = us[0]
            # update state
            x_hist[i + 1] = self._f(x_hist[i], u_hist[i], t)
            # warm start at t + dt
            us_guess = us
        # save
        self._t_hist = t_hist
        self._x_hist = x_hist
        self._u_hist = u_hist
        return t_hist, x_hist, u_hist

    def log(self, log_dir: str=None):
        if log_dir == None:
            log_dir = join(dirname(dirname(abspath(__file__))), 
                           'log_mpc',
                           self._ocp.get_ocp_name()
                           )
        logger = Logger(log_dir)
        logger.save(self._t_hist, self._x_hist, self._u_hist, np.array([]))

    def plot(self, log_dir: str=None):
        if log_dir == None:
            log_dir = join(dirname(dirname(abspath(__file__))), 
                           'log_mpc',
                           self._ocp.get_ocp_name()
                           )
        plotter = Plotter(log_dir, self._ocp.get_ocp_name(), self._t_hist,
                          self._x_hist, self._u_hist, np.ndarray([]))
        plotter.plot()


