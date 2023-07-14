import sympy as sym
import numpy as np
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
        self._ocp = ocp
        self._sim_name = ocp.get_ocp_name()
        self._log_dir = join(dirname(dirname(abspath(__file__))), 'log',
                             self._sim_name)
        pass

    def ocp(self):
        """ Returns OCP.
        """
        return self._ocp

    def set_log_directory(self, log_dir: str):
        """ set directory path of data are logged.
        """
        self._log_dir = log_dir

    def get_log_directory(self):
        """ get directory path of data are logged.
        """
        return self._log_dir

    @abc.abstractmethod
    def set_solver_parameters(self):
        """ Set solver parameters.
        """
        pass

    @abc.abstractmethod
    def reset_initial_conditions(self):
        """ Reset t0, x0, initial guess of us and so.
        """
        pass

    @abc.abstractmethod
    def solve(self):
        pass

class DDPSolver(SolverBase):
    """ Differential Dynamic Programming(DDP) solver.
    """
    def __init__(self, ocp: OCP):
        """ set optimal control problem.
        """
        super().__init__(ocp)
        # OCP
        self._ocp = ocp
        self._sim_name = ocp.get_ocp_name()
        self._log_dir = join(dirname(dirname(abspath(__file__))), 'log',
                             self._sim_name)
        # dimensions of state and input.
        self._n_x = ocp.get_n_x()
        self._n_u = ocp.get_n_u()
        # Horizon length, num of discretization, time step.
        self._T = ocp.get_T()
        self._N = ocp.get_N()
        self._dt = self._T / self._N
        # start time, x0 and initial guess of us
        self._t0 = ocp.get_t0()
        self._x0 = ocp.get_x0()
        self._us_guess = ocp.get_us_guess()
        # solver parameters
        self._max_iter = 200
        self._alphas = np.array([0.5**i for i in range(8)] + [0])
        self._damp_init = 1.0
        self._damp_min = 1e-5
        self._damp_max = 1e5      
        # functions derivatives.
        self._df, self._dl = ocp.get_derivatives()
        self._f, self._fx, self._fu, self._fxx, self._fux, self._fuu = self._df
        self._l, self._lx, self._lu, self._lxx, self._lux, self._luu, \
            self._lf, self._lfx, self._lfxx = self._dl

    def ocp(self):
        """ Returns OCP.
        """
        return self._ocp

    def get_log_directory(self):
        """ Returns directory path where logs are saved.
        """
        return self._log_dir
    
    def set_log_directory(self, log_dir: str):
        """ Set directory path where logs are saved
        """
        self._log_dir = log_dir

    def reset_initial_conditions(self, x0: np.ndarray, us_guess: np.ndarray,
                                 t0: float):
        """ reset t0, x0, and initial guess of us_guess.
        """
        self._x0 = np.ndarray(x0, dtype=float)
        self._us_guess = np.ndarray(us_guess, dtype=float)
        self._t0 = float(t0)
       
    def set_solver_parameters(self, max_iter: int=None, alphas: np.ndarray=None,
                              damp_init: float=None, damp_min: float=None, 
                              damp_max: float=None):
        """ Set solver parameters.

        Args:
            max_iter (int): Number of maximum iterations.
            alphas (np.ndarray): Line search steps.
            damp_init (float): Initial value of damp,\
                coefficient of regularization.
            damp_min (float): Minimum value of damp.
            damp_max (float): Maximum value of damp.
        """
        if max_iter is not None:
            self._max_iter = max_iter
        if alphas is not None:
            self._alphas = np.array(alphas)
        if damp_init is not None:
            self._damp_init = damp_init
        if damp_min is not None:
            self._damp_min = damp_min
        if damp_max is not None:
            self._damp_max = damp_max

    def solve(self, result=True , log=False):
        """ Solve OCP via DDP iteration.

        Args:
            result (bool): If true, result is printed.
            log (bool): If true, results are logged to log_dir.
        
        Returns:
            ts (numpy.ndarray): time history.
            xs (numpy.ndarray): optimal state trajectory. (N * n_x)
            us (numpy.ndarray): optimal control trajectory. (N * n_u)
            Js (numpy.ndarray): costs at each iteration.

        """
        max_iter = self._max_iter
        alphas = self._alphas
        damp_init = self._damp_init
        damp_min = self._damp_min
        damp_max = self._damp_max
        t0 = self._t0
        x0 = self._x0
        us = self._us_guess
        N = self._N
        T = self._T
        dt = self._dt
        # derivatives
        f, fx, fu, fxx, fux, fuu = self._df
        l, lx, lu, lxx, lux, luu, lf, lfx, lfxx = self._dl
        # success flag of solver.
        is_success = False        
        # computational time
        time_start = time.perf_counter()
        # initial rollout
        xs, J = self.rollout(f, l, lf, x0, us, t0, dt)
        Js = [J]
        # dumping coefficient of C-Newton.
        damp = damp_init
        # main iteration
        for iter in range(max_iter):
            print(f'iter: {iter}')
            # backward pass
            ks, Ks, Delta_V = self.backward_pass(
                fx, fu, fxx, fux, fuu, lx, lu, lxx, lux, luu, lfx, lfxx,
                xs, us, t0, dt, damp
            )
            if np.abs(Delta_V) < 1e-5:
                is_success = True
                break
            elif Delta_V > 0:
                damp *= 10.0
                continue
            print('DeltaV: ',Delta_V)
            # forward pass in line search 
            for alpha in alphas:
                xs_new, us_new, J_new = self.forward_pass(
                    f, l, lf, xs, us, t0, dt, ks, Ks, alpha
                )
                # print(f'iter: {iter}, alpha: {alpha}, J: {J}, J_new: {J_new}')
                if J_new < J:
                    # line search success
                    xs = xs_new
                    us = us_new
                    J = J_new
                    damp *= 0.5
                    break
            else:
                # line search failed
                damp *= 2.0
                damp = min(max(damp, damp_min), damp_max)
            Js.append(J)
        ts = np.array([i*self._dt for i in range(N + 1)])
        Js = np.array(Js)
        # computational time
        time_end = time.perf_counter()
        time_elapsed = time_end - time_start
        # results
        if result:
            self.print_result(is_success, iter, Js[-1], time_elapsed)
        # log
        if log:
            self.log_data(self._log_dir, ts, xs, us, Js)
        return ts, xs, us, Js

    def rollout(self, f, l, lf, x0: np.ndarray, us: np.ndarray,
                t0: float, dt: float):
        """ Rollout state trajectory from initial state and input trajectory,\
            with cost is calculated.

        Args:
            f (function): Discrete state equation. x_k+1 = f(x_k, u_k).
            l (function): Stage cost function.
            lf (function): Terminal cost function.
            x0 (np.ndarray): Initial state.
            us (np.ndarray): Input control trajectory.
            t0 (float): Initial time.
            dt (float): Discrete time step.
        """
        N = us.shape[0]
        # time, state trajectory and cost
        xs = np.zeros((N + 1, x0.shape[0]))
        xs[0] = x0
        J = 0.0
        for i in range(N):
            t = t0 + i*dt
            xs[i + 1] = f(xs[i], us[i], t)
            J += l(xs[i], us[i], t)
        t = t0 + i*N
        J += lf(xs[N], t)
        return xs, J

    def backward_pass(self, fx, fu, fxx, fux, fuu, 
                      lx, lu, lxx, lux, luu, lfx, lfxx,
                      xs: np.ndarray, us: np.ndarray, t0: float, dt: float,
                      damp: float=1e-6):
        """ backward pass of DDP.
        Args:
            fx (function):
            fu (function):
            fxx (function):
            fux (function):
            fuu (function):
            lx (function):
            lu (function):
            lxx (function):
            lux (function):
            luu (function):
            xs (numpy.ndarray): nominal state trajectory.\
                Size must be (N+1)*n_u.
            us (numpy.ndarray): nominalcontrol trajectory.\
                Size must be N*n_u.
            t0 (float): Initial time.
            dt (float): Discrete time step.
            damp (float): damping coefficient.
        Returns:
            ks (numpy.ndarray): series of k. Its size is N * n_u
            Ks (numpy.ndarray): series of K. Its size is N * (n_u * n_x)
            Delta_V (float): expecting change of value function at stage 0.
        """
        N = us.shape[0]
        T = N * dt
        n_x = xs.shape[1]
        n_u = us.shape[1]
        dt = T / N
        # feedforward term and feedback coeff.
        ks = np.empty((N, n_u))
        Ks = np.empty((N, n_u, n_x))
        # value function at stage i+1
        Vx = lfx(xs[N], t0 + T)
        Vxx = lfxx(xs[N], t0 + T)
        # expected cost cahnge of all stage
        delta_V = 0
        # daming matrix
        Reg = damp * np.eye(n_u)
        for i in range(N - 1, -1, -1):
            t = t0 + i*dt
            # x and u at satge i
            x, u = xs[i], us[i]
            # derivatives of stage i
            fx_i = fx(x, u, t)
            fu_i = fu(x, u, t)
            fxx_i = fxx(x, u, t)
            fux_i = fux(x, u, t)
            fuu_i = fuu(x, u, t)
            lx_i = lx(x, u, t)
            lu_i = lu(x, u, t)
            lxx_i = lxx(x, u, t)
            lux_i = lux(x, u, t)
            luu_i = luu(x, u, t)
            lfx_i = lfx(x, t)
            lfxx_i = lfxx(x, t)
            # action value derivatives
            Qx = lx_i + fx_i.T @ Vx
            Qu = lu_i + fu_i.T @ Vx
            Qxx = lxx_i + fx_i.T @ Vxx @ fx_i + Vx @ fxx_i.T
            Qux = lux_i + fu_i.T @ Vxx @ fx_i + Vx @ fux_i.T
            Quu = luu_i + fu_i.T @ Vxx @ fu_i + Vx @ fuu_i.T
            # feedforward and feedback terms
            Quu_inv = np.linalg.inv(Quu + Reg)
            k = -Quu_inv @ Qu
            K = -Quu_inv @ Qux
            ks[i] = k
            Ks[i] = K
            # value function of stage i, passed to i-1.
            delta_V_i = 0.5 * k.T @ Quu @ k + k.T @ Qu
            Vx = Qx - K.T @ Quu @ k
            Vxx = Qxx - K.T @ Quu @ K
            delta_V += delta_V_i
        return ks, Ks, delta_V

    def forward_pass(self, f, l, lf, xs: np.ndarray, us: np.ndarray,
                     t0: float, dt: float,
                     ks: np.ndarray, Ks: np.ndarray, alpha: float=1.0):
        """ forward pass of DDP.
        Args:
            f (function): state function.
            l (function): stage cost function.
            lf (function): terminal cost function.
            xs (numpy.ndarray): nominal state trajectory.\
                size must be (N+1)*n_u
            us (numpy.ndarray): nominal control trajectory.\
                size must be N*n_u
            t0 (float): Initial time.
            dt (float): Discrete time step.
            ks (numpy.ndarray): series of k. Size must be N * n_u.
            Ks (numpy.ndarray): series of K. Size must be N * (n_u * n_x).
            alpha (float): step size of line search. 0<= alpha <= 1.0.

        Returns:
            xs_new (numpy.ndarray): new state trajectory.
            us_new (numpy.ndarray): new control trajectory.
            J_new (float) cost along with (xs_new, us_new).
        """
        N = us.shape[0]
        T = N * dt
        # new (xs, us) and cost
        xs_new = np.empty(xs.shape)
        xs_new[0] = xs[0]
        us_new = np.empty(us.shape)
        J_new = 0.0
        for i in range(N):
            t = t0 + i*dt
            us_new[i] = us[i] + alpha * ks[i] + Ks[i] @ (xs_new[i] - xs[i])
            xs_new[i + 1] = f(xs_new[i], us_new[i], t)
            # debug
            try:
                J_new += l(xs_new[i], us_new[i], t)
            except:
                print('i:', i)
                print('t: ', t)
                print('alpha: ', alpha)
                # print('x[i-1], u[i-1]', xs_new[i-1], us_new[i-1])
                # print('x[i], u[i]', xs_new[i], us_new[i])
                # print('ks', ks)
                # print('Ks', Ks)
                print('xs_new', xs_new)
                print('us_new', us_new)
                print(l(xs_new[i], us_new[i], t))
                raise Exception('stop!')
        # terminal cost
        J_new += lf(xs_new[N], t + T)
        return xs_new, us_new, J_new

    @staticmethod
    def print_result(is_success: bool, iter: int, cost: float,
                     computational_time: float):
        print('--- RESULT ---')
        if is_success:
            status = 'success'
        else:
            status = 'failure'
        print(f'status: {status}')
        print(f'iteration: {iter}')
        print(f'cost value: {cost}')
        print(f'computational time: {computational_time}')
        print('--------------')
    
    @staticmethod
    def log_data(log_dir: str, ts: np.ndarray, xs: np.ndarray, us: np.ndarray,
                 Js: np.ndarray):
        logger = Logger(log_dir)
        logger.save(ts, xs, us, Js)
        # plot
        plotter = Plotter.from_log(log_dir)
        plotter.plot(save=True)
