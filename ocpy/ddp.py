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
from ocpy.solverbase import SolverBase


class DDPSolver(SolverBase):
    """ Differential Dynamic Programming(DDP) solver.
    """
    def __init__(self, ocp: OCP):
        """ Set optimal control problem.
        
        Args:
            ocp(ocpy.OCP): optimal control problem. 
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
        self._stop_threshold = 1e-3

        # functions derivatives.
        self._df, self._dl = ocp.get_derivatives()
        self._f, self._fx, self._fu, self._fxx, self._fux, self._fuu = self._df
        self._l, self._lx, self._lu, self._lxx, self._lux, self._luu, \
            self._lf, self._lfx, self._lfxx = self._dl
        
        # pseudo AOT
        DDPSolver.ddp(
            self._f, self._fx, self._fu, self._fxx, self._fux, self._fuu, 
            self._l, self._lx, self._lu, self._lxx, self._lux, self._luu, 
            self._lf, self._lfx, self._lfxx,
            self._t0, self._x0, self._us_guess, self._N, self._T, self._dt,
            1 , self._alphas, self._damp_init, self._damp_min, 
            self._damp_max, self._stop_threshold 
        )

    def ocp(self):
        """ Return OCP.
        """
        return self._ocp

    def get_log_directory(self):
        """ Return directory path where logs are saved.
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
                              damp_max: float=None, stop_threshold: float=None):
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
            self._alphas = np.array(alphas, dtype=float)
        if damp_init is not None:
            self._damp_init = damp_init
        if damp_min is not None:
            self._damp_min = damp_min
        if damp_max is not None:
            self._damp_max = damp_max
        if stop_threshold is not None:
            self._stop_threshold = stop_threshold

    def solve(self, result=True , log=False, plot=False):
        """ Solve OCP via DDP iteration.

        Args:
            result (bool): If true, summary of result is printed.
            log (bool): If true, results are logged to log_dir.
            plot (bool): If true, graphs are generated and saved.
        
        Returns:
            ts (numpy.ndarray): Discretized time history.
            xs (numpy.ndarray): Optimal state trajectory. (N * n_x)
            us (numpy.ndarray): Optimal control trajectory. (N * n_u)
            Js (numpy.ndarray): Costs at each iteration.
        """
        max_iter = self._max_iter
        alphas = self._alphas
        damp_init = self._damp_init
        damp_min = self._damp_min
        damp_max = self._damp_max
        stop_threshold = self._stop_threshold
        t0 = self._t0
        x0 = self._x0
        us = self._us_guess
        N = self._N
        T = self._T
        dt = self._dt
        # derivatives functions.
        f, fx, fu, fxx, fux, fuu = self._df
        l, lx, lu, lxx, lux, luu, lf, lfx, lfxx = self._dl
        # success flag of solver.
        is_success = False
        # computational time
        time_start = time.perf_counter()
        # solve
        ts, xs, us, Js, is_success = DDPSolver.ddp(
            f, fx, fu, fxx, fux, fuu, l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
            t0, x0, us, N, T, dt, max_iter, alphas, damp_init, damp_min, damp_max,
            stop_threshold 
        )
        # computational time
        time_end = time.perf_counter()
        time_elapsed = time_end - time_start
        # number of iterations
        iters = len(Js) - 1
        # results
        if result:
            self.print_result(is_success, iters, Js[-1], time_elapsed)
        # log
        if log:
            self.log_data(self._log_dir, ts, xs, us, Js)
        if plot:
            self.plot_data(self._log_dir, ts, xs, us, Js)
        return ts, xs, us, Js

    @staticmethod
    @numba.njit
    def ddp(f, fx, fu, fxx, fux, fuu, l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
            t0, x0, us, N, T, dt, max_iter, alphas, damp_init, damp_min, damp_max,
            stop_threshold):
        """ DDP algorithm.
        """
    # innner functions
        def rollout(f, l, lf, x0: np.ndarray, us: np.ndarray,
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
                xs[i + 1] = f(xs[i], us[i], t, dt)
                J += l(xs[i], us[i], t, dt)
            t = t0 + i*N
            J += lf(xs[N], t)
            return xs, J

        def vector_dot_tensor(Vx: np.ndarray, fab: np.ndarray):
            """ Tensor dot product between 1d vector and 3d tensor, contraction\
                with each 0 and 1 axis. This is used for product between Vx \
                (derivative of value function) and fqq (hessian of dynamics)
            
            Args:
                Vx (np.ndarray) : n_x-sized 1d array.
                fqq (np.ndarray) : (n_b*n_x*n_a)-sized 3d array.
            
            Returns:
                np.ndarray: (n_b*n_a)-sized 2d array.
            """
            n_b, n_x, n_a = fab.shape
            Vxfab = np.zeros((n_b, n_a))
            for i in range(n_b):
                for j in range(n_x):
                    for k in range(n_a):
                        Vxfab[i][k] += Vx[j] * fab[i][j][k]
            return Vxfab

        def backward_pass(fx, fu, fxx, fux, fuu, 
                          lx, lu, lxx, lux, luu, lfx, lfxx,
                          xs: np.ndarray, us: np.ndarray, t0: float, dt: float,
                          damp: float=1e-6):
            """ Backward pass of DDP.

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
                xs (numpy.ndarray): Nominal state trajectory.\
                    Size must be (N+1)*n_u.
                us (numpy.ndarray): Nominalcontrol trajectory.\
                    Size must be N*n_u.
                t0 (float): Initial time.
                dt (float): Discrete time step.
                damp (float): Damping coefficient.

            Returns:
                ks (numpy.ndarray): Series of k. Its size is N * n_u
                Ks (numpy.ndarray): Series of K. Its size is N * (n_u * n_x)
                Delta_V (float): Expecting change of value function at stage 0.
            """
            N = us.shape[0]
            T = N * dt
            n_x = xs.shape[1]
            n_u = us.shape[1]
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
                fx_i = fx(x, u, t, dt)
                fu_i = fu(x, u, t, dt)
                fxx_i = fxx(x, u, t, dt)
                fux_i = fux(x, u, t, dt)
                fuu_i = fuu(x, u, t, dt)
                lx_i = lx(x, u, t, dt)
                lu_i = lu(x, u, t, dt)
                lxx_i = lxx(x, u, t, dt)
                lux_i = lux(x, u, t, dt)
                luu_i = luu(x, u, t, dt)
                lfx_i = lfx(x, t)
                lfxx_i = lfxx(x, t)
                # action value derivatives
                Qx = lx_i + fx_i.T @ Vx
                Qu = lu_i + fu_i.T @ Vx
                Qxx = lxx_i + fx_i.T @ Vxx @ fx_i + vector_dot_tensor(Vx, fxx_i).T
                Qux = lux_i + fu_i.T @ Vxx @ fx_i + vector_dot_tensor(Vx, fux_i).T
                Quu = luu_i + fu_i.T @ Vxx @ fu_i + vector_dot_tensor(Vx, fuu_i).T
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
        
        def forward_pass(f, l, lf, xs: np.ndarray, us: np.ndarray,
                         t0: float, dt: float,
                         ks: np.ndarray, Ks: np.ndarray, alpha: float=1.0):
            """ Forward pass of DDP.

            Args:
                f (function): State function.
                l (function): Stage cost function.
                lf (function): Terminal cost function.
                xs (numpy.ndarray): Nominal state trajectory.\
                    size must be (N+1)*n_u
                us (numpy.ndarray): Nominal control trajectory.\
                    size must be N*n_u
                t0 (float): Initial time.
                dt (float): Discrete time step.
                ks (numpy.ndarray): Series of k. Size must be N * n_u.
                Ks (numpy.ndarray): Series of K. Size must be N * (n_u * n_x).
                alpha (float): step Size of line search. 0<= alpha <= 1.0.

            Returns:
                xs_new (numpy.ndarray): New state trajectory.
                us_new (numpy.ndarray): New control trajectory.
                J_new (float): Cost along with (xs_new, us_new).
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
                xs_new[i + 1] = f(xs_new[i], us_new[i], t, dt)
                J_new += l(xs_new[i], us_new[i], t, dt)
            # terminal cost
            J_new += lf(xs_new[N], t + T)
            return xs_new, us_new, J_new
    # innner functions end
        # flag
        is_success = False        
        xs, J = rollout(f, l, lf, x0, us, t0, dt)
        Js = np.zeros(max_iter + 1, dtype=float)
        Js[0] = J
        # dumping coefficient of C-Newton.
        damp = damp_init
        # main iteration
        for iters in range(max_iter):
            # print(f'iters: {iters}')
            # backward pass
            ks, Ks, Delta_V = backward_pass(
                fx, fu, fxx, fux, fuu, lx, lu, lxx, lux, luu, lfx, lfxx,
                xs, us, t0, dt, damp
            )
            # print('DeltaV: ',Delta_V)      
            if np.abs(Delta_V) < stop_threshold:
                is_success = True
                break
            elif Delta_V > 0:
                # it's no use line searching
                damp *= 10.0
                Js[iters + 1] = J
                continue
            # forward pass in line search 
            for alpha in alphas:
                xs_new, us_new, J_new = forward_pass(
                    f, l, lf, xs, us, t0, dt, ks, Ks, alpha
                )
                # print(f'iters: {iters}, alpha: {alpha}, J: {J}, J_new: {J_new}')
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
            Js[iters + 1] = J
        ts = np.array([i*dt for i in range(N + 1)])
        Js = Js[0:iters + 1]
        return ts, xs, us, Js, is_success
    
    @staticmethod
    def print_result(is_success: bool, iters: int, cost: float,
                     computational_time: float):
        """ Print summary of result.
        
        Args:
            is_success (bool): Flag of success or failure.
            iters (int): Number of iterations.
            computational_time (float): total computational time.
        """
        print('--- RESULT ---')
        if is_success:
            status = 'success'
        else:
            status = 'failure'
        print(f'status: {status}')
        print(f'iteration: {iters}')
        print(f'cost value: {cost}')
        print(f'computational time: {computational_time} [s]')
        print(f'per update : {computational_time / iters} [s]')
        print('--------------')
    
    @staticmethod
    def log_data(log_dir: str, ts: np.ndarray, xs: np.ndarray, us: np.ndarray,
                 Js: np.ndarray):
        """ Log data.
        
        Args:
            log_dir (str): Directory where data are saved.
            ts (numpy.ndarray): time history.
            xs (numpy.ndarray): optimal state trajectory. (N * n_x)
            us (numpy.ndarray): optimal control trajectory. (N * n_u)
            Js (numpy.ndarray): costs at each iteration.
        """
        logger = Logger(log_dir)
        logger.save(ts, xs, us, Js)

    @staticmethod
    def plot_data(log_dir: str, ts: np.ndarray, xs: np.ndarray, us: np.ndarray,
                  Js: np.ndarray):
        """ Ulot data and save it.
        
        Args:
            log_dir (str): Directory where data are saved.
            ts (numpy.ndarray): time history.
            xs (numpy.ndarray): optimal state trajectory. (N * n_x)
            us (numpy.ndarray): optimal control trajectory. (N * n_u)
            Js (numpy.ndarray): costs at each iteration.
        """
        plotter = Plotter(log_dir, ts, xs, us, Js)
        plotter.plot(save=True)


class iLQRSolver(SolverBase):
    """ Iterative Linear Quadratic Regulator (iLQR) solver.
    """

    def __init__(self, ocp: OCP):
        """ Set optimal control problem.
        
        Args:
            ocp(ocpy.OCP): optimal control problem. 
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
        self._stop_threshold = 1e-3

        # functions derivatives.
        self._df, self._dl = ocp.get_derivatives()
        self._f, self._fx, self._fu, self._fxx, self._fux, self._fuu = self._df
        self._l, self._lx, self._lu, self._lxx, self._lux, self._luu, \
            self._lf, self._lfx, self._lfxx = self._dl
        
        # pseudo AOT
        iLQRSolver.ilqr(
            self._f, self._fx, self._fu, 
            self._l, self._lx, self._lu, self._lxx, self._lux, self._luu, 
            self._lf, self._lfx, self._lfxx,
            self._t0, self._x0, self._us_guess, self._N, self._T, self._dt,
            1 , self._alphas, self._damp_init, self._damp_min, 
            self._damp_max, self._stop_threshold 
        )

    def ocp(self):
        """ Return OCP.
        """
        return self._ocp

    def get_log_directory(self):
        """ Return directory path where logs are saved.
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
                              damp_max: float=None, stop_threshold: float=None):
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
            self._alphas = np.array(alphas, dtype=float)
        if damp_init is not None:
            self._damp_init = damp_init
        if damp_min is not None:
            self._damp_min = damp_min
        if damp_max is not None:
            self._damp_max = damp_max
        if stop_threshold is not None:
            self._stop_threshold = stop_threshold
    def solve(self, result=True , log=False, plot=False):
        """ Solve OCP via iLQR iteration.

        Args:
            result (bool): If true, summary of result is printed.
            log (bool): If true, results are logged to log_dir.
            plot (bool): If true, graphs are generated and saved.
        
        Returns:
            ts (numpy.ndarray): Discretized time history.
            xs (numpy.ndarray): Optimal state trajectory. (N * n_x)
            us (numpy.ndarray): Optimal control trajectory. (N * n_u)
            Js (numpy.ndarray): Costs at each iteration.
        """
        max_iter = self._max_iter
        alphas = self._alphas
        damp_init = self._damp_init
        damp_min = self._damp_min
        damp_max = self._damp_max
        stop_threshold = self._stop_threshold
        t0 = self._t0
        x0 = self._x0
        us = self._us_guess
        N = self._N
        T = self._T
        dt = self._dt
        # derivatives functions.
        f, fx, fu, _, _, _ = self._df
        l, lx, lu, lxx, lux, luu, lf, lfx, lfxx = self._dl
        # success flag of solver.
        is_success = False
        # computational time
        time_start = time.perf_counter()
        # solve
        ts, xs, us, Js, is_success = iLQRSolver.ilqr(
            f, fx, fu, l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
            t0, x0, us, N, T, dt, max_iter, alphas, damp_init, damp_min, damp_max,
            stop_threshold 
        )
        # computational time
        time_end = time.perf_counter()
        time_elapsed = time_end - time_start
        # number of iterations
        iters = len(Js) - 1
        # results
        if result:
            self.print_result(is_success, iters, Js[-1], time_elapsed)
        # log
        if log:
            self.log_data(self._log_dir, ts, xs, us, Js)
        if plot:
            self.plot_data(self._log_dir, ts, xs, us, Js)
        return ts, xs, us, Js

    @staticmethod
    @numba.njit
    def ilqr(f, fx, fu, l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
             t0, x0, us, N, T, dt, max_iter, alphas, damp_init, damp_min, damp_max,
             stop_threshold):
        """ iLQR algorithm.
        """
    # innner functions
        def rollout(f, l, lf, x0: np.ndarray, us: np.ndarray,
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
                xs[i + 1] = f(xs[i], us[i], t, dt)
                J += l(xs[i], us[i], t, dt)
            t = t0 + i*N
            J += lf(xs[N], t)
            return xs, J

        def backward_pass(fx, fu, 
                          lx, lu, lxx, lux, luu, lfx, lfxx,
                          xs: np.ndarray, us: np.ndarray, t0: float, dt: float,
                          damp: float=1e-6):
            """ Backward pass of iLQR.

            Args:
                fx (function):
                fu (function):
                lx (function):
                lu (function):
                lxx (function):
                lux (function):
                luu (function):
                xs (numpy.ndarray): Nominal state trajectory.\
                    Size must be (N+1)*n_u.
                us (numpy.ndarray): Nominalcontrol trajectory.\
                    Size must be N*n_u.
                t0 (float): Initial time.
                dt (float): Discrete time step.
                damp (float): Damping coefficient.

            Returns:
                ks (numpy.ndarray): Series of k. Its size is N * n_u
                Ks (numpy.ndarray): Series of K. Its size is N * (n_u * n_x)
                Delta_V (float): Expecting change of value function at stage 0.
            """
            N = us.shape[0]
            T = N * dt
            n_x = xs.shape[1]
            n_u = us.shape[1]
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
                fx_i = fx(x, u, t, dt)
                fu_i = fu(x, u, t, dt)
                lx_i = lx(x, u, t, dt)
                lu_i = lu(x, u, t, dt)
                lxx_i = lxx(x, u, t, dt)
                lux_i = lux(x, u, t, dt)
                luu_i = luu(x, u, t, dt)
                lfx_i = lfx(x, t)
                lfxx_i = lfxx(x, t)
                # action value derivatives
                Qx = lx_i + fx_i.T @ Vx
                Qu = lu_i + fu_i.T @ Vx
                Qxx = (lxx_i + fx_i.T @ Vxx @ fx_i)#.reshape((n_x, n_x))
                Qux = (lux_i + fu_i.T @ Vxx @ fx_i)#.reshape((n_u, n_x))
                Quu = (luu_i + fu_i.T @ Vxx @ fu_i)#.reshape((n_u, n_u))
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
        
        def forward_pass(f, l, lf, xs: np.ndarray, us: np.ndarray,
                        t0: float, dt: float,
                        ks: np.ndarray, Ks: np.ndarray, alpha: float=1.0):
            """ Forward pass of iLQR.

            Args:
                f (function): State function.
                l (function): Stage cost function.
                lf (function): Terminal cost function.
                xs (numpy.ndarray): Nominal state trajectory.\
                    size must be (N+1)*n_u
                us (numpy.ndarray): Nominal control trajectory.\
                    size must be N*n_u
                t0 (float): Initial time.
                dt (float): Discrete time step.
                ks (numpy.ndarray): Series of k. Size must be N * n_u.
                Ks (numpy.ndarray): Series of K. Size must be N * (n_u * n_x).
                alpha (float): step Size of line search. 0<= alpha <= 1.0.

            Returns:
                xs_new (numpy.ndarray): New state trajectory.
                us_new (numpy.ndarray): New control trajectory.
                J_new (float): Cost along with (xs_new, us_new).
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
                xs_new[i + 1] = f(xs_new[i], us_new[i], t, dt)
                J_new += l(xs_new[i], us_new[i], t, dt)
            # terminal cost
            J_new += lf(xs_new[N], t + T)
            return xs_new, us_new, J_new
    # innner functions end
        # flag
        is_success = False        
        xs, J = rollout(f, l, lf, x0, us, t0, dt)
        Js = np.zeros(max_iter + 1, dtype=float)
        Js[0] = J
        # dumping coefficient of C-Newton.
        damp = damp_init
        # main iteration
        for iters in range(max_iter):
            # print(f'iters: {iters}')
            # backward pass
            ks, Ks, Delta_V = backward_pass(
                fx, fu, lx, lu, lxx, lux, luu, lfx, lfxx,
                xs, us, t0, dt, damp
            )
            # print('DeltaV: ',Delta_V)      
            if np.abs(Delta_V) < stop_threshold:
                is_success = True
                break
            elif Delta_V > 0:
                # it's no use line searching
                damp *= 10.0
                Js[iters + 1] = J
                continue
            # forward pass in line search 
            for alpha in alphas:
                xs_new, us_new, J_new = forward_pass(
                    f, l, lf, xs, us, t0, dt, ks, Ks, alpha
                )
                # print(f'iters: {iters}, alpha: {alpha}, J: {J}, J_new: {J_new}')
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
            Js[iters + 1] = J
        ts = np.array([i*dt for i in range(N + 1)])
        Js = Js[0:iters + 1]
        return ts, xs, us, Js, is_success
    
    @staticmethod
    def print_result(is_success: bool, iters: int, cost: float,
                     computational_time: float):
        """ Print summary of result.
        
        Args:
            is_success (bool): Flag of success or failure.
            iters (int): Number of iterations.
            computational_time (float): total computational time.
        """
        print('--- RESULT ---')
        if is_success:
            status = 'success'
        else:
            status = 'failure'
        print(f'status: {status}')
        print(f'iteration: {iters}')
        print(f'cost value: {cost}')
        print(f'computational time: {computational_time} [s]')
        print(f'per update : {computational_time / iters} [s]')
        print('--------------')
    
    @staticmethod
    def log_data(log_dir: str, ts: np.ndarray, xs: np.ndarray, us: np.ndarray,
                 Js: np.ndarray):
        """ Log data.
        
        Args:
            log_dir (str): Directory where data are saved.
            ts (numpy.ndarray): time history.
            xs (numpy.ndarray): optimal state trajectory. (N * n_x)
            us (numpy.ndarray): optimal control trajectory. (N * n_u)
            Js (numpy.ndarray): costs at each iteration.
        """
        logger = Logger(log_dir)
        logger.save(ts, xs, us, Js)

    @staticmethod
    def plot_data(log_dir: str, ts: np.ndarray, xs: np.ndarray, us: np.ndarray,
                  Js: np.ndarray):
        """ Ulot data and save it.
        
        Args:
            log_dir (str): Directory where data are saved.
            ts (numpy.ndarray): time history.
            xs (numpy.ndarray): optimal state trajectory. (N * n_x)
            us (numpy.ndarray): optimal control trajectory. (N * n_u)
            Js (numpy.ndarray): costs at each iteration.
        """
        plotter = Plotter(log_dir, ts, xs, us, Js)
        plotter.plot(save=True)
