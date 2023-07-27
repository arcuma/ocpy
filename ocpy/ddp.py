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
    def __init__(self, ocp: OCP, init=True):
        """ Set optimal control problem.
        
        Args:
            ocp (OCP): optimal control problem. 
            init (bool=True): If true, call init_solver(). It may take time.
        """
        super().__init__(ocp)
        self._solver_name = 'DDP'
        if init:
            self.init_solver()

    def set_stop_threshold(self, stop_threshold: float=None):
        """ Set stop creteria. If expectation of cost reduction is less than this, \
            iteration stops.

        Args:
            stop_threshold (float): Stop creterion
        """
        if stop_threshold is not None:
            self._stop_threshold = stop_threshold

    def set_solver_parameters(
            self, gamma_ini: float=None, rho_gamma: float=None,
            gamma_min: float=None,  gamma_max: float=None, alphas: np.ndarray=None, 
            max_iters: int=None, stop_threshold: float=None
        ):
        """ Set solver parameters.

        Args:
            gamma_ini (float): Initial value of damping coefficient.
            rho_gamma (float): Increasing/decreasing factor of gamma. (>1)
            gamma_min (float): Minimum value of damp.
            gamma_max (float): Maximum value of damp.
            alphas (np.ndarray): Line search steps.
            max_iters (int): Number of maximum iterations.
            stop_threshold (float): Stop criterion.
        """
        self.set_damping_coefficient(gamma_ini, rho_gamma, gamma_min, gamma_max)
        self.set_alphas(alphas)
        self.set_max_iters(max_iters)
        self.set_stop_threshold(stop_threshold)

    def init_solver(self):
        """ Initialize solver. Call once before you first call solve().
        """
        print("Initializing solver...")
        # caching
        self._solve(
            *self._df, *self._dl, 
            t0=self._t0, x0=self._x0, N=self._N, T=self._T,
            us_guess=self._us_guess, 
            gamma_ini=self._gamma_ini, rho_gamma=self._rho_gamma,
            gamma_min=self._gamma_min, gamma_max=self._gamma_max, 
            alphas=self._alphas,
            max_iters=10, stop_threshold=self._stop_threshold
        )
        print("Initialization Done.")

    def solve(
            self, 
            t0: float=None, x0: np.ndarray=None, T: float=None, N: int=None,
            xs_guess: np.ndarray=None ,us_guess: np.ndarray=None,
            gamma_fixed: float=None, enable_line_search: bool=True,
            result: bool=False, log: bool=False, plot: bool=False
        ):
        """ Solve OCP via DDP iteration.

        Args:
            t0 (float): Initial time.
            x0 (numpy.array): Initial state. Size must be n_x.
            T (float): Horizon length.
            N (int): Discretization grid number.
            xs_guess (numpy.array): Guess of state trajectory. \
                Size must be (N+1)*n_x. Only used in multiple-shooting.
            us_guess (numpy.array): Guess of input trajectory. \
                Size must be N*n_u.
            gamma_fixed (float): If set, damping coefficient is fixed.
            enable_line_search (bool=True): If true, enable line searching.
            result (bool): If true, summary of result is printed.
            log (bool): If true, results are logged to log_dir.
            plot (bool): If true, graphs are generated and saved.
        
        Returns:
            xs (numpy.ndarray): optimal state trajectory. (N + 1) * n_x
            us (numpy.ndarray): optimal control trajectory. N * n_u
            ts (numpy.ndarray): Discretized time history.
            Js (numpy.ndarray): Costs at each iteration.
            time_elapsed (float): Computational time.
            is_success (bool): Success or not.
        """
        if t0 is None:
            t0 = self._t0
        if x0 is None:
            x0 = self._x0
        assert x0.shape[0] == self._n_x
        if T is None:
            T = self._T        
        if N is None:
            N = self._N
        if xs_guess is None:
            xs_guess = self._xs_guess
        assert xs_guess.shape == (N + 1, self._n_x)
        if us_guess is None:
            us_guess = self._us_guess
        assert us_guess.shape == (N, self._n_u)
        if gamma_fixed is None:
            gamma_ini = self._gamma_ini
            rho_gamma = self._rho_gamma
            gamma_min = self._gamma_min
            gamma_max = self._gamma_max
        else:
            gamma_ini =  gamma_min = gamma_max = gamma_fixed
            rho_gamma = 1.0
        if enable_line_search:
            alphas = self._alphas
        else:
            alphas = np.array([1.0])
        max_iters = self._max_iters
        stop_threshold = self._stop_threshold
        # derivatives functions.
        f, fx, fu, fxx, fux, fuu = self._df
        l, lx, lu, lxx, lux, luu, lf, lfx, lfxx = self._dl
        # success flag of solver.
        is_success = False
        # computational time
        time_start = time.perf_counter()
        # solve
        xs, us, ts, Js, is_success = self._solve(
            f, fx, fu, fxx, fux, fuu, l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
            t0, x0, T, N, us_guess, gamma_ini, rho_gamma, gamma_min, gamma_max,
            alphas, max_iters, stop_threshold
        )
        # computational time
        time_end = time.perf_counter()
        time_elapsed = time_end - time_start
        # number of iterations
        iters = len(Js) - 1
        # results
        if result:
            self.print_result(self._solver_name, is_success, iters, Js[-1],
                              time_elapsed)
        # log
        if log:
            self.log_data(self._log_dir, xs, us, ts, Js)
        # plot
        if plot:
            self.plot_data(self._log_dir, xs, us, ts, Js)
        return xs, us, ts, Js, time_elapsed, is_success

    @staticmethod
    @numba.njit
    def _solve(
            f, fx, fu, fxx, fux, fuu, l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
            t0, x0, T, N, us_guess, gamma_ini, rho_gamma, gamma_min, gamma_max,
            alphas, max_iters, stop_threshold
        ):
        """ DDP algorithm.
        """
    ##### INNER FUNCTIONS #####
        def rollout(f, l, lf, x0: np.ndarray, us: np.ndarray, t0: float, dt: float):
            """ Rollout state trajectory from initial state and input trajectory,\
                with cost is calculated.

            Args:
                f (function): State equation.
                l (function): Stage cost function.
                lf (function): Terminal cost function.
                t0 (float): Initial time.                
                x0 (np.ndarray): Initial state.
                us (np.ndarray): Input control trajectory.
                dt (float): Discrete time step.
            """
            N = us.shape[0]
            xs = np.zeros((N + 1, x0.shape[0]))
            xs[0] = x0
            J = 0.0
            for i in range(N):
                xs[i + 1] = xs[i] + f(xs[i], us[i], t0 + i*dt) * dt
                J += l(xs[i], us[i], t0 + i*dt) * dt
            J += lf(xs[N], t0 + i*N)
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
                          gamma: float=1e-6):
            """ Backward pass of DDP.

            Args:
                fx (function): Derivative of f w.r.t. state x.
                fu (function): Derivative of f w.r.t. state u.
                fxx (function): Derivative of f w.r.t. state x and x.
                fux (function): Derivative of f w.r.t. state x and u.
                fuu (function): Derivative of f w.r.t. state u and u.
                lx (function): Derivative of l w.r.t. state x.
                lu (function): Derivative of l w.r.t. state u.
                lxx (function): Derivative of l w.r.t. state x and x.
                lux (function): Derivative of l w.r.t. state u and x.
                luu (function): Derivative of l w.r.t. state u and u.
                lfx (function): Derivative of lf w.r.t. state x.
                lfxx (function): Derivative of l w.r.t. state x and x.
                xs (numpy.ndarray): Nominal state trajectory.\
                    Size must be (N+1)*n_u.
                us (numpy.ndarray): Nominalcontrol trajectory.\
                    Size must be N*n_u.
                t0 (float): Initial time.
                dt (float): Discrete time step.
                gamma (float): Damping coefficient.

            Returns:
                ks (numpy.ndarray): Series of k. Its size is N * n_u
                Ks (numpy.ndarray): Series of K. Its size is N * (n_u * n_x)
                Delta_V (float): Expecting change of value function at stage 0.
            """
            N = us.shape[0]
            T = N * dt
            n_x = xs.shape[1]
            n_u = us.shape[1]
            I = np.eye(n_x)
            # feedforward term and feedback coeff.
            ks = np.empty((N, n_u))
            Ks = np.empty((N, n_u, n_x))
            # value function at stage i+1
            Vx = lfx(xs[N], t0 + T)
            Vxx = lfxx(xs[N], t0 + T)
            # expected cost cahnge of all stage
            delta_V = 0
            # daming matrix
            Reg = gamma * np.eye(n_u)
            for i in range(N - 1, -1, -1):
                # t, x and u at satge i
                t = t0 + i*dt
                x, u = xs[i], us[i]
                # derivatives of stage i
                fx_i = I + fx(x, u, t) * dt
                fu_i = fu(x, u, t) * dt
                fxx_i = fxx(x, u, t) * dt
                fux_i = fux(x, u, t) * dt
                fuu_i = fuu(x, u, t) * dt
                lx_i = lx(x, u, t) * dt
                lu_i = lu(x, u, t) * dt
                lxx_i = lxx(x, u, t) * dt
                lux_i = lux(x, u, t) * dt
                luu_i = luu(x, u, t) * dt
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
        
        def forward_pass(f, l, lf, 
                         xs: np.ndarray, us: np.ndarray, t0: float, dt: float,
                         ks: np.ndarray, Ks: np.ndarray, alpha: float=1.0):
            """ Forward pass of DDP.

            Args:
                f (function): State function.
                l (function): Stage cost function.
                lf (function): Terminal cost function.
                xs (numpy.ndarray): Nominal state trajectory.\
                    Size must be (N+1)*n_u
                us (numpy.ndarray): Nominal control trajectory.\
                    Size must be N*n_u
                t0 (float): Initial time.
                dt (float): Discrete time step.
                ks (numpy.ndarray): Series of k. Size must be N * n_u.
                Ks (numpy.ndarray): Series of K. Size must be N * (n_u * n_x).
                alpha (float): step Size of line search. 0 <= alpha <= 1.0.

            Returns:
                xs_new (numpy.ndarray): New state trajectory.
                us_new (numpy.ndarray): New control trajectory.
                J_new (float): Cost along with (xs_new, us_new).
            """
            N = us.shape[0]
            T = N * dt
            # new (xs, us) and cost value.
            xs_new = np.empty(xs.shape)
            xs_new[0] = xs[0]
            us_new = np.empty(us.shape)
            J_new = 0.0
            for i in range(N):
                t = t0 + i*dt
                us_new[i] = us[i] + alpha * ks[i] + Ks[i] @ (xs_new[i] - xs[i])
                xs_new[i + 1] = xs_new[i] + f(xs_new[i], us_new[i], t) * dt
                J_new += l(xs_new[i], us_new[i], t) * dt
            # terminal cost
            J_new += lf(xs_new[N], t0 + T)
            return xs_new, us_new, J_new
    ##### END INNER FUNCTIONS #####

        dt = T / N
        us = us_guess
        xs, J = rollout(f, l, lf, x0, us, t0, dt)
        Js = np.zeros(max_iters + 1, dtype=float)
        Js[0] = J
        gamma = gamma_ini
        is_success = False
        # main iteration
        for iters in range(1, max_iters + 1):
            # backward pass
            ks, Ks, delta_V = backward_pass(
                fx, fu, fxx, fux, fuu, lx, lu, lxx, lux, luu, lfx, lfxx,
                xs, us, t0, dt, gamma
            )
            if np.abs(delta_V) < stop_threshold:
                is_success = True
                iters -= 1
                break
            elif delta_V > 0:
                gamma *= rho_gamma
                Js[iters] = J
                continue
            # line search 
            for alpha in alphas:
                # forward pass
                xs_new, us_new, J_new = forward_pass(
                    f, l, lf, xs, us, t0, dt, ks, Ks, alpha
                )
                if J_new < J:
                    # line search succeeded
                    xs = xs_new
                    us = us_new
                    J = J_new
                    gamma /= rho_gamma
                    break
            else:
                # line search failed
                gamma *= rho_gamma
            gamma = min(max(gamma, gamma_min), gamma_max)
            Js[iters] = J
        ts = np.array([t0 + i*dt for i in range(N + 1)])
        Js = Js[0:iters + 1]
        return xs, us, ts, Js, is_success
    
    @staticmethod
    def print_result(solver_name: str, is_success: bool, iters: int, cost: float,
                     computational_time: float):
        """ Print summary of result.
        
        Args:
            is_success (bool): Flag of success or failure.
            iters (int): Number of iterations.
            computational_time (float): total computational time.
        """
        print('------------------- RESULT -------------------')
        print(f'solver: {solver_name}')
        if is_success:
            status = 'success'
        else:
            status = 'failure'
        print(f'status: {status}')
        print(f'iteration: {iters}')
        print(f'cost value: {cost}')
        print(f'computational time: {computational_time} [s]')
        print(f'per update : {computational_time / iters} [s]')
        print('----------------------------------------------')
    
    @staticmethod
    def log_data(log_dir: str, xs: np.ndarray, us: np.ndarray, ts: np.ndarray,
                 Js: np.ndarray):
        """ Log data.
        
        Args:
            log_dir (str): Directory where data are saved.
            xs (numpy.ndarray): optimal state trajectory. (N + 1) * n_x
            us (numpy.ndarray): optimal control trajectory. N * n_u
            ts (numpy.ndarray): time history.
            Js (numpy.ndarray): costs at each iteration.
        """
        logger = Logger(log_dir)
        logger.save(xs, us, ts, Js)

    @staticmethod
    def plot_data(log_dir: str, xs: np.ndarray, us: np.ndarray, ts: np.ndarray,
                  Js: np.ndarray):
        """ Plot data and save it.
        
        Args:
            log_dir (str): Directory where data are saved.
            xs (numpy.ndarray): optimal state trajectory. (N + 1) * n_x
            us (numpy.ndarray): optimal control trajectory. N * n_u
            ts (numpy.ndarray): time history.
            Js (numpy.ndarray): costs at each iteration.
        """
        plotter = Plotter(log_dir, xs, us, ts, Js)
        plotter.plot(save=True)


class iLQRSolver(DDPSolver):
    def __init__(self, ocp: OCP, init=True):
        super().__init__(ocp=ocp, init=False)
        self._solver_name = 'iLQR'
        self._df = self._df[0:3]        
        if init:
            self.init_solver()

    def init_solver(self):
        """ Initialize solver. Call once before you first call solve().
        """
        print("Initializing solver...")
        # caching
        self._solve(
            *self._df, *self._dl, 
            t0=self._t0, x0=self._x0, N=self._N, T=self._T,
            us_guess=self._us_guess, 
            gamma_ini=self._gamma_ini, rho_gamma=self._rho_gamma,
            gamma_min=self._gamma_min, gamma_max=self._gamma_max, 
            alphas=self._alphas,
            max_iters=10, stop_threshold=self._stop_threshold
        )
        print("Initialization Done.")

    def solve(
            self, 
            t0: float=None, x0: np.ndarray=None, T: float=None, N: int=None,
            xs_guess: np.ndarray=None ,us_guess: np.ndarray=None,
            gamma_fixed: float=None, enable_line_search: bool=True,
            result: bool=False, log: bool=False, plot: bool=False
        ):
        """ Solve OCP via iLQR iteration.

        Args:
            t0 (float): Initial time.
            x0 (numpy.array): Initial state. Size must be n_x.
            T (float): Horizon length.
            N (int): Discretization grid number.
            xs_guess (numpy.array): Guess of state trajectory. \
                Size must be (N+1)*n_x. Only used in multiple-shooting.
            us_guess (numpy.array): Guess of input trajectory. \
                Size must be N*n_u.
            gamma_fixed (float): If set, damping coefficient is fixed.
            enable_line_search (bool=True): If true, enable line searching.
            result (bool): If true, summary of result is printed.
            log (bool): If true, results are logged to log_dir.
            plot (bool): If true, graphs are generated and saved.
        
        Returns:
            xs (numpy.ndarray): Optimal state trajectory. (N * n_x)
            us (numpy.ndarray): Optimal control trajectory. (N * n_u)
            ts (numpy.ndarray): Discretized time history.
            Js (numpy.ndarray): Costs at each iteration.
            time_elapsed (float): Computational time.
            is_success (bool): Success or not.

        """
        if t0 is None:
            t0 = self._t0
        if x0 is None:
            x0 = self._x0
        assert x0.shape[0] == self._n_x
        if N is None:
            N = self._N
        if T is None:
            T = self._T
        if xs_guess is None:
            xs_guess = self._xs_guess
        assert xs_guess.shape == (N + 1, self._n_x)
        if us_guess is None:
            us_guess = self._us_guess
        assert us_guess.shape == (N, self._n_u)
        if gamma_fixed is None:
            gamma_ini = self._gamma_ini
            rho_gamma = self._rho_gamma
            gamma_min = self._gamma_min
            gamma_max = self._gamma_max
        else:
            gamma_ini =  gamma_min = gamma_max = gamma_fixed
            rho_gamma = 1.0
        if enable_line_search:
            alphas = self._alphas
        else:
            alphas = np.array([1.0])
        max_iters = self._max_iters
        stop_threshold = self._stop_threshold
        # derivatives functions.
        f, fx, fu = self._df
        l, lx, lu, lxx, lux, luu, lf, lfx, lfxx = self._dl
        # success flag of solver.
        is_success = False
        # computational time
        time_start = time.perf_counter()
        # solve
        xs, us, ts, Js, is_success = self._solve(
            f, fx, fu, l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
            t0, x0, T, N, us_guess, gamma_ini, rho_gamma, gamma_min, gamma_max,
            alphas, max_iters, stop_threshold
        )
        # computational time
        time_end = time.perf_counter()
        time_elapsed = time_end - time_start
        # number of iterations
        iters = len(Js) - 1
        # results
        if result:
            self.print_result(self._solver_name, is_success, iters, Js[-1], 
                              time_elapsed)
        # log
        if log:
            self.log_data(self._log_dir, xs, us, ts, Js)
        # plot
        if plot:
            self.plot_data(self._log_dir, xs, us, ts, Js)
        return xs, us, ts, Js, time_elapsed, is_success

    @staticmethod
    @numba.njit
    def _solve(
            f, fx, fu, l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
            t0, x0, T, N, us_guess, gamma_ini, rho_gamma, gamma_min, gamma_max,
            alphas, max_iters, stop_threshold
        ):
        """ iLQR algorithm.
        """
    ##### INNER FUNCTIONS #####
        def rollout(f, l, lf, x0: np.ndarray, us: np.ndarray, t0: float, dt: float):
            """ Rollout state trajectory from initial state and input trajectory,\
                with cost is calculated.

            Args:
                f (function): State equation.
                l (function): Stage cost function.
                lf (function): Terminal cost function.
                t0 (float): Initial time.                
                x0 (np.ndarray): Initial state.
                us (np.ndarray): Input control trajectory.
                dt (float): Discrete time step.
            """
            N = us.shape[0]
            xs = np.zeros((N + 1, x0.shape[0]))
            xs[0] = x0
            J = 0.0
            for i in range(N):
                xs[i + 1] = xs[i] + f(xs[i], us[i], t0 + i*dt) * dt
                J += l(xs[i], us[i], t0 + i*dt) * dt
            J += lf(xs[N], t0 + i*N)
            return xs, J

        def backward_pass(fx, fu,
                          lx, lu, lxx, lux, luu, lfx, lfxx,
                          xs: np.ndarray, us: np.ndarray, t0: float, dt: float,
                          gamma: float=1e-6):
            """ Backward pass of iLQR.

            Args:
                fx (function): Derivative of f w.r.t. state x.
                fu (function): Derivative of f w.r.t. state u.
                lx (function): Derivative of l w.r.t. state x.
                lu (function): Derivative of l w.r.t. state u.
                lxx (function): Derivative of l w.r.t. state x and x.
                lux (function): Derivative of l w.r.t. state u and x.
                luu (function): Derivative of l w.r.t. state u and u.
                lfx (function): Derivative of lf w.r.t. state x.
                lfxx (function): Derivative of l w.r.t. state x and x.
                xs (numpy.ndarray): Nominal state trajectory.\
                    Size must be (N+1)*n_u.
                us (numpy.ndarray): Nominalcontrol trajectory.\
                    Size must be N*n_u.
                t0 (float): Initial time.
                dt (float): Discrete time step.
                gamma (float): Damping coefficient.

            Returns:
                ks (numpy.ndarray): Series of k. Its size is N * n_u
                Ks (numpy.ndarray): Series of K. Its size is N * (n_u * n_x)
                delta_V (float): Expecting change of value function at stage 0.
            """
            N = us.shape[0]
            T = N * dt
            n_x = xs.shape[1]
            n_u = us.shape[1]
            I = np.eye(n_x)
            # feedforward term and feedback coeff.
            ks = np.empty((N, n_u))
            Ks = np.empty((N, n_u, n_x))
            # value function at stage i+1
            Vx = lfx(xs[N], t0 + T)
            Vxx = lfxx(xs[N], t0 + T)
            # expected cost cahnge of all stage
            delta_V = 0
            # daming matrix
            Reg = gamma * np.eye(n_u)
            for i in range(N - 1, -1, -1):
                # t, x and u at satge i
                t = t0 + i*dt
                x, u = xs[i], us[i]
                # derivatives of stage i
                fx_i = I + fx(x, u, t) * dt
                fu_i = fu(x, u, t) * dt
                lx_i = lx(x, u, t) * dt
                lu_i = lu(x, u, t) * dt
                lxx_i = lxx(x, u, t) * dt
                lux_i = lux(x, u, t) * dt
                luu_i = luu(x, u, t) * dt
                # action value derivatives
                Qx = lx_i + fx_i.T @ Vx
                Qu = lu_i + fu_i.T @ Vx
                Qxx = lxx_i + fx_i.T @ Vxx @ fx_i
                Qux = lux_i + fu_i.T @ Vxx @ fx_i
                Quu = luu_i + fu_i.T @ Vxx @ fu_i
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
        
        def forward_pass(f, l, lf, 
                         xs: np.ndarray, us: np.ndarray, t0: float, dt: float,
                         ks: np.ndarray, Ks: np.ndarray, alpha: float=1.0):
            """ Forward pass of iLQR.

            Args:
                f (function): State function.
                l (function): Stage cost function.
                lf (function): Terminal cost function.
                xs (numpy.ndarray): Nominal state trajectory.\
                    Size must be (N+1)*n_u
                us (numpy.ndarray): Nominal control trajectory.\
                    Size must be N*n_u
                t0 (float): Initial time.
                dt (float): Discrete time step.
                ks (numpy.ndarray): Series of k. Size must be N * n_u.
                Ks (numpy.ndarray): Series of K. Size must be N * (n_u * n_x).
                alpha (float): step Size of line search. 0 <= alpha <= 1.0.

            Returns:
                xs_new (numpy.ndarray): New state trajectory.
                us_new (numpy.ndarray): New control trajectory.
                J_new (float): Cost along with (xs_new, us_new).
            """
            N = us.shape[0]
            T = N * dt
            # new (xs, us) and cost value.
            xs_new = np.empty(xs.shape)
            xs_new[0] = xs[0]
            us_new = np.empty(us.shape)
            J_new = 0.0
            for i in range(N):
                t = t0 + i*dt
                us_new[i] = us[i] + alpha * ks[i] + Ks[i] @ (xs_new[i] - xs[i])
                xs_new[i + 1] = xs_new[i] + f(xs_new[i], us_new[i], t) * dt
                J_new += l(xs_new[i], us_new[i], t) * dt
            # terminal cost
            J_new += lf(xs_new[N], t0 + T)
            return xs_new, us_new, J_new
    ##### END INNER FUNCTIONS #####

        dt = T / N
        us = us_guess
        xs, J = rollout(f, l, lf, x0, us, t0, dt)
        Js = np.zeros(max_iters + 1, dtype=float)
        Js[0] = J
        gamma = gamma_ini
        is_success = False
        # main iteration
        for iters in range(1, max_iters + 1):
            # backward pass
            ks, Ks, delta_V = backward_pass(
                fx, fu, lx, lu, lxx, lux, luu, lfx, lfxx,
                xs, us, t0, dt, gamma
            )
            if np.abs(delta_V) < stop_threshold:
                is_success = True
                iters -= 1
                break
            elif delta_V > 0:
                gamma *= rho_gamma
                Js[iters] = J
                continue
            # line search 
            for alpha in alphas:
                # forward pass
                xs_new, us_new, J_new = forward_pass(
                    f, l, lf, xs, us, t0, dt, ks, Ks, alpha
                )
                if J_new < J:
                    # line search succeeded
                    xs = xs_new
                    us = us_new
                    J = J_new
                    gamma /= rho_gamma
                    break
            else:
                # line search failed
                gamma *= rho_gamma
            gamma = min(max(gamma, gamma_min), gamma_max)
            Js[iters] = J
        ts = np.array([t0 + i*dt for i in range(N + 1)])
        Js = Js[0:iters + 1]
        return xs, us, ts, Js, is_success
