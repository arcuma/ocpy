import sympy as sym
import numpy as np
import numba
import time
import matplotlib.pyplot as plt

from ocpy import symutils
from ocpy.ocp import OCP
from ocpy.logger import Logger
from ocpy.plotter import Plotter
from ocpy.solverbase import SolverBase

numba.config.DISABLE_JIT = False


class DDPSolver(SolverBase):
    """ Differential Dynamic Programming(DDP) solver.
    """
    def __init__(self, ocp: OCP, init=True, ddp=True):
        """ Set optimal control problem.
        
        Args:
            ocp (OCP): optimal control problem. 
            init (bool=True): If True, call init_solver(). It may take time.
            ddp (bool=True): If False, iLQR is appllied.
        """
        super().__init__(ocp)
        self._solver_name = 'DDP'
        self._is_ddp = ddp

        ### tolerance (stop condition)
        self._stop_tol = 1e-3

        self._us_guess = np.zeros((self._N, self._n_u))

        self._xs_opt = np.zeros((self._N + 1, self._n_x))
        self._us_opt = np.zeros((self._N, self._n_u))

        self._result['cost_hist'] = None
        self._result['gamma_hist'] = None
        self._result['alpha_hist'] = None
        self._result['xs_opt'] = np.ndarray(0)
        self._result['us_opt'] = np.ndarray(0)
        self._result['ts'] = np.ndarray(0)

        if init:
            self.init_solver()

    def set_guess(self, us_guess: np.ndarray=None):
        """ Set initial guess of variables.

        Args:
            us_guess (np.ndarray): Guess of input trajectory. N*n_u.
        """
        if us_guess is not None:
            us_guess = np.asarray(us_guess, dtype=float)
            assert us_guess.shape == (self._N, self._n_u)
            self._us_guess = us_guess

    def reset_guess(self):
        """ Reset guess to zero.
        """
        self._us_guess = np.zeros((self._N, self._n_u))
    
    def reset_opt(self):
        """ Reset solution.
        """
        self._us_opt = np.zeros((self._N, self._n_u))
        self._xs_opt = np.zeros((self._N, self._n_x))

    def set_stop_tol(self, stop_tol: float=None):
        """ Set stop tolerance.

        Args:
            stop_tol (float): Stop threshold.
        """
        if stop_tol is not None:
            self._stop_tol = stop_tol

    def print_result(self):
        """ Print summary of result.
        """
        is_success = self._result['is_success']
        if is_success:
            status = 'success'
        else:
            status = 'failure'
        noi = self._result['noi']
        computation_time = self._result['computation_time']
        cost = self._result['cost_hist'][-1]

        print('------------------- RESULT -------------------')
        print(f'solver: {self._solver_name}')
        print(f'status: {status}')
        print(f'number of iterations: {noi}')
        print(f'computation time: {computation_time:.6f} [s]')
        if noi >= 1:
            print(f'per update : {computation_time / noi:.6f} [s]')
        print(f'final cost value: {cost: .8f}')
        print('----------------------------------------------')
    
    def log_data(self):
        """ Log data to self._log_dir.
        """

        logger = Logger(self._log_dir)
        logger.save(self._xs_opt, self._us_opt, self._ts,
                    self._result['cost_hist'])

    def plot_data(self, save=True):
        """ Plot data.
        """
        plotter = Plotter(self._log_dir, self._xs_opt, self._us_opt, self._ts,
                          self._result['cost_hist'])
        plotter.plot(save=save)

    def plot_detail(self):
        """ plot result of some parameters.
        """
        result = self.get_result()

        gamma_hist = result['gamma_hist']
        plt.plot(gamma_hist)
        plt.title('gamma')
        plt.show()

        alpha_hist = result['alpha_hist']
        plt.plot(alpha_hist)
        plt.title('alpha')
        plt.show()
        print('average alpha:',sum(alpha_hist / (len(alpha_hist) - 1)))

        cost_hist = result['cost_hist']
        plt.plot(cost_hist)
        plt.title('cost')
        plt.show()

    def init_solver(self):
        """ Initialize solver. Call once before you first call solve().
        """
        print("Initializing solver...")

        ### compile
        max_iters_eva = self._max_iters
        self._max_iters = 1
        self.solve(save_result=False)
        self._max_iters = max_iters_eva

        print("Initialization done.")

    def solve(
            self,
            from_opt=False,
            result=False, log=False, plot=False, save_result=True
        ):
        """ Solve OCP via DDP iteration.

        Args:
            from_opt (bool=False): If true, previous solution is used \
                as initial guess. Mainly for MPC.
            result (bool): If true, summary of result is printed.
            log (bool): If true, results are logged to log_dir.
            plot (bool): If true, graphs are generated (and saved if log==True).
            save_result (bool): If true, results are saved.
        """
        if from_opt is True:
            us_guess = self._us_opt
        else:
            us_guess = self._us_guess

        ### derivatives functions.
        f, fx, fu, fxx, fux, fuu = self._df
        l, lx, lu, lxx, lux, luu, lf, lfx, lfxx = self._dl

        ### success flag
        is_success = False

        time_start = time.perf_counter()

        ### solve
        xs, us, ts, \
        is_success, cost_hist, gamma_hist, alpha_hist = self._solve(
            f, fx, fu, fxx, fux, fuu,
            l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
            self._t0, self._x0, self._T, self._N,
            us_guess,
            self._gamma_init, self._r_gamma, self._gamma_min, self._gamma_max, self._fix_gamma,
            self._alpha_min, self._r_alpha, self._enable_line_search,
            self._stop_tol, self._min_iters, self._max_iters, self._is_ddp
        )

        time_end = time.perf_counter()
        computation_time = time_end - time_start

        ### number of iterations
        noi = len(cost_hist) - 1
        if save_result:
            self._xs_opt = xs
            self._us_opt = us
            self._ts = ts

            self._result['is_success'] = is_success
            self._result['noi'] = noi
            self._result['computation_time'] = computation_time
            self._result['cost_hist'] = cost_hist
            self._result['gamma_hist'] = gamma_hist
            self._result['alpha_hist'] = alpha_hist
            self._result['xs_opt'] = xs
            self._result['us_opt'] = us
            self._result['ts'] = ts

        ### result
        if result:
            self.print_result()
        ### log
        if log:
            self.log_data()
        ### plot
        if plot:
            self.plot_data(save=log)

    @staticmethod
    @numba.njit
    def _solve(
            f, fx, fu, fxx, fux, fuu,
            l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
            t0, x0, T, N, us_guess,
            gamma_init, r_gamma, gamma_min, gamma_max, fix_gamma,
            alpha_min, r_alpha, enable_line_search,
            stop_tol, min_iters, max_iters, is_ddp
        ):
        """ DDP algorithm.

        Returns:
            (xs, us, ts, 
            is_success, cost_hist, gamma_hist, alpha_hist)
        """
        dt = T / N
        us = us_guess
        gamma = gamma_init

        ### initial rollout
        xs, cost = rollout(f, l, lf, t0, x0, dt, us)

        ### cost history
        cost_hist = np.zeros(max_iters + 1, dtype=float)
        cost_hist[0] = cost

        ### gamma history
        gamma_hist = np.zeros(max_iters + 1, dtype=float)
        gamma_hist[0] = gamma

        ### alpha history
        alpha_hist = np.zeros(max_iters + 1, dtype=float)
        alpha_hist[0] = 0.0

        is_success = False

        ### main iteration
        for iters in range(1, max_iters + 1):

            ### backward pass
            ks, Ks, delta_V = backward_pass(
                fx, fu, fxx, fux, fuu,
                lx, lu, lxx, lux, luu, lfx, lfxx,
                t0, dt, xs, us,
                gamma, is_ddp
            )

            ### stop criterion
            if np.abs(delta_V) < stop_tol:
                is_success = True

            if is_success and iters > min_iters:
                iters -= 1
                break
            
            ### step size
            alpha = 1.0
            success_line_search = False

            ### line search
            while True:
                ### forward pass
                xs_new, us_new, cost_new = forward_pass(
                    f, l, lf,
                    t0, dt, xs, us,
                    ks, Ks, alpha
                )

                ### stop condition of line search
                if cost_new < cost:
                    success_line_search = True
                    break

                if (not enable_line_search) or (alpha < alpha_min):
                    break
                
                ### update alpha
                alpha *= r_alpha

            ### modify regularization coefficient
            if not fix_gamma:
                if success_line_search:
                    gamma *= r_gamma
                else:
                    gamma /= r_gamma
                ### clip gamma
                gamma = min(max(gamma, gamma_min), gamma_max)

            ### update trajectory
            xs = xs_new
            us = us_new
            cost = cost_new
    
            cost_hist[iters] = cost
            gamma_hist[iters] = gamma
            alpha_hist[iters] = alpha

        ts = np.array([t0 + i*dt for i in range(N + 1)])
        cost_hist = cost_hist[0:iters + 1]
        gamma_hist = gamma_hist[0:iters + 1]
        alpha_hist = alpha_hist[0:iters + 1]

        return (xs, us, ts, 
                is_success, cost_hist, gamma_hist, alpha_hist)


#### DDP FUNCTIONS ####
@numba.njit
def rollout(f, l, lf, t0: float, x0: np.ndarray, dt: float, us: np.ndarray):
    """ Rollout state trajectory from initial state and input trajectory,\
        with cost is calculated.

    Args:
        f (function): State equation.
        l (function): Stage cost function.
        lf (function): Terminal cost function.
        t0 (float): Initial time.                
        dt (float): Discrete time step.
        x0 (np.ndarray): Initial state.
        us (np.ndarray): Input control trajectory.
    """
    N = us.shape[0]
    xs = np.zeros((N + 1, x0.shape[0]))
    xs[0] = x0
    cost = 0.0
    for i in range(N):
        xs[i + 1] = xs[i] + f(xs[i], us[i], t0 + i*dt) * dt
        cost += l(xs[i], us[i], t0 + i*dt) * dt
    cost += lf(xs[N], t0 + i*N)
    return xs, cost


@numba.njit
def vector_dot_tensor(Vx: np.ndarray, fab: np.ndarray):
    """ Tensor dot product between 1d vector and 3d tensor, contraction\
        with each 0 and 1 axis. This is used for product between Vx \
        (derivative of value function) and fqq (hessian of dynamics)
        
    Args:
        Vx (np.ndarray) : n_x-sized 1d array.
        fab (np.ndarray) : (n_b*n_x*n_a)-sized 3d array.
        
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


@numba.njit
def backward_pass(fx, fu, fxx, fux, fuu, 
                  lx, lu, lxx, lux, luu, lfx, lfxx,
                  t0: float, dt: float, xs: np.ndarray, us: np.ndarray,
                  gamma: float=0.0, is_ddp: bool=True):
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
        lfxx (function): Derivative of lf w.r.t. state x and x.
        t0 (float): Initial time.
        dt (float): Discrete time step.
        xs (np.ndarray): Nominal state trajectory.\
            Size must be (N+1)*n_x.
        us (np.ndarray): Nominalcontrol trajectory.\
            Size must be N*n_u.
        gamma (float): regularization coefficient.
        is_ddp (bool): If False, iLQR is applied.

    Returns:
        ks (np.ndarray): Series of k. Its size is N * n_u
        Ks (np.ndarray): Series of K. Its size is N * (n_u * n_x)
        Delta_V (float): Expecting change of value function at stage 0.
    """
    N = us.shape[0]
    T = N * dt
    n_x = xs.shape[1]
    n_u = us.shape[1]
    I = np.eye(n_x)

    ### feedforward term and feedback coeff.
    ks = np.empty((N, n_u))
    Ks = np.empty((N, n_u, n_x))

    ### value function at stage i+1
    Vx = lfx(xs[N], t0 + T)
    Vxx = lfxx(xs[N], t0 + T)

    ### expected cost change of all stage
    delta_V = 0.0

    ### Regularization matrix
    Reg = gamma * np.eye(n_u)

    for i in range(N - 1, -1, -1):
        ### variables at this stage
        t = t0 + i*dt
        x, u = xs[i], us[i]

        ### derivatives of stage i
        fx_i = I + fx(x, u, t) * dt
        fu_i = fu(x, u, t) * dt
        if is_ddp:
            fxx_i = fxx(x, u, t) * dt
            fux_i = fux(x, u, t) * dt
            fuu_i = fuu(x, u, t) * dt
        lx_i = lx(x, u, t) * dt
        lu_i = lu(x, u, t) * dt
        lxx_i = lxx(x, u, t) * dt
        lux_i = lux(x, u, t) * dt
        luu_i = luu(x, u, t) * dt

        ### action value derivatives
        Qx = lx_i + fx_i.T @ Vx
        Qu = lu_i + fu_i.T @ Vx
        if is_ddp:
            Qxx = lxx_i + fx_i.T @ Vxx @ fx_i + vector_dot_tensor(Vx, fxx_i).T
            Qux = lux_i + fu_i.T @ Vxx @ fx_i + vector_dot_tensor(Vx, fux_i).T
            Quu = luu_i + fu_i.T @ Vxx @ fu_i + vector_dot_tensor(Vx, fuu_i).T
        else:
            Qxx = lxx_i + fx_i.T @ Vxx @ fx_i
            Qux = lux_i + fu_i.T @ Vxx @ fx_i
            Quu = luu_i + fu_i.T @ Vxx @ fu_i

        ### feedforward and feedback terms
        Quu_inv = np.linalg.inv(Quu + Reg)
        k = -Quu_inv @ Qu
        K = -Quu_inv @ Qux
        ks[i] = k
        Ks[i] = K

        ### value function of stage i, passed to stage i-1.
        delta_V_i = 0.5 * k.T @ Quu @ k + k.T @ Qu
        Vx = Qx - K.T @ Quu @ k
        Vxx = Qxx - K.T @ Quu @ K
        
        delta_V += delta_V_i

    return ks, Ks, delta_V


@numba.njit
def forward_pass(f, l, lf, 
                 t0: float, dt: float, xs: np.ndarray, us: np.ndarray,
                 ks: np.ndarray, Ks: np.ndarray, alpha: float=1.0):
    """ Forward pass of DDP.

    Args:
        f (function): State function.
        l (function): Stage cost function.
        lf (function): Terminal cost function.
        t0 (float): Initial time.
        dt (float): Discrete time step.
        xs (np.ndarray): Nominal state trajectory.\
            Size must be (N+1)*n_x
        us (np.ndarray): Nominal control trajectory.\
            Size must be N*n_u
        ks (np.ndarray): Series of k. Size must be N * n_u.
        Ks (np.ndarray): Series of K. Size must be N * (n_u * n_x).
        alpha (float): step Size of line search. 0 <= alpha <= 1.0.

    Returns:
        xs_new (np.ndarray): New state trajectory.
        us_new (np.ndarray): New control trajectory.
        cost_new (float): Cost along with (xs_new, us_new).
    """
    N = us.shape[0]
    T = N * dt

    ### new (xs, us) and cost value.
    xs_new = np.empty(xs.shape)
    xs_new[0] = xs[0]
    us_new = np.empty(us.shape)
    cost_new = 0.0

    for i in range(N):
        t = t0 + i*dt
        us_new[i] = us[i] + alpha * ks[i] + Ks[i] @ (xs_new[i] - xs[i])
        xs_new[i + 1] = xs_new[i] + f(xs_new[i], us_new[i], t) * dt
        cost_new += l(xs_new[i], us_new[i], t) * dt
    ### terminal cost
    cost_new += lf(xs_new[N], t0 + T)

    return xs_new, us_new, cost_new
### END DDP FUNCTIONS ###


class iLQRSolver(DDPSolver):
    """ iterative Linear Quadratic Regulator (iLQR) Solver.
    """
    def __init__(self, ocp: OCP, init=True):
        super().__init__(ocp=ocp, init=False, ddp=False)
        self._solver_name = 'iLQR'
        if init:
            self.init_solver()
