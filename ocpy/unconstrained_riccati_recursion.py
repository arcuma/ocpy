import sympy as sym
import numpy as np
import numba
import abc
import time

from ocpy import symutils
from ocpy.ocp import OCP
from ocpy.logger import Logger
from ocpy.plotter import Plotter
from ocpy.solverbase import SolverBase


class UCRRSolver(SolverBase):
    """ Unconstrained Riccati Recursion Solver. Problem is formulated by \
        multiple-shooting method.
    """
    def __init__(self, ocp: OCP, init=True):
        """ Set optimal control problem.

        Args:
            ocp (ocpy.OCP): optimal control problem. 
            init (bool=True): If True, call init_solver(). It may take time.
        """
        super().__init__(ocp)
        self._solver_name = 'UCRR'

        self._kkt_tol = 1e-2

        self._xs_guess = np.zeros((self._N + 1, self._n_x))
        self._us_guess = np.zeros((self._N, self._n_u))
        self._lamxs_guess = np.zeros((self._N + 1, self._n_x))

        self._lamxs_opt = np.zeros((self._N + 1, self._n_x))

        self._result['cost_hist'] = None
        self._result['kkt_error_hist'] = None
        self._result['dyn_error_hist'] = None
        self._result['gamma_hist'] = None
        self._result['alpha_hist'] = None
        self._result['r_merit_hist'] = None
        self._result['xs_opt'] = np.ndarray(0)
        self._result['us_opt'] = np.ndarray(0)
        self._result['lamxs_opt'] = np.ndarray(0)
        self._result['ts'] = np.ndarray(0)

        if init:
            self.init_solver()

    def set_guess(self, xs_guess: np.ndarray=None ,us_guess: np.ndarray=None,
                  lamxs_guess: np.ndarray=None):
        """ Set initial guess of xs, us, and lamxs.

        Args:
            xs_guess (np.ndarray): Guess of state trajectory. (N + 1)*n_x.
            us_guess (np.ndarray): Guess of input trajectory. N*n_u.
            lamxs_guess (np.ndarray): Guess of costate trajectory. (N + 1)*n_x.
        """
        if us_guess is not None:
            us_guess = np.asarray(us_guess, dtype=float)
            assert us_guess.shape == (self._N, self._n_u)
            self._us_guess = us_guess
        if xs_guess is not None:
            xs_guess = np.asarray(xs_guess, dtype=float)
            assert xs_guess.shape == (self._N + 1, self._n_x)
            self._xs_guess = xs_guess
        if lamxs_guess is not None:
            lamxs_guess = np.asarray(lamxs_guess, dtype=float)
            assert lamxs_guess.shape == (self._N + 1, self._n_x)
            self._lamxs_guess = lamxs_guess

    def reset_guess(self):
        """ Reset guess to zero.
        """
        self._xs_guess = np.zeros((self._N + 1, self._n_x))
        self._us_guess = np.zeros((self._N, self._n_u))
        self._lamxs_guess = np.zeros((self._N + 1, self._n_x))

    def set_kkt_tol(self, kkt_tol: float=None):
        """ Set stop criterion. 

        Args:
            kkt_tol (float): Threshold of KKT error at each epsilon.
        """
        if kkt_tol is not None:
            assert kkt_tol > 0
            self._kkt_tol = kkt_tol

    def init_solver(self):
        """ Initialize solver. Call once before you first call solve().
        """
        print("Initializing solver...")

        # compile
        self.solve(
            gamma_fixed=1e3, max_iters=3
        )

        print("Initialization done.")

    def solve(
            self,
            gamma_fixed: float=None, enable_line_search: bool=False,
            max_iters: int=None, warm_start :bool=False,
            result=False, log=False, plot=False
        ):
        """ Solve OCP via Riccati Recursion iteration.

        Args:
            gamma_fixed (float): If set, regularization coefficient is fixed.
            enable_line_search (bool=True): If true, enable line search.
            warm_start (bool=False): If true, previous solution is used \
                as initial guess. Mainly for MPC.
            result (bool): If true, summary of result is printed.
            log (bool): If true, results are logged to log_dir.
            plot (bool): If true, graphs are generated and saved.
        
        Returns:
            ts (np.ndarray): Discretized Time at each stage.
            xs (np.ndarray): Optimal state trajectory. (N + 1) * n_x.
            us (np.ndarray): Optimal control trajectory. N * n_u.
            is_success (bool): Success or not.
        """
        if gamma_fixed is None:
            gamma_init = self._gamma_init
            r_gamma = self._r_gamma
            gamma_min = self._gamma_min
            gamma_max = self._gamma_max
        else:
            gamma_init =  gamma_min = gamma_max = gamma_fixed
            r_gamma = 1.0

        if enable_line_search:
            alpha_min = self._alpha_min
            r_alpha = self._r_alpha
        else:
            alpha_min = 1.0
            r_alpha = self._r_alpha

        if max_iters is None:
            max_iters = self._max_iters

        if warm_start is True:
            xs_guess = self._xs_opt
            us_guess = self._us_opt
            lamxs_guess = self._lamxs_opt
        else:
            xs_guess = self._xs_guess
            us_guess = self._us_guess
            lamxs_guess = self._lamxs_guess

        # derivatives of functions
        f, fx, fu, fxx, fux, fuu = self._df
        l, lx, lu, lxx, lux, luu, lf, lfx, lfxx = self._dl

        # success flag of solver
        is_success = False

        time_start = time.perf_counter()

        # solve
        xs, us, lamxs, ts, is_success, cost_hist, kkt_error_hist, dyn_error_hist,\
        gamma_hist, alpha_hist, r_merit_hist = self._solve(
                f, fx, fu,
                l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
                self._t0, self._x0, self._T, self._N, 
                xs_guess, us_guess, lamxs_guess,
                gamma_init, r_gamma, gamma_min, gamma_max, alpha_min, r_alpha,
                self._kkt_tol, max_iters
        )

        time_end = time.perf_counter()
        computation_time = time_end - time_start

        # number of iterations
        noi = len(cost_hist) - 1

        self._xs_opt = xs
        self._us_opt = us
        self._lamxs_opt = lamxs
        self._ts = ts

        self._result['is_success'] = is_success
        self._result['noi'] = noi
        self._result['computation_time'] = computation_time
        self._result['cost_hist'] = cost_hist
        self._result['kkt_error_hist'] = kkt_error_hist
        self._result['dyn_error_hist'] = dyn_error_hist
        self._result['gamma_hist'] = gamma_hist
        self._result['alpha_hist'] = alpha_hist
        self._result['r_merit_hist'] = r_merit_hist
        self._result['xs_opt'] = xs
        self._result['us_opt'] = us
        self._result['lamxs_opt'] = lamxs
        self._result['ts'] = ts

        # results
        if result:
            self.print_result()
        # log
        if log:
            self.log_data()
        # plot            
        if plot:
            self.plot_data(save=log)

        return xs, us, ts, is_success

    @staticmethod
    @numba.njit
    def _solve(f, fx, fu, 
               l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
               t0, x0, T, N,
               xs, us, lamxs,
               gamma_init, r_gamma, gamma_min, gamma_max,
               alpha_min, r_alpha, kkt_tol, max_iters):
        """ Riccati Recursion algorighm.

        Returns: (xs, us, lamxs, ts, is_success,
                  cost_hist, kkt_error_hist, dyn_error_hist,
                  gamma_hist, alpha_hist)

        """
        dt = T / N
        ts = np.array([t0 + i * dt for i in range(N + 1)])

        gamma = gamma_init
        r_merit = 1.0

        # check initial KKT error and cost
        kkt_error = eval_kkt_error(f, fx, fu, lx, lu, lfx, 
                                   t0, x0, dt, xs, us, lamxs)
        cost = eval_cost(l, lf, t0, dt, xs, us)
        dyn_error = eval_dynamics_error(f, t0, x0, dt, xs, us)

        # cost history
        cost_hist = np.zeros(max_iters + 1, dtype=float)
        cost_hist[0] = cost

        # KKT error history
        kkt_error_hist = np.zeros(max_iters + 1, dtype=float)
        kkt_error_hist[0] = kkt_error

        # dynamics feasibility history
        dyn_error_hist = np.zeros(max_iters + 1, dtype=float)
        dyn_error_hist[0] = dyn_error

        # gamma history
        gamma_hist = np.zeros(max_iters + 1, dtype=float)
        gamma_hist[0] = gamma

        # alpha history
        alpha_hist = np.zeros(max_iters + 1, dtype=float)
        alpha_hist[0] = 0.0

        r_merit_hist = np.zeros(max_iters + 1, dtype=float)
        r_merit_hist[0] = 0.0

        # success flag
        is_success = False

        for iters in range(1, max_iters + 1):

            if kkt_error < kkt_tol:
                is_success = True
                iters -= 1
                break

            # compute blocks of pertubed KKT system
            kkt_blocks = compute_linearlized_kkt_blocks(
                f, fx, fu, lx, lu, lxx, lux, luu, lfx, lfxx,
                t0, dt, xs, us, lamxs
            )
            As, Bs = kkt_blocks[0:2]
            Qxxs, Quxs, Quus = kkt_blocks[2:5]
            x_bars, lx_bars, lu_bars = kkt_blocks[5:8]

            # backward recursion
            Ps, ps, Ks, ks = backward_recursion(
                As, Bs, Qxxs, Quxs, Quus, x_bars, lx_bars, lu_bars, gamma
            )

            # forward recursion
            dxs, dus, dlamxs = forward_recursion(
                Ps, ps, Ks, ks, As, Bs, x_bars, x0, xs[0]
            )

            # line search
            xs_new, us_new, lamxs_new, cost_new, kkt_error_new, alpha, r_merit_new \
                = line_search(
                    f, fx, fu, l, lx, lu, lf, lfx, t0, x0, dt,
                    xs, us, lamxs, dxs, dus, dlamxs,
                    alpha_min, r_alpha, cost, kkt_error, r_merit
            )

            # evaluate dynamics feasibility
            dyn_error = eval_dynamics_error(f, t0, x0, dt, xs, us)

            # modify regularization coefficient
            if kkt_error_new < kkt_error:
                gamma /= r_gamma
            else:
                gamma *= r_gamma

            # clip gamma
            gamma = min(max(gamma, gamma_min), gamma_max)

            # update variables.
            xs = xs_new
            us = us_new
            lamxs = lamxs_new

            kkt_error = kkt_error_new
            cost = cost_new
            r_merit = r_merit_new

            cost_hist[iters] = cost
            kkt_error_hist[iters] = kkt_error
            dyn_error_hist[iters] = dyn_error
            gamma_hist[iters] = gamma
            alpha_hist[iters] = alpha
            r_merit_hist[iters] = r_merit
        else:
            is_success = False
        
        cost_hist = cost_hist[0:iters + 1]
        kkt_error_hist = kkt_error_hist[0:iters + 1]
        dyn_error_hist = dyn_error_hist[0: iters + 1]
        gamma_hist = gamma_hist[0:iters + 1]
        alpha_hist = alpha_hist[0:iters + 1]
        r_merit_hist = r_merit_hist[0:iters + 1]

        return (xs, us, lamxs, ts, is_success,
                cost_hist, kkt_error_hist, dyn_error_hist,
                gamma_hist, alpha_hist, r_merit_hist)

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
        kkt_error = self._result['kkt_error_hist'][-1]

        print('------------------- RESULT -------------------')
        print(f'solver: {self._solver_name}')
        print(f'status: {status}')
        print(f'number of iterations: {noi}')
        print(f'computation time: {computation_time:.6f} [s]')
        if noi >= 1:
            print(f'per update : {computation_time / noi:.6f} [s]')
        print(f'final cost value: {cost:.8f}')
        print(f'final KKT error: {kkt_error:.8f}')
        print('----------------------------------------------')


    def log_data(self):
        """ Log data to self._log_dir.
        """
        cost_hist = self._result['cost_hist']
        kkt_error_hist = self._result['kkt_error_hist']

        logger = Logger(self._log_dir)
        logger.save(self._xs_opt, self._us_opt, self._ts,
                    cost_hist, kkt_error_hist)

    def plot_data(self, save=True):
        """ Plot data.

        Args:
            save (bool): If True, graph is saved.
        """
        cost_hist = self._result['cost_hist']
        kkt_error_hist = self._result['kkt_error_hist']

        plotter = Plotter(self._log_dir, self._xs_opt, self._us_opt, self._ts,
                          cost_hist, kkt_error_hist)
        plotter.plot(save=save)


@numba.njit
def compute_linearlized_kkt_blocks(
        f, fx, fu, lx, 
        lu, lxx, lux, luu, lfx, lfxx,
        t0: float, dt: float,
        xs: np.ndarray, us: np.ndarray, lamxs: np.ndarray
    ):
    """ Compute blocks of linealized kkt systems.

    Returns: 
        tuple: (As, Bs,
                Qxxs, Qxus, Quus,
                x_bars, lx_bars, lu_bars)
    """
    N = us.shape[0]
    n_x = xs.shape[1]
    n_u = us.shape[1]

    # LHS
    As = np.empty((N, n_x, n_x))
    Bs = np.empty((N, n_x, n_u))
    Qxxs = np.empty((N + 1, n_x, n_x))
    Quxs = np.empty((N, n_u, n_x))
    Quus = np.empty((N, n_u, n_u))
    
    # RHS
    x_bars = np.empty((N, n_x))
    lx_bars = np.empty((N + 1, n_x))
    lu_bars = np.empty((N, n_u))

    I = np.eye(n_x)

    for i in range(N):
        # variables at stage i
        t = t0 + i * dt
        x = xs[i]
        u = us[i]
        lamx = lamxs[i]
        x_1 = xs[i + 1]
        lamx_1 = lamxs[i + 1]

        As[i] = I + fx(x, u, t) * dt
        Bs[i] = fu(x, u, t) * dt

        Qxxs[i] = lxx(x, u, t) * dt
        Quxs[i] = lux(x, u, t) * dt
        Quus[i] = luu(x, u, t) * dt

        x_bars[i] = x + f(x, u, t) * dt - x_1

        lx_bars[i] = -lamx + lamx_1 + (lx(x, u, t) + lamx_1.T @ fx(x, u, t)) * dt
        lu_bars[i] = (lu(x, u, t) + lamx_1.T @ fu(x, u, t)) * dt

    Qxxs[N] = lfxx(xs[N], t0 + N*dt)
    lx_bars[N] = lfx(xs[N], t0 + N*dt) - lamxs[N]

    kkt_blocks = (As, Bs,
                  Qxxs, Quxs, Quus,
                  x_bars, lx_bars, lu_bars)

    return kkt_blocks


@numba.njit
def backward_recursion(
        As, Bs,
        Qxxs, Quxs, Quus,
        x_bars, lx_bars, lu_bars,
        gamma):
    """ Backward recursion.

    Returns:
        tuple (np.ndarray): (Ps, ps, Ks, ks)
    """
    N, n_x, n_u = Bs.shape

    Ps = np.empty((N + 1, n_x, n_x))
    ps = np.empty((N + 1, n_x))
    Ks = np.empty((N, n_u, n_x))
    ks = np.empty((N, n_u))

    # i = N
    Ps[N] = Qxxs[N]
    ps[N] = lx_bars[N]

    # regularlization
    Reg = gamma * np.eye(n_u)

    for i in range(N - 1, -1, -1):
        F = Qxxs[i] + As[i].T @ Ps[i + 1] @ As[i]
        H = Quxs[i] + Bs[i].T @ Ps[i + 1] @ As[i]
        G = Quus[i] + Bs[i].T @ Ps[i + 1] @ Bs[i]

        G_inv = np.linalg.inv(G + Reg)

        Ks[i] = -G_inv @ H
        ks[i] = -G_inv @ (Bs[i].T @ (Ps[i + 1] @ x_bars[i] + ps[i + 1]) + lu_bars[i])
 
        Ps[i] = F - Ks[i].T @ G @ Ks[i]
        ps[i] = As[i].T @ (ps[i + 1] + Ps[i + 1] @ x_bars[i]) + lx_bars[i] + H.T @ ks[i]
        
        # Ps[i] = (Ps[i] + Ps[i].T) / 2
    return Ps, ps, Ks, ks


@numba.njit
def forward_recursion(
        Ps, ps, Ks, ks, As, Bs, x_bars, x0, xs0
    ):
    """ Forward recursion.

    Returns:
        tuple (np.ndarray): (dxs, dus, dlamxs).
    """
    N = Bs.shape[0]

    dxs = np.empty(ps.shape)
    dus = np.empty(ks.shape)
    dlamxs = np.empty(ps.shape)

    dxs[0] = x0 - xs0

    for i in range(N):
        dlamxs[i] = Ps[i] @ dxs[i] + ps[i]
        dus[i] = Ks[i] @ dxs[i] + ks[i]
        dxs[i + 1] = As[i] @ dxs[i] + Bs[i] @ dus[i] + x_bars[i]

    dlamxs[N] = Ps[N] @ dxs[N] + ps[N]

    return dxs, dus, dlamxs


@numba.njit
def line_search(
        f, fx, fu, l, lx, lu, lf, lfx, t0, x0, dt,
        xs, us, lamxs, dxs, dus, dlamxs,
        alpha_min, r_alpha, cost, kkt_error, r_merit):
    """ Line search (multiple-shooting).
    
    Returns:
        tuple : (xs_new, us_new, lamxs_new, cost_new, kkt_error_new, alpha)
    """
    N = us.shape[0]

    c_armijo = 1e-4

    xs_new = np.empty(xs.shape)
    us_new = np.empty(us.shape)
    lamxs_new = np.empty(lamxs.shape)

    merit, deriv_merit, r_merit = eval_merit_and_derivative(
        f, l, lx, lu, lf, lfx,
        t0, x0, dt, xs, us, dxs, dus, r_merit
    )

    alpha = 1.0

    # backtracking line search
    while True:

        xs_new = xs + alpha * dxs
        us_new = us + alpha * dus
        lamxs_new = lamxs + alpha * dlamxs

        merit_new = eval_merit(
            f, l, lf, t0, x0, dt, xs_new, us_new, r_merit
        )

        kkt_error_new = eval_kkt_error(
            f, fx, fu, lx, lu, lfx,
            t0, x0, dt,
            xs_new, us_new, lamxs_new
        )

        cost_new = eval_cost(
            l, lf, t0, dt, xs_new, us_new
        )

        # stop condition
        if merit_new < merit + c_armijo * alpha * deriv_merit:
            break
        if cost_new < cost:
            break
        if (not np.isnan(cost_new)) and kkt_error_new < kkt_error:
            break

        # reached minimum alpha
        if alpha < alpha_min:
            break

        # update alpha
        alpha *= r_alpha
    
    return xs_new, us_new, lamxs_new, cost_new, kkt_error_new, alpha, r_merit


@numba.njit
def eval_merit_and_derivative(
        f, l, lx, lu, lf, lfx, t0, x0, dt, xs, us, dxs, dus, r_merit
    ):

    N = us.shape[0]
    I = np.eye(xs.shape[1])

    r_merit_max = 1e6
    r_merit_min = 1e-6

    merit_cost = 0.0
    merit_constr = 0.0
    deriv_merit_cost = 0.0
    deriv_merit_constr = 0.0

    merit_cost += np.linalg.norm(x0 - xs[0], 1)
    deriv_merit_constr += -np.linalg.norm(x0 - xs[0], 1)

    for i in range(N):
        # variables
        x = xs[i]
        x1 = xs[i + 1]
        u = us[i]
        dx = dxs[i]
        du = dus[i]
        t = t0 + i * dt

        # cost
        merit_cost += l(x, u, t) * dt

        # constraint
        merit_constr += np.linalg.norm(x + f(x, u, t) * dt - x1, 1)

        # cost deriv
        deriv_merit_cost += lx(x, u, t) * dt @ dx
        deriv_merit_cost += lu(x, u, t) * dt @ du

        # constraint deriv
        deriv_merit_constr += -np.linalg.norm(x + f(x, u, t) * dt - x1, 1)
    
    merit_cost += lf(xs[N], t0 + N * dt)
    deriv_merit_cost += lfx(xs[N], t0 + N * dt) @ dxs[N]

    # based on (3.5) of "An interior algorith for ..."
    if merit_constr > 1e-10:
        rho = 0.5
        r_merit_trial = deriv_merit_cost / ((1.0 - rho) * merit_constr)
        r_merit = max(r_merit, r_merit_trial + 1)

    merit = merit_cost + r_merit * merit_constr
    deriv_merit = deriv_merit_cost + r_merit * deriv_merit_constr

    return merit, deriv_merit, r_merit


@numba.njit
def eval_merit(
        f, l, lf, t0, x0, dt, xs, us, r_merit
    ):
    """ Evaluate merit function.

    Returns:
        merit (float): Merit function value.
    """
    N = us.shape[0]

    merit_cost = 0.0
    merit_constr = 0.0

    merit_constr += np.linalg.norm(x0 - xs[0], ord=1)

    for i in range(N):
        x = xs[i]        
        x1 = xs[i + 1]
        u = us[i]
        t = t0 + i * dt

        # cost
        merit_cost += l(x, u, t) * dt

        # constraint
        merit_constr += np.linalg.norm(x + f(x, u, t) * dt - x1, ord=1)
    
    merit_cost += lf(xs[N], t0 + N * dt)

    # merit function value
    merit = merit_cost + r_merit * merit_constr

    return merit


@numba.njit
def eval_cost(l, lf, t0, dt, xs, us):
    """ Evaluate cost value.

    Returns:
        cost (float): Cost along with xs and us.
    """
    N = us.shape[0]

    cost = 0.0

    for i in range(N):
        cost += l(xs[i], us[i], t0 + i * dt) * dt

    cost += lf(xs[N], t0 + N * dt)

    return cost


@numba.njit
def eval_kkt_error(
        f, fx, fu, lx, lu, lfx, t0, x0, dt, xs, us, lamxs, ord=2):
    """ Evaluate KKT error.

    Returns:
        kkt_error (float): Square root of KKT RSS (Residual Sum of Square).
    """
    N = us.shape[0]

    kkt_error = 0.0

    res = x0 - xs[0]
    kkt_error += np.sum(res ** ord)

    for i in range(N):
        x = xs[i]
        u = us[i]
        lamx = lamxs[i]
        x1 = xs[i + 1]
        lamx1 = lamxs[i + 1]
        t = t0 + i * dt

        res = x + f(x, u, t) * dt - x1
        kkt_error += np.sum(res ** ord)

        res = -lamx + lamx1 + (lx(x, u, t) + lamx1.T @ fx(x, u, t)) * dt
        kkt_error += np.sum(res ** ord)

        res = (lu(x, u, t) + lamx1.T @ fu(x, u, t)) * dt
        kkt_error += np.sum(res ** ord)

    res = lfx(xs[N], t0 + N * dt) - lamxs[N]
    kkt_error += np.sum(res ** ord)

    return kkt_error ** (1 / ord)


@numba.njit
def eval_dynamics_error(f, t0, x0, dt, xs, us):
    """ Evaluate feasibility of dynamics.

    Returns:
        dynamics_error (float): Square root of dynamics RSS.
    """
    N = us.shape[0]

    res = x0 - xs[0]
    dynamics_error = np.sum(res ** 2)

    for i in range(N):
        res = xs[i] + f(xs[i], us[i], t0 + i*dt) * dt - xs[i + 1]
        dynamics_error += np.sum(res ** 2)
    
    return np.sqrt(dynamics_error)


@numba.njit
def rollout(f, t0: float, dt: float, xs: np.ndarray, us: np.ndarray):
    """ Rollout (multiple-shooting).

    Returns:
        xs_new (np.ndarray): New trajectory of state.
    """
    N = us.shape[0]

    xs_new = np.empty(xs.shape)
    xs_new[0] = xs[0]

    for i in range(N):
        xs_new = xs[i] + f(xs[i], us[i], t0 + i * dt) * dt

    return xs_new
