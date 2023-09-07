# WIP

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


class RiccatiRecursionSolver(SolverBase):
    """ Riccati Recursion Solver. Problem is formulated by \
        multiple-shooting method.
    """
    def __init__(self, ocp: OCP, init=True):
        """ Set optimal control problem.

        Args:
            ocp (ocpy.OCP): optimal control problem. 
            init (bool=True): If True, call init_solver(). It may take time.
        """
        super().__init__(ocp)
        self._solver_name = 'RiccatiRecursion'

        self._n_g = ocp.get_n_g()
        self._has_ineq_constr = False
        if self._n_g > 0:
            self._has_ineq_constr = True
        self._dg = ocp.get_dg()

        self._epsilon_init = 1e-2
        self._rho_epsilon = 0.1
        self._epsilon_tol = 1e-4

        self._kkt_tol = 1e-4

        self._xs_guess = np.zeros((self._N + 1, self._n_x))
        self._us_guess = np.zeros((self._N, self._n_u))
        self._lmds_guess = np.zeros((self._N + 1, self._n_x))
        self._ss_guess = self.generate_ss(self._xs_guess, self._us_guess)
        self._mus_guess = self.generate_mus(self._ss_guess, self._epsilon_init)

        self._lmds_opt = np.zeros((self._N + 1, self._n_x))
        self._ss_opt = np.ones((self._N, self._n_g))
        self._mus_opt = np.ones((self._N, self._n_g))

        self._result['cost_hist'] = None
        self._result['kkt_error_hist'] = None
        self._result['dyn_feas_hist'] = None
        self._result['gamma_hist'] = None
        self._result['alpha_hist'] = None
        self._result['epsilon_hist'] = None
        self._result['xs_opt'] = np.ndarray(0)
        self._result['us_opt'] = np.ndarray(0)
        self._result['lmds_opt'] = np.ndarray(0)
        self._result['ss_opt'] = np.ndarray(0)
        self._result['mus_opt'] = np.ndarray(0)
        self._result['ts'] = np.ndarray(0)

        if init:
            self.init_solver()

    def set_guess(self, xs_guess: np.ndarray=None ,us_guess: np.ndarray=None,
                  lmds_guess: np.ndarray=None,
                  ss_guess: np.ndarray=None, mus_guess: np.ndarray=None):
        """ Set initial guess of xs, us, and lmds.

        Args:
            xs_guess (np.ndarray): Guess of state trajectory. (N + 1)*n_x.
            us_guess (np.ndarray): Guess of input trajectory. N*n_u.
            lmds_guess (np.ndarray): Guess of costate trajectory. (N + 1)*n_x.
            ss_guess (np.ndarray): Guess of slack variables of ineqality \
                constraints. N*n_g.
            mus_guess (np.ndarray): Guess of lagrange variables of ineqality \
                constraints. N*n_g.
        """
        if xs_guess is not None:
            xs_guess = np.asarray(xs_guess, dtype=float)
            assert xs_guess.shape == (self._N + 1, self._n_x)
            self._xs_guess = xs_guess
        if us_guess is not None:
            us_guess = np.asarray(us_guess, dtype=float)
            assert us_guess.shape == (self._N, self._n_u)
            self._us_guess = us_guess
        if lmds_guess is not None:
            lmds_guess = np.asarray(lmds_guess, dtype=float)
            assert lmds_guess.shape == (self._N + 1, self._n_x)
            self._lmds_guess = lmds_guess
        if ss_guess is not None:
            ss_guess = np.asarray(ss_guess, dtype=float)
            assert ss_guess.shape == (self._N, self._n_g)
            self._ss_guess = ss_guess
        if mus_guess is not None:
            mus_guess = np.asarray(mus_guess, dtype=float)
            assert mus_guess.shape == (self._N, self._n_g)
            self._mus_guess = mus_guess

    def reset_guess(self):
        """ Reset guess to zero.
        """
        self._xs_guess = np.zeros((self._N + 1, self._n_x))
        self._us_guess = np.zeros((self._N, self._n_u))
        self._lmds_guess = np.zeros((self._N + 1, self._n_x))
        self._ss_guess = self.generate_ss(self._xs_guess, self._us_guess)
        self._mus_guess = self.generate_mus(self._ss_guess, self._epsilon_init)
    
    def generate_ss(self, xs: np.ndarray, us: np.ndarray):
        """ Reset trajectory of slack variables of inequality constraints.

        Args:
            xs (np.ndarray): Guess of state trajectory. (N + 1)*n_x.
            us (np.ndarray): Guess of input trajectory. N*n_u.
        
        Returns:
            ss (np.ndarray): Trajectory of slack variables of inequality \
                constraints. N*n_g.
        """
        if not self._has_ineq_constr:
            print('OCP does not have inequality constraint.')
            return np.ones((self._N, self._n_g))

        g = self._dg[0]
        t0 = self._t0
        dt = self._dt

        ss = np.empty((self._N, self._n_g))

        for i in range(ss.shape[0]):
            ss[i] = -g(xs[i], us[i], t0 + i * dt)

        return ss

    def generate_mus(self, ss: np.ndarray, epsilon: float):
        """ Reset trajectory of lagrange variables of inequality constraints.

        Args:
            ss (np.ndarray): Trajectory of slack variables of inequality \
                constraints. N*n_g.
            
        Returns:
            mus (np.ndarray): Trajectory of lagrange variables of inequality \
                constraints. N*n_g.
        """
        if not self._has_ineq_constr:
            print('OCP does not have inequality constraint.')
            return np.ones((self._N, self._n_g))

        mus = epsilon * np.reciprocal(ss)

        return mus

    def set_kkt_tol(self, kkt_tol: float=None):
        """ Set KKT error tolerance. 

        Args:
            kkt_tol (float): Threshold of KKT error at each epsilon.
        """
        if kkt_tol is not None:
            assert kkt_tol > 0
            self._kkt_tol = kkt_tol
    
    def set_epsilon(
            self, epsilon_init: float=None, rho_epsilon: float=None,
            epsilon_tol: float=None):
        """ Set parameters related to barrier function.

        Args:
            epsilon_init (float): Initial value of barrier coefficient.
            rho_epsilon (float): Update rate of barrier coefficient.
            epsilon_tol (float): Stop tolerance of barrier coefficient.
        """
        if epsilon_init is not None:
            self._epsilon_init = epsilon_init
        if rho_epsilon is not None:
            self._rho_epsilon = rho_epsilon
        if epsilon_tol is not None:
            self._epsilon_tol = epsilon_tol

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
            gamma_fixed: float=None, enable_line_search: bool=True,
            max_iters: int=None, warm_start :bool=False,
            result=False, log=False, plot=False
        ):
        """ Solve OCP via Riccati Recursion iteration.

        Args:
            gamma_fixed (float): If set, regularization coefficient is fixed.
            enable_line_search (bool=True): If true, enable line searching.
            max_iters (int): Number of max iterations.
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
            rho_gamma = self._rho_gamma
            gamma_min = self._gamma_min
            gamma_max = self._gamma_max
        else:
            gamma_init =  gamma_min = gamma_max = gamma_fixed
            rho_gamma = 1.0

        if enable_line_search:
            alphas = self._alphas
        else:
            alphas = np.array([1.0])

        if max_iters is None:
            max_iters = self._max_iters

        if warm_start is True:
            xs_guess = self._xs_opt
            us_guess = self._us_opt
            lmds_guess = self._lmds_opt
            ss_guess = self._ss_opt
            mus_guess = self._mus_opt
        else:
            xs_guess = self._xs_guess
            us_guess = self._us_guess
            lmds_guess = self._lmds_guess
            ss_guess = self._ss_guess
            mus_guess = self._mus_guess

        # derivatives of functions
        f, fx, fu, fxx, fux, fuu = self._df
        l, lx, lu, lxx, lux, luu, lf, lfx, lfxx = self._dl
        g, gx, gu, gxx, gux, guu = self._dg

        # success flag of solver
        is_success = False

        time_start = time.perf_counter()

        # solve
        xs, us, lmds, ss, mus, ts, is_success,\
            cost_hist, kkt_error_hist, dyn_feas_hist,\
            gamma_hist, alpha_hist, epsilon_hist = self._solve(
                f, fx, fu,
                l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
                g, gx, gu,
                self._t0, self._x0, self._T, self._N, 
                xs_guess, us_guess, lmds_guess, ss_guess, mus_guess,
                gamma_init, rho_gamma, gamma_min, gamma_max, alphas,
                self._epsilon_init, self._rho_epsilon, self._epsilon_tol,                
                self._kkt_tol, max_iters
            )

        time_end = time.perf_counter()
        computation_time = time_end - time_start

        # number of iterations
        noi = len(cost_hist) - 1

        self._xs_opt = xs
        self._us_opt = us
        self._lmds_opt = lmds
        self._ss_opt = ss
        self._mus_opt = mus
        self._ts = ts

        self._result['is_success'] = is_success
        self._result['noi'] = noi
        self._result['computation_time'] = computation_time
        self._result['cost_hist'] = cost_hist
        self._result['kkt_error_hist'] = kkt_error_hist
        self._result['dyn_feas_hist'] = dyn_feas_hist
        self._result['gamma_hist'] = gamma_hist
        self._result['alpha_hist'] = alpha_hist
        self._result['epsilon_hist'] = epsilon_hist
        self._result['xs_opt'] = xs
        self._result['us_opt'] = us
        self._result['lmds_opt'] = lmds
        self._result['ss_opt'] = ss
        self._result['mus_opt'] = mus
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
               g, gx, gu,
               t0, x0, T, N,
               xs, us, lmds, ss, mus,
               gamma_init, rho_gamma, gamma_min, gamma_max, alphas, 
               epsilon_init, rho_epsiron, epsilon_tol,
               kkt_tol, max_iters):
        """ Riccati Recursion algorighm.

        Returns: (xs, us, lmds, ts, is_success,
                  cost_hist, kkt_error_hist, dyn_feas_hist,
                  gamma_hist, alpha_hist, epsilon_hist)

        """
        dt = T / N
        ts = np.array([t0 + i * dt for i in range(N + 1)])

        gamma = gamma_init
        epsilon = epsilon_init

        # check initial KKT error and cost
        kkt_error = eval_kkt_error(f, fx, fu,
                                   lx, lu, lfx,
                                   g, gx, gu,
                                   t0, x0, dt,
                                   xs, us, lmds, ss, mus,
                                   epsilon)
        cost = eval_cost(l, lf, t0, dt, xs, us)
        dyn_feas = eval_dynamics_feasibility(f, t0, x0, dt, xs, us)

        # cost history
        cost_hist = np.zeros(max_iters + 1, dtype=float)
        cost_hist[0] = cost

        # KKT error history
        kkt_error_hist = np.zeros(max_iters + 1, dtype=float)
        kkt_error_hist[0] = kkt_error

        # dynamics feasibility history
        dyn_feas_hist = np.zeros(max_iters + 1, dtype=float)
        dyn_feas_hist[0] = dyn_feas

        # gamma history
        gamma_hist = np.zeros(max_iters + 1, dtype=float)
        gamma_hist[0] = gamma

        # alpha history
        alpha_hist = np.zeros(max_iters + 1, dtype=float)
        alpha_hist[0] = 0.0

        # epsilon (barrier parameter) history
        epsilon_hist = np.zeros(max_iters + 1, dtype=float)
        epsilon_hist[0] = epsilon

        # success flag
        is_success = False

        for iters in range(1, max_iters + 1):

            # if epsilon <= epsilon_tol:
            #     break

            if kkt_error < kkt_tol:
                is_success = True
                iters -= 1
                break

            # (2.26a) - (2.26g)
            kkt_blocks = compute_linearlized_kkt_blocks(
                f, fx, fu,
                lx, lu, lxx, lux, luu, lfx, lfxx,
                g, gx, gu,
                t0, dt,
                xs, us, lmds, ss, mus
            )
            As, Bs, Cs, Ds = kkt_blocks[0:4]
            Qxxs, Quxs, Quus = kkt_blocks[4:7]
            x_bars, g_bars, lx_bars, lu_bars = kkt_blocks[7:11]

            # turn into unconstrained
            Qxx_ecs, Qux_ecs, Quu_ecs, lx_ecs, lu_ecs = eliminate_constraints(
                Qxxs, Quxs, Quus,
                Cs, Ds,
                g_bars, lx_bars, lu_bars,
                ss, mus, epsilon
            )

            # (2.34) - (2.35e)
            Ps, ps, Ks, ks = backward_recursion(
                As, Bs,
                Qxx_ecs, Qux_ecs, Quu_ecs,
                x_bars, lx_ecs, lu_ecs, gamma
            )

            # (2.26b), (2.36), (2.33)
            dxs, dus, dlmds = forward_recursion(
                x0, xs[0], 
                Ps, ps, Ks, ks, 
                As, Bs, x_bars
            )

            dss, dmus = compute_constraints_steps(
                Cs, Ds, g_bars, epsilon,
                ss, mus,
                dxs, dus
            )

            # line search
            (xs_new, us_new, lmds_new, ss_new, mus_new, cost_new, kkt_error_new,
             alpha, alpha_s_max, alpha_mu_max) = line_search(
                f, fx, fu,
                l, lx, lu, lf, lfx,
                g, gx, gu,
                t0, x0, dt,
                xs, us, lmds, ss, mus,
                dxs, dus, dlmds, dss, dmus,
                epsilon,
                alphas, cost, kkt_error
            )

            # evaluate dynamics feasibility
            dyn_feas = eval_dynamics_feasibility(f, t0, x0, dt, xs_new, us_new)

            # modify regularization coefficient
            if kkt_error_new < kkt_error:
                gamma /= rho_gamma
            else:
                gamma *= rho_gamma

            # clip gamma
            gamma = min(max(gamma, gamma_min), gamma_max)

            # update variables.
            xs = xs_new
            us = us_new
            lmds = lmds_new
            ss = ss_new
            mus = mus_new
            kkt_error = kkt_error_new
            cost = cost_new

            # parameters' history
            cost_hist[iters] = cost
            kkt_error_hist[iters] = kkt_error
            dyn_feas_hist[iters] = dyn_feas
            gamma_hist[iters] = gamma
            alpha_hist[iters] = alpha
            epsilon_hist[iters] = epsilon
        else:
            is_success = False
        
        cost_hist = cost_hist[0:iters + 1]
        kkt_error_hist = kkt_error_hist[0:iters + 1]
        dyn_feas_hist = dyn_feas_hist[0: iters + 1]
        gamma_hist = gamma_hist[0:iters + 1]
        alpha_hist = alpha_hist[0:iters + 1]
        epsilon_hist = epsilon_hist[0: iters + 1]

        return (xs, us, lmds, ss, mus, ts, is_success,
                cost_hist, kkt_error_hist, dyn_feas_hist,
                gamma_hist, alpha_hist, epsilon_hist)

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
        g, gx, gu,
        t0: float, dt: float,
        xs: np.ndarray, us: np.ndarray, lmds: np.ndarray,
        ss: np.ndarray, mus: np.ndarray
    ):
    """ Compute blocks of linealized kkt systems.

    Returns: 
        tuple: (As, Bs, Cs, Ds,
                Qxxs, Qxus, Quus,
                x_bars, g_bars, lx_bars, lu_bars)
    """
    N = us.shape[0]
    n_x = xs.shape[1]
    n_u = us.shape[1]
    n_g = ss.shape[1]

    # blocks of (2.26a) - (2.26g).
    As = np.empty((N, n_x, n_x))
    Bs = np.empty((N, n_x, n_u))
    Cs = np.empty((N, n_g, n_x))
    Ds = np.empty((N, n_g, n_u))
    Qxxs = np.empty((N + 1, n_x, n_x))
    Quxs = np.empty((N, n_u, n_x))
    Quus = np.empty((N, n_u, n_u))
    
    # LHS of (2.23c), (2.25b), (2.25c)
    x_bars = np.empty((N, n_x))
    g_bars = np.empty((N, n_g))
    lx_bars = np.empty((N + 1, n_x))
    lu_bars = np.empty((N, n_u))

    for i in range(N):
        # variables at stage i
        t = t0 + i * dt
        x = xs[i]
        u = us[i]
        lmd = lmds[i]
        x_1 = xs[i + 1]
        lmd_1 = lmds[i + 1]

        # hamiltonian
        Hx = lx(x, u, t) + lmd_1.T @ fx(x, u, t)
        Hu = lu(x, u, t) + lmd_1.T @ fu(x, u, t)

        # blocks
        As[i] = np.eye(n_x) + fx(x, u, t) * dt
        Bs[i] = fu(x, u, t) * dt
        Cs[i] = gx(x, u, t)
        Ds[i] = gu(x, u, t)

        Qxxs[i] = lxx(x, u, t) * dt
        Quxs[i] = lux(x, u, t) * dt
        Quus[i] = luu(x, u, t) * dt

        x_bars[i] = x + f(x, u, t) * dt - x_1
        g_bars[i] = g(x, u, t) + ss[i]

        lx_bars[i] = -lmd + lmd_1 + Hx * dt + gx(x, u, t).T @ mus[i]
        lu_bars[i] = Hu * dt + gu(x, u, t).T @ mus[i]

    Qxxs[N] = lfxx(xs[N], t0 + N*dt)
    lx_bars[N] = lfx(xs[N], t0 + N*dt) - lmds[N]

    kkt_blocks = (As, Bs, Cs, Ds,
                  Qxxs, Quxs, Quus,
                  x_bars, g_bars, lx_bars, lu_bars)

    return kkt_blocks


@numba.njit
def eliminate_constraints(
        Qxxs: np.ndarray, Quxs: np.ndarray, Quus: np.ndarray,
        Cs: np.ndarray, Ds: np.ndarray,
        g_bars: np.ndarray, lx_bars: np.ndarray, lu_bars: np.ndarray,
        ss: np.ndarray, mus: np.ndarray, epsilon: float
    ):
    """ Eliminating variables of inequality constraints and \
        returns unconstrained blocks.
    
    Returns:
        tuple: (Qxx_ecs, Qux_ecs, Quu_ecs
                lx_ecs, lu_ecs)
    """
    N = Quxs.shape[0]
    n_g = Cs.shape[1]

    Qxx_ecs = Qxxs.copy()
    Qux_ecs = Quxs.copy()
    Quu_ecs = Quus.copy()
    lx_ecs = lx_bars.copy()
    lu_ecs = lu_bars.copy()

    eps1 = epsilon * np.ones(n_g)

    for i in range(N):
        S_inv_Nu = np.diag(mus[i] / ss[i])
        s_inv = np.reciprocal(ss[i])
        # lxu_part = s_inv * (mus[i] * g_bars[i] - (mus[i] * ss[i] - eps1))
        lxu_part = s_inv * mus[i] * g_bars[i] - mus[i] + s_inv * eps1

        Qxx_ecs[i] += Cs[i].T @ S_inv_Nu @ Cs[i]
        Qux_ecs[i] += Ds[i].T @ S_inv_Nu @ Cs[i]
        Quu_ecs[i] += Ds[i].T @ S_inv_Nu @ Ds[i]
        lx_ecs[i] += Cs[i].T @ lxu_part
        lu_ecs[i] += Ds[i].T @ lxu_part
    
    return (Qxx_ecs, Qux_ecs, Quu_ecs, lx_ecs, lu_ecs)


@numba.njit
def compute_constraints_steps(
        Cs, Ds,
        g_bars, epsilon,
        ss, mus,
        dxs, dus,
    ):
    """ Compute newton steps of s and mu.

    Returns:
        (dss, dmus)
    """
    N = ss.shape[0]

    dss = np.empty(ss.shape)
    dmus = np.empty(mus.shape)

    eps1 = epsilon * np.ones(ss.shape[1])

    for i in range(N):
        dss[i] = - (Cs[i] @ dxs[i] + Ds[i] @ dus[i] + g_bars[i])
    
    for i in range(N):
        dmus[i] = - (mus[i] * (ss[i] + dss[i]) - eps1) / ss[i]
    
    return dss, dmus


@numba.njit
def backward_recursion(
        As: np.ndarray, Bs: np.ndarray,
        Qxxs: np.ndarray, Quxs: np.ndarray, Quus: np.ndarray,
        x_bars: np.ndarray, lx_bars: np.ndarray, lu_bars: np.ndarray,
        gamma: float):
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
        x0, xs0,
        Ps, ps, Ks, ks, 
        As, Bs, x_bars):
    """ Forward recursion.

    Returns:
        tuple (np.ndarray): (dxs, dus, dlmds).
    """
    N = Bs.shape[0]

    dxs = np.empty(ps.shape)
    dus = np.empty(ks.shape)
    dlmds = np.empty(ps.shape)

    dxs[0] = x0 - xs0

    for i in range(N):
        dlmds[i] = Ps[i] @ dxs[i] + ps[i]
        dus[i] = Ks[i] @ dxs[i] + ks[i]
        dxs[i + 1] = As[i] @ dxs[i] + Bs[i] @ dus[i] + x_bars[i]

    dlmds[N] = Ps[N] @ dxs[N] + ps[N]

    return dxs, dus, dlmds


@numba.njit
def line_search(f, fx, fu, 
                l, lx, lu, lf, lfx,
                g, gx, gu,
                t0, x0, dt,
                xs, us, lmds, ss, mus,
                dxs, dus, dlmds, dss, dmus,
                epsilon,
                alphas, cost, kkt_error):
    """ Line search (multiple-shooting).
    
    Returns:
        tuple : (xs_new, us_new, lmds_new, ss_new, mus_new,
                 cost_new, kkt_error_new, alpha,
                 alpha, alpha_s_max, alpha_mu_max)
    """
    N = us.shape[0]
    n_g = ss.shape[1]

    # updated variables
    xs_new = np.empty(xs.shape)
    us_new = np.empty(us.shape)
    lmds_new = np.empty(lmds.shape)
    ss_new = np.empty(ss.shape)
    mus_new = np.empty(mus.shape)

    # maximum alpha
    alpha_s_max = 1.0
    alpha_mu_max = 1.0

    # margin of the fraction to the boundary rule
    tau = 0.995

    # fraction to the boundary rule
    for i in range(N):

        # s and mu at stage i
        s = ss[i]
        mu = mus[i]
        ds = dss[i]
        dmu = dmus[i]

        for j in range(n_g):
            if ds[j] < 0:
                alpha_s_limit = - tau * s[j] / ds[j]
                alpha_s_max = min(alpha_s_max, alpha_s_limit)
            
            if dmu[j] < 0:
                alpha_mu_limit = - tau * mu[j] / dmu[j]
                alpha_mu_max = min(alpha_mu_max, alpha_mu_limit)
    
    # line search
    for alpha in alphas:

        alpha *= alpha_s_max

        # for i in range(N):
        #     us_new[i] = us[i] + alpha * dus[i]
        #     xs_new[i] = xs[i] + alpha * dxs[i]
        #     lmds_new[i] = lmds[i] + alpha * dlmds[i]
        #     ss_new[i] = ss[i] + alpha * dss[i]
        #     mus_new[i] = mus[i] + alpha * dmus[i]

        # xs_new[N] = xs[N] + alpha * dxs[N]
        # lmds_new[N] = lmds[N] + alpha * dlmds[N]
        
        # new variables
        xs_new = xs + alpha * dxs
        us_new = us + alpha * dus
        lmds_new = lmds + alpha_mu_max * dlmds
        ss_new = ss + alpha * dss
        mus_new = mus + alpha_mu_max * dmus

        cost_new = eval_cost(
            l, lf, t0, dt, xs_new, us_new
        )

        kkt_error_new = eval_kkt_error(
            f, fx, fu,
            lx, lu, lfx,
            g, gx, gu,
            t0, x0, dt,
            xs_new, us_new, lmds_new, ss_new, mus_new,
            epsilon
        )

        if cost_new < cost:
            break

        if (not np.isnan(cost_new)) and kkt_error_new < kkt_error:
            break
    
    return (xs_new, us_new, lmds_new, ss_new, mus_new,
            cost_new, kkt_error_new,
            alpha, alpha_s_max, alpha_mu_max)


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
def eval_kkt_error(f, fx, fu, 
                   lx, lu, lfx, 
                   g, gx, gu,
                   t0, x0, dt,
                   xs, us, lmds, ss, mus,
                   epsilon):
    """ Evaluate KKT error.

    Returns:
        kkt_error (float): Square root of KKT RSS (Residual Sum of Square).
    """
    N = us.shape[0]
    n_g = ss.shape[1]

    kkt_error = 0.0

    # initial state
    res = x0 - xs[0]
    kkt_error += np.sum(res ** 2)

    eps1 = epsilon * np.ones(n_g)

    for i in range(N):
        # variables at state i
        x = xs[i]
        u = us[i]
        lmd = lmds[i]
        s = ss[i]
        mu = mus[i]
        x1 = xs[i + 1]
        lmd1 = lmds[i + 1]
        t = t0 + i * dt
        # gradient of hamiltonian
        Hx = lx(x, u, t) + lmd1.T @ fx(x, u, t)
        Hu = lu(x, u, t) + lmd1.T @ fu(x, u, t)

        # dynamics
        res = x + f(x, u, t) * dt - x1
        kkt_error += np.sum(res ** 2)

        # inequality constraints with slack variables.
        res = g(x, u, t) + s
        kkt_error += np.sum(res ** 2)

        # Lx
        res = -lmd + lmd1 + Hx * dt + gx(x, u, t).T @ mu
        kkt_error += np.sum(res ** 2)

        # Lu
        res = Hu * dt + gu(x, u, t).T @ mu
        kkt_error += np.sum(res ** 2)

        # complementary condition
        res = mu * s - eps1
        kkt_error += np.sum(res ** 2)

    # Lx[N]
    res = lfx(xs[N], t0 + N * dt) - lmds[N]
    kkt_error += np.sum(res ** 2)

    return np.sqrt(kkt_error)


@numba.njit
def eval_dynamics_feasibility(f, t0, x0, dt, xs, us):
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
