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

        self._mu_init = 1e-1
        self._r_mu = 0.01
        self._mu_min = 1e-8
        self._update_mu = True

        self._kkt_tol = 1e-4

        self._xs_guess = np.zeros((self._N + 1, self._n_x))
        self._us_guess = np.zeros((self._N, self._n_u))
        self._ss_guess = self.generate_ss(self._xs_guess, self._us_guess)
        self._lamxs_guess = np.zeros((self._N + 1, self._n_x))
        self._lamss_guess = self.generate_lamss(self._ss_guess, self._mu_init)

        self._lamxs_opt = np.zeros((self._N + 1, self._n_x))
        self._ss_opt = np.ones((self._N, self._n_g))
        self._lamss_opt = np.ones((self._N, self._n_g))

        self._result['cost_hist'] = None
        self._result['kkt_error_hist'] = None
        self._result['kkt_error_mu_hist'] = None
        self._result['dyn_error_hist'] = None
        self._result['gamma_hist'] = None
        self._result['alpha_hist'] = None
        self._result['mu_hist'] = None
        self._result['r_merit_hist'] = None
        self._result['xs_opt'] = np.ndarray(0)
        self._result['us_opt'] = np.ndarray(0)
        self._result['ss_opt'] = np.ndarray(0)
        self._result['lamxs_opt'] = np.ndarray(0)
        self._result['lamss_opt'] = np.ndarray(0)
        self._result['ts'] = np.ndarray(0)

        if init:
            self.init_solver()

    def set_guess(self, xs_guess: np.ndarray=None ,us_guess: np.ndarray=None,
                  ss_guess: np.ndarray=None,
                  lamxs_guess: np.ndarray=None, lamss_guess: np.ndarray=None):
        """ Set initial guess of xs, us, and lamxs.

        Args:
            xs_guess (np.ndarray): Guess of state trajectory. (N + 1)*n_x.
            us_guess (np.ndarray): Guess of input trajectory. N*n_u.
            ss_guess (np.ndarray): Guess of slack variables of ineqality \
                constraints. N*n_g.
            lamxs_guess (np.ndarray): Guess of costate trajectory. (N + 1)*n_x.
            lamss_guess (np.ndarray): Guess of lagrange variables of ineqality \
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

        if ss_guess is None:
            self._ss_guess = self.generate_ss(self._xs_guess, self._us_guess)
        else:
            ss_guess = np.asarray(ss_guess, dtype=float)
            assert ss_guess.shape == (self._N, self._n_g)
            self._ss_guess = ss_guess

        if lamxs_guess is not None:
            lamxs_guess = np.asarray(lamxs_guess, dtype=float)
            assert lamxs_guess.shape == (self._N + 1, self._n_x)
            self._lamxs_guess = lamxs_guess

        if lamss_guess is None:
            self._lamss_guess = self.generate_lamss(self._ss_guess, self._mu_init)
        else:
            lamss_guess = np.asarray(lamss_guess, dtype=float)
            assert lamss_guess.shape == (self._N, self._n_g)
            self._lamss_guess = lamss_guess

    def reset_guess(self):
        """ Reset guess to zero.
        """
        self._xs_guess = np.zeros((self._N + 1, self._n_x))
        self._us_guess = np.zeros((self._N, self._n_u))
        self._ss_guess = self.generate_ss(self._xs_guess, self._us_guess)
        self._lamxs_guess = np.zeros((self._N + 1, self._n_x))
        self._lamss_guess = self.generate_lamss(self._ss_guess, self._mu_init)
    
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

    def generate_lamss(self, ss: np.ndarray, mu: float):
        """ Reset trajectory of lagrange variables of inequality constraints.

        Args:
            ss (np.ndarray): Trajectory of slack variables of inequality \
                constraints. N*n_g.
            
        Returns:
            lamss (np.ndarray): Trajectory of lagrange variables of inequality \
                constraints. N*n_g.
        """
        if not self._has_ineq_constr:
            print('OCP does not have inequality constraint.')
            return np.ones((self._N, self._n_g))

        lamss = mu * np.reciprocal(ss)

        return lamss

    def set_kkt_tol(self, kkt_tol: float=None):
        """ Set KKT error tolerance. 

        Args:
            kkt_tol (float): Threshold of KKT error at each mu.
        """
        if kkt_tol is not None:
            assert kkt_tol > 0
            self._kkt_tol = kkt_tol
    
    def set_barrier_param(
            self, mu_init: float=None, r_mu: float=None,
            mu_min: float=None):
        """ Set parameters related to barrier function.

        Args:
            mu_init (float): Initial value of barrier coefficient.
            r_mu (float): Update rate of barrier coefficient.
            mu_min (float): Stop tolerance of barrier coefficient.
        """
        if mu_init is not None:
            self._mu_init = mu_init
        if r_mu is not None:
            self._r_mu = r_mu
        if mu_min is not None:
            self._mu_min = mu_min

    def init_solver(self):
        """ Initialize solver. Call once before you first call solve().
        """
        print("Initializing solver...")

        # compile
        self.solve(
            max_iters=3
        )

        print("Initialization done.")

    def solve(
            self,
            update_gamma: bool=False, enable_line_search: bool=False,
            update_mu: bool=True,
            max_iters: int=None, warm_start :bool=False,
            result=False, log=False, plot=False
        ):
        """ Solve OCP via Riccati Recursion iteration.

        Args:
            update_gamma (bool): If True, regularization coefficient is updated.
            enable_line_search (bool=True): If true, enable line searching.
            update_mu (bool=True): If True, barrier parameter is updated \
                while iteration.
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
        if update_gamma:
            gamma_init = self._gamma_init
            r_gamma = self._r_gamma
            gamma_min = self._gamma_min
            gamma_max = self._gamma_max
        else:
            gamma_init =  gamma_min = gamma_max = self._gamma_init
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
            ss_guess = self._ss_opt
            lamxs_guess = self._lamxs_opt
            lamss_guess = self._lamss_opt
        else:
            xs_guess = self._xs_guess
            us_guess = self._us_guess
            ss_guess = self._ss_guess
            lamxs_guess = self._lamxs_guess
            lamss_guess = self._lamss_guess

        # derivatives of functions
        f, fx, fu, fxx, fux, fuu = self._df
        l, lx, lu, lxx, lux, luu, lf, lfx, lfxx = self._dl
        g, gx, gu, gxx, gux, guu = self._dg

        # success flag of solver
        is_success = False

        time_start = time.perf_counter()

        # solve
        xs, us, ss, lamxs, lamss, ts, is_success,\
        cost_hist, kkt_error_hist, kkt_error_mu_hist, dyn_error_hist,\
        gamma_hist, alpha_hist, mu_hist, r_merit_hist = self._solve(
                f, fx, fu,
                l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
                g, gx, gu,
                self._t0, self._x0, self._T, self._N, 
                xs_guess, us_guess, ss_guess, lamxs_guess, lamss_guess,
                gamma_init, r_gamma, gamma_min, gamma_max, alpha_min, r_alpha,
                self._mu_init, self._r_mu, self._mu_min,
                update_mu,
                self._kkt_tol, max_iters
        )

        time_end = time.perf_counter()
        computation_time = time_end - time_start

        # number of iterations
        noi = len(cost_hist) - 1

        self._xs_opt = xs
        self._us_opt = us
        self._ss_opt = ss
        self._lamxs_opt = lamxs
        self._lamss_opt = lamss
        self._ts = ts

        self._result['is_success'] = is_success
        self._result['noi'] = noi
        self._result['computation_time'] = computation_time
        self._result['cost_hist'] = cost_hist
        self._result['kkt_error_hist'] = kkt_error_hist
        self._result['kkt_error_mu_hist'] = kkt_error_mu_hist
        self._result['dyn_error_hist'] = dyn_error_hist
        self._result['gamma_hist'] = gamma_hist
        self._result['alpha_hist'] = alpha_hist
        self._result['mu_hist'] = mu_hist
        self._result['r_merit_hist'] = r_merit_hist
        self._result['xs_opt'] = xs
        self._result['us_opt'] = us
        self._result['ss_opt'] = ss
        self._result['lamxs_opt'] = lamxs
        self._result['lamss_opt'] = lamss
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

    @staticmethod
    @numba.njit
    def _solve(f, fx, fu, 
               l, lx, lu, lxx, lux, luu, lf, lfx, lfxx,
               g, gx, gu,
               t0, x0, T, N,
               xs, us, ss, lamxs, lamss,
               gamma_init, r_gamma, gamma_min, gamma_max, alpha_min, r_alpha, 
               mu_init, r_mu, mu_min, update_mu,
               kkt_tol, max_iters):
        """ Riccati Recursion algorighm.

        Returns: (xs, us, lamxs, ts, is_success,
                  cost_hist, kkt_error_hist, dyn_error_hist,
                  gamma_hist, alpha_hist, mu_hist)

        """
        dt = T / N
        ts = np.array([t0 + i * dt for i in range(N + 1)])

        gamma = gamma_init
        mu = mu_init
        r_merit = 1.0

        # check initial KKT error and cost
        kkt_error = eval_kkt_error(
            f, fx, fu, lx, lu, lfx, g, gx, gu, t0, x0, dt,
            xs, us, ss, lamxs, lamss, 0
        )
        kkt_error_mu = eval_kkt_error(
            f, fx, fu, lx, lu, lfx, g, gx, gu, t0, x0, dt,
            xs, us, ss, lamxs, lamss, mu
        )
        cost = eval_cost(l, lf, t0, dt, xs, us)
        dyn_error = eval_dynamics_error(f, t0, x0, dt, xs, us)

        # cost history
        cost_hist = np.zeros(max_iters + 1, dtype=float)
        cost_hist[0] = cost

        # KKT error history
        kkt_error_hist = np.zeros(max_iters + 1, dtype=float)
        kkt_error_hist[0] = kkt_error

        # barrier KKT error history
        kkt_error_mu_hist = np.zeros(max_iters + 1, dtype=float)
        kkt_error_mu_hist[0] = kkt_error_mu

        # dynamics feasibility history
        dyn_error_hist = np.zeros(max_iters + 1, dtype=float)
        dyn_error_hist[0] = dyn_error

        # gamma history
        gamma_hist = np.zeros(max_iters + 1, dtype=float)
        gamma_hist[0] = gamma

        # alpha history
        alpha_hist = np.zeros(max_iters + 1, dtype=float)
        alpha_hist[0] = 0.0

        # mu (barrier parameter) history
        mu_hist = np.zeros(max_iters + 1, dtype=float)
        mu_hist[0] = mu

        # penalty coefficient of constraints violation in merit function
        r_merit_hist = np.zeros(max_iters + 1, dtype=float)
        r_merit_hist[0] = r_merit

        # success flag
        is_success = False

        for iters in range(1, max_iters + 1):

            # stop condition
            # if update_mu == False, kkt_error_mu is used for creterion.
            if update_mu:
                if kkt_error < kkt_tol:
                    is_success = True
                    iters -= 1
                    break
            else:
                if kkt_error_mu < kkt_tol:
                    is_success = True
                    iters -= 1
                    break

            # compute blocks of pertubed KKT system
            kkt_blocks = compute_linearlized_kkt_blocks(
                f, fx, fu, lx, lu, lxx, lux, luu, lfx, lfxx, g, gx, gu,
                t0, dt, xs, us, ss, lamxs, lamss
            )
            As, Bs, Cs, Ds = kkt_blocks[0:4]
            Qxxs, Quxs, Quus = kkt_blocks[4:7]
            x_bars, g_bars, lx_bars, lu_bars = kkt_blocks[7:11]

            # turn into unconstrained
            Qxx_ecs, Qux_ecs, Quu_ecs, lx_ecs, lu_ecs = eliminate_constraints(
                Qxxs, Quxs, Quus, Cs, Ds, g_bars, lx_bars, lu_bars,
                ss, lamss, mu
            )

            # backward recursion 
            Ps, ps, Ks, ks = backward_recursion(
                As, Bs, Qxx_ecs, Qux_ecs, Quu_ecs, x_bars, lx_ecs, lu_ecs, gamma
            )

            # forward recursion
            dxs, dus, dlamxs = forward_recursion(
                Ps, ps, Ks, ks, As, Bs, x_bars, x0, xs[0]
            )

            # get newton step of slack variables and lagrange variables
            dss, dlamss = compute_slack_step(
                Cs, Ds, g_bars, mu,
                ss, lamss,
                dxs, dus
            )

            # line search
            xs, us, ss, lamxs, lamss, cost, kkt_error, kkt_error_mu, \
            alpha, alpha_dual, r_merit_new = line_search(
                    f, fx, fu, l, lx, lu, lf, lfx, g, gx, gu, t0, x0, dt,
                    xs, us, ss, lamxs, lamss, dxs, dus, dss, dlamxs, dlamss,
                    mu, alpha_min, r_alpha, cost, kkt_error_mu, r_merit
            )

            # evaluate dynamics feasibility
            dyn_error = eval_dynamics_error(f, t0, x0, dt, xs, us)

            # modify regularization coefficient
            if alpha_min < alpha:
                gamma /= r_gamma
            else:
                gamma *= r_gamma
            
            # clip gamma
            gamma = min(max(gamma, gamma_min), gamma_max)

            # update barrier parameter
            if update_mu:
                th = 10.0
                kkt_tol_mu = max(mu * th, kkt_tol - mu)
                # kkt_tol_mu = mu
                if kkt_error_mu < kkt_tol_mu and mu >= kkt_tol / 100:
                    mu *= r_mu

            # parameters' history
            cost_hist[iters] = cost
            kkt_error_hist[iters] = kkt_error
            kkt_error_mu_hist[iters] = kkt_error_mu
            dyn_error_hist[iters] = dyn_error
            gamma_hist[iters] = gamma
            alpha_hist[iters] = alpha
            mu_hist[iters] = mu
            r_merit_hist[iters] = r_merit
        else:
            is_success = False
        
        cost_hist = cost_hist[0:iters + 1]
        kkt_error_hist = kkt_error_hist[0:iters + 1]
        kkt_error_mu_hist = kkt_error_mu_hist[0:iters + 1]
        dyn_error_hist = dyn_error_hist[0: iters + 1]
        gamma_hist = gamma_hist[0:iters + 1]
        alpha_hist = alpha_hist[0:iters + 1]
        mu_hist = mu_hist[0: iters + 1]
        r_merit_hist = r_merit_hist[0: iters + 1]

        return (xs, us, ss, lamxs, lamss, ts, is_success,
                cost_hist, kkt_error_hist, kkt_error_mu_hist, dyn_error_hist,
                gamma_hist, alpha_hist, mu_hist, r_merit_hist)

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
def compute_new_mu(
        ss: np.ndarray, lamss: np.ndarray
    ):
    """ Compute new barrier parameter. Duality measure is used.

    Args:
        ss (np.ndarray): Trajectory of slack variables of inequality.
        lamss (np.ndarray): Trajectory of Lagrange variables of inequality

    Returns:
        mu_new (float):
    """
    N = ss.shape[0]
    n_g = ss.shape[1]

    # reduction factor
    sigma = 0.8

    # ss \bigodot lamss
    ss_lamss = 0.0
    for i in range(N):
        ss_lamss += ss[i].T @ lamss[i]
    
    # duality measure
    mu_new = ss_lamss / (N * n_g)

    mu_new *= sigma

    return mu_new


@numba.njit
def compute_linearlized_kkt_blocks(
        f, fx, fu, lx, 
        lu, lxx, lux, luu, lfx, lfxx,
        g, gx, gu,
        t0: float, dt: float,
        xs: np.ndarray, us: np.ndarray, ss: np.ndarray,
        lamxs: np.ndarray, lamss: np.ndarray
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

    # LHS
    As = np.empty((N, n_x, n_x))
    Bs = np.empty((N, n_x, n_u))
    Cs = np.empty((N, n_g, n_x))
    Ds = np.empty((N, n_g, n_u))
    Qxxs = np.empty((N + 1, n_x, n_x))
    Quxs = np.empty((N, n_u, n_x))
    Quus = np.empty((N, n_u, n_u))
    
    # RHS
    x_bars = np.empty((N, n_x))
    g_bars = np.empty((N, n_g))
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

        # hamiltonian
        Hx = lx(x, u, t) + lamx_1.T @ fx(x, u, t)
        Hu = lu(x, u, t) + lamx_1.T @ fu(x, u, t)

        # blocks
        As[i] = I + fx(x, u, t) * dt
        Bs[i] = fu(x, u, t) * dt
        Cs[i] = gx(x, u, t)
        Ds[i] = gu(x, u, t)

        Qxxs[i] = lxx(x, u, t) * dt
        Quxs[i] = lux(x, u, t) * dt
        Quus[i] = luu(x, u, t) * dt

        x_bars[i] = x + f(x, u, t) * dt - x_1
        g_bars[i] = g(x, u, t) + ss[i]

        lx_bars[i] = -lamx + lamx_1 + Hx * dt + gx(x, u, t).T @ lamss[i]
        lu_bars[i] = Hu * dt + gu(x, u, t).T @ lamss[i]

    Qxxs[N] = lfxx(xs[N], t0 + N*dt)
    lx_bars[N] = lfx(xs[N], t0 + N*dt) - lamxs[N]

    kkt_blocks = (As, Bs, Cs, Ds,
                  Qxxs, Quxs, Quus,
                  x_bars, g_bars, lx_bars, lu_bars)

    return kkt_blocks


@numba.njit
def eliminate_constraints(
        Qxxs: np.ndarray, Quxs: np.ndarray, Quus: np.ndarray,
        Cs: np.ndarray, Ds: np.ndarray,
        g_bars: np.ndarray, lx_bars: np.ndarray, lu_bars: np.ndarray,
        ss: np.ndarray, lamss: np.ndarray, mu: float
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

    mu1 = mu * np.ones(n_g)

    for i in range(N):
        S_inv_Nu = np.diag(lamss[i] / ss[i])
        s_inv = np.reciprocal(ss[i])
        # lxu_part = s_inv * (lamss[i] * g_bars[i] - (lamss[i] * ss[i] - mu1))
        lxu_part = s_inv * lamss[i] * g_bars[i] - lamss[i] + s_inv * mu1

        Qxx_ecs[i] += Cs[i].T @ S_inv_Nu @ Cs[i]
        Qux_ecs[i] += Ds[i].T @ S_inv_Nu @ Cs[i]
        Quu_ecs[i] += Ds[i].T @ S_inv_Nu @ Ds[i]
        lx_ecs[i] += Cs[i].T @ lxu_part
        lu_ecs[i] += Ds[i].T @ lxu_part
    
    return (Qxx_ecs, Qux_ecs, Quu_ecs, lx_ecs, lu_ecs)


@numba.njit
def compute_slack_step(
        Cs, Ds,
        g_bars, mu,
        ss, lamss,
        dxs, dus,
    ):
    """ Compute newton step of s and lams.

    Returns:
        (dss, dlamss)
    """
    N = ss.shape[0]

    dss = np.empty(ss.shape)
    dlamss = np.empty(lamss.shape)

    mu1 = mu * np.ones(ss.shape[1])

    for i in range(N):
        dss[i] = - (Cs[i] @ dxs[i] + Ds[i] @ dus[i] + g_bars[i])
    
    for i in range(N):
        dlamss[i] = - (lamss[i] * (ss[i] + dss[i]) - mu1) / ss[i]
    
    return dss, dlamss


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
        f, fx, fu, l, lx, lu, lf, lfx, g, gx, gu, t0, x0, dt, 
        xs, us, ss, lamxs, lamss, dxs, dus, dss, dlamxs, dlamss,
        mu, alpha_min, r_alpha, cost, kkt_error_mu, r_merit
    ):
    """ Line search (multiple-shooting).
    
    Returns:
        tuple : (xs_new, us_new, ss_new, lamxs_new, lamss_new,
                 cost_new, kkt_error_new,
                 alpha_primal, alpha_dual, r_merit)
    """
    N = us.shape[0]
    n_g = ss.shape[1]

    c_armijo = 1e-4

    # updated variables
    xs_new = np.empty(xs.shape)
    us_new = np.empty(us.shape)
    ss_new = np.empty(ss.shape)
    lamxs_new = np.empty(lamxs.shape)
    lamss_new = np.empty(lamss.shape)

    # maximum alpha
    alpha_s_max = 1.0
    alpha_lams_max = 1.0

    # margin of the fraction to the boundary rule
    tau = 0.995

    # merit value at nominal state
    merit, deriv_merit, r_merit = eval_merit_and_derivative(
        f, l, lx, lu, lf, lfx, g, t0, x0, dt, 
        xs, us, ss, dxs, dus, dss, mu, r_merit
    )
 
    # fraction to the boundary rule
    for i in range(N):

        # s and lams at stage i
        s = ss[i]
        lams = lamss[i]
        ds = dss[i]
        dlams = dlamss[i]

        for j in range(n_g):
            if ds[j] < 0:
                alpha_s_limit = - tau * s[j] / ds[j]
                alpha_s_max = min(alpha_s_max, alpha_s_limit)
            
            if dlams[j] < 0:
                alpha_lams_limit = - tau * lams[j] / dlams[j]
                alpha_lams_max = min(alpha_lams_max, alpha_lams_limit)
    
    alpha_primal = alpha_s_max
    alpha_dual = alpha_lams_max

    # backtracking line search
    while True:
        # new variables
        xs_new = xs + alpha_primal * dxs
        us_new = us + alpha_primal * dus
        ss_new = ss + alpha_primal * dss
        lamxs_new = lamxs + alpha_dual * dlamxs
        lamss_new = lamss + alpha_dual * dlamss

        # evaluate merit function
        merit_new = eval_merit(
            f, l, lf, g, t0, x0, dt, xs_new, us_new, ss_new, mu, r_merit
        )
        
        # evaluate new cost
        cost_new = eval_cost(
            l, lf, t0, dt, xs_new, us_new
        )

        # evaluate new barrier KKT error
        kkt_error_mu_new = eval_kkt_error(
            f, fx, fu, lx, lu, lfx, g, gx, gu, t0, x0, dt,
            xs_new, us_new, ss_new, lamxs_new, lamss_new, mu
        )

        # stop condition
        if merit_new < merit + c_armijo * alpha_primal * deriv_merit:
            break
        if cost_new < cost:
            break
        if (not np.isnan(cost_new)) and kkt_error_mu_new < kkt_error_mu:
            break

        # reached minimum alpha
        if alpha_primal < alpha_min:
            break

        # update alpha
        alpha_primal *= r_alpha
    
    # eval new kkt error
    kkt_error_new = eval_kkt_error(
        f, fx, fu, lx, lu, lfx, g, gx, gu, t0, x0, dt,
        xs_new, us_new, ss_new, lamxs_new, lamss_new, 0
    )

    return (xs_new, us_new, ss_new, lamxs_new, lamss_new,
            cost_new, kkt_error_new, kkt_error_mu_new,
            alpha_primal, alpha_dual, r_merit)

@numba.njit
def eval_merit_and_derivative(
        f, l, lx, lu, lf, lfx, g, t0, x0, dt,
        xs, us, ss, dxs, dus, dss, mu, r_merit
    ):
    """ Evaluate merit function and its directional derivatives of metit function.

    Returns:
        merit, deriv_merit, r_merit
    """
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
        s = ss[i]
        dx = dxs[i]
        du = dus[i]
        ds = dss[i]
        t = t0 + i * dt

        # cost
        merit_cost += l(x, u, t) * dt
        merit_cost += -mu * np.log(ss[i]).sum()

        # constraint
        merit_constr += np.linalg.norm(x + f(x, u, t) * dt - x1, 1)
        merit_constr += np.linalg.norm(g(x, u, t) + s, 1)

        # cost deriv
        deriv_merit_cost += lx(x, u, t) * dt @ dx
        deriv_merit_cost += lu(x, u, t) * dt @ du
        deriv_merit_cost += -mu * (1.0/s) @ ds

        # constraint deriv
        deriv_merit_constr += -np.linalg.norm(x + f(x, u, t) * dt - x1, 1)
        deriv_merit_constr += -np.linalg.norm(g(x, u, t) + s, 1)
    
    merit_cost += lf(xs[N], t0 + N * dt)
    deriv_merit_cost += lfx(xs[N], t0 + N * dt) @ dxs[N]

    # based on (3.5) of "An interior algorith for ..."
    if merit_constr > 1e-10:
        rho = 0.1
        r_merit_trial = deriv_merit_cost / ((1.0 - rho) * merit_constr)
        r_merit = max(r_merit, r_merit_trial)

    merit = merit_cost + r_merit * merit_constr
    deriv_merit = deriv_merit_cost + r_merit * deriv_merit_constr

    return merit, deriv_merit, r_merit


@numba.njit
def eval_merit(
        f, l, lf, g, t0, x0, dt, xs, us, ss, mu, r_merit
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
        s = ss[i]
        t = t0 + i * dt

        # cost
        merit_cost += l(x, u, t) * dt
        merit_cost += -mu * np.log(s).sum()

        # constraint
        merit_constr += np.linalg.norm(x + f(x, u, t) * dt - x1, ord=1)
        merit_constr += np.linalg.norm(g(x, u, t) + s, ord=1)
    
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
        f, fx, fu, lx, lu, lfx, g, gx, gu, t0, x0, dt,
        xs, us, ss, lamxs, lamss, mu, ord=2):
    """ Evaluate l1 KKT error.

    Returns:
        kkt_error (float): Square root of KKT RSS (Residual Sum of Square).
    
    Note:
        This function evaluate barrier-KKT error. If mu is set 0, \
            exact KKT error is evaluated.
    """
    N = us.shape[0]
    n_g = ss.shape[1]

    kkt_error = 0.0

    dyn_error = 0.0
    ineq_error = 0.0
    Lx_error = 0.0
    Lu_error = 0.0
    cmpl_error = 0.0

    # initial state
    res = x0 - xs[0]
    dyn_error += np.sum(np.abs(res) ** ord)

    mu1 = mu * np.ones(n_g)

    for i in range(N):
        # variables at state i
        x = xs[i]
        u = us[i]
        s = ss[i]
        lamx = lamxs[i]
        lams = lamss[i]
        x1 = xs[i + 1]
        lamx1 = lamxs[i + 1]
        t = t0 + i * dt
        # gradient of hamiltonian
        Hx = lx(x, u, t) + lamx1.T @ fx(x, u, t)
        Hu = lu(x, u, t) + lamx1.T @ fu(x, u, t)

        # dynamics
        res = x + f(x, u, t) * dt - x1
        dyn_error += np.sum(np.abs(res) ** ord)

        # inequality constraints with slack variables.
        res = g(x, u, t) + s
        ineq_error += np.sum(np.abs(res) ** ord)

        # Lx
        res = -lamx + lamx1 + Hx * dt + gx(x, u, t).T @ lams
        Lx_error += np.sum(np.abs(res) ** ord)

        # Lu
        res = Hu * dt + gu(x, u, t).T @ lams
        Lu_error += np.sum(np.abs(res) ** ord)

        # complementary condition
        res = lams * s - mu1
        cmpl_error += np.sum(np.abs(res) ** ord)

    # Lx[N]
    res = lfx(xs[N], t0 + N * dt) - lamxs[N]
    Lx_error += np.sum(np.abs(res) ** ord)

    kkt_error = dyn_error + ineq_error + Lx_error + Lu_error + cmpl_error

    return kkt_error ** (1.0 / ord)


@numba.njit
def eval_dynamics_error(f, t0, x0, dt, xs, us, ord=2):
    """ Evaluate feasibility of dynamics.

    Returns:
        dynamics_error (float): Square root of dynamics RSS.
    """
    N = us.shape[0]

    res = x0 - xs[0]
    dynamics_error = np.sum(np.abs(res) ** ord)

    for i in range(N):
        res = xs[i] + f(xs[i], us[i], t0 + i*dt) * dt - xs[i + 1]
        dynamics_error += np.sum(np.abs(res) ** ord)
    
    return dynamics_error ** (1.0 / ord)


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
