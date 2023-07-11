import sympy as sy
import numpy as np
import time
from os.path import join, abspath, dirname

from ocpy import symutils
from ocpy.logger import Logger
from ocpy.plotter import Plotter
from ocpy.ocp import OCP


class DDP:
    """ Differential Dynamic Programming(DDP) solver.
    """
    def __init__(self, ocp: OCP):
        """ set optimal control problem.
        """
        # OCP
        self._ocp = ocp
        self._sim_name = ocp.get_ocp_name()
        self._log_dir = join(dirname(dirname(abspath(__file__))), 'log',
                             self._sim_name)
        # dimensions
        self._n_x = ocp.get_n_x()
        self._n_u = ocp.get_n_u()
        self._T = ocp.get_T()
        self._N = ocp.get_N()
        self._dt = self._T / self._N
        # x0 and initial us
        self._x0 = ocp._x0
        self._us_guess = ocp._us_guess
        # functions of derivatives.
        self._df, self._dl = ocp.get_derivatives()
        self._f, self._fx, self._fu, self._fxx, self._fux, self._fuu = self._df
        self._l, self._lx, self._lu, self._lxx, self._lux, self._luu, \
            self._lf, self._lfx, self._lfxx = self._dl

    def log_directory(self):
        """ Returns target directory where logs are saved.
        """
        return self._log_dir

    def solve(self, t0: float, x0: np.ndarray, us_guess: np.ndarray, max_iter=10,
              alphas=0.5**np.arange(5), damp_init=1e-3, damp_min=1e-3,
              damp_max=1e3, log=False):
        """ Solve OCP via DDP iteration.

        Args:
            t0 (float): initial time.
            x0 (numpy.array): initial state. size must be n_x.
            us_guess (numpy.array): guess of input trajectory. \
                size must be (N * n_u).
            max_iter (int): maximum iteration.
            alphas (np.array): step sizes of line search.
            damp_init (float): initial damping value for backward pass
            damp_min (float): minimum dapming value.
            damp_max (float): maximum damping value.
            log (bool): If true, results are logged to log_dir.
        
        Returns:
            ts (numpy.ndarray): time history.
            xs (numpy.ndarray): optimal state trajectory. (N * n_x)
            us (numpy.ndarray): optimal control trajectory. (N * n_u)
            Js (numpy.ndarray): costs at each iteration.

        """
        time_start = time.perf_counter()
        x0 = np.array(x0)
        us = np.array(us_guess)
        N = self._N
        T = self._T
        # initial rollout
        xs, J = self.rollout(x0, us, t0)
        Js = [J]
        damp = damp_init
        # success flag
        is_success = False
        # iteration
        for iter in range(max_iter):
            print(f'iter: {iter}')
            ks, Ks, Delta_V0 = self.backward_pass(self._df, self._dl, xs, us,
                                                  t0, T, damp)
            if np.abs(Delta_V0) < 1e-5:
                is_success = True
                break
            print(Delta_V0)    
                # break
            alphas = np.array([0.5 ** i for i in range(5)] + [0])
            # line search
            for alpha in alphas:
                xs_new, us_new, J_new = self.forward_pass(
                    self._f, self._l, self._lf, xs, us, t0, T, ks, Ks, alpha)
                # print(f'iter: {iter}, alpha: {alpha}, J: {J}, J_new: {J_new}')
                if J_new < J:
                    # line search successed
                    xs = xs_new
                    us = us_new
                    J = J_new
                    damp *= 0.5
                    break
            else:
                # failed in line search
                damp *= 2.0
                damp = min(max(damp, damp_min), damp_max)
            Js.append(J)
        # time grids
        ts = np.array([i*self._dt for i in range(N + 1)])
        # transform Js from list into np.ndarray (for np.append() is slow)
        Js = np.array(Js)
        # computational time
        time_end = time.perf_counter()
        time_elapsed = time_end - time_start
        # results
        print('---')
        if is_success:
            status = 'success'
        else:
            status = 'fail'
        print(f'status: {status}')
        print(f'iteration: {iter}')
        print(f'cost value: {Js[-1]}')
        print(f'computational time: {time_elapsed}')
        print('---')
        # log
        if log:
            self._log_dir = join(dirname(dirname(abspath(__file__))), 'log',
                           self._sim_name)
            logger = Logger(self._log_dir)
            logger.save(ts, xs, us, Js)
            # plot
            plotter = Plotter.from_log(self._log_dir, self._sim_name)
            plotter.plot(save=True)
        return ts, xs, us, Js

    def rollout(self, x0: np.ndarray, us: np.ndarray, t0: float=0.0):
        """ Rollout state trajectory from initial state and input trajectory.
        Args:
            x0 (np.ndarray): Initial state.
            us (np.ndarray): Input control trajectory.
            t0 (float): initial time.
        """
        f = self._f
        l, lf = self._l, self._lf
        N = self._N
        dt = self._dt
        # time, state trajectory and cost
        xs = np.zeros((N + 1, x0.shape[0]))
        xs[0] = x0
        J = 0.0
        for i in range(N):
            xs[i + 1] = f(xs[i], us[i], t0 + i*dt)
            J += l(xs[i], us[i], t0 + i*dt)
        J += lf(xs[N], t0 + N*dt)
        return xs, J

    def backward_pass(self, df, dl, xs: np.ndarray, us: np.ndarray,
                      t0: float, T: float, damp: float=1e-6):
        """ backward pass of DDP.
        Args:
            df (list): Derivatives of state function, f, fx, fu, fxx, fux, fuu.
            dl (list): Derivatives of state function, \
                l, lx, lu, lxx, lux, luu, lf, lfx, lfxx.
            xs (numpy.ndarray): nominal state trajectory.\
                size must be (N+1)*n_u
            us (numpy.ndarray): nominalcontrol trajectory.\
                size must be N*n_u
            damp (float): damping coefficient of LM method.
        Returns:
            ks (numpy.ndarray): series of k. Its size is N * n_u
            Ks (numpy.ndarray): series of K. Its size is N * (n_u * n_x)
            Delta_V (float): expecting change of value function at stage 0.
        """
        N = us.shape[0]
        n_x = xs.shape[1]
        n_u = us.shape[1]
        dt = T / N
        # functions of derivatives
        f_f, fx_f, fu_f, fxx_f, fux_f, fuu_f = df 
        l_f, lx_f, lu_f, lxx_f, lux_f, luu_f, lf_f, lfx_f, lfxx_f = dl 
        # feedforward term and feedback coeff.
        ks = np.empty((N, n_u))
        Ks = np.empty((N, n_u, n_x))
        # value function at stage N
        Vx = lfx_f(xs[N], t0 + T)
        Vxx = lfxx_f(xs[N], t0 + T)
        # expected cost cahnge of total stage
        delta_V = 0
        # daming term.
        Reg = damp * np.eye(n_u)
        for i in range(N - 1, -1, -1):
            t = t0 + i*dt
            # x and u at satge i
            x, u = xs[i], us[i]
            # values of derivatives at stage i
            fx, fu, fxx, fux, fuu = fx_f(x, u, t), fu_f(x, u, t), fxx_f(x, u, t),\
                fux_f(x, u, t), fuu_f(x, u, t)
            lx, lu, lxx, lux, luu = lx_f(x, u, t), lu_f(x, u, t), lxx_f(x, u, t),\
                lux_f(x, u, t), luu_f(x, u, t)
            # action value derivatives
            Qx = lx + fx.T @ Vx
            Qu = lu + fu.T @ Vx
            Qxx = lxx + fx.T @ Vxx @ fx + Vx @ fxx.T
            Qux = lux + fu.T @ Vxx @ fx + Vx @ fux.T
            Quu = luu + fu.T @ Vxx @ fu + Vx @ fuu.T
            # feedforward and feedback terms
            Quu_inv = np.linalg.inv(Quu + Reg)
            k = -Quu_inv @ Qu
            K = -Quu_inv @ Qux
            ks[i] = k
            Ks[i] = K
            # value function of stage i
            Vx = Qx - K.T @ Quu @ k
            Vxx = Qxx - K.T @ Quu @ K
            delta_V += 0.5 * k.T @ Quu @ k + k.T @ Qu
        return ks, Ks, delta_V

    def forward_pass(self, f, l, lf, xs: np.ndarray, us: np.ndarray,
                     t0: float, T: float,
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
            ks (numpy.ndarray): series of k. Size must be N * n_u
            Ks (numpy.ndarray): series of K. Size must be N * (n_u * n_x)
            alpha (float): step size of line search. 0<= alpha <= 1.0
        Returns:
            xs_new (numpy.ndarray): new state trajectory.
            us_new (numpy.ndarray): new control trajectory.
            J_new (float) cost along with (xs_new, us_new).
        """
        N = us.shape[0]
        dt = T / N
        # new (xs, us) and cost
        xs_new = np.empty(xs.shape)
        xs_new[0] = xs[0]
        us_new = us.copy()
        J_new = 0.0
        for i in range(N):
            t = t0 + i*dt
            us_new[i] += alpha * ks[i] + Ks[i] @ (xs_new[i] - xs[i])
            xs_new[i + 1] = f(xs_new[i], us_new[i], t)
            J_new += l(xs_new[i], us_new[i], t)
        # terminal cost
        J_new += lf(xs_new[N], t + T)
        return xs_new, us_new, J_new
    