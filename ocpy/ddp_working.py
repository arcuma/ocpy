import sympy as sy
import numpy as np

from ocpy import symutils
from ocpy.ocp import OCP


class DDP:
    """ Differential Dynamic Programming(DDP) solver.
    """
    def __init__(self, ocp: OCP):
        """ set optimal control problem.
        """
        # OCP
        self._ocp = ocp
        # dimensions
        self._n_x = ocp.get_n_x()
        self._n_u = ocp.get_n_u()
        self._T = ocp.get_T
        self._N = ocp.get_N()
        self._dt = self._T / self._N
        # functions of derivatives.
        self._df, self._dl = ocp.get_derivatives()
        self._f, self._fx, self._fu, self._fxx, self._fux, self._fuu = self._df
        self._l, self._lx, self._lu, self._lxx, self._lux, self._luu \
            = self._lf, self._lfx, self._lfxx = self._df

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
        xs = np.zeros((N, x0.shape[0]))
        xs[0] = x0
        J = 0.0
        for i in range(N):
            xs[i + 1] = f(xs[i], us[i], t0 + i*dt)
            J += l(xs[i], us[i], t0 + i*dt)
        J += lf(xs, t0 + N*dt)
        return xs, J

    def forward_recursion(self, xs, us, ks, Ks, alpha=1.0):
        pass