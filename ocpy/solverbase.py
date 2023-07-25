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
        """ Return OCP.
        """
        return self._ocp

    def set_log_directory(self, log_dir: str):
        """ Set directory path of data are logged.
        """
        self._log_dir = log_dir

    def get_log_directory(self) -> str:
        """ Get directory path of data are logged.
        """
        return self._log_dir

    @abc.abstractmethod
    def set_solver_parameters(self):
        """ Set solver parameters.
        """
        pass

    @abc.abstractmethod
    def reset_initial_conditions(self):
        """ Reset t0, x0, initial guess of us, etc.
        """
        pass

    @abc.abstractmethod
    def solve(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def print_result():
        pass

    @staticmethod
    @abc.abstractmethod
    def log_data():
        pass

    @staticmethod
    @abc.abstractmethod
    def plot_data():
        pass
