import sympy as sym
import numpy as np
import abc


class SolverBase(abc.ABC):
    """ Abstract solver class.
    """
    def __init__(self, ocp):
        super().__init__()
        self._ocp = ocp
        pass


    @abc.abstractmethod
    def solve(self):
        pass

    @abc.abstractmethod
    def log_directory(self):
        pass
