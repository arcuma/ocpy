import numpy as np
import os
from os.path import abspath, dirname, join


class Logger:
    """ Save simulation data.
    """

    def __init__(self, log_dir: str=None):
        """ Constructor.

        Args:
            log_dir (str): Directory in which data will be saved.
        """
        if log_dir == None:
            self._log_dir = join(dirname(dirname(abspath(__file__))), 'log')
        else:
            self._log_dir = abspath(log_dir)

    def save(self, xs: np.ndarray, us: np.ndarray, ts: np.ndarray, 
             cost_hist: np.ndarray=None, kkt_error_hist: np.ndarray=None):
        """ Save simulation data at log_dir.

        Args:
            xs (np.ndarray): State trajectory.
            us (np.ndarray): Control input trajectory.
            ts (np.ndarray): Time at each stage.
            cost_hist (numpy.ndarray=None): costs of each iteration.
            kkt_error_hist (numpy.ndarray=None): KKT error of each iteration .

        Returns:
            log_dir (str): Target directory
        """
        log_dir = self._log_dir
        os.makedirs(log_dir, exist_ok=True)

        with open(join(log_dir, 'x_log.txt'), mode='w') as x_log:
            np.savetxt(x_log, xs)

        with open(join(log_dir, 'u_log.txt'), mode='w') as u_log:
            np.savetxt(u_log, us)

        with open(join(log_dir, 't_log.txt'), mode='w') as t_log:
            np.savetxt(t_log, ts)

        if cost_hist is not None:
            with open(join(log_dir, 'cost_log.txt'), mode='w') as cost_log:
                np.savetxt(cost_log, cost_hist)
        elif os.path.isfile(join(log_dir, 'cost_log.txt')):
            os.remove(join(log_dir, 'cost_log.txt'))

        if kkt_error_hist is not None:
            with open(join(log_dir, 'kkt_log.txt'), mode='w') as cost_log:
                np.savetxt(cost_log, kkt_error_hist)
        elif os.path.isfile(join(log_dir, 'kkt_log.txt')):
            os.remove(join(log_dir, 'kkt_log.txt'))

        print("Data are saved at " + log_dir)

        return log_dir
