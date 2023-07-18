import numpy as np
import os
from os.path import abspath, dirname, join


class Logger:
    """ Class of saving simulation data .
    """

    def __init__(self, log_dir: str=None):
        """ Constructor.

        Args:
            log_dir (str): Directory in which data will be saved.
        """
        if log_dir == None:
            log_dir = join(dirname(dirname(abspath(__file__))), 'log')
        else:
            log_dir = abspath(log_dir)
        self._log_dir = log_dir

    def save(self, ts, xs, us, J_hist):
        """ Save simulation data at log_dir.

        Args:
            ts (np.ndarray): Time history.
            xs (np.ndarray): State trajectory.
            us (np.ndarray): Control input trajectory.
            Js (np.ndarray): Cost value of each iteration.

        Returns:
            log_dir (str): Target directory
        """
        log_dir = self._log_dir
        os.makedirs(log_dir, exist_ok=True)
        # save
        with open(join(log_dir, 't_log.txt'), mode='w') as t_log:
            np.savetxt(t_log, ts)
        with open(join(log_dir, 'x_log.txt'), mode='w') as x_log:
            np.savetxt(x_log, xs)
        with open(join(log_dir, 'u_log.txt'), mode='w') as u_log:
            np.savetxt(u_log, us)
        with open(join(log_dir, 'J_log.txt'), mode='w') as J_log:
            np.savetxt(J_log, J_hist)
        
        print("Data are saved at " + log_dir)
        return log_dir
        