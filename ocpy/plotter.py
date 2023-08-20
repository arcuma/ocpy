import numpy as np
import math
import os
from os.path import abspath, dirname, join, isfile
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import warnings


class Plotter:
    """ Class of plotting simulation results.
    """
    def __init__(self, log_dir: str, 
                 xs: np.ndarray, us: np.ndarray,  ts: np.ndarray, 
                 cost_hist: np.ndarray=None, kkt_error_hist: np.ndarray=None):
        """ Constructor.

        Args:
            log_dir (str): Direcory in which graph will be saved.
            xs (np.ndarray): State trajectory.
            us (np.ndarray): Control input trajectory.
            ts (np.ndarray): Discrete Time at each stage.
            cost_hist (numpy.ndarray=None): costs of each iteration.
            kkt_error_hist (numpy.ndarray=None): KKT error of each iteration .
        """
        # hold data
        N = ts.size - 1
        self._N_x = xs.shape[0]
        self._N_u = us.shape[0]
        self._N_t = ts.size
        if self._N_t != self._N_x:
            warnings.warn("Size of time and state is different.")
        if self._N_u not in(self._N_x - 1, self._N_x):
            warnings.warn("Size of input should be equal to (or 1 less than) "\
                          "state.")
        self._log_dir = abspath(log_dir)
        self._ts = ts
        self._xs = xs if xs.ndim > 1 else xs.reshape((-1, 1))
        self._us = us if us.ndim > 1 else us.reshape((-1, 1))
        self._costs = cost_hist
        self._kkts = kkt_error_hist
        self._dtau = ts[1] - ts[0]
        self._n_x = self._xs.shape[1]
        self._n_u = self._us.shape[1]
        self._num_graphs = self._n_x + self._n_u + 1
        # set style
        sns.set_style('ticks')
        sns.set_palette('deep')
        sns.set_context('paper')
        mathtext = {'rm': 'serif',
                    'it': 'serif:itelic',
                    'bf': 'serif:bold',
                    'fontset': 'cm'}
        plt.rc('mathtext', **mathtext)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.rcParams['lines.linewidth'] = 1
        plt.rcParams['axes.linewidth'] = 0.5
    

    @staticmethod
    def from_log(log_dir: str):
        """ Create instance from logs.

        Args:
            log_dir (str): Direcory in which logs are stored.
            dat_name (str): Name of dat_vs_iters.
        """
        log_dir = abspath(log_dir)
        xs = np.genfromtxt(join(log_dir, 'x_log.txt'))
        us = np.genfromtxt(join(log_dir, 'u_log.txt'))
        ts = np.genfromtxt(join(log_dir, 't_log.txt'))
        if isfile(join(join(log_dir, 'J_log.txt'))):
            costs = np.genfromtxt(join(log_dir, 'J_log.txt'))
        else:
            costs = None
        if isfile(join(join(log_dir, 'kkt_log.txt'))):
            kkts = np.genfromtxt(join(log_dir, 'kkt_log.txt'))
        else:
            kkts = None
        return Plotter(log_dir, xs, us, ts, costs, kkts)

    
    def plot(
            self, fig_scale=4.0, font_scale_label=2.0, font_scale_ticks=1.3,
            wspace_scale=1.0, hspace_scale=2.3,
            show=True, save=False
        ):
        """ Plotting data.

        Args:
            fig_scale (float): Figure scale.
            font_scale_label (float): Font scale of axis label.
            font_scale_ticks (float): Font scale of axis ticks.
            wspace_scale (float): Horizontal space scale.
            hspace_scale (float): Vertical space scale.
            show (bool): If true, display graph.
            save (bool): If true, save graph to log_dir.
        """
        plt.rcParams['font.size'] = 24
        # variables
        n_x = self._n_x
        n_u = self._n_u
        N_x = self._N_x
        N_u = self._N_u
        N_t = self._N_t
        ts = self._ts
        xs = self._xs
        us = self._us
        costs = self._costs
        kkts = self._kkts
        # number of columns of graph
        cols = math.ceil(math.sqrt(n_x + n_u))
        # number of rows of graph. 
        rows_x = math.ceil(n_x / cols)
        rows_u = math.ceil(n_u / cols)
        rows = rows_x + rows_u + 1
        # rows * cols 
        fig, axes = plt.subplots(rows, cols)
        # figure size
        fig.set_figheight(fig_scale * rows)
        fig.set_figwidth(2.5 * fig_scale * cols)
        # space between graphs
        fig.subplots_adjust(wspace=wspace_scale/cols, hspace=hspace_scale/rows)
        # state
        for i in range(rows_x):
            for j in range(cols):
                idx = i*cols + j
                if idx < n_x:
                    axes[i][j].plot(ts[:N_x], xs[:N_x, idx])
                    axes[i][j].tick_params(labelsize=10*font_scale_ticks)
                    axes[i][j].set_xlabel(r'${\rm Time} [s]$', 
                                          fontsize=10*font_scale_label)
                    axes[i][j].set_ylabel(r'$x_{' + str(idx) + r'}$',
                                          fontsize=10*font_scale_label)
                    axes[i][j].set_xlim(ts[0], ts[-1])
                else:
                    fig.delaxes(axes[i][j])
        # control
        for i in range(rows_x, rows_x + rows_u):
            for j in range(cols):
                idx = (i - rows_x)*cols + j
                if idx < n_u:
                    axes[i][j].plot(ts[:N_u], us[:N_u, idx])
                    axes[i][j].tick_params(labelsize=10*font_scale_ticks)
                    axes[i][j].set_xlabel(r'${\rm Time} [s]$',
                                          fontsize=10*font_scale_label)
                    axes[i][j].set_ylabel(r'$u_{' + str(idx) + r'}$',
                                          fontsize=10*font_scale_label)
                    axes[i][j].set_xlim(ts[0], ts[-1])
                else:
                    fig.delaxes(axes[i][j])
        # costs and kkts
        for j in range(cols):
            i = rows_x + rows_u
            if j == 0 and costs is not None:
                num_iters = costs.size - 1
                arr_iters = np.arange(num_iters + 1)
                axes[i][j].plot(arr_iters, costs)
                axes[i][j].tick_params(labelsize=10*font_scale_ticks)
                axes[i][j].set_xlabel(r'${\rm iteration}$',
                                      fontsize=10*font_scale_label)
                axes[i][j].set_ylabel(r'$J$',
                                      fontsize=10*font_scale_label)
                axes[i][j].set_yscale('log')
                axes[i][j].set_xlim(0, num_iters)
                axes[i][j].xaxis.set_major_locator(MaxNLocator(integer=True))
            elif j == 1 and kkts is not None:
                num_iters = kkts.size - 1
                arr_iters = np.arange(num_iters + 1)
                axes[i][j].plot(arr_iters, kkts)
                axes[i][j].tick_params(labelsize=10*font_scale_ticks)
                axes[i][j].set_xlabel(r'${\rm iteration}$',
                                      fontsize=10*font_scale_label)
                axes[i][j].set_ylabel(r'$\rm{KKT error}$',
                                      fontsize=10*font_scale_label)
                axes[i][j].set_yscale('log')
                axes[i][j].set_xlim(0, num_iters)
                axes[i][j].xaxis.set_major_locator(MaxNLocator(integer=True))
            else:
                fig.delaxes(axes[i][j])
        if save:
            os.makedirs(self._log_dir, exist_ok=True)
            plt.savefig(join(self._log_dir, 'result.pdf'),
                        bbox_inches='tight', pad_inches=0.1)
            print('Graphs are saved at ' + join(self._log_dir, 'result.pdf'))
        if show:
            plt.show()


# test
if __name__ == '__main__':
    # sim_name = 'hexacopter'
    sim_name = 'cartpole'
    log_dir = join(dirname(dirname(abspath(__file__))), 'log', sim_name)
    plotter = Plotter.from_log(log_dir)
    plotter.plot(save=False)
    print('a')
