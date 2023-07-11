import numpy as np
import math
import os
from os.path import abspath, dirname, join
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


class Plotter:
    """ Class of plotting simulation results.
    """
    def __init__(self, log_dir: str, sim_name: str, ts: np.ndarray,
                 xs: np.ndarray, us: np.ndarray, J_hist: np.ndarray):
        """ Constrctor.

        Args:
            log_dir (str): Direcory in which graph will be saved.
            sim_name (str): Simulation name.
            ts (np.ndarray): Discrete time history.
            xs (np.ndarray): State trajectory.
            us (np.ndarray): Control input trajectory.
        """
        # hold data
        N = ts.size - 1        
        self._N = N        
        self._log_dir = log_dir
        self._sim_name = sim_name
        self._ts = ts
        self._xs = xs if xs.ndim > 1 else xs.reshape((N + 1, 1))
        self._us = us if us.ndim > 1 else us.reshape((N, 1))
        self._J_hist = J_hist
        self._dtau = ts[1] - ts[0]
        self._n_x = xs.shape[1] if xs.ndim > 1 else 1
        self._n_u = us.shape[1] if us.ndim > 1 else 1
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
    def from_log(log_dir: str, sim_name: str=None):
        """ Create instance from logs.

        Args:
            log_dir (str): Direcory in which logs are stored.
            sim_name (str): Simulation name.
        """
        ts = np.genfromtxt(join(log_dir, 't_log.txt'))
        xs = np.genfromtxt(join(log_dir, 'x_log.txt'))
        us = np.genfromtxt(join(log_dir, 'u_log.txt'))
        J_hist = np.genfromtxt(join(log_dir, 'J_log.txt'))
        return Plotter(log_dir, sim_name, ts, xs, us, J_hist)

    
    def plot(self, fig_scale=3, font_scale=1, wspace_scale=0.3, hspace_scale=2,
             show=True, save=False):
        """ Plotting data.

        Args:
            fig_scale (float): Figure scale.
            font_scale (float): Font scale.
            wspace_scale (float): Horizontal space scale.
            hspace_scale (float): Vertical space scale.
            show (bool): If true, display graph.
            save (bool): If true, save graph to log_dir.
        """
        plt.rcParams['font.size'] = 24
        # variables
        n_x = self._n_x
        n_u = self._n_u
        N = self._N
        ts = self._ts
        xs = self._xs
        us = self._us
        J_hist = self._J_hist
        # columns of graph
        cols = math.ceil(math.sqrt(n_x + n_u))
        # rows of graph. 
        rows_x = math.ceil(n_x / cols)
        rows_u = math.ceil(n_u / cols)
        # for x, u, and J
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
                    axes[i][j].plot(ts, xs[:, idx])
                    axes[i][j].set_xlabel(r'${\rm Time} [s]$')
                    axes[i][j].set_ylabel(r'$x_{' + str(idx) + r'}$')
                    axes[i][j].set_xlim(ts[0], ts[N])
                else:
                    fig.delaxes(axes[i][j])
        # control
        for i in range(rows_x, rows_x + rows_u):
            for j in range(cols):
                idx = (i - rows_x)*cols + j
                if idx < n_u:
                    axes[i][j].plot(ts[:-1], us[:, idx])
                    axes[i][j].set_xlabel(r'${\rm Time} [s]$')
                    axes[i][j].set_ylabel(r'$u_{' + str(idx) + r'}$')
                    axes[i][j].set_xlim(ts[0], ts[N])
                else:
                    fig.delaxes(axes[i][j])
        # cost value
        for j in range(cols):
            i = rows_x + rows_u
            if j == 0:
                iter = J_hist.size
                iters = np.arange(iter)
                axes[i][j].plot(iters, J_hist)
                axes[i][j].set_xlabel(r'${\rm iteration}$')
                axes[i][j].set_ylabel(r'$J$')
                axes[i][j].set_yscale('log')
                axes[i][j].set_xlim(0, iter-1)
                axes[i][j].xaxis.set_major_locator(MaxNLocator(integer=True))
            else:
                fig.delaxes(axes[i][j])
        if save:
            plt.savefig(join(self._log_dir, 'result.pdf'),
                        bbox_inches='tight', pad_inches=0.1)
            print('Graphs are saved at ' + join(self._log_dir, 'result.pdf'))
        if show:
            plt.show()


# test
if __file__ == '__main__':
    sim_name = 'lqr'
    log_dir = join(dirname(dirname(abspath(__file__))), 'log', sim_name)
    plotter = Plotter.from_log(log_dir, sim_name)
    plotter.plot(save=True)


if False:
    sim_name = 'lqr'
    log_dir = join(dirname(dirname(abspath(__file__))), 'log')
    # ts = np.genfromtxt(join(log_dir, sim_name, 't_log.txt'))
    # xs = np.genfromtxt(join(log_dir, sim_name, 'x_log.txt'))
    # us = np.genfromtxt(join(log_dir, sim_name, 'u_log.txt'))
    # Js = np.genfromtxt(join(log_dir, sim_name, 'J_log.txt'))
    
    print(log_dir)
    plotter = Plotter('lqr', log_dir)
    num_graphs = plotter._num_graphs
    print('num_graphs: ', num_graphs)

    n_x, n_u = plotter._n_x, plotter._n_u
    ts = plotter._ts
    xs = plotter._xs
    us = plotter._us
    us = us.reshape((us.size, 1))
    
    J_hist = plotter._J_hist

    # rows and colums of subplot
    col = math.ceil(math.sqrt(n_x + n_u))
    rows_x = math.ceil(n_x / col)
    row_u = math.ceil(n_u / col)
    row = rows_x + row_u + 1

    fig, axes = plt.subplots(row, col)
    print(row, col)

    #plot
    # state
    for i in range(rows_x):
        for j in range(col):
            idx = i*col + j
            if idx < n_x:
                axes[i][j].plot(ts, xs[:, idx])
            else:
                fig.delaxes(axes[i][j])
    # control
    for i in range(rows_x, rows_x + row_u):
        for j in range(col):
            idx = (i - rows_x)*col + j
            if idx < n_u:
                axes[i][j].plot(ts[:-1], us[:, idx])
            else:
                fig.delaxes(axes[i][j])
    # cost value
    for j in range(col):
        i = rows_x + row_u
        if j == 0:
            iter = J_hist.size
            iters = np.arange(iter)
            axes[i][j].plot(iters, J_hist)
        else:
            fig.delaxes(axes[i][j])
    plt.show()
    
