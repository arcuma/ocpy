import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import seaborn as sns
import sys
from os.path import join


class CartPoleAnimator:
    """ Class of generating animation of cartpole.
    """
    def __init__(self, log_dir: str, sim_name: str='cartpole'):
        """ Constructor.
        
        Args:
            log_dir (str): Direcory in which logs are stored.
            sim_name (str): Simulation name.
        """
        # Load data
        self._log_dir = log_dir
        self._sim_name = sim_name
        self._xs = np.genfromtxt(join(log_dir, 'x_log.txt'))
        self._us = np.genfromtxt(join(log_dir, 'u_log.txt'))
        self._ts = np.genfromtxt(join(log_dir, 't_log.txt'))
        if self._xs.ndim == 1:
            self._xs = self._xs.reshape((-1, 1))
        if self._us.ndim == 1:
            self._us = self._us.reshape((-1, 1))
        # when OC, us=(u0, ... ,uN-1) while xs=(x0, ..., xN-1)
        if self._us.shape[0] == self._xs.shape[0] - 1:
            self._us = np.append(self._us, [self._us[-1]], axis=0)
        self._n_x = self._xs.shape[1]
        self._n_u = self._us.shape[1]
        self._N = self._ts.size - 1
        self._dtau = self._ts[1] - self._ts[0]
        # cartpole
        self._cart_width = 0.5
        self._cart_height = 0.25
        self._pole_length = 0.5
        self._ball_r = 0.050
        # drawing range
        x_axis_lim = max(abs(np.amax(self._xs[:, 0])),
                    abs(np.amin(self._xs[:, 0])),
                    1.0) + self._pole_length + self._ball_r
        self._x_min = -x_axis_lim
        self._x_max = x_axis_lim
        scale = 2 * x_axis_lim
        r = 0.3
        aspect = 9/16
        self._y_min = -scale * r * aspect
        self._y_max = +scale * (1-r) * aspect
        # frame skip rate
        self._skip_rate = 1
        self._play_frames = (int)(self._ts.size / self._skip_rate)

    
    def generate_animation(self, save=True):
        """ Genarating animation.

        Args:
            save (bool): If True, animation is saved to log_dir.
        """
        self._fig = plt.figure(figsize=(16, 9))
        self._ax = plt.axes(xlim=(self._x_min, self._x_max),
                      ylim=(self._y_min, self._y_max))
        # 1:1 aspect ration
        self._ax.set_aspect('equal')
        state = self._xs[0, :]
        control = self._us[0, :]
        # cart position
        cart_center_x = state[0]
        cart_lu_x = cart_center_x - self._cart_width / 2
        cart_lu_y = 0
        cart_center_y = cart_lu_y + self._cart_height / 2
        # cart
        self._cart = patches.Rectangle(
            xy=(cart_lu_x, cart_lu_y), width=self._cart_width,
            height=self._cart_height
            )
        # pole
        l = self._pole_length
        pole_s_x = cart_center_x
        pole_s_y = cart_center_y
        pole_t_x = pole_s_x + l * np.sin(state[1])
        pole_t_y = pole_s_y - l * np.cos(state[1])
        self._pole = patches.Polygon(
            xy=[[pole_s_x, pole_s_y], [pole_t_x, pole_t_y]],
            ec='blue',
            linewidth=3.0
            )
        # weight
        self._ball = patches.Circle(
            xy=(pole_t_x, pole_t_y), radius=self._ball_r
            )
        # force arrow
        self._arrow = patches.Arrow(
            x=pole_s_x,
            y=pole_s_y,
            dx=control[0] * 0.05,
            dy=0.0,
            width=0.1,
            color='red',
            alpha=0.5
            )
        # add shapes
        self._ax.add_patch(self._cart)
        self._ax.add_patch(self._ball)        
        self._ax.add_patch(self._pole)
        self._ax.add_patch(self._arrow)
        self._groud = self._ax.axhline(y=0)
        # time display
        self._time_text = self._ax.text(
            0.9, 0.05,
            f'{self._ts[0]} [s]',
            transform=self._ax.transAxes,
            fontsize=16
            )
        self._variable_text = self._ax.text(
            x=0.88, y=0.72,
            s=r'$x: $'+f'{state[0]:.3f}\n'
                +r'$\theta$: '+f'{state[1]:.3f}\n'
                +r'$\dot{x}$: '+f'{state[2]:.3f}\n'
                +r'$\dot{\theta}$: '+f'{state[3]:.3f}\n'
                +r'$u$: '+f'{control[0]:.3f}\n',
            transform=self._ax.transAxes,
            fontsize=16
            )
        anim = FuncAnimation(
            self._fig,
            self._update_animation,
            frames=self._play_frames,
            interval=1000*self._dtau*self._skip_rate,
            blit=True
            )
        # save movie
        if save:
            anim.save(join(self._log_dir, self._sim_name) + '.mp4', dpi=120,
            writer='ffmpeg',
            fps=int(1/(self._dtau * self._skip_rate))
            )
            print('Animation was saved at ' + self._log_dir + ' .')
        plt.show()


    def _update_animation(self, i):
        """ callback function handed to FuncAnimation.
        """
        # current frame
        frame = i * self._skip_rate
        state = self._xs[frame, :]
        control = self._us[frame, :]
        # cart position
        cart_center_x = state[0]
        cart_lu_x = cart_center_x - self._cart_width / 2
        cart_lu_y = 0
        cart_center_y = cart_lu_y + self._cart_height / 2
        self._cart.set_x(cart_lu_x)
        # pole position
        pole_s_x = cart_center_x
        pole_s_y = cart_center_y
        pole_t_x = pole_s_x + self._pole_length * np.sin(state[1])
        pole_t_y = pole_s_y - self._pole_length * np.cos(state[1])
        self._pole.set_xy([[pole_s_x, pole_s_y], [pole_t_x, pole_t_y]])
        # weight position
        self._ball.set_center((pole_t_x, pole_t_y))
        # remake and repatch arrow
        self._arrow.remove()
        self._arrow = patches.Arrow(
            x=pole_s_x,
            y=pole_s_y,
            dx=control[0] * 0.05,
            dy=0.0,
            width=0.1,
            color='red',
            alpha=0.5
        )
        self._ax.add_patch(self._arrow)
        # time text
        self._time_text.set_text(
            '{0:.1f} [s]'.format(frame * self._dtau)
            )
        # variable text
        self._variable_text.set_text(
            r'$x: $'+f'{state[0]:5.3f}\n'
            +r'$\theta$: '+f'{state[1]:.3f}\n'
            +r'$\dot{x}$: '+f'{state[2]:.3f}\n'
            +r'$\dot{\theta}$: '+f'{state[3]:.3f}\n'
            +r'$u$: '+f'{control[0]:.3f}\n'
            )
        return (self._cart, self._pole, self._ball, self._arrow,
                self._time_text, self._variable_text)


# test
if __name__ == '__main__':
    log_dir = '/home/ohtsukalab/src/DDPython/log/cartpole'
    sim_name = 'cartpole'
    anim = CartPoleAnimator(log_dir, sim_name)
    anim.generate_animation(save=False)
