import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.art3d as art3d
import seaborn as sns
import sys
from os.path import join


class CartPoleAnimator:
    """ Class of generating animation of cartpole.
    """
    def __init__(self, log_dir: str, file_name: str='cartpole'):
        """ Constructor.
        
        Args:
            log_dir (str): Direcory in which logs are stored.
            file_name (str): Animation is saved as (filename).mp4.
        """
        # Load data
        self._log_dir = log_dir
        self._file_name = file_name
        self._xs = np.genfromtxt(join(log_dir, 'x_log.txt'))
        self._us = np.genfromtxt(join(log_dir, 'u_log.txt'))
        self._ts = np.genfromtxt(join(log_dir, 't_log.txt'))
        if self._xs.ndim == 1:
            self._xs = self._xs.reshape((-1, 1))
        if self._us.ndim == 1:
            self._us = self._us.reshape((-1, 1))
        # when OC, us=(u0, ... ,uN-1) while xs=(x0, ..., xN)
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
        self._total_frames = (int)(self._ts.size / self._skip_rate)
    
    def generate_animation(self, save: bool=True, skip_rate: int=1):
        """ Genarating animation.

        Args:
            save (bool): If True, animation is saved to log_dir.
            skip_rate (int): Skip rate.
        """
        # frame skip
        self._skip_rate = skip_rate
        self._total_frames = (int)(self._ts.shape[0] / skip_rate)
        self._fig = plt.figure(figsize=(16, 9))
        self._ax = plt.axes(xlim=(self._x_min, self._x_max),
                            ylim=(self._y_min, self._y_max))
        # 1:1 aspect ration
        self._ax.set_aspect('equal')
        state = self._xs[0, :]
        control = self._us[0, :]
        # cart position
        cart_center_x = state[0]
        cart_ll_x = cart_center_x - self._cart_width / 2
        cart_ll_y = 0
        cart_center_y = cart_ll_y + self._cart_height / 2
        # cart
        self._cart = patches.Rectangle(
            xy=(cart_ll_x, cart_ll_y), width=self._cart_width,
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
            frames=self._total_frames,
            interval=1000*self._dtau*self._skip_rate,
            blit=True
            )
        # save movie
        if save:
            anim.save(join(self._log_dir, self._file_name) + '.mp4', dpi=120,
                      writer='ffmpeg',
                      fps=int(1/(self._dtau * self._skip_rate))
            )
            print('Animation was saved at ' + self._log_dir + ' .')
        plt.show()

    def _update_animation(self, i):
        """ Callback function handed to FuncAnimation.
        """
        # current frame
        frame = i * self._skip_rate
        state = self._xs[frame, :]
        control = self._us[frame, :]
        # cart position
        cart_center_x = state[0]
        cart_ll_x = cart_center_x - self._cart_width / 2
        cart_ll_y = 0
        cart_center_y = cart_ll_y + self._cart_height / 2
        self._cart.set_x(cart_ll_x)
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


class HexacopterAnimator:
    """ Class generating animation of hexacopter.
    """
    def __init__(self, log_dir, file_name: str='hexacopter'):
        """ Constructor.

        Args:
            log_dir (str): Directory in which logs are stored.
            file_name (str): Animation is saved as (filename).mp4.
        """
        self._log_dir = log_dir
        self._file_name = file_name
        self._ts = np.genfromtxt(join(log_dir, "t_log.txt"))
        self._xs = np.genfromtxt(join(log_dir, "x_log.txt"))
        self._us = np.genfromtxt(join(log_dir, "u_log.txt"))
        if self._xs.ndim == 1:
            self._xs = self._xs.reshape((-1, 1))
        if self._us.ndim == 1:
            self._us = self._us.reshape((-1, 1))
        # when OC, us=(u0, ... ,uN-1) while xs=(x0, ..., xN)
        if self._us.shape[0] == self._xs.shape[0] - 1:
            self._us = np.append(self._us, [self._us[-1]], axis=0)
        self._dt = self._ts[1] - self._ts[0]
        # size of hexacopter
        self._r_body=0.10
        self._r_prop=0.05
        self._l_arm=0.2
        # frame skip rate
        self._skip_rate = 1
        self._total_frames = (int)(self._ts.size / self._skip_rate)      

    def generate_animation(self, save: bool=True, skip_rate: int=1):
        """ Genarating animation.

        Args:
            save (bool): If True, animation is saved to log_dir.
            skip_rate (int): Skip rate.
        """
        # frame skip
        self._skip_rate = skip_rate
        self._total_frames = (int)(self._ts.shape[0] / skip_rate)
        # matplotlib
        self._fig = plt.figure(figsize=(13, 13))
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._ax.set_xlabel('x')
        self._ax.set_ylabel('y')
        self._ax.set_zlabel('z')
        self._ax.set_xlim(-4.0, 4.0)
        self._ax.set_ylim(-4.0, 4.0)
        self._ax.set_zlim(-0.0, 8.0)
        # initial state
        state = self._xs[0, :]
        # drone
        self._lines = []
        for i in range(6):
            line = art3d.Line3D([], [], [], color='tab:blue', linewidth=3.0)
            self._ax.add_line(line)
            self._lines.append(line)
        # body frame
        self._frame = [0] * 3
        self._framecolor = ['r', 'g', 'b']
        for i in range(3):
            self._frame[i] = self._ax.quiver(
                *np.zeros(3), *np.zeros(3), color=self._framecolor[i]
            )
        # time text
        self._ax.tick_params(color='white')
        self._time_text = self._ax.text2D(
            0.85,
            0,
            f'0.0 [s]',
            transform=self._ax.transAxes,
            fontsize=14
        )
        anim = FuncAnimation(
            self._fig,
            self._update_animation,
            frames=self._total_frames,
            interval=1000*self._dt*self._skip_rate,
            blit=False
        )
        if save:
            anim.save(join(self._log_dir, self._file_name) + '.mp4', dpi=120,
                      writer='ffmpeg',
                      fps=int(1/(self._dt * self._skip_rate))
            )
            print('Animation was saved at ' + self._log_dir + ' .')
        plt.show()

    def _update_animation(self, i):
        """ Callback function handed to FuncAnimation.
        """
        frame = self._skip_rate * i
        t = frame * self._dt
        state = self._xs[frame, :]
        self.draw_frame(t, state)

    def draw_frame(self, t, state):
        """ Draw a frame of a certain instant.
        """
        pos = state[0:3]
        rpy = state[3:6]
        wRb = self.rpy_to_rotmat(rpy)
        # drone
        for i in range(6):
            theta = (1/6 + i/3) * np.pi
            bl = self._l_arm * np.array([np.cos(theta), np.sin(theta), 0])
            dest = pos + wRb @ bl
            # print(pos, dir)
            self._lines[i].set_data_3d([pos[0], dest[0]],
                                       [pos[1], dest[1]],
                                       [pos[2], dest[2]]
            )
        # body frame
        for i in range(3):
            self._frame[i].remove()
            dest = wRb[:, i] * 0.8
            self._frame[i] = self._ax.quiver(
                *pos, *dest, color=self._framecolor[i]
            )
        # time text
        self._time_text.set_text(f'{t:.1f} [s]')
        
    @staticmethod
    def rpy_to_rotmat(rpy: np.ndarray):
        """ Transform Roll-Pitch-Yaw expression into rotation matrix.
        """
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_euler('xyz', rpy)
        return rotation.as_matrix()


class PendubotAnimator:
    """ Class of generating animation of pendubot.
    """
    def __init__(self, log_dir: str, file_name: str='pendubot'):
        """ Constructor.

        Args:
            log_dir (str): Directory in which logs are stored.
            file_name (str): Animation is saved as (filename).mp4
        """
        # Load data
        self._log_dir = log_dir
        self._file_name = file_name
        self._xs = np.genfromtxt(join(log_dir, 'x_log.txt'))
        self._us = np.genfromtxt(join(log_dir, 'u_log.txt'))
        self._ts = np.genfromtxt(join(log_dir, 't_log.txt'))
        if self._xs.ndim == 1:
            self._xs = self._xs.reshape((-1, 1))
        if self._us.ndim == 1:
            self._us = self._us.reshape((-1, 1))
        # when OC, us=(u0, ... ,uN-1) while xs=(x0, ..., xN)
        if self._us.shape[0] == self._xs.shape[0] - 1:
            self._us = np.append(self._us, [self._us[-1]], axis=0)
        self._n_x = self._xs.shape[1]
        self._n_u = self._us.shape[1]
        self._N = self._ts.size - 1
        self._dtau = self._ts[1] - self._ts[0]
        # drawing range
        self._x_min = -1
        self._x_max = 1
        self._y_min = -0.75
        self._y_max = 0.75
        # two link arm
        self._l1 = 0.3
        self._l2 = 0.3
        # frame skip rate
        self._skip_rate = 1
        self._total_frames = (int)(self._ts.size / self._skip_rate)

    def generate_animation(self, save: bool=True, skip_rate: int=1):
        """ Genarating animation.

        Args:
            save (bool): If True, animation is saved to log_dir.
            skip_rate (int): Skip rate.
        """
        # frame
        self._skip_rate = skip_rate
        self._total_frames = (int)(self._ts.size / self._skip_rate)
        self._fig = plt.figure(figsize=(12, 9))
        self._ax = plt.axes(xlim=(self._x_min, self._x_max),
                            ylim=(self._y_min, self._y_max))
        # aspect ratio
        self._ax.set_aspect('equal')
        # x0 and u0
        state = self._xs[0, :]
        control = self._us[0, :]
        # link1
        theta1 = state[0]
        link1_sx = 0
        link1_sy = 0
        link1_tx = link1_sx + self._l1 * np.sin(theta1)
        link1_ty = link1_sy - self._l1 * np.cos(theta1)
        self._link1 = patches.Polygon(
            xy=[[link1_sx, link1_sy], [link1_tx, link1_ty]],
            ec='tab:blue',
            linewidth=6.0
        )
        # link2
        theta2 = state[1]
        link2_sx = link1_tx
        link2_sy = link1_ty
        link2_tx = link2_sx + self._l2 * np.sin(theta1 + theta2)
        link2_ty = link2_sy - self._l2 * np.cos(theta1 + theta2)
        self._link2 = patches.Polygon(
            xy=[[link2_sx, link2_sy], [link2_tx, link2_ty]],
            ec='tab:green',
            linewidth=6.0
        )
        # add patch to ax
        self._ax.add_patch(self._link1)
        self._ax.add_patch(self._link2)
        # display
        self._time_text = self._ax.text(
            0.9, 0.05,
            f'{self._ts[0]} [s]',
            transform=self._ax.transAxes,
            fontsize=16
            )
        self._variable_text = self._ax.text(
            x=0.85, y=0.72,
            s=r'$\theta_1: $'+f'{state[0]:.3f}\n'
                +r'$\theta_2$: '+f'{state[1]:.3f}\n'
                +r'$\dot{\theta_1}$: '+f'{state[2]:.3f}\n'
                +r'$\dot{\theta_2}$: '+f'{state[3]:.3f}\n'
                +r'$u$: '+f'{control[0]:.3f}\n',
            transform=self._ax.transAxes,
            fontsize=16
            )
        anim = FuncAnimation(
            self._fig,
            self._update_animation,
            frames=self._total_frames,
            interval=1000*self._dtau*self._skip_rate,
            blit=True
            )
        # save movie
        if save:
            anim.save(join(self._log_dir, self._file_name) + '.mp4', dpi=120),
            writer='pillow',
            fps=int(1/(self._dtau * self._skip_rate)
            )
            print('Animation was saved at ' + self._log_dir + ' .')
            plt.show()

    def _update_animation(self, i):
        """ Callback function handed to FuncAnimation.
        """
        # current frame
        frame = i * self._skip_rate
        state = self._xs[frame, :]
        control = self._us[frame, :]
        # link1
        theta1 = state[0]
        link1_sx = 0
        link1_sy = 0
        link1_tx = link1_sx + self._l1 * np.sin(theta1)
        link1_ty = link1_sy - self._l1 * np.cos(theta1)
        self._link1.set_xy([[link1_sx, link1_sy], [link1_tx, link1_ty]])
        # link2
        theta2 = state[1]
        link2_sx = link1_tx
        link2_sy = link1_ty
        link2_tx = link2_sx + self._l2 * np.sin(theta1 + theta2)
        link2_ty = link2_sy - self._l2 * np.cos(theta1 + theta2)
        self._link2.set_xy([[link2_sx, link2_sy], [link2_tx, link2_ty]])
        # time text
        self._time_text.set_text(
            '{0:.1f} [s]'.format(frame * self._dtau)
            )
        # variable text
        self._variable_text.set_text(
            r'$\theta_1: $'+f'{state[0]:.3f}\n'
            +r'$\theta_2$: '+f'{state[1]:.3f}\n'
            +r'$\dot{\theta}_1$: '+f'{state[2]:.3f}\n'
            +r'$\dot{\theta}_2$: '+f'{state[3]:.3f}\n'
            +r'$u$: '+f'{control[0]:.3f}\n',
            )
        return self._link1, self._link2


# test
if __name__ == '__main__':
    # log_dir = '/home/ohtsukalab/src/DDPython/log/cartpole'
    # file_name = 'cartpole'
    # anim = CartPoleAnimator(log_dir, file_name)
    # anim.generate_animation(save=False)
    animator = HexacopterAnimator('log/hexacopter', 'hexacopter')
    animator.generate_animation(True)
