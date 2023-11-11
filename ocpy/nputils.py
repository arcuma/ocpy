import numpy as np
import numba


@numba.njit
def l1_directional_derivatives(x: np.ndarray, dx: np.ndarray):
    """ Compute directional derivatives of l1-norm.
        See P628 at (Nocedal, Wright 2004)

    Args:
        x (np.ndarray): Point.
        dx (np.ndarray): Direction.
    """
    n_x = x.shape[0]

    ### directional derivative of ||x||_1 along with dx.
    d_n1 = 0.0

    for i in range(n_x):

        if x[i] > 0:
           d_n1 += dx[i]
        elif x[i] < 0:
            d_n1 += -dx[i]
        else:
            d_n1 += abs(dx[i])
    
    return d_n1
