import path_ocpy

from ocpy import OCP
from ocpy import symutils

import numpy as np
import sympy as sym
from sympy import sin, cos, tan, exp, log, ln, sinh, cosh, tanh, diff, sqrt


def ocp_pendubot_unconstrained(constrain: str=None, mu=1e-2):

    sim_name = 'pendubot'

    ### Dimensions of state and input
    n_x = 4
    n_u = 1

    ### Define ocp class
    ocp = OCP(sim_name, n_x, n_u)

    ### Get symbols
    t = ocp.get_t()
    x = ocp.get_x()
    u = ocp.get_u()

    ### Symbolic expressions of constants.
    m1, m2, l1, l2, d1, d2, J1, J2, g_c = ocp.define_scalar_constants([
        ('m1', 0.2), ('m2', 0.7), ('l1', 0.3), ('l2', 0.3), ('d1', 0.15), 
        ('d2', 0.257), ('J1', 0.006), ('J2', 0.051), ('g_c', 9.80665)
    ])

    u_min, u_max = ocp.define_scalar_constants([
        ('u_min', -5), ('u_max', 5)
    ])

    ### Cost weight
    q = ocp.define_vector_constant('q', [1, 1, 0.1, 0.1])
    r = ocp.define_vector_constant('r', [0.1])
    q_f = ocp.define_vector_constant('q_{f}', [1, 1, 0.1, 0.1])
    Q = symutils.diag(q)
    Q_f = symutils.diag(q_f)
    R = symutils.diag(r)

    ### Reference state. 
    x_ref = ocp.define_vector_constant('x_{ref}', [np.pi, 0, 0, 0])

    ### State space representation.
    f = ocp.get_zero_vector(n_x)
    f[0] = x[2]
    f[1] = x[3]
    f[2] = -(sin(x[0] + x[1]) * d2 * g_c * m2 + g_c * (d1 * m1 + l1 * m2) * sin(x[0]) - 0.2e1 * d2 * (x[2] + x[3] / 0.2e1) * l1 * x[3] * m2 * sin(x[1]) - u[0]) / (0.2e1 * d2 * m2 * l1 * cos(x[1]) + d1 * d1 * m1 + d2 * d2 * m2 + l1 * l1 * m2 + J1 + J2)
    f[3] = (g_c * d2 * l1 * m2 * (d1 * m1 + l1 * m2) * sin(x[0] - x[1]) / 0.2e1 - d2 * d2 * g_c * l1 * m2 * m2 * sin(x[0] + 0.2e1 * x[1]) / 0.2e1 - (d1 * d1 * m1 - d1 * l1 * m1 / 0.2e1 + l1 * l1 * m2 / 0.2e1 + J1) * m2 * g_c * d2 * sin(x[0] + x[1]) - l1 * l1 * m2 * m2 * d2 * d2 * (pow(x[2], 0.2e1) + x[2] * x[3] + pow(x[3], 0.2e1) / 0.2e1) * sin(0.2e1 * x[1]) - l1 * m2 * d2 * ((pow(x[2] + x[3], 0.2e1) * d2 * d2 + pow(x[2], 0.2e1) * l1 * l1) * m2 + (d1 * d1 * m1 + J1 + J2) * pow(x[2], 0.2e1) + 0.2e1 * J2 * x[2] * x[3] + J2 * pow(x[3], 0.2e1)) * sin(x[1]) + g_c * (d2 * d2 * l1 * m2 * m2 / 0.2e1 + (d1 * d2 * d2 * m1 + J2 * l1) * m2 + J2 * d1 * m1) * sin(x[0]) - u[0] * (d2 * m2 * l1 * cos(x[1]) + d2 * d2 * m2 + J2)) / (0.2e1 * d2 * m2 * l1 * cos(x[1]) + d1 * d1 * m1 + d2 * d2 * m2 + l1 * l1 * m2 + J1 + J2) / (d2 * d2 * m2 + J2)

    ### Stage cost and terminal cost.
    l = 0.5 * (x - x_ref).T * Q * (x - x_ref) + 0.5 * u.T * R * u
    lf = 0.5 * (x - x_ref).T * Q_f * (x - x_ref)

    ### penalty / barrier function
    if constrain == 'penalty':
        l += ocp.get_penalty(u, [u_min], [u_max], mu=mu)
    elif constrain == 'barrier':
        l += ocp.get_barrier(u, [u_min], [u_max], mu=mu)

    ### Horizon length and discretization grids.
    T = 5.0
    N = 200

    ### Initial condition
    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    ### Define ocp
    ocp.define_unconstrained(f, l, lf, t0, x0, T, N)

    return ocp


def ocp_pendubot():

    sim_name = 'pendubot'

    ### Dimensions of state and input
    n_x = 4
    n_u = 1
    n_g = 2 * n_u

    ### Define ocp class
    ocp = OCP(sim_name, n_x, n_u, n_g)

    ### Get symbols
    t = ocp.get_t()
    x = ocp.get_x()
    u = ocp.get_u()

    ### Symbolic expressions of constants.
    m1, m2, l1, l2, d1, d2, J1, J2, g_c = ocp.define_scalar_constants([
        ('m1', 0.2), ('m2', 0.7), ('l1', 0.3), ('l2', 0.3), ('d1', 0.15), 
        ('d2', 0.257), ('J1', 0.006), ('J2', 0.051), ('g_c', 9.80665)
    ])

    u_min, u_max = ocp.define_scalar_constants([
        ('u_min', -5), ('u_max', 5)
    ])

    ### Cost weight
    q = ocp.define_vector_constant('q', [1, 1, 0.1, 0.1])
    r = ocp.define_vector_constant('r', [0.1])
    q_f = ocp.define_vector_constant('q_{f}', [10, 10, 0.1, 0.1])
    Q = symutils.diag(q)
    Q_f = symutils.diag(q_f)
    R = symutils.diag(r)

    ### Reference state. 
    x_ref = ocp.define_vector_constant('x_{ref}', [np.pi, 0, 0, 0])

    ### State space representation.
    f = ocp.get_zero_vector(n_x)
    f[0] = x[2]
    f[1] = x[3]
    f[2] = -(sin(x[0] + x[1]) * d2 * g_c * m2 + g_c * (d1 * m1 + l1 * m2) * sin(x[0]) - 0.2e1 * d2 * (x[2] + x[3] / 0.2e1) * l1 * x[3] * m2 * sin(x[1]) - u[0]) / (0.2e1 * d2 * m2 * l1 * cos(x[1]) + d1 * d1 * m1 + d2 * d2 * m2 + l1 * l1 * m2 + J1 + J2)
    f[3] = (g_c * d2 * l1 * m2 * (d1 * m1 + l1 * m2) * sin(x[0] - x[1]) / 0.2e1 - d2 * d2 * g_c * l1 * m2 * m2 * sin(x[0] + 0.2e1 * x[1]) / 0.2e1 - (d1 * d1 * m1 - d1 * l1 * m1 / 0.2e1 + l1 * l1 * m2 / 0.2e1 + J1) * m2 * g_c * d2 * sin(x[0] + x[1]) - l1 * l1 * m2 * m2 * d2 * d2 * (pow(x[2], 0.2e1) + x[2] * x[3] + pow(x[3], 0.2e1) / 0.2e1) * sin(0.2e1 * x[1]) - l1 * m2 * d2 * ((pow(x[2] + x[3], 0.2e1) * d2 * d2 + pow(x[2], 0.2e1) * l1 * l1) * m2 + (d1 * d1 * m1 + J1 + J2) * pow(x[2], 0.2e1) + 0.2e1 * J2 * x[2] * x[3] + J2 * pow(x[3], 0.2e1)) * sin(x[1]) + g_c * (d2 * d2 * l1 * m2 * m2 / 0.2e1 + (d1 * d2 * d2 * m1 + J2 * l1) * m2 + J2 * d1 * m1) * sin(x[0]) - u[0] * (d2 * m2 * l1 * cos(x[1]) + d2 * d2 * m2 + J2)) / (0.2e1 * d2 * m2 * l1 * cos(x[1]) + d1 * d1 * m1 + d2 * d2 * m2 + l1 * l1 * m2 + J1 + J2) / (d2 * d2 * m2 + J2)

    ### Stage cost and terminal cost.
    l = 0.5 * (x - x_ref).T * Q * (x - x_ref) + 0.5 * u.T * R * u
    lf = 0.5 * (x - x_ref).T * Q_f * (x - x_ref)

    ### constraints. g(x, u, t) <= 0.
    g = ocp.get_zero_vector(n_g)
    g[0] = u_min - u[0]
    g[1] = u[0] - u_max

    ### Horizon length and discretization grids.
    T = 5.0
    N = 200

    ### Initial condition
    t0 = 0.0
    x0 = np.array([0.0, 0.0, 0.0, 0.0])

    ### Define ocp
    ocp.define(f, l, lf, g, t0=t0, x0=x0, T=T, N=N)

    return ocp
