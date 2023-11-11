import path_ocpy

from ocpy import OCP
from ocpy import symutils

import numpy as np
import sympy as sym
from sympy import sin, cos, tan, exp, log, ln, sinh, cosh, tanh, diff, sqrt


def ocp_cartpole_unconstrained(constrain: str=None, mu=1e-2):

    sim_name = 'cartpole'

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
    m_c, m_p, l, g_c, u_min, u_max = ocp.define_scalar_constants([
        ('m_c', 2), ('m_p', 0.2), ('l', 0.5), ('g', 9.80665), 
        ('u_min', -15),  ('u_max', 15)
    ])

    ### Cost weight
    q = ocp.define_vector_constant('q', [2.5, 10, 0.01, 0.01])
    r = ocp.define_vector_constant('r', [1])
    q_f = ocp.define_vector_constant('q_{f}', [2.5, 10, 0.01, 0.01])
    Q = symutils.diag(q)
    Q_f = symutils.diag(q_f)
    R = symutils.diag(r)

    ### Reference state. 
    x_ref = ocp.define_vector_constant('x_{ref}', [0, np.pi, 0, 0])

    ### State space representation.
    f = ocp.get_zero_vector(n_x)
    f[0] = x[2]
    f[1] = x[3]
    f[2] = (u[0] + m_p*sin(x[1])*(l*x[3]*x[3] + g_c*cos(x[1])) )/( m_c+m_p*sin(x[1])*sin(x[1]) )
    f[3] = (-u[0] * cos(x[1]) - m_p*l*x[3]*x[3]*cos(x[1])*sin(x[1]) 
            - (m_c+m_p)*g_c*sin(x[1]) )/( l*(m_c + m_p*sin(x[1])*sin(x[1])))

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


def ocp_cartpole():

    sim_name = 'cartpole'

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
    m_c, m_p, l, g_c, u_min, u_max = ocp.define_scalar_constants([
        ('m_c', 2), ('m_p', 0.2), ('l', 0.5), ('g', 9.80665), 
        ('u_min', -15),  ('u_max', 15)
    ])

    ### Cost weight
    q = ocp.define_vector_constant('q', [2.5, 10, 0.01, 0.01])
    r = ocp.define_vector_constant('r', [1])
    q_f = ocp.define_vector_constant('q_{f}', [2.5, 10, 0.01, 0.01])
    Q = symutils.diag(q)
    Q_f = symutils.diag(q_f)
    R = symutils.diag(r)

    ### Reference state. 
    x_ref = ocp.define_vector_constant('x_{ref}', [0, np.pi, 0, 0])

    ### State space representation.
    f = ocp.get_zero_vector(n_x)
    f[0] = x[2]
    f[1] = x[3]
    f[2] = (u[0] + m_p*sin(x[1])*(l*x[3]*x[3] + g_c*cos(x[1])) )/( m_c+m_p*sin(x[1])*sin(x[1]) )
    f[3] = (-u[0] * cos(x[1]) - m_p*l*x[3]*x[3]*cos(x[1])*sin(x[1]) 
            - (m_c+m_p)*g_c*sin(x[1]) )/( l*(m_c + m_p*sin(x[1])*sin(x[1])))

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
