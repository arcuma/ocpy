import path_ocpy

from ocpy import OCP
from ocpy import symutils

import numpy as np
import sympy as sym
from sympy import sin, cos, tan, exp, log, ln, sinh, cosh, tanh, diff, sqrt


def ocp_hexacopter_unconstrained(constrain: str=None, mu=1e-2):

    sim_name = 'hexacopter'

    n_x = 12
    n_u = 6

    # Define ocp class
    ocp = OCP(sim_name, n_x, n_u)

    # Get symbols
    t = ocp.get_t()
    x = ocp.get_x()
    u = ocp.get_u()

    # Symbolic expressions of constants.
    m, l, k, Ixx, Iyy, Izz, gamma, g_c = ocp.define_scalar_constants(
        [('m', 1.44), ('l', 0.23), ('k', 1.6e-09),
        ('I_xx', 0.0348), ('I_yy', 0.0459), ('I_zz', 0.0977),
        ('gamma', 0.01), ('g_c', 9.80665)]
    )
    z_ref, u_min, u_max, epsilon = ocp.define_scalar_constants(
        [('z_ref', 5), ('u_min', 0.144), ('u_max', 6), ('epsilon', 0.01)]
    )

    # Symbolic expressions of constants.
    # q = ocp.define_vector_constant(
    #     'q',  [1, 1, 1, 0.01, 0.01, 0, 0.01, 0.01, 0.01, 0.1, 0.1, 0.001])
    # q_f = ocp.define_vector_constant(
    #     'q_{f}',  [1, 1, 1, 0.01, 0.01, 1, 0.01, 0.01, 0.01, 0.1, 0.1, 0.001])
    q = ocp.define_vector_constant(
        'q',  [20, 20, 20, 0.01, 0.01, 1, 0.01, 0.01, 0.01, 0.1, 0.1, 0.001])
    r = ocp.define_vector_constant(
        'r',  [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    q_f = ocp.define_vector_constant(
        'q_{f}',  [20, 20, 20, 0.01, 0.01, 1, 0.01, 0.01, 0.01, 0.1, 0.1, 0.001])
    Q = symutils.diag(q)
    R = symutils.diag(r)
    Q_f = symutils.diag(q_f)

    # Reference position and velocity.
    p_ref = ocp.get_zero_vector(3)
    # p_ref[0] = sin(2*t)
    # p_ref[1] = 1 - cos(2*t)
    # p_ref[2] = z_ref + 2*sin(2*t)
    p_ref[0] = 2*sin(2*t)
    p_ref[1] = 1*sin(4*t)
    p_ref[2] = z_ref
    p_ref_diff = p_ref.diff(t)

    # Reference state
    x_ref = ocp.get_zero_vector(n_x)
    x_ref[0:3, :] = p_ref
    x_ref[6:9, :] = p_ref_diff

    # Input
    U1 = sum(u[i] for i in range(n_u))
    U2 = l*(-u[0]/2 - u[1] - u[2]/2 + u[3]/2 + u[4]+ u[5]/2)
    U3 = l*(-(sqrt(3)/2)*u[0] + (sqrt(3)/2)*u[2] + (sqrt(3)/2)*u[3] - (sqrt(3)/2)*u[5])
    U4 = k*(-u[0] + u[1] - u[2] + u[3] - u[4] + u[5]) - gamma * x[11]

    # State space representation.
    f = ocp.get_zero_vector(n_x)
    f[0] = x[6]
    f[1] = x[7]
    f[2] = x[8]
    f[3] = x[9]
    f[4] = x[10]
    f[5] = x[11]
    f[6] = (cos(x[5])*sin(x[4])*cos(x[3]) + sin(x[5])*sin(x[3]))*U1/m
    f[7] = (sin(x[5])*sin(x[4])*cos(x[3]) - cos(x[5])*sin(x[3]))*U1/m
    f[8] = -g_c + (cos(x[3])*cos(x[4]))*U1/m
    f[9] = ((Iyy-Izz)/Ixx)*x[10]*x[11] + U2/Ixx
    f[10] = ((Izz-Ixx)/Iyy)*x[9]*x[11] + U3/Iyy
    f[11] = ((Ixx-Iyy)/Izz)*x[9]*x[10] + U4/Izz

    # Reference input.
    u_ref = ocp.get_zero_vector(n_u)
    for i in range(n_u):
        u_ref[i] = (m*g_c) / 6

    # Stage cost and terminal cost
    l = 0.5 * (x - x_ref).T * Q * (x - x_ref) + 0.5 * (u - u_ref).T * R * (u - u_ref)
    lf = 0.5 * (x - x_ref).T * Q_f * (x - x_ref)

    # penalty / barrier function
    u_lb = np.array([u_min for i in range(n_u)])
    u_ub = np.array([u_max for i in range(n_u)])
    if constrain == 'penalty':
        l += ocp.get_penalty(u, u_lb, u_ub, mu=mu)
    elif constrain == 'barrier':
        l += ocp.get_barrier(u, u_lb, u_ub, mu=mu)

    # Horizon length and discretization grids.
    T = 5.0
    N = 200

    # Initial condition
    t0 = 0.0
    x0 = np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Define ocp
    ocp.define_unconstrained(f, l, lf, t0, x0, T, N)

    return ocp


def ocp_hexacopter():

    sim_name = 'hexacopter'

    n_x = 12
    n_u = 6
    n_g = 2 * n_u

    # Define ocp class
    ocp = OCP(sim_name, n_x, n_u, n_g)

    # Get symbols
    t = ocp.get_t()
    x = ocp.get_x()
    u = ocp.get_u()

    # Symbolic expressions of constants.
    m, l, k, Ixx, Iyy, Izz, gamma, g_c = ocp.define_scalar_constants(
        [('m', 1.44), ('l', 0.23), ('k', 1.6e-09),
        ('I_xx', 0.0348), ('I_yy', 0.0459), ('I_zz', 0.0977),
        ('gamma', 0.01), ('g_c', 9.80665)]
    )
    z_ref, u_min, u_max, epsilon = ocp.define_scalar_constants(
        [('z_ref', 5), ('u_min', 0.144), ('u_max', 6), ('epsilon', 0.01)]
    )

    # Symbolic expressions of constants.
    # q = ocp.define_vector_constant(
    #     'q',  [1, 1, 1, 0.01, 0.01, 0, 0.01, 0.01, 0.01, 0.1, 0.1, 0.001])
    # q_f = ocp.define_vector_constant(
    #     'q_{f}',  [1, 1, 1, 0.01, 0.01, 1, 0.01, 0.01, 0.01, 0.1, 0.1, 0.001])
    q = ocp.define_vector_constant(
        'q',  [20, 20, 20, 0.01, 0.01, 1, 0.01, 0.01, 0.01, 0.1, 0.1, 0.001])
    r = ocp.define_vector_constant(
        'r',  [0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    q_f = ocp.define_vector_constant(
        'q_{f}',  [20, 20, 20, 0.01, 0.01, 1, 0.01, 0.01, 0.01, 0.1, 0.1, 0.001])
    Q = symutils.diag(q)
    R = symutils.diag(r)
    Q_f = symutils.diag(q_f)

    # Reference position and velocity.
    p_ref = ocp.get_zero_vector(3)
    # p_ref[0] = sin(2*t)
    # p_ref[1] = 1 - cos(2*t)
    # p_ref[2] = z_ref + 2*sin(2*t)
    p_ref[0] = 2*sin(2*t)
    p_ref[1] = 1*sin(4*t)
    p_ref[2] = z_ref
    p_ref_diff = p_ref.diff(t)

    # Reference state
    x_ref = ocp.get_zero_vector(n_x)
    x_ref[0:3, :] = p_ref
    x_ref[6:9, :] = p_ref_diff

    # Input
    U1 = sum(u[i] for i in range(n_u))
    U2 = l*(-u[0]/2 - u[1] - u[2]/2 + u[3]/2 + u[4]+ u[5]/2)
    U3 = l*(-(sqrt(3)/2)*u[0] + (sqrt(3)/2)*u[2] + (sqrt(3)/2)*u[3] - (sqrt(3)/2)*u[5])
    U4 = k*(-u[0] + u[1] - u[2] + u[3] - u[4] + u[5]) - gamma * x[11]

    # State space representation.
    f = ocp.get_zero_vector(n_x)
    f[0] = x[6]
    f[1] = x[7]
    f[2] = x[8]
    f[3] = x[9]
    f[4] = x[10]
    f[5] = x[11]
    f[6] = (cos(x[5])*sin(x[4])*cos(x[3]) + sin(x[5])*sin(x[3]))*U1/m
    f[7] = (sin(x[5])*sin(x[4])*cos(x[3]) - cos(x[5])*sin(x[3]))*U1/m
    f[8] = -g_c + (cos(x[3])*cos(x[4]))*U1/m
    f[9] = ((Iyy-Izz)/Ixx)*x[10]*x[11] + U2/Ixx
    f[10] = ((Izz-Ixx)/Iyy)*x[9]*x[11] + U3/Iyy
    f[11] = ((Ixx-Iyy)/Izz)*x[9]*x[10] + U4/Izz

    # Reference input.
    u_ref = ocp.get_zero_vector(n_u)
    for i in range(n_u):
        u_ref[i] = (m*g_c) / 6

    # Stage cost and terminal cost
    l = 0.5 * (x - x_ref).T * Q * (x - x_ref) + 0.5 * (u - u_ref).T * R * (u - u_ref)
    lf = 0.5 * (x - x_ref).T * Q_f * (x - x_ref)

    # constraints. g(x, u, t) <= 0.
    g = ocp.get_zero_vector(n_g)
    for i in range(n_u):
        g[2 * i] = u_min - u[i]
        g[2 * i + 1] = u[i] - u_max

    # Horizon length and discretization grids.
    T = 5.0
    N = 200

    # Initial condition
    t0 = 0.0
    x0 = np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Define ocp
    ocp.define(f, l, lf, g, t0=t0, x0=x0, T=T, N=N)

    return ocp