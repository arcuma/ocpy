"""
utility module that helps symbolic matrix calculation.
"""

import sympy
import numpy as np
from sympy import ImmutableDenseNDimArray as Tensor
import copy


def define_scalor(name: str) -> sympy.Symbol:
    return sympy.Symbol(name)


def define_vector(name: str, n: int) -> sympy.Matrix:
    v = sympy.Matrix([[sympy.symbols(name + f'[{i}]')] for i in range(n)])
    return v


def define_matrix(name: str, m: int, n: int) -> sympy.Matrix:
    M = sympy.Matrix([[sympy.symbols(name + f'[{i}][{j}]') for j in range(n)]
                       for i in range(m)])
    return M


def diff_scalar(f, x):
    """ Calculate derivative of  scalar function f w.r.t. x. \\
        Args:
            f (1 * 1 sympy.Matrix or sympy.Symbol) : Scalar function 
            x (n_x * 1 sympy.Matrix): Vector variable
        
        Returns:
            fx (n_x * 1 sympy.Matrix): Derivative of f w.r.t. x.
    """
    if not isinstance(f, sympy.Matrix):
        f = sympy.Matrix([f])
    fx = f.jacobian(x).T
    return fx


def diff_vector(f, x):
    """ Calculate derivative of  scalar function f w.r.t. x. \\
        Args:
            f (n_f * 1 sympy.Matrix) : Vector function 
            x (n_x * 1 sympy.Matrix): Vector variable
        
        Returns:
            fx (n_x * n_x sympy.Matrix): Derivative of f w.r.t. x.
    """
    return f.jacobian(x)


def diff_matrix(M, x):
    """ Calculate derivative of matrix function M(x) , w.r.t. x. \\
        Args:
            M (m * n sympy.Matrix): Matrix function
            x (n_x sympy.Matrix): Vector variable
        
        Returns:
            M_x (n_x * m * n): Derivative of M w.r.t. x, a 3 order tensor Mx.
            
        Note : 
            Mx is sympy.MutableDenseNDimArray and \
            no longer a sympy.matrices.dense.MutableDenseMatrix
    """
    n_x = x.shape[0]
    Mx = Tensor(Tensor(sympy.diff(M, x[i])) for i in range(n_x))
    return Mx


def vector_dot_tensor_1(v, T):
    """ Tensor product between vector v and 3th tensor T, contraction with 1st axis.
        Args:
            v (m * 1 sympy Matrix): Vector. Correspond to jacobian of V
            T (l * m * n sympy Array): 3rd tensor T. Correspond to Hessian of f
        Returns:
            M (l * n sympy Matrix) : Output matrix.
    """
    assert len(v) == T.shape[1]
    l, m, n = T.shape
    M = sympy.Matrix.zeros(l, n)
    for i in range(l):
        for j in range(m):
            for k in range(n):
                M[i, k] += v[j] * T[i, j, k]
    return M.T


def simplify(f: sympy.Symbol | list[sympy.Symbol]):
    """ Simplify symbolic expression.
    Args:
        f (sympy of list[sympy]): Symbolic function(s).
    """
    if isinstance(f, list):
        for func in f:
            func.simplify()
    else:
        f.simplify()


def substitute_constants(
        f: sympy.Symbol | sympy.Matrix | sympy.Array,
        scalar_dict: dict=None, vector_dict: dict=None, matrix_dict: dict=None,
        dt: sympy.Symbol=None, dt_value: float=None):
    """ Substitute constants such as mass, length, costfunction coefs etc..

    Args:
        f (sympy) : Symbolic function.
        scalar_dict (dict, {"name": (symbol, value)}) : Scalar constants
        vector_dict (dict, {"name": (symbol, value)}) : Vector constants
        matrix_dict (dict, {"name": (symbol, value)}) : Matrix constants
        dt (sympy.Symobl): Time discretization step.
        dt_value (float): Value of dt.
        
    Returns:
        f_subs : Function f in which constatnts are substituted.
    """
    f_subs = copy.copy(f)
    if scalar_dict is not None:
        assert type(scalar_dict) == dict
        for sym, val in scalar_dict.values():
            f_subs = f_subs.subs(sym, val)
    if vector_dict is not None:
        assert type(vector_dict) == dict
        for sym_vec, val_vec in vector_dict.values():
            for sym, val in zip(sym_vec, val_vec):
                f_subs = f_subs.subs(sym, val)
    if matrix_dict is not None:
        assert type(matrix_dict) == dict
        for sym_mat, val_mat in matrix_dict.values():
            m, n = sym_mat.shape
            for i in range(m):
                for j in range(n):
                    f_subs = f_subs.subs(sym_mat[i, j], val_mat[i][j])
    if dt is not None:
        f_subs = f_subs.subs(dt, dt_value)
    return f_subs


def substitute_constants_list(
        func_list: list, 
        scalar_dict: dict=None, vector_dict: dict=None, matrix_dict: dict=None,
        dt: sympy.Symbol=None, dt_value: float=None):
    """ Substitute constants such as mass, length, costfunction coefs etc.
    Args:
        func_list (list) : list of symbolic function
        scalar_dict (dict, {"name": (symbol, value)}) : Scalar constants
        vector_dict (dict, {"name": (symbol, value)}) : Vector constants
        matrix_dict (dict, {"name": (symbol, value)}) : Matrix constants
        dt (sympy.Symobl): discretization step.
        dt_value (float): value of dt.
    Returns:
        f_subs : Function f in which constatnts are substituted.
    """
    functions_subs = []
    for f in func_list:
        functions_subs.append(substitute_constants(f, scalar_dict, vector_dict,
                                                    matrix_dict, dt, dt_value))
    return functions_subs


def lambdify(args: list, f: sympy.Symbol | sympy.Matrix | sympy.Array,
             dim_reduction=True):
    """ call sympy.lambdify to transform funciton into fast numpy ufunc.
        Args:
            args (list) : Arguments of function f. 
                If f = f(x, y, z),  args=[x, y, z].
            f (sympy symbol or matrix) : Function.
            dim_reduction (bool=True): If true, m*1 or 1*n Matrix are transform \
                into 1d ndarray.
        Returns:
            f_ufunc : numpy ufunc.
    """
    if isinstance(f, sympy.Matrix):
        m, n = f.shape
        # convert into 1d array
        if (m == 1 or n == 1) and dim_reduction:
            if n == 1:
                f = f.T
            f = sympy.Array(f)[0]
            f = sympy.lambdify(args, f, "numpy")
            # unless this operation, f_ufunc returns list, not ndarray. 
            f_ufunc = lambda *args: np.array(f(*args))
            return f_ufunc
    elif isinstance(f, sympy.Array):
        f = sympy.lambdify(args, f, "numpy")
        f_ufunc = lambda *args: np.array(f(*args))
        return f_ufunc
    f_ufunc = sympy.lambdify(args, f, "numpy")
    return f_ufunc


def lambdify_list(args: list, f_list: list):
    """ Call sympy.lambdify to transform funciton into fast numpy ufunc.
        Args:
            args (list) : Arguments of function f. \
                If f = f(x, y, z),  args are [x, y, z].
            f_list (list (sympy symbol or matrix)) : list of functions.
        Returns:
            f_ufunc : numpy ufunc.
    """
    ufunc_list = []
    for f in f_list:
        ufunc_list.append(lambdify(args, f))
    return ufunc_list
