"""
Utility module that helps SymPy operation.
"""

import sympy as sym
import numpy as np
from numba import njit
import copy


def define_scalor(name: str) -> sym.Symbol:
    return sym.Symbol(name)


def define_vector(name: str, n: int) -> sym.Matrix:
    v = sym.Matrix([[sym.symbols(name + f'[{i}]')] for i in range(n)])
    return v


def define_matrix(name: str, m: int, n: int) -> sym.Matrix:
    M = sym.Matrix([[sym.symbols(name + f'[{i}][{j}]') for j in range(n)]
                       for i in range(m)])
    return M


def diff_scalar(f: sym.Matrix | sym.Symbol , x: sym.Matrix) -> sym.Matrix:
    """ Calculate derivative of  scalar function f w.r.t. x.

    Args:
        f (1 * 1 sym.Matrix or sympy.Symbol) : Scalar function.
        x (n_x * 1 sym.Matrix): Vector variable.
        
    Returns:
            fx (n_x * 1 sym.Matrix): Derivative of f w.r.t. x.
    """
    if not isinstance(f, sym.Matrix):
        f = sym.Matrix([f])
    fx = f.jacobian(x).T
    return fx


def diff_vector(f: sym.Matrix, x: sym.Matrix) -> sym.Matrix:
    """ Calculate derivative of  scalar function f w.r.t. x.

    Args:
        f (n_f * 1 sym.Matrix) : Vector function.
        x (n_x * 1 sym.Matrix): Vector variable.
        
    Returns:
            fx (n_x * n_x sym.Matrix): Derivative of f w.r.t. x.
    """
    return f.jacobian(x)


def diff_matrix(M: sym.Matrix, x: sym.Matrix) -> sym.Array:
    """ Calculate derivative of matrix function M(x) , w.r.t. x.

    Args:
        M (m * n sym.Matrix): Matrix function
        x (n_x sym.Matrix): Vector variable
    
    Returns:
        M_x (n_x * m * n): Derivative of M w.r.t. x, a 3 order tensor Mx.
        
    Note : 
        Mx is sym.MutableDenseNDimArray so \
        no longer a sym.Matrix.
    """
    Mx = sym.diff(M, x.T)[0]
    ### For MutableDenseNDimArray don't have attribute subs()
    return sym.ImmutableDenseNDimArray(Mx)


def diff_matrix_2(M: sym.Matrix, x: sym.Matrix) -> sym.Array:
    """ Calculate derivative of matrix function M(x) , w.r.t. x. \
        The dimension of result is different to diff_matrix().

    Args:
        M (m * n sym.Matrix): Matrix function
        x (n_x sym.Matrix): Vector variable
    
    Returns:
        M_x (m * n * n_x): Derivative of M w.r.t. x, a 3rd order tensor Mx.
    """
    m, n = M.shape
    n_x = x.shape[0]
    Mx = sym.MutableDenseNDimArray(np.zeros((m, n, n_x)))
    for i in range(m):
        for j in range(n):
            for k in range(n_x):
                Mx[i, j, k] = sym.diff(M[i, j], x[k])
    return sym.ImmutableDenseNDimArray(Mx)

def diag(v: sym.Matrix):
    """ Transform vector into diagonal matrix.

    Args:
        v (sym.Matrix): Symbolic vector.

    Returns:
        D (sym.Matrix): Symbolic diagonal matrix.
    """
    return sym.diag(*v)


def vector_dot_tensor_sym(v, T):
    """ Tensor product between vector v and 3th tensor T, \
        contraction with 2nd axis.
    Args:

        v (m * 1 sym Matrix): Vector. Correspond to jacobian of V
        T (l * m * n sym Array): 3rd tensor T. Correspond to Hessian of f

    Returns:
        M (l * n sym Matrix) : Output matrix.
    """
    assert len(v) == T.shape[1]
    l, m, n = T.shape
    M = sym.Matrix.zeros(l, n)
    for i in range(l):
        for j in range(m):
            for k in range(n):
                M[i, k] += v[j] * T[i, j, k]
    return M.T


def simplify(f: sym.Symbol | list[sym.Symbol]):
    """ Simplify symbolic expression.

    Args:
        f (sym of list[sympy]): Symbolic function(s).
    """
    if isinstance(f, (list, tuple)):
        for func in f:
            func.simplify()
    else:
        f.simplify()


def substitute_constants(
        f: sym.Symbol | sym.Matrix | sym.Array,
        scalar_dict: dict=None, vector_dict: dict=None, matrix_dict: dict=None,
        dt: sym.Symbol=None, dt_value: float=None):
    """ Substitute constants such as mass, length, costfunction coefs etc..

    Args:
        f (sym) : Symbolic function.
        scalar_dict (dict, {"name": (symbol, value)}) : Scalar constants
        vector_dict (dict, {"name": (symbol, value)}) : Vector constants
        matrix_dict (dict, {"name": (symbol, value)}) : Matrix constants
        dt (sym.Symobl): Time discretization step.
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
        dt: sym.Symbol=None, dt_value: float=None):
    """ Substitute constants such as mass, length, costfunction coefs etc.

    Args:
        func_list (list) : list of symbolic function
        scalar_dict (dict, {"name": (symbol, value)}) : Scalar constants
        vector_dict (dict, {"name": (symbol, value)}) : Vector constants
        matrix_dict (dict, {"name": (symbol, value)}) : Matrix constants
        dt (sym.Symobl): discretization step.
        dt_value (float): value of dt.

    Returns:
        f_subs : Function f in which constatnts are substituted.
    """
    functions_subs = []
    for f in func_list:
        functions_subs.append(substitute_constants(f, scalar_dict, vector_dict,
                                                   matrix_dict, dt, dt_value))
    return functions_subs


def lambdify(args: list, f: sym.Symbol | sym.Matrix | sym.Array,
             dim_reduction=True, numba_njit=True):
    """ call sym.lambdify() to convert symbolic funciton into \
        fast numerical function.

    Args:
        args (list) : Arguments of function f. 
            That is, if f = f(x, y, z),  args is [x, y, z].
        f (sym symbol or matrix) : Function.
        dim_reduction (bool=True): If true, m*1 or 1*n Matrix are transformed \
            into 1d ndarray.
        numba_njit (bool=True): If true, function is wrapped by numba.njit.
            
    Returns:
        f_num (lambdifygenerated) : Fast lambda function.
    """
    f = copy.copy(f)
    if numba_njit:
        if isinstance(f, sym.Matrix):
            m, n = f.shape
            ### for needed that numba know datatype is float.
            f += 1e-128*np.ones((m, n))
            ### convert into 1d array
            if (m == 1 or n == 1) and dim_reduction:
                if n == 1:
                    f = f.T
                f = sym.Array(f)[0]
                f_num = sym.lambdify(args, f, modules='numpy')
                return njit(f_num)
                ### expired
                # f_numpy = lambda *args: np.array(f_list_njit(*args))
                # return njit(f_numpy)
        elif isinstance(f, sym.Array):
            l, m, n = f.shape
            f = sym.MutableDenseNDimArray(f)
            ### for needed that numba know datatype is float.
            f += 1e-128 * np.ones((l, m, n))
            f_list_njit = njit(sym.lambdify(args, f, modules='numpy'))
            ### for 3D array, this operation is needed to turn into ndarray.
            f_num = lambda *args: np.array(f_list_njit(*args))
            return njit(f_num)
        f_num = sym.lambdify(args, f, modules='numpy')
        return njit(f_num)
    else:
        if isinstance(f, sym.Matrix):
            m, n = f.shape
            ### convert into 1d array
            if (m == 1 or n == 1) and dim_reduction:
                if n == 1:
                    f = f.T
                f = sym.Array(f)[0]
                f = sym.lambdify(args, f, "numpy")
                ### unless this operation, f_num returns list, not ndarray. 
                f_num = lambda *args: np.array(f(*args))
                return f_num
        elif isinstance(f, sym.Array):
            f = sym.lambdify(args, f, "numpy")
            f_num = lambda *args: np.array(f(*args))
            return f_num
        f_num = sym.lambdify(args, f, "numpy")
        return f_num


def lambdify_list(args: list, f_list: list, dim_reduction: bool=True,
                  numba_njit :bool=True):
    """ Call sym.lambdify to transform funciton into fast numpy function.

    Args:
        args (list) : Arguments of function f. \
            If f = f(x, y, z),  args are [x, y, z].
        f_list (list (sym symbol or matrix)) : list of functions.
        dim_reduction (bool=True): If true, m*1 or 1*n Matrix are transformed \
            into 1d ndarray.
        numba_njit (bool=True): If true, function is wrapped by numba.njit.
    Returns:
        f_num_list (list) : Fast lambda function.

    """
    f_num_list = []
    for f in f_list:
        f_num_list.append(lambdify(args, f), dim_reduction, numba_njit)
    return f_num_list
