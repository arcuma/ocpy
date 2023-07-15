import sympy
import numpy as np

from ocpy import symutils


class SymDynamics:
    """ Create Dynamics class. Derivatives are calculated via sympy.
    """
    
    def __init__(self, x: sympy.Matrix, u: sympy.Matrix, t: sympy.Symbol,
                 f: sympy.Matrix):
        """ Create Dynamics class. Derivatives are calculated via sympy.

        Args:
            x (sympy.Matrix): State vector.
            u (sympy.Matrix): Control input vector.
            t (sympy.Symbol): Time.
            f (sympy.Matrix): State equation, f(x, u, t)
        """
        fx = symutils.diff_vector(f, x)
        fu = symutils.diff_vector(f, u)
        fxx = symutils.diff_matrix(fx, x)
        fux = symutils.diff_matrix(fu, x)
        fuu = symutils.diff_matrix(fu, u)
        self.f = f
        self.fx = fx
        self.fu = fu
        self.fxx = fxx
        self.fux = fux
        self.fuu = fuu
        self.df = [f, fx, fu, fxx, fux, fuu]

    @staticmethod
    def InitByManual(x: sympy.Matrix, u: sympy.Matrix, t: sympy.Symbol,
                     f: sympy.Matrix, fx: sympy.Matrix, fu: sympy.Matrix,
                     fxx: sympy.Matrix, fux: sympy.Matrix, fuu: sympy.Matrix):
        """ Initialize dynamics derivatives manually.

        Args:
            x (sympy.Matrix): State vector.
            u (sympy.Matrix): Control input vector.
            t (sympy.Symbol): Time.
            f (sympy.Matrix): Function of state equation.
            fx (sympy.Matrix): Derivative of f w.r.t. state x.
            fu (sympy.Matrix): Derivative of f w.r.t. input u.
            fxx (sympy.Array): Derivative of f w.r.t. x and x.
            fux (sympy.Array): Derivative of f w.r.t. u and x.
            fux (sympy.Array): Derivative of f w.r.t. u and u.
            continuous (bool): Is continous model.
        """
        assert f.shape == x.shape
        symdyn = SymDynamics(t, x, u, sympy.zeros(*x.shape))
        symdyn.f = f
        symdyn.fx = fx
        symdyn.fu = fu
        symdyn.fxx = fxx
        symdyn.fux = fux
        symdyn.fuu = fuu
        symdyn.df = [f, fx, fu, fxx, fux, fuu]
        return symdyn

    def get_derivatives(self):
        """ Return derivatives of dynamics.

            Returns:
                df (list):[f, fx, fu, fxx, fux, fuu]
        """
        return self.df

    def substitute_constatnts(self, scalar_dict: dict=None, vector_dict: dict=None, 
                              matrix_dict: dict=None):
        """ Substitute symbolic constatnts into specic values 
                for numerical calculation.

            Args:
                df_sym (list): derivatives of f, [f, fx, fu, fxx, fux, fuu].
                scalar_dict (dict) : {"name": (symbol, value)}) 
                vector_dict (dict) : {"name": (symbol, value)}) 
                matrix_dict (dict) : {"name": (symbol, value)}) 
            Returns:
                df_subs (list) : constants-substituted symbolic
                    dynamics derivatives.
        """
        self.df_subs = symutils.substitute_constants_list(
            self.df, scalar_dict, vector_dict, matrix_dict
        )
        return self.df_subs

class NumDynamics:
    """ Generate numerical function of dynamics from symbolic expression.
    """
    def __init__(self, x: sympy.Matrix, u: sympy.Matrix, t: sympy.Symbol,
                 f_sym: sympy.Matrix, fx_sym: sympy.Matrix, fu_sym: sympy.Matrix,
                 fxx_sym: sympy.Matrix, fux_sym: sympy.Matrix,
                 fuu_sym: sympy.Matrix):
        """ Turn symbolic dynamics into fast universal function.

        Args:
            x (sympy.Matrix): state vector.
            u (sympy.Matrix): control input vector.
            t (sympy.Symbol): Time.
            f_sym (sympy.Matrix): Function of state equation.
            fx_sym (sympy.Matrix): Derivative of f w.r.t. state x.
            fu_sym (sympy.Matrix): Derivative of f w.r.t. input u.
            fxx_sym (sympy.Array): Derivative of f w.r.t. x and x.
            fux_sym (sympy.Array): Derivative of f w.r.t. u and x.
            fux_sym (sympy.Array): Derivative of f w.r.t. u and u.

        Note:
            Confirm all symbolic constants (e.g. mass, lentgth,) are substituted.
        """
        f_ufunc = symutils.lambdify([x, u, t], f_sym)
        fx_ufunc = symutils.lambdify([x, u, t], fx_sym)
        fu_ufunc = symutils.lambdify([x, u, t], fu_sym)
        fxx_ufunc = symutils.lambdify([x, u, t], fxx_sym)
        fux_ufunc = symutils.lambdify([x, u, t], fux_sym)
        fuu_ufunc = symutils.lambdify([x, u, t], fuu_sym)
        self.f_ufunc = f_ufunc
        self.fx_ufunc = fx_ufunc
        self.fu_ufunc = fu_ufunc
        self.fxx_ufunc = fxx_ufunc
        self.fux_ufunc = fux_ufunc
        self.fuu_ufunc = fuu_ufunc
        self.df = [f_ufunc, fx_ufunc, fu_ufunc, fxx_ufunc, fux_ufunc, fuu_ufunc]
    
    def get_derivatives(self):
        """ Return dynamics ufunction.

        Returns:
            df (list) : ufunc list of [f, fx, fu, fxx, fux, fuu], 
            whose arguments are x, u, t.
        """
        return self.df
