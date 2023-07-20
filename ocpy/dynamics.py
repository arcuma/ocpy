import sympy as sym
import numpy as np

from ocpy import symutils


class SymDynamics:
    """ Create Dynamics class. Derivatives are calculated via sym.
    """
    
    def __init__(self, x: sym.Matrix, u: sym.Matrix, t: sym.Symbol,
                 f: sym.Matrix):
        """ Create Dynamics class. Derivatives are calculated via sym.

        Args:
            x (sym.Matrix): State vector.
            u (sym.Matrix): Control input vector.
            t (sym.Symbol): Time.
            f (sym.Matrix): State equation, f(x, u, t)
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
    def InitByManual(x: sym.Matrix, u: sym.Matrix, t: sym.Symbol,
                     f: sym.Matrix, fx: sym.Matrix, fu: sym.Matrix,
                     fxx: sym.Matrix, fux: sym.Matrix, fuu: sym.Matrix):
        """ Initialize dynamics derivatives manually.

        Args:
            x (sym.Matrix): State vector.
            u (sym.Matrix): Control input vector.
            t (sym.Symbol): Time.
            f (sym.Matrix): Function of state equation.
            fx (sym.Matrix): Derivative of f w.r.t. state x.
            fu (sym.Matrix): Derivative of f w.r.t. input u.
            fxx (sym.Array): Derivative of f w.r.t. x and x.
            fux (sym.Array): Derivative of f w.r.t. u and x.
            fux (sym.Array): Derivative of f w.r.t. u and u.
            continuous (bool): Is continous model.
        """
        assert f.shape == x.shape
        symdyn = SymDynamics(t, x, u, sym.zeros(*x.shape))
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

    def substitute_constatnts(self, df_sym: list,
                              scalar_dict: dict=None, vector_dict: dict=None, 
                              matrix_dict: dict=None, dt: sym.Symbol=None,
                              dt_value: float=None,):
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
            self.df, scalar_dict, vector_dict, matrix_dict, dt, dt_value
        )
        return self.df_subs

class NumDynamics:
    """ Generate numerical function of dynamics from symbolic expression.
    """
    def __init__(self, x: sym.Matrix, u: sym.Matrix, t: sym.Symbol, dt: sym.Symbol,
                 f_sym: sym.Matrix, fx_sym: sym.Matrix, fu_sym: sym.Matrix,
                 fxx_sym: sym.Matrix, fux_sym: sym.Matrix,
                 fuu_sym: sym.Matrix):
        """ Turn symbolic dynamics into fast universal function.

        Args:
            x (sym.Matrix): state vector.
            u (sym.Matrix): control input vector.
            t (sym.Symbol): Time.
            f_sym (sym.Matrix): Function of state equation.
            fx_sym (sym.Matrix): Derivative of f w.r.t. state x.
            fu_sym (sym.Matrix): Derivative of f w.r.t. input u.
            fxx_sym (sym.Array): Derivative of f w.r.t. x and x.
            fux_sym (sym.Array): Derivative of f w.r.t. u and x.
            fux_sym (sym.Array): Derivative of f w.r.t. u and u.

        Note:
            Confirm all symbolic constants (e.g. mass, lentgth,) are substituted.
        """
        args = [x, u, t, dt]
        f_ufunc = symutils.lambdify(args, f_sym)
        fx_ufunc = symutils.lambdify(args, fx_sym)
        fu_ufunc = symutils.lambdify(args, fu_sym)
        fxx_ufunc = symutils.lambdify(args, fxx_sym)
        fux_ufunc = symutils.lambdify(args, fux_sym)
        fuu_ufunc = symutils.lambdify(args, fuu_sym)
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
