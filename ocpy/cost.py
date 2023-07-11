import sympy
import numpy as np

from ocpy import symutils


class SymCost:
    """Symbolic Cost class.
    """
    def __init__(self, x: sympy.Matrix, u: sympy.Matrix, t: sympy.Symbol,
                 l: sympy.Symbol, lf: sympy.Symbol):    
        """Symbolic Cost class.
    
            Args:
                x (sympy.Matrix): state vector.
                u (sympy.Matrix): control input vector.
                t (sympy.Symbol): Time.
                l (sympy.Symbol): Stage cost.
                lf (sympy.Symbol): Terminal cost.
        """
        lx = symutils.diff_scalar(l, x)
        lu = symutils.diff_scalar(l, u)
        lxx = symutils.diff_vector(lx, x)
        lux = symutils.diff_vector(lu, x)
        luu = symutils.diff_vector(lu, u)
        lfx = symutils.diff_scalar(lf, x)
        lfxx = symutils.diff_vector(lfx, x)
        self.l = l
        self.lx = lx
        self.lu = lu
        self.lxx = lxx
        self.lux = lux
        self.luu = luu
        self.lfx = lfx
        self.lfxx = lfxx
        self.dl = [l, lx, lu, lxx, lux, luu, lf, lfx, lfxx]

    
    @staticmethod
    def InitByManual(x: sympy.Matrix, u: sympy.Matrix, t: sympy.Symbol,
                     l: sympy.Symbol, lx: sympy.Matrix, lu: sympy.Matrix,
                     lxx: sympy.Matrix, lux: sympy.Matrix, luu: sympy.Matrix,
                     lf: sympy.Symbol, lfx: sympy.Matrix, lfxx: sympy.Matrix):
        """ Initialize dynamics derivatives manually.
        Args:
            x (sympy.Matrix): state vector.
            u (sympy.Matrix): control input vector.
            t (sympy.Symbol): Time.
            l (sympy.Matrix): stage cost
            lx (sympy.Symbol): derivative of l w.r.t. state x
            lu (sympy.Symbol): derivative of l w.r.t. input u
            lxx (sympy.Matrix): derivative of lx w.r.t. x.
            lux (sympy.Matrix): derivative of lu w.r.t. x.
            luu (sympy.Matrix): derivative of lu w.r.t. u.
            lf (sympy.Matrix): terminal cost
            lfx (sympy.Matrix): derivative of lf w.r.t. state x
            lfxx (sympy.Matrix): derivative of lfx w.r.t. state x
        """
        symcost = SymCost(x, u, t, sympy.zeros(1)[0], sympy.zeros(1)[0])
        symcost.l = l
        symcost.lx = lx
        symcost.lu = lu
        symcost.lxx = lxx
        symcost.lux = lux
        symcost.luu = luu
        symcost.lfx = lfx
        symcost.lfxx = lfxx
        symcost.dl = [l, lx, lu, lxx, lux, luu, lf, lfx, lfxx]


    def get_derivatives(self):
        """Get derivatives of dynamics.
            Returns:
                dl (list): [l, lx, lu, lxx, lux, luu, lf, lfx, lfxx]
        """
        return self.dl
    

    def substitute_constatnts(self, dl_sym, dt=None, dt_value=None, 
                              scalar_dict=None, vector_dict=None, matrix_dict=None):
        """ Substitute symbolic constatnts into specic values \
                for numerical calculation.

        Args:
            df_sym (list): derivatives of l,\
                [l, lx, lu, lxx, lux, luu, lf, lfx, lfxx].
            dt (Sympy.symbol): discretization 
            dt_value (float): value of dt
            scalar_dict (dict): {"name": (symbol, value)}).
            vector_dict (dict): {"name": (symbol, value)}).
            matrix_dict (dict): {"name": (symbol, value)}).

        Returns:
            dl_subs (list) : constants-substituted symbolic \
                cost function derivatives.
        """
        self.df_subs = symutils.substitute_constants_list(
            dl_sym, scalar_dict, vector_dict, matrix_dict
        )
        return self.df_subs


class NumCost:
    """ Turn symbolic dynamics into fast universal function.

    Args:
        x (sympy.Matrix): state vector.
        u (sympy.Matrix): control input vector.
        l (sympy.Symbol): stage cost
        lx (sympy.Matrix): derivative of l w.r.t. state x
        lu (sympy.Matrix): derivative of l w.r.t. input u
        lxx (sympy.Matrix): derivative of lx w.r.t. x.
        lux (sympy.Matrix): derivative of lu w.r.t. x.
        luu (sympy.Matrix): derivative of lu w.r.t. u.
        lf (sympy.Matrix): terminal cost
        lfx (sympy.Matrix): derivative of lf w.r.t. state x
        lfxx (sympy.Matrix): derivative of lfx w.r.t. state x

    Note:
        Confirm all constant symbolcs (e.g. Q, R,...) are substituted.
    """
    def __init__(self, x, u, t, l_sym, lx_sym, lu_sym, lxx_sym, lux_sym, luu_sym,
                 lf_sym, lfx_sym, lfxx_sym):
        l_ufunc = symutils.lambdify([x, u, t], l_sym)
        lx_ufunc = symutils.lambdify([x, u, t], lx_sym)
        lu_ufunc = symutils.lambdify([x, u, t], lu_sym)
        lxx_ufunc = symutils.lambdify([x, u, t], lxx_sym)
        lux_ufunc = symutils.lambdify([x, u, t], lux_sym)
        luu_ufunc = symutils.lambdify([x, u, t], luu_sym)
        lf_ufunc = symutils.lambdify([x, t], lf_sym)
        lfx_ufunc = symutils.lambdify([x, t], lfx_sym)
        lfxx_ufunc = symutils.lambdify([x, t], lfxx_sym)
        self.dl = [l_ufunc, lx_ufunc, lu_ufunc, lxx_ufunc, lux_ufunc, luu_ufunc,
                   lf_ufunc, lfx_ufunc, lfxx_ufunc]


    def get_derivatives(self):
        """ Returns dynamics ufunction.

            Returns:
                dl (list): ufunction list of 
                [l_ufunc, lx_ufunc, lu_ufunc, lxx_ufunc, lux_ufunc,  luu_ufunc,
                 lf_ufunc, lfx_ufunc, lfxx_ufunc]
        """
        return self.dl
