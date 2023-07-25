import sympy as sym
import numpy as np

from ocpy import symutils


class SymCost:
    """ Symbolic Cost class.
    """
    def __init__(self, x: sym.Matrix, u: sym.Matrix, t: sym.Symbol,
                 l: sym.Symbol, lf: sym.Symbol):    
        """ Symbolic Cost class.
    
        Args:
            x (sym.Matrix): State vector.
            u (sym.Matrix): Control input vector.
            t (sym.Symbol): Time.
            l (sym.Symbol): Stage cost.
            lf (sym.Symbol): Terminal cost.
        """
        lx   = symutils.diff_scalar(l, x)
        lu   = symutils.diff_scalar(l, u)
        lxx  = symutils.diff_vector(lx, x)
        lux  = symutils.diff_vector(lu, x)
        luu  = symutils.diff_vector(lu, u)
        lfx  = symutils.diff_scalar(lf, x)
        lfxx = symutils.diff_vector(lfx, x)
        self.l    = l
        self.lx   = lx
        self.lu   = lu
        self.lxx  = lxx
        self.lux  = lux
        self.luu  = luu
        self.lf   = lf
        self.lfx  = lfx
        self.lfxx = lfxx
        self.dl = (l, lx, lu, lxx, lux, luu, lf, lfx, lfxx)

    @staticmethod
    def InitByManual(x: sym.Matrix, u: sym.Matrix, t: sym.Symbol,
                     l: sym.Symbol, lx: sym.Matrix, lu: sym.Matrix,
                     lxx: sym.Matrix, lux: sym.Matrix, luu: sym.Matrix,
                     lf: sym.Symbol, lfx: sym.Matrix, lfxx: sym.Matrix):
        """ Initialize dynamics derivatives manually.

        Args:
            x (sym.Matrix): State vector.
            u (sym.Matrix): Control input vector.
            t (sym.Symbol): Time.
            l (sym.Matrix): Stage cost
            lx (sym.Symbol): Derivative of l w.r.t. state x
            lu (sym.Symbol): Derivative of l w.r.t. input u
            lxx (sym.Matrix): Derivative of lx w.r.t. x.
            lux (sym.Matrix): Derivative of lu w.r.t. x.
            luu (sym.Matrix): Derivative of lu w.r.t. u.
            lf (sym.Matrix): Terminal cost
            lfx (sym.Matrix): Derivative of lf w.r.t. state x
            lfxx (sym.Matrix): Derivative of lfx w.r.t. state x
        """
        symcost = SymCost(x, u, t, sym.zeros(1)[0, 0], sym.zeros(1)[0, 0])
        symcost.l    = l
        symcost.lx   = lx
        symcost.lu   = lu
        symcost.lxx  = lxx
        symcost.lux  = lux
        symcost.luu  = luu
        symcost.lf   = lf
        symcost.lfx  = lfx
        symcost.lfxx = lfxx
        symcost.dl = (l, lx, lu, lxx, lux, luu, lf, lfx, lfxx)

    def get_derivatives(self):
        """ Get derivatives of dynamics.

        Returns:
            dl (tuple): (l, lx, lu, lxx, lux, luu, lf, lfx, lfxx)
        """
        return self.dl

    def substitute_constatnts(self,
                              scalar_dict: dict=None, vector_dict: dict=None, 
                              matrix_dict: dict=None, dt: sym.Symbol=None,
                              dt_value: float=None):
        """ Substitute symbolic constatnts into specic values \
                for numerical calculation.

        Args:
            scalar_dict (dict): {"name": (symbol, value)}).
            vector_dict (dict): {"name": (symbol, value)}).
            matrix_dict (dict): {"name": (symbol, value)}).
            dt (sym.Symbol): Discretization step.
            dt_value (float): Value of dt.

        Returns:
            dl_subs (tuple) : Constants-substituted symbolic \
                cost function derivatives.
        """
        dl_subs = symutils.substitute_constants_list(
            self.dl, scalar_dict, vector_dict, matrix_dict, dt, dt_value
        )
        self.dl_subs = tuple(dl_subs)
        return self.dl_subs

class NumCost:
    """ Turn symbolic dynamics into fast universal function.
    """
    def __init__(self, x: sym.Matrix, u: sym.Matrix, t: sym.Symbol, dt: sym.Symbol,
                 l_sym: sym.Matrix, lx_sym: sym.Matrix, lu_sym: sym.Matrix, 
                 lxx_sym: sym.Matrix, lux_sym: sym.Matrix, luu_sym: sym.Matrix,
                 lf_sym: sym.Matrix, lfx_sym: sym.Matrix, lfxx_sym: sym.Matrix):
        """ Turn symbolic dynamics into fast universal function.
        Args:
            x (sym.Matrix): State vector.
            u (sym.Matrix): Control input vector.
            l (sym.Symbol): Stage cost
            lx (sym.Matrix): Derivative of l w.r.t. state x
            lu (sym.Matrix): Derivative of l w.r.t. input u
            lxx (sym.Matrix): Derivative of lx w.r.t. x.
            lux (sym.Matrix): Derivative of lu w.r.t. x.
            luu (sym.Matrix): Derivative of lu w.r.t. u.
            lf (sym.Matrix): Terminal cost
            lfx (sym.Matrix): Derivative of lf w.r.t. state x
            lfxx (sym.Matrix): Derivative of lfx w.r.t. state x

        Note:
            Confirm that all constant symbolcs (e.g. Q, R,...) are substituted.
        """
        dl_sym = [l_sym, lx_sym, lu_sym, lxx_sym, lux_sym, luu_sym,
                  lf_sym, lfx_sym, lfxx_sym]
        dl = []
        for i, func_sym in enumerate(dl_sym):
            args = [x, u, t] if i < 6 else [x, t]
            dim_reduction = True if i in [1, 2, 7] else False
            dl.append(symutils.lambdify(args, func_sym, dim_reduction))
        self.l    = dl[0]
        self.lx   = dl[1]
        self.lu   = dl[2]
        self.lxx  = dl[3]
        self.lux  = dl[4]
        self.luu  = dl[5]
        self.lf   = dl[6]
        self.lfx  = dl[7]
        self.lfxx = dl[8]
        self.dl   = tuple(dl)

    def get_derivatives(self):
        """ Return dynamics ufunction.

            Returns:
                dl (tuple): (l, lx, lu, lxx, lux, luu, lf, lfx, lfxx)
            
            Note:
                Arguments of l are [x, u, t], and of lf are [x, t].
        """
        return self.dl
