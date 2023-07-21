import sympy as sym
import numpy as np

from ocpy import symutils


class SymIneqConstraints:
    """ Create inequality constraints class. Derivatives are calculated via sympy.
    """

    def __init__(self, x: sym.Matrix, u: sym.Matrix, t: sym.Symbol, g: sym.Matrix):
        """ Create inequality constraints class. Derivatives are calculated via sympy.
        Args:
            x (sym.Matrix): State vector.
            u (sym.Matrix): Control input vector.
            t (sym.Symbol): Time.
            g (sym.Matrix): Stack of equality constraints. g(x, u, t) <= 0.
        """
        gx  = symutils.diff_vector(g, x)
        gu  = symutils.diff_vector(g, u)
        gxx = symutils.diff_matrix(gx, x)
        gux = symutils.diff_matrix(gu, x)
        guu = symutils.diff_matrix(gu, u)
        self.g   = g
        self.gx  = gx
        self.gu  = gu
        self.gxx = gxx
        self.gux = gux
        self.guu = guu
        self.dg = [g, gx, gu, gxx, gux, guu]
    
    @staticmethod
    def InitByManual(x: sym.Matrix, u: sym.Matrix, t: sym.Symbol,
                     g: sym.Matrix, gx: sym.Matrix, gu: sym.Matrix,
                     gxx: sym.Matrix, gux: sym.Matrix, guu: sym.Matrix): 
        """ Initialize inequality constraints derivatives manually.

        Args:
            x (sym.Matrix): State vector.
            u (sym.Matrix): Control input vector.
            t (sym.Symbol): Time.
            g (sym.Matrix): Stack of inequality constraints.
            gx (sym.Matrix): Derivative of g w.r.t. state x.
            gu (sym.Matrix): Derivative of g w.r.t. input u.
            gxx (sym.Array): Derivative of g w.r.t. x and x.
            gux (sym.Array): Derivative of g w.r.t. u and x.
            gux (sym.Array): Derivative of g w.r.t. u and u.
            continuous (bool): Is continous model.
        """
        symineq = SymIneqConstraints(t, x, u, sym.zeros(1, 1))
        symineq.g   = g
        symineq.gx  = gx
        symineq.gu  = gu
        symineq.gxx = gxx
        symineq.gux = gux
        symineq.guu = guu
        symineq.dg = [g, gx, gu, gxx, gux, guu]
        return symineq

    def get_derivatives(self):
        """ Return derivatives of inequality constraints.

            Returns:
                dg (list):[g, gx, gu, gxx, gux, guu]
        """
        return self.dg
    
    def dimension(self):
        """ Dimension of inequality constraints.
        """
        return self.dg[0].shape[0]

    def substitute_constatnts(self,
                              scalar_dict: dict=None, vector_dict: dict=None, 
                              matrix_dict: dict=None, dt: sym.Symbol=None,
                              dt_value: float=None,):
        """ Substitute symbolic constatnts into specic values \
                for numerical calculation.

            Args:
                dg_sym (list): derivatives of g. [g, gx, gu, gxx, gux, guu].
                scalar_dict (dict) : {"name": (symbol, value)}) 
                vector_dict (dict) : {"name": (symbol, value)}) 
                matrix_dict (dict) : {"name": (symbol, value)}) 
            Returns:
                dg_subs (list) : constants-substituted symbolic \
                    inequality constraints derivatives.
        """
        self.dg_subs = symutils.substitute_constants_list(
            self.dg, scalar_dict, vector_dict, matrix_dict, dt, dt_value
        )
        return self.dg_subs


class NumIneqConstraints:
    """ Generate numerical function of inequality constraints \
        from symbolic expression.

    """
    def __init__(self, x: sym.Matrix, u: sym.Matrix, t: sym.Symbol, dt: sym.Symbol,
                 g_sym: sym.Matrix, gx_sym: sym.Matrix, gu_sym: sym.Matrix,
                 gxx_sym: sym.Matrix, gux_sym: sym.Matrix,
                 guu_sym: sym.Matrix):
        """ Turn symbolic inequality constraints into fast universal function.

        Args:
            x (sym.Matrix): state vector.
            u (sym.Matrix): control input vector.
            t (sym.Symbol): Time.
            g_sym (sym.Matrix): Stack of inequality constraints. g(x, u, t) <= 0.
            gx_sym (sym.Matrix): Derivative of g w.r.t. state x.
            gu_sym (sym.Matrix): Derivative of g w.r.t. input u.
            gxx_sym (sym.Array): Derivative of g w.r.t. x and x.
            gux_sym (sym.Array): Derivative of g w.r.t. u and x.
            gux_sym (sym.Array): Derivative of g w.r.t. u and u.

        Note:
            Confirm all symbolic constants (e.g. mass, lentgth,) are substituted.   
        """
        args = [x, u, t]
        dg_sym = [g_sym, gx_sym, gu_sym, gxx_sym, gux_sym, guu_sym]
        dg = []
        for i, func_sym in enumerate(dg_sym):
            args = [x, u, t]
            dim_reduction = True if i == 0 else False
            dg.append(symutils.lambdify(args, func_sym, dim_reduction))
        self.dg_sym = dg_sym
        self.g   = dg[0]
        self.gx  = dg[1]
        self.gu  = dg[2]
        self.gxx = dg[3]
        self.gux = dg[4]
        self.guu = dg[5]
        self.dg  = dg

    def get_derivatives(self):
        """ return inequality constraints derivarives ufunction.

        Returns:
            dg (list): ufunc list of [g, gx, gu, gxx, gux, guu],\
            whose arguments are x, u, t
        """
        return self.dg_sym[0].shape[0]
    

class SymEqConstraints:
    """ Create equality constraints class. Derivatives are calculated via sympy.
    """

    def __init__(self, x: sym.Matrix, u: sym.Matrix, t: sym.Symbol, h: sym.Matrix):
        """ Create equality constraints class. Derivatives are calculated via sympy.
        
        """
        hx  = symutils.diff_vector(h, x)
        hu  = symutils.diff_vector(h, u)
        hxx = symutils.diff_matrix(hx, x)
        hux = symutils.diff_matrix(hu, x)
        huu = symutils.diff_matrix(hu, u)
        self.h   = h
        self.hx  = hx
        self.hu  = hu
        self.hxx = hxx
        self.hux = hux
        self.huu = huu
        self.dh = [h, hx, hu, hxx, hux, huu]
    
    @staticmethod
    def InitByManual(x: sym.Matrix, u: sym.Matrix, t: sym.Symbol,
                     h: sym.Matrix, hx: sym.Matrix, hu: sym.Matrix,
                     hxx: sym.Matrix, hux: sym.Matrix, huu: sym.Matrix): 
        """ Initialize equality constraints derivatives manually.

        Args:
            x (sym.Matrix): State vector.
            u (sym.Matrix): Control input vector.
            t (sym.Symbol): Time.
            h (sym.Matrix): Stack of equality constraints. g(x, u, t) <= 0.
            hx (sym.Matrix): Derivative of h w.r.t. state x.
            hu (sym.Matrix): Derivative of h w.r.t. input u.
            hxx (sym.Array): Derivative of h w.r.t. x and x.
            hux (sym.Array): Derivative of h w.r.t. u and x.
            hux (sym.Array): Derivative of h w.r.t. u and u.
            continuous (bool): Is continous model.
        """
        symeq = SymEqConstraints(t, x, u, sym.zeros(1, 1))
        symeq.h   = h
        symeq.hx  = hx
        symeq.hu  = hu
        symeq.hxx = hxx
        symeq.hux = hux
        symeq.huu = huu
        symeq.dh = [h, hx, hu, hxx, hux, huu]
        return symeq

    def get_derivatives(self):
        """ Return derivatives of equality constraints.

            Returns:
                dg (list):[h, hx, hu, hxx, hux, huu]
        """
        return self.dh
    
    def dimension(self):
        """ Dimension of inequality constraints.
        """
        return self.dh[0].shape[0]

    def substitute_constatnts(self,
                              scalar_dict: dict=None, vector_dict: dict=None, 
                              matrix_dict: dict=None, dt: sym.Symbol=None,
                              dt_value: float=None,):
        """ Substitute symbolic constatnts into specic values \
                for numerical calculation.

            Args:
                dh_sym (list): derivatives of h, [h, hx, hu, hxx, hux, huu].
                scalar_dict (dict) : {"name": (symbol, value)}) 
                vector_dict (dict) : {"name": (symbol, value)}) 
                matrix_dict (dict) : {"name": (symbol, value)}) 
            Returns:
                dh_subs (list) : constants-substituted symbolic
                    equality constraints derivatives.
        """
        self.dh_subs = symutils.substitute_constants_list(
            self.dh, scalar_dict, vector_dict, matrix_dict, dt, dt_value
        )
        return self.dh_subs


class NumEqConstraints:
    """ Generate numerical function of equality constraints \
        from symbolic expression.

    """
    def __init__(self, x: sym.Matrix, u: sym.Matrix, t: sym.Symbol, dt: sym.Symbol,
                 h_sym: sym.Matrix, hx_sym: sym.Matrix, hu_sym: sym.Matrix,
                 hxx_sym: sym.Matrix, hux_sym: sym.Matrix,
                 huu_sym: sym.Matrix):
        """ Turn symbolic inequality constraints into fast universal function.

        Args:
            x (sym.Matrix): state vector.
            u (sym.Matrix): control input vector.
            t (sym.Symbol): Time.
            h_sym (sym.Matrix): Stack of inequality constraints. h(x, u, t) <= 0.
            hx_sym (sym.Matrix): Derivative of h w.r.t. state x.
            hu_sym (sym.Matrix): Derivative of h w.r.t. input u.
            hxx_sym (sym.Array): Derivative of h w.r.t. x and x.
            hux_sym (sym.Array): Derivative of h w.r.t. u and x.
            hux_sym (sym.Array): Derivative of h w.r.t. u and u.

        Note:
            Confirm all symbolic constants (e.g. mass, lentgth,) are substituted.   
        """
        args = [x, u, t]
        dh_sym = [h_sym, hx_sym, hu_sym, hxx_sym, hux_sym, huu_sym]
        dh = []
        for i, func_sym in enumerate(dh_sym):
            args = [x, u, t]
            dim_reduction = True if i == 0 else False
            dh.append(symutils.lambdify(args, func_sym, dim_reduction))
        self.dh_sym = dh_sym
        self.h   = dh[0]
        self.hx  = dh[1]
        self.hu  = dh[2]
        self.hxx = dh[3]
        self.hux = dh[4]
        self.huu = dh[5]
        self.dh  = dh

    def get_derivatives(self):
        """ return inequality constraints derivarives ufunction.

        Returns:
            dh (list): ufunc list of [h, hx, hu, hxx, hux, huu],\
            whose arguments are x, u, t
        """
        return self.dh_sym[0].shape[0]
