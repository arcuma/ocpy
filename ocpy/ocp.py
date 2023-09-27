import sympy as sym
import numpy as np
from IPython.display import display, Math

from ocpy import symutils
from ocpy.dynamics import SymDynamics, NumDynamics
from ocpy.cost import SymCost, NumCost
from ocpy.constraints import SymIneqConstraints, NumIneqConstraints,\
    SymEqConstraints, NumEqConstraints


class OCP:
    """ Class that describes optimal control problem.
    """
    def __init__(self, ocp_name: str, n_x: int, n_u: int, n_g=0, n_h=0):
        """ Constructor.

        Args:
            ocp_name (str): Simulation name.
            n_x (int): Dimension of state.
            n_u (int): Dimension of input.
            n_g (int): Dimension of inequality constraints.
            n_h (int): Dimension of equality constraints.
        """
        self._ocp_name = ocp_name
        self._n_x = n_x
        self._n_u = n_u
        self._n_g = n_g
        self._n_h = n_h

        self._has_ineq_constraints = False
        if n_g > 0:
            self._has_ineq_constraints = True

        self._has_eq_constraints = False
        if n_h > 0:
            self._has_eq_constraints = True

        # state and input (symbol)
        self._x = symutils.define_vector('x', n_x)
        self._u = symutils.define_vector('u', n_u)
        self._t = sym.Symbol('t')
        self._dt = sym.Symbol('dt')

        # dictionary holding symbols and values of constants
        self._scalar_dict = {}
        self._vector_dict = {}
        self._matrix_dict = {}

        # flags
        self._is_lambdified = False

        # in define()
        self._T = None
        self._N = None
        self._dt_value = None
        self._t0 = None
        self._x0 = None
        self._is_continuous = None
        self._f_original = None
        self._l_original = None
        self._sym_dynamics = None
        self._sym_cost = None
        self._sym_ineq_constraints = None
        self._sym_eq_constraints = None
        self._df_sym = None
        self._dl_sym = None
        self._dg_sym = None
        self._dh_sym = None
        self._is_ocp_defined = False

        # in lambdify()
        self._df_subs = None
        self._dl_subs = None
        self._dg_subs = None
        self._dh_subs = None
        self._num_dynamics = None
        self._num_cost = None
        self._num_ineq_constraints = None
        self._num_eq_constraints = None
        self._df_num = None
        self._dl_num = None
        self._dg_num = None
        self._dh_num = None
        self._is_lambdified = False

    def define(self,
            f: sym.Matrix, l: sym.Symbol, lf: sym.Symbol,
            g: sym.Matrix=None, h: sym.Matrix=None,
            t0: float=None, x0: np.ndarray=None, T: float=None, N: int=None,
            is_continuous: bool=True, simplification: bool=False
        ):
        """ Define optimal control problem. 

        Args:
            f (sym.Matrix): State function.
            l (sym.Symbol): Stage cost.
            lf (sym.Symbol): Terminal cost.
            g (sym.Matrix): Inequality constraints. \
                If there is no inequality constraints, pass None.
            h (sym.Matrix): Equality constraints. \
                If there is no equality constraints, pass None.
            t0 (float): Initial time.
            x0 (np.ndarray): Initial state. Size must be n_x.
            T (float): Horizon length.
            N (int): Discretization grid number.
            is_continuous (bool=True): Is dynamics and costs are continuous-time.\
                If true, they will be discretized. Default is False.
            simplification (bool=False): If True, functions are simplified.\
                Simplification may take time. Default is False.
        """
        x, u, t, dt = self._x, self._u, self._t, self._dt
        n_x, n_u = self._n_x, self._n_u

        t0 = float(t0) if t0 is not None else 0.0

        if x0 is not None:
            x0 = np.array(x0, dtype=float)
            assert x0.shape[0] == n_x
        else:
            x0 = np.zeros(n_x)
        
        if T is None:
            T = 5.0
            print(f"T was set to the defalut value {T}.")
        if N is None:
            N = 200
            print(f"N was set to the defalut value {N}.")

        T = float(T)
        N = int(N)

        # if l and lf are 1x1 Matrix, turn it into Symbol
        if isinstance(l, sym.Matrix):
            l = l[0, 0]
        if isinstance(lf, sym.Matrix):
            lf = lf[0, 0]

        # symbolic derivatives of dynamics and cost.
        if is_continuous:
            sym_dynamics = SymDynamics(x, u, t, f)
            sym_cost = SymCost(x, u, t, l, lf)
        else:
            # if discrete, regard it discretized by forward-Euler method.
            sym_dynamics = SymDynamics(x, u, t, (f - x) / dt)
            sym_cost = SymCost(x, u, t, l / dt, lf)
        # (f, fx, fu, fxx, fux, fuu)
        df_sym = sym_dynamics.get_derivatives()
        # (l, lx, lu, lxx, lux, luu, lf, lfx, lfxx)
        dl_sym = sym_cost.get_derivatives()

        # inequality constraints
        if g is not None:
            sym_ineq_constraints = SymIneqConstraints(x, u, t, g)
            dg_sym = sym_ineq_constraints.get_derivatives()
            # hold
            self._sym_ineq_constraints = sym_ineq_constraints
            self._dg_sym = dg_sym

        # equality constraints
        if h is not None:
            sym_eq_constraints = SymEqConstraints(x, u, t, h)
            dh_sym = sym_eq_constraints.get_derivatives()
            # hold
            self._sym_ineq_constraints = sym_eq_constraints
            self._dh_sym = dh_sym

        # simplify
        if simplification:
            symutils.simplify(df_sym)
            symutils.simplify(dl_sym)
            if self._has_ineq_constraints:
                symutils.simplify(dg_sym)
            if self._has_eq_constraints:
                symutils.simplify(dh_sym)

        # hold
        self._t0 = t0
        self._x0 = x0
        self._T = T
        self._N = N
        self._dt_value = T / N
        self._is_continuous = is_continuous
        self._sym_dynamics = sym_dynamics
        self._sym_cost = sym_cost
        self._f_original = f
        self._l_original = l
        self._df_sym = df_sym
        self._dl_sym = dl_sym
        self._is_ocp_defined = True
        
        # generate lambda function.
        self.lambdify()

    def define_unconstrained(self,
            f: sym.Matrix, l: sym.Symbol, lf: sym.Symbol,
            t0: float=None, x0: np.ndarray=None, T: float=None, N: int=None, 
            is_continuous: bool=True, simplification: bool=False
        ):
        """ Define optimal control problem. 

        Args:
            f (sym.Matrix): State function.
            l (sym.Symbol): Stage cost.
            lf (sym.Symbol): Terminal cost.
            t0 (float): Initial time.
            x0 (np.ndarray): Initial state. Size must be n_x.
            T (float): Horizon length.
            N (int): Discretization grid number.
            is_continuous (bool=True): Is dynamics and costs are continuous-time.\
                If true, they will be discretized. Default is False.
            simplification (bool=False): If True, functions are simplified.\
                Simplification may take time. Default is False.
        """
        self.define(f=f, l=l, lf=lf, g=None, h=None,t0=t0 , x0=x0, T=T, N=N,
                    is_continuous=is_continuous, simplification=simplification)

    def lambdify(self):
        """ Generate sympy symbolic expression into numpy function.\

        Returns:
            tuple: (df, dl) = ((f, fx, fu, fxx, fux, fuu), \
                (l, lx, lu, lxx, lux, luu, lf, lfx, lfxx))
        """
        assert self._is_ocp_defined

        x = self._x
        u = self._u
        t = self._t
        dt = self._dt

        # dynamics
        df_subs = symutils.substitute_constants_list(
            self._df_sym, self._scalar_dict, self._vector_dict, self._matrix_dict,
            self._dt, self._dt_value)
        num_dynamics = NumDynamics(x, u, t, dt, *df_subs)
        df_num = num_dynamics.get_derivatives()

        # cost
        dl_subs = symutils.substitute_constants_list(            
            self._dl_sym, self._scalar_dict, self._vector_dict, self._matrix_dict,
            self._dt, self._dt_value)
        num_cost = NumCost(x, u, t, dt, *dl_subs)
        dl_num = num_cost.get_derivatives()

        # inequality constraints
        if self._has_ineq_constraints:
            dg_subs = symutils.substitute_constants_list(
                self._dg_sym, self._scalar_dict, self._vector_dict,
                self._matrix_dict,
                )
            num_ineq_constraints = NumIneqConstraints(
                x, u, t, dt, *dg_subs
                )
            dg_num = num_ineq_constraints.get_derivatives()
            self._dg_subs = dg_subs
            self._num_ineq_constraints = num_ineq_constraints
            self._dg_num = dg_num

        # equality constraints
        if self._has_eq_constraints:
            dh_subs = symutils.substitute_constants_list(
                self._dh_sym, self._scalar_dict, self._vector_dict,
                self._matrix_dict,
                )
            num_eq_constraints = NumEqConstraints(
                x, u, t, dt, *dh_subs
                )
            dh_num = num_eq_constraints.get_derivatives()
            self._dh_subs = dh_subs
            self._num_eq_constrants = num_eq_constraints
            self._dh_num = dh_num

        # hold
        self._df_subs = df_subs
        self._num_dynamics = num_dynamics
        self._df_num = df_num
        self._dl_subs = dl_subs
        self._num_cost = num_cost
        self._dl_num = dl_num
        self._is_lambdified = True

    def set_initial_condition(self, t0: float=None, x0: np.ndarray=None):
        """ Set t0 and x0.

        Args:
            t0 (float): Initial time of horizon.
            x0 (float): Initial state of horizon.
        """
        if t0 is not None:
            self._t0 = float(t0)
        if x0 is not None:
            x0 = np.array(x0, dtype=float).reshape(-1)
            assert x0.shape[0] == self._n_x
            self._x0 = np.array(x0, dtype=float)
    
    def set_horizon(self, T: float=None, N: float=None):
        """ Set T and N.

        Args:
            T (float): Horizon length.
            N (int): Discretization grid number.
        """
        if T is not None:
            assert T > 0
            self._T = float(T)
        if N is not None:
            assert isinstance(N, int) and N > 0
            self._N = N
        self._dt_val = self._N / self._T

    def set_parameters(self, t0: float=None, x0: np.ndarray=None,
                       T: float=None, N: int=None):
        """ Set parameters.

        Args:
            t0 (float): Initial time of horizon.
            x0 (float): Initial state of horizon.
            T (float): Horizon length.
            N (int): Discretization grid number.
        """
        assert self._is_ocp_defined
        self.set_initial_condition(t0, x0)
        self.set_horizon(T, N)

    def get_initial_condition(self):
        """ Return t0 and x0.
        """
        return self._t0, self._x0
    
    def get_horizon(self):
        """ Return T and N.
        """
        return self._T, self._N

    def get_t0(self) -> float:
        """ Return t0.
        """
        assert self._is_ocp_defined
        return self._t0

    def get_x0(self) -> np.ndarray:
        """ Return initial state x0.
        """
        assert self._is_ocp_defined
        return self._x0

    def get_T(self) -> float:
        """ Return horizon length.
        """
        assert self._is_ocp_defined
        return self._T

    def get_N(self) -> int:
        """ Return number of discretization grids.
        """
        assert self._is_ocp_defined
        return self._N

    def get_dt_value(self) -> float:
        """ Return value of dt (=T/N).
        """
        assert self._is_ocp_defined
        return self._dt_value

    # symbolic
    def get_df_symbolic(self) -> tuple:
        """ Return symbolic derivatives of dynamics.

        Returns:
            tuple: (f, fx, fu, fxx, fux, fuu)
        """
        assert self._is_ocp_defined
        return self._df_sym
    
    def get_dl_symbolic(self) -> tuple:
        """ Return symbolic derivatives of costs.

        Returns:
            tuple: (l, lx, lu, lxx, lux, luu, lf, lfx, lfxx)
        """
        assert self._is_ocp_defined
        return self._dl_sym
    
    def get_dg_symbolic(self) -> tuple:
        """ Return symbolic derivatives of inequality constraints.
            If no inequality constraints, return None.

        Returns:
            tuple: (g, gx, gu, gxx, gux, guu)
        """
        assert self._is_ocp_defined
        return self._dg_sym

    def get_dh_symbolic(self) -> tuple:
        """ Return symbolic derivatives of equality constraints.
            If no equality constraints, return None.

        Returns:
            tuple: (h, hx, hu, hxx, hux, huu)
        """
        assert self._is_ocp_defined
        return self._dg_sym
    
    # constants-substituted symbolic
    def get_df_symbolic_substituted(self) -> tuple:
        """ Return constants-substituted symbolic derivatives of dynamics.

        Returns:
            tuple: (f, fx, fu, fxx, fux, fuu)
        """
        assert self._is_lambdified
        return self._df_subs

    def get_dl_symbolic_substituted(self) -> tuple:
        """ Return constants-substituted symbolic derivatives of costs.

        Returns:
            tuple: (l, lx, lu, lxx, lux, luu, lf, lfx, lfxx)
        """
        assert self._is_lambdified
        return self._dl_subs

    def get_dg_symbolic_substituted(self) -> tuple:
        """ Return constants-substituted symbolic derivatives \
            of inequality constraints.
            If no inequality constraints, return None.

        Returns:
            tuple: (g, gx, gu, gxx, gux, guu)
        """
        assert self._is_lambdified
        return self._dg_subs

    def get_dh_symbolic_substituted(self) -> tuple:
        """ Return constants-substituted symbolic derivatives \
            equality constraints.
            If no equality constraints, return None.

        Returns:
            tuple: (h, hx, hu, hxx, hux, huu)
        """
        assert self._is_lambdified
        return self._dh_subs

    # numba-numpy lambda functions
    def get_df(self) -> tuple:
        """ Return derivatives of dynamics.

        Returns:
            tuple (f, fx, fu, fxx, fux, fuu)
        """
        assert self._is_lambdified
        return self._df_num
    
    def get_dl(self) -> tuple:
        """ Return derivatives of cost.

        Returns:
            tuple (l, lx, lu, lxx, lux, luu, lf, lfx, lfxx)
        """
        assert self._is_lambdified
        return self._dl_num

    def get_dg(self) -> tuple:
        """ Return derivatives of inequality constraints.
            If no inequality constraints, return None.

        Returns:
            tuple (g, gx, gu, gxx, gux, guu)
        """
        assert self._is_lambdified
        return self._dg_num

    def get_dh(self) -> tuple:
        """ Return derivatives of equality constraints.
            If no equality constraints, return None.

        Returns:
            tuple (h, hx, hu, hxx, hux, huu)
        """
        assert self._is_lambdified
        return self._dh_num
    
    def get_ocp_name(self) -> str:
        return self._ocp_name

    def get_x(self) -> sym.Matrix:
        """ Return sym expression of x.
        """
        return self._x.copy()

    def get_u(self) -> sym.Matrix:
        """ Return sym expression of u.
        """
        return self._u.copy()
    
    def get_t(self) -> sym.Symbol:
        """ Return sym symbol of t.
        """
        return self._t
    
    def get_dt(self) -> sym.Symbol:
        """ Return sym symbol of dt.
        """
        return self._dt

    def get_n_x(self) -> int:
        """ Return dimension of x.
        """
        return self._n_x

    def get_n_u(self) -> int:
        """ Return dimension of u.
        """
        return self._n_u
    
    def get_n_g(self) -> int:
        """ Return dimension of inequality constraints g.
        """
        return self._n_g
    
    def get_n_h(self) -> int:
        """ Return dimension of equality constraints h.
        """
        return self._n_h

    def get_f_empty(self):
        """ Return n_x*1-size symbolic zero vector.
        """
        return sym.zeros(self._n_x, 1)
    
    @staticmethod
    def get_zero_vector(m: int) -> sym.Matrix:
        """ Return symbolic m*1-size zero Matrix.

        Args:
            m (int): Dimension of vector.
        """
        return sym.zeros(m, 1)

    @staticmethod   
    def get_zero_matrix(m: int, n: int) -> sym.Matrix:
        """ Return symbolic m*n-size zero Matrix.

        Args:
            m (int): dimension of rows
            n (int): dimension of columns
        """
        return sym.zeros(m, n)
    
    @staticmethod
    def get_barrier(w: sym.Matrix, w_min: np.ndarray=None, w_max: np.ndarray=None,
                    mu: float=1e-2, matrix_expr=True):
        """ Get logarithm barrier function for constraint w_min <= w <= w_max.

        Args:
            w (sym.Matrix): Vector. (Assuming control input).
            w_min (np.ndarray): Minimum value of w.
            w_max (np.ndarray): Minimum value of w.
            mu (float): Barrier coefficient.
            matrix_expr (float): If True, return 1x1 Matrix. Else Symbol.
        """
        n_w = w.shape[0]

        cost = 0
        if w_min is not None:
            assert n_w == len(w_min)
            for i in range(n_w):
                cost += -mu * sym.ln(w[i] - w_min[i])
        if w_max is not None:
            assert n_w == len(w_max)
            for i in range(n_w):
                cost += -mu * sym.ln(w_max[i] - w[i])
        
        if matrix_expr:
            cost = sym.Matrix([cost])
        
        return cost
        
    @staticmethod
    def get_penalty(w: sym.Matrix, w_min: np.ndarray=None, w_max: np.ndarray=None,
                    mu: float=1e-2, matrix_expr=True):
        """ Get exponential penalty function for constraint w_min <= w <= w_max.

        Args:
            w (sym.Matrix): Vector. (Assuming control input).
            w_min (np.ndarray): Minimum value of w.
            w_max (np.ndarray): Minimum value of w.
            mu (float): Barrier coefficient.
            matrix_expr (float): If True, return 1x1 Matrix. Else Symbol.
        """
        n_w = w.shape[0]

        cost = 0
        if w_min is not None:
            assert n_w == len(w_min)
            for i in range(n_w):
                cost += mu * sym.exp(-(w[i] - w_min[i]))
        if w_max is not None:
            assert n_w == len(w_max)
            for i in range(n_w):
                cost += mu * sym.exp(-(w_max[i] - w[i]))
        
        if matrix_expr:
            cost = sym.Matrix([cost])
        
        return cost

    def define_scalar_constant(self, constant_name: str, value: float):
        """
        Args:
            constant_name (str): Name of constant.
            value (float): Constant value.

        Returns:
            scalar_symbol (sym.Symbol) : sympy symbol of constant.
        """
        scalar_symbol = sym.Symbol(constant_name)
        self._scalar_dict[constant_name] = (scalar_symbol, value)
        return scalar_symbol

    def define_scalar_constants(self, constant_list: list):
        """
        Args:
            constant_list (list): list of tuple('name', value)

        Returns:
            scalar_symbols (list) : list of symbols.
        """
        scalar_symbols = []
        for name, val in constant_list:
            scalar_symbol = self.define_scalar_constant(name, val)
            scalar_symbols.append(scalar_symbol)
        return scalar_symbols

    def define_vector_constant(self, constant_name: str, 
                               vector_value: np.ndarray | list):
        """
        Args:
            constant_name (str): Name of constant.
            vector_value (np.ndarray): 1d ndarray.
        Returns:
            vector_symbol (sym.Matrix) : n*1-size sympy.Matrix
        """
        if type(vector_value) == list:
            vector_value = np.array(vector_value)
        vec_symbol = symutils.define_vector(constant_name, len(vector_value))
        self._vector_dict[constant_name] = (vec_symbol, vector_value)
        return vec_symbol

    def define_vector_constants(self, constant_list: list):
        """
        Args:
            constant_list (list): list of tuple('name', value)
        Returns:
            vector_symbols (list) : list[sym.Matrix]. list of vector symbols.
        """
        vec_symbols = []
        for name, value in constant_list:
            vec_symbol = self.define_vector_constant(name, value)
            vec_symbols.append(vec_symbol)
        return vec_symbols

    def define_matrix_constant(self, constant_name: str,
                               matrix_value: np.ndarray | list):
        """
        Args:
            constant_name (str): Name of constant.
            vector_value (np.ndarray): 2d ndarray.

        Returns:
            matrix_symbol (sym.Matrix) : m*n-size sympy.Matrix
        """
        if isinstance(matrix_value, list):
            matrix_value = np.array(matrix_value)
        matrix_symbol = symutils.define_matrix(constant_name, *(matrix_value.shape))
        self._matrix_dict[constant_name] = (matrix_symbol, matrix_value)
        return matrix_symbol.copy()

    def define_matrix_constants(self, constant_list: list):
        """
        Args:
            constant_list (list): list of tuple('name', value)

        Returns:
            matrix_symbols (list) : list[sym.Matrix]. list of Matrix symbols.
        """
        matrix_symbols = []
        for name, value in constant_list:
            matrix_symbol = self.define_matrix_constant(name, value)
            matrix_symbols.append(matrix_symbol)
        return matrix_symbols   

    def display(self):
        """ Display problem.
        """
        display('State equation:')
        display(Math(r"\dot{x} = f(x, u, t) \equiv %s" % sym.latex(self._df_sym[0])))
        display('Stage cost:')
        display(Math(r"l(x, u, t) = %s" % sym.latex(self._dl_sym[0])))
        display('Terminal cost:')
        display(Math(r"l_f(x, t) = %s" % sym.latex(self._dl_sym[6])))
        if self._has_ineq_constraints:
            display('Inequality constraints:')
            display(Math(r"g(x, u, t) = %s \leq 0" % sym.latex(self._dg_sym[0])))
        if self._has_eq_constraints:
            display('Equality constraints:')
            display(Math(r"h(x, u, t) = %s \leq 0" % sym.latex(self._dh_sym[0])))
        display('Horizon length')
        display(Math(r"T = %s" % sym.latex(self._T)))
        display('Number of stage')
        display(Math(r"N = %s" % sym.latex(self._N)))
        display('Initial time')
        display(Math(r"t_0 = %s" % sym.latex(self._t0)))
        display('Initial state')
        display(Math(r"x_0 = %s" % sym.latex(sym.Matrix(self._x0))))

