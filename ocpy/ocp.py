import sympy as sym
import numpy as np

from ocpy import symutils
from ocpy.dynamics import SymDynamics, NumDynamics
from ocpy.cost import SymCost, NumCost
from ocpy.constraints import SymIneqConstraints, NumIneqConstraints,\
    SymEqConstraints, NumEqConstraints


class OCP:
    """ Class that describes optimal control problem.
    """
    def __init__(self, n_x: int, n_u: int, ocp_name: str):
        self._n_x = n_x
        self._n_u = n_u
        self._ocp_name = ocp_name
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
        self._us_guess = None
        self._is_continuous = None
        self._sym_dynamics = None
        self._sym_cost = None
        self._f_original = None
        self._l_original = None
        self._lf_original = None
        self._df_sym = None
        self._dl_sym = None
        self._dg_sym = None
        self._dh_sym = None
        self._has_ineq_constraints = False
        self._has_eq_constraints = False
        self._is_ocp_defined = False
        # in lambdify()
        self._df_subs = None
        self._dl_subs = None
        self._dg_subs = None
        self._dh_subs = None
        self._num_dynamics = None
        self._num_cost = None
        self._df_ufunc = None
        self._dl_ufunc = None
        self._dg_ufunc = None
        self._dh_ufunc = None
        self._is_lambdified = False

    def define(
            self, f: sym.Matrix, l: sym.Symbol, lf: sym.Symbol,
            g: sym.Matrix, h: sym.Matrix,
            T: float, N: int, t0: float, x0: np.ndarray, us_guess: np.ndarray,
            is_continuous: bool=True, simplification: bool=False):
        """ Define optimal control problem. 

        Args:
            f (sym.Matrix): State function.
            l (sym.Symbol): Stage cost.
            lf (sym.Symbol): Terminal cost.
            g (sym.Matrix): Inequality constraints. \
                If there is no inequality, pass None.
            h (sym.Matrix): Equality constraints. \
                If there is no equality, pass None.
            T (float): Horizon length
            N (int): Discretization grid number.
            t0 (float): Initial time.
            x0 (numpy.array): Initial state. size must be n_x.
            us_guess (numpy.array, optional): Guess of input trajectory. \
                size must be (N * n_u).
            is_continuous (bool=True): Is dynamics and costs are continuous-time.\
                If true, they will be discretized. Default is False.
            simplification (bool=False): If True, functions are simplified.\
                Simplification may take time. Default is False.
        """
        x, u, t, dt = self._x, self._u, self._t, self._dt
        n_x, n_u = self._n_x, self._n_u
        # turn x0 and us_guess into ndarray if not.
        if x0 is not None:
            x0 = np.array(x0, dtype=float)
            assert x0.shape[0] == n_x
        else:
            x0 = np.zeros(n_x)
        if us_guess is not None:
            us_guess = np.array(us_guess, dtype=float)
            assert us_guess.shape == (N, n_u)
        else:
            us_guess = np.zeros((N, n_u))
        # if l and lf are 1x1 Matrix, turn it into Symbol
        if isinstance(l, sym.Matrix):
            l = l[0, 0]
        if isinstance(lf, sym.Matrix):
            lf = lf[0, 0]
        # symbolic derivatives of dynamics and cost.
        if is_continuous:
            # discretize by Forward-Euler method.
            sym_dynamics = SymDynamics(x, u, t, x + f * dt)
            sym_cost = SymCost(x, u, t, l * dt, lf)
        else:
            sym_dynamics = SymDynamics(x, u, t, f) 
            sym_cost = SymCost(x, u, t, l, lf)
        # f, fx, fu, fxx, fux, fuu
        df_sym = sym_dynamics.get_derivatives()
        # l, lx, lu, lxx, lux, luu, lf, lfx, lfxx
        dl_sym = sym_cost.get_derivatives()
        # inequality constraints
        if g is not None:
            sym_ineq_constraints = SymIneqConstraints(x, u, t, g)
            dg_sym = sym_ineq_constraints.get_derivatives()
            has_ineq_constraints = True
            # hold
            self._sym_ineq_constraints = sym_ineq_constraints
            self._dg_sym = dg_sym
            self._has_ineq_constraints = has_ineq_constraints
        # equality constraints
        if h is not None:
            sym_eq_constraints = SymEqConstraints(x, u, t, h)
            dh_sym = sym_eq_constraints.get_derivatives()
            has_eq_constraints = True
            # hold
            self._sym_ineq_constraints = sym_eq_constraints
            self._dh_sym = dh_sym
            self._has_eq_constraints = has_eq_constraints
        # simplify
        if simplification:
            symutils.simplify(df_sym)
            symutils.simplify(dl_sym)
            if self._has_ineq_constraints:
                symutils.simplify(dg_sym)
            if self._has_eq_constraints:
                symutils.simplify(dh_sym)
        # hold
        self._T = T
        self._N = N
        self._dt_value = T / N
        self._t0 = t0
        self._x0 = x0
        self._us_guess = us_guess
        self._is_continuous = is_continuous
        self._sym_dynamics = sym_dynamics
        self._sym_cost = sym_cost
        self._f_original = f
        self._l_original = l
        self._lf_original = lf
        self._df_sym = df_sym
        self._dl_sym = dl_sym
        self._is_ocp_defined = True
        # generate ufunc
        self.lambdify()

    def define_unconstrained(self, f: sym.Matrix, l: sym.Symbol, lf: sym.Symbol,
               T: float, N: int, t0: float, x0: np.ndarray, us_guess: np.ndarray,
               is_continuous: bool=True, simplification: bool=False):
        """ Define optimal control problem. 

        Args:
            f (sym.Matrix): State function.
            l (sym.Symbol): Stage cost.
            lf (sym.Symbol): Terminal cost.
            T (float): Horizon length
            N (int): Discretization grid number.
            t0 (float): Initial time.
            x0 (numpy.array): Initial state. size must be n_x.
            us_guess (numpy.array, optional): Guess of input trajectory. \
                size must be (N * n_u).
            is_continuous (bool=True): Is dynamics and costs are continuous-time.\
                If true, they will be discretized. Default is False.
            simplification (bool=False): If True, functions are simplified.\
                Simplification may take time. Default is False.
        """
        self.define(f=f, l=l, lf=lf, g=None, h=None, T=T, N=N, t0=t0, x0=x0,
                    us_guess=us_guess, is_continuous=is_continuous, 
                    simplification=simplification)

    def lambdify(self) -> tuple[list, list]:
        """ Generate sympy symbolic expression into numpy function.\

        Returns:
            tuple: (df, dl) = ([f, fx, fu, fxx, fux, fuu], \
                [l, lx, lu, lxx, lux, luu, lf, lfx, lfxx])
        """
        assert self._is_ocp_defined
        x = self._x
        u = self._u
        t = self._t
        dt = self._dt
        # dynamics
        df_subs = symutils.substitute_constants_list(
            self._df_sym, self._scalar_dict, self._vector_dict, self._matrix_dict,
            ) #self._dt, self._dt_value)
        num_dynamics = NumDynamics(x, u, t, dt, *df_subs)
        df_ufunc = num_dynamics.get_derivatives()
        # cost
        dl_subs = symutils.substitute_constants_list(            
            self._dl_sym, self._scalar_dict, self._vector_dict, self._matrix_dict,
            ) #self._dt, self._dt_value)
        num_cost = NumCost(x, u, t, dt, *dl_subs)
        dl_ufunc = num_cost.get_derivatives()
        # inequality constraints
        if self._has_ineq_constraints:
            dg_subs = symutils.substitute_constants_list(
                self._dg_sym, self._scalar_dict, self._vector_dict,
                self._matrix_dict,
                )
            num_ineq_constraints = NumIneqConstraints(
                x, u, t, dt, *dg_subs
                )
            dg_ufunc = num_ineq_constraints.get_derivatives()
            self._dg_subs = dg_subs
            self._num_ineq_constrants = num_ineq_constraints
            self._dg_ufunc = dg_ufunc
        # equality constraints
        if self._has_eq_constraints:
            dh_subs = symutils.substitute_constants_list(
                self._dh_sym, self._scalar_dict, self._vector_dict,
                self._matrix_dict,
                )
            num_eq_constraints = NumEqConstraints(
                x, u, t, dt, *dh_subs
                )
            dh_ufunc = num_eq_constraints.get_derivatives()
            self._dh_subs = dh_subs
            self._num_eq_constrants = num_eq_constraints
            self._dh_ufunc = dh_ufunc
        # hold
        self._df_subs = df_subs
        self._num_dynamics = num_dynamics
        self._df_ufunc = df_ufunc
        self._dl_subs = dl_subs
        self._num_cost = num_cost
        self._dl_ufunc = dl_ufunc
        self._is_lambdified = True

    def reset_parameters(self, t0: float, T: float, N: int,
                         x0: np.ndarray, us_guess: np.ndarray):
        """ Reset parameters.

        Args:
            t0 (float): Initial time.
            T (float): Horizon length.
            N (int): Discretization grid.
            x0 (np.ndarray): Initial state
            us_guess (np.ndarray): Initial input guess.
        """
        assert self._is_ocp_defined
        self._t0 = float(t0)
        self._T = float(T)
        self._N = N
        self._dt_value = T / N
        self.reset_x0(x0)
        self.reset_us_guess(us_guess)
        # for dt_value being changed.
        self.lambdify()

    def reset_x0(self, x0: np.ndarray | list) -> np.ndarray:
        """ Reset x0. If list is given, transformed into ndarray.

        Args:
            x0 (numpy.ndarray): Initial state. size must be n_x.
        """
        if x0 is None:
            return
        x0 = np.array(x0, dtype=float)
        assert x0.shape[0] == self._n_x
        self._x0 = x0
        return x0

    def reset_us_guess(self, us_guess: np.ndarray | list) -> np.ndarray:
        """ Reset us_guess. If list is given, transformed into ndarray.

        Args:
            us (numpy.ndarray): Initial guess of input trajectory. \
                Size must be (N * n_u).
        """
        if us_guess is None:
            return
        us_guess = np.array(us_guess, dtype=float)
        assert us_guess.shape == (self._N, self._n_u)
        self._us_guess = us_guess
        return us_guess

    # symbolic
    def get_df_symbolic(self) -> list:
        """ Return symbolic derivatives of dynamics.

        Returns:
            list: [f, fx, fu, fxx, fux, fuu]
        """
        assert self._is_ocp_defined
        return self._df_sym
    
    def get_dl_symbolic(self) -> list:
        """ Return symbolic derivatives of costs.

        Returns:
            list: [l, lx, lu, lxx, lux, luu, lf, lfx, lfxx]
        """
        assert self._is_ocp_defined
        return self._dl_sym
    
    def get_dg_symbolic(self) -> list:
        """ Return symbolic derivatives of inequality constraints.

        Returns:
            list: [g, gx, gu, gxx, gux, guu]
        """
        assert self._is_ocp_defined
        return self._dg_sym

    def get_dh_symbolic(self) -> list:
        """ Return symbolic derivatives of equality constraints.

        Returns:
            list: [h, hx, hu, hxx, hux, huu]
        """
        assert self._is_ocp_defined
        return self._dg_sym
    
    # constants-substituted symbolic
    def get_df_symbolic_substituted(self) -> list:
        """ Return constants-substituted symbolic derivatives of dynamics.

        Returns:
            list: [f, fx, fu, fxx, fux, fuu]
        """
        assert self._is_lambdified
        return self._df_subs

    def get_dl_symbolic_substituted(self) -> list:
        """ Return constants-substituted symbolic derivatives of costs.

        Returns:
            list: [l, lx, lu, lxx, lux, luu, lf, lfx, lfxx]
        """
        assert self._is_lambdified
        return self._dl_subs

    def get_dg_symbolic_substituted(self) -> list:
        """ Return constants-substituted symbolic derivatives \
            of inequality constraints.

        Returns:
            list: [g, gx, gu, gxx, gux, guu]
        """
        assert self._is_lambdified
        return self._dg_subs

    def get_dh_symbolic_substituted(self) -> list:
        """ Return constants-substituted symbolic derivatives \
            equality constraints.

        Returns:
            list: [h, hx, hu, hxx, hux, huu]
        """
        assert self._is_lambdified
        return self._dh_subs

    # ufunc
    def get_df(self) -> list:
        """ Return derivatives of dynamics.

        Returns:
            list [f, fx, fu, fxx, fux, fuu]
        """
        assert self._is_lambdified
        return self._df_ufunc
    
    def get_dl(self) -> list:
        """ Return derivatives of cost.

        Returns:
            list [l, lx, lu, lxx, lux, luu, lf, lfx, lfxx]
        """
        assert self._is_lambdified
        return self._dl_ufunc

    def get_dg(self) -> list:
        """ Return derivatives of inequality constraints.

        Returns:
            list [g, gx, gu, gxx, gux, guu]
        """
        assert self._is_lambdified
        return self._dg_ufunc

    def get_dh(self) -> list:
        """ Return derivatives of equality constraints.

        Returns:
            list [h, hx, hu, hxx, hux, huu]
        """
        assert self._is_lambdified
        return self._dh_ufunc
    
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
    
    def get_f_empty(self):
        """ Return n_x*1-size symbolic zero vector.
        """
        return sym.zeros(self._n_x, 1)
    
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

    def get_us_guess(self) -> np.ndarray:
        """ Return initial guess of us.
        """
        assert self._is_ocp_defined
        return self._us_guess
    
    @staticmethod
    def zero_vector(m: int) -> sym.Matrix:
        """ symbolic m*1-size zero Matrix.

        Args:
            m (int): Dimension of vector.
        """
        return sym.zeros(m, 1)

    @staticmethod   
    def zero_matrix(m: int, n: int) -> sym.Matrix:
        """ symbolic m*n-size zero Matrix.

        Args:
            m (int): dimension of rows
            n (int): dimension of columns
        """
        return sym.zeros(m, n)

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
            vector_value (np.ndarray): 1d numpy array.
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
            vector_value (np.ndarray): 2d numpy array.

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

    @staticmethod
    def SetAllAtOnce(ocp_name: str,
            x: sym.Matrix, u: sym.Matrix, t:sym.Symbol,  dt: sym.Symbol, 
            f: sym.Matrix, l: sym.Symbol, lf: sym.Symbol,
            g: sym.Matrix, h: sym.Matrix,
            T: float, N: int, t0: float, x0: np.ndarray, us_guess: np.ndarray,
            scalar_dict: dict=None, vector_dict: dict=None,  matrix_dict: dict=None,
            is_continuous=True, simplification=False):
        """ Define optimal control problem. If symbolic constatnts are included, \
            pass them as dict{name: (symbol, value)} for substitution.

        Args:
            x (sym.Matrix): State vector.
            u (sym.Matrix): Control input vector.
            t (sym.Symbol): Time.
            dt (sym.Symbol): Time discretization step. Its value is T/N.  
            f (sym.Matrix): State function.
            l (sym.Symbol): Stage cost.
            lf (sym.Symbol): Terminal cost.
            g (sym.Matrix): Inequality constraints.
            h (sym.Matrix): Inequality constraints.
            T (float): Horizon length.
            N (int): Discretization grids.
            t0 (float): Initial time.
            x0 (numpy.array): Initial state. size must be n_x.
            us (numpy.array, optional): Guess of input trajectory. \
                size must be (N * n_u).
            scalar_dict (dict) : {"name": (symbol, value)})
            vector_dict (dict) : {"name": (symbol, value)}) 
            matrix_dict (dict) : {"name": (symbol, value)}) 
            is_continuous (bool): Is dynamics and costs are continuous-time.
                If true, they will be discretized.

        Returns:
            OCP: ocp class instance.
        """
        n_x = x.shape[0]
        n_u = u.shape[0]
        ocp = OCP(n_x, n_u, ocp_name)
        ocp._scalar_dict = scalar_dict
        ocp._vector_dict = vector_dict
        ocp._matrix_dict = matrix_dict
        ocp.define(f, l, lf, g, h, T, N, t0, x0, us_guess,
            is_continuous, simplification)
        return ocp
    
    @staticmethod
    def SampleOCPCartpole(simplification: bool=False):
        """ Return sample cartpole OCP.
        """
        from sympy import sin, cos, ln
        n_x = 4
        n_u = 1
        cartpole_ocp = OCP(n_x, n_u, 'cartpole')
        t = cartpole_ocp.get_t()
        x = cartpole_ocp.get_x()
        u = cartpole_ocp.get_u()
        # define constants
        m_c, m_p, l, g, u_min, u_max, u_eps \
            = cartpole_ocp.define_scalar_constants(
                [('m_c', 2), ('m_p', 0.1), ('l', 0.5), ('g', 9.80665), 
                ('u_min', -20),  ('u_max', 20), ('u_eps', 0.001)])
        q = cartpole_ocp.define_vector_constant('q', [2.5, 10, 0.01, 0.01])
        r = cartpole_ocp.define_vector_constant('r', [1])
        q_f = cartpole_ocp.define_vector_constant('q_f', [2.5, 10, 0.01, 0.01])
        x_ref = cartpole_ocp.define_vector_constant('x_ref', [0, np.pi, 0, 0])
        # diagonal weight    
        Q = sym.diag(*q)
        Qf = sym.diag(*q_f)
        R = sym.diag(*r)
        # state equation
        f = cartpole_ocp.zero_vector(n_x)
        f[0] = x[2]
        f[1] = x[3]
        f[2] = (u[0] + m_p*sin(x[1])*(l*x[1]*x[1] + g*cos(x[1]))) \
                /( m_c+m_p*sin(x[1])*sin(x[1]))
        f[3] = (-u[0] * cos(x[1]) - m_p*l*x[1]*x[1]*cos(x[1])*sin(x[1]) 
                - (m_c+m_p)*g*sin(x[1])) / ( l*(m_c + m_p*sin(x[1])*sin(x[1])))
        # barrier function for inequality constraints.
        u_barrier = sym.Matrix([
            sum(-ln(u[i] - u_min) - ln(u_max - u[i]) for i in range(n_u)) * 1e-5
        ])
        # cost function
        l = (x - x_ref).T * Q * (x - x_ref) + u.T * R * u + u_barrier
        lf = (x - x_ref).T * Qf * (x - x_ref)
        # horizon
        T = 5.0
        N = 200
        # initial state and solution guess
        t0 = 0.0
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        us_guess = np.zeros((N, n_u))
        cartpole_ocp.define_unconstrained(f, l, lf, T, N, t0, x0, us_guess, 
            is_continuous=True, simplification=simplification)
        return cartpole_ocp
