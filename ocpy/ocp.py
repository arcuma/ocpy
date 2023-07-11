import sympy
import numpy as np

from ocpy import symutils
from ocpy.dynamics import SymDynamics, NumDynamics
from ocpy.cost import SymCost, NumCost


class OCP:
    """ Class that describes optimal control problem.
    """
    def __init__(self, n_x: int, n_u: int):
        self._n_x = n_x
        self._n_u = n_u
        # state and input
        self._x = symutils.define_vector('x', n_x)
        self._u = symutils.define_vector('u', n_u)
        self._t = sympy.Symbol('t')
        self._dt = sympy.Symbol('dt')
        # dictionary holding symbols and values of constants
        self._scalar_dict = {}
        self._vector_dict = {}
        self._matrix_dict = {}
        self._is_ocp_defined = False
        self._is_lambdified = False

    def define(self, f: sympy.Matrix, l: sympy.Symbol, lf: sympy.Symbol,
               T: float, N: int, x0: np.ndarray, us_guess: np.ndarray,
               is_continuous: bool=True, simplification: bool=False):
        """ Define optimal control problem. 

        Args:
            f (sympy.Matrix): State function
            l (sympy.Symbol): Stage cost
            lf (sympy.Symbol): Terminal cost
            T (float): Horizon length
            N (int): Discretization grid number.
            x0 (numpy.array): Initial state. size must be n_x.
            us (numpy.array, optional): Guess of input trajectory. \
                size must be (N * n_u).
            is_continuous (bool): Is dynamics and costs are continuous-time.\
                If true, they will be discretized. Default is False.
            simplification (bool): If True, functions are simplified by simplify().\
                Simplification may take time. Default is False.
        """
        x, u, t, dt = self._x, self._u, self._t, self._dt
        n_x, n_u = self._n_x, self._n_u
        # turn x0 and us_guess into ndarray if not.
        if x0 is not None:
            if not isinstance(x0, np.ndarray):
                x0 = np.array(x0, dtype=float)
                assert x0.shape[0] == n_x
        else:
            x0 = np.array(n_x)
        if us_guess is not None:
            if not isinstance(us_guess, np.ndarray):
                us_guess = np.array(us_guess, dtype=float)
            assert us_guess.shape == (N, n_u)
        else:
            us_guess = np.zeros((N, n_u))
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
        # simplify
        if simplification:
            symutils.simplify(df_sym)
            symutils.simplify(dl_sym)
        # hold
        self._is_ocp_defined = True
        self._T = T
        self._N = N
        self._dt_value = T / N
        self._x0 = x0
        self._us_guess = us_guess
        self._is_continuous = is_continuous
        self._sym_dynamics = sym_dynamics
        self._sym_cost = sym_cost
        self._df_sym = df_sym
        self._dl_sym = dl_sym
        # generate ufunc
        self.lambdify()

    def lambdify(self) -> tuple[list, list]:
        """ Generate sympy symbolic expression into numpy function.\
        Confirm
        
        Args:

        Returns:
            tuple: (df, dl) = ([f, fx, fu, fxx, fux, fuu], \
                [l, lx, lu, lxx, lux, luu, lf, lfx, lfxx])
        """
        assert self._is_ocp_defined
        # Substitute symbolic constants.
        df_subs = symutils.substitute_constants_list(
            self._df_sym, self._scalar_dict, self._vector_dict, self._matrix_dict,
            self._dt, self._dt_value)
        dl_subs = symutils.substitute_constants_list(            
            self._dl_sym, self._scalar_dict, self._vector_dict, self._matrix_dict,
            self._dt, self._dt_value)
        # lambdify
        num_dynamics = NumDynamics(self._x, self._u, self._t, *df_subs)
        num_cost = NumCost(self._x, self._u, self._t, *dl_subs)
        # f, fx, fu, fxx, fux, fuu
        df_ufunc = num_dynamics.get_derivatives()
        # l, lx, lu, lxx, lux, luu, lf, lfx, lfxx
        dl_ufunc = num_cost.get_derivatives()
        # hold
        self._is_lambdified = True
        self._df_subs = df_subs
        self._dl_subs = dl_subs
        self._df_ufunc = df_ufunc
        self._dl_ufunc = dl_ufunc
        self._num_dynamics = num_dynamics
        self._num_cost = num_cost
        # [f, fx, fu, fxx, fux, fuu], [l, lx, lu, lxx, lux, luu, lf, lfx, lfxx]
        return df_ufunc, dl_ufunc

    def get_symbolic_derivatives(self) -> tuple[list, list]:
        """ Returns symbolic derivatives of dynamics and costs.

        Returns:
            tuple: (df, dl) = ([f, fx, fu, fxx, fux, fuu], \
                [l, lx, lu, lxx, lux, luu, lf, lfx, lfxx])
        """
        assert self._is_ocp_defined
        return self._df_sym, self._dl_sym
    
    def get_symbolic_derivatives_substituted(self) -> tuple[list, list]:
        """ Returns constants-substituted symbolic derivatives of dynamics and costs.

        Returns:
            tuple: (df, dl) = ([f, fx, fu, fxx, fux, fuu], \
                [l, lx, lu, lxx, lux, luu, lf, lfx, lfxx])
        """
        assert self._is_lambdified
        return self._df_subs, self._dl_subs

    def get_derivatives(self) -> tuple[list, list]:
        """ Returns derivatives of dynamics and costs.

        Returns:
            tuple: (df, dl) = ([f, fx, fu, fxx, fux, fuu], \
                [l, lx, lu, lxx, lux, luu, lf, lfx, lfxx])
        """
        assert self._is_lambdified
        return self._df_ufunc, self._dl_ufunc

    def reset_x0(self, x0: np.ndarray | list) -> np.ndarray:
        """ reset x0. If list is given, transformed into ndarray.

        Args:
            x0 (numpy.ndarray): Initial state. size must be n_x.
        """
        if x0 is not None:
            if not isinstance(x0, np.ndarray):
                x0 = np.array(x0, dtype=float)
                assert x0.shape[0] == self._n_x
        else:
            x0 = np.array(self._n_x)
        self._x0 = x0
        return x0

    def reset_us_guess(self, us_guess: np.ndarray | list) -> np.ndarray:
        """ reset us_guess. If list is given, transformed into ndarray.

        Args:
            us (numpy.ndarray): Guess of input trajectory. \
                Size must be (N * n_u).
        """
        if us_guess is not None:
            if not isinstance(us_guess, np.ndarray):
                us_guess = np.array(us_guess, dtype=float)
            assert us_guess.shape == (self._N, self._n_u)
        else:
            us_guess = np.zeros((self._N, self._n_u))
        self._us_guess = us_guess
        return us_guess

    @staticmethod
    def SetAllAtOnce(
            x: sympy.Matrix, u: sympy.Matrix, t:sympy.Symbol,  dt: sympy.Symbol, 
            f: sympy.Matrix, l: sympy.Symbol, lf: sympy.Symbol,
            T: float, N: int, x0: np.ndarray, us_guess: np.ndarray,
            scalar_dict: dict=None, vector_dict: dict=None,  matrix_dict: dict=None,
            is_continuous=True):
        """ Define optimal control problem. If symbolic constatnts are included, \
            pass them as dict{name: (symbol, value)} for substitution.

        Args:
            x (sympy.Matrix): State vector.
            u (sympy.Matrix): Control input vector.
            t (sympy.Symbol): Time.
            dt (sympy.Symbol): Time discretization step. Its value is T/N.  
            f (sympy.Matrix): State function
            l (sympy.Symbol): Stage cost
            lf (sympy.Symbol): Terminal cost
            T (float): Horizon length
            N (int): Discretization grid number.
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
        ocp = OCP(n_x, n_u)
        ocp._scalar_dict = scalar_dict
        ocp._vector_dict = vector_dict
        ocp._matrix_dict = matrix_dict
        ocp.define(f, l, lf, T, N, x0, us_guess, is_continuous)
        return ocp
    
    @staticmethod
    def SampleModelCartpole(simplification: bool=False):
        """ Returns sample cartpole OCP.
        """
        from sympy import sin, cos
        n_x = 4
        n_u = 1
        cartpole_ocp = OCP(n_x, n_u)
        t = cartpole_ocp.get_t()
        x = cartpole_ocp.get_x()
        u = cartpole_ocp.get_u()
        # define constants
        m_c, m_p, l, g, u__min, u_max, u_eps \
            = cartpole_ocp.define_scalar_constants(
                [('m_c', 2), ('m_p', 0.1), ('l', 0.5), ('g', 9.80665), 
                ('u_min', -20),  ('u_max', 20), ('u_eps', 0.001)])
        q = cartpole_ocp.define_vector_constant('q', [2.5, 10, 0.01, 0.01])
        r = cartpole_ocp.define_vector_constant('r', [1])
        q_f = cartpole_ocp.define_vector_constant('q_f', [2.5, 10, 0.01, 0.01])
        x_ref = cartpole_ocp.define_vector_constant('x_ref', [0, np.pi, 0, 0])
        # diagonal weight    
        Q = sympy.diag(*q)
        Qf = sympy.diag(*q_f)
        R = sympy.diag(*r)
        # state equation
        f = cartpole_ocp.get_f_empty()
        f[0] = x[2]
        f[1] = x[3]
        f[2] = (u[0] + m_p*sin(x[1])*(l*x[1]*x[1] + g*cos(x[1]))) \
                /( m_c+m_p*sin(x[1])*sin(x[1]))
        f[3] = (-u[0] * cos(x[1]) - m_p*l*x[1]*x[1]*cos(x[1])*sin(x[1]) 
                - (m_c+m_p)*g*sin(x[1])) / ( l*(m_c + m_p*sin(x[1])*sin(x[1])))
        # cost function
        l = (x - x_ref).T * Q * (x - x_ref) + u.T * R * u
        lf = (x - x_ref).T * Qf * (x - x_ref)
        # horizon
        T = 5.0
        N = 200
        # initial state and solution guess
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        us_guess = np.zeros((N, n_u))
        cartpole_ocp.define(f, l, lf, T, N, x0, us_guess, is_continuous=True)
        return cartpole_ocp

    def get_x(self) -> sympy.Matrix:
        """ Returns sympy expression of x.
        """
        return self._x.copy()

    def get_u(self) -> sympy.Matrix:
        """ Returns sympy expression of u.
        """
        return self._u.copy()
    
    def get_t(self) -> sympy.Symbol:
        """ Returns sympy symbol of t.
        """
        return self._t
    
    def get_dt(self) -> sympy.Symbol:
        """ Returns sympy symbol of dt.
        """
        return self._dt

    def get_n_x(self) -> int:
        """ Returns dimension of x.
        """
        return self._n_x

    def get_n_u(self) -> int:
        """ Returns dimension of u.
        """
        return self._n_u
    
    def get_f_empty(self):
        """ Returns n_x*1-size symbolic zero vector.
        """
        return sympy.zeros(self._n_x, 1)
    
    def get_T(self) -> float:
        """ Returns horizon length.
        """
        assert self._is_ocp_defined
        return self._T

    def get_N(self) -> int:
        """ Returns number of discretization grids.
        """
        assert self._is_ocp_defined
        return self._N

    def get_dt_value(self) -> float:
        """ Returns number of discretization grids.
        """
        assert self._is_ocp_defined
        return self._dt_value
    
    def get_function_template(self) -> tuple:
        """ Returns symbolic explessions of state function, \
            stage cost and terminal cost. The size of f equals to n_x.

        Returns:
            (Matrix, Symbol, Symbol)
        """
        f = symutils.define_vector('f', self._n_x)
        l = sympy.Symbol('l')
        lf = sympy.Symbol('lf')
        return f, l, lf

    def define_scalar_constant(self, constant_name: str, value: float):
        """
        Args:
            constant_name (str): 
            value (float):

        Returns:
            scalar_symbol (sympy.Symbol) : sympy symbol of constant.
        """
        scalar_symbol = sympy.Symbol(constant_name)
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
            constant_name (str): 
            vector_value (np.ndarray): 1d numpy array.
        Returns:
            vector_symbol (sympy.Matrix) : n*1-size sympy.Matrix
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
            vector_symbols (list) : list[sympy.Matrix]. list of vector symbols.
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
            constant_name (str): 
            vector_value (np.ndarray): 2d numpy array.

        Returns:
            matrix_symbol (sympy.Matrix) : m*n-size sympy.Matrix
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
            matrix_symbols (list) : list[sympy.Matrix]. list of Matrix symbols.
        """
        matrix_symbols = []
        for name, value in constant_list:
            matrix_symbol = self.define_matrix_constant(name, value)
            matrix_symbols.append(matrix_symbol)
        return matrix_symbols   
