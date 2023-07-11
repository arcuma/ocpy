import sympy
import numpy as np
import time
from os.path import abspath, join, dirname

from ocpy import symutils
from ocpy.logger import Logger
from ocpy.plotter import Plotter



class DDP:
    """ Class of solving OCP by Differential Dynamic Programming.
    """
    def __init__(self, n_x, n_u, sim_name):
        """ Constructor.

        Args:
            n_x (int): Dimention of state.
            n_u (int): Dimention of control input.
            sim_name (str): simulation name. it is used to make log directory.
        """
        assert type(n_x) == int and n_x > 0
        assert type(n_u) == int and n_u > 0
        self._n_x = n_x
        self._n_u = n_u
        self._sim_name = sim_name
        self._t_sym = sympy.Symbol('t')
        self._x_sym = sympy.Matrix(sympy.symbols(f'x[0:{self._n_x}]'))
        self._u_sym = sympy.Matrix(sympy.symbols(f'u[0:{self._n_u}]'))
        self._dt = sympy.Symbol('dt')
        self._dt_val = None
        self._T = None
        self._N = None
        self._us_guess = None
        self._x0 = np.zeros(self._n_x)
        # symolic expressions 
        self._f_sym = sympy.Matrix(np.zeros(self._n_x)) # continuous state_eq
        self._F_sym = sympy.Matrix(np.zeros(self._n_x)) # discrete state_eq
        self._Fx_sym = None
        self._Fu_sym = None
        self._Fxx_sym = None
        self._Fux_sym = None
        self._Fuu_sym = None
        self._l_sym = None
        self._L_sym = None
        self._Lx_sym = None
        self._Lu_sym = None
        self._Lxx_sym = None
        self._Lux_sym = None
        self._Luu_sym = None
        self._lf_sym = None
        self._lfx_sym = None
        self._lfxx_sym = None
        self._V_sym = None
        self._Vx_sym = None
        self._Vxx_sym = None
        self._Q_sym = None
        self._Qx_sym = None
        self._Qu_sym = None
        self._Qxx_sym = None
        self._Qux_sym = None
        self._Quu_sym = None
        # constant values
        self._scalar_dict = {}
        self._vector_dict = {}
        self._matrix_dict = {}
        # np.ufunc
        self._f_ufunc = None
        self._F_ufunc = None
        self._Fx_ufunc = None
        self._Fu_ufunc = None
        self._Fxx_ufunc = None
        self._Fux_ufunc = None
        self._Fuu_ufunc = None
        self._l_ufunc = None
        self._L_ufunc = None
        self._Lx_ufunc = None
        self._Lu_ufunc = None
        self._Lxx_ufunc = None
        self._Lux_ufunc = None
        self._Luu_ufunc = None
        self._lf_ufunc = None
        self._lfx_ufunc = None
        self._lfxx_ufunc = None
        self._V_ufunc = None
        self._Vx_ufunc = None
        self._Vxx_ufunc = None
        self._Q_ufunc = None
        self._Qx_ufunc = None
        self._Qu_ufunc = None
        self._Qxx_ufunc = None
        self._Qux_ufunc = None
        self._Quu_ufunc = None
        # trajectory
        self._xs = None
        self._us = None 
        # feedforward and feedback gain
        self._ks = None
        self._Ks = None
        # flags
        self._max_iter = 100
        self._is_function_set = False
        self.debug = True
        self.simplification = True
        # log directory
        self._log_dir = join(dirname(dirname(abspath(__file__))), 'log',
                sim_name)

    def log_directory(self):
        return self._log_dir

    def define_t(self):
        """ returns time symbol t.
            Args: None
            Returns:
                t (sympy.Symbol): time variable
        """
        return sympy.symbols('t')
    
    def define_x(self):
        """ returns state variable x.
            Args: None
            Returns:
                x (n_x * 1 sympy.Matrix): state variable
        """
        return sympy.Matrix(sympy.symbols(f'x[0:{self._n_x}]'))
    
    def define_u(self):
        """ returns input variable u.
            Args: None
            Returns:
                u (n_u * 1 sympy.Matrix): input variable
        """
        return sympy.Matrix(sympy.symbols(f'u[0:{self._n_u}]'))
    
    def get_t(self) -> sympy.Symbol:
        return self._t_sym

    def get_x(self) -> sympy.Matrix:
        return self._x_sym.copy()
    
    def get_u(self) -> sympy.Matrix:
        return self._u_sym.copy()
    
    def get_f(self) -> sympy.Matrix:
        return self._f_sym.copy()

    def f_eval(self, x_val, u_val):
        assert type(x_val) == np.ndarray
        assert type(u_val) == np.ndarray
        assert len(x_val) == self._n_x
        assert len(u_val) == self._n_u

        f_ufunc = sympy.lambdify((self._x, self._u), self._f, "numpy")
    

    ### set const values 
    def define_scalar_variable(self, variable_name: str, value: float):
        """
        Args:
            variable_name (str): 
            value: (float): 
        Returns:
            scalar_symbol (sympy.Symbol) : sympy symbol of variable.
        """
        scalar_symbol = sympy.Symbol(variable_name)
        self._scalar_dict[variable_name] = (scalar_symbol, value)
        return scalar_symbol

    def define_scalar_variables(self, variable_list: list):
        """
        Args:
            variable_list (list): list[(name, value)].
        Returns:
            scalar_symbols (list) : list[Symbol]. sympy symbol of variable.
        """
        scalar_symbols = []
        for variable in variable_list:
            scalar_symbol = self.define_scalar_variable(variable[0], variable[1])
            scalar_symbols.append(scalar_symbol)
        return scalar_symbols


    def define_vector_variable(self, variable_name: str, 
                               vector_value: np.ndarray | list):
        """
        Args:
            variable_name (str): 
            vector_value (np.array): 1d numpy array.
        Returns:
            vector_symbol (sympy.Matrix) : n*1 sympy.Matrix
        """
        if type(vector_value) == list:
            vector_value = np.array(vector_value)
        n = len(vector_value)
        vec_symbol = sympy.Matrix(sympy.symbols(f'{variable_name}[0:{n}]'))
        self._vector_dict[variable_name] = (vec_symbol, vector_value)
        return vec_symbol
    
    def define_vector_variables(self, variable_list: list):
        """
        Args:
            variable_list (list): list[(name, value)].
        Returns:
            vector_symbols (list) : list[sympy.Matrix]. list of vector symbols.
        """
        vec_symbols = []
        for variable in variable_list:
            vec_symbol = self.define_vector_variable(variable[0], variable[1])
            vec_symbols.append(vec_symbol)
        return vec_symbols
    
    def define_matrix_variable(self, variable_name: str, matrix_value: np.ndarray):
        """
        Args:
            variable_name (str): 
            vector_value (np.array): 2d numpy array.
        Returns:
            matrix_symbol (sympy.Matrix) : n*1 sympy.Matrix
        """
        matrix_symbol = symutils.define_matrix(variable_name, *(matrix_value.shape))
        self._matrix_dict[variable_name] = (matrix_symbol, matrix_value)
        return matrix_symbol.copy()

    def define_matrix_variables(self, variable_list: list):
        """
        Args:
            variable_list (list): list[(name, value)].
        Returns:
            vector_symbols (list) : list[sympy.Matrix]. list of Matrix symbols.
        """
        matrix_symbols = []
        for name, value in variable_list:
            matrix_symbol = self.define_matrix_variable(name, value)
            matrix_symbols.append(matrix_symbol)
        return matrix_symbols

    def set_state_equation(self, f):
        self._f_sym = f

    def set_stage_cost(self, l):
        self._l_sym = l

    def set_terminal_cost(self, lf):
        self._lf_sym = lf

    def set_functions(self, f, l, lf):
        self.set_state_equation(f)
        self.set_stage_cost(l)
        self.set_terminal_cost(lf)

    def set_OCP_horizon(self, T, N):
        self._T = float(T)
        self._N = N
        self._dt_val = T / N
        self.define_scalar_variable('dt', self._dt_val)


    def defineOCP(self, f, l, lf, T, N, x0, us_guess=None):
        """ Define optimal control problem. 

        Args:
            f (sympy.Matrix): state function
            l (sympy.Symbol): stage cost
            lf (sympy.Symbol): terminal cost
            T (float): horizon length
            N (int): discretization grid
            x0 (numpy.array): initial state. size must be n_x.
            us (numpy.array): guess of input trajectory. size must be (N * n_u).
        """
        if x0 is not None:
            if not isinstance(x0, np.ndarray):
                x0 = np.array(x0, dtype=float)
                assert x0.shape[0] == self._n_x
        else:
            x0 = np.array(self._n_x)
        if us_guess is not None:
            if not isinstance(us_guess, np.ndarray):
                us_guess = np.array(us_guess, dtype=float)
            assert us_guess.shape == (N, self._n_u)
        else:
            us_guess = np.zeros((N, self._n_u))

        self.set_OCP_horizon(T, N)
        # variables
        x = self._x_sym
        u = self._u_sym
        dt = self._dt
        # derivatives of dynamics
        F = x + f * dt  # forward euler
        Fx = symutils.diff_vector(F, x)
        Fu = symutils.diff_vector(F, u)
        Fxx = symutils.diff_matrix(Fx, x)
        Fux = symutils.diff_matrix(Fu, x)
        Fuu = symutils.diff_matrix(Fu, u)
        # derivatives of cost function
        if isinstance(l, sympy.Matrix):
            l = l[0,0]
        L = l * dt # discrete stage cost
        Lx = symutils.diff_scalar(L, x)
        Lu = symutils.diff_scalar(L, u)
        Lxx = symutils.diff_vector(Lx, x)
        Lux = symutils.diff_vector(Lu, x)
        Luu = symutils.diff_vector(Lu, u)
        if isinstance(lf, sympy.Matrix):
            lf = lf[0, 0]
        lfx = symutils.diff_scalar(lf, x)
        lfxx = symutils.diff_vector(lfx, x)
        # value function (of stage k+1)
        V = sympy.Symbol('V')
        Vx = symutils.define_vector('V_{x}', self._n_x)
        Vxx = symutils.define_matrix('V_{xx}', self._n_x, self._n_x)
        # derivative of action value function
        Q = sympy.Symbol('Q')
        Qx = Lx + Fx.T * Vx
        Qu = Lu + Fu.T * Vx
        Qxx = Lxx + Fx.T * Vxx * Fx + symutils.diff_vector(Vx.T * Fx, x)
        Qux = Lux + Fu.T * Vxx * Fx + symutils.diff_vector(Vx.T * Fu, x)
        Quu = Luu + Fu.T * Vxx * Fu + symutils.diff_vector(Vx.T * Fu, u)
        # set initial state and solution guess
        self._x0 = x0
        self._us_guess = us_guess
        # set symbolic expressions
        self._f_sym = f
        self._F_sym = F
        self._Fx_sym = Fx
        self._Fu_sym = Fu
        self._Fxx_sym = Fxx
        self._Fux_sym = Fux
        self._Fuu_sym = Fuu
        self._l_sym = l
        self._L_sym = L
        self._Lx_sym = Lx
        self._Lu_sym = Lu
        self._Lxx_sym = Lxx
        self._Lux_sym = Lux
        self._Luu_sym = Luu
        self._lf_sym = lf
        self._lfx_sym = lfx
        self._lfxx_sym = lfxx
        self._V_sym = V
        self._Vx_sym = Vx
        self._Vxx_sym = Vxx
        self._Q_sym = Q
        self._Qx_sym = Qx
        self._Qu_sym = Qu
        self._Qxx_sym = Qxx
        self._Qux_sym = Qux
        self._Quu_sym = Quu
        # functions were set
        self._is_function_set = True
        self.init_solver()


    def init_solver(self):
        """ Preprocess state functions and cost functions (and its derivatives).\
            Substitute constant variables and creating ufuncs.
        """
        print('initializing solver...')
        # substitute constatnts 
        f_subs     = symutils.substitute_constants(self._f_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        F_subs     = symutils.substitute_constants(self._F_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Fx_subs    = symutils.substitute_constants(self._Fx_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Fu_subs    = symutils.substitute_constants(self._Fu_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Fxx_subs   = symutils.substitute_constants(self._Fxx_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Fux_subs   = symutils.substitute_constants(self._Fux_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Fuu_subs   = symutils.substitute_constants(self._Fuu_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        l_subs     = symutils.substitute_constants(self._l_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        L_subs     = symutils.substitute_constants(self._L_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Lx_subs    = symutils.substitute_constants(self._Lx_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Lu_subs    = symutils.substitute_constants(self._Lu_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Lxx_subs   = symutils.substitute_constants(self._Lxx_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Lux_subs   = symutils.substitute_constants(self._Lux_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Luu_subs   = symutils.substitute_constants(self._Luu_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        lf_subs    = symutils.substitute_constants(self._lf_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        lfx_subs   = symutils.substitute_constants(self._lfx_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        lfxx_subs  = symutils.substitute_constants(self._lfxx_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        V_subs     = symutils.substitute_constants(self._V_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Vx_subs    = symutils.substitute_constants(self._Vx_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Vxx_subs   = symutils.substitute_constants(self._Vxx_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Q_subs     = symutils.substitute_constants(self._Q_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Qx_subs    = symutils.substitute_constants(self._Qx_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Qu_subs    = symutils.substitute_constants(self._Qu_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Qxx_subs   = symutils.substitute_constants(self._Qxx_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Qux_subs   = symutils.substitute_constants(self._Qux_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        Quu_subs   = symutils.substitute_constants(self._Quu_sym,
            self._scalar_dict, self._vector_dict, self._matrix_dict)
        # create ufunc
        x = self._x_sym
        u = self._u_sym
        V = self._V_sym
        Vx = self._Vx_sym
        Vxx = self._Vxx_sym
        self._f_ufunc    = symutils.lambdify([x, u], f_subs)
        self._F_ufunc    = symutils.lambdify([x, u], F_subs) #
        self._Fx_ufunc   = symutils.lambdify([x, u], Fx_subs)
        self._Fu_ufunc   = symutils.lambdify([x, u], Fu_subs)
        self._Fxx_ufunc  = symutils.lambdify([x, u], Fxx_subs)
        self._Fux_ufunc  = symutils.lambdify([x, u], Fux_subs)
        self._Fuu_ufunc  = symutils.lambdify([x, u], Fuu_subs)
        self._l_ufunc    = symutils.lambdify([x, u], l_subs)
        self._L_ufunc    = symutils.lambdify([x, u], L_subs)
        self._Lx_ufunc   = symutils.lambdify([x, u], Lx_subs)
        self._Lu_ufunc   = symutils.lambdify([x, u], Lu_subs)
        self._Lxx_ufunc  = symutils.lambdify([x, u], Lxx_subs)
        self._Lux_ufunc  = symutils.lambdify([x, u], Lux_subs)
        self._Luu_ufunc  = symutils.lambdify([x, u], Luu_subs)
        self._lf_ufunc   = symutils.lambdify([x], lf_subs)
        self._lfx_ufunc  = symutils.lambdify([x], lfx_subs) #
        self._lfxx_ufunc = symutils.lambdify([x], lfxx_subs) #
        self._V_ufunc    = symutils.lambdify([V], V_subs)
        self._Vx_ufunc   = symutils.lambdify([Vx], Vx_subs)
        self._Vxx_ufunc  = symutils.lambdify([Vxx], Vxx_subs)
        self._Q_ufunc    = symutils.lambdify([x, u, V], Q_subs)
        self._Qx_ufunc   = symutils.lambdify([x, u, Vx], Qx_subs) #
        self._Qu_ufunc   = symutils.lambdify([x, u, Vx], Qu_subs) #
        self._Qxx_ufunc  = symutils.lambdify([x, u, Vx, Vxx], Qxx_subs) #
        self._Qux_ufunc  = symutils.lambdify([x, u, Vx, Vxx], Qux_subs) #
        self._Quu_ufunc  = symutils.lambdify([x, u, Vx, Vxx], Quu_subs) #
        # rollout with initial guess
        F = self._F_ufunc
        N = self._N
        n_x = self._n_x
        n_u = self._n_u
        # xs = np.zeros((N + 1, n_x))
        # us = self._us_guess
        # xs[0] = self._x0
        # for i in range(self._N):
        #     xs[i + 1] = F(xs[i], us[i])
        # # allocate memory        
        # self._xs = xs
        # self._us = us
        # self._Qx = np.zeros((N + 1, n_x))
        # self._Qu = np.zeros((N + 1, n_u))
        # self._Qxx = np.zeros((N + 1, n_x, n_x))
        # self._Qux = np.zeros((N + 1, n_u, n_x))
        # self._Quu = np.zeros((N + 1, n_u, n_u))
        # self._Vx = np.zeros((N + 1, n_x))
        # self._Vxx = np.zeros((N + 1, n_x, n_x))
        # self._Delta_V = np.zeros(N + 1)
        # self._k = np.zeros((N, n_u))
        # self._K = np.zeros((N, n_u, n_x))
        print('initialization done.')


    def solve(self, x0=None, us_guess=None, max_iter=10,
              alphas=0.5**np.arange(5), damp_init=1e-3,
              damp_min=1e-3, damp_max=1e3, log=False):
        """ Solving OCP via DDP iteration.

        Args:
            x0 (numpy.array): initial state. size must be n_x.
            us_guess (numpy.array): guess of input trajectory. \
                size must be (N * n_u).
            max_iter (int): maximum iteration.
            alphas (np.array): step sizes of line search.
            damp_init (float): initial damping value for backward pass
            damp_min (float): minimum dapming value.
            damp_max (float): maximum damping value.
            log (bool): If true, results are logged to log_dir.
        
        Returns:
            xs (numpy.ndarray): optimal state trajectory. (N * n_x)
            us (numpy.ndarray): optimal control trajectory. (N * n_u)

        """
        assert self._is_function_set
        N, n_x, n_u = self._N, self._n_x, self._n_u
        if us_guess is not None:
            us = np.array(us_guess)
            assert us.shape == (N, n_u)
        else:
            us = self._us_guess
        if x0 is not None:
            x0 = np.array(x0)
            assert x0.shape[0] == n_x
        else:
            x0 = self._x0

        t_start = time.perf_counter()
        # initial rollout along with x0 and us
        xs, J = self.rollout(x0, us)
        J_hist = [J]
        # damping coefficient of LM
        damp = damp_init
        # step size of line search
        alphas = np.array(0.5**i for i in range(5))
        # flag
        is_success = False
        # DDP iteration
        for iter in range(max_iter):
            print(f'iter: {iter}')
            ks, Ks, Delta_V0 = self.backward_recursion(xs, us, damp)
            if np.abs(Delta_V0) < 1e-5:
                is_success = True
                break
            print(Delta_V0)    
                # break
            alphas = np.array([0.5 ** i for i in range(5)] + [0])
            
            for alpha in alphas:
                xs_new, us_new, J_new = self.forward_recursion(xs, us, ks, Ks, alpha)
                # print(f'iter: {iter}, alpha: {alpha}, J: {J}, J_new: {J_new}')
                if J_new < J:
                    # line search successed
                    xs = xs_new
                    us = us_new
                    J = J_new
                    damp *= 0.5
                    break
            else:
                # failed in line search
                damp *= 2.0
                damp = min(max(damp, damp_min), damp_max)
            J_hist.append(J)

        # time grids
        ts = np.array([i*self._dt_val for i in range(N + 1)])
        # transform J_hist from list into np.ndarray (for np.append() is slow)
        J_hist = np.array(J_hist)
        # computational time
        t_end = time.perf_counter()
        t_elapsed = t_end - t_start
        # results
        print('---')
        if is_success:
            status = 'success'
        else:
            status = 'fail'
        print(f'status: {status}')
        print(f'iteration: {iter}')
        print(f'cost value: {J_hist[-1]}')
        print(f'computational time: {t_elapsed}')
        print('---')
        # log
        if log:
            self._log_dir = join(dirname(dirname(abspath(__file__))), 'log',
                           self._sim_name)
            logger = Logger(self._log_dir)
            logger.save(ts, xs, us, J_hist)
            # plot
            plotter = Plotter.from_log(self._log_dir, self._sim_name)
            plotter.plot(save=True)
        return ts, xs, us, J_hist
        

    def rollout(self, x0: np.ndarray, us: np.ndarray):
        """ rollout state trajectory.
        Args:
            x0 (numpy.ndarray): initial state. size must be n_x
            us (numpy.ndarray): control trajectory. size must be (N*n_u)
        Returns:
            xs (numpy.ndarray): rollled-out state trajectory.
            J (float): cost value along with (xs, us)
        """ 
        N, n_x, n_u = self._N, self._n_x, self._n_u
        F = self._F_ufunc
        L = self._L_ufunc
        lf = self._lf_ufunc
        xs = np.empty((N + 1, n_x))
        xs[0] = x0
        J = 0
        for i in range(N):
            xs[i + 1] = F(xs[i], us[i])
            J += L(xs[i], us[i])
        J += lf(xs[N])
        return xs, J
    

    def backward_recursion(self, xs, us, damp=1e-6):
        """ backward pass of DDP.
        Args:
            xs (numpy.ndarray): nominal state trajectory.\
                size must be (N+1)*n_u
            us (numpy.ndarray): nominalcontrol trajectory.\
                size must be N*n_u
            damp (float): damping coefficient of LM method.
        Returns:
            ks (numpy.ndarray): series of k. Its size is N * n_u
            Ks (numpy.ndarray): series of K. Its size is N * (n_u * n_x)
            Delta_V0 (float): expecting change of value function at stage 0.
        """ 
        N = self._N
        n_x = self._n_x
        n_u = self._n_u
        # ufunction
        Fx_ufunc = self._Fx_ufunc
        Fu_ufunc = self._Fu_ufunc
        Fxx_ufunc = self._Fxx_ufunc
        Fux_ufunc = self._Fux_ufunc
        Fuu_ufunc = self._Fuu_ufunc
        Lx_ufunc = self._Lx_ufunc
        Lu_ufunc = self._Lu_ufunc
        Lxx_ufunc = self._Lxx_ufunc
        Lux_ufunc = self._Lux_ufunc
        Luu_ufunc = self._Luu_ufunc
        lfx_ufunc = self._lfx_ufunc
        lfxx_ufunc = self._lfxx_ufunc
        # feedforward and feedback gain
        ks = np.empty((N, n_u))
        Ks = np.empty((N, n_u, n_x))
        # derivatives of value function of stage i + 1
        Vx = lfx_ufunc(xs[N])
        Vxx = lfxx_ufunc(xs[N])
        Delta_V0 = 0.0
        for i in range(N - 1, -1, -1):
            Fx = Fx_ufunc(xs[i], us[i])
            Fu = Fu_ufunc(xs[i], us[i])
            Fxx = Fxx_ufunc(xs[i], us[i])
            Fux = Fux_ufunc(xs[i], us[i])
            Fuu = Fuu_ufunc(xs[i], us[i])
            Lx = Lx_ufunc(xs[i], us[i])
            Lu = Lu_ufunc(xs[i], us[i])
            Lxx = Lxx_ufunc(xs[i], us[i])
            Lux = Lux_ufunc(xs[i], us[i])
            Luu = Luu_ufunc(xs[i], us[i])
            # action value function
            Qx = Lx + Fx.T @ Vx
            Qu = Lu + Fu.T @ Vx
            Qxx = Lxx + Fx.T @ Vxx @ Fx + Vx @ Fxx.T
            Qux = Lux + Fu.T @ Vxx @ Fx + Vx @ Fux.T
            Quu = Luu + Fu.T @ Vxx @ Fu + Vx @ Fuu.T
            # state feedforward/feedback system
            Quu_inv = np.linalg.inv(Quu + damp * np.eye(n_u))
            k = -Quu_inv @ Qu
            K = -Quu_inv @ Qux 
            ks[i] = k
            Ks[i] = K
            # derivatives of value function
            Vx = Qx - K.T @ Quu @ k
            Vxx = Qxx - K.T @ Quu @ K
            Delta_V0 += 0.5 * k.T @ Quu @ k + k.T @ Qu
        # save
        self._ks = ks
        self._Ks = Ks
        # print(Delta_V0)
        return ks, Ks, Delta_V0


    def forward_recursion(self, xs, us, ks, Ks, alpha=1.0):
        """ forward pass of DDP.
        Args:
            xs (numpy.ndarray): nominal state trajectory.\
                size must be (N+1)*n_u
            us (numpy.ndarray): nominal control trajectory.\
                size must be N*n_u
            ks (numpy.ndarray): series of k. Size must be N * n_u
            Ks (numpy.ndarray): series of K. Size must be N * (n_u * n_x)
            alpha (float): step size of line search. 0<= alpha <= 1.0
        Returns:
            xs_new (numpy.ndarray): new state trajectory.
            us_new (numpy.ndarray): new control trajectory.
            J_new (float): entire cost along with (xs_new, us_new)
        """
        N = self._N
        F = self._F_ufunc
        L = self._L_ufunc
        lf = self._lf_ufunc
        xs_new = np.empty(xs.shape)
        xs_new[0] = xs[0]
        us_new = us.copy()
        J0_new = 0.0
        for i in range(N):
            us_new[i] += alpha * ks[i] + Ks[i] @ (xs_new[i] - xs[i])
            xs_new[i + 1] = F(xs_new[i], us_new[i])
            J0_new += L(xs_new[i], us_new[i])
        J0_new += lf(xs_new[N])
        return xs_new, us_new, J0_new
