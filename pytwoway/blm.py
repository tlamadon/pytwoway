'''
We implement the non-linear estimator from Bonhomme Lamadon & Manresa
'''

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.sparse import csc_matrix, diags
from scipy.stats import norm
from qpsolvers import solve_qp
from matplotlib import pyplot as plt
from multiprocessing import Pool
import itertools
import warnings
from pytwoway import jitter_scatter
from bipartitepandas.util import ParamsDict, to_list, _is_subtype
from tqdm import tqdm

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq1(a):
    return a >= 1
def _gteq0(a):
    return a >= 0
def _gt0(a):
    return a > 0
def _min_gt0(a):
    return np.min(a) > 0
def _lstdct(a):
    return (isinstance(a[0], list) and isinstance(a[1], ParamsDict))
def _dctlstdct(a):
    return all([_lstdct(sub_a) for sub_a in a.values()])

# Define default parameter dictionaries
_blm_params_default = ParamsDict({
    ## Class parameters ##
    'nl': (6, 'type_constrained', (int, _gteq1),
        '''
            (default=6) Number of worker types.
        ''', '>= 1'),
    'nk': (10, 'type_constrained', (int, _gteq1),
        '''
            (default=10) Number of firm types.
        ''', '>= 1'),
    'categorical_controls': (None, 'dict_of_type_none', ParamsDict,
        '''
            (default=None) Dictionary linking column names to instances of tw.categorical_control_params(). Each instance specifies a new categorical control variable and how its starting values should be generated. Run tw.categorical_control_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
        ''', None),
    'continuous_controls': (None, 'dict_of_type_none', ParamsDict,
        '''
            (default=None) Dictionary linking column names to instances of tw.continuous_control_params(). Each instance specifies a new continuous control variable and how its starting values should be generated. Run tw.continuous_control_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
        ''', None),
    ## Starting values ##
    'a1_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A1 (mean of fixed effects in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A1 (mean of fixed effects in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A2 (mean of fixed effects in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2 (mean of fixed effects in second period).
        ''', '>= 0'),
    's1_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S1 (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's1_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S1 (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's2_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2 (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    's2_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S2 (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    'pk1_prior': (None, 'array_of_type_constrained_none', (('float', 'int'), _min_gt0),
        '''
            (default=None) Dirichlet prior for pk1 (probability of being at each combination of firm types for movers). Must have length nl. None is equivalent to np.ones(nl).
        ''', 'min > 0'),
    'pk0_prior': (None, 'array_of_type_constrained_none', (('float', 'int'), _min_gt0),
        '''
            (default=None) Dirichlet prior for pk0 (probability of being at each firm type for stayers). Must have length nl. None is equivalent to np.ones(nl).
        ''', 'min > 0'),
    'verbose': (0, 'set', [0, 1, 2],
        '''
            (default=0) If 0, print no output; if 1, print additional output; if 2, print maximum output.
        ''', None),
    ## fit_movers() and fit_stayers() parameters ##
    'return_qi': (False, 'type', bool,
        '''
            (default=False) If True, return qi matrix after first loop.
        ''', None),
    # fit_movers() parameters ##
    'n_iters_movers': (100, 'type_constrained', (int, _gteq1),
        '''
            (default=100) Number of iterations for EM for movers.
        ''', '>= 1'),
    'threshold_movers': (1e-7, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1e-7) Threshold to break EM loop for movers.
        ''', '>= 0'),
    'update_a': (True, 'type', bool,
        '''
            (default=True) If False, do not update A1 or A2.
        ''', None),
    'update_s': (True, 'type', bool,
        '''
            (default=True) If False, do not update S1 or S2.
        ''', None),
    'update_pk1': (True, 'type', bool,
        '''
            (default=True) If False, do not update pk1.
        ''', None),
    'fixb': (False, 'type', bool,
        '''
            (default=False) If True, set constraints for A1/A2/A1_cat/A2_cat/A1_cts/A2_cts so that A2 = np.mean(A2, axis=1) + A1 - np.mean(A1, axis=1).
        ''', None),
    'linear': (False, 'type', bool,
        '''
            (default=False) If True, set constraints for A1/A2/A1_cat/A2_cat/A1_cts/A2_cts so that for a given firm type, the change in effect between worker types is linear.
        ''', None),
    'stationary_A': (False, 'type', bool,
        '''
            (default=False) If True, set constraints so that A1 = A2, A1_cat = A2_cat, and A1_cts = A2_cts.
        ''', None),
    'stationary_S': (False, 'type', bool,
        '''
            (default=False) If True, set constraints so that S1 = S2, S1_cat = S2_cat, and S1_cts = S2_cts.
        ''', None),
    'cons_a': (None, 'type_constrained_none', (tuple, _lstdct),
        '''
            (default=None) Constraints on A1 and A2, where first entry gives list of string constraint names and second entry gives dictionary of constraint parameters. None is equivalent to ([], tw.constraint_params()).
        ''', 'first entry gives list of constraints, second entry gives dictionary of constraint parameters'),
    'cons_s': (None, 'type_constrained_none', (tuple, _lstdct),
        '''
            (default=None) Constraints on S1 and S2, where first entry gives list of constraints and second entry gives dictionary of constraint parameters. None is equivalent to ([], tw.constraint_params()).
        ''', 'first entry gives list of constraints, second entry gives dictionary of constraint parameters'),
    'cons_a_cat': (None, 'type_constrained_none', (dict, _dctlstdct),
        '''
            (default=None) Dictionary of constraints on A1_cat and A2_cat, where each column name links to a tuple where the first entry gives list of string constraint names and the second entry gives dictionary of constraint parameters. None is equivalent to {}.
        ''', 'first entry gives list of constraints, second entry gives dictionary of constraint parameters'),
    'cons_s_cat': (None, 'type_constrained_none', (dict, _dctlstdct),
        '''
            (default=None) Dictionary of constraints on S1_cat and S2_cat, where each column name links to a tuple where the first entry gives list of string constraint names and the second entry gives dictionary of constraint parameters. None is equivalent to {}.
        ''', 'first entry gives list of constraints, second entry gives dictionary of constraint parameters'),
    'cons_a_cts': (None, 'type_constrained_none', (dict, _dctlstdct),
        '''
            (default=None) Dictionary of constraints on A1_cts and A2_cts, where each column name links to a tuple where the first entry gives list of string constraint names and the second entry gives dictionary of constraint parameters. None is equivalent to {}.
        ''', 'first entry gives list of constraints, second entry gives dictionary of constraint parameters'),
    'cons_s_cts': (None, 'type_constrained_none', (dict, _dctlstdct),
        '''
            (default=None) Dictionary of constraints on S1_cts and S2_cts, where each column name links to a tuple where the first entry gives list of string constraint names and the second entry gives dictionary of constraint parameters. None is equivalent to {}.
        ''', 'first entry gives list of constraints, second entry gives dictionary of constraint parameters'),
    's_lower_bound': (1e-10, 'type_constrained', ((float, int), _gt0),
        '''
            (default=1e-10) Lower bound on estimated S1, S2, S1_cat, S2_cat, S1_cts, and S2_cts. Must be greater than 0.
        ''', '> 0'),
    'd_prior_movers': (1.0001, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1.0001) Account for probabilities being too small by adding (d_prior - 1) to pk1.
        ''', '>= 1'),
    # fit_stayers() parameters ##
    'n_iters_stayers': (100, 'type_constrained', (int, _gteq1),
        '''
            (default=100) Number of iterations for EM for stayers.
        ''', '>= 1'),
    'threshold_stayers': (1e-7, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1e-7) Threshold to break EM loop for stayers.
        ''', '>= 0'),
    'd_prior_stayers': (1.0001, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1.0001) Account for probabilities being too small by adding (d_prior - 1) to pk0.
        ''', '>= 1')
})

def blm_params(update_dict=None):
    '''
    Dictionary of default blm_params. Run tw.blm_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values; None is equivalent to {}

    Returns:
        (ParamsDict) dictionary of blm_params
    '''
    new_dict = _blm_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

_constraint_params_default = ParamsDict({
    'gap_akmmono': (0, 'type', (float, int),
        '''
            (default=0) Used for akmmono constraint.
        ''', None),
    'gap_mono_k': (0, 'type', (float, int),
        '''
            (default=0) Used for mono_k constraint.
        ''', None),
    'lower_bound': (0, 'type', (float, int),
        '''
            (default=0) Used for bounded_below constraint to determine lower bound.
        ''', None),
    'upper_bound': (0, 'type', (float, int),
        '''
            (default=0) Used for bounded_above constraint to determine upper bound.
        ''', None),
    'n_periods': (2, 'type_constrained', (int, _gteq1),
        '''
            (default=2) Number of periods in the data.
        ''', '>= 1'),
    'nt': (4, 'type', (float, int),
        '''
            (default=4)
        ''', None)
})

def constraint_params(update_dict=None):
    '''
    Dictionary of default constraint_params. Run tw.constraint_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values; None is equivalent to {}

    Returns:
        (ParamsDict) dictionary of constraint_params
    '''
    new_dict = _constraint_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

class QPConstrained:
    '''
    Solve a quadratic programming model of the following form:
        min_x(1/2 x.T @ P @ x + q.T @ x)
        s.t.    Gx <= h
                Ax = b

    Arguments:
        nl (int): number of worker types
        nk (int): number of firm types
    '''

    def __init__(self, nl, nk):
        # Store attributes
        self.nl = nl
        self.nk = nk

        # Inequality constraint matrix
        self.G = np.array([])
        # Inequality constraint bound
        self.h = np.array([])
        # Equality constraint matrix
        self.A = np.array([])
        # Equality constraint bound
        self.b = np.array([])

    def add_constraint_builtin(self, constraint, params=None):
        '''
        Add a built-in constraint.

        Arguments:
            constraint (str): name of constraint to add. Valid constraints are 'linear', 'stable_across_worker_types', 'stable_across_time_full', 'stable_across_time_partial_1', 'stable_across_time_partial_2', 'akmmono', 'mono_k', 'fixb', 'bounded_below', 'bounded_above', 'stationary', 'none', and 'sum'.
            params (ParamsDict): dictionary of parameters for constraint. Run tw.constraint_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.constraint_params().
        '''
        if params is None:
            params = constraint_params()
        # Unpack attributes
        nl, nk = self.nl, self.nk

        G = np.array([])
        h = np.array([])
        A = np.array([])
        b = np.array([])
        if constraint == 'linear':
            n_periods = params['n_periods']
        #     LL = np.zeros(shape=(n_periods * (nl - 1), n_periods * nl))
        #     for period in range(n_periods):
        #         row_shift = period * (nl - 1)
        #         col_shift = period * nl
        #         for l in range(nl - 1):
        #             LL[l + row_shift, l + col_shift] = 1
        #             LL[l + row_shift, l + col_shift + 1] = - 1
        #     KK = np.zeros(shape=(nk - 1, nk))
        #     for k in range(nk - 1):
        #         KK[k, k] = 1
        #         KK[k, k + 1] = - 1
        #     A = - np.kron(LL, KK)
        #     b = - np.zeros(shape=A.shape[0])
            A = np.zeros(shape=(n_periods * (nl - 2) * nk, n_periods * nl * nk))
            for period in range(n_periods):
                row_shift = period * (nl - 2) * nk
                col_shift = period * nl * nk
                for k in range(nk):
                    for l in range(nl - 2):
                        A[row_shift + l, col_shift + l] = 1
                        A[row_shift + l, col_shift + l + 1] = 2
                        A[row_shift + l, col_shift + l + 2] = -1
                    row_shift += (nl - 2)
                    col_shift += nl

            b = - np.zeros(shape=A.shape[0])

        elif constraint == 'stable_across_worker_types':
            n_periods = params['n_periods']
            A = np.zeros(shape=(n_periods * (nl - 1) * nk, n_periods * nl * nk))
            for period in range(n_periods):
                row_shift = period * (nl - 1) * nk
                col_shift = period * nl * nk
                for k in range(nk):
                    for l in range(nl - 1):
                        A[row_shift + l, col_shift + nk * l] = 1
                        A[row_shift + l, col_shift + nk * (l + 1)] = -1
                    row_shift += (nl - 1)
                    col_shift += 1

            b = - np.zeros(shape=A.shape[0])

        elif constraint == 'stable_across_time_full':
            # If not also using 'stable_across_worker_types' or 'linear' constraint
            n_periods = params['n_periods']
            A = np.zeros(shape=((n_periods - 1) * nl * nk, n_periods * nl * nk))
            col_shift = nl * nk
            for row in range((n_periods - 1) * nl * nk):
                A[row, row] = 1
                A[row, row + col_shift] = -1

            b = - np.zeros(shape=A.shape[0])

        elif constraint == 'stable_across_time_partial_1':
            # If also using 'stable_across_worker_types' constraint but not 'linear' constraint
            n_periods = params['n_periods']
            A = np.zeros(shape=((n_periods - 1) * nk, n_periods * nl * nk))
            for period in range(n_periods - 1):
                row_shift = period * nk
                col_shift = period * nl * nk
                for k in range(nk):
                    A[row_shift + k, col_shift + nl * k] = 1
                    A[row_shift + k, col_shift + nl * nk + nl * k] = -1

            b = - np.zeros(shape=A.shape[0])

        elif constraint == 'stable_across_time_partial_2':
            # If also using 'stable_across_worker_types' or 'linear' constraint
            n_periods = params['n_periods']
            nl_adj = min(nl, 2)
            A = np.zeros(shape=((n_periods - 1) * nl_adj * nk, n_periods * nl * nk))
            for period in range(n_periods - 1):
                row_shift = period * nl_adj * nk
                col_shift = period * nl * nk
                for k in range(nk):
                    for l in range(nl_adj):
                        A[row_shift + k + l, col_shift + nl * k + l] = 1
                        A[row_shift + k + l, col_shift + nl * nk + nl * k + l] = -1
                    row_shift += (nl_adj - 1)

            b = - np.zeros(shape=A.shape[0])

        elif constraint == 'akmmono':
            gap = params['gap_akmmono']
            LL = np.zeros(shape=(nl - 1, nl))
            for l in range(nl - 1):
                LL[l, l] = 1
                LL[l, l + 1] = - 1
            KK = np.zeros(shape=(nk - 1, nk))
            for k in range(nk - 1):
                KK[k, k] = 1
                KK[k, k + 1] = - 1
            G = np.kron(np.eye(nl), KK)
            h = - gap * np.ones(shape=(nl * (nk - 1)))

            A = - np.kron(LL, KK)
            b = - np.zeros(shape=A.shape[0])

        elif constraint == 'mono_k':
            gap = params['gap_mono_k']
            KK = np.zeros(shape=(nk - 1, nk))
            for k in range(nk - 1):
                KK[k, k] = 1
                KK[k, k + 1] = - 1
            G = np.kron(np.eye(nl), KK)
            h = - gap * np.ones(shape=(nl * (nk - 1)))

        elif constraint == 'fixb':
            # if len(self.G) > 0 or len(self.A) > 0:
            #     self.clear_constraints()
            #     warnings.warn("Constraint 'fixb' requires different dimensions than other constraints, existing constraints have been removed. It is recommended to manually run clear_constraints() prior to adding the constraint 'fixb' in order to ensure you are not unintentionally removing existing constraints.")
            nt = params['nt']
            KK = np.zeros(shape=(nk - 1, nk))
            for k in range(nk - 1):
                KK[k, k] = 1
                KK[k, k + 1] = - 1
            A = - np.kron(np.eye(nl), KK)
            MM = np.zeros(shape=(nt - 1, nt))
            for m in range(nt - 1):
                MM[m, m] = 1
                MM[m, m + 1] = - 1
            A = - np.kron(MM, A)
            b = - np.zeros(shape=nl * (nk - 1) * (nt - 1))

        elif constraint == 'bounded_below':
            lower_bound = params['lower_bound']
            n_periods = params['n_periods']
            G = - np.eye(n_periods * nl * nk)
            h = - lower_bound * np.ones(shape=n_periods * nl * nk)

        elif constraint == 'bounded_above':
            upper_bound = params['upper_bound']
            n_periods = params['n_periods']
            G = np.eye(n_periods * nl * nk)
            h = upper_bound * np.ones(shape=n_periods * nl * nk)

        elif constraint == 'stationary':
            LL = np.zeros(shape=(nl - 1, nl))
            for l in range(nl - 1):
                LL[l, l] = 1
                LL[l, l + 1] = - 1
            A = np.kron(LL, np.eye(nk))
            b = - np.zeros(shape=(nl - 1) * nk)

        elif constraint == 'none':
            G = - np.zeros(shape=(1, nk * nl))
            h = - np.array([0])

        elif constraint == 'sum':
            A = - np.kron(np.eye(nl), np.ones(shape=nk).T)
            b = - np.zeros(shape=nl)

        else:
            raise NotImplementedError('Invalid constraint {}.'.format(constraint))

        # Add constraints to attributes
        self.add_constraint_manual(G=G, h=h, A=A, b=b)

    def add_constraints_builtin(self, constraints, params=None):
        '''
        Add a built-in constraint.

        Arguments:
            constraints (list of str): names of constraints to add
            params (ParamsDict): dictionary of parameters for constraints. Run tw.constraint_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.constraint_params().
        '''
        if params is None:
            params = constraint_params()
        for constraint in constraints:
            self.add_constraint_builtin(constraint=constraint, params=params)

    def add_constraint_manual(self, G=None, h=None, A=None, b=None):
        '''
        Manually add a constraint. If setting inequality constraints, must set both G and h to have the same dimension 0. If setting equality constraints, must set both A and b to have the same dimension 0.

        Arguments:
            G (NumPy Array): inequality constraint matrix; None is equivalent to np.array([])
            h (NumPy Array): inequality constraint bound; None is equivalent to np.array([])
            A (NumPy Array): equality constraint matrix; None is equivalent to np.array([])
            b (NumPy Array): equality constraint bound; None is equivalent to np.array([])
        '''
        if G is None:
            G = np.array([])
        if h is None:
            h = np.array([])
        if A is None:
            A = np.array([])
        if b is None:
            b = np.array([])

        if len(G) > 0:
            # If inequality constraints
            if len(self.G) > 0:
                self.G = np.concatenate((self.G, G), axis=0)
                self.h = np.concatenate((self.h, h), axis=0)
            else:
                self.G = G
                self.h = h
        if len(A) > 0:
            # If equality constraints
            if len(self.A) > 0:
                self.A = np.concatenate((self.A, A), axis=0)
                self.b = np.concatenate((self.b, b), axis=0)
            else:
                self.A = A
                self.b = b

    def pad(self, l=0, r=0):
        '''
        Add padding to the left and/or right of C matrix.

        Arguments:
            l (int): how many columns to add on left
            r (int): how many columns to add on right
        '''
        if len(self.G) > 0:
            self.G = np.concatenate((
                    np.zeros(shape=(self.G.shape[0], l)),
                    self.G,
                    np.zeros(shape=(self.G.shape[0], r)),
                ), axis=1)
        else:
            self.G = np.zeros(shape=l + r)
        if len(self.A) > 0:
            self.A = np.concatenate((
                    np.zeros(shape=(self.A.shape[0], l)),
                    self.A,
                    np.zeros(shape=(self.A.shape[0], r)),
                ), axis=1)
        else:
            self.A = np.zeros(shape=l + r)

    def clear_constraints(self):
        '''
        Remove all constraints.
        '''
        self.G = np.array([])
        self.h = np.array([])
        self.A = np.array([])
        self.b = np.array([])

    def check_feasible(self):
        '''
        Check that constraints are feasible.

        Returns:
            (bool): True if constraints feasible, False otherwise
        '''
        # -----  Simulate an OLS -----
        rng = np.random.default_rng()
        # Parameters
        n = 2 * self.nl * self.nk # self.A.shape[1]
        k = self.nl * self.nk
        # Regressors
        x = rng.normal(size=k)
        M = rng.normal(size=(n, k))
        # Dependent
        Y = M @ x

        # ----- Create temporary solver -----
        cons = QPConstrained(self.nl, self.nk)
        cons.G = self.G
        cons.h = self.h
        cons.A = self.A
        cons.b = self.b

        # ----- Map to qpsolvers -----
        P = M.T @ M
        q = - M.T @ Y

        # ----- Run solver -----
        cons.solve(P, q)

        return cons.res is not None

    def solve(self, P, q):
        '''
        Solve a quadratic programming model of the following form:
            min_x(1/2 x.T @ P @ x + q.T @ x)
            s.t.    Gx <= h
                    Ax = b

        Arguments:
            P (NumPy Array): P in quadratic programming problem
            q (NumPy Array): q in quadratic programming problem

        Returns:
            (NumPy Array): x that solves quadratic programming problem
        '''
        if len(self.G) > 0 and len(self.A) > 0:
            self.res = solve_qp(P=P, q=q, G=self.G, h=self.h, A=self.A, b=self.b)
        elif len(self.G) > 0:
            self.res = solve_qp(P=P, q=q, G=self.G, h=self.h)
        elif len(self.A) > 0:
            self.res = solve_qp(P=P, q=q, A=self.A, b=self.b)
        else:
            self.res = solve_qp(P=P, q=q)

def lognormpdf(x, mu, sd):
    return - 0.5 * np.log(2 * np.pi) - np.log(sd) - (x - mu) ** 2 / (2 * sd ** 2)

class BLMModel:
    '''
    Class for solving the BLM model using a single set of starting values.

    Arguments:
        blm_params (ParamsDict): dictionary of parameters for BLM estimation. Run tw.blm_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.blm_params().
        rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)
    '''
    def __init__(self, blm_params=None, rng=None):
        if blm_params is None:
            blm_params = blm_params()
        if rng is None:
            rng = np.random.default_rng(None)

        # Store parameters
        self.params = blm_params
        self.rng = rng
        nl, nk = self.params.get_multiple(('nl', 'nk'))
        self.nl, self.nk = nl, nk

        # Log likelihood for movers
        self.lik1 = None
        # Path of log likelihoods for movers
        self.liks1 = np.array([])
        # Log likelihood for stayers
        self.lik0 = None
        # Path of log likelihoods for stayers
        self.liks0 = np.array([])
        # Connectedness measure of data
        self.connectedness = None
        # Parameter dimensions
        dims = (nl, nk)
        self.dims = dims

        ## Unpack control variable parameters ##
        cat_dict = self.params['categorical_controls']
        cts_dict = self.params['continuous_controls']
        ## Check if control variable parameters are None ##
        if cat_dict is None:
            cat_dict = {}
        if cts_dict is None:
            cts_dict = {}
        ## Create dictionary of all control variables ##
        controls_dict = cat_dict.copy()
        controls_dict.update(cts_dict.copy())
        ## Control variable ordering ##
        cat_cols = sorted(cat_dict.keys())
        cts_cols = sorted(cts_dict.keys())
        ## Store control variable attributes ##
        # Dictionaries #
        self.controls_dict = controls_dict
        self.cat_dict = cat_dict
        self.cts_dict = cts_dict
        # Lists #
        self.cat_cols = cat_cols
        self.cts_cols = cts_cols

        # Check that no control variables appear multiple times
        control_cols = cat_cols + cts_cols
        if len(control_cols) > len(set(control_cols)):
            for col in control_cols:
                if control_cols.count(col) > 1:
                    raise ValueError(f'Control variable names must be unique, but {col!r} appears multiple times.')

        # Check if there are any control variables
        self.any_controls = len(control_cols) > 0
        # Check if any control variables interact with worker type
        self.any_worker_type_interactions = any([col_dict['worker_type_interaction'] for col_dict in controls_dict.values()])
        # Check if any control variables don't interact with worker type
        self.any_non_worker_type_interactions = any([not col_dict['worker_type_interaction'] for col_dict in controls_dict.values()])

        ## Generate starting values ##
        a1_mu, a1_sig, a2_mu, a2_sig, s1_low, s1_high, s2_low, s2_high, pk1_prior, pk0_prior = self.params.get_multiple(('a1_mu', 'a1_sig', 'a2_mu', 'a2_sig', 's1_low', 's1_high', 's2_low', 's2_high', 'pk1_prior', 'pk0_prior'))
        # Model for Y1 | Y2, l, k for movers and stayers
        self.A1 = rng.normal(loc=a1_mu, scale=a1_sig, size=dims)
        self.S1 = rng.uniform(low=s1_low, high=s1_high, size=dims)
        # Model for Y4 | Y3, l, k for movers and stayers
        self.A2 = rng.normal(loc=a2_mu, scale=a2_sig, size=dims)
        self.S2 = rng.uniform(low=s2_low, high=s2_high, size=dims)
        # Model for p(K | l, l') for movers
        if pk1_prior is None:
            pk1_prior = np.ones(nl)
        self.pk1 = rng.dirichlet(alpha=pk1_prior, size=nk ** 2)
        # Model for p(K | l, l') for stayers
        if pk0_prior is None:
            pk0_prior = np.ones(nl)
        self.pk0 = rng.dirichlet(alpha=pk0_prior, size=nk)

        ### Control variables ###
        ## Categorical ##
        self.A1_cat = {col:
                rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=(nl, controls_dict[col]['n']))
                if controls_dict[col]['worker_type_interaction'] else
                rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=controls_dict[col]['n'])
            for col in cat_cols}
        self.A2_cat = {col:
                rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=(nl, controls_dict[col]['n']))
                if controls_dict[col]['worker_type_interaction'] else
                rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=controls_dict[col]['n'])
            for col in cat_cols}
        self.S1_cat = {col:
                rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=(nl, controls_dict[col]['n']))
                if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=controls_dict[col]['n'])
            for col in cat_cols}
        self.S2_cat = {col:
                rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=(nl, controls_dict[col]['n']))
                if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=controls_dict[col]['n'])
            for col in cat_cols}
        # # Stationary #
        # for col in cat_cols:
        #     if controls_dict[col]['stationary_A']:
        #         self.A2_cat[col] = self.A1_cat[col]
        #     if controls_dict[col]['stationary_S']:
        #         self.S2_cat[col] = self.S1_cat[col]
        ## Continuous ##
        self.A1_cts = {col:
                rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=nl)
                if controls_dict[col]['worker_type_interaction'] else
                rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=1)
            for col in cts_cols}
        self.A2_cts = {col:
                rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=nl)
                if controls_dict[col]['worker_type_interaction'] else
                rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=1)
            for col in cts_cols}
        self.S1_cts = {col:
                rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=nl)
                if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=1)
            for col in cts_cols}
        self.S2_cts = {col:
                rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=nl)
                if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=1)
            for col in cts_cols}
        # # Stationary #
        # for col in cts_cols:
        #     if controls_dict[col]['stationary_A']:
        #         self.A2_cts[col] = self.A1_cts[col]
        #     if controls_dict[col]['stationary_S']:
        #         self.S2_cts[col] = self.S1_cts[col]

        for l in range(nl):
            self.A1[l] = np.sort(self.A1[l], axis=0)
            self.A2[l] = np.sort(self.A2[l], axis=0)

        # if self.fixb:
        #     self.A2 = np.mean(self.A2, axis=1) + self.A1 - np.mean(self.A1, axis=1)

        # if self.stationary:
        #     self.A2 = self.A1

    # def reset_params(self):
    #     nl = self.nl
    #     nk = self.nk
    #     # Model for Y1 | Y2, l, k for movers and stayers
    #     self.A1 = np.tile(sorted(rng.normal(size=nl)), (nk, 1))
    #     self.S1 = np.ones(shape=(nk, nl))
    #     # Model for Y4 | Y3, l, k for movers and stayers
    #     self.A2 = self.A1.copy()
    #     self.S2 = np.ones(shape=(nk, nl))
    #     # Model for p(K | l, l') for movers
    #     self.pk1 = np.ones(shape=(nk * nk, nl)) / nl
    #     # Model for p(K | l, l') for stayers
    #     self.pk0 = np.ones(shape=(nk, nl)) / nl

    def _sum_by_non_nl(self, ni, C1, C2, compute_A=True, compute_S=True):
        '''
        Compute A1_sum/A2_sum/S1_sum_sq/S2_sum_sq for non-worker-interaction terms.

        Arguments:
            ni (int): number of observations
            C1 (dict of NumPy Arrays): dictionary linking column names to control variable data for the first period
            C2 (dict of NumPy Arrays): dictionary linking column names to control variable data for the second period
            compute_A (bool): if True, compute and return A terms
            compute_S (bool): if True, compute and return S terms

        Returns:
            (tuple of NumPy Arrays): (A1_sum, A2_sum, S1_sum_sq, S2_sum_sq), where each term gives the sum of estimated effects for control variables that do not interact with worker type (A terms are dropped if compute_A=False, and S terms are dropped if compute_S=False)
        '''
        if (not compute_A) and (not compute_S):
            raise ValueError('compute_A=False and compute_S=False. Must specify at least one to be True.')

        if not self.any_non_worker_type_interactions:
            # If all control variables interact with worker type
            if compute_A and compute_S:
                return [0] * 4
            return [0] * 2

        A1_cat, A2_cat, S1_cat, S2_cat = self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat
        A1_cts, A2_cts, S1_cts, S2_cts = self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        controls_dict = self.controls_dict

        if compute_A:
            A1_sum = np.zeros(ni)
            A2_sum = np.zeros(ni)
        if compute_S:
            S1_sum_sq = np.zeros(ni)
            S2_sum_sq = np.zeros(ni)

        ## Categorical ##
        for col in cat_cols:
            if not controls_dict[col]['worker_type_interaction']:
                if compute_A:
                    A1_sum += A1_cat[col][C1[col]]
                    A2_sum += A2_cat[col][C2[col]]
                if compute_S:
                    S1_sum_sq += S1_cat[col][C1[col]] ** 2
                    S2_sum_sq += S2_cat[col][C2[col]] ** 2
        ## Continuous ##
        for col in cts_cols:
            if not controls_dict[col]['worker_type_interaction']:
                if compute_A:
                    A1_sum += A1_cts[col] * C1[col]
                    A2_sum += A2_cts[col] * C2[col]
                if compute_S:
                    S1_sum_sq += S1_cts[col] ** 2
                    S2_sum_sq += S2_cts[col] ** 2

        if compute_A and compute_S:
            return (A1_sum, A2_sum, S1_sum_sq, S2_sum_sq)
        if compute_A:
            return (A1_sum, A2_sum)
        if compute_S:
            return (S1_sum_sq, S2_sum_sq)
    
    def _sum_by_nl_l(self, ni, l, C1, C2, compute_A=True, compute_S=True):
        '''
        Compute A1_sum/A2_sum/S1_sum_sq/S2_sum_sq to account for worker-interaction terms for a particular worker type.

        Arguments:
            ni (int): number of observations
            l (int): worker type (must be in range(0, nl))
            C1 (dict of NumPy Arrays): dictionary linking column names to control variable data for the first period
            C2 (dict of NumPy Arrays): dictionary linking column names to control variable data for the second period
            compute_A (bool): if True, compute and return A terms
            compute_S (bool): if True, compute and return S terms

        Returns:
            (tuple of NumPy Arrays): (A1_sum_l, A2_sum_l, S1_sum_sq_l, S2_sum_sq_l), where each term gives the sum of estimated effects for control variables that interact with worker type, specifically for worker type l (A terms are dropped if compute_A=False, and S terms are dropped if compute_S=False)
        '''
        if (not compute_A) and (not compute_S):
            raise ValueError('compute_A=False and compute_S=False. Must specify at least one to be True.')

        if not self.any_worker_type_interactions:
            # If no control variables interact with worker type
            if compute_A and compute_S:
                return [0] * 4
            return [0] * 2

        A1_cat, A2_cat, S1_cat, S2_cat = self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat
        A1_cts, A2_cts, S1_cts, S2_cts = self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        controls_dict = self.controls_dict

        if compute_A:
            A1_sum_l = np.zeros(ni)
            A2_sum_l = np.zeros(ni)
        if compute_S:
            S1_sum_sq_l = np.zeros(ni)
            S2_sum_sq_l = np.zeros(ni)

        ## Categorical ##
        for col in cat_cols:
            if controls_dict[col]['worker_type_interaction']:
                if compute_A:
                    A1_sum_l += A1_cat[col][l, C1[col]]
                    A2_sum_l += A2_cat[col][l, C2[col]]
                if compute_S:
                    S1_sum_sq_l += S1_cat[col][l, C1[col]] ** 2
                    S2_sum_sq_l += S2_cat[col][l, C2[col]] ** 2
        ## Continuous ##
        for col in cts_cols:
            if controls_dict[col]['worker_type_interaction']:
                if compute_A:
                    A1_sum_l += A1_cts[col][l] * C1[col]
                    A2_sum_l += A2_cts[col][l] * C2[col]
                if compute_S:
                    S1_sum_sq_l += S1_cts[col][l] ** 2
                    S2_sum_sq_l += S2_cts[col][l] ** 2

        if compute_A and compute_S:
            return (A1_sum_l, A2_sum_l, S1_sum_sq_l, S2_sum_sq_l)
        if compute_A:
            return (A1_sum_l, A2_sum_l)
        if compute_S:
            return (S1_sum_sq_l, S2_sum_sq_l)

    def compute_connectedness_measure(self, all=False):
        '''
        Computes graph connectedness measure among the movers within each type and updates self.connectedness to be the smallest value.

        Arguments:
            all (bool): if True, set self.connectedness to be the vector of connectedness for all worker types instead of the minimum
        '''
        nl, nk = self.nl, self.nk
        EV = np.zeros(shape=nl)
        pk1 = np.reshape(self.pk1, (nk, nk, nl))
        pr = (self.NNm.T * pk1.T).T

        for kk in range(nl):
            # Compute adjacency matrix
            A = pr[:, :, kk]
            A /= A.sum()
            A = (A + A.T) / 2
            D = np.diag(np.sum(A, axis=1) ** (-0.5))
            L = np.eye(nk) - D @ A @ D
            evals, evecs = np.linalg.eig(L)
            EV[kk] = sorted(evals)[1]

        if all:
            self.connectedness = EV
        self.connectedness = np.abs(EV).min()

    def fit_movers(self, jdata, compute_NNm=True):
        '''
        EM algorithm for movers.

        Arguments:
            jdata (Pandas DataFrame): movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # Unpack parameters
        params = self.params
        nl, nk, ni = self.nl, self.nk, jdata.shape[0]
        A1, A2, S1, S2 = self.A1, self.A2, self.S1, self.S2
        A1_cat, A2_cat, S1_cat, S2_cat = self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat
        A1_cts, A2_cts, S1_cts, S2_cts = self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        cat_dict, cts_dict = self.cat_dict, self.cts_dict
        controls_dict = self.controls_dict
        fixb, linear, stationary_A, stationary_S = params.get_multiple(('fixb', 'linear', 'stationary_A', 'stationary_S'))
        any_controls, any_non_worker_type_interactions = self.any_controls, self.any_non_worker_type_interactions

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        G1 = jdata.loc[:, 'g1'].to_numpy().astype(int, copy=False)
        G2 = jdata.loc[:, 'g2'].to_numpy().astype(int, copy=False)
        # Control variables
        C1 = {}
        C2 = {}
        for col in cat_cols:
            # Get subcolumns associated with col
            subcols = to_list(jdata.col_reference_dict[col])
            n_subcols = len(subcols)
            if n_subcols == 1:
                # If column is constant over time
                C1[col] = jdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
                C2[col] = jdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
            elif n_subcols == 2:
                # If column can change over time
                C1[col] = jdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
                C2[col] = jdata.loc[:, subcols[1]].to_numpy().astype(int, copy=False)
            else:
                raise NotImplementedError(f'Column names must have either one or two associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
        for col in cts_cols:
            # Get subcolumns associated with col
            subcols = to_list(jdata.col_reference_dict[col])
            n_subcols = len(subcols)
            if n_subcols == 1:
                # If column is constant over time
                C1[col] = jdata.loc[:, subcols[0]].to_numpy()
                C2[col] = jdata.loc[:, subcols[0]].to_numpy()
            elif n_subcols == 2:
                # If column can change over time
                C1[col] = jdata.loc[:, subcols[0]].to_numpy()
                C2[col] = jdata.loc[:, subcols[1]].to_numpy()
            else:
                raise NotImplementedError(f'Column names must have either one or two associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
        ## Sparse matrix representations ##
        GG1 = csc_matrix((np.ones(ni), (range(ni), G1)), shape=(ni, nk))
        GG2 = csc_matrix((np.ones(ni), (range(ni), G2)), shape=(ni, nk))
        CC1 = {col: csc_matrix((np.ones(ni), (range(ni), C1[col])), shape=(ni, controls_dict[col]['n'])) for col in cat_cols}
        CC2 = {col: csc_matrix((np.ones(ni), (range(ni), C2[col])), shape=(ni, controls_dict[col]['n'])) for col in cat_cols}

        # Transition probability matrix
        GG12 = csc_matrix((np.ones(ni), (range(ni), G1 + nk * G2)), shape=(ni, nk ** 2))

        # Matrix of prior probabilities
        pk1 = self.pk1
        # Matrix of posterior probabilities
        qi = np.ones(shape=(ni, nl))
        # Log pdfs
        lp = np.zeros(shape=(ni, nl))
        # Log likelihood for movers
        lik1 = None
        # Path of log likelihoods for movers
        liks1 = []
        prev_lik = np.inf
        # Fix error from bad initial guesses causing probabilities to be too low
        d_prior = params['d_prior_movers']

        ### Constraints ###
        ## General ##
        cons_a = QPConstrained(nl, nk)
        cons_s = QPConstrained(nl, nk)
        if params['cons_a'] is not None:
            cons_a.add_constraints_builtin(*params['cons_a'])
        if params['cons_s'] is not None:
            cons_s.add_constraints_builtin(*params['cons_s'])

        # Set lower bound on standard deviations
        cons_s.add_constraint_builtin('bounded_below', {'lower_bound': params['s_lower_bound'], 'n_periods': 2})
        if fixb:
            cons_a.add_constraint_builtin('fixb', {'nt': 2})
        if linear:
            cons_a.add_constraint_builtin('linear', {'n_periods': 2})
        if stationary_A:
            cons_a.add_constraint_builtin('stable_across_time_full', {'n_periods': 2})
        if stationary_S:
            cons_s.add_constraint_builtin('stable_across_time_full', {'n_periods': 2})
        ## Categorical ##
        cons_a_cat = {}
        cons_s_cat = {}
        for col in cat_cols:
            col_dict = controls_dict[col]
            cons_a_cat[col] = QPConstrained(nl, col_dict['n'])
            cons_s_cat[col] = QPConstrained(nl, col_dict['n'])

            # Set lower bound on standard deviations
            cons_s_cat[col].add_constraint_builtin('bounded_below', {'lower_bound': params['s_lower_bound'], 'n_periods': 2})
            if fixb:
                cons_a_cat[col].add_constraint_builtin('fixb', {'nt': 2})
            if col_dict['worker_type_interaction']:
                if linear:
                    cons_a_cat[col].add_constraint_builtin('linear', {'n_periods': 2})
                    if stationary_A or col_dict['stationary_A']:
                        cons_a_cat[col].add_constraint_builtin('stable_across_time_partial_2', {'n_periods': 2})
                    if stationary_S or col_dict['stationary_S']:
                        cons_s_cat[col].add_constraint_builtin('stable_across_time_partial_2', {'n_periods': 2})
                else:
                    if stationary_A or col_dict['stationary_A']:
                        cons_a_cat[col].add_constraint_builtin('stable_across_time_full', {'n_periods': 2})
                    if stationary_S or col_dict['stationary_S']:
                        cons_s_cat[col].add_constraint_builtin('stable_across_time_full', {'n_periods': 2})
            else:
                cons_a_cat[col].add_constraint_builtin('stable_across_worker_types', {'n_periods': 2})
                cons_s_cat[col].add_constraint_builtin('stable_across_worker_types', {'n_periods': 2})

                if stationary_A or col_dict['stationary_A']:
                    cons_a_cat[col].add_constraint_builtin('stable_across_time_partial_1', {'n_periods': 2})
                if stationary_S or col_dict['stationary_S']:
                    cons_s_cat[col].add_constraint_builtin('stable_across_time_partial_1', {'n_periods': 2})
        if params['cons_a_cat'] is not None:
            for col, cons_a_col in params['cons_a_cat'].items():
                cons_a_cat[col].add_constraints_builtin(*cons_a_col)
        if params['cons_s_cat'] is not None:
            for col, cons_s_col in params['cons_s_cat'].items():
                cons_s_cat[col].add_constraints_builtin(*cons_s_col)
        ## Continuous ##
        cons_a_cts = {}
        cons_s_cts = {}
        for col in cts_cols:
            col_dict = controls_dict[col]
            cons_a_cts[col] = QPConstrained(nl, 1)
            cons_s_cts[col] = QPConstrained(nl, 1)
            # Set lower bound on standard deviations
            cons_s_cts[col].add_constraint_builtin('bounded_below', {'lower_bound': params['s_lower_bound'], 'n_periods': 2})
            if fixb:
                cons_a_cts[col].add_constraint_builtin('fixb', {'nt': 2})
            if col_dict['worker_type_interaction']:
                if linear:
                    cons_a_cts[col].add_constraint_builtin('linear', {'n_periods': 2})
                    if stationary_A or col_dict['stationary_A']:
                        cons_a_cts[col].add_constraint_builtin('stable_across_time_partial_2', {'n_periods': 2})
                    if stationary_S or col_dict['stationary_S']:
                        cons_s_cts[col].add_constraint_builtin('stable_across_time_partial_2', {'n_periods': 2})
                else:
                    if stationary_A or col_dict['stationary_A']:
                        cons_a_cts[col].add_constraint_builtin('stable_across_time_full', {'n_periods': 2})
                    if stationary_S or col_dict['stationary_S']:
                        cons_s_cts[col].add_constraint_builtin('stable_across_time_full', {'n_periods': 2})
            else:
                cons_a_cts[col].add_constraint_builtin('stable_across_worker_types', {'n_periods': 2})
                cons_s_cts[col].add_constraint_builtin('stable_across_worker_types', {'n_periods': 2})

                if stationary_A or col_dict['stationary_A']:
                    cons_a_cts[col].add_constraint_builtin('stable_across_time_partial_1', {'n_periods': 2})
                if stationary_S or col_dict['stationary_S']:
                    cons_s_cts[col].add_constraint_builtin('stable_across_time_partial_1', {'n_periods': 2})
        if params['cons_a_cts'] is not None:
            for col, cons_a_col in params['cons_a_cts'].items():
                cons_a_cts[col].add_constraints_builtin(*cons_a_col)
        if params['cons_s_cts'] is not None:
            for col, cons_s_col in params['cons_s_cts'].items():
                cons_s_cts[col].add_constraints_builtin(*cons_s_col)

        for iter in range(params['n_iters_movers']):

            # ---------- E-Step ----------
            # We compute the posterior probabilities for each row
            # We iterate over the worker types, should not be be
            # too costly since the vector is quite large within each iteration
            if any_controls > 0:
                ## Account for control variables ##
                if iter == 0:
                    A1_sum, A2_sum, S1_sum_sq, S2_sum_sq = self._sum_by_non_nl(ni=ni, C1=C1, C2=C2)
                else:
                    S1_sum_sq, S2_sum_sq = self._sum_by_non_nl(ni=ni, C1=C1, C2=C2, compute_A=False)

                KK = G1 + nk * G2
                for l in range(nl):
                    # Update A1_sum/A2_sum/S1_sum_sq/S2_sum_sq to account for worker-interaction terms
                    A1_sum_l, A2_sum_l, S1_sum_sq_l, S2_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2)
                    lp1 = lognormpdf(Y1, A1_sum + A1_sum_l + A1[l, G1], np.sqrt(S1_sum_sq + S1_sum_sq_l + S1[l, G1] ** 2))
                    lp2 = lognormpdf(Y2, A2_sum + A2_sum_l + A2[l, G2], np.sqrt(S2_sum_sq + S2_sum_sq_l + S2[l, G2] ** 2))
                    lp[:, l] = np.log(pk1[KK, l]) + lp1 + lp2
            else:
                KK = G1 + nk * G2
                for l in range(nl):
                    lp1 = lognormpdf(Y1, A1[l, G1], S1[l, G1])
                    lp2 = lognormpdf(Y2, A2[l, G2], S2[l, G2])
                    lp[:, l] = np.log(pk1[KK, l]) + lp1 + lp2
            del lp1, lp2

            # We compute log sum exp to get likelihoods and probabilities
            qi = np.exp(lp.T - logsumexp(lp, axis=1)).T
            # # Add dirichlet prior
            # qi += d_prior - 1
            # # Normalize rows to sum to 1
            # qi = (qi.T / np.sum(qi, axis=1).T).T
            if params['return_qi']:
                return qi
            lik1 = logsumexp(lp, axis=1).mean() # FIXME should this be returned?
            # lik_prior = (params['d_prior'] - 1) * np.sum(np.log(pk1))
            # lik1 += lik_prior
            liks1.append(lik1)
            if params['verbose'] == 2:
                print('loop {}, liks {}'.format(iter, lik1))

            if abs(lik1 - prev_lik) < params['threshold_movers']:
                break
            prev_lik = lik1

            # ---------- M-step ----------
            # For now we run a simple ols, however later we
            # want to add constraints!
            # see https://scaron.info/blog/quadratic-programming-in-python.html

            # The regression has 2 * nl * nk parameters and nl * ni rows
            # We do not necessarily want to construct the duplicated data by nl
            # Instead we will construct X'X and X'Y by looping over nl
            # We also note that X'X is block diagonal with 2*nl matrices of dimensions nk^2
            ## General ##
            # Shift for period 2
            ts = nl * nk
            # Only store the diagonal
            XwXd = np.zeros(shape=2 * ts)
            if params['update_a']:
                XwY = np.zeros(shape=2 * ts)

            ## Categorical ##
            if len(cat_cols) > 0:
                ts_cat = {col: nl * col_dict['n'] for col, col_dict in cat_dict.items()}
                XwX_cat = {col: np.zeros(shape=2 * col_ts) for col, col_ts in ts_cat.items()}
                if params['update_a']:
                    XwY_cat = {col: np.zeros(shape=2 * col_ts) for col, col_ts in ts_cat.items()}
            ### Continuous ###
            if len(cts_cols) > 0:
                XwX_cts = {col: np.zeros(shape=2 * nl) for col in cts_cols}
                if params['update_a']:
                    XwY_cts = {col: np.zeros(shape=2 * nl) for col in cts_cols}

            if iter == 0:
                if any_non_worker_type_interactions:
                    Y1_adj = Y1.copy()
                    Y2_adj = Y2.copy()
                    Y1_adj -= A1_sum
                    Y2_adj -= A2_sum
                else:
                    Y1_adj = Y1
                    Y2_adj = Y2

            ## Update A ##
            for l in range(nl):
                # (We might be better off trying this within numba or something)
                l_index, r_index = l * nk, (l + 1) * nk
                # Shared weighted terms
                GG1_weighted = GG1.T @ diags(qi[:, l] / S1[l, G1])
                GG2_weighted = GG2.T @ diags(qi[:, l] / S2[l, G2])
                ## Compute XwXd terms ##
                XwXd[l_index: r_index] = (GG1_weighted @ GG1).diagonal()
                XwXd[ts + l_index: ts + r_index] = (GG2_weighted @ GG2).diagonal()
                if params['update_a']:
                    # Update A1_sum and A2_sum to account for worker-interaction terms
                    A1_sum_l, A2_sum_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, compute_S=False)
                    ## Compute XwY terms ##
                    XwY[l_index: r_index] = GG1_weighted @ (Y1_adj - A1_sum_l)
                    XwY[ts + l_index: ts + r_index] = GG2_weighted @ (Y2_adj - A2_sum_l)
                    del A1_sum_l, A2_sum_l
            del GG1_weighted, GG2_weighted

            # print('A1 before:')
            # print(A1)
            # print('A2 before:')
            # print(A2)
            # print('S1 before:')
            # print(S1)
            # print('S2 before:')
            # print(S2)
            # print('A1_cat_wi before:')
            # print(A1_cat_wi)
            # print('A2_cat_wi before:')
            # print(A2_cat_wi)
            # print('S1_cat_wi before:')
            # print(S1_cat_wi)
            # print('S2_cat_wi before:')
            # print(S2_cat_wi)

            # We solve the system to get all the parameters (note: this won't work if XwX is sparse)
            XwX = np.diag(XwXd)
            if params['update_a']:
                try:
                    cons_a.solve(XwX, -XwY)
                    res_a1, res_a2 = cons_a.res[: len(cons_a.res) // 2], cons_a.res[len(cons_a.res) // 2:]
                    # if pd.isna(res_a1).any() or pd.isna(res_a2).any():
                    #     raise ValueError('Estimated A1/A2 has NaN values')
                    A1 = np.reshape(res_a1, self.dims)
                    A2 = np.reshape(res_a2, self.dims)

                except ValueError as e:
                    # If constraints inconsistent, keep A1 and A2 the same
                    if params['verbose'] in [1, 2]:
                        print(f'Passing A1/A2: {e}')
                    # stop
                    pass

            ## Categorical ##
            for col in cat_cols:
                col_n = cat_dict[col]['n']
                if not cat_dict[col]['worker_type_interaction']:
                    Y1_adj += A1_cat[col][C1[col]]
                    Y2_adj += A2_cat[col][C2[col]]
                for l in range(nl):
                    l_index, r_index = l * col_n, (l + 1) * col_n
                    ## Compute shared terms ##
                    if cat_dict[col]['worker_type_interaction']:
                        S1_cat_l = S1_cat[col][l, C1[col]]
                        S2_cat_l = S2_cat[col][l, C2[col]]
                    else:
                        S1_cat_l = S1_cat[col][C1[col]]
                        S2_cat_l = S2_cat[col][C2[col]]
                    CC1_weighted = CC1[col].T @ diags(qi[:, l] / S1_cat_l)
                    CC2_weighted = CC2[col].T @ diags(qi[:, l] / S2_cat_l)
                    del S1_cat_l, S2_cat_l
                    ## Compute XwX_cat terms ##
                    XwX_cat[col][l_index: r_index] = (CC1_weighted @ CC1[col]).diagonal()
                    XwX_cat[col][ts_cat[col] + l_index: ts_cat[col] + r_index] = (CC2_weighted @ CC2[col]).diagonal()
                    if params['update_a']:
                        # Update A1_sum and A2_sum to account for worker-interaction terms
                        A1_sum_l, A2_sum_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, compute_S=False)
                        if cat_dict[col]['worker_type_interaction']:
                            A1_sum_l -= A1_cat[col][l, C1[col]]
                            A2_sum_l -= A2_cat[col][l, C2[col]]
                        ## Compute XwY_cat terms ##
                        XwY_cat[col][l_index: r_index] = CC1_weighted @ (Y1_adj - A1_sum_l - A1[l, G1])
                        XwY_cat[col][ts_cat[col] + l_index: ts_cat[col] + r_index] = CC2_weighted @ (Y2_adj - A2_sum_l - A2[l, G2])
                        del A1_sum_l, A2_sum_l
                del CC1_weighted, CC2_weighted

                # We solve the system to get all the parameters (note: this won't work if XwX_cat is sparse)
                XwX_cat[col] = np.diag(XwX_cat[col])
                if params['update_a']:
                    try:
                        a_solver = cons_a_cat[col]
                        a_solver.solve(XwX_cat[col], -XwY_cat[col])
                        res_a1, res_a2 = a_solver.res[: len(a_solver.res) // 2], a_solver.res[len(a_solver.res) // 2:]
                        # if pd.isna(res_a1).any() or pd.isna(res_a2).any():
                        #     raise ValueError(f'Estimated A1_cat/A2_cat has NaN values for column {col!r}')
                        if cat_dict[col]['worker_type_interaction']:
                            A1_cat[col] = np.reshape(res_a1, (nl, col_n))
                            A2_cat[col] = np.reshape(res_a2, (nl, col_n))
                        else:
                            A1_cat[col] = res_a1[: col_n]
                            A2_cat[col] = res_a2[: col_n]

                    except ValueError as e:
                        # If constraints inconsistent, keep A1_cat and A2_cat the same
                        if params['verbose'] in [1, 2]:
                            print(f'Passing A1_cat/A2_cat for column {col!r}: {e}')
                        # stop
                        pass
                if not cat_dict[col]['worker_type_interaction']:
                    Y1_adj -= A1_cat[col][C1[col]]
                    Y2_adj -= A2_cat[col][C2[col]]
            ## Continuous ##
            for col in cts_cols:
                if not cts_dict[col]['worker_type_interaction']:
                    Y1_adj += A1_cts[col] * C1[col]
                    Y2_adj += A2_cts[col] * C2[col]
                for l in range(nl):
                    ## Compute shared terms ##
                    if cts_dict[col]['worker_type_interaction']:
                        S1_cts_l = S1_cts[col][l]
                        S2_cts_l = S2_cts[col][l]
                    else:
                        S1_cts_l = S1_cts[col]
                        S2_cts_l = S2_cts[col]
                    CC1_weighted = C1[col].T @ diags(qi[:, l] / S1_cts_l)
                    CC2_weighted = C2[col].T @ diags(qi[:, l] / S2_cts_l)
                    del S1_cts_l, S2_cts_l
                    ## Compute XwX_cts terms ##
                    XwX_cts[col][l] = (CC1_weighted @ C1[col])
                    XwX_cts[col][nl + l] = (CC2_weighted @ C2[col])
                    if params['update_a']:
                        # Update A1_sum and A2_sum to account for worker-interaction terms
                        A1_sum_l, A2_sum_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, compute_S=False)
                        if cts_dict[col]['worker_type_interaction']:
                            A1_sum_l -= A1_cts[col][l] * C1[col]
                            A2_sum_l -= A2_cts[col][l] * C2[col]
                        ## Compute XwY_cts terms ##
                        XwY_cts[col][l] = CC1_weighted @ (Y1_adj - A1_sum_l - A1[l, G1])
                        XwY_cts[col][nl + l] = CC2_weighted @ (Y2_adj - A2_sum_l - A2[l, G2])
                        del A1_sum_l, A2_sum_l
                del CC1_weighted, CC2_weighted

                # We solve the system to get all the parameters (note: this won't work if XwX_cts is sparse)
                XwX_cts[col] = np.diag(XwX_cts[col])
                if params['update_a']:
                    try:
                        a_solver = cons_a_cts[col]
                        a_solver.solve(XwX_cts[col], -XwY_cts[col])
                        res_a1, res_a2 = a_solver.res[: len(a_solver.res) // 2], a_solver.res[len(a_solver.res) // 2:]
                        # if pd.isna(res_a1).any() or pd.isna(res_a2).any():
                        #     raise ValueError(f'Estimated A1_cts/A2_cts has NaN values for column {col!r}')
                        if cts_dict[col]['worker_type_interaction']:
                            A1_cts[col] = res_a1
                            A2_cts[col] = res_a2
                        else:
                            A1_cts[col] = res_a1[0]
                            A2_cts[col] = res_a2[0]

                    except ValueError as e:
                        # If constraints inconsistent, keep A1_cts and A2_cts the same
                        if params['verbose'] in [1, 2]:
                            print(f'Passing A1_cts/A2_cts for column {col!r}: {e}')
                        # stop
                        pass
                if not cts_dict[col]['worker_type_interaction']:
                    Y1_adj -= A1_cts[col] * C1[col]
                    Y2_adj -= A2_cts[col] * C2[col]

            if any_non_worker_type_interactions:
                # Update A1_sum and A2_sum
                A1_sum = Y1 - Y1_adj
                A2_sum = Y2 - Y2_adj

            if params['update_s']:
                # Next we extract the variances
                if iter == 0:
                    XwS = np.zeros(shape=2 * ts)

                    ## Categorical ##
                    if len(cat_cols) > 0:
                        XwS_cat = {col: np.zeros(shape=2 * col_ts) for col, col_ts in ts_cat.items()}
                    ## Continuous ##
                    if len(cts_cols) > 0:
                        XwS_cts = {col: np.zeros(shape=2 * nl) for col in cts_cols}

                ## Update S ##
                for l in range(nl):
                    # Update A1_sum and A2_sum to account for worker-interaction terms
                    A1_sum_l, A2_sum_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, compute_S=False)
                    eps1_l_sq = (Y1_adj - A1_sum_l - A1[l, G1]) ** 2
                    eps2_l_sq = (Y2_adj - A2_sum_l - A2[l, G2]) ** 2
                    del A1_sum_l, A2_sum_l
                    ## XwS terms ##
                    l_index, r_index = l * nk, (l + 1) * nk
                    XwS[l_index: r_index] = GG1.T @ diags(qi[:, l] / S1[l, G1]) @ eps1_l_sq
                    XwS[ts + l_index: ts + r_index] = GG2.T @ diags(qi[:, l] / S2[l, G2]) @ eps2_l_sq
                    ## Categorical ##
                    for col in cat_cols:
                        col_n = cat_dict[col]['n']
                        l_index, r_index = l * col_n, (l + 1) * col_n
                        if cat_dict[col]['worker_type_interaction']:
                            S1_cat_l = S1_cat[col][l, C1[col]]
                            S2_cat_l = S2_cat[col][l, C2[col]]
                        else:
                            S1_cat_l = S1_cat[col][C1[col]]
                            S2_cat_l = S2_cat[col][C2[col]]
                        ## XwS_cat terms ##
                        XwS_cat[col][l_index: r_index] = CC1[col].T @ diags(qi[:, l] / S1_cat_l) @ eps1_l_sq
                        XwS_cat[col][ts_cat[col] + l_index: ts_cat[col] + r_index] = CC2[col].T @ diags(qi[:, l] / S2_cat_l) @ eps2_l_sq
                        del S1_cat_l, S2_cat_l
                    ## Continuous ##
                    for col in cts_cols:
                        if cts_dict[col]['worker_type_interaction']:
                            S1_cts_l = S1_cts[col][l]
                            S2_cts_l = S2_cts[col][l]
                        else:
                            S1_cts_l = S1_cts[col]
                            S2_cts_l = S2_cts[col]
                        ## XwS_cts terms ##
                        XwS_cts[col][l] = C1[col].T @ diags(qi[:, l] / S1_cts_l) @ eps1_l_sq
                        XwS_cts[col][nl + l] = C2[col].T @ diags(qi[:, l] / S2_cts_l) @ eps2_l_sq
                        del S1_cts_l, S2_cts_l
                    del eps1_l_sq, eps2_l_sq

                try:
                    cons_s.solve(XwX, -XwS)
                    res_s1, res_s2 = cons_s.res[: len(cons_s.res) // 2], cons_s.res[len(cons_s.res) // 2:]
                    # if pd.isna(res_s1).any() or pd.isna(res_s2).any():
                    #     raise ValueError('Estimated S1/S2 has NaN values')
                    S1 = np.sqrt(np.reshape(res_s1, self.dims))
                    S2 = np.sqrt(np.reshape(res_s2, self.dims))

                except ValueError as e:
                    # If constraints inconsistent, keep S1 and S2 the same
                    if params['verbose'] in [1, 2]:
                        print(f'Passing S1/S2: {e}')
                    # stop
                    pass
                ## Categorical ##
                for col in cat_cols:
                    try:
                        col_n = cat_dict[col]['n']
                        s_solver = cons_s_cat[col]
                        s_solver.solve(XwX_cat[col], -XwS_cat[col])
                        res_s1, res_s2 = s_solver.res[: len(s_solver.res) // 2], s_solver.res[len(s_solver.res) // 2:]
                        # if pd.isna(res_s1).any() or pd.isna(res_s2).any():
                        #     raise ValueError(f'Estimated S1_cat/S2_cat has NaN values for column {col!r}')
                        if cat_dict[col]['worker_type_interaction']:
                            S1_cat[col] = np.sqrt(np.reshape(res_s1, (nl, col_n)))
                            S2_cat[col] = np.sqrt(np.reshape(res_s2, (nl, col_n)))
                        else:
                            S1_cat[col] = np.sqrt(res_s1[: col_n])
                            S2_cat[col] = np.sqrt(res_s2[: col_n])

                    except ValueError as e:
                        # If constraints inconsistent, keep S1_cat and S2_cat the same
                        if params['verbose'] in [1, 2]:
                            print(f'Passing S1_cat/S2_cat for column {col!r}: {e}')
                        # stop
                        pass
                ## Continuous ##
                for col in cts_cols:
                    try:
                        s_solver = cons_s_cts[col]
                        s_solver.solve(XwX_cts[col], -XwS_cts[col])
                        res_s1, res_s2 = s_solver.res[: len(s_solver.res) // 2], s_solver.res[len(s_solver.res) // 2:]
                        # if pd.isna(res_s1).any() or pd.isna(res_s2).any():
                        #     raise ValueError(f'Estimated S1_cts/S2_cts has NaN values for column {col!r}')
                        if cts_dict[col]['worker_type_interaction']:
                            S1_cts[col] = np.sqrt(res_s1)
                            S2_cts[col] = np.sqrt(res_s2)
                        else:
                            S1_cts[col] = np.sqrt(res_s1[0])
                            S2_cts[col] = np.sqrt(res_s2[0])

                    except ValueError as e:
                        # If constraints inconsistent, keep S1_cts and S2_cts the same
                        if params['verbose'] in [1, 2]:
                            print(f'Passing S1_cts/S2_cts for column {col!r}: {e}')
                        # stop
                        pass

            # print('A1 after:')
            # print(A1)
            # print('A2 after:')
            # print(A2)
            # print('S1 after:')
            # print(S1)
            # print('S2 after:')
            # print(S2)
            # print('A1_cat_wi after:')
            # print(A1_cat_wi)
            # print('A2_cat_wi after:')
            # print(A2_cat_wi)
            # print('S1_cat_wi after:')
            # print(S1_cat_wi)
            # print('S2_cat_wi after:')
            # print(S2_cat_wi)

            if params['update_pk1']:
                pk1 = GG12.T @ qi
                # Add dirichlet prior
                pk1 += d_prior - 1
                # Normalize rows to sum to 1
                pk1 = (pk1.T / np.sum(pk1, axis=1).T).T

        self.A1, self.A2, self.S1, self.S2 = A1, A2, S1, S2
        self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat = A1_cat, A2_cat, S1_cat, S2_cat
        self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts = A1_cts, A2_cts, S1_cts, S2_cts
        self.pk1, self.lik1, self.liks1 = pk1, lik1, np.array(liks1)

        # Update NNm
        if compute_NNm:
            self.NNm = jdata.groupby('g1')['g2'].value_counts().unstack(fill_value=0).to_numpy()

    def fit_stayers(self, sdata, compute_NNs=True):
        '''
        EM algorithm for stayers.

        Arguments:
            sdata (Pandas DataFrame): stayers
            compute_NNs (bool): if True, compute vector giving the number of stayers at each firm type (e.g. entry (1) gives the number of stayers at firm type 1)
        '''
        # Unpack parameters
        params = self.params
        nl, nk, ni = self.nl, self.nk, sdata.shape[0]
        A1, A2, S1, S2 = self.A1, self.A2, self.S1, self.S2
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        any_controls = self.any_controls
        # Fix error from bad initial guesses causing probabilities to be too low
        d_prior = params['d_prior_stayers']

        # Store wage outcomes and groups
        Y1 = sdata['y1'].to_numpy()
        Y2 = sdata['y2'].to_numpy()
        G1 = sdata['g1'].to_numpy().astype(int, copy=False)
        G2 = sdata['g2'].to_numpy().astype(int, copy=False)
        GG1 = csc_matrix((np.ones(ni), (range(ni), G1)), shape=(ni, nk))
        if any_controls:
            ## Account for control variables ##
            C1 = {}
            C2 = {}
            for col in cat_cols:
                # Get subcolumns associated with col
                subcols = to_list(sdata.col_reference_dict[col])
                n_subcols = len(subcols)
                if n_subcols == 1:
                    # If column is constant over time
                    C1[col] = sdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
                    C2[col] = sdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
                elif n_subcols == 2:
                    # If column can change over time
                    C1[col] = sdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
                    C2[col] = sdata.loc[:, subcols[1]].to_numpy().astype(int, copy=False)
                else:
                    raise NotImplementedError(f'Column names must have either one or two associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
            for col in cts_cols:
                # Get subcolumns associated with col
                subcols = to_list(sdata.col_reference_dict[col])
                n_subcols = len(subcols)
                if n_subcols == 1:
                    # If column is constant over time
                    C1[col] = sdata.loc[:, subcols[0]].to_numpy()
                    C2[col] = sdata.loc[:, subcols[0]].to_numpy()
                elif n_subcols == 2:
                    # If column can change over time
                    C1[col] = sdata.loc[:, subcols[0]].to_numpy()
                    C2[col] = sdata.loc[:, subcols[1]].to_numpy()
                else:
                    raise NotImplementedError(f'Column names must have either one or two associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')

        # Matrix of prior probabilities
        pk0 = self.pk0
        # Matrix of posterior probabilities
        qi = np.ones(shape=(ni, nl))
        # Log pdfs
        lp_stable = np.zeros(shape=(ni, nl))
        lp = np.zeros(shape=(ni, nl))
        # Log likelihood for stayers
        lik0 = None
        # Path of log likelihoods for stayers
        liks0 = []
        prev_lik = np.inf

        if any_controls:
            ## Account for control variables ##
            A1_sum, A2_sum, S1_sum_sq, S2_sum_sq = self._sum_by_non_nl(ni=ni, C1=C1, C2=C2)

            for l in range(nl):
                # Update A1_sum/S1_sum_sq to account for worker-interaction terms
                A1_sum_l, A2_sum_l, S1_sum_sq_l, S2_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2)
                lp1 = lognormpdf(Y1, A1_sum + A1_sum_l + A1[l, G1], np.sqrt(S1_sum_sq + S1_sum_sq_l + S1[l, G1] ** 2))
                lp2 = lognormpdf(Y2, A2_sum + A2_sum_l + A2[l, G2], np.sqrt(S2_sum_sq + S2_sum_sq_l + S2[l, G2] ** 2))
                lp_stable[:, l] = lp1 + lp2
        else:
            for l in range(nl):
                lp1 = lognormpdf(Y1, A1[l, G1], S1[l, G1])
                lp2 = lognormpdf(Y2, A2[l, G2], S2[l, G2])
                lp_stable[:, l] = lp1 + lp2
        del lp1, lp2

        for iter in range(params['n_iters_stayers']):

            # ---------- E-Step ----------
            # We compute the posterior probabilities for each row
            # We iterate over the worker types, should not be be
            # too costly since the vector is quite large within each iteration
            for l in range(nl):
                lp[:, l] = lp_stable[:, l] + np.log(pk0[G1, l])

            # We compute log sum exp to get likelihoods and probabilities
            qi = np.exp(lp.T - logsumexp(lp, axis=1)).T
            if params['return_qi']:
                return qi
            lik0 = logsumexp(lp, axis=1).mean() # FIXME should this be returned?
            liks0.append(lik0)
            if params['verbose'] == 2:
                print('loop {}, liks {}'.format(iter, lik0))

            if abs(lik0 - prev_lik) < params['threshold_stayers']:
                break
            prev_lik = lik0

            # ---------- M-step ----------
            pk0 = GG1.T @ qi
            # Add dirichlet prior
            pk0 += d_prior - 1
            # Normalize rows to sum to 1
            pk0 = (pk0.T / np.sum(pk0, axis=1).T).T

        self.pk0, self.lik0, self.liks0 = pk0, lik0, np.array(liks0)

        # Update NNs
        if compute_NNs:
            NNs = sdata['g1'].value_counts(sort=False)
            NNs.sort_index(inplace=True)
            self.NNs = NNs.to_numpy()

    def fit_movers_cstr_uncstr(self, jdata, compute_NNm=True):
        '''
        Run fit_movers(), first constrained, then using results as starting values, run unconstrained.

        Arguments:
            jdata (Pandas DataFrame): movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        ## First, simulate parameters but keep A fixed ##
        ## Second, use estimated parameters as starting point to run with A constrained to be linear ##
        ## Finally use estimated parameters as starting point to run without constraints ##
        # Save original parameters
        user_params = self.params.copy()
        ##### Loop 1 #####
        # First run fixm = True, which fixes A but updates S and pk
        self.params['update_a'] = False
        self.params['update_s'] = True
        self.params['update_pk1'] = True
        if self.params['verbose'] in [1, 2]:
            print('Running fixm movers')
        self.fit_movers(jdata, compute_NNm=False)
        ##### Loop 2 #####
        # Now update A
        self.params['update_a'] = True
        # Set constraints
        self.params['linear'] = True
        if self.params['verbose'] in [1, 2]:
            print('Running constrained movers')
        self.fit_movers(jdata, compute_NNm=False)
        ##### Loop 3 #####
        # Remove constraints
        self.params['linear'] = False
        if self.params['verbose'] in [1, 2]:
            print('Running unconstrained movers')
        self.fit_movers(jdata, compute_NNm=compute_NNm)
        ##### Compute connectedness #####
        self.compute_connectedness_measure()
        # Restore original parameters
        self.params = user_params

    def fit_A(self, jdata, compute_NNm=True):
        '''
        Run fit_movers() and update A while keeping S and pk1 fixed.

        Arguments:
            jdata (Pandas DataFrame): movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # Save original parameters
        user_params = self.params.copy()
        # Update parameters
        self.params['update_a'] = True
        self.params['update_s'] = False
        self.params['update_pk1'] = False
        # Estimate
        if self.params['verbose'] in [1, 2]:
            print('Running fit_A')
        self.fit_movers(jdata, compute_NNm=compute_NNm)
        # Restore original parameters
        self.params = user_params

    def fit_S(self, jdata, compute_NNm=True):
        '''
        Run fit_movers() and update S while keeping A and pk1 fixed.

        Arguments:
            jdata (Pandas DataFrame): movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # Save original parameters
        user_params = self.params.copy()
        # Update parameters
        self.params['update_a'] = False
        self.params['update_s'] = True
        self.params['update_pk1'] = False
        # Estimate
        if self.params['verbose'] in [1, 2]:
            print('Running fit_S')
        self.fit_movers(jdata, compute_NNm=compute_NNm)
        # Restore original parameters
        self.params = user_params

    def fit_pk(self, jdata, compute_NNm=True):
        '''
        Run fit_movers() and update pk1 while keeping A and S fixed.

        Arguments:
            jdata (Pandas DataFrame): movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # Save original parameters
        user_params = self.params.copy()
        # Update parameters
        self.params['update_a'] = False
        self.params['update_s'] = False
        self.params['update_pk1'] = True
        # Estimate
        if self.params['verbose'] in [1, 2]:
            print('Running fit_pk')
        self.fit_movers(jdata, compute_NNm=compute_NNm)
        # Restore original parameters
        self.params = user_params

    def _sort_matrices(self, firm_effects=False, reverse=False):
        '''
        Sort arrays by cluster means.

        Arguments:
            firm_effects (bool): if True, also sort by average firm effect
            reverse (bool): if True, sort in reverse order
        '''
        nk = self.nk
        controls_dict = self.controls_dict
        ## Compute sum of all effects ##
        A_sum = self.A1 + self.A2
        for control_dict in (self.A1_cat, self.A2_cat):
            for control_col, control_array in control_dict.items():
                if controls_dict[control_col]['worker_type_interaction']:
                    A_sum = (A_sum.T + np.mean(control_array, axis=1)).T
        ## Sort worker effects ##
        worker_effect_order = np.mean(self.A1 + self.A2, axis=1).argsort()
        if reverse:
            worker_effect_order = list(reversed(worker_effect_order))
        self.A1 = self.A1[worker_effect_order, :]
        self.A2 = self.A2[worker_effect_order, :]
        self.S1 = self.S1[worker_effect_order, :]
        self.S2 = self.S2[worker_effect_order, :]
        self.pk1 = self.pk1[:, worker_effect_order]
        self.pk0 = self.pk0[:, worker_effect_order]
        # Sort control variables #
        for control_dict in (self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat):
            for control_col, control_array in control_dict.items():
                if controls_dict[control_col]['worker_type_interaction']:
                    control_dict[control_col] = control_array[worker_effect_order, :]
        for control_dict in (self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts):
            for control_col, control_array in control_dict.items():
                if controls_dict[control_col]['worker_type_interaction']:
                    control_dict[control_col] = control_array[worker_effect_order]

        if firm_effects:
            ## Sort firm effects ##
            firm_effect_order = np.mean(self.A1 + self.A2, axis=0).argsort()
            if reverse:
                firm_effect_order = list(reversed(firm_effect_order))
            self.A1 = self.A1[:, firm_effect_order]
            self.A2 = self.A2[:, firm_effect_order]
            self.S1 = self.S1[:, firm_effect_order]
            self.S2 = self.S2[:, firm_effect_order]
            self.pk0 = self.pk0[firm_effect_order, :]
            # Reorder part 1: e.g. nk=2, and type 0 > type 1, then 0, 1, 2, 3 would reorder to 1, 0, 3, 2 (i.e. reorder within groups)
            pk1_order_1 = np.tile(firm_effect_order, nk) + nk * np.repeat(range(nk), nk)
            self.pk1 = self.pk1[pk1_order_1, :]
            # Reorder part 2: e.g. nk=2, and type 0 > type 1, then 0, 1, 2, 3 would reorder to 2, 3, 0, 1 (i.e. reorder between groups)
            pk1_order_2 = nk * np.repeat(firm_effect_order, nk) + np.tile(range(nk), nk)
            self.pk1 = self.pk1[pk1_order_2, :]

    def plot_A1(self, dpi=None):
        '''
        Plot self.A1.

        Arguments:
            dpi (float): dpi for plot
        '''
        # Sort A1 by average effect over firms
        sorted_A1 = self.A1.T[np.mean(self.A1.T, axis=1).argsort()].T
        sorted_A1 = sorted_A1[np.mean(sorted_A1, axis=1).argsort()]

        if dpi is not None:
            plt.figure(dpi=dpi)
        for l in range(self.nl):
            plt.plot(sorted_A1[l, :], label=f'Worker type {l}')
        plt.legend()
        plt.xlabel('Firm type')
        plt.ylabel('A1')
        plt.xticks(range(self.nk))
        plt.show()

    def plot_A2(self, dpi=None):
        '''
        Plot self.A2.

        Arguments:
            dpi (float): dpi for plot
        '''
        # Sort A2 by average effect over firms
        sorted_A2 = self.A2.T[np.mean(self.A2.T, axis=1).argsort()].T
        sorted_A2 = sorted_A2[np.mean(sorted_A2, axis=1).argsort()]

        if dpi is not None:
            plt.figure(dpi=dpi)
        for l in range(self.nl):
            plt.plot(sorted_A2[l, :], label=f'Worker type {l}')
        plt.legend()
        plt.xlabel('Firm type')
        plt.ylabel('A2')
        plt.xticks(range(self.nk))
        plt.show()

class BLMEstimator:
    '''
    Class for solving the BLM model using multiple sets of starting values.

    Arguments:
        params (ParamsDict): dictionary of parameters for BLM estimation. Run tw.blm_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.blm_params().
    '''

    def __init__(self, params=None):
        if params is None:
            params = blm_params()

        self.params = params
        # No initial model
        self.model = None
        # No likelihoods yet
        self.liks_high = None
        self.liks_low = None
        # No paths of likelihoods yet
        self.liks_all = None
        # No connectedness yet
        self.connectedness_high = None
        self.connectedness_low = None

    def _sim_model(self, jdata, rng=None):
        '''
        Generate model and run fit_movers_cstr_uncstr() given parameters.

        Arguments:
            jdata (Pandas DataFrame): movers
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        model = BLMModel(self.params.copy(), rng)
        model.fit_movers_cstr_uncstr(jdata)
        return model

    def fit(self, jdata, sdata, n_init=10, n_best=1, ncore=1, rng=None):
        '''
        EM model for movers and stayers.

        Arguments:
            jdata (Pandas DataFrame): movers
            sdata (Pandas DataFrame): stayers
            n_init (int): number of starting values
            n_best (int): take the n_best estimates with the highest likelihoods, and then take the estimate with the highest connectedness
            ncore (int): number of cores for multiprocessing
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Run sim_model()
        if ncore > 1:
            ## Multiprocessing
            # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
            seeds = rng.bit_generator._seed_seq.spawn(n_init)
            with Pool(processes=ncore) as pool:
                sim_model_lst = pool.starmap(self._sim_model, tqdm([(jdata, np.random.default_rng(seed)) for seed in seeds], total=n_init))
        else:
            sim_model_lst = itertools.starmap(self._sim_model, tqdm([(jdata, rng) for _ in range(n_init)], total=n_init))

        # Sort by likelihoods FIXME better handling of lik1 is None
        sorted_zipped_models = sorted([(model.lik1, model) for model in sim_model_lst if model.lik1 is not None], reverse=True, key=lambda a: a[0])
        sorted_lik_models = [model for _, model in sorted_zipped_models]

        # Save likelihood vs. connectedness for all models
        liks_high = np.zeros(shape=n_best) # Save likelihoods for n_best
        connectedness_high = np.zeros(shape=n_best) # Save connectedness for n_best
        liks_low = np.zeros(shape=n_init - n_best) # Save likelihoods for not n_best
        connectedness_low = np.zeros(shape=n_init - n_best) # Save connectedness for not n_best
        liks_all = [] # Save paths of likelihoods
        for i, model in enumerate(sorted_lik_models):
            liks_all.append(model.liks1)
            if i < n_best:
                liks_high[i] = model.lik1
                connectedness_high[i] = model.connectedness
            else:
                liks_low[i - n_best] = model.lik1
                connectedness_low[i - n_best] = model.connectedness
        self.liks_high = liks_high
        self.connectedness_high = connectedness_high
        self.liks_low = liks_low
        self.connectedness_low = connectedness_low
        self.liks_all = liks_all

        # Take the n_best best estimates and find the lowest connectedness
        best_lik_models = sorted_lik_models[: n_best]
        sorted_zipped_models = sorted([(model.connectedness, model) for model in best_lik_models], reverse=True, key=lambda a: a[0])
        best_model = sorted_zipped_models[0][1]

        if self.params['verbose'] in [1, 2]:
            print('liks_max:', best_model.lik1)
        self.model = best_model
        # Using best estimated parameters from fit_movers(), run fit_stayers()
        if self.params['verbose'] in [1, 2]:
            print('Running stayers')
        self.model.fit_stayers(sdata)
        # FIXME matrices shouldn't sort, because then you can't merge estimated effects into the original dataframe
        # self.model._sort_matrices()

    def plot_A1(self, dpi=None):
        '''
        Plot self.model.A1.

        Arguments:
            dpi (float): dpi for plot
        '''
        if self.model is not None:
            self.model.plot_A1(dpi)
        else:
            warnings.warn('Estimation has not yet been run.')

    def plot_A2(self, dpi=None):
        '''
        Plot self.model.A2.

        Arguments:
            dpi (float): dpi for plot
        '''
        if self.model is not None:
            self.model.plot_A2(dpi)
        else:
            warnings.warn('Estimation has not yet been run.')

    def plot_liks_connectedness(self, jitter=False, dpi=None):
        '''
        Plot likelihoods vs connectedness for the estimations run.

        Arguments:
            jitter (bool): if True, jitter points to prevent overlap
            dpi (float): dpi for plot
        '''
        if (self.model is not None) and (self.liks_high is not None) and (self.connectedness_high is not None) and (self.liks_low is not None) and (self.connectedness_low is not None):
            if dpi is not None:
                plt.figure(dpi=dpi)
            # So best estimation only graphed once, drop index from liks_high and connectedness_high
            liks_high_lst = list(self.liks_high)
            connectedness_high_lst = list(self.connectedness_high)
            drop_index = list(zip(liks_high_lst, connectedness_high_lst)).index((self.model.lik1, self.model.connectedness))
            del liks_high_lst[drop_index]
            del connectedness_high_lst[drop_index]
            # Now graph
            if jitter:
                plot = jitter_scatter
            else:
                plot = plt.scatter
            plot(self.liks_low, self.connectedness_low, marker='o', facecolors='None', edgecolors='C0')
            plot(liks_high_lst, connectedness_high_lst, marker='^', facecolors='None', edgecolors='C1')
            plt.scatter(self.model.lik1, self.model.connectedness, marker=(6, 2, 45), facecolors='C2')
            plt.xlabel('Likelihood')
            plt.ylabel('Connectedness')
            plt.show()
        else:
            warnings.warn('Estimation has not yet been run.')

####################
##### Old Code #####
####################
# ------------- Simulating functions ---------------------
# # Using the model, simulates a dataset of stayers
# def m2_mixt_simulate_stayers_withx(model, NNsx):
#     '''
#     Returns:
#         sdatae (Pandas DataFrame):
#     '''
#     G1 = np.zeros(shape=np.sum(NNsx)) - 1
#     G2 = np.zeros(shape=np.sum(NNsx)) - 1
#     Y1 = np.zeros(shape=np.sum(NNsx)) - 1
#     Y2 = np.zeros(shape=np.sum(NNsx)) - 1
#     K = np.zeros(shape=np.sum(NNsx)) - 1
#     X = np.zeros(shape=np.sum(NNsx)) - 1

#     A1 = model.A1
#     A2 = model.A2
#     S1 = model.S1
#     S2 = model.S2
#     pk0 = model.pk0
#     nk = model.nk
#     nf = model.nf
#     nx = len(NNsx)

#     # ------ Impute K, Y1, Y4 on jdata ------- #
#     i = 0
#     for l1 in range(nf):
#         for x in range(nx):
#             I = np.arange(i, i + NNsx[x, l1])
#             ni = len(I)
#             G1[I] = l1

#             # Draw k
#             draw_vals = np.arange(nk)
#             Ki = np.random.RandomState().choice(draw_vals, size=ni, replace=True, p=pk0[x, l1, :])
#             K[I] = Ki
#             X[I] = x

#             # Draw Y2, Y3
#             Y1[I] = A1[l1, Ki] + S1[l1, Ki] * np.random.RandomState().normal(size=ni)
#             Y2[I] = A2[l1, Ki] + S2[l1, Ki] * np.random.RandomState().normal(size=ni)

#             i = i + NNsx[x,l1]

#     sdatae = pd.DataFrame(data={'k': K, 'y1': Y1, 'y2': Y2, 'g1': G1, 'g2': G1, 'x': X})

#     return sdatae

# def m2_mixt_impute_movers(model, jdatae):
#     '''
#     '''
#     A1 = model.A1
#     S1 = model.S1
#     pk1 = model.pk1
#     A2 = model.A2
#     S2 = model.S2
#     nk = model.nk
#     nf = model.nf

#     # ------ Impute K, Y1, Y4 on jdata ------- #
#     jdatae.sim = jdatae.copy(deep=True)
#     # Generate Ki, Y1, Y4
#     # FIXME the follow code probably doesn't run
#     ni = len(jdatae)
#     jj = jdatae['g1'] + nf * (jdatae['g2'] - 1)
#     draw_vals = np.arange(nk)
#     Ki = np.random.RandomState().choice(draw_vals, size=ni, replace=True, p=pk1[jj, :])
#     # Draw Y1, Y4
#     Y1 = A1[jdatae['g1'], Ki] + S1[jdatae['g1'], Ki] * np.random.RandomState().normal(size=ni)
#     Y2 = A2[jdatae['g2'], Ki] + S2[jdatae['g2'], Ki] * np.random.RandomState().normal(size=ni)
#     # Append Ki, Y1, Y4 to jdatae.sim
#     jdatae.sim[['k_imp', 'y1_imp', 'y2_imp']] = [Ki, Y1, Y2]

#     return jdatae.sim

# def m2_mixt_impute_stayers(model, sdatae):
#     '''
#     '''
#     A1 = model.A1
#     S1 = model.S1
#     pk0 = model.pk0
#     A2 = model.A2
#     S2 = model.S2
#     nk = model.nk
#     nf = model.nf

#     # ------ Impute K, Y2, Y3 on sdata ------- #
#     sdatae.sim = sdatae.copy(deep=True)
#     # Generate Ki, Y2, Y3
#     # FIXME the follow code probably doesn't run
#     ni = len(sdatae)
#     draw_vals = np.arange(nk)
#     Ki = np.random.RandomState().choice(draw_vals, size=ni, replace=True, p=pk0[sdatae['x'], sdatae['g1'], :])
#     # Draw Y2, Y3
#     Y1 = A1[sdatae['g1'], Ki] + S1[sdatae['g1'], Ki] * np.random.RandomState().normal(size=ni)
#     Y2 = A2[sdatae['g1'], Ki] + S2[sdatae['g1'], Ki] * np.random.RandomState().normal(size=ni) # False for movers
#     # Append Ki, Y2, Y3 to sdatae.sim
#     sdatae.sim[['k_imp', 'y1_imp', 'y2_imp']] = [Ki, Y1, Y2]

#     return sdatae.sim


# # -------------------- Estimating functions -----------------------------


# # Estimates the static model parameters for movers
# def m2_mixt_movers(model, jdatae, ctrl):
#     # @comment Most important function
#     self.start_time = time.time()
#     # tic <- tic.new() FIXME

#     dprior = ctrl.dprior
#     model0 = ctrl.model0
#     taum = ctrl.tau

#     ### ----- GET MODEL  ---
#     nk = model.nk
#     nf = model.nf
#     A1 = model.A1
#     S1 = model.S1
#     A2 = model.A2
#     S2 = model.S2
#     pk1 = model.pk1

#     # ----- GET DATA
#     # Movers
#     Y1m = jdatae.y1
#     Y2m = jdatae.y2
#     G1m = jdatae.g1
#     G2m = jdatae.g2
#     GGm = G1m + nf * (G2m - 1)
#     Nm = len(jdatae)

#     # Get the constraints
#     CS1 = cons.pad(cons.get(ctrl.cstr_type[1], ctrl.cstr_val[1], nk, nf), nk * nf * 0, nk * nf * 1) # FIXME
#     CS2 = cons.pad(cons.get(ctrl.cstr_type[2], ctrl.cstr_val[2], nk, nf), nk * nf * 1, 0) # FIXME
#     # Combine them
#     CS = cons.bind(CS1, CS2) # FIXME

#     # Create the stationary contraints
#     if ctrl.fixb:
#         CS2 = cons.fixb(nk, nf, 2)
#         CS  = cons.bind(CS2, CS)

#     # Create a constraint for the variances
#     if ctrl.model_var:
#         CSw = cons.none(nk, nf * 2)
#     else:
#         CS1 = cons.pad(cons.mono_k(nk, nf), nk * nf * 0, nk * nf * 3)
#         CS2 = cons.pad(cons.mono_k(nk, nf),nk * nf * 1, nk * nf * 2)
#         CSw = cons.bind(CS1, CS2)
#         CSw.meq = len(CSw.H)

#     # Prepare matrices aggregated at the type level
#     Dkj1f = np.kron(np.kron(np.eye(nf), np.ones((nf, 1))), np.eye(nk)) # A[k,l] coefficients for g1
#     Dkj2f = np.kron(np.kron(np.ones((nf, 1)), np.eye(nf)), np.eye(nk)) # A[k,l] coefficients for g2

#     # Regression matrix for the variance
#     XX = pd.append([
#         pd.concat([Dkj1f, np.zeros(shape=Dkj2f.shape)], axis=1),
#         pd.concat([np.zeros(shape=Dkj1f.shape), Dkj2f], axis=1)], ignore_index=True)

#     ## --- Prepare regressions covariates --- #
#     # Create the dependent variables

#     lik_old = - np.inf
#     lik = - np.inf
#     lik_best = - np.inf
#     liks = 0
#     likm = 0

#     lpt1 = np.zeros(shape=(Nm, nk))
#     lpt2 = np.zeros(shape=(Nm, nk))
#     lp = np.zeros(shape=(Nm, nk))

#     # tic("prep") FIXME
#     stop = False
#     for step in range(ctrl.maxiter):
#         model1 = {'nk': nk, 'nf': nk, 'A1': A1, 'A2': A2, 'S1': S1, 'S2':S2, 'pk1': pk1, 'dprior': dprior}

#         ### ---------- E STEP ------------- #
#         # Compute the tau probabilities and the likelihood
#         if pd.isna(taum[1]) or step > 1:
#             # For efficiency we want to group by (l1,l2)
#             for l1 in range(nf):
#                 for l2 in range(nf):
#                     I = (G1m == l1) & (G2m == l2)
#                     ll = l1 + nf * (l2 - 1)
#                     if np.sum(I) > 0:
#                         for k in range(nk):
#                             lpt1[I, k] = lognormpdf(Y1m[I], A1[l1, k], S1[l1, k])
#                             lpt2[I, k] = lognormpdf(Y2m[I], A2[l2, k], S2[l2, k])

#                         # Sum the log of the periods
#                         lp[I, k] = np.log(pk1[ll, k]) + lpt1[I, k] + lpt2[I, k]

#             liks = np.sum(logRowSumExp(lp))
#             taum = np.exp(lp - spread(logRowSumExp(lp), 2, nk)) # Normalize the k probabilities Pr(k|Y1, Y2, Y3, Y4, l)

#             # Compute prior
#             lik_prior = (dprior - 1) * np.sum(np.log(pk1))
#             lik = liks + BLM.fit_A(jdata)lik_prior
#         else:
#             pass
#             # cat("skiping first max step, using supplied posterior probabilities\n") FIXME

#         # tic("estep") FIXME

#         if stop:
#             break

#         # ---------- MAX STEP ------------- #
#         # taum = makePosteriorStochastic(tau = taum,m = ctrl$stochastic) # if we want to implement stochastic EM

#         # we start by recovering the posterior weight, and the variances for each term
#         rwm = (taum + ctrl.posterior_reg).T

#         if not ctrl.fixm:
#             DYY = np.zeros(shape=(nk, nf, nf, 2))
#             WWT = np.zeros(shape=(nk, nf, nf, 2)) + 1e-7

#             for l1 in range(nf):
#                 for l2 in range(nf):
#                     I = (G1m == l1) & (G2m == l2)
#                     if np.sum(I) > 0:
#                         for k in range(nk):
#                             # Compute the posterior weight, it's not time specific
#                             ww = np.sum(taum[I, k] + ctrl.posterior_reg)

#                             # Construct dependent for each time period k,l2,l1,
#                             DYY[k, l2, l1, 1] = np.sum(Y1m[I] * (taum[I, k] + ctrl.posterior_reg)) / ww
#                             DYY[k, l2, l1, 2] = np.sum(Y2m[I] * (taum[I, k] + ctrl.posterior_reg)) / ww

#                             # Scaling the weight by the time specific variance
#                             WWT[k, l2, l1, 1] = ww / np.maximum(ctrl.sd_floor, S1[l1, k] ** 2)
#                             WWT[k, l2, l1, 2] = ww / np.maximum(ctrl.sd_floor, S2[l2, k] ** 2)

#             WWT = WWT / np.sum(WWT)
#             fit = slm.wfitc(XX, as.numeric(DYY), as.numeric(WWT), CS).solution # FIXME
#             is = 1
#             A1 = rdim(fit[is: (is + nk * nf - 1)], nk, nf).T
#             is = is + nk * nf # FIXME
#             A2 = rdim(fit[is: (is + nk * nf - 1)], nk, nf).T
#             is = is + nk * nf # FIXME

#             # compute the variances!!!!
#             DYY_bar = np.zeros(shape=(nk, nf, nf, 2))
#             DYY_bar[] = XX @ fit # FIXME
#             DYYV = np.zeros(shape=(nk, nf, nf, 2))

#             for l1 in range(nf):
#                 for l2 in range(nf):
#                     I = (G1m == l1) & (G2m == l2)
#                     if np.sum(I) > 0:
#                         for k in range(nk):
#                             # Construct dependent for each time period k, l2, l1
#                             ww = np.sum(taum[I, k] + ctrl.posterior_reg)
#                             DYYV[k, l2, l1, 1] = np.sum((Y1m[I] - DYY_bar[k, l2, l1, 1]) ** 2 * (taum[I, k] + ctrl.posterior_reg)) / ww
#                             DYYV[k, l2, l1, 2] = np.sum((Y2m[I] - DYY_bar[k, l2, l1, 2]) ** 2 * (taum[I, k] + ctrl.posterior_reg)) / ww

#             fitv = slm.wfitc(XX, np.array(dDYYV), np.array(WWT), CSw).solution # FIXME
#             is = 1
#             S1 = np.sqrt(rdim(fitv[is: (is + nk * nf - 1)], nk, nf).T); is = is + nk * nf # FIXME
#             S2 = sqrt(rdim(fitv[is: (is + nk * nf - 1)], nk, nf).T); is = is + nk * nf
#             S1[S1 < ctrl.sd_floor] = ctrl.sd_floor # Having a variance of exactly 0 creates problem in the likelihood
#             S2[S2 < ctrl.sd_floor] = ctrl.sd_floor

#         # tic("mstep-ols") FIXME

#         ## -------- PK probabilities ------------ #
#         ## --- Movers --- #
#         for l1 in range(nf):
#             for l2 in range(nf):
#                 jj = l1 + nf * (l2 - 1)
#                 I = (GGm == jj)
#                 if np.sum(I) > 1:
#                     pk1[jj, :] = np.sum(taum[I, :], axis=0)
#                 elif np.sum(I) == 0: # This deals with the case where the cell is empty
#                     pk1[jj, :] = 1 / nk
#                 else:
#                     pk1[jj, :] = taum[I, :]

#                 pk1[jj, :] = (pk1[jj, :] + dprior - 1) / (np.sum(pk1[jj, :] + dprior - 1))

#         #check_lik = computeLik(Y1m,Y2m,Y3m,Y4m,A12,B12,S12,A43,B43,S43,A2ma,A2mb,S2m,A3ma,A3mb,B32m,S3m)
#         #if (check_lik<lik) cat("lik did not go down on pk1 update\n")

#         # tic("mstep-pks") FIXME

#         # Checking model fit
#         if (np.sum(np.isnan(model0)) == 0) & (step % ctrl.nplot == ctrl.nplot - 1):
#             I1 = sorted(np.sum(A1, axis=0))
#             I2 = sorted(np.sum(model0.A1, axis=0))
#             # FIXME everything below
#             rr = addmom(A2[:, I1], model0.A2[:, I2], 'A2')
#             rr = addmom(A1[:, I1], model0.A1[:, I2], 'A1', rr)
#             rr = addmom(S2[:, I1], model0.S2[:, I2], 'S2', rr, type='var')
#             rr = addmom(S1[:, I1], model0.S1[:, I2], 'S1', rr, type='var')
#             rr = addmom(pk1, model0.pk1, 'pk1', rr, type='pr')

#             print(ggplot(rr, aes(x=val2, y=val1,color=type)) + geom_point() + facet_wrap(~name, scale='free') + theme_bw() + geom_abline(linetype=2)) # FIXME
#         else:
#             if (step % ctrl.nplot) == (ctrl.nplot - 1):
#                 plt.bar(A1) # wplot(A1)
#                 plt.show()

#         # -------- Check convergence ------- #
#         dlik = (lik - lik_old) / np.abs(lik_old)
#         lik_old = lik
#         lik_best = np.maximum(lik_best, lik)
#         if step % ctrl.ncat == 0:
#             # flog.info('[%3i][%s] lik=%4.4f dlik=%4.4e liks=%4.4e likm=%4.4e', step, ctrl.textapp, lik, dlik, liks, likm) FIXME
#             pass
#         if step > 10 and np.abs(dlik) < ctrl.tol:
#             break

#         # tic("loop-wrap") FIXME

#     # flog.info('[%3i][%s][final] lik=%4.4f dlik=%4.4e liks=%4.4e likm=%4.4e', step, ctrl.textapp, lik, dlik, liks, likm) FIXME

    # # Y1 | Y2
    # model.A1 = A1
    # model.S1 = S1
    # model.A2 = A2
    # model.S2 = S2
    # ## -- Movers --
    # model.pk1 = pk1

    # model.NNm = acast(jdatae[:, .N, list(g1, g2)], g1~g2, fill=0, value.var='N') # FIXME
    # model.likm = lik

    # end_time = time.time()
    # self = pd.DataFrame() # FIXME
    # self.res = {} # FIXME
    # self.res['total_time'] = end_time - self.start_time
    # del self.start_time

    # # self.res['tic'] = tic() FIXME
    # self.res['model'] = model
    # self.res['lik'] = lik
    # self.res['step'] = step
    # self.res['dlik'] = dlik
    # self.res['ctrl'] = ctrl
    # self.res['liks'] = liks
    # self.res['likm'] = likm

    # return self.res

# Use the marginal distributions to extract type distributions within each cluster and observable characteristics
def m2_mixt_stayers(model, sdata, ctrl):
    '''
    We set a linear programing problem to maximize likelihood subject to non negetivity and summing to one
    '''
    # The objective weights are the the density evaluated at each k
    nk  = model.nk
    nf  = model.nf
    Y1  = sdata.y1   # Firm id in period 1
    G1  = sdata.g1   # Wage in period 1
    X   = sdata.x    # Observable category
    # @todo add code in case X is missing, just set it to one
    nx = len(np.unique(X))
    N = len(Y1)
    Wmu = model.A1.T
    Wsg = model.S1.T

    # We create the index for the movement
    # This needs to take into account the observable X
    G1x = X + nx * (G1 - 1) # Joint in index for movement
    G1s = csc_matrix(np.zeros(shape=nf * nx), shape=(N, nf * nx))
    II = np.arange(N * G1x) # FIXME was 1:N + N * (G1x - 1)
    G1s[II] = 1
    tot_count = spread(np.sum(G1s, axis=0), 2, nk).T # FIXME
    empty_cells = tot_count[1, :] == 0

    #PI = rdirichlet(nf*nx,rep(1,nk))
    PI = rdim(model.pk0, nf * nx, nk) # FIXME
    PI_old = PI

    lik_old = np.inf
    iter_start = 1

    for count in range(iter_start, ctrl.maxiter):
        # The coeffs on the pis are the sum of the norm pdf
        norm1 = norm.ppf(spread(Y1, 2, nk), Wmu[:, G1].T, Wsg[:, G1].T) # FIXME
        tau = PI[G1x, :] * norm1
        tsum = np.sum(tau, axis=1)
        tau = tau / spread(tsum, 2, nk) # FIXME
        lik = - np.sum(np.log(tsum))

        PI = (tau.T @ G1s / tot_count).T
        PI[empty_cells, :] = 1 / nk * np.ones(shape=(np.sum(empty_cells), nk))

        dPI = np.abs(PI - PI_old)
        max_change = np.max(dPI)
        mean_change = np.mean(dPI)
        PI_old = PI

        if not np.isfinite(lik):
            status = - 5
            break

        prg = (lik_old - lik) / lik
        lik_old = lik

        if count % ctrl.ncat == ctrl.ncat - 1:
            # flog.info('[%3i][%s] lik=%4.4e inc=%4.4e max-pchg=%4.4e mean-pchg=%4.4e', count, ctrl.textapp, lik, prg, max_change, mean_change) # FIXME
            # flush.console() # FIXME
            pass

        if max_change < ctrl.tol:
            status = 1
            msg = 'converged'
            break

    model.pk0 = rdim(PI, nx, nf, nk) # FIXME
    model.liks = lik
    model.NNs = sdata[:, len(sdata) - 1, g1][sorted(g1)][:, N - 1] # FIXME g1 is not defined

    return model
