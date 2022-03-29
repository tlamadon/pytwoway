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
from bipartitepandas.util import ParamsDict, to_list
from tqdm import tqdm

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq1(a):
    return a >= 1
def _gteq0(a):
    return a >= 0
def _lstdct(a):
    return (isinstance(a[0], list) and isinstance(a[1], dict))

# Define default parameter dictionaries
_blm_params_default = ParamsDict({
    ## Class parameters
    'nl': (6, 'type_constrained', (int, _gteq1),
        '''
            (default=6) Number of worker types.
        ''', '>= 1'),
    'nk': (10, 'type_constrained', (int, _gteq1),
        '''
            (default=10) Number of firm types.
        ''', '>= 1'),
    'custom_independent_dict': (None, 'type_none', dict,
        '''
            (default=None) Dictionary of custom general column names (to use as controls) linked to the number of types for that column, where the estimated parameters should be independent of worker/firm type pairs. In other words, any column listed as a member of this parameter will have the same parameter estimated for each worker-firm type pair (the parameter value can still differ over time). None is equivalent to {}.
        ''', None),
    'custom_dependent_dict': (None, 'type_none', dict,
        '''
            (default=None) Dictionary of custom general column names (to use as controls) linked to the number of types for that column, where the estimated parameters should be dependent on worker/firm type pairs. In other words, any column listed as a member of this parameter will have a different parameter estimated for each worker-firm type pair (the parameter value can still differ over time). None is equivalent to {}.
        ''', None),
    'custom_nosplit_columns': (None, 'type_none', (list, tuple),
        '''
            (default=None) List of custom general column names (to use as controls), where the estimated parameter values should be constant over the pre- and post-periods. Any column not listed will be estimated separately for the pre- and post-periods. Only works for columns listed as independent of worker-firm type pairs. None is equivalent to [].
        ''', None),
    'fixb': (False, 'type', bool,
        '''
            (default=False) If True, set A2 = np.mean(A2, axis=1) + A1 - np.mean(A1, axis=1).
        ''', None),
    'stationary': (False, 'type', bool,
        '''
            (default=False) If True, set A1 = A2.
        ''', None),
    'verbose': (0, 'set', [0, 1, 2],
        '''
            (default=0) If 0, print no output; if 1, print additional output; if 2, print maximum output.
        ''', None),
    ## fit_movers() and fit_stayers parameters
    'return_qi': (False, 'type', bool,
        '''
            (default=False) If True, return qi matrix after first loop.
        ''', None),
    # fit_movers() parameters
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
    'cons_a': ((['lin'], {'n_periods': 2}), 'type_constrained', (tuple, _lstdct),
        '''
            (default=(['lin'], {'n_periods': 2})) Constraints on A1 and A2, where first entry gives list of constraints and second entry gives dictionary of constraint parameters.
        ''', 'first entry gives list of constraints, second entry gives dictionary of constraint parameters'),
    'cons_s': ((['biggerthan'], {'gap_bigger': 1e-7, 'n_periods': 2}), 'type_constrained', (tuple, _lstdct),
        '''
            (default=(['biggerthan'], {'gap_bigger': 1e-7, 'n_periods': 2})) Constraints on S1 and S2, where first entry gives list of constraints and second entry gives dictionary of constraint parameters.
        ''', 'first entry gives list of constraints, second entry gives dictionary of constraint parameters'),
    # fit_stayers() parameters
    'n_iters_stayers': (100, 'type_constrained', (int, _gteq1),
        '''
            (default=100) Number of iterations for EM for stayers.
        ''', '>= 1'),
    'threshold_stayers': (1e-7, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1e-7) Threshold to break EM loop for stayers.
        ''', '>= 0'),
    'd_prior': (1.0001, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1.0001) Account for probabilities being too small by adding (d_prior - 1).
        ''', '>= 1')
})

def blm_params(update_dict={}):
    '''
    Dictionary of default blm_params. Run tw.blm_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of blm_params
    '''
    new_dict = _blm_params_default.copy()
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
    'gap_bigger': (0, 'type', (float, int),
        '''
            (default=0) Used for biggerthan constraint to determine bound.
        ''', None),
    'gap_smaller': (0, 'type', (float, int),
        '''
            (default=0) Used for smallerthan constraint to determine bound.
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

def constraint_params(update_dict={}):
    '''
    Dictionary of default constraint_params. Run tw.constraint_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of constraint_params
    '''
    new_dict = _constraint_params_default.copy()
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
        n_dep (int): number of dependent control variable types
        n_indep_split (int): number of independent control variable types that can change over time
        n_indep_nosplit (int): number of independent control variable types that are constant over time
    '''

    def __init__(self, nl, nk, n_dep, n_indep_split, n_indep_nosplit):
        # Store attributes
        self.nl = nl
        self.nk = nk
        self.n_dep = max(1, n_dep)
        self.n_indep_split = n_indep_split
        self.n_indep_nosplit = n_indep_nosplit
        self.cum_params = nk * n_dep + n_indep_split + n_indep_nosplit

        # Inequality constraint matrix
        self.G = np.array([])
        # Inequality constraint bound
        self.h = np.array([])
        # Equality constraint matrix
        self.A = np.array([])
        # Equality constraint bound
        self.b = np.array([])

    def add_constraint_builtin(self, constraint, params=constraint_params()):
        '''
        Add a built-in constraint.

        Arguments:
            constraint (str): name of constraint to add
            params (ParamsDict): dictionary of parameters for constraint. Run tw.constraint_params().describe_all() for descriptions of all valid parameters.
        '''
        # Unpack attributes
        nl, nk = self.nl, self.nk
        n_dep, n_indep_split, n_indep_nosplit, cum_params = self.n_dep, self.n_indep_split, self.n_indep_nosplit, self.cum_params

        G = np.array([])
        h = np.array([])
        A = np.array([])
        b = np.array([])
        if constraint in ['lin', 'lin_add', 'akm']:
            n_periods = params['n_periods']
            LL = np.zeros(shape=(n_periods * (nl - 1), n_periods * nl))
            for period in range(n_periods):
                row_shift = period * (nl - 1)
                col_shift = period * nl
                for l in range(nl - 1):
                    LL[l + row_shift, l + col_shift] = 1
                    LL[l + row_shift, l + col_shift + 1] = - 1
            KK = np.zeros(shape=(cum_params - 1, cum_params))
            for k in range(cum_params - 1):
                KK[k, k] = 1
                KK[k, k + 1] = - 1
            # NOTE: commented out code below does the constraint solely for dependent parameters
            # KK = np.zeros(shape=(nk * n_dep - 1, nk * n_dep))
            # for k in range(nk * n_dep - 1):
            #     KK[k, k] = 1
            #     KK[k, k + 1] = - 1
            # LL_KK = - np.kron(LL, KK)
            # A = np.zeros(shape=(LL_KK.shape[0], n_periods * nl * cum_params))
            # A[:, : LL_KK.shape[1] // 2] = LL_KK[:, : LL_KK.shape[1] // 2]
            # A[:, A.shape[1] // 2: A.shape[1] // 2 + LL_KK.shape[1] // 2] = LL_KK[:, LL_KK.shape[1] // 2:]
            A = - np.kron(LL, KK)
            b = - np.zeros(shape=A.shape[0])

        elif constraint == 'stable_within_time':
            n_periods = params['n_periods']
            A = np.zeros(shape=(n_periods * (nl - 1) * (n_indep_split + n_indep_nosplit), n_periods * nl * cum_params))
            for period in range(n_periods):
                row_shift = period * (nl - 1) * (n_indep_split + n_indep_nosplit)
                col_shift = period * nl * cum_params + nl * nk * n_dep
                for _ in range(n_indep_split + n_indep_nosplit):
                    for k in range(nl - 1):
                        A[row_shift + k, col_shift + k * nl] = 1
                        A[row_shift + k, col_shift + (k + 1) * nl] = -1
                    row_shift += (nl - 1)
                    col_shift += 1

            b = - np.zeros(shape=A.shape[0])

        elif constraint == 'stable_across_time':
            n_periods = params['n_periods']
            A = np.zeros(shape=((n_periods - 1) * n_indep_nosplit, n_periods * nl * cum_params))
            for period in range(n_periods - 1):
                row_shift = period * n_indep_nosplit
                col_shift = period * nl * cum_params + nl * (nk * n_dep + n_indep_split)
                for k in range(n_indep_nosplit):
                    A[row_shift + k, col_shift + k] = 1
                    A[row_shift + k, col_shift + nl * cum_params + k] = -1

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
            if len(self.G) > 0 or len(self.A) > 0:
                self.clear_constraints()
                warnings.warn("Constraint 'fixb' requires different dimensions than other constraints, existing constraints have been removed. It is recommended to manually run clear_constraints() prior to adding the constraint 'fixb' in order to ensure you are not unintentionally removing existing constraints.")
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

        elif constraint in ['biggerthan', 'greaterthan']:
            gap = params['gap_bigger']
            n_periods = params['n_periods']
            G = - np.eye(n_periods * nl * cum_params)
            h = - gap * np.ones(shape=n_periods * nl * cum_params)

        elif constraint in ['smallerthan', 'lessthan']:
            gap = params['gap_smaller']
            n_periods = params['n_periods']
            G = np.eye(n_periods * nl * cum_params)
            h = gap * np.ones(shape=n_periods * nl * cum_params)

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

    def add_constraints_builtin(self, constraints, params=constraint_params()):
        '''
        Add a built-in constraint.

        Arguments:
            constraints (list of str): names of constraints to add
            params (ParamsDict): dictionary of parameters for constraints. Run tw.constraint_params().describe_all() for descriptions of all valid parameters.
        '''
        for constraint in constraints:
            self.add_constraint_builtin(constraint=constraint, params=params)

    def add_constraint_manual(self, G=np.array([]), h=np.array([]), A=np.array([]), b=np.array([])):
        '''
        Manually add a constraint. If setting inequality constraints, must set both G and h to have the same dimension 0. If setting equality constraints, must set both A and b to have the same dimension 0.

        Arguments:
            G (NumPy Array): inequality constraint matrix
            h (NumPy Array): inequality constraint bound
            A (NumPy Array): equality constraint matrix
            b (NumPy Array): equality constraint bound
        '''
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
        blm_params (ParamsDict): dictionary of parameters for BLM estimation. Run tw.blm_params().describe_all() for descriptions of all valid parameters.
        rng (np.random.Generator): NumPy random number generator
    '''
    def __init__(self, blm_params=blm_params(), rng=np.random.default_rng(None)):
        # Store parameters
        self.params = blm_params
        nl = self.params['nl']
        nk = self.params['nk']
        self.nl = nl
        self.nk = nk
        self.fixb = self.params['fixb']
        self.stationary = self.params['stationary']
        self.rng = rng
        ## Unpack custom column parameters
        custom_dep_dict = self.params['custom_dependent_dict']
        custom_indep_dict = self.params['custom_independent_dict']
        custom_nosplit_cols = self.params['custom_nosplit_columns']
        ## Check if custom column parameters are None
        if custom_dep_dict is None:
            custom_dep_dict = {}
        if custom_indep_dict is None:
            custom_indep_dict = {}
        if custom_nosplit_cols is None:
            custom_nosplit_cols = []
        ## Create dictionary of all custom columns
        custom_cols_dict = custom_dep_dict.copy()
        custom_cols_dict.update(custom_indep_dict.copy())
        ## Custom column order
        custom_dep_cols = sorted(custom_dep_dict.keys())
        custom_indep_split_cols = [col for col in sorted(custom_indep_dict.keys()) if col not in custom_nosplit_cols]
        custom_indep_nosplit_cols = [col for col in sorted(custom_indep_dict.keys()) if col in custom_nosplit_cols]
        custom_cols = custom_dep_cols + custom_indep_split_cols + custom_indep_nosplit_cols
        ## Store custom column attributes
        self.custom_dep_dict = custom_dep_dict
        self.custom_indep_dict = custom_indep_dict
        self.custom_cols_dict = custom_cols_dict
        # self.custom_nosplit_cols = custom_nosplit_cols
        self.custom_dep_cols = custom_dep_cols
        self.custom_indep_split_cols = custom_indep_split_cols
        self.custom_indep_nosplit_cols = custom_indep_nosplit_cols
        self.custom_cols = custom_cols

        for col in custom_indep_dict.keys():
            if col in custom_dep_cols:
                # Make sure independent and dependent custom columns don't overlap
                raise NotImplementedError(f'Custom independent columns and custom dependent columns cannot overlap, but input lists column {col!r} as a member of both.')
        for col in custom_nosplit_cols:
            if col not in custom_cols:
                # Make sure all custom no-split columns are actually used
                raise NotImplementedError(f'All custom columns listed not be split over time must be included as control variables, but {col!r} is not included.')
            if col in custom_dep_cols:
                # Make sure dependent columns are not split
                raise NotImplementedError(f'Cannot set custom columns that are dependent on worker-firm type pairs to not split, but input indicates column {col!r} should both be dependent and not split.')

        dims = [nl, nk]
        for col in custom_dep_cols:
            # dims must account for all dependent columns (and make sure the columns are in the correct order)
            dims.append(custom_dep_dict[col])
        self.dims = dims
        
        # Model for Y1 | Y2, l, k for movers and stayers
        self.A1 = np.tile(sorted(rng.normal(scale=2, size=nl)), list(reversed(dims[1:])) + [1]).T
        self.S1 = np.ones(shape=dims)
        # Model for Y4 | Y3, l, k for movers and stayers
        self.A2 = self.A1.copy()
        self.S2 = np.ones(shape=dims)
        # Model for p(K | l, l') for movers
        self.pk1 = rng.dirichlet(alpha=np.ones(nl), size=nk ** 2) # np.ones(shape=(nk ** 2, nl)) / nl
        # Model for p(K | l, l') for stayers
        self.pk0 = np.ones(shape=(nk, nl)) / nl
        ## Control variables
        # Split
        self.A1_indep = {col: np.sort(rng.normal(size=custom_indep_dict[col])) for col in custom_indep_split_cols}
        self.S1_indep = {col: np.sort(rng.normal(size=custom_indep_dict[col])) for col in custom_indep_split_cols}
        self.A2_indep = {col: np.sort(rng.normal(size=custom_indep_dict[col])) for col in custom_indep_split_cols}
        self.S2_indep = {col: np.sort(rng.normal(size=custom_indep_dict[col])) for col in custom_indep_split_cols}
        # No-split
        self.A_indep = {col: np.sort(rng.normal(size=custom_indep_dict[col])) for col in custom_indep_nosplit_cols}
        self.S_indep = {col: np.sort(rng.normal(size=custom_indep_dict[col])) for col in custom_indep_nosplit_cols}

        # Log likelihood for movers
        self.lik1 = None
        # Path of log likelihoods for movers
        self.liks1 = np.array([])
        # Log likelihood for stayers
        self.lik0 = None
        # Path of log likelihoods for stayers
        self.liks0 = np.array([])

        self.connectedness = None

        for l in range(nl):
            self.A1[l] = np.sort(self.A1[l], axis=0)
            self.A2[l] = np.sort(self.A2[l], axis=0)

        if self.fixb:
            self.A2 = np.mean(self.A2, axis=1) + self.A1 - np.mean(self.A1, axis=1)

        if self.stationary:
            self.A2 = self.A1

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
        A1_indep, A2_indep, A_indep, S1_indep, S2_indep, S_indep = self.A1_indep, self.A2_indep, self.A_indep, self.S1_indep, self.S2_indep, self.S_indep

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        G1 = jdata.loc[:, 'g1'].to_numpy().astype(int, copy=False)
        G2 = jdata.loc[:, 'g2'].to_numpy().astype(int, copy=False)
        # Control variables
        C = {}
        C1 = {}
        C2 = {}
        for col in self.custom_cols:
            # Get subcolumns associated with col
            subcols = to_list(jdata.col_reference_dict[col])
            if col in self.custom_indep_nosplit_cols:
                # If parameter associated with column is constant over time
                if len(subcols) != 1:
                    raise NotImplementedError(f'No-split general columns must have one associated column, but {col!r} does not.')
                C[col] = jdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
            else:
                # If parameter associated with column can change over time
                if len(subcols) != 2:
                    raise NotImplementedError(f'Split general columns must have two associated columns, but {col!r} does not.')
                C1[col] = jdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
                C2[col] = jdata.loc[:, subcols[1]].to_numpy().astype(int, copy=False)
        ## Sparse matrix representations ##
        # GG1 = csc_matrix((np.ones(ni), (range(ni), G1)), shape=(ni, nk))
        # GG2 = csc_matrix((np.ones(ni), (range(ni), G2)), shape=(ni, nk))
        # CC = {col: csc_matrix((np.ones(ni), (range(ni), vec)), shape=(ni, self.custom_cols_dict[col])) for col, vec in C.items()}
        # CC1 = {col: csc_matrix((np.ones(ni), (range(ni), vec)), shape=(ni, self.custom_cols_dict[col])) for col, vec in C1.items()}
        # CC2 = {col: csc_matrix((np.ones(ni), (range(ni), vec)), shape=(ni, self.custom_cols_dict[col])) for col, vec in C2.items()}
        n_param_cols = 1 + len(self.custom_indep_dict)
        CC1_indicator = np.zeros([ni, n_param_cols])
        CC2_indicator = np.zeros([ni, n_param_cols])
        CC1_indicator[:, 0] = G1
        CC2_indicator[:, 0] = G2

        n_dep_params = nk
        for col in self.custom_cols:
            ## First, set indicators for dependent columns ##
            if col in self.custom_dep_cols:
                # If column depends on worker-firm type pair
                CC1_indicator[:, 0] += n_dep_params * C1[col]
                CC2_indicator[:, 0] += n_dep_params * C2[col]
                n_dep_params *= self.custom_cols_dict[col]
        cum_params = n_dep_params
        i = 0
        for col in self.custom_cols:
            ## Second, set indicators for independent split columns ##
            if col in self.custom_indep_split_cols:
                # If parameter associated with column can change over time
                i += 1
                CC1_indicator[:, i] = cum_params + C1[col]
                CC2_indicator[:, i] = cum_params + C2[col]
                cum_params += self.custom_cols_dict[col]
        n_indep_split_params = cum_params - n_dep_params
        for col in self.custom_cols:
            ## Third, set indicators for independent no-split columns ##
            if col in self.custom_indep_nosplit_cols:
                # If parameter associated with column is constant over time
                i += 1
                CC1_indicator[:, i] = cum_params + C[col]
                CC2_indicator[:, i] = cum_params + C[col]
                cum_params += self.custom_cols_dict[col]
        n_indep_nosplit_params = cum_params - n_dep_params - n_indep_split_params

        CC1 = csc_matrix((np.ones(ni * n_param_cols), (np.repeat(range(ni), n_param_cols), CC1_indicator.flatten())), shape=(ni, cum_params))
        CC2 = csc_matrix((np.ones(ni * n_param_cols), (np.repeat(range(ni), n_param_cols), CC2_indicator.flatten())), shape=(ni, cum_params))

        # Transition probability matrix
        GG1 = csc_matrix((np.ones(ni), (range(ni), G1)), shape=(ni, nk))
        GG2 = csc_matrix((np.ones(ni), (range(ni), G2)), shape=(ni, nk))
        GG12 = csc_matrix((np.ones(ni), (range(ni), G1 + nk * G2)), shape=(ni, nk ** 2))
        del GG1, GG2

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
        d_prior = params['d_prior']

        # Constraints FIXME should this be nk or cum_params or something else?
        cons_a = QPConstrained(nl, nk, n_dep_params // nk, n_indep_split_params, n_indep_nosplit_params)
        cons_a.add_constraints_builtin(['stable_within_time', 'stable_across_time'], {'n_periods': 2})
        if len(params['cons_a']) > 0:
            cons_a.add_constraints_builtin(params['cons_a'][0], params['cons_a'][1])
        cons_s = QPConstrained(nl, nk, n_dep_params // nk, n_indep_split_params, n_indep_nosplit_params)
        cons_s.add_constraints_builtin(['stable_within_time', 'stable_across_time'], {'n_periods': 2})
        if len(params['cons_s']) > 0:
            cons_s.add_constraints_builtin(params['cons_s'][0], params['cons_s'][1])

        for iter in range(params['n_iters_movers']):

            # -------- E-Step ---------
            # We compute the posterior probabilities for each row
            # We iterate over the worker types, should not be be
            # too costly since the vector is quite large within each iteration
            ## Independent custom columns
            if len(self.custom_cols) > len(self.custom_dep_cols):
                if iter == 0:
                    A1_sum = np.zeros(ni)
                    A2_sum = np.zeros(ni)
                S1_sum_sq = np.zeros(ni)
                S2_sum_sq = np.zeros(ni)
                for col in self.custom_indep_split_cols:
                    # If parameter associated with column can change over time
                    if iter == 0:
                        A1_sum += A1_indep[col][C1[col]]
                        A2_sum += A2_indep[col][C2[col]]
                    S1_sum_sq += S1_indep[col][C1[col]] ** 2
                    S2_sum_sq += S2_indep[col][C2[col]] ** 2
                for col in self.custom_indep_nosplit_cols:
                    # If parameter associated with column is constant over time
                    if iter == 0:
                        A1_sum += A_indep[col][C[col]]
                        A2_sum += A_indep[col][C[col]]
                    S1_sum_sq += S_indep[col][C[col]] ** 2
                    S2_sum_sq += S_indep[col][C[col]] ** 2

                KK = G1 + nk * G2
                for l in range(nl):
                    idx_one = (l, G1, *[C1[col] for col in self.custom_dep_cols])
                    idx_two = (l, G2, *[C2[col] for col in self.custom_dep_cols])
                    lp1 = lognormpdf(Y1, A1_sum + A1[idx_one], np.sqrt(S1_sum_sq + S1[idx_one] ** 2))
                    lp2 = lognormpdf(Y2, A2_sum + A2[idx_two], np.sqrt(S2_sum_sq + S2[idx_two] ** 2))
                    lp[:, l] = np.log(pk1[KK, l]) + lp1 + lp2
            else:
                KK = G1 + nk * G2
                for l in range(nl):
                    idx_one = (l, G1, *[C1[col] for col in self.custom_dep_cols])
                    idx_two = (l, G2, *[C2[col] for col in self.custom_dep_cols])
                    lp1 = lognormpdf(Y1, A1[idx_one], S1[idx_one])
                    lp2 = lognormpdf(Y2, A2[idx_two], S2[idx_two])
                    lp[:, l] = np.log(pk1[KK, l]) + lp1 + lp2
            del idx_one, idx_two

            # We compute log sum exp to get likelihoods and probabilities
            qi = np.exp(lp.T - logsumexp(lp, axis=1)).T
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

            # --------- M-step ----------
            # For now we run a simple ols, however later we
            # want to add constraints!
            # see https://scaron.info/blog/quadratic-programming-in-python.html

            # The regression has 2 * nl * nk parameters and nl * ni rows
            # We do not necessarily want to construct the duplicated data by nl
            # Instead we will construct X'X and X'Y by looping over nl
            # We also note that X'X is block diagonal with 2*nl matrices of dimensions nk^2
            ts = nl * cum_params # Shift for period 2
            # ts_indep = # Shift for period 2 for independent control variables
            XwXd = np.zeros(shape=2 * ts) # Only store the diagonal
            if params['update_a']:
                XwY = np.zeros(shape=2 * ts)
            for l in range(nl):
                # (We might be better off trying this within numba or something)
                l_dep_index, r_dep_index = l * n_dep_params, (l + 1) * n_dep_params
                l_indep_split_index, r_indep_split_index = nl * n_dep_params + l * n_indep_split_params, nl * n_dep_params + (l + 1) * n_indep_split_params
                l_indep_nosplit_index, r_indep_nosplit_index = nl * (n_dep_params + n_indep_split_params) + l * n_indep_nosplit_params, nl * (n_dep_params + n_indep_split_params) + (l + 1) * n_indep_nosplit_params
                ## Compute shared terms ##
                # Variances
                idx_one = (l, G1, *[C1[col] for col in self.custom_dep_cols])
                idx_two = (l, G2, *[C2[col] for col in self.custom_dep_cols])
                S_l1 = S1[idx_one]
                S_l2 = S2[idx_two]
                if len(self.custom_cols) > len(self.custom_dep_cols):
                    S_l1 = np.sqrt(S1_sum_sq + S_l1 ** 2)
                    S_l2 = np.sqrt(S2_sum_sq + S_l2 ** 2)
                del idx_one, idx_two
                # Shared weighted terms
                CC1_weighted = CC1.T @ diags(qi[:, l] / S_l1)
                CC2_weighted = CC2.T @ diags(qi[:, l] / S_l2)
                ## Compute XwXd terms ##
                # Split dependent and independent (put dependent at beginning, independent at end)
                XwXd_1 = (CC1_weighted @ CC1).diagonal()
                XwXd_2 = (CC2_weighted @ CC2).diagonal()
                XwXd[l_dep_index: r_dep_index] = XwXd_1[: n_dep_params]
                XwXd[l_indep_split_index: r_indep_split_index] = XwXd_1[n_dep_params: n_dep_params + n_indep_split_params]
                XwXd[l_indep_nosplit_index: r_indep_nosplit_index] = XwXd_1[n_dep_params + n_indep_split_params:]
                XwXd[ts + l_dep_index: ts + r_dep_index] = XwXd_2[: n_dep_params]
                XwXd[ts + l_indep_split_index: ts + r_indep_split_index] = XwXd_2[n_dep_params: n_dep_params + n_indep_split_params]
                XwXd[ts + l_indep_nosplit_index: ts + r_indep_nosplit_index] = XwXd_2[n_dep_params + n_indep_split_params:]
                if params['update_a']:
                    ## Compute XwY terms ##
                    # Split dependent and independent (put dependent at beginning, independent at end)
                    XwY_1 = CC1_weighted @ Y1
                    XwY_2 = CC2_weighted @ Y2
                    XwY[l_dep_index: r_dep_index] = XwY_1[: n_dep_params]
                    XwY[l_indep_split_index: r_indep_split_index] = XwY_1[n_dep_params: n_dep_params + n_indep_split_params]
                    XwY[l_indep_nosplit_index: r_indep_nosplit_index] = XwY_1[n_dep_params + n_indep_split_params:]
                    XwY[ts + l_dep_index: ts + r_dep_index] = XwY_2[: n_dep_params]
                    XwY[ts + l_indep_split_index: ts + r_indep_split_index] = XwY_2[n_dep_params: n_dep_params + n_indep_split_params]
                    XwY[ts + l_indep_nosplit_index: ts + r_indep_nosplit_index] = XwY_2[n_dep_params + n_indep_split_params:]
            del XwXd_1, XwXd_2

            # We solve the system to get all the parameters (note: this won't work if XwX is sparse)
            # print('A1 before:')
            # print(A1)
            # print('A2 before:')
            # print(A2)
            # print('S1 before:')
            # print(S1)
            # print('S2 before:')
            # print(S2)
            # print('A1_indep before:')
            # print(A1_indep)
            # print('A2_indep before:')
            # print(A2_indep)
            # print('A_indep before:')
            # print(A_indep)
            XwX = np.diag(XwXd)
            if params['update_a']:
                try:
                    cons_a.solve(XwX, -XwY)
                    res_a1, res_a2 = cons_a.res[: len(cons_a.res) // 2], cons_a.res[len(cons_a.res) // 2:]
                    A1 = np.reshape(res_a1[: nl * n_dep_params], self.dims)
                    A2 = np.reshape(res_a2[: nl * n_dep_params], self.dims)
                    params_count = nl * n_dep_params
                    for col in self.custom_indep_split_cols:
                        # If parameter associated with column can change over time
                        n_col_params = self.custom_cols_dict[col]
                        A1_indep[col] = res_a1[params_count: params_count + n_col_params]
                        A2_indep[col] = res_a2[params_count: params_count + n_col_params]
                        params_count += n_col_params
                    for col in self.custom_indep_nosplit_cols:
                        # If parameter associated with column is constant over time
                        n_col_params = self.custom_cols_dict[col]
                        A_indep[col] = res_a1[params_count: params_count + n_col_params]
                        params_count += n_col_params
                except ValueError as e:
                    # If constraints inconsistent, keep A1 and A2 the same
                    if params['verbose'] in [1, 2]:
                        print(f'{e}, passing 1')
                    stop
                    pass

            if params['update_s']:
                # Next we extract the variances
                XwS = np.zeros(shape=2 * ts)
                if len(self.custom_cols) > len(self.custom_dep_cols):
                    A1_sum = np.zeros(ni)
                    A2_sum = np.zeros(ni)
                    for col in self.custom_indep_split_cols:
                        # If parameter associated with column can change over time
                        A1_sum += A1_indep[col][C1[col]]
                        A2_sum += A2_indep[col][C2[col]]
                    for col in self.custom_indep_nosplit_cols:
                        # If parameter associated with column is constant over time
                        A1_sum += A_indep[col][C[col]]
                        A2_sum += A_indep[col][C[col]]
                for l in range(nl):
                    l_dep_index, r_dep_index = l * n_dep_params, (l + 1) * n_dep_params
                    l_indep_split_index, r_indep_split_index = nl * n_dep_params + l * n_indep_split_params, nl * n_dep_params + (l + 1) * n_indep_split_params
                    l_indep_nosplit_index, r_indep_nosplit_index = nl * (n_dep_params + n_indep_split_params) + l * n_indep_nosplit_params, nl * (n_dep_params + n_indep_split_params) + (l + 1) * n_indep_nosplit_params
                    # Means and variances
                    idx_one = (l, G1, *[C1[col] for col in self.custom_dep_cols])
                    idx_two = (l, G2, *[C2[col] for col in self.custom_dep_cols])
                    A_l1 = A1[idx_one]
                    A_l2 = A2[idx_two]
                    S_l1 = S1[idx_one]
                    S_l2 = S2[idx_two]
                    if len(self.custom_cols) > len(self.custom_dep_cols):
                        A_l1 += A1_sum
                        A_l2 += A2_sum
                        S_l1 = np.sqrt(S1_sum_sq + S_l1 ** 2)
                        S_l2 = np.sqrt(S2_sum_sq + S_l2 ** 2)
                    del idx_one, idx_two
                    # Split dependent and independent (put dependent at beginning, independent at end)
                    XwS_1 = CC1.T @ diags(qi[:, l] / S_l1) @ ((Y1 - A_l1) ** 2)
                    XwS_2 = CC2.T @ diags(qi[:, l] / S_l2) @ ((Y2 - A_l2) ** 2)
                    XwS[l_dep_index: r_dep_index] = XwS_1[: n_dep_params]
                    XwS[l_indep_split_index: r_indep_split_index] = XwS_1[n_dep_params: n_dep_params + n_indep_split_params]
                    XwS[l_indep_nosplit_index: r_indep_nosplit_index] = XwS_1[n_dep_params + n_indep_split_params:]
                    XwS[ts + l_dep_index: ts + r_dep_index] = XwS_2[: n_dep_params]
                    XwS[ts + l_indep_split_index: ts + r_indep_split_index] = XwS_2[n_dep_params: n_dep_params + n_indep_split_params]
                    XwS[ts + l_indep_nosplit_index: ts + r_indep_nosplit_index] = XwS_2[n_dep_params + n_indep_split_params:]
                del XwS_1, XwS_2

                try:
                    cons_s.solve(XwX, -XwS)
                    # FIXME using np.maximum(0, S) to set lower bound on variance
                    res_s1, res_s2 = np.sqrt(cons_s.res[: len(cons_s.res) // 2]), np.sqrt(cons_s.res[len(cons_s.res) // 2:])
                    S1 = np.reshape(res_s1[: nl * n_dep_params], self.dims)
                    S2 = np.reshape(res_s2[: nl * n_dep_params], self.dims)
                    params_count = nl * n_dep_params
                    for col in self.custom_indep_split_cols:
                        # If parameter associated with column can change over time
                        n_col_params = self.custom_cols_dict[col]
                        S1_indep[col] = res_s1[params_count: params_count + n_col_params]
                        S2_indep[col] = res_s2[params_count: params_count + n_col_params]
                        params_count += n_col_params
                    for col in self.custom_indep_nosplit_cols:
                        # If parameter associated with column is constant over time
                        n_col_params = self.custom_cols_dict[col]
                        S_indep[col] = res_s1[params_count: params_count + n_col_params]
                        params_count += n_col_params
                except ValueError as e:
                    # If constraints inconsistent, keep S1 and S2 the same
                    if params['verbose'] in [1, 2]:
                        print(f'{e}, passing 2')
                    stop
                    pass
            print('res a:')
            print(cons_a.res)
            print('res s:')
            print(cons_s.res)
            # print('A1 after:')
            # print(A1)
            # print('A2 after:')
            # print(A2)
            # print('S1 after:')
            # print(S1)
            # print('S2 after:')
            # print(S2)
            # print('A1_indep after:')
            # print(A1_indep)
            # print('A2_indep after:')
            # print(A2_indep)
            # print('A_indep after:')
            # print(A_indep)
            stop
            if params['update_pk1']:
                pk1 = GG12.T @ qi
                # for l in range(nl):
                #     pk1[:, l] = GG12.T * qi[:, l]
                # Normalize rows to sum to 1, and add dirichlet prior
                # pk1 = np.maximum(pk1, 0) # FIXME this isn't okay, but it prevents a bug with NaNs
                pk1 += d_prior - 1
                pk1 = (pk1.T / np.sum(pk1, axis=1).T).T

        self.A1, self.A2, self.S1, self.S2 = A1, A2, S1, S2
        self.A1_indep, self.A2_indep, self.A_indep, self.S1_indep, self.S2_indep, self.S_indep = A1_indep, A2_indep, A_indep, S1_indep, S2_indep, S_indep
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
        A1, S1 = self.A1, self.S1
        A1_indep, A_indep, S1_indep, S_indep = self.A1_indep, self.A_indep, self.S1_indep, self.S_indep

        # Store wage outcomes and groups
        Y1 = sdata['y1'].to_numpy()
        G1 = sdata['g1'].to_numpy().astype(int)
        # Control variables
        C = {}
        C1 = {}
        for col in self.custom_cols:
            # Get subcolumns associated with col
            subcols = to_list(sdata.col_reference_dict[col])
            if col in self.custom_indep_nosplit_cols:
                # If parameter associated with column is constant over time
                if len(subcols) != 1:
                    raise NotImplementedError(f'No-split general columns must have one associated column, but {col!r} does not.')
                C[col] = sdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
            else:
                # If parameter associated with column can change over time
                if len(subcols) != 2:
                    raise NotImplementedError(f'Split general columns must have two associated columns, but {col!r} does not.')
                C1[col] = sdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)

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

        GG1 = csc_matrix((np.ones(ni), (range(ni), G1)), shape=(ni, nk))

        if len(self.custom_cols) > len(self.custom_dep_cols):
            # Custom columns
            A1_sum = np.zeros(ni)
            S1_sum_sq = np.zeros(ni)
            for col in self.custom_indep_nosplit_cols:
                # If parameter associated with column is constant over time
                A1_sum += A_indep[col][C[col]]
                S1_sum_sq += S_indep[col][C[col]] ** 2
            for col in self.custom_indep_split_cols:
                # If parameter associated with column can change over time
                A1_sum += A1_indep[col][C1[col]]
                S1_sum_sq += S1_indep[col][C1[col]] ** 2

            for l in range(nl):
                idx_l = (l, G1, *[C1[col] for col in self.custom_dep_cols])
                lp_stable[:, l] = lognormpdf(Y1, A1_sum + A1[idx_l], np.sqrt(S1_sum_sq + S1[idx_l] ** 2))
        else:
            for l in range(nl):
                idx_l = (l, G1, *[C1[col] for col in self.custom_dep_cols])
                lp_stable[:, l] = lognormpdf(Y1, A1[idx_l], S1[idx_l])
        del idx_l

        for iter in range(params['n_iters_stayers']):

            # -------- E-Step ---------
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

            # --------- M-step ----------
            pk0 = GG1.T @ qi
            # for l in range(nl):
            #     pk0[:, l] = GG1.T * qi[:, l]
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
        # First, simulate parameters but keep A fixed
        # Second, use estimated parameters as starting point to run with A constrained to be linear
        # Finally use estimated parameters as starting point to run without constraints
        # self.reset_params() # New parameter guesses
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
        self.params['cons_a'] = (['lin'], {'n_periods': 2})
        if self.params['verbose'] in [1, 2]:
            print('Running constrained movers')
        self.fit_movers(jdata, compute_NNm=False)
        ##### Loop 3 #####
        # Remove constraints
        self.params['cons_a'] = ([], {})
        if self.params['verbose'] in [1, 2]:
            print('Running unconstrained movers')
        self.fit_movers(jdata, compute_NNm=compute_NNm)
        ##### Compute connectedness #####
        self.compute_connectedness_measure()

    def fit_A(self, jdata, compute_NNm=True):
        '''
        Run fit_movers() and update A while keeping S and pk1 fixed.

        Arguments:
            jdata (Pandas DataFrame): movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # New parameter guesses
        # self.reset_params()
        self.params['update_a'] = True
        self.params['update_s'] = False
        self.params['update_pk1'] = False
        self.params['cons_a'] = ([], {})
        if self.params['verbose'] in [1, 2]:
            print('Running fit_A')
        self.fit_movers(jdata, compute_NNm=compute_NNm)

    def fit_S(self, jdata, compute_NNm=True):
        '''
        Run fit_movers() and update S while keeping A and pk1 fixed.

        Arguments:
            jdata (Pandas DataFrame): movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # New parameter guesses
        # self.reset_params()
        self.params['update_a'] = False
        self.params['update_s'] = True
        self.params['update_pk1'] = False
        self.params['cons_a'] = ([], {})
        if self.params['verbose'] in [1, 2]:
            print('Running fit_S')
        self.fit_movers(jdata, compute_NNm=compute_NNm)

    def fit_pk(self, jdata, compute_NNm=True):
        '''
        Run fit_movers() and update pk1 while keeping A and S fixed.

        Arguments:
            jdata (Pandas DataFrame): movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # New parameter guesses
        # self.reset_params()
        self.params['update_a'] = False
        self.params['update_s'] = False
        self.params['update_pk1'] = True
        self.params['cons_a'] = ([], {})
        if self.params['verbose'] in [1, 2]:
            print('Running fit_pk')
        self.fit_movers(jdata, compute_NNm=compute_NNm)

    def _sort_matrices(self):
        '''
        Sort arrays by cluster means.
        '''
        n_dims = len(self.A1.shape)
        nk = self.nk
        ## Sort worker effects ##
        worker_effect_order = np.mean(self.A1, axis=tuple(range(n_dims)[1:])).argsort()
        self.A1 = self.A1[worker_effect_order, :]
        self.A2 = self.A2[worker_effect_order, :]
        self.S1 = self.S1[worker_effect_order, :]
        self.S2 = self.S2[worker_effect_order, :]
        self.pk1 = self.pk1[:, worker_effect_order]
        self.pk0 = self.pk0[:, worker_effect_order]
        ## Sort firm effects ##
        firm_effect_order = np.mean(self.A1, axis=(0, *range(n_dims)[2:])).argsort()
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
        # Collapse by mean over control variables
        # FIXME should this weight by number of observations per group?
        n_dims = len(self.A1.shape)
        sorted_A1 = np.mean(self.A1, axis=tuple(np.arange(n_dims)[2:]))
        # Sort A1 by average effect over firms
        sorted_A1 = sorted_A1.T[np.mean(sorted_A1.T, axis=1).argsort()].T
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

class BLMEstimator:
    '''
    Class for solving the BLM model using multiple sets of starting values.

    Arguments:
        params (ParamsDict): dictionary of parameters for BLM estimation. Run tw.blm_params().describe_all() for descriptions of all valid parameters.
    '''

    def __init__(self, params=blm_params()):
        self.params = params
        self.model = None # No initial model
        self.liks_high = None # No likelihoods yet
        self.connectedness_high = None # No connectedness yet
        self.liks_low = None # No likelihoods yet
        self.connectedness_low = None # No connectedness yet
        self.liks_all = None # No paths of likelihoods yet

    def _sim_model(self, jdata, rng=np.random.default_rng(None)):
        '''
        Generate model and run fit_movers_cstr_uncstr() given parameters.

        Arguments:
            jdata (Pandas DataFrame): movers
            rng (np.random.Generator): NumPy random number generator
        '''
        model = BLMModel(self.params, rng)
        model.fit_movers_cstr_uncstr(jdata)
        return model

    def fit(self, jdata, sdata, n_init=10, n_best=1, ncore=1, rng=np.random.default_rng(None)):
        '''
        EM model for movers and stayers.

        Arguments:
            jdata (Pandas DataFrame): movers
            sdata (Pandas DataFrame): stayers
            n_init (int): number of starting values
            n_best (int): take the n_best estimates with the highest likelihoods, and then take the estimate with the highest connectedness
            ncore (int): number of cores for multiprocessing
            rng (np.random.Generator): NumPy random number generator
        '''
        # Run sim_model()
        if ncore > 1:
            ## Multiprocessing
            # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
            seeds = rng.bit_generator._seed_seq.spawn(n_init)
            with Pool(processes=ncore) as pool:
                sim_model_lst = pool.starmap(self._sim_model, tqdm([(jdata, np.random.default_rng(seed)) for seed in seeds], total=n_init))
        else:
            sim_model_lst = itertools.starmap(self._sim_model, tqdm([(jdata, rng) for _ in range(n_init)], total=n_init))

        # Sort by likelihoods
        sorted_zipped_models = sorted([(model.lik1, model) for model in sim_model_lst], reverse=True, key=lambda a: a[0])
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
        self.model._sort_matrices()

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
