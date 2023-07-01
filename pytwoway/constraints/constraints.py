'''
Classes for generating quadratic programming constraints for BLM.

All classes are for solving a quadratic programming model of the following form:
        min_x(1/2 x.T @ P @ x + q.T @ x)
        s.t.    Gx <= h
                Ax = b
'''
'''
NOTE: use classes rather than nested functions because nested functions cannot be pickled (source: https://stackoverflow.com/a/12022055/17333120).

NOTE: parameters are ordered with precedence of (time, worker type, firm type). As an example, if nt=2, nl=2, and nk=3, then the parameters will be ordered as follows:
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2)
'''
import numpy as np
from qpsolvers import solve_qp
from scipy.optimize import linprog
from scipy.sparse import csc_matrix
from bipartitepandas.util import to_list

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
        self.G = None
        # Inequality constraint bound
        self.h = None
        # Equality constraint matrix
        self.A = None
        # Equality constraint bound
        self.b = None
        # Lower bound
        self.lb = None
        # Upper bound
        self.ub = None
        # Columns to drop (these columns are normalized to 0)
        self.drop_cols = None
        self.drop_cols_mask = None
        # Bounds (for linprog)
        self.bounds = None

    def add_constraints(self, constraints):
        '''
        Add a built-in constraint.

        Arguments:
            constraints (object or list of objects): constraint objects with class method ._get_constraints() that defines constraints to add
        '''
        for constraint in to_list(constraints):
            self._add_constraint(**constraint._get_constraints(nl=self.nl, nk=self.nk))

    def _add_constraint(self, G=None, h=None, A=None, b=None, lb=None, ub=None, drop_cols=None):
        '''
        Manually add a constraint. If setting inequality constraints, must set both G and h to have the same dimension 0. If setting equality constraints, must set both A and b to have the same dimension 0.

        Arguments:
            G (NumPy Array or None): inequality constraint matrix; None is equivalent to np.array([])
            h (NumPy Array or None): inequality constraint bound; None is equivalent to np.array([])
            A (NumPy Array or None): equality constraint matrix; None is equivalent to np.array([])
            b (NumPy Array or None): equality constraint bound; None is equivalent to np.array([])
            lb (NumPy Array or None): lower bound; None means there is no lower bound
            ub (NumPy Array or None): upper bound; None means there is no upper bound
            drop_cols (list of int or None): indices of columns to drop (these columns are normalized to 0); None is equivalent to []
        '''
        if G is not None:
            # If inequality constraints
            if self.G is not None:
                self.G = np.concatenate((self.G, G), axis=0)
                self.h = np.concatenate((self.h, h), axis=0)
            else:
                self.G = G
                self.h = h
        if A is not None:
            # If equality constraints
            if self.A is not None:
                self.A = np.concatenate((self.A, A), axis=0)
                self.b = np.concatenate((self.b, b), axis=0)
            else:
                self.A = A
                self.b = b
        if lb is not None:
            # If lower bound constraints
            if self.lb is not None:
                self.lb = np.maximum(self.lb, lb)
            else:
                self.lb = lb
        if ub is not None:
            # If upper bound constraints
            if self.ub is not None:
                self.ub = np.minimum(self.ub, ub)
            else:
                self.ub = ub
        if drop_cols is not None:
            # If columns to normalize
            if self.drop_cols is not None:
                self.drop_cols += drop_cols
            else:
                self.drop_cols = drop_cols
            # np.unique also sorts
            self.drop_cols = list(np.unique(self.drop_cols))

    def pad(self, l=0, r=0):
        '''
        Add padding to the left and/or right of C matrix.

        Arguments:
            l (int): how many columns to add on left
            r (int): how many columns to add on right
        '''
        if self.G is not None:
            self.G = np.concatenate((
                    np.zeros(shape=(self.G.shape[0], l)),
                    self.G,
                    np.zeros(shape=(self.G.shape[0], r)),
                ), axis=1)
        else:
            self.G = np.zeros(shape=l + r)
        if self.A is not None:
            self.A = np.concatenate((
                    np.zeros(shape=(self.A.shape[0], l)),
                    self.A,
                    np.zeros(shape=(self.A.shape[0], r)),
                ), axis=1)
        else:
            self.A = np.zeros(shape=l + r)
        if self.drop_cols is not None:
            for i in range(len(self.drop_cols)):
                self.drop_cols[i] += l

    def clear_constraints(self, inequality=True, equality=True, bounds=True, drop_cols=True, drop_cols_mask=True):
        '''
        Clear constraints.

        Arguments:
            inequality (bool): if True, clear inequality constraints
            equality (bool): if True, clear equality constraints
            bounds (bool): if True, clear lower and upper bound constraints
            drop_cols (bool): if True, clear columns to normalize to 0
            drop_cols_mask (bool): if True, clear mask for columns to normalize to 0
        '''
        if inequality:
            self.G = None
            self.h = None
        if equality:
            self.A = None
            self.b = None
        if bounds:
            self.lb = None
            self.ub = None
            self.bounds = None
        if drop_cols:
            self.drop_cols = None
        if drop_cols_mask:
            self.drop_cols_mask = None

    def check_feasible(self):
        '''
        Check that constraints are feasible.

        Returns:
            (bool): True if constraints feasible, False otherwise
        '''
        # -----  Simulate an OLS -----
        rng = np.random.default_rng(None)
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
        cons.lb = self.lb
        cons.ub = self.ub
        cons.drop_cols = self.drop_cols
        cons.drop_cols_mask = self.drop_cols_mask

        # ----- Map to qpsolvers -----
        P = M.T @ M
        q = - M.T @ Y

        # ----- Run solver -----
        cons.solve(P, q)

        return cons.res is not None

    def solve(self, P, q, solver='quadprog', integrality=0, verbose=False, **kwargs):
        '''
        Solve a quadratic programming model of the following form:
            min_x(1/2 x.T @ P @ x + q.T @ x)
            s.t.    Gx <= h
                    Ax = b

        Arguments:
            P (NumPy Array): P in quadratic programming problem
            q (NumPy Array): q in quadratic programming problem
            solver (str): solver to use
            integrality (int): (for use with linprog) indicates the type of integrality constraint on each decision variable. 0: Continuous variable, no integrality constraint; 1: Integer variable, decision variable must be an integer within bounds; 2: Semi-continuous variable, decision variable must be within bounds or take value 0; 3: Semi-integer variable, decision variable must be an integer within bounds or take value 0.
            verbose (bool): if True, print extra output
            **kwargs: parameters for solver

        Returns:
            (NumPy Array): x that solves quadratic programming problem
        '''
        G, h = self.G, self.h
        A, b = self.A, self.b
        lb, ub = self.lb, self.ub
        drop_cols, drop_cols_mask = self.drop_cols, self.drop_cols_mask

        if solver in ['ecos', 'gurobi', 'mosek', 'osqp', 'qpswift', 'scs']:
            ## Make matrices sparse (if using sparse solver) ##
            if (G is not None) and not isinstance(G, csc_matrix):
                G = csc_matrix(G)
                self.G = G
            if (A is not None) and not isinstance(A, csc_matrix):
                A = csc_matrix(A)
                self.A = A

        if drop_cols is not None:
            # Drop columns that will be normalized to 0
            n_params = P.shape[0]
            if drop_cols_mask is None:
                # Source: https://stackoverflow.com/a/12518492/17333120
                drop_cols_mask = np.zeros(n_params)
                drop_cols_mask[drop_cols] = 1
                drop_cols_mask = drop_cols_mask.astype(bool)
            P = P[~drop_cols_mask, :][:, ~drop_cols_mask]
            q = q[~drop_cols_mask]
            if G is not None:
                G = G[:, ~drop_cols_mask]
            if A is not None:
                A = A[:, ~drop_cols_mask]
            if lb is not None:
                lb = lb[~drop_cols_mask]
            if ub is not None:
                ub = ub[~drop_cols_mask]

        if solver == 'linprog':
            bounds = self.bounds
            if bounds is None:
                if (lb is None) and (ub is None):
                    bounds = (None, None)
                elif lb is not None:
                    bounds = list(zip(lb, [None] * len(lb)))
                elif ub is not None:
                    bounds = list(zip([None] * len(ub), ub))
                else:
                    bounds = list(zip(lb, ub))
                self.bounds = bounds
            self.res = linprog(c=q, A_ub=G, b_ub=h, A_eq=A, b_eq=b, bounds=bounds, integrality=integrality).x
        else:
            self.res = solve_qp(P=P, q=q, G=G, h=h, A=A, b=b, lb=lb, ub=ub, solver=solver, verbose=verbose, **kwargs)

        if drop_cols is not None:
            # Add back in zeros for normalized parameters
            res = np.zeros(n_params)
            res[~drop_cols_mask] = self.res
            self.res = res

class Linear():
    '''
    Generate BLM constraints so that for a fixed firm type, worker types effects must change linearly.

    Arguments:
        nnt (int or list of ints or None): time periods to constrain. This should be set to 0 if Linear() is being used in conjunction with Stationary(). None is equivalent to range(nt).
        nt (int): number of time periods
        dynamic (bool): if True, using dynamic BLM estimator
    '''

    def __init__(self, nnt=None, nt=2, dynamic=False):
        if nnt is None:
            self.nnt = range(nt)
        else:
            self.nnt = to_list(nnt)
        self.nt = nt
        self.dynamic = dynamic

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': None, 'h': None, 'A': A, 'b': b}, where G, h, A, and b are defined in the quadratic programming model
        '''
        ## Unpack parameters ##
        nnt, nt, dynamic = self.nnt, self.nt, self.dynamic

        ## Initialize variables ##
        # A starts with 4 dimensions
        A = np.zeros(shape=(len(nnt) * (nl - 2) * nk, nt, nl, nk))
        i = 0

        ## Generate constraints ##
        for period in nnt:
            ## Iterate over periods ##
            for k in range(nk):
                ## Iterate over firm types ##
                for l in range(nl - 2):
                    ## Iterate over worker types ##
                    A[i, period, l, k] = 1
                    A[i, period, l + 1, k] = -2
                    A[i, period, l + 2, k] = 1
                    i += 1

        if dynamic:
            # Use dynamic BLM dimensions (i, l, period, k)
            A = A.transpose((0, 2, 1, 3))

        # Reshape A to 2 dimensions
        A = A.reshape((len(nnt) * (nl - 2) * nk, nt * nl * nk))

        b = - np.zeros(shape=A.shape[0])

        return {'G': None, 'h': None, 'A': A, 'b': b}

class LinearAdditive():
    '''
    Generate BLM constraints so that for a fixed firm type, worker types effects must change linear-additively.

    Arguments:
        nnt (int or list of ints or None): time periods to constrain. This should be set to 1 if LinearAdditive() is being used in conjunction with Stationary(). None is equivalent to range(nt).
        nt (int): number of time periods
        dynamic (bool): if True, using dynamic BLM estimator
    '''

    def __init__(self, nnt=None, nt=2, dynamic=False):
        if nnt is None:
            self.nnt = range(nt)
        else:
            self.nnt = to_list(nnt)
        self.nt = nt
        self.dynamic = dynamic

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': None, 'h': None, 'A': A, 'b': b}, where G, h, A, and b are defined in the quadratic programming model
        '''
        ## Unpack parameters ##
        nnt, nt, dynamic = self.nnt, self.nt, self.dynamic

        ## Initialize variables ##
        # A starts with 4 dimensions
        A = np.zeros(shape=(len(nnt) * ((nl - 2) * nk + (nk - 1)), nt, nl, nk))
        i = 0

        ### Generate constraints ###
        ## Linear ##
        for period in nnt:
            ## Iterate over periods ##
            for k in range(nk):
                ## Iterate over firm types ##
                for l in range(nl - 2):
                    ## Iterate over worker types ##
                    A[i, period, l, k] = 1
                    A[i, period, l + 1, k] = -2
                    A[i, period, l + 2, k] = 1
                    i += 1

        ## Additive ##
        for period in nnt:
            ## Iterate over periods ##
            for k in range(nk - 1):
                ## Iterate over firm types ##
                A[i, period, 0, k] = 1
                A[i, period, 1, k] = -1
                A[i, period, 0, k + 1] = -1
                A[i, period, 1, k + 1] = 1
                i += 1

        if dynamic:
            # Use dynamic BLM dimensions (i, l, period, k)
            A = A.transpose((0, 2, 1, 3))

        # Reshape A to 2 dimensions
        A = A.reshape((len(nnt) * ((nl - 2) * nk + (nk - 1)), nt * nl * nk))

        b = - np.zeros(shape=A.shape[0])

        return {'G': None, 'h': None, 'A': A, 'b': b}

class Monotonic():
    '''
    Generate BLM constraints so that for a fixed firm type, worker types effects must increase (or decrease) monotonically.

    Arguments:
        md (float): minimum difference between consecutive types
        increasing (bool): if True, monotonic increasing; if False, monotonic decreasing
        nt (int): number of time periods
        dynamic (bool): if True, using dynamic BLM estimator
    '''

    def __init__(self, md=0, increasing=True, nt=2, dynamic=False):
        self.md = md
        self.increasing = increasing
        self.nt = nt
        self.dynamic = dynamic

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': G, 'h': h, 'A': None, 'b': None}, where G, h, A, and b are defined in the quadratic programming model
        '''
        ## Unpack parameters ##
        md, increasing, nt, dynamic = self.md, self.increasing, self.nt, self.dynamic

        ## Initialize variables ##
        # G starts with 4 dimensions
        G = np.zeros(shape=(nt * (nl - 1) * nk, nt, nl, nk))
        i = 0

        ## Generate constraints ##
        for period in range(nt):
            ## Iterate over periods ##
            for k in range(nk):
                ## Iterate over firm types ##
                for l in range(nl - 1):
                    ## Iterate over worker types ##
                    G[i, period, l, k] = 1
                    G[i, period, l + 1, k] = -1
                    i += 1

        if dynamic:
            # Use dynamic BLM dimensions (i, l, period, k)
            G = G.transpose((0, 2, 1, 3))

        # Reshape G to 2 dimensions
        G = G.reshape((nt * (nl - 1) * nk, nt * nl * nk))

        h = - md * np.ones(shape=G.shape[0])

        if not increasing:
            G *= -1

        return {'G': G, 'h': h, 'A': None, 'b': None}

class MonotonicMean():
    '''
    Generate BLM constraints so that the mean of worker types effects over all firm types must increase (or decrease) monotonically.

    Arguments:
        md (float): minimum difference between consecutive types
        increasing (bool): if True, monotonic increasing; if False, monotonic decreasing
        cross_period_mean (bool): if True, rather than checking means are monotonic for each period separately, consider the mean worker effects over all periods jointly
        nnt (int or list of ints or None): time periods to constrain; None is equivalent to range(nt)
        nt (int): number of time periods
        dynamic (bool): if True, using dynamic BLM estimator
    '''

    def __init__(self, md=0, increasing=True, cross_period_mean=False, nnt=None, nt=2, dynamic=False):
        self.md = md
        self.increasing = increasing
        self.cross_period_mean = cross_period_mean
        if nnt is None:
            self.nnt = range(nt)
        else:
            self.nnt = to_list(nnt)
        self.nt = nt
        self.dynamic = dynamic

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': G, 'h': h, 'A': None, 'b': None}, where G, h, A, and b are defined in the quadratic programming model
        '''
        ## Unpack parameters ##
        md, increasing, cross_period_mean, nnt, nt, dynamic = self.md, self.increasing, self.cross_period_mean, self.nnt, self.nt, self.dynamic

        ## Initialize variables ##
        # G starts with 4 dimensions
        if cross_period_mean:
            G = np.zeros(shape=((nl - 1), nt, nl, nk))
        else:
            G = np.zeros(shape=(len(nnt) * (nl - 1), nt, nl, nk))
        i = 0

        ## Generate constraints ##
        for period in nnt:
            ## Iterate over periods ##
            if cross_period_mean:
                i = 0
            for l in range(nl - 1):
                ## Iterate over worker types ##
                for k in range(nk):
                    ## Iterate over firm types ##
                    if cross_period_mean:
                        G[i, period, l, k] = (1 / (len(nnt) * nk))
                        G[i, period, l + 1, k] = - (1 / (len(nnt) * nk))
                    else:
                        G[i, period, l, k] = (1 / nk)
                        G[i, period, l + 1, k] = - (1 / nk)
                i += 1

        if dynamic:
            # Use dynamic BLM dimensions (i, l, period, k)
            G = G.transpose((0, 2, 1, 3))

        # Reshape G to 2 dimensions
        if cross_period_mean:
            G = G.reshape(((nl - 1), nt * nl * nk))
        else:
            G = G.reshape((len(nnt) * (nl - 1), nt * nl * nk))

        h = - md * np.ones(shape=G.shape[0])

        if not increasing:
            G *= -1

        return {'G': G, 'h': h, 'A': None, 'b': None}

class MinFirmType():
    '''
    Generate BLM constraints so that the mean of a given firm type is less than or equal to the mean of every other firm type.

    Arguments:
        min_firm_type (int): minimum firm type
        md (float): minimum difference between the lowest firm type and other types
        is_min (bool): if True, constraint firm type to be the lowest firm type; if False, constraint it to be the highest firm type
        cross_period_mean (bool): if True, rather than checking means are monotonic for each period separately, consider the mean worker effects over all periods jointly
        nnt (int or list of ints or None): time periods to constrain; None is equivalent to range(nt)
        nt (int): number of time periods
        dynamic (bool): if True, using dynamic BLM estimator
    '''

    def __init__(self, min_firm_type, md=0, is_min=True, cross_period_mean=False, nnt=None, nt=2, dynamic=False):
        self.min_firm_type = min_firm_type
        self.md = md
        self.is_min = is_min
        self.cross_period_mean = cross_period_mean
        if nnt is None:
            self.nnt = range(nt)
        else:
            self.nnt = to_list(nnt)
        self.nt = nt
        self.dynamic = dynamic

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': G, 'h': h, 'A': None, 'b': None}, where G, h, A, and b are defined in the quadratic programming model
        '''
        ## Unpack parameters ##
        min_firm_type, md, is_min, cross_period_mean, nnt, nt, dynamic = self.min_firm_type, self.md, self.is_min, self.cross_period_mean, self.nnt, self.nt, self.dynamic

        ## Initialize variables ##
        # G starts with 4 dimensions
        if cross_period_mean:
            G = np.zeros(shape=((nk - 1), nt, nl, nk))
        else:
            G = np.zeros(shape=(len(nnt) * (nk - 1), nt, nl, nk))
        i = 0

        ## Generate constraints ##
        for period in nnt:
            ## Iterate over periods ##
            if cross_period_mean:
                i = 0
            for k in list(range(min_firm_type)) + list(range(min_firm_type + 1, nk)):
                ## Iterate over firm types ##
                for l in range(nl):
                    ## Iterate over worker types ##
                    # First, consider minimum firm type
                    if cross_period_mean:
                        G[i, period, l, min_firm_type] = (1 / (len(nnt) * nl))
                    else:
                        G[i, period, l, min_firm_type] = (1 / nl)
                    # Second, consider all other firm types
                    if cross_period_mean:
                        G[i, period, l, k] = - (1 / (len(nnt) * nl))
                    else:
                        G[i, period, l, k] = - (1 / nl)
                i += 1

        if dynamic:
            # Use dynamic BLM dimensions (i, l, period, k)
            G = G.transpose((0, 2, 1, 3))

        # Reshape G to 2 dimensions
        if cross_period_mean:
            G = np.reshape(G, ((nk - 1), nt * nl * nk))
        else:
            G = np.reshape(G, (len(nnt) * (nk - 1), nt * nl * nk))

        h = - md * np.ones(shape=G.shape[0])

        if not is_min:
            G *= -1

        return {'G': G, 'h': h, 'A': None, 'b': None}

class NoWorkerTypeInteraction():
    '''
    Generate BLM constraints so that for a fixed firm type, worker type effects must all be the same.

    Arguments:
        nnt (int or list of ints or None): time periods to constrain; None is equivalent to range(nt)
        nt (int): number of time periods
        dynamic (bool): if True, using dynamic BLM estimator
    '''

    def __init__(self, nnt=None, nt=2, dynamic=False):
        if nnt is None:
            self.nnt = range(nt)
        else:
            self.nnt = to_list(nnt)
        self.nt = nt
        self.dynamic = dynamic

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': None, 'h': None, 'A': A, 'b': b}, where G, h, A, and b are defined in the quadratic programming model
        '''
        ## Unpack parameters ##
        nnt, nt, dynamic = self.nnt, self.nt, self.dynamic

        ## Initialize variables ##
        # A starts with 4 dimensions
        A = np.zeros(shape=(len(nnt) * (nl - 1) * nk, nt, nl, nk))
        i = 0

        ## Generate constraints ##
        for period in nnt:
            ## Iterate over periods ##
            for k in range(nk):
                ## Iterate over firm types ##
                for l in range(nl - 1):
                    ## Iterate over worker types ##
                    A[i, period, l, k] = 1
                    A[i, period, l + 1, k] = -1
                    i += 1

        if dynamic:
            # Use dynamic BLM dimensions (i, l, period, k)
            A = A.transpose((0, 2, 1, 3))

        # Reshape A to 2 dimensions
        A = A.reshape((len(nnt) * (nl - 1) * nk, nt * nl * nk))

        b = - np.zeros(shape=A.shape[0])

        return {'G': None, 'h': None, 'A': A, 'b': b}

class NormalizeLowest():
    '''
    Generate BLM constraints so that the lowest worker-firm type pair has effect 0.

    Arguments:
        min_firm_type (int): lowest firm type
        cross_period_normalize (bool): if True, rather than normalizing for each period separately, normalize the mean of the lowest worker-firm type pair effect over all periods jointly
        nnt (int or list of ints or None): time periods to constrain; None is equivalent to range(nt)
        nt (int): number of time periods
        dynamic (bool): if True, using dynamic BLM estimator
    '''

    def __init__(self, min_firm_type, cross_period_normalize=False, nnt=None, nt=2, dynamic=False):
        self.min_firm_type = min_firm_type
        self.cross_period_normalize = cross_period_normalize
        if nnt is None:
            self.nnt = range(nt)
        else:
            self.nnt = to_list(nnt)
        self.nt = nt
        self.dynamic = dynamic

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': None, 'h': None, 'A': A, 'b': b}, where G, h, A, and b are defined in the quadratic programming model
        '''
        ## Unpack parameters ##
        min_firm_type, cross_period_normalize, nnt, nt, dynamic = self.min_firm_type, self.cross_period_normalize, self.nnt, self.nt, self.dynamic

        ## Initialize variables ##
        # A starts with 4 dimensions
        if cross_period_normalize:
            A = np.zeros(shape=(1, nt, nl, nk))
        else:
            A = np.zeros(shape=(len(nnt), nt, nl, nk))
            i = 0

        ## Generate constraints ##
        for period in nnt:
            ## Iterate over periods ##
            if cross_period_normalize:
                A[0, period, 0, min_firm_type] = (1 / len(nnt))
            else:
                A[i, period, 0, min_firm_type] = 1
                i += 1

        if dynamic:
            # Use dynamic BLM dimensions (i, l, period, k)
            A = A.transpose((0, 2, 1, 3))

        # Reshape A to 2 dimensions
        if cross_period_normalize:
            A = A.reshape((1, nt * nl * nk))

            b = - np.zeros(shape=A.shape[0])

            return {'G': None, 'h': None, 'A': A, 'b': b}
        else:
            # Keep track of columns to normalize, and drop them before estimation
            A = A.reshape((len(nnt), nt * nl * nk))
            drop_cols = list(np.nonzero(A.sum(axis=0) > 0.5)[0])

            return {'G': None, 'h': None, 'A': None, 'b': None, 'drop_cols': drop_cols}

class NormalizeAllWorkerTypes():
    '''
    Generate BLM constraints so that all worker-firm type pairs that include the lowest firm type have effect 0.

    Arguments:
        min_firm_type (int): lowest firm type
        cross_period_normalize (bool): if True, rather than normalizing for each period separately, normalize the mean of all worker-firm type pair effects over all periods jointly
        nnt (int or list of ints or None): time periods to constrain; None is equivalent to range(nt)
        nt (int): number of time periods
        dynamic (bool): if True, using dynamic BLM estimator
    '''

    def __init__(self, min_firm_type, cross_period_normalize=False, nnt=None, nt=2, dynamic=False):
        self.min_firm_type = min_firm_type
        self.cross_period_normalize = cross_period_normalize
        if nnt is None:
            self.nnt = range(nt)
        else:
            self.nnt = to_list(nnt)
        self.nt = nt
        self.dynamic = dynamic

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': None, 'h': None, 'A': A, 'b': b}, where G, h, A, and b are defined in the quadratic programming model
        '''
        ## Unpack parameters ##
        min_firm_type, cross_period_normalize, nnt, nt, dynamic = self.min_firm_type, self.cross_period_normalize, self.nnt, self.nt, self.dynamic

        ## Initialize variables ##
        # A starts with 4 dimensions
        if cross_period_normalize:
            A = np.zeros(shape=(nl, nt, nl, nk))
        else:
            A = np.zeros(shape=(len(nnt) * nl, nt, nl, nk))
            i = 0

        ## Generate constraints ##
        for period in nnt:
            ## Iterate over periods ##
            for l in range(nl):
                ## Iterate over worker types ##
                if cross_period_normalize:
                    A[l, period, l, min_firm_type] = (1 / len(nnt))
                else:
                    A[i, period, l, min_firm_type] = 1
                    i += 1

        if dynamic:
            # Use dynamic BLM dimensions (i, l, period, k)
            A = A.transpose((0, 2, 1, 3))

        # Reshape A to 2 dimensions
        if cross_period_normalize:
            A = A.reshape((nl, nt * nl * nk))

            b = - np.zeros(shape=A.shape[0])

            return {'G': None, 'h': None, 'A': A, 'b': b}
        else:
            # Keep track of columns to normalize, and drop them before estimation
            A = A.reshape((len(nnt) * nl, nt * nl * nk))
            drop_cols = list(np.nonzero(A.sum(axis=0) > 0.5)[0])

            return {'G': None, 'h': None, 'A': None, 'b': None, 'drop_cols': drop_cols}

class NormalizeAllFirmTypes():
    '''
    Generate BLM constraints so that all worker-firm type pairs that include the lowest worker type have effect 0.

    Arguments:
        cross_period_normalize (bool): if True, rather than normalizing for each period separately, normalize the mean of all worker-firm type pair effects over all periods jointly
        nnt (int or list of ints or None): time periods to constrain; None is equivalent to range(nt)
        nt (int): number of time periods
        dynamic (bool): if True, using dynamic BLM estimator
    '''

    def __init__(self, cross_period_normalize=False, nnt=None, nt=2, dynamic=False):
        self.cross_period_normalize = cross_period_normalize
        if nnt is None:
            self.nnt = range(nt)
        else:
            self.nnt = to_list(nnt)
        self.nt = nt
        self.dynamic = dynamic

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': None, 'h': None, 'A': A, 'b': b}, where G, h, A, and b are defined in the quadratic programming model
        '''
        ## Unpack parameters ##
        cross_period_normalize, nnt, nt, dynamic = self.cross_period_normalize, self.nnt, self.nt, self.dynamic

        ## Initialize variables ##
        # A starts with 4 dimensions
        if cross_period_normalize:
            A = np.zeros(shape=(nk, nt, nl, nk))
        else:
            A = np.zeros(shape=(len(nnt) * nk, nt, nl, nk))
            i = 0

        ## Generate constraints ##
        for period in nnt:
            ## Iterate over periods ##
            for k in range(nk):
                ## Iterate over firm types ##
                if cross_period_normalize:
                    A[k, period, 0, k] = (1 / len(nnt))
                else:
                    A[i, period, 0, k] = 1
                    i += 1

        if dynamic:
            # Use dynamic BLM dimensions (i, l, period, k)
            A = A.transpose((0, 2, 1, 3))

        # Reshape A to 2 dimensions
        if cross_period_normalize:
            A = A.reshape((nk, nt * nl * nk))

            b = - np.zeros(shape=A.shape[0])

            return {'G': None, 'h': None, 'A': A, 'b': b}
        else:
            # Keep track of columns to normalize, and drop them before estimation
            A = A.reshape((len(nnt) * nk, nt * nl * nk))
            drop_cols = list(np.nonzero(A.sum(axis=0) > 0.5)[0])

            return {'G': None, 'h': None, 'A': None, 'b': None, 'drop_cols': drop_cols}

class Stationary():
    '''
    Generate BLM constraints so that worker-firm pair effects are the same in all periods.

    Arguments:
        nwt (int): number of worker types to constrain. This is used in conjunction with Linear(), as only two worker types are required to be constrained in this case; or in conjunction with NoWorkerTypeInteraction(), as only one worker type is required to be constrained in this case. Setting nwt=-1 constrains all worker types.
        nnt (int or list of ints or None): time periods to constrain; None is equivalent to range(nt)
        nt (int): number of time periods
        dynamic (bool): if True, using dynamic BLM estimator
    '''

    def __init__(self, nwt=-1, nnt=None, nt=2, dynamic=False):
        self.nwt = nwt
        if (nwt < -1) or (nwt == 0):
            raise NotImplementedError(f'nwt must equal -1 or be positive, but input specifies nwt={nwt}.')
        if nnt is None:
            self.nnt = range(nt)
        else:
            self.nnt = to_list(nnt)
        self.nt = nt
        self.dynamic = dynamic

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': None, 'h': None, 'A': A, 'b': b}, where G, h, A, and b are defined in the quadratic programming model
        '''
        ## Unpack parameters ##
        nwt, nnt, nt, dynamic = self.nwt, self.nnt, self.nt, self.dynamic

        ## Initialize variables ##
        if nwt == -1:
            nl_adj = nl
        else:
            nl_adj = min(nl, nwt)
        # A starts with 4 dimensions
        A = np.zeros(shape=((len(nnt) - 1) * nl_adj * nk, nt, nl, nk))
        i = 0

        ## Generate constraints ##
        for period in nnt[1:]:
            ## Iterate over periods ##
            for l in range(nl_adj):
                ## Iterate over worker types ##
                for k in range(nk):
                    ## Iterate over firm types ##
                    # Baseline is first period in nnt
                    A[i, nnt[0], l, k] = 1
                    # Comparisons are remaining periods in nnt
                    A[i, period, l, k] = -1
                    i += 1

        if dynamic:
            # Use dynamic BLM dimensions (i, l, period, k)
            A = A.transpose((0, 2, 1, 3))

        # Reshape A to 2 dimensions
        A = A.reshape(((len(nnt) - 1) * nl_adj * nk, nt * nl * nk))

        b = - np.zeros(shape=A.shape[0])

        return {'G': None, 'h': None, 'A': A, 'b': b}

class StationaryFirmTypeVariation():
    '''
    Generate BLM constraints so that the firm type induced variation of worker-firm pair effects is the same in all periods. In particular, this is equivalent to setting A2 = (np.mean(A2, axis=1) + A1.T - np.mean(A1, axis=1)).T.

    Arguments:
        nnt (int or list of ints or None): time periods to constrain; None is equivalent to [0, 1]
        nt (int): number of time periods
        R_version (bool): if True, use R version of stationary firm-type variation constraint (this is called fixb in the R code, but this version is not recommended)
        dynamic (bool): if True, using dynamic BLM estimator
    '''

    def __init__(self, nnt=None, nt=2, R_version=False, dynamic=False):
        if nnt is None:
            self.nnt = [0, 1]
        else:
            self.nnt = to_list(nnt)
        self.nt = nt
        self.R_version = R_version
        self.dynamic = dynamic

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': None, 'h': None, 'A': A, 'b': b}, where G, h, A, and b are defined in the quadratic programming model
        '''
        ## Unpack parameters ##
        nnt, nt, R_version, dynamic = self.nnt, self.nt, self.R_version, self.dynamic

        if not R_version:
            ## Initialize variables ##
            # A starts with 4 dimensions
            A = np.zeros(shape=((len(nnt) - 1) * nl * nk, nt, nl, nk))
            i = 0

            ## Generate constraints ##
            for period in nnt[1:]:
                ## Iterate over periods ##
                for l in range(nl):
                    ## Iterate over worker types ##
                    for k1 in range(nk):
                        ## Iterate over firm types ##
                        for k2 in range(nk):
                            ## Iterate over firm types ##
                            # Baseline is first period in nnt
                            A[i, nnt[0], l, k2] = - (1 / nk)
                            # Comparisons are remaining periods in nnt
                            A[i, period, l, k2] = (1 / nk)
                        # Baseline is first period in nnt
                        A[i, nnt[0], l, k1] += 1
                        # Comparisons are remaining periods in nnt
                        A[i, period, l, k1] -= 1
                        i += 1

            if dynamic:
                # Use dynamic BLM dimensions (i, l, period, k)
                A = A.transpose((0, 2, 1, 3))

            # Reshape A to 2 dimensions
            A = A.reshape(((len(nnt) - 1) * nl * nk, nt * nl * nk))
        else:
            # NOTE: in reality, this is fixb from the R code
            # NOTE: the other method doesn't work with the dynamic model
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
            A = A.reshape((A.shape[0], nt, nl, nk))

            if dynamic:
                # Use dynamic BLM dimensions (i, l, period, k)
                A = A.transpose((0, 2, 1, 3))

            # Reshape A to 2 dimensions
            A = A.reshape((A.shape[0], nt * nl * nk))

        b = - np.zeros(shape=A.shape[0])

        return {'G': None, 'h': None, 'A': A, 'b': b}

class FirmSum():
    '''
    Generate BLM constraints that constrain the sum of all firm effects.

    Arguments:
        b (float): equality bound
        cross_period_sum (bool): if True, rather than constraining sums for each period separately, constrain the sum over all periods jointly
        nnt (int or list of ints or None): time periods to constrain; None is equivalent to range(nt)
        nt (int): number of time periods
    '''

    def __init__(self, b=0, cross_period_sum=False, nnt=None, nt=2):
        self.b = b
        self.cross_period_sum = cross_period_sum
        if nnt is None:
            self.nnt = range(nt)
        else:
            self.nnt = to_list(nnt)
        self.nt = nt

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': None, 'h': None, 'A': A, 'b': b}, where G, h, A, and b are defined in the quadratic programming model
        '''
        ## Unpack parameters ##
        b, cross_period_sum, nnt, nt = self.b, self.cross_period_sum, self.nnt, self.nt

        ## Initialize variables ##
        # A starts with 4 dimensions
        if cross_period_sum:
            A = np.zeros(shape=(nk, nt, nl, nk))
        else:
            A = np.zeros(shape=(len(nnt) * nk, nt, nl, nk))
        i = 0

        ## Generate constraints ##
        for period in nnt:
            ## Iterate over periods ##
            for k in range(nk):
                ## Iterate over firm types ##
                for l in range(nl):
                    ## Iterate over worker types ##
                    if cross_period_sum:
                        A[k, period, l, k] = 1
                    else:
                        A[i, period, l, k] = 1
                i += 1

        # Reshape A to 2 dimensions
        if cross_period_sum:
            A = A.reshape((nk, nt * nl * nk))
        else:
            A = A.reshape((len(nnt) * nk, nt * nl * nk))

        return {'G': None, 'h': None, 'A': A, 'b': b}

class WorkerSum():
    '''
    Generate BLM constraints that constrain the sum of all worker effects.

    Arguments:
        b (float): equality bound
        cross_period_sum (bool): if True, rather than constraining sums for each period separately, constrain the sum over all periods jointly
        nnt (int or list of ints or None): time periods to constrain; None is equivalent to range(nt)
        nt (int): number of time periods
    '''

    def __init__(self, b=0, cross_period_sum=False, nnt=None, nt=2):
        self.b = b
        self.cross_period_sum = cross_period_sum
        if nnt is None:
            self.nnt = range(nt)
        else:
            self.nnt = to_list(nnt)
        self.nt = nt

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': None, 'h': None, 'A': A, 'b': b}, where G, h, A, and b are defined in the quadratic programming model
        '''
        ## Unpack parameters ##
        b, cross_period_sum, nnt, nt = self.b, self.cross_period_sum, self.nnt, self.nt

        ## Initialize variables ##
        # A starts with 4 dimensions
        if cross_period_sum:
            A = np.zeros(shape=(nl, nt, nl, nk))
        else:
            A = np.zeros(shape=(len(nnt) * nl, nt, nl, nk))
        i = 0

        ## Generate constraints ##
        for period in nnt:
            ## Iterate over periods ##
            for l in range(nl):
                ## Iterate over worker types ##
                for k in range(nk):
                    ## Iterate over firm types ##
                    if cross_period_sum:
                        A[l, period, l, k] = 1
                    else:
                        A[i, period, l, k] = 1
                i += 1

        # Reshape A to 2 dimensions
        if cross_period_sum:
            A = A.reshape((nl, nt * nl * nk))
        else:
            A = A.reshape((len(nnt) * nl, nt * nl * nk))

        return {'G': None, 'h': None, 'A': A, 'b': b}

class BoundedBelow():
    '''
    Generate BLM constraints so that worker-firm pair effects are bounded below.

    Arguments:
        lb (float or NumPy Array): lower bound
        nt (int): number of time periods
    '''

    def __init__(self, lb=0, nt=2):
        self.lb = lb
        self.nt = nt

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': G, 'h': h, 'A': None, 'b': None}, where G, h, A, and b are defined in the quadratic programming model
        '''
        nt, lb = self.nt, self.lb
        h = lb * np.ones(shape=nt * nl * nk)

        return {'G': None, 'h': None, 'A': None, 'b': None, 'lb': h, 'ub': None}

class BoundedAbove():
    '''
    Generate BLM constraints so that worker-firm pair effects are bounded above.

    Arguments:
        ub (float or NumPy Array): upper bound
        nt (int): number of time periods
    '''

    def __init__(self, ub=0, nt=2):
        self.ub = ub
        self.nt = nt

    def _get_constraints(self, nl, nk):
        '''
        Generate constraint arrays.

        Arguments:
            nl (int): number of worker types
            nk (int): number of firm types

        Returns:
            (dict of NumPy Arrays): {'G': G, 'h': h, 'A': None, 'b': None}, where G, h, A, and b are defined in the quadratic programming model
        '''
        nt, ub = self.nt, self.ub
        h = ub * np.ones(shape=nt * nl * nk)

        return {'G': None, 'h': None, 'A': None, 'b': None, 'lb': None, 'ub': h}
