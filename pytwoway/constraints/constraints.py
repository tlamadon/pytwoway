'''
Classes for generating quadratic programming constraints for BLM. Note: use classes rather than nested functions because nested functions cannot be pickled (source: https://stackoverflow.com/a/12022055/17333120).

All classes are for solving a quadratic programming model of the following form:
        min_x(1/2 x.T @ P @ x + q.T @ x)
        s.t.    Gx <= h
                Ax = b

Note that parameters are ordered with precedence of (time, worker type, firm type). As an example, if nt=2, nl=2, and nk=3, then the parameters will be ordered as follows:
    (0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0), (0, 1, 1), (0, 1, 2), (1, 0, 0), (1, 0, 1), (1, 0, 2), (1, 1, 0), (1, 1, 1), (1, 1, 2)
'''
import numpy as np
from qpsolvers import solve_qp
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
        self.G = np.array([])
        # Inequality constraint bound
        self.h = np.array([])
        # Equality constraint matrix
        self.A = np.array([])
        # Equality constraint bound
        self.b = np.array([])

    def add_constraints(self, constraints):
        '''
        Add a built-in constraint.

        Arguments:
            constraints (object or list of objects): constraint objects with class method ._get_constraints() that defines constraints to add
        '''
        for constraint in to_list(constraints):
            self._add_constraint(**constraint._get_constraints(nl=self.nl, nk=self.nk))

    def _add_constraint(self, G=None, h=None, A=None, b=None):
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

    def clear_constraints(self, inequality=True, equality=True):
        '''
        Clear constraints.

        Arguments:
            inequality (bool): if True, clear inequality constraints
            equality (bool): if True, clear equality constraints
        '''
        if inequality:
            self.G = np.array([])
            self.h = np.array([])
        if equality:
            self.A = np.array([])
            self.b = np.array([])

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

class Linear():
    '''
    Generate BLM constraints so that for a fixed firm type, worker types effects must change linearly.

    Arguments:
        nnt (int): number of time periods to constrain. This should be set to 1 if Linear() is being used in conjunction with Stationary().
        nt (int): number of time periods
    '''

    def __init__(self, nnt=2, nt=2):
        self.nnt = nnt
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
        nt, nnt = self.nt, self.nnt
        A = np.zeros(shape=(nnt * (nl - 2) * nk, nt * nl * nk))
        for period in range(nnt):
            row_shift = period * (nl - 2) * nk
            col_shift = period * nl * nk
            for k in range(nk):
                for l in range(nl - 2):
                    A[row_shift + l, col_shift + nk * l] = 1
                    A[row_shift + l, col_shift + nk * (l + 1)] = -2
                    A[row_shift + l, col_shift + nk * (l + 2)] = 1
                row_shift += (nl - 2)
                col_shift += 1

        b = - np.zeros(shape=A.shape[0])

        return {'G': None, 'h': None, 'A': A, 'b': b}

class Monotonic():
    '''
    Generate BLM constraints so that for a fixed firm type, worker types effects must increase (or decrease) monotonically.

    Arguments:
        md (float): minimum difference between consecutive types
        increasing (bool): if True, monotonic increasing; if False, monotonic decreasing
        nt (int): number of time periods
    '''

    def __init__(self, md=0, increasing=True, nt=2):
        self.md = md
        self.increasing = increasing
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
        nt, md, increasing = self.nt, self.md, self.increasing
        G = np.zeros(shape=(nt * (nl - 1) * nk, nt * nl * nk))
        for period in range(nt):
            row_shift = period * (nl - 1) * nk
            col_shift = period * nl * nk
            for k in range(nk):
                for l in range(nl - 1):
                    G[row_shift + l, col_shift + nk * l] = 1
                    G[row_shift + l, col_shift + nk * (l + 1)] = -1
                row_shift += (nl - 1)
                col_shift += 1

        h = - md * np.ones(shape=G.shape[0])

        if not increasing:
            G *= -1

        return {'G': G, 'h': h, 'A': None, 'b': None}

class NoWorkerTypeInteraction():
    '''
    Generate BLM constraints so that for a fixed firm type, worker types effects must all be the same.

    Arguments:
        nt (int): number of time periods
    '''

    def __init__(self, nt=2):
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
        nt = self.nt
        A = np.zeros(shape=(nt * (nl - 1) * nk, nt * nl * nk))
        for period in range(nt):
            row_shift = period * (nl - 1) * nk
            col_shift = period * nl * nk
            for k in range(nk):
                for l in range(nl - 1):
                    A[row_shift + l, col_shift + nk * l] = 1
                    A[row_shift + l, col_shift + nk * (l + 1)] = -1
                row_shift += (nl - 1)
                col_shift += 1

        b = - np.zeros(shape=A.shape[0])

        return {'G': None, 'h': None, 'A': A, 'b': b}

class Stationary():
    '''
    Generate BLM constraints so that worker-firm pair effects are the same in all periods.

    Arguments:
        nwt (int): number of worker types to constrain. This is used in conjunction with Linear(), as only two worker types are required to be constrained in this case; or in conjunction with NoWorkerTypeInteraction(), as only one worker type is required to be constrained in this case. Setting ns=-1 constrains all worker types.
        nt (int): number of time periods
    '''

    def __init__(self, nwt=-1, nt=2):
        self.nwt = nwt
        if (nwt < -1) or (nwt == 0):
            raise NotImplementedError(f'nwt must equal -1 or be positive, but input specifies nwt={nwt}.')
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
        nt, nwt = self.nt, self.nwt
        if nwt == -1:
            nl_adj = nl
        else:
            nl_adj = min(nl, nwt)
        A = np.zeros(shape=((nt - 1) * nl_adj * nk, nt * nl * nk))
        for period in range(nt - 1):
            row_shift = period * nl_adj * nk
            col_shift = period * nl * nk
            for k in range(nk):
                for l in range(nl_adj):
                    A[row_shift + k + l, col_shift + nl * k + l] = 1
                    A[row_shift + k + l, col_shift + nl * nk + nl * k + l] = -1
                row_shift += (nl_adj - 1)

        b = - np.zeros(shape=A.shape[0])

        return {'G': None, 'h': None, 'A': A, 'b': b}

class StationaryFirmTypeVariation():
    '''
    Generate BLM constraints so that the firm type induced variation of worker-firm pair effects is the same in all periods. In particular, this is equivalent to setting A2 = (np.mean(A2, axis=1) + A1.T - np.mean(A1, axis=1)).T.

    Arguments:
        nt (int): number of time periods
    '''

    def __init__(self, nt=2):
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
        nt = self.nt
        A = np.zeros(shape=((nt - 1) * nl * nk, nt * nl * nk))
        for period in range(nt - 1):
            row_shift = period * nl * nk
            col_shift = period * nl * nk
            for l in range(nl):
                for k1 in range(nk):
                    for k2 in range(nk):
                        A[row_shift + k1, col_shift + k2] = -(1 / nk)
                        A[row_shift + k1, col_shift + nl * nk + k2] = (1 / nk)
                    A[row_shift + k1, col_shift + k1] += 1
                    A[row_shift + k1, col_shift + nl * nk + k1] -= 1
                row_shift += nk
                col_shift += nk

        b = - np.zeros(shape=A.shape[0])

        return {'G': None, 'h': None, 'A': A, 'b': b}

class BoundedBelow():
    '''
    Generate BLM constraints so that worker-firm pair effects are bounded below.

    Arguments:
        lb (float): lower bound
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
        G = - np.eye(nt * nl * nk)
        h = - lb * np.ones(shape=nt * nl * nk)

        return {'G': G, 'h': h, 'A': None, 'b': None}

class BoundedAbove():
    '''
    Generate BLM constraints so that worker-firm pair effects are bounded above.

    Arguments:
        ub (float): upper bound
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
        G = np.eye(nt * nl * nk)
        h = ub * np.ones(shape=nt * nl * nk)

        return {'G': G, 'h': h, 'A': None, 'b': None}
