'''
    We implement the non-linear estimator from Bonhomme Lamadon & Manresa
'''

import numpy as np
from numpy import pi
import pandas as pd
from scipy.special import logsumexp
from scipy.sparse import csc_matrix, diags
from scipy.stats import norm
from qpsolvers import solve_qp
from matplotlib import pyplot as plt
import time
import argparse
import warnings
from util import update_dict

####################
##### New Code #####
####################
class QPConstrained:
    '''
    Solve a quadratic programming model of the following form:
        min_x(1/2 x.T @ P @ x + q.T @ x)
        s.t.    Gx <= h
                Ax = b

    Params:
        nl (int): number of worker types
        nk (int): number of firm types
    '''

    def __init__(self, nl, nk):
        self.nl = nl
        self.nk = nk

        self.G = np.array([]) # Inequality constraint matrix
        self.h = np.array([]) # Inequality constraint bound
        self.A = np.array([]) # Equality constraint matrix
        self.b = np.array([]) # Equality constraint bound

        self.default_constraints = {
            'gap_akmmono': 0, # Used for akmmono constraint
            'gap_mono_k': 0, # Used for mono_k constraint
            'gap_bigger': 0, # Used for biggerthan constraint to determine bound
            'gap_smaller': 0, # Used for smallerthan constraint to determine bound,
            'n_periods': 1, # Number of periods in the data
            'nt': 4
        }

    def add_constraint_builtin(self, constraint, constraint_params={}):
        '''
        Add a built-in constraint.

        Params:
            constraint (str): name of constraint to add
            constraint_params (dict): parameters
        '''
        nl = self.nl
        nk = self.nk
        params = update_dict(self.default_constraints, constraint_params)
        G = None
        h = None
        A = None
        b = None
        if constraint in ['lin', 'lin_add', 'akm']:
            LL = np.zeros(shape=(nl - 1, nl))
            for l in range(nl - 1):
                LL[l, l] = 1
                LL[l, l + 1] = - 1
            KK = np.zeros(shape=(nk - 1, nk))
            for k in range(nk - 1):
                KK[k, k] = 1
                KK[k, k + 1] = - 1
            A = - np.kron(LL, KK)
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
            for i in range(nt - 1):
                MM[i, i] = 1
                MM[i, i + 1] = - 1
            A = - np.kron(MM, A)
            b = - np.zeros(shape=nl * (nk - 1) * (nt - 1))

        elif constraint in ['biggerthan', 'greaterthan']:
            gap = params['gap_bigger']
            n_periods = params['n_periods']
            G = - np.eye(n_periods * nk * nl)
            h = - gap * np.ones(shape=n_periods * nk * nl)

        elif constraint in ['smallerthan', 'lessthan']:
            gap = params['gap_smaller']
            n_periods = params['n_periods']
            G = np.eye(n_periods * nk * nl)
            h = gap * np.ones(shape=n_periods * nk * nl)

        elif constraint == 'lin_para':
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
            warnings.warn('Invalid constraint entered {}.'.format(constraint))
            return

        # Add constraints to attributes
        self.add_constraint_manual(G=G, h=h, A=A, b=b)

    def add_constraints_builtin(self, constraints, constraint_params={}):
        '''
        Add a built-in constraint.

        Params:
            constraints (list of str): names of constraint to add
            constraint_params (dict): parameters
        '''
        for constraint in constraints:
            self.add_constraint_builtin(constraint, constraint_params)

    def add_constraint_manual(self, G=None, h=None, A=None, b=None):
        '''
        Manually add a constraint. If setting inequality constraints, must set both G and h to have the same dimension 0. If setting equality constraints, must set both A and b to have the same dimension 0.

        Params:
            G (NumPy Array): inequality constraint matrix
            h (NumPy Array): inequality constraint bound
            A (NumPy Array): equality constraint matrix
            b (NumPy Array): equality constraint bound
        '''
        if G is not None: # If you have inequality constraints
            if len(self.G) > 0:
                self.G = np.concatenate((self.G, G), axis=0)
                self.h = np.concatenate((self.h, h), axis=0)
            else:
                self.G = G
                self.h = h
        if A is not None: # If you have equality constraints
            if len(self.A) > 0:
                self.A = np.concatenate((self.A, A), axis=0)
                self.b = np.concatenate((self.b, b), axis=0)
            else:
                self.A = A
                self.b = b

    def pad(self, l=0, r=0):
        '''
        Add padding to the left and/or right of C matrix.

        Params:
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
            (bool): True if constraints feasiable, False otherwise
        '''
        # -------  simulate an OLS -------
        n = 100
        k = 10
        # parameters

        x = np.random.normal(size=k)
        # regressors
        M = np.random.normal(size=(n, k))
        # dependent
        Y = M @ x 

        # =-------- map to quadprog ---------
        cons = QPConstrained(k, 1)
        P = M.T @ M
        q = - M.T @ Y
        try:
            self.solve(P, q)
            return True
        except ValueError:
            return False

    def solve(self, P, q):
        '''
        Solve a quadratic programming model of the following form:
        min_x(1/2 x.T @ P @ x + q.T @ x)
        s.t.    Gx <= h
                Ax = b
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
    return - 0.5 * np.log(2 * pi) - np.log(sd) - (x - mu) ** 2 / (2 * sd ** 2)

class BLMEstimator:

    def __init__(self, user_params):
        '''
            Initialize the model
        '''
        default_BLM = {
            'nl': 6, # Number of worker types
            'nk': 10, # Number of firm types
            'fixb': False,
            'stationary': False
        }
        params = update_dict(default_BLM, user_params)
        nl = params['nl']
        nk = params['nk']
        self.nl = nl # Number of worker types
        self.nk = nk # Number of firm types

        # model for Y1|Y2,l,k for movers and stayes
        self.A1 = 0.9 * (1 + 0.5 * np.random.normal(size=(nk, nl)))
        self.S1 = 0.3 * (1 + 0.5 * np.random.uniform(size=(nk, nl)))
        # model for Y4|Y3,l,k for movers and stayes
        self.A2 = 0.9 * (1 + 0.5 * np.random.normal(size=(nk, nl)))
        self.S2 = 0.3 * (1 + 0.5 * np.random.uniform(size=(nk, nl)))
        # model for p(K | l ,l') for movers
        self.pk1 = np.random.dirichlet(alpha=[1] * nl, size=nk * nk)
        # model for p(K | l ,l') for stayers
        self.pk0 = np.random.dirichlet(alpha=np.ones(shape=nk), size=(nk, nl)) # np.random.dirichlet(alpha=[1] * nk, size=nl)
        # model.pk0 = np.expand_dims(model.pk0, axis=0)

        self.NNm = np.zeros(shape=(nk, nk)).astype(int) + 10 # np.random.randint(low=0, high=max(nl // 2 + 1, 1), size=[nl, nl]) # FIXME new code
        self.NNs = np.zeros(shape=nk).astype(int) + 10 # np.random.randint(low=nl // 2 + 1, high=max(nl, nl // 2 + 2), size=nl * nl) # FIXME new code

        for l in range(nl):
            self.A1[:, l] = sorted(self.A1[:, l])
            self.A2[:, l] = sorted(self.A2[:, l])

        if params['fixb']:
            self.A2 = np.mean(self.A2, axis=0) + self.A1 - np.mean(self.A1, axis=0)

        if params['stationary']:
            self.A2 = self.A1

        # # Mean of wages by firm and worker type
        # self.A1 = np.zeros((self.nk, self.nl))
        # self.A2 = np.zeros((self.nk, self.nl))
        # # Starndard deviation of wages by firm and worker type
        # self.S1 = np.ones((self.nk, self.nl))        
        # self.S2 = np.ones((self.nk, self.nl))        

        # # Model for p(K | l ,l') for movers
        # self.pk1 = np.ones((self.nk, self.nk, self.nl)) / self.nl
        # # Model for p(K | l ,l') for stayers
        # self.pk0 = np.ones((self.nk, self.nl)) / self.nl

    def fit(self, jdata, user_params={}):
        '''
            We write the EM algorithm for the movers
        '''
        nl = self.nl
        nk = self.nk
        ni = len(jdata)

        # update params
        default_fit = {
            'maxiter': 1000 # Max number of iterations
        }
        params = update_dict(default_fit, user_params)

        # store wage outcomes and groups
        Y1 = jdata['y1'].to_numpy() 
        Y2 = jdata['y2'].to_numpy()
        J1 = jdata['j1'].to_numpy()
        J2 = jdata['j2'].to_numpy()

        # matrix of posterior probabilities
        qi = np.ones(shape=(ni, nl))

        # constraints
        cons_a = QPConstrained(nl, nk)
        cons_s = QPConstrained(nl, nk)
        cons_s.add_constraints_builtin(['biggerthan'], {'gap_bigger': 0, 'n_periods': 2})

        lp = np.zeros(shape=(ni, nl))
        JJ1 = csc_matrix((np.ones(ni), (jdata.index, jdata['j1'])), shape=(ni, nk))
        JJ2 = csc_matrix((np.ones(ni), (jdata.index, jdata['j1'])), shape=(ni, nk))

        for iter in range(params['maxiter']):

            # -------- E-Step ---------
            # we compute the posterior probabiluties for each row
            # we iterate over the worker types, should not be be 
            # to costly since the vector is quite large within each iteration
            for l in range(nl): 
                lp1 = lognormpdf(Y1, self.A1[J1, l], self.S1[J1, l])
                lp2 = lognormpdf(Y2, self.A2[J2, l], self.S2[J2, l])
                KK = J1 + nk * J2
                lp[:, l] = np.log(self.pk1[KK, l]) + lp1 + lp2 # FIXME added new middle dimension to lp

            # we compute log sum exp to get likelihoods and probabilities
            qi = np.exp(lp.T - logsumexp(lp, axis=1)).T # FIXME changed logsumexp from axis=2 to axis=1
            liks = logsumexp(lp, axis=0).sum() # FIXME should this be returend?

            # --------- M-step ----------
            # for now we run a simple ols, however later we
            # want to add constraints!
            # see https://scaron.info/blog/quadratic-programming-in-python.html

            # the regression has 2 * nl * nk parameters and nl * ni rows
            # we do not necessarly want to construct the duplicated data by nl
            # instead we will construct X'X and X'Y by looping over nl
            # we also note that X'X is block diagonal with 2*nl matrices of dimensions nk^2
            ts = nl * nk # shift for period 2 FIXME used to be called t2, I assumed it is ts
            XwX = np.zeros(shape=(2 * ts, 2 * ts)) # np.zeros(shape=(2 * nl * ni, 2 * nl * ni)) # np.zeros(shape=(nl * nk + ts, nl * nk + ts)) # FIXME new line
            XwY = np.zeros(shape=2 * ts) # np.zeros(shape=2 * nl * ni) # np.zeros(shape=nl * nk + ts) # FIXME new line
            for l in range(nl):
                l_index = l * nk
                r_index = (l + 1) * nk
                XwX[l_index: r_index, l_index: r_index] = (JJ1.T @ (diags(qi[:, l] / self.S1[J1, l]) @ JJ1)).todense()
                # here want to compute the matrix multiplication with a diagonal mattrix in the middle, 
                # we might be better off trying this within numba or something.
                XwY[l_index: r_index] = JJ1.T @ (diags(qi[:, l] / self.S1[J1, l]) @ Y1)
                XwX[l_index + ts: r_index + ts, l_index + ts: r_index + ts] = (JJ2.T @ (diags(qi[:, l] / self.S2[J2, l]) @ JJ2)).todense()
                XwY[l_index + ts: r_index + ts] = JJ2.T @ (diags(qi[:, l] / self.S2[J2, l]) @ Y2)
            
            # we solve the system to get all the parameters
            # we need to add the constraints here using quadprog
            cons_a.solve(XwX, XwY)
            res_a = cons_a.res
            self.A1 = np.reshape(res_a, [nk, nl, 2])[:, :, 0]
            self.A2 = np.reshape(res_a, [nk, nl, 2])[:, :, 1]

            XwS = np.zeros(shape=2 * ts) # np.zeros(shape=2 * nl * ni)
            # next we extract the variances
            for l in range(nl):
                l_index = l * nk
                r_index = (l + 1) * nk
                XwS[l_index: r_index] = - JJ1.T @ (diags(qi[:, l] / self.S1[J1, l]) @ ((Y1 - self.A1[J1, l]) ** 2)) # FIXME changed to -
                XwS[l_index + ts: r_index + ts] = - JJ2.T @ (diags(qi[:, l] / self.S2[J2, l]) @ ((Y2 - self.A2[J2, l]) ** 2)) # FIXME changed to -

            cons_s.solve(XwX, XwS) # we need to constraint the parameters to be all positive
            res_s = cons_s.res
            self.S1 = np.sqrt(np.reshape(res_s, [nk, nl, 2])[:, :, 0])
            self.S2 = np.sqrt(np.reshape(res_s, [nk, nl, 2])[:, :, 1])

    def fit_A(self, jdata, user_params={}):
        '''
            We write the EM algorithm for the movers
        '''
        nl = self.nl
        nk = self.nk
        ni = len(jdata)

        # update params
        default_fit = {
            'maxiter': 1000 # Max number of iterations
        }
        params = update_dict(default_fit, user_params)

        # store wage outcomes and groups
        Y1 = jdata['y1'].to_numpy() 
        Y2 = jdata['y2'].to_numpy()
        J1 = jdata['j1'].to_numpy()
        J2 = jdata['j2'].to_numpy()

        # matrix of posterior probabilities
        qi = np.ones(shape=(ni, nl))

        # constraints
        cons_a = QPConstrained(nl, nk)
        cons_s = QPConstrained(nl, nk)
        cons_s.add_constraints_builtin(['biggerthan'], {'gap_bigger': 0, 'n_periods': 2})

        lp = np.zeros(shape=(ni, nl))
        JJ1 = csc_matrix((np.ones(ni), (jdata.index, jdata['j1'])), shape=(ni, nk))
        JJ2 = csc_matrix((np.ones(ni), (jdata.index, jdata['j1'])), shape=(ni, nk))

        for iter in range(params['maxiter']):

            # -------- E-Step ---------
            # we compute the posterior probabiluties for each row
            # we iterate over the worker types, should not be be 
            # to costly since the vector is quite large within each iteration
            for l in range(nl): 
                lp1 = lognormpdf(Y1, self.A1[J1, l], self.S1[J1, l])
                lp2 = lognormpdf(Y2, self.A2[J2, l], self.S2[J2, l])
                KK = J1 + nk * J2
                lp[:, l] = np.log(self.pk1[KK, l]) + lp1 + lp2 # FIXME added new middle dimension to lp

            # we compute log sum exp to get likelihoods and probabilities
            qi = np.exp(lp.T - logsumexp(lp, axis=1)).T # FIXME changed logsumexp from axis=2 to axis=1
            liks = logsumexp(lp, axis=0).sum() # FIXME should this be returend?

            # --------- M-step ----------
            # for now we run a simple ols, however later we
            # want to add constraints!
            # see https://scaron.info/blog/quadratic-programming-in-python.html

            # the regression has 2 * nl * nk parameters and nl * ni rows
            # we do not necessarly want to construct the duplicated data by nl
            # instead we will construct X'X and X'Y by looping over nl
            # we also note that X'X is block diagonal with 2*nl matrices of dimensions nk^2
            ts = nl * nk # shift for period 2
            XwX = np.zeros(shape=(2 * ts, 2 * ts))
            XwY = np.zeros(shape=2 * ts)
            for l in range(nl):
                l_index = l * nk
                r_index = (l + 1) * nk
                XwX[l_index: r_index, l_index: r_index] = (JJ1.T @ (diags(qi[:, l] / self.S1[J1, l]) @ JJ1)).todense()
                # here want to compute the matrix multiplication with a diagonal mattrix in the middle, 
                # we might be better off trying this within numba or something.
                XwY[l_index: r_index] = JJ1.T @ (diags(qi[:, l] / self.S1[J1, l]) @ Y1)
                XwX[l_index + ts: r_index + ts, l_index + ts: r_index + ts] = (JJ2.T @ (diags(qi[:, l] / self.S2[J2, l]) @ JJ2)).todense()
                XwY[l_index + ts: r_index + ts] = JJ2.T @ (diags(qi[:, l] / self.S2[J2, l]) @ Y2)

            # we solve the system to get all the parameters
            # we need to add the constraints here using quadprog
            cons_a.solve(XwX, XwY)
            res_a = cons_a.res
            self.A1 = np.reshape(res_a, [nk, nl, 2])[:, :, 0]
            self.A2 = np.reshape(res_a, [nk, nl, 2])[:, :, 1]

    def fit_S(self, jdata, user_params={}):
        '''
            We write the EM algorithm for the movers
        '''
        nl = self.nl
        nk = self.nk
        ni = len(jdata)

        # update params
        default_fit = {
            'maxiter': 1000 # Max number of iterations
        }
        params = update_dict(default_fit, user_params)

        # store wage outcomes and groups
        Y1 = jdata['y1'].to_numpy() 
        Y2 = jdata['y2'].to_numpy()
        J1 = jdata['j1'].to_numpy()
        J2 = jdata['j2'].to_numpy()

        # matrix of posterior probabilities
        qi = np.ones(shape=(ni, nl))

        # constraints
        cons_a = QPConstrained(nl, nk)
        cons_s = QPConstrained(nl, nk)
        cons_s.add_constraints_builtin(['biggerthan'], {'gap_bigger': 0, 'n_periods': 2})

        lp = np.zeros(shape=(ni, nl))
        JJ1 = csc_matrix((np.ones(ni), (jdata.index, jdata['j1'])), shape=(ni, nk))
        JJ2 = csc_matrix((np.ones(ni), (jdata.index, jdata['j1'])), shape=(ni, nk))

        for iter in range(params['maxiter']):

            # -------- E-Step ---------
            # we compute the posterior probabiluties for each row
            # we iterate over the worker types, should not be be 
            # to costly since the vector is quite large within each iteration
            for l in range(nl): 
                lp1 = lognormpdf(Y1, self.A1[J1, l], self.S1[J1, l])
                lp2 = lognormpdf(Y2, self.A2[J2, l], self.S2[J2, l])
                KK = J1 + nk * J2
                lp[:, l] = np.log(self.pk1[KK, l]) + lp1 + lp2 # FIXME added new middle dimension to lp

            # we compute log sum exp to get likelihoods and probabilities
            qi = np.exp(lp.T - logsumexp(lp, axis=1)).T # FIXME changed logsumexp from axis=2 to axis=1
            liks = logsumexp(lp, axis=0).sum() # FIXME should this be returend?

            # --------- M-step ----------
            # for now we run a simple ols, however later we
            # want to add constraints!
            # see https://scaron.info/blog/quadratic-programming-in-python.html

            # the regression has 2 * nl * nk parameters and nl * ni rows
            # we do not necessarly want to construct the duplicated data by nl
            # instead we will construct X'X and X'Y by looping over nl
            # we also note that X'X is block diagonal with 2*nl matrices of dimensions nk^2
            ts = nl * nk # shift for period 2 FIXME used to be called t2, I assumed it is ts
            XwX = np.zeros(shape=(2 * ts, 2 * ts)) # np.zeros(shape=(2 * nl * ni, 2 * nl * ni)) # np.zeros(shape=(nl * nk + ts, nl * nk + ts)) # FIXME new line
            for l in range(nl):
                l_index = l * nk
                r_index = (l + 1) * nk
                XwX[l_index: r_index, l_index: r_index] = (JJ1.T @ (diags(qi[:, l] / self.S1[J1, l]) @ JJ1)).todense()
                # here want to compute the matrix multiplication with a diagonal mattrix in the middle, 
                # we might be better off trying this within numba or something.
                XwX[l_index + ts: r_index + ts, l_index + ts: r_index + ts] = (JJ2.T @ (diags(qi[:, l] / self.S2[J2, l]) @ JJ2)).todense()

            XwS = np.zeros(shape=2 * ts) # np.zeros(shape=2 * nl * ni)
            # next we extract the variances
            for l in range(nl):
                l_index = l * nk
                r_index = (l + 1) * nk
                XwS[l_index: r_index] = - JJ1.T @ (diags(qi[:, l] / self.S1[J1, l]) @ ((Y1 - self.A1[J1, l]) ** 2)) # FIXME changed to -
                XwS[l_index + ts: r_index + ts] = - JJ2.T @ (diags(qi[:, l] / self.S2[J2, l]) @ ((Y2 - self.A2[J2, l]) ** 2)) # FIXME changed to -

            cons_s.solve(XwX, XwS) # we need to constraint the parameters to be all positive
            res_s = cons_s.res
            self.S1 = np.sqrt(np.reshape(res_s, [nk, nl, 2])[:, :, 0])
            self.S2 = np.sqrt(np.reshape(res_s, [nk, nl, 2])[:, :, 1])

    def fit_qi(self, jdata, user_params={}):
        '''
            We write the EM algorithm for the movers
        '''
        nl = self.nl
        nk = self.nk
        ni = len(jdata)

        # update params
        default_fit = {
            'maxiter': 1000 # Max number of iterations
        }
        params = update_dict(default_fit, user_params)

        # store wage outcomes and groups
        Y1 = jdata['y1'].to_numpy() 
        Y2 = jdata['y2'].to_numpy()
        J1 = jdata['j1'].to_numpy()
        J2 = jdata['j2'].to_numpy()

        # matrix of posterior probabilities
        qi = np.ones(shape=(ni, nl))

        # constraints
        cons_a = QPConstrained(nl, nk)
        cons_s = QPConstrained(nl, nk)
        cons_s.add_constraints_builtin(['biggerthan'], {'gap_bigger': 0, 'n_periods': 2})

        lp = np.zeros(shape=(ni, nl))
        JJ1 = csc_matrix((np.ones(ni), (jdata.index, jdata['j1'])), shape=(ni, nk))
        JJ2 = csc_matrix((np.ones(ni), (jdata.index, jdata['j1'])), shape=(ni, nk))

        for iter in range(params['maxiter']):

            # -------- E-Step ---------
            # we compute the posterior probabiluties for each row
            # we iterate over the worker types, should not be be 
            # to costly since the vector is quite large within each iteration
            for l in range(nl): 
                lp1 = lognormpdf(Y1, self.A1[J1, l], self.S1[J1, l])
                lp2 = lognormpdf(Y2, self.A2[J2, l], self.S2[J2, l])
                KK = J1 + nk * J2
                lp[:, l] = np.log(self.pk1[KK, l]) + lp1 + lp2 # FIXME added new middle dimension to lp

            # we compute log sum exp to get likelihoods and probabilities
            return np.exp(lp.T - logsumexp(lp, axis=1)).T # FIXME changed logsumexp from axis=2 to axis=1

    def sim_model(self, fixb=False, stationary=False, fsize=10, mmult=1, smult=1):
        '''
        Create a random model for EM with endogenous mobility with multinomial pr.

        Returns:
            sim (dict):
                'jdata': movers
                'sdata': stayers
        '''
        model = self._m2_mixt_new(fixb=fixb, stationary=stationary)
        sim = self._m2_mixt_simulate_sim(self, model, fsize=fsize, mmult=mmult, smult=smult)
        return sim

    def _m2_mixt_new(self, fixb=False, stationary=False):
        '''
        Returns:
            model (Pandas DataFrame):
        '''
        nl = self.nl
        nk = self.nk

        model = argparse.Namespace()

        # model for Y1|Y2,l,k for movers and stayes
        model.A1 = 0.9 * (1 + 0.5 * np.random.normal(size=(nk, nl)))
        model.S1 = 0.3 * (1 + 0.5 * np.random.uniform(size=(nk, nl)))
        # model for Y4|Y3,l,k for movers and stayes
        model.A2 = 0.9 * (1 + 0.5 * np.random.normal(size=(nk, nl)))
        model.S2 = 0.3 * (1 + 0.5 * np.random.uniform(size=(nk, nl)))
        # model for p(K | l ,l') for movers
        model.pk1 = np.random.dirichlet(alpha=[1] * nl, size=nk * nk)
        # model for p(K | l ,l') for stayers
        model.pk0 = np.random.dirichlet(alpha=np.ones(shape=nk), size=(nk, nl)) # np.random.dirichlet(alpha=[1] * nk, size=nl)
        # model.pk0 = np.expand_dims(model.pk0, axis=0)

        model.NNm = np.zeros(shape=(nk, nk)).astype(int) + 10 # Matrix of movers per group (nk x nk)
        model.NNs = np.zeros(shape=nk).astype(int) + 10 # Matrix of stayers per group (nk)

        for l in range(nl):
            model.A1[:, l] = sorted(model.A1[:, l])
            model.A2[:, l] = sorted(model.A2[:, l])

        if fixb:
            model.A2 = np.mean(model.A2, axis=0) + model.A1 - np.mean(model.A1, axis=0)

        if stationary:
            model.A2 = model.A1

        return model

    def _m2_mixt_simulate_movers(self, model, NNm):
        '''
        Using the model, simulates a dataset of movers.

        Returns:
            jdatae (Pandas DataFrame):
        '''
        J1 = np.zeros(shape=np.sum(NNm)).astype(int) - 1
        J2 = np.zeros(shape=np.sum(NNm)).astype(int) - 1
        Y1 = np.zeros(shape=np.sum(NNm))
        Y2 = np.zeros(shape=np.sum(NNm))
        L = np.zeros(shape=np.sum(NNm)).astype(int) - 1

        A1 = model.A1
        A2 = model.A2
        S1 = model.S1
        S2 = model.S2
        pk1 = model.pk1
        nl = self.nl
        nk = self.nk

        i = 0
        for k1 in range(nk): # FIXME changed from nl to nk
            for k2 in range(nk): # FIXME changed from nl to nk
                I = np.arange(i, i + NNm[k1, k2])
                ni = len(I)
                jj = k1 + nk * k2 # k1 + nk * (k2 - 1)
                J1[I] = k1
                J2[I] = k2

                # Draw k
                draw_vals = np.arange(nl)
                Li = np.random.choice(draw_vals, size=ni, replace=True, p=pk1[jj, :])
                L[I] = Li

                # Draw Y2, Y3
                Y1[I] = A1[k1, Li] + S1[k1, Li] * np.random.normal(size=ni)
                Y2[I] = A2[k2, Li] + S2[k2, Li] * np.random.normal(size=ni)

                i += NNm[k1, k2]

        jdatae = pd.DataFrame(data={'l': L, 'y1': Y1, 'y2': Y2, 'j1': J1, 'j2': J2})

        return jdatae

    def _m2_mixt_simulate_stayers(self, model, NNs):
        '''
        Using the model, simulates a dataset of stayers.

        Returns:
            sdatae (Pandas DataFrame):
        '''
        J1 = np.zeros(shape=np.sum(NNs)).astype(int) - 1
        J2 = np.zeros(shape=np.sum(NNs)).astype(int) - 1
        Y1 = np.zeros(shape=np.sum(NNs))
        Y2 = np.zeros(shape=np.sum(NNs))
        K  = np.zeros(shape=np.sum(NNs)).astype(int) - 1

        A1 = model.A1
        A2 = model.A2
        S1 = model.S1
        S2 = model.S2
        pk0 = model.pk0
        nl = self.nl
        nk = self.nk

        # ------ Impute K, Y1, Y4 on jdata ------- #
        i = 0
        for k1 in range(nk):
            I = np.arange(i, i + NNs[k1])
            ni = len(I)
            J1[I] = k1

            # Draw k
            draw_vals = np.arange(nl)
            Ki = np.random.choice(draw_vals, size=ni, replace=True, p=pk0[0, k1, :])
            K[I] = Ki

            # Draw Y2, Y3
            Y1[I] = A1[k1, Ki] + S1[k1, Ki] * np.random.normal(size=ni)
            Y2[I] = A2[k1, Ki] + S2[k1, Ki] * np.random.normal(size=ni)

            i += NNs[k1]

        sdatae = pd.DataFrame(data={'k': K, 'y1': Y1, 'y2': Y2, 'j1': J1, 'j2': J1, 'x': 1})

        return sdatae

    def _m2_mixt_simulate_sim(self, model, fsize, mmult=1, smult=1):
        '''
        Simulates data (movers and stayers) and attached firms ids. Firms have all same expected size.

        Returns:
            sim (dict):
                'jdata': movers
                'sdata': stayers
        '''
        jdata = self._m2_mixt_simulate_movers(model, model.NNm * mmult)
        sdata = self._m2_mixt_simulate_stayers(model, model.NNs * smult)

        # Create some firm ids
        sdata['f1'] = np.hstack(np.roll(sdata.groupby('j1').apply(lambda df: np.random.randint(low=0, high=len(df) // fsize + 1, size=len(df))), -1)) # Random number generation, roll is required because f1 is -1 for empty rows but they appear at the end of the dataframe
        sdata['f1'] = 'F' + (sdata['j1'].astype(int) + sdata['f1']).astype(str)
        sdata['j1b'] = sdata['j1']
        sdata['j1true'] = sdata['j1']
        jdata['j1c'] = jdata['j1']
        jdata['j1true'] = jdata['j1']
        jdata['j2true'] = jdata['j2']
        jdata['f1'] = np.hstack(jdata.groupby('j1c').apply(lambda df: np.random.choice(sdata.loc[sdata['j1b'].isin(jdata['j1c']), 'f1'].unique(), size=len(df))))
        jdata['j2c'] = jdata['j2']
        jdata['f2'] = np.hstack(jdata.groupby('j2c').apply(lambda df: np.random.choice(sdata.loc[sdata['j1b'].isin(jdata['j2c']), 'f1'].unique(), size=len(df))))
        jdata = jdata.drop(['j1c', 'j2c'], axis=1)
        sdata = sdata.drop(['j1b'], axis=1)
        sdata['f2'] = sdata['f1']

        sim = {'jdata': jdata, 'sdata': sdata}
        return sim

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
#     J1 = np.zeros(shape=np.sum(NNsx)) - 1
#     J2 = np.zeros(shape=np.sum(NNsx)) - 1
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
#             J1[I] = l1

#             # Draw k
#             draw_vals = np.arange(nk)
#             Ki = np.random.choice(draw_vals, size=ni, replace=True, p=pk0[x, l1, :])
#             K[I] = Ki
#             X[I] = x

#             # Draw Y2, Y3
#             Y1[I] = A1[l1, Ki] + S1[l1, Ki] * np.random.normal(size=ni)
#             Y2[I] = A2[l1, Ki] + S2[l1, Ki] * np.random.normal(size=ni)

#             i = i + NNsx[x,l1]

#     sdatae = pd.DataFrame(data={'k': K, 'y1': Y1, 'y2': Y2, 'j1': J1, 'j2': J1, 'x': X})

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
#     jj = jdatae['j1'] + nf * (jdatae['j2'] - 1)
#     draw_vals = np.arange(nk)
#     Ki = np.random.choice(draw_vals, size=ni, replace=True, p=pk1[jj, :])
#     # Draw Y1, Y4
#     Y1 = A1[jdatae['j1'], Ki] + S1[jdatae['j1'], Ki] * np.random.normal(size=ni)
#     Y2 = A2[jdatae['j2'], Ki] + S2[jdatae['j2'], Ki] * np.random.normal(size=ni)
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
#     Ki = np.random.choice(draw_vals, size=ni, replace=True, p=pk0[sdatae['x'], sdatae['j1'], :])
#     # Draw Y2, Y3
#     Y1 = A1[sdatae['j1'], Ki] + S1[sdatae['j1'], Ki] * np.random.normal(size=ni)
#     Y2 = A2[sdatae['j1'], Ki] + S2[sdatae['j1'], Ki] * np.random.normal(size=ni) # False for movers
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
#     J1m = jdatae.j1
#     J2m = jdatae.j2
#     JJm = J1m + nf * (J2m - 1)
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
#     Dkj1f = np.kron(np.kron(np.eye(nf), np.ones((nf, 1))), np.eye(nk)) # A[k,l] coefficients for j1
#     Dkj2f = np.kron(np.kron(np.ones((nf, 1)), np.eye(nf)), np.eye(nk)) # A[k,l] coefficients for j2

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
#                     I = (J1m == l1) & (J2m == l2)
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
#             lik = liks + lik_prior
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
#                     I = (J1m == l1) & (J2m == l2)
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
#                     I = (J1m == l1) & (J2m == l2)
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
#                 I = (JJm == jj)
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

    # model.NNm = acast(jdatae[:, .N, list(j1, j2)], j1~j2, fill=0, value.var='N') # FIXME
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
    J1  = sdata.j1   # Wage in period 1
    X   = sdata.x    # Observable category
    # @todo add code in case X is missing, just set it to one
    nx = len(np.unique(X))
    N = len(Y1)
    Wmu = model.A1.T
    Wsg = model.S1.T

    # We create the index for the movement
    # This needs to take into account the observable X
    J1x = X + nx * (J1 - 1) # Joint in index for movement
    J1s = csc_matrix(np.zeros(shape=nf * nx), shape=(N, nf * nx))
    II = np.arange(N * J1x) # FIXME was 1:N + N * (J1x - 1)
    J1s[II] = 1
    tot_count = spread(np.sum(J1s, axis=0), 2, nk).T # FIXME
    empty_cells = tot_count[1, :] == 0

    #PI = rdirichlet(nf*nx,rep(1,nk))
    PI = rdim(model.pk0, nf * nx, nk) # FIXME
    PI_old = PI

    lik_old = np.inf
    iter_start = 1

    for count in range(iter_start, ctrl.maxiter):
        # The coeffs on the pis are the sum of the norm pdf
        norm1 = norm.ppf(spread(Y1, 2, nk), Wmu[:, J1].T, Wsg[:, J1].T) # FIXME
        tau = PI[J1x, :] * norm1
        tsum = np.sum(tau, axis=1)
        tau = tau / spread(tsum, 2, nk) # FIXME
        lik = - np.sum(np.log(tsum))

        PI = (tau.T @ J1s / tot_count).T
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
    model.NNs = sdata[:, len(sdata) - 1, j1][sorted(j1)][:, N - 1] # FIXME j1 is not defined

    return model
