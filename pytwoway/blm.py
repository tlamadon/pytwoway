"""
    We implement the non-linear estimator from Bonhomme Lamadon & Manresa
"""

import numpy as np
from scipy.special import logsumexp

def lognormpdf(x,mu,sd):
    return( -0.5*np.log(2*pi) - np.log(sd) - (x - mu)**2/(2*sd**2) )

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    """
    wrapper for quadprog 
    """

    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -numpy.vstack([A, G]).T
        qp_b = -numpy.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


class BLMEstimator:

    def __init__(self, opts):
        """
            Initialize the model
        """

        self.nl = 6
        self.nk = 10

        # mean of wages by firm and worker type
        self.A1    = np.zeros((self.nk, self.nl))
        self.A2    = np.zeros((self.nk, self.nl))
        # starndard deviation of wages by firm and worker type
        self.S1    = np.ones( (self.nk ,self.nl))        
        self.S2    = np.ones( (self.nk ,self.nl))        

        # model for p(K | l ,l') for movers
        self.pk1    = np.ones((self.nk,self.nk,self.nl)) / self.nl        
        # model for p(K | l ,l') for stayers
        self.pk0    = np.ones((self.nk,self.nl)) / self.nl        


    def fit(self,jdata):
        """ 
            We write the EM algorithm for the movers
        """
        ni = nrow(jdata)

        # store wage outcomes and groups
        Y1 = jdata.y1.to_numpy() 
        Y2 = jdata.y2.to_numpy()
        J1 = jdata.j1.to_numpy()
        J2 = jdata.j2.to_numpy()

        # matrix of posterior probabilities
        qi =  np.ones( (ni ,self.nl))

        for iter in range(maxiter):

            # -------- E-Step ---------
            # we compute the posterior probabiluties for each row
            # we iterate over the worker types, should not be be 
            # to costly since the vector is quite large within each iteration
            for l in range(p.nl): 
                lp1 = lognormpdf( Y1, self.A1[J1,l], self.S1[J1,l])
                lp2 = lognormpdf( Y2, self.A2[J2,l], self.S2[J2,l])
                lp[:,l] = np.log( self.pk1[J1,J2,l] ) + lp1 + lp2
        
            # we compute log sum exp to get likelihoods and probabilities
            qi = np.exp(  lp -  logsumexp(lp,axis=2) )
            liks = logsumexp(lp,axis=0).sum()

            # --------- M-step ----------
            # for now we run a simple ols, however later we
            # want to add constraints!
            # see https://scaron.info/blog/quadratic-programming-in-python.html

            # the regression has 2 * nl * nk parameters and nl * ni rows
            # we do not necessarly want to construct the duplicated data by nl
            # instead we will construct X'X and X'Y by looping over nl
            # we also note that X'X is block diagonal with 2*nl matrices of dimensions nk^2
            for l in range(p.nl): 
                lr = range( l*p.nk, (l+1)*p.nk) # range that selects corresponding block
                t2 = p.nk*p.nl                  # shift for period 2
                XwX[lr,lr]       = J1m * (qi[:,l]/S1[J1,l]) * J1m.transpose() 
                # here want to compute the matrix multiplication with a diagonal mattrix in the middle, 
                # we might be better off trying this within numba or something.
                XwY[lr]          = J1m * (qi[:,l]/S1[J1,l]) * Y1
                XwX[lr+ts,lr+ts] = J2m * (qi[:,l]/S2[J2,l]) * J2m.transpose() 
                XwY[lr+ts]       = J2m * (qi[:,l]/S2[J2,l]) * Y2
            
            # we solve the system to get all the parameters
            # we need to add the constraints here using quadprog
            res_a   = solve(XwX, XwY)
            self.A1 = reshape(res_a, [nl,nk,2])[:,:,1]
            self.A2 = reshape(res_a, [nl,nk,2])[:,:,2]

            # next we extract the variances
            for l in range(p.nl): 
                lr = range( l*p.nk, (l+1)*p.nk) # range that selects corresponding block
                t2 = p.nk*p.nl                  # shift for period 2
                XwS[lr]          = J1m * (qi[:,l]/self.S1[J1,l]) * (Y1 - self.A1[J1,l])
                XwS[lr+ts]       = J2m * (qi[:,l]/self.S2[J2,l]) * (Y2 - self.A2[J2,l])

            res_s = solve(XwX, XwS)
            self.S1 = reshape(res_s, [nl,nk,2])[:,:,1]
            self.S2 = reshape(res_s, [nl,nk,2])[:,:,2]



