'''
    We implement the non-linear estimator from Bonhomme Lamadon & Manresa
'''

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from numpy import pi
import argparse

# Create a random model for EM with
# endogenous mobility with multinomial pr

def m2_mixt_new(nk, nf, fixb=False, stationary=False):
    '''
    Returns:
        model (Pandas DataFrame):
    '''
    model = argparse.Namespace()

    # model for Y1|Y2,l,k for movers and stayes
    model.A1 = 0.9 * (1 + 0.5 * np.random.normal(size=[nf, nk]))
    model.S1 = 0.3 * (1 + 0.5 * np.random.uniform(size=[nf, nk]))
    # model for Y4|Y3,l,k for movers and stayes
    model.A2 = 0.9 * (1 + 0.5 * np.random.normal(size=[nf, nk]))
    model.S2 = 0.3 * (1 + 0.5 * np.random.uniform(size=[nf, nk]))
    # model for p(K | l ,l') for movers
    model.pk1 = np.random.dirichlet(alpha=[1] * nk, size=nf * nf)
    # model for p(K | l ,l') for stayers
    model.pk0 = np.random.dirichlet(alpha=[1] * nk, size=nf)
    model.pk0 = np.expand_dims(model.pk0, axis=0)

    model.NNm = np.random.randint(low=1, high=nf + 1, size=[nf, nf])

    model.nk = nk
    model.nf = nf

    for l in range(nf):
        model.A1[l, :] = sorted(model.A1[l, :])
        model.A2[l, :] = sorted(model.A2[l, :])

    if fixb:
        model.A3 = pd.pivot_table(np.mean(model.A2, axis=1), columns=[2], values=nk) + model.A1 - pd.pivot_table(np.mean(model.A1, axis=1), columns=[2], values=nk)
        # model$A2 = spread(as.data.frame(rowMeans(model$A2)),2,nk) + model$A1 - spread(as.data.frame(rowMeans(model$A1)),2,nk)
        # NOTE: I couldn't get this code to run in R

    if stationary:
        model.A2 = model.A1

    return model

# ------------- Simulating functions ---------------------

# Using the model, simulates a dataset of movers
def m2_mixt_simulate_movers(model, NNm=np.nan):
    '''
    Returns:
        jdatae (Pandas DataFrame):
    '''
    J1 = np.zeros(shape=np.sum(NNm))
    J2 = np.zeros(shape=np.sum(NNm))
    Y1 = np.zeros(shape=np.sum(NNm))
    Y2 = np.zeros(shape=np.sum(NNm))
    K = np.zeros(shape=np.sum(NNm))

    A1 = model.A1
    A2 = model.A2
    S1 = model.S1
    S2 = model.S2
    pk1 = model.pk1
    nk = model.nk
    nf = model.nf

    i = 1
    for l1 in range(nf):
        for l2 in range(nf):
            I = np.arange(i, i + NNm[l1, l2])
            ni = len(I)
            jj = l1 + nf * (l2 - 1)
            J1[I] = l1
            J2[I] = l2

            # Draw k
            draw_vals = np.arange(1, nk + 1)
            Ki = np.random.choice(draw_vals, size=ni, replace=True, p=pk1[jj, :])
            K[I] = Ki

            # Draw Y2, Y3
            Y1[I] = A1[l1, Ki] + S1[l1, Ki] * np.random.normal(size=ni)
            Y2[I] = A2[l2, Ki] + S2[l2, Ki] * np.random.normal(size=ni)

            i += NNm[l1, l2]

    jdatae = pd.DataFrame(data={'k': K, 'y1': Y1, 'y2': Y2, 'j1': J1, 'j2': J2})

    return jdatae

# Using the model, simulates a dataset of stayers
def m2_mixt_simulate_stayers(model, NNs):
    '''
    Returns:
        sdatae (Pandas DataFrame):
    '''
    J1 = np.zeros(shape=np.sum(NNs))
    J2 = np.zeros(shape=np.sum(NNs))
    Y1 = np.zeros(shape=np.sum(NNs))
    Y2 = np.zeros(shape=np.sum(NNs))
    K  = np.zeros(shape=np.sum(NNs))

    A1 = model.A1
    A2 = model.A2
    S1 = model.S1
    S2 = model.S2
    pk0 = model.pk0
    nk = model.nk
    nf = model.nf

    # ------ Impute K, Y1, Y4 on jdata ------- #
    i = 1
    for l1 in range(nf):
        I = np.arange(i, i + NNs[l1])
        ni = len(I)
        J1[I] = l1

        # Draw k
        draw_vals = np.arange(1, nk + 1)
        Ki = np.random.choice(draw_vals, size=ni, replace=True, p=pk0[1, l1, :])
        K[I] = Ki

        # Draw Y2, Y3
        Y1[I] = A1[l1, Ki] + S1[l1, Ki] * np.random.normal(size=ni)
        Y2[I] = A2[l1, Ki] + S2[l1, Ki] * np.random.normal(size=ni)

        i += NNs[l1]

    sdatae = pd.DataFrame(data={'k': K, 'y1': Y1, 'y2': Y2, 'j1': J1, 'j2': J1, 'x': 1})

    return sdatae

# Using the model, simulates a dataset of stayers
def m2_mixt_simulate_stayers_withx(model, NNsx):
    '''
    Returns:
        sdatae (Pandas DataFrame):
    '''
    J1 = np.zeros(shape=np.sum(NNsx))
    J2 = np.zeros(shape=np.sum(NNsx))
    Y1 = np.zeros(shape=np.sum(NNsx))
    Y2 = np.zeros(shape=np.sum(NNsx))
    K = np.zeros(shape=np.sum(NNsx))
    X = np.zeros(shape=np.sum(NNsx))

    A1 = model.A1
    A2 = model.A2
    S1 = model.S1
    S2 = model.S2
    pk0 = model.pk0
    nk = model.nk
    nf = model.nf
    nx = len(NNsx)

    # ------ Impute K, Y1, Y4 on jdata ------- #
    i = 1
    for l1 in range(nf):
        for x in range(nx):
            I = np.arange(i, i + NNsx[x, l1])
            ni = len(I)
            J1[I] = l1

            # Draw k
            draw_vals = np.arange(1, nk + 1)
            Ki = np.random.choice(draw_vals, size=ni, replace=True, p=pk0[x, l1, :])
            K[I] = Ki
            X[I] = x

            # Draw Y2, Y3
            Y1[I] = A1[l1, Ki] + S1[l1, Ki] * np.random.normal(size=ni)
            Y2[I] = A2[l1, Ki] + S2[l1, Ki] * np.random.normal(size=ni)

            i = i + NNsx[x,l1]

    sdatae = pd.DataFrame(data={'k': K, 'y1': Y1, 'y2': Y2, 'j1': J1, 'j2': J1, 'x': X})

    return sdatae

def m2_mixt_impute_movers(model, jdatae):
    '''
    '''
    A1 = model.A1
    S1 = model.S1
    pk1 = model.pk1
    A2 = model.A2
    S2 = model.S2
    nk = model.nk
    nf = model.nf

    # ------ Impute K, Y1, Y4 on jdata ------- #
    jdatae.sim = jdatae.copy(deep=True)
    # Generate Ki, Y1, Y4
    # FIXME the follow code probably doesn't run
    ni = len(jdatae)
    jj = jdatae['j1'] + nf * (jdatae['j2'] - 1)
    draw_vals = np.arange(1, nk + 1)
    Ki = np.random.choice(draw_vals, size=ni, replace=True, p=pk1[jj, :])
    # Draw Y1, Y4
    Y1 = A1[jdatae['j1'], Ki] + S1[jdatae['j1'], Ki] * np.random.normal(size=ni)
    Y2 = A2[jdatae['j2'], Ki] + S2[jdatae['j2'], Ki] * np.random.normal(size=ni)
    # Append Ki, Y1, Y4 to jdatae.sim
    jdatae.sim[['k_imp', 'y1_imp', 'y2_imp']] = [Ki, Y1, Y2]

    return jdatae.sim

def m2_mixt_impute_stayers(model, sdatae):
    '''
    '''
    A1 = model.A1
    S1 = model.S1
    pk0 = model.pk0
    A2 = model.A2
    S2 = model.S2
    nk = model.nk
    nf = model.nf

    # ------ Impute K, Y2, Y3 on sdata ------- #
    sdatae.sim = sdatae.copy(deep=True)
    # Generate Ki, Y2, Y3
    # FIXME the follow code probably doesn't run
    ni = len(sdatae)
    draw_vals = np.arange(1, nk + 1)
    Ki = np.random.choice(draw_vals, size=ni, replace=True, p=pk0[sdatae['x'], sdatae['j1'], :])
    # Draw Y2, Y3
    Y1 = A1[sdatae['j1'], Ki] + S1[sdatae['j1'], Ki] * np.random.normal(size=ni)
    Y2 = A2[sdatae['j1'], Ki] + S2[sdatae['j1'], Ki] * np.random.normal(size=ni) # False for movers
    # Append Ki, Y2, Y3 to sdatae.sim
    sdatae.sim[['k_imp', 'y1_imp', 'y2_imp']] = [Ki, Y1, Y2]

    return sdatae.sim

# Simulates data (movers and stayers) and attached firms ids. Firms have all same expected size
# model = m2_mixt_new(nk, nf, fixb=False, stationary=False)
def m2_mixt_simulate_sim(model, fsize, mmult=1, smult=1):
    '''
    '''
    jdata = m2_mixt_simulate_movers(model, model.NNm * mmult)
    sdata = m2_mixt_simulate_stayers(model, model.NNs * smult)

    # Create some firm ids
    sdata <- sdata[,f1 := paste("F",j1 + model$nf*(sample.int(.N/fsize,.N,replace=T)-1),sep=""),j1]
    sdata <- sdata[,j1b:=j1]
    sdata <- sdata[,j1true := j1]
    jdata <- jdata[,j1true := j1][,j2true := j2]
    jdata <- jdata[,j1c:=j1]
    jdata <- jdata[,f1:=sample( unique(sdata[j1b %in% j1c,f1]) ,.N,replace=T),j1c]
    jdata <- jdata[,j2c:=j2]
    jdata <- jdata[,f2:=sample( unique(sdata[j1b %in% j2c,f1])  ,.N,replace=T),j2c]
    jdata.j2c = None
    jdata.j1c = None
    sdata.j1b = None
    sdata['f2'] = sdata['f1']
    sdata[,f2:=f1]

    sim = {'sdata': sdata, 'jdata': jdata}
    return sim

####################
##### New Code #####
####################

def lognormpdf(x, mu, sd):
    return - 0.5 * np.log(2 * pi) - np.log(sd) - (x - mu) ** 2 / (2 * sd ** 2)

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



