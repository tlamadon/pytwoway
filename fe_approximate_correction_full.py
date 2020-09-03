"""
    This is a program that will computes a bunch of estimates 
    from an event study data set:
     - AKM variance decomposition
     - Andrews bias correction
     - KSS bias correction


From R, you can export data:
    f1s = jdata[,unique(c(f1,f2))]
    fids = data.table(f1=f1s,nfid=1:length(f1s))
    setkey(fids,f1)
    setkey(jdata,f1)
    jdata[,f1i := fids[jdata,nfid]]
    setkey(sdata,f1)
    sdata[,f1i := fids[sdata,nfid]]
    setkey(jdata,f2)
    jdata[,f2i := fids[jdata,nfid]]
    
    jdata = as.data.frame(jdata)
    sdata = as.data.frame(sdata)
    saveRDS(sdata,file="~/Dropbox/paper-small-firm-effects/results/simsdata.rds")
    saveRDS(jdata,file="~/Dropbox/paper-small-firm-effects/results/simjdata.rds")
"""

import pyamg
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix,coo_matrix,diags,linalg
import time
import pyreadr
import os
import logging
from multiprocessing import Pool, TimeoutError, set_start_method
from timeit import default_timer as timer
import itertools
import pickle
import time
import argparse
import json
import glob, sys

def pipe_qcov(df,e1,e2):
    v1 = df.eval(e1)
    v2 = df.eval(e2)
    return(np.cov(v1,v2)[0][1])

# try to use tqdm
try:
    from tqdm import tqdm,trange
except ImportError:
    trange = range

class FEsolver:
    """
    Uses multigrid and partialing out to solve two way Fixed Effect model
    """

    def __init__(self,adata):
        nf = adata.f1i.max()
        nw = adata.wid.max()
        nn = len(adata)
        logger.info("data nf:{} nw:{} nn:{}".format(nf, nw, nn))

        # Matrices for the cross-section
        J = csc_matrix((np.ones(nn), (adata.index, adata.f1i - 1)), shape=(nn, nf))
        J = J[:, range(nf - 1)]  # normalizing one firm to 0
        W = csc_matrix((np.ones(nn), (adata.index, adata.wid - 1)), shape=(nn, nw))
        Dw = diags((W.transpose() * W).diagonal())
        Dwinv = diags(1.0 / ((W.transpose() * W).diagonal()))

        logger.info("prepare linear solver")
        # finally we create M
        M = J.transpose() * J - J.transpose() * W * Dwinv * W.transpose() * J
        self.ml = pyamg.ruge_stuben_solver(M)

        # L = diags(fes.M.diagonal()) - fes.M
        # r = linalg.eigsh(L,k=2,which='LM')

        # create cross-section matrices
        mdata = adata.query('cs==1')
        mdata = mdata.set_index(pd.Series(range(len(mdata))))

        nnq = len(mdata)
        Jq = csc_matrix((np.ones(nnq), (mdata.index, mdata.f1i - 1)), shape=(nnq, nf))
        self.Jq = Jq[:, range(nf - 1)]  # normalizing one firm to 0
        self.Wq = csc_matrix((np.ones(nnq), (mdata.index, mdata.wid - 1)), shape=(nnq, nw))
        self.nnq = nnq
        self.Yq = mdata['y1']

        # save all variables
        self.nf = nf
        self.nn = nn
        self.nw = nw
        self.M = M
        self.J = J
        self.W = W
        self.Dwinv = Dwinv
        self.last_invert_time=0

    def __getstate__(self):
        """ defines how the model is pickled """
        odict = {k: self.__dict__[k] for k in self.__dict__.keys() - {'ml'}}
        return odict

    def __setstate__(self, dict):
        """ defines how the model is unpickled """
        # need to recreate the simple model and the seasrch representation
        self.__dict__ = dict     # make dict our attribute dictionary
        self.ml = pyamg.ruge_stuben_solver(self.M)

    @staticmethod
    def load(filename):
        fes= None
        with open(filename, "rb") as infile:
            fes =  pickle.load(infile)
        return fes

    def save(self,filename):
        with open(filename, "wb") as output_file:
            pickle.dump(self, output_file)

    def solve(self,Y):
        psi, alpha = self.mult_Atranspose(Y)
        psi_hat, alpha_hat = self.mult_AAinv(psi, alpha)
        return(psi_hat, alpha_hat)

    def mult_A(self,psi,alpha):
        return self.J*psi + self.W*alpha

    def mult_Atranspose(self,v):
        return self.J.transpose()*v , self.W.transpose()*v

    def mult_AAinv(self,psi,alpha):
        # inter1 = self.ml.solve( psi , tol=1e-10 )
        # inter2 = self.ml.solve(  , tol=1e-10 )
        # psi_out = inter1 - inter2

        start = timer()
        psi_out = self.ml.solve( psi - self.J.transpose() * (self.W * (self.Dwinv * alpha)), tol=1e-10 )
        self.last_invert_time = timer() - start

        alpha_out = - self.Dwinv * (self.W.transpose() * (self.J * psi_out)) + self.Dwinv * alpha
        return psi_out,alpha_out

    def proj(self,y):
        return(self.mult_A(*self.solve(y)))

    def leverage_approx(self,ndraw_pii):
        """ computes an approximate leverage using ndraw_pii """
        Pii = np.zeros(self.nn)

        # we compute the different draws
        for r in trange(ndraw_pii):
            R2  = 2 * np.random.binomial(1, 0.5, self.nn) - 1
            Pii += 1 / ndraw_pii * np.power( self.proj(R2) , 2.0)
 
        return(Pii)

def weighted_var(v,w):
    m0 = np.sum( w * v) / np.sum(w)
    v0 = np.sum( w * ( v-m0 )**2 ) / np.sum(w) 
    return v0

def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


# simple wrapper to call the leverage function
def leverage_approx_func(fes,ndraw_pii): 
    res = fes.leverage_approx(ndraw_pii)
    logger.info("done with batch")
    return res


def main(args):
    ncore = args['ncore']   # number of cores to use
    ndraw_pii = args['ndraw_pii']  # number of draws to compute leverage
    ndraw_trace = args['ndraw_tr']  # number of draws to compute hetero correction
    compute_hetero = args['hetero']

    start_time = time.time()

    res = {}
    res['ndt'] = ndraw_trace
    res['ndp'] = ndraw_pii
    res['cores'] = ncore

    # ----------------- LOADING AND PREPARING DATA -----------------
    # import sdata/jdata
    logger.info("loading the data")

    for i in range(0,30):
        try:
            data = args['data']
            sdata = data[data['m'] == 0].reset_index(drop=True)
            jdata = data[data['m'] == 1].reset_index(drop=True)
        except pyreadr.custom_errors.LibrdataError:
            logger.info("can't read file, waiting 30s and trying again {}/30".format(i+1))
            time.sleep(30)
            continue
        break

    logger.info("data movers={} stayers={}".format(len(jdata),len(sdata)))
    res['nm'] = len(jdata)
    res['ns'] = len(sdata)
    res['n_firms']=len(np.unique(pd.concat([jdata['f1i'],jdata['f2i'],sdata['f1i']], ignore_index=True)))
    res['n_workers']=len(np.unique(pd.concat([jdata['wid'], sdata['wid']], ignore_index=True)))
    res['n_movers']=len(np.unique(pd.concat([jdata['wid']], ignore_index=True)))
    #res['year_max'] = int(sdata['year'].max())
    #res['year_min'] = int(sdata['year'].min())

    # make wids unique per rows
    jdata.set_index( np.arange(res['nm'])+ 1)
    sdata.set_index( np.arange(res['ns'])+ 1 + res['nm'])
    jdata['wid'] = np.arange(res['nm']) + 1
    sdata['wid'] = np.arange(res['ns'])+ 1 + res['nm']

    # combining the 2 data-sets
    adata = pd.concat( [ sdata[['wid','f1i','y1']].assign(cs=1,m=0),
                         jdata[['wid','f1i','y1']].assign(cs=1,m=1),
                         jdata[['wid','f2i','y2']].rename(columns={'f2i':'f1i','y2':'y1'}).assign(cs=0,m=1) ])
    adata = adata.set_index(pd.Series(range(len(adata))))
    adata['wid'] = adata['wid'].astype('category').cat.codes + 1

    # extract some stats
    fdata = adata.groupby('f1i').agg({'m':'sum', 'y1':'mean', 'wid':'count' })
    res['mover_quantiles'] = weighted_quantile(fdata['m'], np.linspace(0,1,11), fdata['wid'] ).tolist()
    res['size_quantiles'] = weighted_quantile(fdata['wid'], np.linspace(0,1,11), fdata['wid'] ).tolist()
    res['between_firm_var'] = weighted_var(fdata['y1'], fdata['wid'] )
    res['var_y'] = adata.query('cs==1')['y1'].var()

    # extract woodcock moments using sdata and jdata
    # get averages by firms for stayers
    #dsf  = adata.query('cs==1').groupby('f1i').agg(y1sj=('y1','mean'), nsj=('y1','count'))
    #ds   = pd.merge(adata.query('cs==1'), dsf, on="f1i")
    #ds.eval("y1s_lo    = (nsj * y1sj - y1) / (nsj - 1)",inplace=True)
    #res['woodcock_var_psi']   = ds.query('nsj  > 1').pipe(pipe_qcov, 'y1', 'y1s_lo')
    #res['woodcock_var_alpha'] = np.minimum( jdata.pipe(pipe_qcov, 'y1','y2'), adata.query('cs==1')['y1'].var() - res['woodcock_var_psi'] )
    #res['woodcock_var_eps'] = adata.query('cs==1')['y1'].var() - res['woodcock_var_alpha'] - res['woodcock_var_psi']
    #logger.info("[woodcock] var psi = {}", res['woodcock_var_psi'])
    #logger.info("[woodcock] var alpha = {}", res['woodcock_var_alpha'])
    #logger.info("[woodcock] var eps = {}", res['woodcock_var_eps'])

    del(sdata)
    del(jdata)

    if args['statsonly']:
        with open(args['out'],'w') as outfile:
            json.dump(res,outfile)
        logger.info("saved results to {}".format(args['out']))
        logger.info(" --statsonly was passed as argument, so we skip all estimation.")
        logger.info("------ DONE -------")
        sys.exit()
 


    # ------------- creating fixed effect solver -----------
    fes = FEsolver(adata)
    Y = adata.y1

    # try to pickle the object to see its size
    # fes.save("tmp.pkl")

    logger.info("extract firm effects")
    psi_hat, alpha_hat = fes.solve(Y)
    logger.info("solver time {:2.4f} seconds".format(fes.last_invert_time))
    logger.info("expected total time {:2.4f} minutes".format( (ndraw_trace * (1 + compute_hetero) + ndraw_pii * compute_hetero ) *  fes.last_invert_time / 60    ))
    E = Y - fes.mult_A(psi_hat  , alpha_hat)

    res["solver_time"] = fes.last_invert_time

    fe_rsq = 1 - np.power(E, 2).mean() / np.power(Y, 2).mean()
    logger.info("Fixed effect R-square {:2.4f}".format(fe_rsq))

    var_fe = np.var( fes.Jq * psi_hat)
    cov_fe = np.cov( fes.Jq * psi_hat, fes.Wq * alpha_hat)[0][1]
    tot_var  = np.var(Y)
    logger.info("[fe] var_psi={:2.4f} cov={:2.4f} tot={:2.4f}".format(var_fe,cov_fe,np.var(Y)))

    var_e = fes.nn/(fes.nn - fes.nw - fes.nf + 1 ) * np.power(E ,2).mean()
    logger.info("[ho] variance of residuals {:2.4f}".format(var_e))

    # -------- COMPUTING leverages Pii ---------
    # we start by computing the sigma_i
    Pii = np.zeros(fes.nn)
    Sii = np.zeros(fes.nn)
    if compute_hetero:

        if (len(args['levfile'])>1):
            logger.info("[he] starting heteroskedastic correction, loading precomputed files")

            files = glob.glob("{}*".format(args['levfile'])) 
            logger.info("[he] found {} files to get leverages from".format(len(files)))
            res['lev_file_count']=len(files)
            assert len(files)>0, "didn't find any leverage files!"


            for f in files:
                pp = np.load(f)
                Pii += pp/len(files)

        elif (ncore>1):
            logger.info("[he] starting heteroskedastic correction p2={}, using {} cores, batch size {}".format(ndraw_pii,ncore,args['batch']))
            set_start_method("spawn")
            with Pool(processes=ncore) as pool:
                Pii_all =  pool.starmap( leverage_approx_func, [ (fes,args['batch']) for _ in range( ndraw_pii // args['batch'] )] )

            for pp in Pii_all:
                Pii += pp/len(Pii_all)
        else:
            Pii_all =  list(itertools.starmap( leverage_approx_func, [ (fes,args['batch']) for _ in range( ndraw_pii // args['batch'] )] ))
            
            for pp in Pii_all:
                Pii += pp/len(Pii_all)
                    
                
        I = 1.0 * adata.eval('m==1')
        max_leverage = (I*Pii).max()

        # we attach the computed Pii to the data.frame
        adata["Pii"] = Pii
        logger.info("[he] Leverage range {:2.4f} to {:2.4f}".format(adata.query('m==1').Pii.min(),adata.query('m==1').Pii.max()))

        # we give stayers the variance estimate at the firm level
        adata["Sii"] = Y * E / (1 - Pii)
        S_j = adata.query('m==1').rename(columns={'Sii':'Sii_j'}).groupby('f1i')['Sii_j'].agg('mean')

        adata = pd.merge(adata,S_j,on="f1i")
        adata['Sii'] = np.where(adata['m']==1, adata['Sii'], adata['Sii_j'])
        Sii = adata['Sii']

        logger.info("[he] variance of residuals in heteroskedastic case: {:2.4f}".format(Sii.mean()))

    # ------ computing trace approximation ------
    logger.info("starting trace correction ndraws={}, using {} cores".format(ndraw_trace,ncore))
    tr_var_ho_all = np.zeros(ndraw_trace)
    tr_cov_ho_all = np.zeros(ndraw_trace)
    tr_var_he_all = np.zeros(ndraw_trace)
    tr_cov_he_all = np.zeros(ndraw_trace)
    for r in trange(ndraw_trace):

        Zpsi   = 2 * np.random.binomial(1, 0.5, fes.nf - 1) - 1
        Zalpha = 2 * np.random.binomial(1, 0.5, fes.nw) - 1

        # terms for homoskedastic
        R1 = fes.Jq * Zpsi
        psi1,alpha1 = fes.mult_AAinv(Zpsi,Zalpha)
        R2_psi = fes.Jq * psi1
        R2_alpha = fes.Wq * alpha1

        # trace corrections
        tr_var_ho_all[r] = np.cov(R1,R2_psi)[0][1]
        tr_cov_ho_all[r] = np.cov(R1,R2_alpha)[0][1]

        # terms for heteroskedastic
        if compute_hetero:
            psi2,alpha2 = fes.mult_AAinv(*fes.mult_Atranspose(Sii * fes.mult_A(Zpsi,Zalpha)))
            R3_psi = fes.Jq * psi2

            # trace corrections
            tr_var_he_all[r] = np.cov(R2_psi,R3_psi)[0][1]
            tr_cov_he_all[r] = np.cov(R2_alpha,R3_psi)[0][1]

        logger.debug("[traces] step {}/{} done.".format(r, ndraw_trace))


    end_time = time.time()

    # collecting the results
    res['tot_var'] = tot_var
    res['eps_var_ho'] = var_e
    res['eps_var_fe'] = np.var(E)
    res['tr_var_ho'] = np.mean(tr_var_ho_all)
    res['tr_cov_ho'] = np.mean(tr_cov_ho_all)
    res['tr_var_he'] = np.mean(tr_var_he_all)
    res['tr_cov_he'] = np.mean(tr_cov_he_all)
    logger.info("[ho] VAR tr={:2.4f} (sd={:2.4e})".format(res['tr_var_ho'], np.std(tr_var_ho_all)))
    logger.info("[ho] COV tr={:2.4f} (sd={:2.4e})".format(res['tr_cov_ho'], np.std(tr_cov_ho_all)))
    if compute_hetero:
        res['eps_var_he'] = Sii.mean()
        res['min_lev'] = adata.query('m==1').Pii.min()
        res['max_lev'] = adata.query('m==1').Pii.max()
        res['tr_var_ho_sd'] = np.std(tr_var_ho_all)
        res['tr_cov_ho_sd'] = np.std(tr_cov_ho_all)
        res['tr_var_he_sd'] = np.std(tr_var_he_all)
        res['tr_cov_he_sd'] = np.std(tr_cov_he_all)
        logger.info("[he] VAR tr={:2.4f} (sd={:2.4e})".format(res['tr_var_he'], np.std(tr_var_he_all)))
        logger.info("[he] COV tr={:2.4f} (sd={:2.4e})".format(res['tr_cov_he'], np.std(tr_cov_he_all)))

    # # ----- FINAL ------
    logger.info("[ho] VAR fe={:2.4f} bc={:2.4f}".format(var_fe, var_fe - var_e * res['tr_var_ho'] ))
    logger.info("[ho] COV fe={:2.4f} bc={:2.4f}".format(cov_fe, cov_fe - var_e * res['tr_cov_ho'] ))
    if compute_hetero:
        logger.info("[he] VAR fe={:2.4f} bc={:2.4f}".format(var_fe, var_fe - res['tr_var_he']))
        logger.info("[he] COV fe={:2.4f} bc={:2.4f}".format(cov_fe, cov_fe - res['tr_cov_he']))

    res['var_y']  = np.var(fes.Yq)
    res['var_fe'] = var_fe
    res['cov_fe'] = cov_fe
    res['var_ho'] = var_fe - var_e * res['tr_var_ho']
    res['cov_ho'] = cov_fe - var_e * res['tr_cov_ho']

    if compute_hetero:
        res['var_he'] = var_fe - res['tr_var_he']
        res['cov_he'] = cov_fe - res['tr_cov_he']

    res["total_time"] = end_time-start_time
    # Saving to file
    with open(args['out'],'w') as outfile:
        json.dump(res,outfile)
    logger.info("saved results to {}".format(args['out']))

    logger.info("------ DONE -------")

# Begin logging
logger = logging.getLogger('akm')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('akm_spam.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
