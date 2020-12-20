"""
    Estimates the CRE model and computes posterior using Trace
    Approximation.
"""

import pyamg
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, coo_matrix, diags, linalg, eye
import time
import pyreadr
import os
import logging
from multiprocessing import Pool, TimeoutError
from timeit import default_timer as timer

import argparse
import json
import itertools

# try to use tqdm
try:
    from tqdm import tqdm,trange
except ImportError:
    trange = range

def pipe_qcov(df,e1,e2):
    v1 = df.eval(e1)
    v2 = df.eval(e2)
    return(np.cov(v1,v2)[0][1])

def expand_grid(data_dict):
    rows = itertools.product(*data_dict.values())
    return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def pd_to_np(df,colr,colc,colv,nr,nc):
    row_index = df[colr].to_numpy()
    col_index = df[colc].to_numpy()
    values = df[colv].to_numpy()
    A = np.zeros((nr,nc))

    for i in range(nr):
        for j in range(nc):
            I = (row_index==i+1) & (col_index==j+1)
            if I.sum()>0:
                A[i,j] = values[I][0]

    return(A)
    # pd_to_np(df,'i','j','v',3,3)


class CREsolver:
    """
    Uses multigrid and partialing out to solve two way Fixed Effect model
    """
    def __init__(self,adata,sdata,jdata, wo_btw=False):
        self.nf = adata.f1i.max()
        self.nc = jdata.j1.max()
        self.nw = adata.wid.max()
        self.nn = len(adata)
        self.nm = len(jdata)
        self.ns = len(sdata)

        mdata = adata.query('cs==1')
        mdata = mdata.set_index(pd.Series(range(len(mdata))))
        Yq = mdata['y1']

        logger.info("data nf:{} nw:{} nn:{} nc:{}".format(self.nf, self.nw, self.nn, self.nc))

        sdata,jdata = self.estimate_between_cluster(sdata,jdata,wo_btw=wo_btw)
        self.estimate_within_cluster(sdata,jdata)
        self.estimate_within_parameters()

        # compute the between terms
        self.res = {}
        cdata = pd.concat( [sdata[['y1','psi1_tmp','mx','f1i']],jdata[['y1','psi1_tmp','mx','f1i']]],axis=0)
        cov_mat_between = cdata.cov()
        self.res['var_bw'] = cov_mat_between['psi1_tmp'].get('psi1_tmp')
        self.res['cov_bw'] = cov_mat_between['psi1_tmp'].get('mx')

        # compute the within terms
        self.res['var_wt'] = self.within_params['var_psi']
        self.res['cov_wt'] = (self.ns * self.within_params['cov_AsPsi1'] +self.nm * self.within_params['cov_Am1Psi1'])/(self.ns +self.nm)
        self.res['tot_var'] = self.res['var_bw'] + self.res['var_wt']
        self.res['var_y']=np.var(Yq)

        logger.info('[cre] VAR bw={:4f} wt={:4f} tot={:4f}'.format(self.res['var_bw'],self.res['var_wt'],self.res['var_bw'] + self.res['var_wt']))
        logger.info('[cre] COV bw={:4f} wt={:4f} tot={:4f}'.format(self.res['cov_bw'],self.res['cov_wt'],self.res['cov_bw'] + self.res['cov_wt']))

        # ------ STARTING POSTERIOR FOR VAR IN DIFF -----------
        jdata['val'] = 1
        J1 = csc_matrix((jdata.val, (jdata.index, jdata.f1i - 1)), shape=(self.nm, self.nf))
        J2 = csc_matrix((jdata.val, (jdata.index, jdata.f2i - 1)), shape=(self.nm, self.nf))
        Jd = J2 - J1
        Yd = jdata.eval('y2 - y1') # create the difference Y
        mdata = adata.query('cs==1')
        mdata = mdata[ ~pd.isnull(mdata['f1i'])]

        nnq = len(mdata)
        Jq = csc_matrix((np.ones(nnq), (range(nnq), mdata.f1i - 1)), shape=(nnq, self.nf)) # get the weighting for the cross-section
        Yq = mdata['y1']
        self.nnq = len(cdata)
        self.Jd = Jd
        self.Yd = Yd
        self.Jq = Jq

        logger.info("prepare linear solver")
        M = 1/self.within_params['var_psi'] * eye(self.nf) + 1/self.within_params['var_eps'] * Jd.transpose() * Jd
        self.ml = pyamg.ruge_stuben_solver(M)

        # store the prior in diff
        jdata_f = pd.concat( [jdata[['f1i','j1']],jdata[['f2i','j2']].rename(columns={'f2i':'f1i','j2':'j1'})]).drop_duplicates()
        Jf = csc_matrix((np.ones(len(jdata_f)), (jdata_f.f1i - 1,jdata_f.j1-1)), shape=(len(jdata_f) , self.nc))
        self.Mud = Jf * self.between_params['Afill']

        self.last_invert_time=0

    def estimate_between_cluster(self,sdata,jdata,wo_btw=False):
        """
        Takes sdata and jdata and extracts cluster levels means of firm effects and average value of worker effects
        :param sdata:
        :param jdata:
        :return:
        """
        # Matrices for group level estimation
        J1c = csc_matrix((np.ones(self.nm), (jdata.index, jdata.j1 - 1)), shape=(self.nm, self.nc))
        J2c = csc_matrix((np.ones(self.nm), (jdata.index, jdata.j2 - 1)), shape=(self.nm, self.nc))
        Jc  = J2c - J1c
        Jc  = Jc[:, range(self.nc - 1)]  # normalizing last group to 0
        Yc  = jdata['y2'] - jdata['y1']

        pb = {} # parameters between clusters

        # Extract CRE means
        Mc = Jc.transpose() * Jc
        A  = linalg.spsolve(Mc,Jc.transpose() * Yc)
        if wo_btw:
            A = A*0.0
        pb['Afill'] = np.append(A,0)

        jdata['psi1_tmp'] = pb['Afill'][jdata['j1']-1]
        jdata['psi2_tmp'] = pb['Afill'][jdata['j2']-1]

        EEm = jdata.assign(mx = lambda df:  0.5*(df.y2 - df.psi2_tmp +  df.y1 - df.psi1_tmp)).groupby(['j1','j2'])['mx'].agg('mean')
        if wo_btw:
            EEm = EEm*0.0
        jdata = pd.merge(jdata,EEm,on=('j1','j2'))
        pb['EEm'] = pd_to_np(EEm.reset_index(),'j1','j2','mx',self.nc,self.nc)
        #pb['EEm'] = np.array(EEm.values).reshape(self.nc,self.nc)
        #print(pd_to_np(EEm.reset_index(),'j1','j2','mx',self.nc,self.nc) - np.array(EEm.values).reshape(self.nc,self.nc) )

        sdata['psi1_tmp'] = pb['Afill'][sdata['j1']-1]
        Em = sdata.assign(mx = lambda df:  df.y1 - df.psi1_tmp ).groupby(['j1'])['mx'].agg('mean')
        if wo_btw:
            Em = Em*0.0
        sdata = pd.merge(sdata,Em,on=('j1'))
        pb['Em'] = np.array(Em.values)

        # # let's also regress residuals on j1,j2
        # Ed = Yc - Jc * A
        # Jcv  = J2c + J1c
        # Mcv = Jcv.transpose() * Jcv
        # S  = linalg.spsolve(Mcv, Jcv.transpose() * (Yc * Ed))
        # self.Esd = np.sqrt(np.maximum(0,S))

        self.between_params = pb

        return(sdata,jdata)

    def estimate_within_cluster(self,sdata,jdata):

        res = {}

        # we construct wages net of between group means
        dm = jdata.eval( 'y1n =  y1 - psi1_tmp - mx ' ) \
                  .eval( 'y2n =  y2 - psi2_tmp - mx ') \
                  [['y1n', 'y2n', 'j1', 'j2', 'f1i', 'f2i']]
        ds  = sdata.eval( 'y1n = y1 - psi1_tmp - mx' )[['y1n', 'j1', 'f1i']]

        # get averages by firms for stayers
        dsf  = ds.groupby('f1i').agg(y1sj =('y1n','mean'), nsj = ('y1n','count'))
        # get averages by firms for movers leaving the firm
        dm1f = dm.groupby('f1i').agg(y1m1j=('y1n','mean'), y2m1j=('y2n','mean'), nm1j= ('y1n','count'))
        # get averages by firms for movers joining the firm
        dm2f = dm.groupby('f2i').agg(y1m2j=('y1n','mean'), y2m2j=('y2n','mean'), nm2j= ('y2n','count'))

        # get averages by firms and jo (cluster worker moves to) to create leave same cluster destination out
        dm1c = dm.groupby(['f1i','j2']).agg(y1m1c=('y1n','mean'), y2m1c=('y2n','mean'), nm1c= ('y1n','count'))
        dm2c = dm.groupby(['f2i','j1']).agg(y1m2c=('y1n','mean'), y2m2c=('y2n','mean'), nm2c= ('y2n','count'))

        # merge averages back into data
        ds  = pd.merge(ds,  dsf,  on = "f1i")
        ds  = pd.merge(ds,  dm1f, on = "f1i")
        ds  = pd.merge(ds,  dm2f, left_on = "f1i", right_on = "f2i")
        dm  = pd.merge(dm,  dm1f, on = "f1i")
        dm  = pd.merge(dm,  dm2f, on = "f2i")
        dm  = pd.merge(dm,  dm1c, on = ["f1i","j2"])
        dm  = pd.merge(dm,  dm2c, on = ["f2i","j1"])

        # create leaveout means
        ds.eval("y1s_lo    = (nsj * y1sj - y1n) / (nsj - 1)",inplace=True)
        # for each observation we remove from the mean value, all movers that move the
        # same cluster, this includes the individuals himself, as well as workers that move
        # to or from the same firm (we want to not use joint moves as the psi in the other period would be the same
        # and hence would be corrolated)
        dm.eval("y1m1j_lo  = (nm1j * y1m1j -  nm1c * y1m1c) / (nm1j - nm1c)",inplace=True)
        dm.eval("y2m1j_lo  = (nm1j * y2m1j -  nm1c * y2m1c) / (nm1j - nm1c)",inplace=True)
        dm.eval("y1m2j_lo  = (nm2j * y1m2j -  nm2c * y1m2c) / (nm2j - nm2c)",inplace=True)
        dm.eval("y2m2j_lo  = (nm2j * y2m2j -  nm2c * y2m2c) / (nm2j - nm2c)",inplace=True)

        # compute the moments involving stayers
        res['y1s_y1s']   = ds.query('nsj  > 1').pipe(pipe_qcov, 'y1n', 'y1s_lo')
        res['y1s_y1s_count']   = ds.query('nsj  > 1').shape[0]
        res['y1s_var'] = ds['y1n'].var()
        res['y1s_var_count'] = ds.shape[0]
        res['y1m_var'] = dm['y1n'].var()
        res['y1m_var_count'] = dm.shape[0]
        res['y2m_var'] = dm['y2n'].var()
        res['y2m_var_count'] = dm.shape[0]

        # compute the moments involving movers leaving the firm
        res['y1s_y1m1']  = ds.query('nm1j > 0').pipe(pipe_qcov, 'y1n', 'y1m1j')
        res['y1s_y1m1_count']  = ds.query('nm1j > 0').shape[0]
        res['y1s_y2m1']  = ds.query('nm1j > 0').pipe(pipe_qcov, 'y1n', 'y2m1j')
        res['y1s_y2m1_count']  = ds.query('nm1j > 0').shape[0]
        res['y1m1_y1m1'] = dm.query('nm1j > nm1c').pipe(pipe_qcov, 'y1n', 'y1m1j_lo')
        res['y1m1_y1m1_count'] = dm.query('nm1j > nm1c').shape[0]
        res['y2m1_y1m1'] = dm.query('nm1j > nm1c').pipe(pipe_qcov, 'y2n', 'y1m1j_lo')
        res['y2m1_y1m1_count'] = dm.query('nm1j > nm1c').shape[0]
        res['y2m1_y2m1'] = dm.query('nm1j > nm1c').pipe(pipe_qcov, 'y2n', 'y2m1j_lo')
        res['y2m1_y2m1_count'] = dm.query('nm1j > nm1c').shape[0]

        # compute the moments involving movers arriving at the firm
        res['y1s_y1m2']  = ds.query('nm2j > 0').pipe(pipe_qcov, 'y1n', 'y1m2j')
        res['y1s_y1m2_count']  = ds.query('nm2j > 0').shape[0]
        res['y1s_y2m2']  = ds.query('nm2j > 0').pipe(pipe_qcov, 'y1n', 'y2m2j')
        res['y1s_y2m2_count']  = ds.query('nm2j > 0').shape[0]
        res['y1m2_y1m2'] = dm.query('nm2j > nm2c').pipe(pipe_qcov, 'y1n', 'y1m2j_lo')
        res['y1m2_y1m2_count'] = dm.query('nm2j > nm2c').shape[0]
        res['y2m2_y1m2'] = dm.query('nm2j > nm2c').pipe(pipe_qcov, 'y2n', 'y1m2j_lo')
        res['y2m2_y1m2_count'] = dm.query('nm2j > nm2c').shape[0]
        res['y2m2_y2m2'] = dm.query('nm2j > nm2c').pipe(pipe_qcov, 'y2n', 'y2m2j_lo')
        res['y2m2_y2m2_count'] = dm.query('nm2j > nm2c').shape[0]

        # total variance of wages in differences for movers
        res['dym_dym'] = dm.query('j1 != j2').eval('y2n-y1n').var()
        res['dym_dym_count'] = dm.query('j1 != j2').shape[0]
        res['y1m_y2m'] = dm.query('j1 != j2').pipe(pipe_qcov, 'y1n', 'y2n')
        res['y1m_y2m_count'] = dm.query('j1 != j2').shape[0]

        self.moments_within = res

    def estimate_within_parameters(self):

        pw = {}
        # using movers leaving from firm
        pw['cov_Am1Am1']  = self.moments_within["y2m1_y2m1"]
        pw['cov_Am1Psi1'] = self.moments_within["y2m1_y1m1"] - pw['cov_Am1Am1']
        pw['var_psi_m1']  = self.moments_within["y1m1_y1m1"] - pw['cov_Am1Am1'] - 2 * pw['cov_Am1Psi1']

        # using movers arriving in firm
        pw[ 'cov_Am2Am2' ]  = self.moments_within['y1m2_y1m2']
        pw[ 'cov_Am2Psi2' ] = self.moments_within['y2m2_y1m2'] - pw['cov_Am2Am2']
        pw[ 'var_psi_m2' ]  = self.moments_within['y2m2_y2m2'] - pw['cov_Am2Am2'] - 2 * pw['cov_Am2Psi2']

        # looking at stayers
        pw[ 'cov_AsAm1' ] = self.moments_within['y1s_y2m1'] - pw['cov_Am1Psi1']
        pw[ 'cov_AsAm2' ] = self.moments_within['y1s_y1m2'] - pw['cov_Am2Psi2']
        pw[ 'psi_plus_cov1' ] = self.moments_within['y1s_y1m1'] - self.moments_within['y1s_y2m1']
        pw[ 'psi_plus_cov2' ] = self.moments_within['y1s_y2m2'] - self.moments_within['y1s_y1m2']

        pw[ 'var_psi' ] = (pw['var_psi_m2'] + pw['var_psi_m1']) / 2
        pw[ 'cov_AsPsi1' ] = pw['psi_plus_cov1'] + pw['psi_plus_cov2'] - pw['var_psi']
        pw[ 'cov_AsAs' ] = self.moments_within["y1s_y1s"] - pw['var_psi'] - 2 * pw['cov_AsPsi1']

        pw[ 'var_eps' ] = np.maximum(0 , self.moments_within['dym_dym'] - 2 * pw[ 'var_psi' ])

        self.within_params = pw

    # def estimate_within_woodcock(self):

    #     pw = {}
    #     # using movers leaving from firm
    #     pw['woodock_var_psi'] = (self.moments_within["y1s_y1s"]*self.moments_within["y1s_y1s_count"] + 
    #                             self.moments_within["y1m1_y1m1"]*self.moments_within["y1m1_y1m1_count"] +
    #                             self.moments_within["y2m2_y2m2"]*self.moments_within["y2m2_y2m2_count"]) / (
    #                                 self.moments_within["y1s_y1s_count"] +self.moments_within["y1m1_y1m1_count"] +
    #                                 self.moments_within["y2m2_y2m2_count"])
    #     pw['woodock_var_alpha'] = self.moments_within["y1s_y1s"]

    #     pw['woodock_var_eps']  = ( self.moments_within["y1s_var"]*self.moments_within["y1s_var_count"] + 
    #                                self.moments_within["y1m_var"]*self.moments_within["y1s_var_count"] + 
    #                                self.moments_within["y2m_var"]*self.moments_within["y2s_var_count"] ) / 
    #                                (self.moments_within["y1s_var_count"] +self.moments_within["y1s_var_count"] + self.moments_within["y2s_var_count"]  ) 

    #     self.within_params_woodcock = pw

    def compute_posterior_var(self,ndraw_trace):
        """
        compute the posterior variance of the CRE model
        :return:
        """

        # we first compute the direct term
        v1 = 1/self.within_params['var_psi'] * self.Mud
        v1 += 1/self.within_params['var_eps'] * self.Jd.transpose() * self.Yd
        v2 = self.Jq * self.ml.solve( v1 )
        t1 = np.var(v2)

        v3 = self.Jq * self.Mud

        # next we look at the trace term
        tr_var_pos_all = np.zeros(ndraw_trace)
        for r in trange(ndraw_trace):
            Zpsi   = 2 * np.random.binomial(1, 0.5, self.nf) - 1

            R1 = self.Jq * Zpsi
            R2 = self.Jq * self.ml.solve( Zpsi )
            tr_var_pos_all[r] = np.cov(R1,R2)[0][1]

        t2 = np.mean(tr_var_pos_all)
        self.posterior_var = t1 + t2

        logger.info("[cre]  posterior variance of psi = {:4f}".format(self.posterior_var) )

def main(args):
    ncore           = args['ncore']   # number of cores to use
    ndraw_trace     = args['ndraw_tr']  # number of draws to compute hetero correction

    """
    In R do 
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

    res = {}
    res["ndt"] = ndraw_trace

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

    sdata['j1'] = sdata['j1'].astype(int)
    sdata['j2'] = sdata['j2'].astype(int)
    jdata['j1'] = jdata['j1'].astype(int)
    jdata['j2'] = jdata['j2'].astype(int)

    logger.info("data movers={} stayers={}".format(len(jdata),len(sdata)))
    res['nm'] = len(jdata)
    res['ns'] = len(sdata)
    res['n_firms']=len(np.unique(pd.concat([jdata['f1i'],jdata['f2i'],sdata['f1i']], ignore_index=True)))
    res['n_workers']=len(np.unique(pd.concat([jdata['wid'], sdata['wid']], ignore_index=True)))

    # make wids unique per rows
    jdata.set_index( np.arange(res['nm'])+ 1)
    sdata.set_index( np.arange(res['ns'])+ 1 + res['nm'])

    # combining the 2 data-sets
    adata = pd.concat( [ sdata[['wid','f1i','y1']].assign(cs=1,m=0),
                         jdata[['wid','f1i','y1']].assign(cs=1,m=1),
                         jdata[['wid','f2i','y2']].rename(columns={'f2i':'f1i','y2':'y1'}).assign(cs=0,m=1) ])
    adata = adata.set_index(pd.Series(range(len(adata))))
    adata['wid'] = adata['wid'].astype('category').cat.codes + 1

    res['var_y'] = adata.query('cs==1')['y1'].var()
    logger.info("total variance: {:0.4f}".format(res['var_y']))

    fes = CREsolver(adata,sdata,jdata,wo_btw=args['wobtw'])

    res.update(fes.moments_within)
    res.update(fes.within_params)
    res.update(fes.res)
    #res.update(fes.between_params)

    if args['posterior']:
        fes.compute_posterior_var(ndraw_trace)
        res['var_posterior'] = fes.posterior_var

    # Saving to file
    # Convert results into strings to prevent JSON errors
    for key, val in res.items():
            res[key] = str(val)

    with open(args['out'],'w') as outfile:
        json.dump(res, outfile)
    logger.info("saved results to {}".format(args['out']))

    logger.info("------ DONE -------")

    return res

# Begin logging
logger = logging.getLogger('cre')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('cre_spam.log')
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
