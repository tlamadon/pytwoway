'''
Defines class CREEstimator, which uses a trace approximation to estimate the CRE model.
'''
# import logging
# from pathlib import Path
import pyamg
import numpy as np
import pandas as pd
from bipartitepandas.util import ParamsDict, logger_init
from scipy.sparse import csc_matrix, coo_matrix, diags, linalg, eye
import time
# import pyreadr
# import os
# from multiprocessing import Pool, TimeoutError
# from timeit import default_timer as timer
# import argparse
import json
# import itertools
from tqdm.auto import tqdm, trange

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq1(a):
    return a >= 1

# Define default parameter dictionary
_cre_params_default = ParamsDict({
    'ncore': (1, 'type_constrained', (int, _gteq1),
        '''
            (default=1) Number of cores to use.
        ''', '>= 1'),
    'posterior': (False, 'type', bool,
        '''
            (default=False) If True, compute posterior variance.
        ''', None),
    'wo_btw': (False, 'type', bool,
        '''
            (default=False) If True, sets between variation to 0, pure RE.
        ''', None),
    'ndraw_trace': (5, 'type_constrained', (int, _gteq1),
        '''
            (default=5) Number of draws to use in trace approximations.
        ''', '>= 1'),
    'outputfile': (None, 'type_none', str,
        '''
            (default=None) Outputfile where results will be saved in json format. If None, results will not be saved.
        ''', None)
    # 'ndp': (50, 'type_constrained', (int, _gteq1), # FIXME not used
    #     '''
    #         (default=50) Number of draws to use in approximation of leverages.
    #     ''', '>= 1')
})

def cre_params(update_dict=None):
    '''
    Dictionary of default cre_params. Run tw.cre_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict or None): user parameter values; None is equivalent to {}

    Returns:
        (ParamsDict) dictionary of cre_params
    '''
    new_dict = _cre_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

def _pipe_qcov(df, e1, e2):
    v1 = df.eval(e1)
    v2 = df.eval(e2)
    return(np.cov(v1, v2)[0][1])

# def expand_grid(data_dict): # FIXME This is not used anywhere
#     rows = itertools.product(*data_dict.values())
#     return pd.DataFrame.from_records(rows, columns=data_dict.keys())

def _pd_to_np(df, colr, colc, colv, nr, nc):
    row_index = df[colr].to_numpy()
    col_index = df[colc].to_numpy()
    values = df[colv].to_numpy()
    A = np.zeros((nr, nc))

    for i in range(nr):
        for j in range(nc):
            I = (row_index == i) & (col_index == j)
            if I.sum() > 0:
                A[i, j] = values[I][0]

    return(A)
    # _pd_to_np(df, 'i', 'j', 'v', 3, 3)

class CREEstimator:
    '''
    Uses multigrid and partialing out to solve two way Fixed Effect model.
    '''
    def __init__(self, data, params=None):
        '''
        Arguments:
            data (Pandas DataFrame): cross-section labor data. Data contains the following columns:

                i (worker id)

                j1 (firm id 1)

                j2 (firm id 2)

                y1 (compensation 1)

                y2 (compensation 2)

                t1 (last period of observation 1)

                t2 (last period of observation 2)

                w1 (weight 1)

                w2 (weight 2)

                m (0 if stayer, 1 if mover)

                cs (0 if not in cross section, 1 if in cross section)
            cre_params (ParamsDict or None): dictionary of parameters for CRE estimation. Run tw.cre_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.cre_params().
        '''
        # Start logger
        logger_init(self)
        # self.logger.info('initializing CREEstimator object')

        if params is None:
            params = cre_params()

        self.adata = data

        self.params = params
        self.res = {} # Results dictionary
        self.summary = {} # Summary results dictionary

        ## Save some commonly used parameters as attributes
        # Number of cores to use
        self.ncore = params['ncore']
        # Number of draws to use in trace approximations
        self.ndraw_trace = params['ndraw_trace']
        # If True, sets between variation to 0, pure RE
        self.wo_btw = params['wo_btw']

        ## Store some parameters in results dictionary
        self.res['cores'] = self.ncore
        self.res['ndt'] = self.ndraw_trace

        # self.logger.info('CREEstimator object initialized')

    def fit(self, rng=None):
        '''
        Run CRE solver.

        Arguments:
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        self.logger.info('----- STARTING CRE ESTIMATION -----')
        self.start_time = time.time()

        # Begin cleaning and analysis
        self.__prep_vars() # Prepare data

        # Generate stayers and movers, and set indices so they don't overlap
        jdata = self.adata[(self.adata['m'] > 0) & (self.adata['cs'] == 1)].reset_index(drop=True)
        self.mn = len(jdata) # Number of observations from movers # FIXME I renamed from nm to mn, since nm makes it seem like it's the number of movers, while mn gives movers, where n is the total number of observations
        ########################################
        ##### NOTE: if you code crashes on the next line, it is likely because your data is not collapsed #####
        ########################################
        sdata = self.adata[(self.adata['m'] == 0) & (self.adata['cs'] == 1)].set_index(np.arange(self.ns) + self.mn) # FIXME could be np.arange(len(sdata)) + self.mn for non-collapsed data, since stayers can have multiple observations - but this has issues at the moment because consecutive observations at the same firm for movers should be marked as stayers
        sdata, jdata = self.__estimate_between_cluster(sdata, jdata)
        self._estimate_within_cluster(sdata, jdata)
        self.__estimate_within_parameters()

        cdata = pd.concat([sdata[['y1', 'psi1_tmp', 'mx', 'j1']], jdata[['y1', 'psi1_tmp', 'mx', 'j1']]], axis=0)
        Yq = self.__get_Yq() # Pandas series with the cross-section income

        self.__collect_res(cdata, Yq) # Collect results
        self.__prior_in_diff(jdata) # Store the prior in diff

        if self.params['posterior']:
            self.logger.info('preparing ')
            self.__prep_posterior_var(jdata, cdata)

            self.logger.info('computing posterior variance')
            self.__compute_posterior_var(rng)
            self.res['var_posterior'] = self.posterior_var

        end_time = time.time()

        self.res['total_time'] = end_time - self.start_time
        del self.start_time

        self.__save_res() # Save results to json

        self.logger.info('----- DONE WITH CRE ESTIMATION -----')

    def __prep_vars(self):
        '''
        Generate some initial class attributes and results.
        '''
        '''
        In R do
            f1s = jdata[,unique(c(f1,f2))]
            fids = data.table(f1=f1s,nfid=1:length(f1s))
            setkey(fids,f1)
            setkey(jdata,f1)
            jdata[,j1 := fids[jdata,nfid]]
            setkey(sdata,f1)
            sdata[,j1 := fids[sdata,nfid]]
            setkey(jdata,f2)
            jdata[,j2 := fids[jdata,nfid]]

            jdata = as.data.frame(jdata)
            sdata = as.data.frame(sdata)
            saveRDS(sdata,file="~/Dropbox/paper-small-firm-effects/results/simsdata.rds")
            saveRDS(jdata,file="~/Dropbox/paper-small-firm-effects/results/simjdata.rds")
        '''
        self.logger.info('preparing the data')

        # self.adata = self.params['data']
        # self.adata['i'] = self.adata['i'].astype('category').cat.codes + 1 # FIXME i should already be correct
        self.adata[['g1', 'g2']] = self.adata[['g1', 'g2']].astype(int) # Clusters generated as Int64 which isn't compatible with indexing

        self.nf = max(self.adata['j1'].max(), self.adata['j2'].max()) + 1 # Number of firms
        self.nw = self.adata['i'].max() + 1 # Number of workers
        self.nc = max(self.adata['g1'].max(), self.adata['g2'].max()) + 1 # Number of clusters
        self.logger.info(f'data firms={self.nf} workers={self.nw} clusters={self.nc} observations={len(self.adata)}')

        nm = self.adata.loc[self.adata['m'] == 1, 'i'].nunique() # Number of movers
        self.ns = self.nw - nm # Number of stayers
        self.logger.info(f'data movers={nm} stayers={self.ns}')

        self.res['n_firms'] = self.nf
        self.res['n_workers'] = self.nw
        self.res['n_movers'] = nm
        self.res['n_stayers'] = self.ns

        # data = self.params['data']
        # sdata = self.adata[(self.adata['m'] == 0) & (self.adata['cs'] == 1)].reset_index(drop=True)
        # jdata = self.adata[(self.adata['m'] == 1) & (self.adata['cs'] == 1)].reset_index(drop=True)

        # sdata['g1'] = sdata['g1'].astype(int)
        # sdata['g2'] = sdata['g2'].astype(int)
        # jdata['g1'] = jdata['g1'].astype(int)
        # jdata['g2'] = jdata['g2'].astype(int)

        # self.jdata = jdata
        # self.sdata = sdata

        # FIXME commented out the below because wids are made unique in BipartiteData, but the below code may still be necessary
        # # Make wids unique per row
        # jdata.set_index(np.arange(self.res['nm']) + 1)
        # sdata.set_index(np.arange(self.res['ns']) + 1 + self.res['nm'])
        # self.jdata = jdata
        # self.sdata = sdata

        # # Combine the 2 data-sets
        # self.adata = pd.concat([sdata[['i', 'j1', 'y1']].assign(cs=1, m=0),
        #                         jdata[['i', 'j1', 'y1']].assign(cs=1, m=1),
        #                         jdata[['i', 'j2', 'y2']].rename(columns={'j2': 'j1', 'y2': 'y1'}).assign(cs=0, m=1)])
        # self.adata = self.adata.set_index(pd.Series(range(len(self.adata))))
        # self.adata['i'] = self.adata['i'].astype('category').cat.codes + 1

        # self.nf = self.res['n_firms']
        # self.nc = max(jdata.g1.max(), jdata.g2.max())
        # self.nw = self.res['n_workers']
        # self.nn = len(self.adata)

    def __estimate_between_cluster(self, sdata, jdata):
        '''
        Takes sdata and jdata and extracts cluster levels means of firm effects and average value of worker effects.

        Arguments:
            sdata (Pandas DataFrame): stayers
            jdata (Pandas DataFrame): movers

        Returns:
            sdata (Pandas DataFrame): @ FIXME update this
            jdata (Pandas DataFrame): @ FIXME update this
        '''
        # Matrices for group level estimation
        J1c = csc_matrix((np.ones(self.mn), (jdata.index, jdata['g1'])), shape=(self.mn, self.nc))
        J2c = csc_matrix((np.ones(self.mn), (jdata.index, jdata['g2'])), shape=(self.mn, self.nc))
        Jc = J2c - J1c
        Jc = Jc[:, range(self.nc - 1)]  # Normalizing last group to 0
        Yc = jdata['y2'] - jdata['y1']

        pb = {} # Parameters between clusters

        # Extract CRE means
        Mc = Jc.transpose() * Jc
        A = linalg.spsolve(Mc, Jc.transpose() * Yc)
        if self.wo_btw:
            A = A * 0.0
        pb['Afill'] = np.append(A, 0)

        jdata['psi1_tmp'] = pb['Afill'][jdata['g1']]
        jdata['psi2_tmp'] = pb['Afill'][jdata['g2']]

        EEm = jdata.assign(mx=lambda df: 0.5 * (df['y2'] - df['psi2_tmp'] + df['y1'] - df['psi1_tmp'])).groupby(['g1', 'g2'])['mx'].agg('mean')
        if self.wo_btw:
            EEm = EEm * 0.0
        jdata = pd.merge(jdata, EEm, on=('g1', 'g2'))
        pb['EEm'] = _pd_to_np(EEm.reset_index(), 'g1', 'g2', 'mx', self.nc, self.nc)
        #pb['EEm'] = np.array(EEm.values).reshape(self.nc, self.nc)
        #print(_pd_to_np(EEm.reset_index(), 'g1', 'g2', 'mx', self.nc, self.nc) - np.array(EEm.values).reshape(self.nc, self.nc))

        sdata['psi1_tmp'] = pb['Afill'][sdata['g1']]
        Em = sdata.assign(mx = lambda df: df['y1'] - df['psi1_tmp']).groupby(['g1'])['mx'].agg('mean')
        if self.wo_btw:
            Em = Em * 0.0
        sdata = pd.merge(sdata, Em, on=('g1'))
        pb['Em'] = np.array(Em.values)

        # # Let's also regress residuals on g1,g2
        # Ed = Yc - Jc * A
        # Jcv = J2c + J1c
        # Mcv = Jcv.transpose() * Jcv
        # S = linalg.spsolve(Mcv, Jcv.transpose() * (Yc * Ed))
        # self.Esd = np.sqrt(np.maximum(0, S))

        self.between_params = pb

        return sdata, jdata

    def _estimate_within_cluster(self, sdata, jdata):
        '''
        @ FIXME add description of function

        Arguments:
            sdata (Pandas DataFrame): stayers
            jdata (Pandas DataFrame): movers
        '''
        res = {}

        # We construct wages net of between group means
        dm = jdata.eval('y1n = y1 - psi1_tmp - mx') \
                  .eval('y2n =  y2 - psi2_tmp - mx') \
                  [['y1n', 'y2n', 'g1', 'g2', 'j1', 'j2']]
        ds  = sdata.eval('y1n = y1 - psi1_tmp - mx')[['y1n', 'g1', 'j1']]

        # Get averages by firms for stayers
        dsf  = ds.groupby('j1').agg(y1sj=('y1n', 'mean'), nsj=('y1n', 'count'))
        # Get averages by firms for movers leaving the firm
        dm1f = dm.groupby('j1').agg(y1m1j=('y1n', 'mean'), y2m1j=('y2n', 'mean'), nm1j=('y1n', 'count'))
        # Get averages by firms for movers joining the firm
        dm2f = dm.groupby('j2').agg(y1m2j=('y1n', 'mean'), y2m2j=('y2n', 'mean'), nm2j=('y2n', 'count'))

        # Get averages by firms and jo (cluster worker moves to) to create leave same cluster destination out
        dm1c = dm.groupby(['j1', 'g2']).agg(y1m1c=('y1n', 'mean'), y2m1c=('y2n', 'mean'), nm1c= ('y1n', 'count'))
        dm2c = dm.groupby(['j2', 'g1']).agg(y1m2c=('y1n', 'mean'), y2m2c=('y2n', 'mean'), nm2c= ('y2n', 'count'))

        # Merge averages back into data
        ds = pd.merge(ds, dsf, on='j1')
        ds = pd.merge(ds, dm1f, on='j1')
        ds = pd.merge(ds, dm2f, left_on='j1', right_on='j2')
        dm = pd.merge(dm, dm1f, on='j1')
        dm = pd.merge(dm, dm2f, on='j2')
        dm = pd.merge(dm, dm1c, on=['j1', 'g2'])
        dm = pd.merge(dm, dm2c, on=['j2', 'g1'])

        # Create leaveout means
        ds.eval('y1s_lo = (nsj * y1sj - y1n) / (nsj - 1)', inplace=True)
        # For each observation we remove from the mean value, all movers that move the
        # same cluster, this includes the individuals himself, as well as workers that move
        # to or from the same firm (we want to not use joint moves as the psi in the other period would be the same
        # and hence would be corrolated)
        dm.eval('y1m1j_lo = (nm1j * y1m1j - nm1c * y1m1c) / (nm1j - nm1c)', inplace=True)
        dm.eval('y2m1j_lo = (nm1j * y2m1j - nm1c * y2m1c) / (nm1j - nm1c)', inplace=True)
        dm.eval('y1m2j_lo = (nm2j * y1m2j - nm2c * y1m2c) / (nm2j - nm2c)', inplace=True)
        dm.eval('y2m2j_lo = (nm2j * y2m2j - nm2c * y2m2c) / (nm2j - nm2c)', inplace=True)

        # Compute the moments involving stayers
        res['y1s_y1s'] = ds.query('nsj > 1').pipe(_pipe_qcov, 'y1n', 'y1s_lo')
        res['y1s_y1s_count'] = ds.query('nsj > 1').shape[0]
        res['y1s_var'] = ds['y1n'].var()
        res['y1s_var_count'] = ds.shape[0]
        res['y1m_var'] = dm['y1n'].var()
        res['y1m_var_count'] = dm.shape[0]
        res['y2m_var'] = dm['y2n'].var()
        res['y2m_var_count'] = dm.shape[0]

        # Compute the moments involving movers leaving the firm
        res['y1s_y1m1'] = ds.query('nm1j > 0').pipe(_pipe_qcov, 'y1n', 'y1m1j')
        res['y1s_y1m1_count'] = ds.query('nm1j > 0').shape[0]
        res['y1s_y2m1'] = ds.query('nm1j > 0').pipe(_pipe_qcov, 'y1n', 'y2m1j')
        res['y1s_y2m1_count'] = ds.query('nm1j > 0').shape[0]
        res['y1m1_y1m1'] = dm.query('nm1j > nm1c').pipe(_pipe_qcov, 'y1n', 'y1m1j_lo')
        res['y1m1_y1m1_count'] = dm.query('nm1j > nm1c').shape[0]
        res['y2m1_y1m1'] = dm.query('nm1j > nm1c').pipe(_pipe_qcov, 'y2n', 'y1m1j_lo')
        res['y2m1_y1m1_count'] = dm.query('nm1j > nm1c').shape[0]
        res['y2m1_y2m1'] = dm.query('nm1j > nm1c').pipe(_pipe_qcov, 'y2n', 'y2m1j_lo')
        res['y2m1_y2m1_count'] = dm.query('nm1j > nm1c').shape[0]

        # Compute the moments involving movers arriving at the firm
        res['y1s_y1m2'] = ds.query('nm2j > 0').pipe(_pipe_qcov, 'y1n', 'y1m2j')
        res['y1s_y1m2_count'] = ds.query('nm2j > 0').shape[0]
        res['y1s_y2m2'] = ds.query('nm2j > 0').pipe(_pipe_qcov, 'y1n', 'y2m2j')
        res['y1s_y2m2_count'] = ds.query('nm2j > 0').shape[0]
        res['y1m2_y1m2'] = dm.query('nm2j > nm2c').pipe(_pipe_qcov, 'y1n', 'y1m2j_lo')
        res['y1m2_y1m2_count'] = dm.query('nm2j > nm2c').shape[0]
        res['y2m2_y1m2'] = dm.query('nm2j > nm2c').pipe(_pipe_qcov, 'y2n', 'y1m2j_lo')
        res['y2m2_y1m2_count'] = dm.query('nm2j > nm2c').shape[0]
        res['y2m2_y2m2'] = dm.query('nm2j > nm2c').pipe(_pipe_qcov, 'y2n', 'y2m2j_lo')
        res['y2m2_y2m2_count'] = dm.query('nm2j > nm2c').shape[0]

        # Total variance of wages in differences for movers
        res['dym_dym'] = dm.query('g1 != g2').eval('y2n-y1n').var()
        res['dym_dym_count'] = dm.query('g1 != g2').shape[0]
        res['y1m_y2m'] = dm.query('g1 != g2').pipe(_pipe_qcov, 'y1n', 'y2n')
        res['y1m_y2m_count'] = dm.query('g1 != g2').shape[0]

        self.moments_within = res
        self.res.update(self.moments_within)

    def __estimate_within_parameters(self):
        '''
        @ FIXME add description of function
        '''
        pw = {}
        # Using movers leaving from firm
        pw['cov_Am1Am1'] = self.moments_within['y2m1_y2m1']
        pw['cov_Am1Psi1'] = self.moments_within['y2m1_y1m1'] - pw['cov_Am1Am1']
        pw['var_psi_m1'] = self.moments_within['y1m1_y1m1'] - pw['cov_Am1Am1'] - 2 * pw['cov_Am1Psi1']

        # Using movers arriving in firm
        pw['cov_Am2Am2'] = self.moments_within['y1m2_y1m2']
        pw['cov_Am2Psi2'] = self.moments_within['y2m2_y1m2'] - pw['cov_Am2Am2']
        pw['var_psi_m2'] = self.moments_within['y2m2_y2m2'] - pw['cov_Am2Am2'] - 2 * pw['cov_Am2Psi2']

        # looking at stayers
        pw['cov_AsAm1'] = self.moments_within['y1s_y2m1'] - pw['cov_Am1Psi1']
        pw['cov_AsAm2'] = self.moments_within['y1s_y1m2'] - pw['cov_Am2Psi2']
        pw['psi_plus_cov1'] = self.moments_within['y1s_y1m1'] - self.moments_within['y1s_y2m1']
        pw['psi_plus_cov2'] = self.moments_within['y1s_y2m2'] - self.moments_within['y1s_y1m2']

        pw['var_psi'] = (pw['var_psi_m2'] + pw['var_psi_m1']) / 2
        pw['cov_AsPsi1'] = pw['psi_plus_cov1'] + pw['psi_plus_cov2'] - pw['var_psi']
        pw['cov_AsAs'] = self.moments_within['y1s_y1s'] - pw['var_psi'] - 2 * pw['cov_AsPsi1']

        pw['var_eps'] = np.maximum(0, self.moments_within['dym_dym'] - 2 * pw['var_psi'])

        self.within_params = pw
        self.res.update(self.within_params)

    # def estimate_within_woodcock(self):

    #     pw = {}
    #     # Using movers leaving from firm
    #     pw['woodock_var_psi'] = (self.moments_within['y1s_y1s'] * self.moments_within['y1s_y1s_count'] +
    #                             self.moments_within['y1m1_y1m1'] * self.moments_within['y1m1_y1m1_count'] +
    #                             self.moments_within['y2m2_y2m2'] * self.moments_within['y2m2_y2m2_count']) / (
    #                                 self.moments_within['y1s_y1s_count'] + self.moments_within['y1m1_y1m1_count'] +
    #                                 self.moments_within['y2m2_y2m2_count'])
    #     pw['woodock_var_alpha'] = self.moments_within['y1s_y1s']

    #     pw['woodock_var_eps']  = ( self.moments_within['y1s_var'] * self.moments_within['y1s_var_count'] +
    #                                self.moments_within['y1m_var'] * self.moments_within['y1s_var_count'] +
    #                                self.moments_within['y2m_var'] * self.moments_within['y2s_var_count'] ) /
    #                                (self.moments_within['y1s_var_count'] + self.moments_within['y1s_var_count'] + self.moments_within['y2s_var_count'])

    #     self.within_params_woodcock = pw

    def __get_Yq(self):
        '''
        Generate Yq, the Pandas series with the cross-section income.

        Returns:
            Yq (Pandas Series): cross-section income
        '''
        mdata = self.adata[self.adata['cs'] == 1] # FIXME changed from adata.query('cs==1') (I ran %timeit and slicing is faster)
        mdata = mdata.reset_index(drop=True) # FIXME changed from set_index(pd.Series(range(len(mdata))))
        Yq = mdata['y1']

        return Yq

    def __collect_res(self, cdata, Yq):
        '''
        Compute the within terms.

        Arguments:
            cdata (Pandas DataFrame): movers and stayers
            Yq (Pandas Series): income for movers
        '''
        self.res['var_y'] = Yq.var()
        self.logger.info(f"total variance: {self.res['var_y']:0.4f}")

        # Compute the between terms
        cov_mat_between = cdata.cov()
        self.res['var_bw'] = cov_mat_between['psi1_tmp'].get('psi1_tmp')
        self.res['cov_bw'] = cov_mat_between['psi1_tmp'].get('mx')

        # Compute the within terms
        self.res['var_wt'] = self.within_params['var_psi']
        self.res['cov_wt'] = (self.ns * self.within_params['cov_AsPsi1'] + self.mn * self.within_params['cov_Am1Psi1']) / (self.ns + self.mn)
        self.res['tot_var'] = self.res['var_bw'] + self.res['var_wt']
        self.res['tot_cov'] = self.res['cov_bw'] + self.res['cov_wt']
        self.res['var_y'] = np.var(Yq)

        self.logger.info(f"[cre] VAR bw={self.res['var_bw']:4f} wt={self.res['var_wt']:4f} tot={self.res['var_bw'] + self.res['var_wt']:4f}")
        self.logger.info(f"[cre] COV bw={self.res['cov_bw']:4f} wt={self.res['cov_wt']:4f} tot={self.res['cov_bw'] + self.res['cov_wt']:4f}")

        # Create summary variable
        self.summary['var_y'] = self.res['var_y']
        self.summary['var_bw'] = self.res['var_bw']
        self.summary['cov_bw'] = self.res['cov_bw']
        self.summary['var_tot'] = self.res['tot_var']
        self.summary['cov_tot'] = self.res['tot_cov']

    def __prior_in_diff(self, jdata):
        '''
        Store the prior in diff.

        Arguments:
            jdata (Pandas DataFrame): movers
        '''
        jdata_f = pd.concat([jdata[['j1', 'g1']], jdata[['j2', 'g2']].rename(columns={'j2': 'j1', 'g2': 'g1'})]).drop_duplicates()
        Jf = csc_matrix((np.ones(len(jdata_f)), (jdata_f['j1'], jdata_f['g1'])), shape=(self.nf, self.nc)) # Need self.nf because recall there can be firms in j2 that aren't in j1
        self.Mud = Jf * self.between_params['Afill']

    def __prep_posterior_var(self, jdata, cdata):
        '''
        Prepare data for computing the posterior variance of the CRE model.

        Arguments:
            jdata (Pandas DataFrame): movers
            cdata (Pandas DataFrame): movers and stayers, dataframe created when computing the between terms
        '''
        jdata['val'] = 1
        J1 = csc_matrix((jdata['val'], (jdata.index, jdata['j1'])), shape=(self.mn, self.nf))
        J2 = csc_matrix((jdata['val'], (jdata.index, jdata['j2'])), shape=(self.mn, self.nf))
        Jd = J2 - J1
        Yd = jdata.eval('y2 - y1') # Create the difference Y
        mdata = self.adata.query('cs == 1')
        mdata = mdata[~pd.isnull(mdata['j1'])]

        nnq = len(mdata)
        Jq = csc_matrix((np.ones(nnq), (range(nnq), mdata['j1'])), shape=(nnq, self.nf)) # Get the weighting for the cross-section
        # Yq = mdata['y1'] # FIXME commented this out
        # self.nnq = len(cdata) # FIXME commented this out
        self.Jd = Jd
        self.Yd = Yd
        self.Jq = Jq

        self.logger.info('preparing linear solver')
        M = 1 / self.within_params['var_psi'] * eye(self.nf) + 1 / self.within_params['var_eps'] * Jd.transpose() * Jd
        self.ml = pyamg.ruge_stuben_solver(M)

    def __compute_posterior_var(self, rng=None):
        '''
        Compute the posterior variance of the CRE model.

        Arguments:
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # We first compute the direct term
        v1 = 1 / self.within_params['var_psi'] * self.Mud
        v1 += 1 / self.within_params['var_eps'] * self.Jd.transpose() * self.Yd
        v2 = self.Jq * self.ml.solve(v1)
        t1 = np.var(v2)

        # v3 = self.Jq * self.Mud # FIXME commented this out

        # Next we look at the trace term
        tr_var_pos_all = np.zeros(self.ndraw_trace)
        pbar = trange(self.ndraw_trace)
        pbar.set_description('cre')
        for r in pbar:
            Zpsi = 2 * rng.binomial(1, 0.5, self.nf) - 1

            R1 = self.Jq * Zpsi
            R2 = self.Jq * self.ml.solve(Zpsi)
            tr_var_pos_all[r] = np.cov(R1, R2)[0][1]

        t2 = np.mean(tr_var_pos_all)
        self.posterior_var = t1 + t2

        self.logger.info(f'[cre] posterior variance of psi = {self.posterior_var:4f}')

    def __save_res(self):
        '''
        Save results as json.
        '''
        outputfile = self.params['outputfile']
        if outputfile is not None:
            # Convert results into strings to prevent JSON errors
            for key, val in self.res.items():
                self.res[key] = str(val)

            with open(outputfile, 'w') as outfile:
                json.dump(self.res, outfile)

            self.logger.info(f'saved results to {outputfile}')
        else:
            self.logger.info('outputfile=None, so results not saved')
