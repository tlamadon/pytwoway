'''
Computes a bunch of estimates from an event study data set:

    - AKM variance decomposition
    - Andrews bias correction
    - KSS bias correction

Does this through class FEsolver
'''

import pyamg
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, coo_matrix, diags, linalg
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

# Try to use tqdm
try:
    from tqdm import tqdm,trange
except ImportError:
    trange = range

# def pipe_qcov(df, e1, e2): # FIXME I moved this from above, also this is used only in commented out code
#     v1 = df.eval(e1)
#     v2 = df.eval(e2)
#     return np.cov(v1, v2)[0][1]

class FEsolver:
    '''
    Uses multigrid and partialing out to solve two way Fixed Effect model

    @ FIXME I think delete everything below this, it's basically contained in the class/functions within the class

    takes as an input this adata
    and creates associated A = [J W] matrix which are AKM dummies

    provides methods to do A x Y but also (A'A)^-1 A'Y solve method

    Arguments:
        params (dictionary): dictionary of parameters

            Dictionary parameters:

                data (Pandas DataFrame): labor data. Contains the following columns:

                    wid (worker id)

                    y1 (compensation 1)

                    y2 (compensation 2)

                    f1i (firm id 1)

                    f2i (firm id 2)

                    m (0 if stayer, 1 if mover)

                ncore (int): number of cores to use

                ndraw_pii (int): number of draws to compute leverage

                ndraw_tr (int): number of draws to compute heteroskedastic correction

                hetero (bool): if True, compute heteroskedastic correction

                statsonly (bool): if True, return only basic statistics

                out (string): if statsonly is True, this is the file where the statistics will be saved

                batch (): @ FIXME I don't know what this is

                Q (str): which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'
    '''

    def __init__(self, params):
        logger.info('initializing FEsolver object')
        self.params = params
        self.res = {} # Results dictionary

        # Save some commonly used parameters as attributes
        self.ncore = self.params['ncore'] # number of cores to use
        self.ndraw_pii = self.params['ndraw_pii'] # number of draws to compute leverage
        self.ndraw_trace = self.params['ndraw_tr'] # number of draws to compute hetero correction
        self.compute_hetero = self.params['hetero']

        # Store some parameters in results dictionary
        self.res['cores'] = self.ncore
        self.res['ndp'] = self.ndraw_pii
        self.res['ndt'] = self.ndraw_trace

        logger.info('FEsolver object initialized')

    def __getstate__(self):
        '''
        Defines how the model is pickled.
        '''
        odict = {k: self.__dict__[k] for k in self.__dict__.keys() - {'ml'}}
        return odict

    def __setstate__(self, d):
        '''
        Defines how the model is unpickled.

        Arguments:
            d (dictionary): attribute dictionary
        '''
        # Need to recreate the simple model and the search representation
        self.__dict__ = d # Make d the attribute dictionary
        self.ml = pyamg.ruge_stuben_solver(self.M)

    @staticmethod
    def load(filename):
        '''
        Load files for heteroskedastic correction.

        Arguments:
            filename (string): file to load

        Returns:
            fes: loaded file
        '''
        fes = None
        with open(filename, 'rb') as infile:
            fes = pickle.load(infile)
        return fes

    def save(self, filename):
        '''
        Save FEsolver class to filename as pickle.

        Arguments:
            filename (string): filename to save to
        '''
        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)

    def run_1(self):
        '''
        Run FE solver, part 1. Before run_2(), modify adata to allow creation of Q matrix.
        '''
        self.start_time = time.time()

        # Begin cleaning and analysis
        self.prep_data() # Prepare data
        self.init_prepped_adata() # Use cleaned adata to generate some attributes
        self.compute_early_stats() # Use cleaned data to compute some statistics

    def run_2(self):
        '''
        Run FE solver, part 2.
        '''
        if self.params['statsonly']: # If only returning early statistics
            self.save_early_stats()

        else: # If running analysis
            self.create_fe_solver() # Solve FE model
            self.compute_trace_approximation_fe() # Compute trace approxmation

            # If computing heteroskedastic correction
            if self.compute_hetero:
                self.compute_leverages_Pii() # Solve he model
                self.compute_trace_approximation_he() # Compute trace approximation

            self.collect_res() # Collect all results

        end_time = time.time()

        self.res['total_time'] = end_time - self.start_time
        del self.start_time

        self.save_res() # Save results to json

        logger.info('------ DONE -------')

    def prep_data(self):
        '''
        Do some initial data cleaning.
        '''
        logger.info('Preparing the data')

        data = self.params['data']
        sdata = data[data['m'] == 0].reset_index(drop=True)
        jdata = data[data['m'] == 1].reset_index(drop=True)

        logger.info('Data movers={} stayers={}'.format(len(jdata), len(sdata)))
        self.res['nm'] = len(jdata)
        self.res['ns'] = len(sdata)
        self.res['n_firms'] = len(np.unique(pd.concat([jdata['f1i'], jdata['f2i'], sdata['f1i']], ignore_index=True)))
        self.res['n_workers'] = len(np.unique(pd.concat([jdata['wid'], sdata['wid']], ignore_index=True)))
        self.res['n_movers'] = len(np.unique(pd.concat([jdata['wid']], ignore_index=True)))
        #res['year_max'] = int(sdata['year'].max())
        #res['year_min'] = int(sdata['year'].min())

        # Make wids unique per row
        jdata.set_index(np.arange(self.res['nm']) + 1)
        sdata.set_index(np.arange(self.res['ns']) + 1 + self.res['nm'])
        jdata['wid'] = np.arange(self.res['nm']) + 1
        sdata['wid'] = np.arange(self.res['ns']) + 1 + self.res['nm']

        # Combine the 2 data-sets
        # self.adata = pd.concat([sdata[['wid', 'f1i', 'y1']].assign(cs=1, m=0), jdata[['wid', 'f1i', 'y1']].assign(cs=1, m=1), jdata[['wid', 'f2i', 'y2']].rename(columns={'f2i': 'f1i', 'y2': 'y1'}).assign(cs=0, m=1)]) # FIXME updated below
        self.adata = pd.concat([sdata[['wid', 'f1i', 'f2i', 'y1']].assign(cs=1, m=0), jdata[['wid', 'f1i', 'f2i', 'y1']].assign(cs=1, m=1), jdata[['wid', 'f1i', 'f2i', 'y2']].rename({'f1i': 'f2i', 'f2i': 'f1i', 'y2': 'y1'}, axis=1).assign(cs=0, m=1)]) # FIXME For some reason I didn't rename the last group's y2 to y1 before, but I'm changing it from y2 to y1 because it isn't working otherwise - make sure in the future to confirm this is the right thing to do (note that y2 is never used anywhere else in the code so it almost certainly is supposed to be labeled as y1, especially given that is how it was done in the original code above)
        self.adata = self.adata.reset_index(drop=True) # FIXME changed from set_index(pd.Series(range(len(self.adata))))
        self.adata['wid'] = self.adata['wid'].astype('category').cat.codes + 1

    def init_prepped_adata(self):
        '''
        Use prepped adata to initialize class attributes.

        Arguments:
            adata (Pandas DataFrame): labor data.

                Contains the following columns:

                    wid (worker id)

                    y1 (compensation 1)

                    y2 (compensation 2)

                    f1i (firm id 1)

                    f2i (firm id 2)

                    m (0 if stayer, 1 if mover)
        '''
        nf = self.adata.f1i.max() # Number of firms
        nw = self.adata.wid.max() # Number of workers
        nn = len(self.adata) # Number of observations
        self.nf = nf
        self.nw = nw
        self.nn = nn
        logger.info('data nf:{} nw:{} nn:{}'.format(nf, nw, nn))

        # Matrices for the cross-section
        J = csc_matrix((np.ones(nn), (self.adata.index, self.adata.f1i - 1)), shape=(nn, nf)) # Firms
        J = J[:, range(nf - 1)]  # Normalize one firm to 0
        self.J = J
        W = csc_matrix((np.ones(nn), (self.adata.index, self.adata.wid - 1)), shape=(nn, nw)) # Workers
        self.W = W
        # Dw = diags((W.T * W).diagonal()) # FIXME changed from .transpose() to .T ALSO commented this out since it's not used
        Dwinv = diags(1.0 / ((W.T * W).diagonal())) # FIXME changed from .transpose() to .T
        self.Dwinv = Dwinv

        logger.info('Prepare linear solver')

        # Finally create M
        M = J.T * J - J.T * W * Dwinv * W.T * J # FIXME changed from .transpose() to .T
        self.M = M
        self.ml = pyamg.ruge_stuben_solver(M)

        # L = diags(fes.M.diagonal()) - fes.M
        # r = linalg.eigsh(L,k=2,which='LM')

        # # Create cross-section matrices
        # # cs == 1 ==> looking at y1 for movers (cs = cross section)
        # # Create Q matrix
        # if self.params['Q'] == 'cov(alpha, psi)': # Default
        #     mdata = self.adata[self.adata['cs'] == 1] # FIXME changed from adata.query('cs==1') (I ran %timeit and slicing is faster)
        #     mdata = mdata.reset_index(drop=True) # FIXME changed from set_index(pd.Series(range(len(mdata))))

        #     nnq = len(mdata) # Number of observations
        #     self.nnq = nnq
        #     Jq = csc_matrix((np.ones(nnq), (mdata.index, mdata.f1i - 1)), shape=(nnq, nf))
        #     self.Jq = Jq[:, range(nf - 1)]  # Normalizing one firm to 0
        #     self.Wq = csc_matrix((np.ones(nnq), (mdata.index, mdata.wid - 1)), shape=(nnq, nw))
        #     self.Yq = mdata['y1']
        # elif self.params['Q'] == 'cov(psi_t, psi_{t+1})':
        #     mdata = self.adata[self.adata['m'] == 1] # FIXME changed from adata.query('cs==1') (I ran %timeit and slicing is faster)
        #     mdata = mdata.reset_index(drop=True) # FIXME changed from set_index(pd.Series(range(len(mdata))))
        #     mdata_1 = mdata[mdata['cs'] == 1].reset_index(drop=True) # Firm 1 for movers
        #     mdata_2 = mdata[mdata['cs'] == 0].reset_index(drop=True) # Firm 2 for movers

        #     nnq = len(mdata_1) # Number of observations
        #     nm = len(mdata_1['wid'].unique()) # Number of movers
        #     self.nnq = nnq
        #     J1 = csc_matrix((np.ones(nnq), (mdata_1.index, mdata_1.f1i - 1)), shape=(nnq, nf))
        #     self.J1 = J1[:, range(nf - 1)]  # Normalizing one firm to 0
        #     J2 = csc_matrix((np.ones(nnq), (mdata_2.index, mdata_2.f1i - 1)), shape=(nnq, nf))
        #     self.J2 = J2[:, range(nf - 1)]  # Normalizing one firm to 0
        #     self.Yq = mdata_1['y1']

        # Save time variable
        self.last_invert_time = 0

    def weighted_quantile(self, values, quantiles, sample_weight=None, values_sorted=False, old_style=False): # FIXME was formerly a function outside the class
        '''
        Very close to numpy.percentile, but supports weights.
        NOTE: quantiles should be in [0, 1]!

        Arguments:
            :param values: numpy.array with data
            :param quantiles: array-like with many quantiles needed
            :param sample_weight: array-like of the same length as `array`
            :param values_sorted: bool, if True, then will avoid sorting of initial array
            :param old_style: if True, will correct output to be consistent with numpy.percentile.

        Returns:
            :return: numpy.array with computed quantiles.
        '''
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

    def weighted_var(self, v, w): # FIXME was formerly a function outside the class
        '''
        Compute weighted variance @ FIXME I don't know what this function really does

        Arguments:
            v: @ FIXME I don't know what this is
            w: @ FIXME I don't know what this is

        Returns:
            v0: @ FIXME I don't know what this is
        '''
        m0 = np.sum(w * v) / np.sum(w)
        v0 = np.sum(w * (v - m0) ** 2) / np.sum(w)

        return v0

    def compute_early_stats(self):
        '''
        Compute some early statistics.
        '''
        fdata = self.adata.groupby('f1i').agg({'m':'sum', 'y1':'mean', 'wid':'count' })
        self.res['mover_quantiles'] = self.weighted_quantile(fdata['m'], np.linspace(0, 1, 11), fdata['wid']).tolist()
        self.res['size_quantiles'] = self.weighted_quantile(fdata['wid'], np.linspace(0, 1, 11), fdata['wid']).tolist()
        self.res['between_firm_var'] = self.weighted_var(fdata['y1'], fdata['wid'])
        self.res['var_y'] = self.adata[self.adata['cs'] == 1]['y1'].var() # FIXME changed from adata.query('cs==1') (I ran %timeit and slicing is faster)

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

    def save_early_stats(self):
        '''
        Save the early statistics computed in compute_early_stats().
        '''
        with open(self.params['out'], 'w') as outfile:
            json.dump(self.res, outfile)
        logger.info('saved results to {}'.format(self.params['out']))
        logger.info('--statsonly was passed as argument, so we skip all estimation.')
        logger.info('------ DONE -------')
        # sys.exit() # FIXME I don't think this is necessary (does it even work?) since this is now a class object

    def construct_Q(self):
        '''
        Generate columns in adata necessary to construct Q.
        '''
        if self.params['Q'] == 'cov(alpha, psi)':
            # Which rows to select
            self.adata['Jq'] = self.adata['cs'] == 1
            # Rows for csc_matrix
            self.adata['Jq_row'] = self.adata['Jq'].cumsum() - 1
            # Columns for csc_matrix
            self.adata['Jq_col'] = self.adata['f1i'] - 1
            self.adata['Wq'] = self.adata['cs'] == 1
            self.adata['Wq_row'] = self.adata['Wq'].cumsum() - 1
            self.adata['Wq_col'] = self.adata['wid'] - 1

        elif self.params['Q'] == 'cov(psi_t, psi_{t+1})':
            self.adata['Jq'] = (self.adata['m'] == 1) & (self.adata['cs'] == 1)
            self.adata['Jq_row'] = self.adata['Jq'].cumsum() - 1
            self.adata['Jq_col'] = self.adata['f1i'] - 1
            self.adata['Wq'] = (self.adata['m'] == 1) & (self.adata['cs'] == 0)
            self.adata['Wq_row'] = self.adata['Wq'].cumsum() - 1
            self.adata['Wq_col'] = self.adata['f1i'] - 1 # Recall f1i, f2i swapped for m==1 and cs==0

        elif self.params['Q'] == 'cov(psi_i, psi_j)': # Code doesn't work
            self.adata['Jq'] = (self.adata['m'] == 1) & (self.adata['cs'] == 1)
            self.adata['Jq_row'] = self.adata['f1i'] - 1
            self.adata['Jq_col'] = self.adata['f1i'] - 1
            self.adata['Wq'] = (self.adata['m'] == 1) & (self.adata['cs'] == 0)
            # Recall f1i, f2i swapped for m==1 and cs==0
            self.adata['Wq_row'] = self.adata['f2i'] - 1
            self.adata['Wq_col'] = self.adata['f1i'] - 1

    def create_fe_solver(self):
        '''
        Solve FE model.
        '''
        self.Y = self.adata.y1

        # try to pickle the object to see its size
        # self.save('tmp.pkl') # FIXME should we delete these 2 lines?

        logger.info('Extract firm effects')

        self.psi_hat, self.alpha_hat = self.solve(self.Y)

        logger.info('Solver time {:2.4f} seconds'.format(self.last_invert_time))
        logger.info('Expected total time {:2.4f} minutes'.format( (self.ndraw_trace * (1 + self.compute_hetero) + self.ndraw_pii * self.compute_hetero) * self.last_invert_time / 60))

        self.E = self.Y - self.mult_A(self.psi_hat, self.alpha_hat)

        self.res['solver_time'] = self.last_invert_time

        fe_rsq = 1 - np.power(self.E, 2).mean() / np.power(self.Y, 2).mean()
        logger.info('Fixed effect R-square {:2.4f}'.format(fe_rsq))

        # FIXME This section moved into compute_trace_approximation_fe()
        # # FIXME Need to figure out when this section can be run
        # self.tot_var = np.var(self.Y)
        # self.var_fe = np.var(self.Jq * self.psi_hat)
        # self.cov_fe = np.cov(self.Jq * self.psi_hat, self.Wq * self.alpha_hat)[0][1]
        # logger.info('[fe] var_psi={:2.4f} cov={:2.4f} tot={:2.4f}'.format(self.var_fe, self.cov_fe, self.tot_var))
        # # FIXME Section ends here

        self.var_e = self.nn / (self.nn - self.nw - self.nf + 1) * np.power(self.E, 2).mean()
        logger.info('[ho] variance of residuals {:2.4f}'.format(self.var_e))

    def compute_leverages_Pii(self):
        '''
        Compute leverages for heteroskedastic correction.
        '''
        self.Pii = np.zeros(self.nn)
        self.Sii = np.zeros(self.nn)

        if len(self.params['levfile']) > 1:
            logger.info('[he] starting heteroskedastic correction, loading precomputed files')

            files = glob.glob('{}*'.format(self.params['levfile']))
            logger.info('[he] found {} files to get leverages from'.format(len(files)))
            self.res['lev_file_count'] = len(files)
            assert len(files) > 0, "Didn't find any leverage files!"

            for f in files:
                pp = np.load(f)
                self.Pii += pp / len(files)

        elif self.ncore > 1:
            logger.info('[he] starting heteroskedastic correction p2={}, using {} cores, batch size {}'.format(self.ndraw_pii, self.ncore, self.params['batch']))
            set_start_method('spawn')
            with Pool(processes=self.ncore) as pool:
                Pii_all = pool.starmap(self.leverage_approx, [self.params['batch'] for _ in range(self.ndraw_pii // self.params['batch'])])

            for pp in Pii_all:
                Pii += pp / len(Pii_all)

        else:
            Pii_all = list(itertools.starmap(self.leverage_approx, [self.params['batch'] for _ in range(self.ndraw_pii // self.params['batch'])]))

            for pp in Pii_all:
                self.Pii += pp / len(Pii_all)

        I = 1.0 * self.adata.eval('m == 1')
        max_leverage = (I * self.Pii).max()

        # Attach the computed Pii to the dataframe
        self.adata['Pii'] = self.Pii
        logger.info('[he] Leverage range {:2.4f} to {:2.4f}'.format(self.adata.query('m == 1').Pii.min(), self.adata.query('m == 1').Pii.max()))

        # Give stayers the variance estimate at the firm level
        self.adata['Sii'] = self.Y * self.E / (1 - self.Pii)
        S_j = self.adata.query('m == 1').rename(columns={'Sii': 'Sii_j'}).groupby('f1i')['Sii_j'].agg('mean')

        self.adata = pd.merge(self.adata, S_j, on='f1i')
        self.adata['Sii'] = np.where(self.adata['m'] == 1, self.adata['Sii'], self.adata['Sii_j'])
        self.Sii = self.adata['Sii']

        logger.info('[he] variance of residuals in heteroskedastic case: {:2.4f}'.format(self.Sii.mean()))

    def compute_trace_approximation_fe(self):
        '''
        Compute FE trace approximation for arbitrary Q.
        '''
        logger.info('Starting FE trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))

        # Construct Jq, Wq matrices
        Jq = self.adata[self.adata['Jq'] == 1].reset_index(drop=True)
        self.Yq = Jq['y1']
        nJ = len(Jq)
        nJ_row = Jq['Jq_row'].max() + 1 # FIXME len(Jq['Jq_row'].unique())
        nJ_col = Jq['Jq_col'].max() + 1 # FIXME len(Jq['Jq_col'].unique())
        Jq = csc_matrix((np.ones(nJ), (Jq['Jq_row'], Jq['Jq_col'])), shape=(nJ_row, nJ_col))
        if nJ_col == self.nf: # If looking at firms, normalize one to 0
            Jq = Jq[:, range(self.nf - 1)]

        Wq = self.adata[self.adata['Wq'] == 1].reset_index(drop=True)
        nW = len(Wq)
        nW_row = Wq['Wq_row'].max() + 1 # FIXME len(Wq['Wq_row'].unique())
        nW_col = Wq['Wq_col'].max() + 1 # FIXME len(Wq['Wq_col'].unique())
        Wq = csc_matrix((np.ones(nW), (Wq['Wq_row'], Wq['Wq_col'])), shape=(nW_row, nW_col)) # FIXME Should we use nJ because require Jq, Wq to have the same size?
        if nW_col == self.nf: # If looking at firms, normalize one to 0
            Wq = Wq[:, range(self.nf - 1)]

        # Compute some stats
        # FIXME Need to figure out when this section can be run
        self.tot_var = np.var(self.Y)
        logger.info('[fe]')
        try:
            # print('psi', self.psi_hat)
            self.var_fe = np.var(Jq * self.psi_hat)
            logger.info('var_psi={:2.4f}'.format(self.var_fe))
        except ValueError: # If dimension mismatch
            pass
        try:
            self.cov_fe = np.cov(Jq * self.psi_hat, Wq * self.alpha_hat)[0][1]
            logger.info('cov={:2.4f} tot={:2.4f}'.format(self.cov_fe, self.tot_var))
        except ValueError: # If dimension mismatch
            pass
        # FIXME Section ends here

        # Begin trace approximation
        self.tr_var_ho_all = np.zeros(self.ndraw_trace)
        self.tr_cov_ho_all = np.zeros(self.ndraw_trace)

        for r in trange(self.ndraw_trace):
            # Generate -1 or 1
            Zpsi = 2 * np.random.binomial(1, 0.5, self.nf - 1) - 1
            Zalpha = 2 * np.random.binomial(1, 0.5, self.nw) - 1

            R1 = Jq * Zpsi
            psi1, alpha1 = self.mult_AAinv(Zpsi, Zalpha)
            try:
                R2_psi = Jq * psi1
                # Trace correction
                self.tr_var_ho_all[r] = np.cov(R1, R2_psi)[0][1]
            except ValueError: # If dimension mismatch
                try:
                    del self.tr_var_ho_all
                except AttributeError: # Once deleted
                    pass
            try:
                R2_alpha = Wq * alpha1
                # Trace correction
                self.tr_cov_ho_all[r] = np.cov(R1, R2_alpha)[0][1]
            except ValueError: # If dimension mismatch
                try:
                    del self.tr_cov_ho_all
                except AttributeError: # Once deleted
                    pass

            logger.debug('FE [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    # def compute_trace_approximation_fe(self):
    #     '''
    #     Purpose:
    #         Compute FE trace approximation.
    #     '''
    #     logger.info('Starting FE trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))
    #     self.tr_var_ho_all = np.zeros(self.ndraw_trace)
    #     self.tr_cov_ho_all = np.zeros(self.ndraw_trace)

    #     for r in trange(self.ndraw_trace):
    #         # Generate -1 or 1
    #         Zpsi = 2 * np.random.binomial(1, 0.5, self.nf - 1) - 1
    #         Zalpha = 2 * np.random.binomial(1, 0.5, self.nw) - 1

    #         R1 = self.Jq * Zpsi
    #         psi1, alpha1 = self.mult_AAinv(Zpsi, Zalpha)
    #         R2_psi = self.Jq * psi1
    #         R2_alpha = self.Wq * alpha1

    #         # Trace corrections
    #         self.tr_var_ho_all[r] = np.cov(R1, R2_psi)[0][1]
    #         self.tr_cov_ho_all[r] = np.cov(R1, R2_alpha)[0][1]

    #         logger.debug('FE [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    # def compute_trace_approximation_j1j2(self):
    #     '''
    #     Purpose:
    #         covariance between psi before and after the move among movers
    #     '''
    #     logger.info('Starting FE trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))
    #     self.tr_var_ho_all = np.zeros(self.ndraw_trace)

    #     for r in trange(self.ndraw_trace):
    #         # Generate -1 or 1
    #         Zpsi = 2 * np.random.binomial(1, 0.5, self.nf - 1) - 1
    #         Zalpha = 2 * np.random.binomial(1, 0.5, self.nw) - 1

    #         R1 = self.J1 * Zpsi
    #         psi1, _ = self.mult_AAinv(Zpsi, Zalpha)
    #         R2_psi = self.J2 * psi1

    #         # Trace corrections
    #         self.tr_var_ho_all[r] = np.cov(R1, R2_psi)[0][1]
    #         logger.debug('FE [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    def compute_trace_approximation_he(self):
        '''
        Compute heteroskedastic trace approximation.
        '''
        logger.info('Starting he trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))
        self.tr_var_he_all = np.zeros(self.ndraw_trace)
        self.tr_cov_he_all = np.zeros(self.ndraw_trace)

        for r in trange(self.ndraw_trace):
            Zpsi = 2 * np.random.binomial(1, 0.5, self.nf - 1) - 1
            Zalpha = 2 * np.random.binomial(1, 0.5, self.nw) - 1

            psi1, alpha1 = self.mult_AAinv(Zpsi, Zalpha)
            R2_psi = self.Jq * psi1
            R2_alpha = self.Wq * alpha1

            psi2, alpha2 = self.mult_AAinv(*self.mult_Atranspose(self.Sii * self.mult_A(Zpsi, Zalpha)))
            R3_psi = self.Jq * psi2

            # Trace corrections
            self.tr_var_he_all[r] = np.cov(R2_psi, R3_psi)[0][1]
            self.tr_cov_he_all[r] = np.cov(R2_alpha, R3_psi)[0][1]

            logger.debug('he [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    def collect_res(self):
        '''
        Collect all results.
        '''
        self.res['tot_var'] = self.tot_var
        self.res['eps_var_ho'] = self.var_e
        self.res['eps_var_fe'] = np.var(self.E)
        self.res['tr_var_ho'] = np.mean(self.tr_var_ho_all)
        logger.info('[ho] VAR tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_var_ho'], np.std(self.tr_var_ho_all)))

        # FIXME Need to figure out when this section can be run
        try:
            self.res['tr_cov_ho'] = np.mean(self.tr_cov_ho_all)
            logger.info('[ho] COV tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_cov_ho'], np.std(self.tr_cov_ho_all)))
        except AttributeError: # If no cov
            pass
        # FIXME Section ends here

        if self.compute_hetero:
            self.res['eps_var_he'] = self.Sii.mean()
            self.res['min_lev'] = self.adata.query('m == 1').Pii.min()
            self.res['max_lev'] = self.adata.query('m == 1').Pii.max()
            self.res['tr_var_he'] = np.mean(self.tr_var_he_all)
            self.res['tr_cov_he'] = np.mean(self.tr_cov_he_all)
            self.res['tr_var_ho_sd'] = np.std(self.tr_var_ho_all)
            self.res['tr_cov_ho_sd'] = np.std(self.tr_cov_ho_all)
            self.res['tr_var_he_sd'] = np.std(self.tr_var_he_all)
            self.res['tr_cov_he_sd'] = np.std(self.tr_cov_he_all)
            logger.info('[he] VAR tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_var_he'], np.std(self.tr_var_he_all)))
            logger.info('[he] COV tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_cov_he'], np.std(self.tr_cov_he_all)))

        # ----- FINAL ------
        # FIXME Need to figure out when this section can be run
        try:
            logger.info('[ho] VAR fe={:2.4f}'.format(self.var_fe))
        except AttributeError: # If no var fe
            pass
        try:
            logger.info('[ho] VAR bc={:2.4f}'.format(self.var_fe - self.var_e * self.res['tr_var_ho']))
        except AttributeError: # If no var bc
            pass
        try:
            logger.info('[ho] COV fe={:2.4f}'.format(self.cov_fe))
        except AttributeError: # If no cov fe
            pass
        try:
            logger.info('[ho] COV bc={:2.4f}'.format(self.cov_fe - self.var_e * self.res['tr_cov_ho']))
        except AttributeError: # If no cov bc
            pass
        # FIXME Section ends here

        if self.compute_hetero:
            logger.info('[he] VAR fe={:2.4f} bc={:2.4f}'.format(self.var_fe, self.var_fe - self.res['tr_var_he']))
            logger.info('[he] COV fe={:2.4f} bc={:2.4f}'.format(self.cov_fe, self.cov_fe - self.res['tr_cov_he']))

        self.res['var_y'] = np.var(self.Yq)
        # FIXME Need to figure out when this section can be run
        try:
            self.res['var_fe'] = self.var_fe
        except AttributeError:
            pass
        try:
            self.res['cov_fe'] = self.cov_fe
        except AttributeError:
            pass
        try:
            self.res['var_ho'] = self.var_fe - self.var_e * self.res['tr_var_ho']
        except AttributeError:
            pass
        try:
            self.res['cov_ho'] = self.cov_fe - self.var_e * self.res['tr_cov_ho']
        except AttributeError:
            pass
        # FIXME Section ends here

        if self.compute_hetero:
            self.res['var_he'] = self.var_fe - self.res['tr_var_he']
            self.res['cov_he'] = self.cov_fe - self.res['tr_cov_he']

    def save_res(self):
        '''
        Save results as json.
        '''
        # Convert results into strings to prevent JSON errors
        for key, val in self.res.items():
            self.res[key] = str(val)

        with open(self.params['out'], 'w') as outfile:
            json.dump(self.res, outfile)

        logger.info('Saved results to {}'.format(self.params['out']))

    def get_akm_estimates(self):
        '''
        Return estimated psi_hats linked to firm ids and alpha_hats linked to worker ids.

        Returns:
            alpha_hat_dict (dict): dictionary linking firm ids to estimated firm fixed effects
            alpha_hat_dict (dict): dictionary linking worker ids to estimated worker fixed effects
        '''
        fids = np.arange(self.adata.f1i.max()) + 1
        wids = np.arange(self.adata.wid.max()) + 1
        psi_hat_dict = dict(zip(fids, np.concatenate([self.psi_hat, np.array([0])]))) # Add 0 for normalized firm
        alpha_hat_dict = dict(zip(wids, self.alpha_hat))

        return psi_hat_dict, alpha_hat_dict

    def solve(self, Y):
        '''
        Compute (A'A)^-1 A'Y, the least squares estimate of A [psi_hat, alpha_hat] = Y.

        Arguments:
            Y (Pandas DataFrame): labor data

        Returns:
            psi_hat: estimated firm fixed effects @ FIXME correct datatype
            alpha_hat: estimated worker fixed effects @ FIXME correct datatype
        '''
        J_transpose_Y, W_transpose_Y = self.mult_Atranspose(Y) # This gives A'Y
        psi_hat, alpha_hat = self.mult_AAinv(J_transpose_Y, W_transpose_Y)

        return psi_hat, alpha_hat

    def mult_A(self, psi, alpha):
        '''
        Multiplies A = [J W] stored in the object by psi and alpha (used, for example, to compute estimated outcomes and sample errors).

        Arguments:
            psi: firm fixed effects @ FIXME correct datatype
            alpha: worker fixed effects @ FIXME correct datatype

        Returns:
            J_psi + W_alpha (CSC Matrix): firms * firm fixed effects + workers * worker fixed effects
        '''
        # J_psi = self.J * psi
        # W_alpha = self.W * alpha

        return self.J * psi + self.W * alpha # J_psi + W_alpha

    def mult_Atranspose(self, v):
        '''
        Multiplies the transpose of A = [J W] stored in the object by v.

        Arguments:
            v: what to multiply by @ FIXME correct datatype

        Returns:
            J_transpose_V (CSC Matrix): firms * v
            W_transpose_V (CSC Matrix): workers * v
        '''
        # J_transpose_V = self.J.T * v # FIXME changed from .transpose() to .T
        # W_transpose_V = self.W.T * v # FIXME changed from .transpose() to .T

        return self.J.T * v, self.W.T * v # J_transpose_V, W_transpose_V

    def mult_AAinv(self, psi, alpha):
        '''
        Multiplies gamma = [psi alpha] by (A'A)^(-1) where A = [J W] stored in the object.

        Arguments:
            psi: firm fixed effects @ FIXME correct datatype
            alpha: worker fixed effects @ FIXME correct datatype

        Returns:
            psi_out: estimated firm fixed effects @ FIXME correct datatype
            alpha_out: estimated worker fixed effects @ FIXME correct datatype
        '''
        # inter1 = self.ml.solve( psi , tol=1e-10 )
        # inter2 = self.ml.solve(  , tol=1e-10 )
        # psi_out = inter1 - inter2

        start = timer()
        psi_out = self.ml.solve(psi - self.J.T * (self.W * (self.Dwinv * alpha)), tol=1e-10) # FIXME changed from .transpose() to .T
        self.last_invert_time = timer() - start

        alpha_out = - self.Dwinv * (self.W.T * (self.J * psi_out)) + self.Dwinv * alpha # FIXME changed from .transpose() to .T

        return psi_out, alpha_out

    def proj(self, y): # FIXME should this y be Y?
        '''
        Solve y, then project onto X space of data stored in the object.

        Arguments:
            y (Pandas DataFrame): labor data

        Returns:
            Projection of psi, alpha solved from y onto X space
        '''
        return self.mult_A(*self.solve(y))

    def leverage_approx(self, ndraw_pii):
        '''
        Compute an approximate leverage using ndraw_pii.

        Arguments:
            ndraw_pii (int): number of draws

        Returns:
            Pii (NumPy Array): @ FIXME I don't know what this function does, so I don't know what Pii is
        '''
        Pii = np.zeros(self.nn)

        # Compute the different draws
        for r in trange(ndraw_pii):
            R2  = 2 * np.random.binomial(1, 0.5, self.nn) - 1
            Pii += 1 / ndraw_pii * np.power(self.proj(R2), 2.0)

        logger.info('Done with batch')

        return Pii

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
