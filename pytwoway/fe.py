'''
Computes a bunch of estimates from an event study data set:

    - AKM variance decomposition
    - Andrews bias correction
    - KSS bias correction

Does this through class FEEstimator
'''
import logging
from pathlib import Path
import pyamg
import numpy as np
import pandas as pd
from bipartitepandas import logger_init
from scipy.sparse import csc_matrix, coo_matrix, diags, linalg
import time
# import pyreadr
import os
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
    from tqdm import tqdm, trange
except ImportError:
    trange = range

# def pipe_qcov(df, e1, e2): # FIXME I moved this from above, also this is used only in commented out code
#     v1 = df.eval(e1)
#     v2 = df.eval(e2)
#     return np.cov(v1, v2)[0][1]

class FEEstimator:
    '''
    Uses multigrid and partialing out to solve two way fixed effect model.

    @ FIXME I think delete everything below this, it's basically contained in the class/functions within the class

    takes as an input this adata
    and creates associated A = [J W] matrix which are AKM dummies

    provides methods to do A x Y but also (A'A)^-1 A'Y solve method
    '''

    def __init__(self, params):
        '''
        Arguments:
            params (dict): dictionary of parameters for FE estimation

                Dictionary parameters:

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

                    ncore (int): number of cores to use

                    batch (int): batch size to send in parallel

                    ndraw_pii (int): number of draws to use in approximation for leverages

                    levfile (str): file to load precomputed leverages`

                    ndraw_tr (int): number of draws to use in approximation for traces

                    h2 (bool): if True, compute h2 correction

                    out (str): outputfile where results are saved

                    statsonly (bool): if True, return only basic statistics

                    Q (str): which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'
        '''
        # Start logger
        logger_init(self)
        # self.logger.info('initializing FEEstimator object')

        self.params = params
        self.res = {} # Results dictionary
        self.summary = {} # Summary results dictionary

        # Save some commonly used parameters as attributes
        self.ncore = self.params['ncore'] # Number of cores to use
        self.ndraw_pii = self.params['ndraw_pii'] # Number of draws to compute leverage
        self.ndraw_trace = self.params['ndraw_tr'] # Number of draws to compute h2 correction
        self.compute_h2 = self.params['h2']

        # Store some parameters in results dictionary
        self.res['cores'] = self.ncore
        self.res['ndp'] = self.ndraw_pii
        self.res['ndt'] = self.ndraw_trace

        # self.logger.info('FEEstimator object initialized')

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
            d (dict): attribute dictionary
        '''
        # Need to recreate the simple model and the search representation
        self.__dict__ = d # Make d the attribute dictionary
        self.ml = pyamg.ruge_stuben_solver(self.M)

    @staticmethod
    def __load(filename):
        '''
        Load files for h2 correction.

        Arguments:
            filename (string): file to load

        Returns:
            fes: loaded file
        '''
        fes = None
        with open(filename, 'rb') as infile:
            fes = pickle.load(infile)
        return fes

    def __save(self, filename):
        '''
        Save FEEstimator class to filename as pickle.

        Arguments:
            filename (string): filename to save to
        '''
        with open(filename, 'wb') as outfile:
            pickle.dump(self, outfile)

    def fit_1(self):
        '''
        Run FE solver, part 1. Before fit_2(), modify adata to allow creation of Q matrix.
        '''
        self.start_time = time.time()

        # Begin cleaning and analysis
        self.__prep_vars() # Prepare data
        self.__prep_JWM() # Use cleaned adata to generate some attributes
        self.__compute_early_stats() # Use cleaned data to compute some statistics

    def fit_2(self):
        '''
        Run FE solver, part 2.
        '''
        if self.params['statsonly']: # If only returning early statistics
            self.__save_early_stats()

        else: # If running analysis
            self.__create_fe_solver() # Solve FE model
            self.__compute_trace_approximation_fe() # Compute trace approxmation

            # If computing h2 correction
            if self.compute_h2:
                self.__compute_leverages_Pii() # Solve h2 model
                self.__compute_trace_approximation_h2() # Compute trace approximation

            self.__collect_res() # Collect all results

        end_time = time.time()

        self.res['total_time'] = end_time - self.start_time
        del self.start_time

        self.__save_res() # Save results to json

        self.logger.info('------ DONE -------')

    def __prep_vars(self):
        '''
        Generate some initial class attributes and results.
        '''
        self.logger.info('preparing the data')

        self.adata = self.params['data']
        # self.adata['i'] = self.adata['i'].astype('category').cat.codes + 1 # FIXME commented out because i should already be correct

        self.nf = max(self.adata['j1'].max(), self.adata['j2'].max()) + 1 # Number of firms
        self.nw = self.adata['i'].max() + 1 # Number of workers
        self.nn = len(self.adata) # Number of observations
        self.logger.info('data firms={} workers={} observations={}'.format(self.nf, self.nw, self.nn))

        self.res['n_firms'] = self.nf
        self.res['n_workers'] = self.nw
        self.res['n_movers'] = len(np.unique(self.adata[self.adata['m'] == 1]['i']))
        self.res['n_stayers'] = self.res['n_workers'] - self.res['n_movers']
        self.logger.info('data movers={} stayers={}'.format(self.res['n_movers'], self.res['n_stayers']))

        #res['year_max'] = int(sdata['year'].max())
        #res['year_min'] = int(sdata['year'].min())

        # # Make j values unique per row
        # jdata.set_index(np.arange(self.res['nm']) + 1)
        # sdata.set_index(np.arange(self.res['ns']) + 1 + self.res['nm'])
        # jdata['i'] = np.arange(self.res['nm']) + 1
        # sdata['i'] = np.arange(self.res['ns']) + 1 + self.res['nm']

        # # Combine the 2 data-sets
        # # self.adata = pd.concat([sdata[['i', 'j1', 'y1']].assign(cs=1, m=0), jdata[['i', 'j1', 'y1']].assign(cs=1, m=1), jdata[['i', 'j2', 'y2']].rename(columns={'j2': 'j1', 'y2': 'y1'}).assign(cs=0, m=1)]) # FIXME updated below
        # self.adata = pd.concat([sdata[['i', 'j1', 'j2', 'y1']].assign(cs=1, m=0), jdata[['i', 'j1', 'j2', 'y1']].assign(cs=1, m=1), jdata[['i', 'j1', 'j2', 'y2']].rename({'j1': 'j2', 'j2': 'j1', 'y2': 'y1'}, axis=1).assign(cs=0, m=1)]) # FIXME For some reason I didn't rename the last group's y2 to y1 before, but I'm changing it from y2 to y1 because it isn't working otherwise - make sure in the future to confirm this is the right thing to do (note that y2 is never used anywhere else in the code so it almost certainly is supposed to be labeled as y1, especially given that is how it was done in the original code above)
        # self.adata = self.adata.reset_index(drop=True) # FIXME changed from set_index(pd.Series(range(len(self.adata))))
        # self.adata['i'] = self.adata['i'].astype('category').cat.codes + 1

    def __prep_JWM(self):
        '''
        Generate J, W, and M matrices.
        '''
        # Matrices for the cross-section
        J = csc_matrix((np.ones(self.nn), (self.adata.index, self.adata.j1)), shape=(self.nn, self.nf)) # Firms
        J = J[:, range(self.nf - 1)]  # Normalize one firm to 0
        self.J = J
        W = csc_matrix((np.ones(self.nn), (self.adata.index, self.adata.i)), shape=(self.nn, self.nw)) # Workers
        self.W = W
        # Dw = diags((W.T * W).diagonal()) # FIXME changed from .transpose() to .T ALSO commented this out since it's not used
        Dwinv = diags(1.0 / ((W.T * W).diagonal())) # FIXME changed from .transpose() to .T
        self.Dwinv = Dwinv

        self.logger.info('Prepare linear solver')

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
        #     Jq = csc_matrix((np.ones(nnq), (mdata.index, mdata.j1 - 1)), shape=(nnq, nf))
        #     self.Jq = Jq[:, range(nf - 1)]  # Normalizing one firm to 0
        #     self.Wq = csc_matrix((np.ones(nnq), (mdata.index, mdata.i - 1)), shape=(nnq, nw))
        #     self.Yq = mdata['y1']
        # elif self.params['Q'] == 'cov(psi_t, psi_{t+1})':
        #     mdata = self.adata[self.adata['m'] == 1] # FIXME changed from adata.query('cs==1') (I ran %timeit and slicing is faster)
        #     mdata = mdata.reset_index(drop=True) # FIXME changed from set_index(pd.Series(range(len(mdata))))
        #     mdata_1 = mdata[mdata['cs'] == 1].reset_index(drop=True) # Firm 1 for movers
        #     mdata_2 = mdata[mdata['cs'] == 0].reset_index(drop=True) # Firm 2 for movers

        #     nnq = len(mdata_1) # Number of observations
        #     nm = len(mdata_1['i'].unique()) # Number of movers
        #     self.nnq = nnq
        #     J1 = csc_matrix((np.ones(nnq), (mdata_1.index, mdata_1.j1 - 1)), shape=(nnq, nf))
        #     self.J1 = J1[:, range(nf - 1)]  # Normalizing one firm to 0
        #     J2 = csc_matrix((np.ones(nnq), (mdata_2.index, mdata_2.j1 - 1)), shape=(nnq, nf))
        #     self.J2 = J2[:, range(nf - 1)]  # Normalizing one firm to 0
        #     self.Yq = mdata_1['y1']

        # Save time variable
        self.last_invert_time = 0

    def __weighted_quantile(self, values, quantiles, sample_weight=None, values_sorted=False, old_style=False): # FIXME was formerly a function outside the class
        '''
        Very close to numpy.percentile, but supports weights.
        NOTE: quantiles should be in [0, 1]!

        Arguments:
            values (NumPy Array): data
            quantiles (array-like): quantiles to compute
            sample_weight (array-like): weighting, must be same length as `array` (is `array` supposed to be quantiles?)
            values_sorted (bool): if True, skips sorting of initial array
            old_style (bool): if True, changes output to be consistent with numpy.percentile

        Returns:
            (NumPy Array): computed quantiles
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

    def __weighted_var(self, v, w): # FIXME was formerly a function outside the class
        '''
        Compute weighted variance. @ FIXME I don't know what this function really does

        Arguments:
            v: @ FIXME I don't know what this is
            w: @ FIXME I don't know what this is

        Returns:
            v0: @ FIXME I don't know what this is
        '''
        m0 = np.sum(w * v) / np.sum(w)
        v0 = np.sum(w * (v - m0) ** 2) / np.sum(w)

        return v0

    def __compute_early_stats(self):
        '''
        Compute some early statistics.
        '''
        fdata = self.adata.groupby('j1').agg({'m':'sum', 'y1':'mean', 'i':'count' })
        self.res['mover_quantiles'] = self.__weighted_quantile(fdata['m'], np.linspace(0, 1, 11), fdata['i']).tolist()
        self.res['size_quantiles'] = self.__weighted_quantile(fdata['i'], np.linspace(0, 1, 11), fdata['i']).tolist()
        self.res['between_firm_var'] = self.__weighted_var(fdata['y1'], fdata['i'])
        self.res['var_y'] = self.adata[self.adata['cs'] == 1]['y1'].var() # FIXME changed from adata.query('cs==1') (I ran %timeit and slicing is faster)
        self.logger.info('total variance: {:0.4f}'.format(self.res['var_y']))

        # extract woodcock moments using sdata and jdata
        # get averages by firms for stayers
        #dsf  = adata.query('cs==1').groupby('j1').agg(y1sj=('y1','mean'), nsj=('y1','count'))
        #ds   = pd.merge(adata.query('cs==1'), dsf, on="j1")
        #ds.eval("y1s_lo    = (nsj * y1sj - y1) / (nsj - 1)",inplace=True)
        #res['woodcock_var_psi']   = ds.query('nsj  > 1').pipe(pipe_qcov, 'y1', 'y1s_lo')
        #res['woodcock_var_alpha'] = np.minimum( jdata.pipe(pipe_qcov, 'y1','y2'), adata.query('cs==1')['y1'].var() - res['woodcock_var_psi'] )
        #res['woodcock_var_eps'] = adata.query('cs==1')['y1'].var() - res['woodcock_var_alpha'] - res['woodcock_var_psi']
        #self.logger.info("[woodcock] var psi = {}", res['woodcock_var_psi'])
        #self.logger.info("[woodcock] var alpha = {}", res['woodcock_var_alpha'])
        #self.logger.info("[woodcock] var eps = {}", res['woodcock_var_eps'])

    def __save_early_stats(self):
        '''
        Save the early statistics computed in compute_early_stats().
        '''
        with open(self.params['out'], 'w') as outfile:
            json.dump(self.res, outfile)
        self.logger.info('saved results to {}'.format(self.params['out']))
        self.logger.info('--statsonly was passed as argument, so we skip all estimation.')
        self.logger.info('------ DONE -------')
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
            self.adata['Jq_col'] = self.adata['j1']
            self.adata['Wq'] = self.adata['cs'] == 1
            self.adata['Wq_row'] = self.adata['Wq'].cumsum() - 1
            self.adata['Wq_col'] = self.adata['i']

        elif self.params['Q'] == 'cov(psi_t, psi_{t+1})':
            self.adata['Jq'] = (self.adata['m'] == 1) & (self.adata['cs'] == 1)
            self.adata['Jq_row'] = self.adata['Jq'].cumsum() - 1
            self.adata['Jq_col'] = self.adata['j1']
            self.adata['Wq'] = (self.adata['m'] == 1) & (self.adata['cs'] == 0)
            self.adata['Wq_row'] = self.adata['Wq'].cumsum() - 1
            self.adata['Wq_col'] = self.adata['j1'] # Recall j1, j2 swapped for m==1 and cs==0

        elif self.params['Q'] == 'cov(psi_i, psi_j)': # Code doesn't work
            self.adata['Jq'] = (self.adata['m'] == 1) & (self.adata['cs'] == 1)
            self.adata['Jq_row'] = self.adata['j1']
            self.adata['Jq_col'] = self.adata['j1']
            self.adata['Wq'] = (self.adata['m'] == 1) & (self.adata['cs'] == 0)
            # Recall j1, j2 swapped for m==1 and cs==0
            self.adata['Wq_row'] = self.adata['j2']
            self.adata['Wq_col'] = self.adata['j1']

    def __construct_Jq_Wq(self):
        '''
        Construct Jq and Wq matrices.

        Returns:
            Jq (Pandas DataFrame): left matrix for computing Q
            Wq (Pandas DataFrame): right matrix for computing Q
        '''
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

        return Jq, Wq

    def __create_fe_solver(self):
        '''
        Solve FE model.
        '''
        self.Y = self.adata.y1

        # try to pickle the object to see its size
        # self.save('tmp.pkl') # FIXME should we delete these 2 lines?

        self.logger.info('extract firm effects')

        self.psi_hat, self.alpha_hat = self.__solve(self.Y)

        self.logger.info('solver time {:2.4f} seconds'.format(self.last_invert_time))
        self.logger.info('expected total time {:2.4f} minutes'.format( (self.ndraw_trace * (1 + self.compute_h2) + self.ndraw_pii * self.compute_h2) * self.last_invert_time / 60))

        self.E = self.Y - self.__mult_A(self.psi_hat, self.alpha_hat)

        self.res['solver_time'] = self.last_invert_time

        fe_rsq = 1 - np.power(self.E, 2).mean() / np.power(self.Y, 2).mean()
        self.logger.info('fixed effect R-square {:2.4f}'.format(fe_rsq))

        # FIXME This section moved into compute_trace_approximation_fe()
        # # FIXME Need to figure out when this section can be run
        # self.tot_var = np.var(self.Y)
        # self.var_fe = np.var(self.Jq * self.psi_hat)
        # self.cov_fe = np.cov(self.Jq * self.psi_hat, self.Wq * self.alpha_hat)[0][1]
        # self.logger.info('[fe] var_psi={:2.4f} cov={:2.4f} tot={:2.4f}'.format(self.var_fe, self.cov_fe, self.tot_var))
        # # FIXME Section ends here

        self.var_e = self.nn / (self.nn - self.nw - self.nf + 1) * np.power(self.E, 2).mean()
        self.logger.info('[ho] variance of residuals {:2.4f}'.format(self.var_e))

    def __compute_leverages_Pii(self):
        '''
        Compute leverages for h2 correction.
        '''
        self.Pii = np.zeros(self.nn)
        self.Sii = np.zeros(self.nn)

        if len(self.params['levfile']) > 1:
            self.logger.info('[h2] starting h2 correction, loading precomputed files')

            files = glob.glob('{}*'.format(self.params['levfile']))
            self.logger.info('[h2] found {} files to get leverages from'.format(len(files)))
            self.res['lev_file_count'] = len(files)
            assert len(files) > 0, "Didn't find any leverage files!"

            for f in files:
                pp = np.load(f)
                self.Pii += pp / len(files)

        elif self.ncore > 1:
            self.logger.info('[h2] starting h2 correction p2={}, using {} cores, batch size {}'.format(self.ndraw_pii, self.ncore, self.params['batch']))
            set_start_method('spawn')
            with Pool(processes=self.ncore) as pool:
                Pii_all = pool.starmap(self.__leverage_approx, [self.params['batch'] for _ in range(self.ndraw_pii // self.params['batch'])])

            for pp in Pii_all:
                Pii += pp / len(Pii_all)

        else:
            Pii_all = list(itertools.starmap(self.__leverage_approx, [[self.params['batch']] for _ in range(self.ndraw_pii // self.params['batch'])]))

            for pp in Pii_all:
                self.Pii += pp / len(Pii_all)

        I = 1.0 * self.adata.eval('m == 1')
        max_leverage = (I * self.Pii).max()

        # Attach the computed Pii to the dataframe
        self.adata['Pii'] = self.Pii
        self.logger.info('[h2] Leverage range {:2.4f} to {:2.4f}'.format(self.adata.query('m == 1').Pii.min(), self.adata.query('m == 1').Pii.max()))

        # Give stayers the variance estimate at the firm level
        self.adata['Sii'] = self.Y * self.E / (1 - self.Pii)
        S_j = self.adata.query('m == 1').rename(columns={'Sii': 'Sii_j'}).groupby('j1')['Sii_j'].agg('mean')

        self.adata = pd.merge(self.adata, S_j, on='j1')
        self.adata['Sii'] = np.where(self.adata['m'] == 1, self.adata['Sii'], self.adata['Sii_j'])
        self.Sii = self.adata['Sii']

        self.logger.info('[h2] variance of residuals in h2 case: {:2.4f}'.format(self.Sii.mean()))

    def __compute_trace_approximation_fe(self):
        '''
        Compute FE trace approximation for arbitrary Q.
        '''
        self.logger.info('Starting FE trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))

        Jq, Wq = self.__construct_Jq_Wq()

        # Compute some stats
        # FIXME Need to figure out when this section can be run
        self.tot_var = np.var(self.Y)
        self.logger.info('[fe]')
        try:
            # print('psi', self.psi_hat)
            self.var_fe = np.var(Jq * self.psi_hat)
            self.logger.info('var_psi={:2.4f}'.format(self.var_fe))
        except ValueError: # If dimension mismatch
            pass
        try:
            self.cov_fe = np.cov(Jq * self.psi_hat, Wq * self.alpha_hat)[0][1]
            self.logger.info('cov={:2.4f} tot={:2.4f}'.format(self.cov_fe, self.tot_var))
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
            psi1, alpha1 = self.__mult_AAinv(Zpsi, Zalpha)
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

            self.logger.debug('FE [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    # def compute_trace_approximation_fe(self):
    #     '''
    #     Purpose:
    #         Compute FE trace approximation.
    #     '''
    #     self.logger.info('Starting FE trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))
    #     self.tr_var_ho_all = np.zeros(self.ndraw_trace)
    #     self.tr_cov_ho_all = np.zeros(self.ndraw_trace)

    #     for r in trange(self.ndraw_trace):
    #         # Generate -1 or 1
    #         Zpsi = 2 * np.random.binomial(1, 0.5, self.nf - 1) - 1
    #         Zalpha = 2 * np.random.binomial(1, 0.5, self.nw) - 1

    #         R1 = self.Jq * Zpsi
    #         psi1, alpha1 = self.__mult_AAinv(Zpsi, Zalpha)
    #         R2_psi = self.Jq * psi1
    #         R2_alpha = self.Wq * alpha1

    #         # Trace corrections
    #         self.tr_var_ho_all[r] = np.cov(R1, R2_psi)[0][1]
    #         self.tr_cov_ho_all[r] = np.cov(R1, R2_alpha)[0][1]

    #         self.logger.debug('FE [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    # def compute_trace_approximation_j1j2(self):
    #     '''
    #     Purpose:
    #         covariance between psi before and after the move among movers
    #     '''
    #     self.logger.info('Starting FE trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))
    #     self.tr_var_ho_all = np.zeros(self.ndraw_trace)

    #     for r in trange(self.ndraw_trace):
    #         # Generate -1 or 1
    #         Zpsi = 2 * np.random.binomial(1, 0.5, self.nf - 1) - 1
    #         Zalpha = 2 * np.random.binomial(1, 0.5, self.nw) - 1

    #         R1 = self.J1 * Zpsi
    #         psi1, _ = self.__mult_AAinv(Zpsi, Zalpha)
    #         R2_psi = self.J2 * psi1

    #         # Trace corrections
    #         self.tr_var_ho_all[r] = np.cov(R1, R2_psi)[0][1]
    #         self.logger.debug('FE [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    def __compute_trace_approximation_h2(self):
        '''
        Compute h2 trace approximation.
        '''
        self.logger.info('Starting h2 trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))
        self.tr_var_h2_all = np.zeros(self.ndraw_trace)
        self.tr_cov_h2_all = np.zeros(self.ndraw_trace)

        Jq, Wq = self.__construct_Jq_Wq()

        for r in trange(self.ndraw_trace):
            Zpsi = 2 * np.random.binomial(1, 0.5, self.nf - 1) - 1
            Zalpha = 2 * np.random.binomial(1, 0.5, self.nw) - 1

            psi1, alpha1 = self.__mult_AAinv(Zpsi, Zalpha)
            R2_psi = Jq * psi1
            R2_alpha = Wq * alpha1

            psi2, alpha2 = self.__mult_AAinv(*self.__mult_Atranspose(self.Sii * self.__mult_A(Zpsi, Zalpha)))
            R3_psi = Jq * psi2

            # Trace corrections
            self.tr_var_h2_all[r] = np.cov(R2_psi, R3_psi)[0][1]
            self.tr_cov_h2_all[r] = np.cov(R2_alpha, R3_psi)[0][1]

            self.logger.debug('h2 [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    def __collect_res(self):
        '''
        Collect all results.
        '''
        self.res['tot_var'] = self.tot_var
        self.res['eps_var_ho'] = self.var_e
        self.res['eps_var_fe'] = np.var(self.E)
        self.res['tr_var_ho'] = np.mean(self.tr_var_ho_all)
        self.logger.info('[ho] VAR tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_var_ho'], np.std(self.tr_var_ho_all)))

        # FIXME Need to figure out when this section can be run
        try:
            self.res['tr_cov_ho'] = np.mean(self.tr_cov_ho_all)
            self.logger.info('[ho] COV tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_cov_ho'], np.std(self.tr_cov_ho_all)))
        except AttributeError: # If no cov
            pass
        # FIXME Section ends here

        if self.compute_h2:
            self.res['eps_var_h2'] = self.Sii.mean()
            self.res['min_lev'] = self.adata.query('m == 1').Pii.min()
            self.res['max_lev'] = self.adata.query('m == 1').Pii.max()
            self.res['tr_var_h2'] = np.mean(self.tr_var_h2_all)
            self.res['tr_cov_h2'] = np.mean(self.tr_cov_h2_all)
            self.res['tr_var_ho_sd'] = np.std(self.tr_var_ho_all)
            self.res['tr_cov_ho_sd'] = np.std(self.tr_cov_ho_all)
            self.res['tr_var_h2_sd'] = np.std(self.tr_var_h2_all)
            self.res['tr_cov_h2_sd'] = np.std(self.tr_cov_h2_all)
            self.logger.info('[h2] VAR tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_var_h2'], np.std(self.tr_var_h2_all)))
            self.logger.info('[h2] COV tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_cov_h2'], np.std(self.tr_cov_h2_all)))

        # ----- FINAL ------
        # FIXME Need to figure out when this section can be run
        try:
            self.logger.info('[ho] VAR fe={:2.4f}'.format(self.var_fe))
        except AttributeError: # If no var fe
            pass
        try:
            self.logger.info('[ho] VAR bc={:2.4f}'.format(self.var_fe - self.var_e * self.res['tr_var_ho']))
        except AttributeError: # If no var bc
            pass
        try:
            self.logger.info('[ho] COV fe={:2.4f}'.format(self.cov_fe))
        except AttributeError: # If no cov fe
            pass
        try:
            self.logger.info('[ho] COV bc={:2.4f}'.format(self.cov_fe - self.var_e * self.res['tr_cov_ho']))
        except AttributeError: # If no cov bc
            pass
        # FIXME Section ends here

        if self.compute_h2:
            self.logger.info('[h2] VAR fe={:2.4f} bc={:2.4f}'.format(self.var_fe, self.var_fe - self.res['tr_var_h2']))
            self.logger.info('[h2] COV fe={:2.4f} bc={:2.4f}'.format(self.cov_fe, self.cov_fe - self.res['tr_cov_h2']))

        self.res['var_y'] = np.var(self.Yq)
        # FIXME Need to figure out when this section can be run
        try:
            self.res['var_fe'] = self.var_fe
            self.summary['var_fe'] = self.res['var_fe']
        except AttributeError:
            pass
        try:
            self.res['cov_fe'] = self.cov_fe
            self.summary['cov_fe'] = self.res['cov_fe']
        except AttributeError:
            pass
        try:
            self.res['var_ho'] = self.var_fe - self.var_e * self.res['tr_var_ho']
            self.summary['var_ho'] = self.res['var_ho']
        except AttributeError:
            pass
        try:
            self.res['cov_ho'] = self.cov_fe - self.var_e * self.res['tr_cov_ho']
            self.summary['cov_ho'] = self.res['cov_ho']
        except AttributeError:
            pass
        # FIXME Section ends here

        if self.compute_h2:
            self.res['var_h2'] = self.var_fe - self.res['tr_var_h2']
            self.res['cov_h2'] = self.cov_fe - self.res['tr_cov_h2']

        # Create summary variable
        self.summary['var_y'] = self.res['var_y']

    def __save_res(self):
        '''
        Save results as json.
        '''
        # Convert results into strings to prevent JSON errors
        for key, val in self.res.items():
            self.res[key] = str(val)

        with open(self.params['out'], 'w') as outfile:
            json.dump(self.res, outfile)

        self.logger.info('saved results to {}'.format(self.params['out']))

    def get_fe_estimates(self):
        '''
        Return estimated psi_hats linked to firm ids and alpha_hats linked to worker ids.

        Returns:
            alpha_hat_dict (dict): dictionary linking firm ids to estimated firm fixed effects
            alpha_hat_dict (dict): dictionary linking worker ids to estimated worker fixed effects
        '''
        j_vals = np.arange(self.nf) # np.arange(self.adata.j1.max()) + 1
        i_vals = np.arange(self.nw) # np.arange(self.adata.i.max()) + 1
        psi_hat_dict = dict(zip(j_vals, np.concatenate([self.psi_hat, np.array([0])]))) # Add 0 for normalized firm
        alpha_hat_dict = dict(zip(i_vals, self.alpha_hat))

        return psi_hat_dict, alpha_hat_dict

    def __solve(self, Y):
        '''
        Compute (A'A)^-1 A'Y, the least squares estimate of A [psi_hat, alpha_hat] = Y.

        Arguments:
            Y (Pandas DataFrame): labor data

        Returns:
            psi_hat: estimated firm fixed effects @ FIXME correct datatype
            alpha_hat: estimated worker fixed effects @ FIXME correct datatype
        '''
        J_transpose_Y, W_transpose_Y = self.__mult_Atranspose(Y) # This gives A'Y
        psi_hat, alpha_hat = self.__mult_AAinv(J_transpose_Y, W_transpose_Y)

        return psi_hat, alpha_hat

    def __mult_A(self, psi, alpha):
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

    def __mult_Atranspose(self, v):
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

    def __mult_AAinv(self, psi, alpha):
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

    def __proj(self, y): # FIXME should this y be Y?
        '''
        Solve y, then project onto X space of data stored in the object.

        Arguments:
            y (Pandas DataFrame): labor data

        Returns:
            Projection of psi, alpha solved from y onto X space
        '''
        return self.__mult_A(*self.__solve(y))

    def __leverage_approx(self, ndraw_pii):
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
            Pii += 1 / ndraw_pii * np.power(self.__proj(R2), 2.0)

        self.logger.info('Done with batch')

        return Pii
