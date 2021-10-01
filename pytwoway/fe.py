'''
Computes a bunch of estimates from an event study data set:

    - AKM variance decomposition
    - Andrews bias correction
    - KSS bias correction

Does this through class FEEstimator
'''
import warnings
from pathlib import Path
import pyamg
import numpy as np
import pandas as pd
from bipartitepandas import update_dict, logger_init
from scipy.sparse import csc_matrix, coo_matrix, diags, linalg
import time
# import pyreadr
import os
from multiprocessing import Pool, TimeoutError, set_start_method
from timeit import default_timer as timer
import itertools
import pickle
import time
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

    def __init__(self, data, params):
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
            params (dict): dictionary of parameters for FE estimation

                Dictionary parameters:

                    ncore (int, default=1): number of cores to use

                    batch (int, default=1): batch size to send in parallel

                    ndraw_pii (int, default=50): number of draws to use in approximation for leverages

                    levfile (str, default=''): file to load precomputed leverages`

                    ndraw_tr (int, default=5): number of draws to use in approximation for traces

                    he (bool, default=False): if True, compute heteroskedastic correction

                    out (str, default='res_fe.json'): outputfile where results are saved

                    statsonly (bool, default=False): if True, return only basic statistics

                    feonly (bool, default=False): if True, compute only fixed effects and not variances

                    Q (str, default='cov(alpha, psi)'): which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'

                    seed (int, default=None): NumPy RandomState seed
        '''
        # Start logger
        logger_init(self)
        # self.logger.info('initializing FEEstimator object')

        self.adata = data
        try:
            self.adata.sort_values(['i', 't'], inplace=True)
        except KeyError:
            self.adata.sort_values(['i', 't1'], inplace=True)

        # Define default parameter dictionaries
        default_params = {
            'ncore': 1, # Number of cores to use
            'batch': 1, # Batch size to send in parallel
            'ndraw_pii': 50, # Number of draws to use in approximation for leverages
            'levfile': '', # File to load precomputed leverages
            'ndraw_tr': 5, # Number of draws to use in approximation for traces
            'he': False, # If True, compute heteroskedastic correction
            'out': 'res_fe.json', # Outputfile where results are saved
            'statsonly': False, # If True, return only basic statistics
            'feonly': False, # If True, compute only fixed effects and not variances
            'Q': 'cov(alpha, psi)', # Which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'
            # 'con': False, # Computes the smallest eigen values, this is the filepath where these results are saved FIXME not used
            # 'logfile': '', # Log output to a logfile FIXME not used
            # 'check': False # If True, compute the non-approximated estimates as well FIXME not used
            'seed': None # np.random.RandomState() seed
        }

        self.params = update_dict(default_params, params)
        self.res = {} # Results dictionary
        self.summary = {} # Summary results dictionary

        # Save some commonly used parameters as attributes
        self.ncore = self.params['ncore'] # Number of cores to use
        self.ndraw_pii = self.params['ndraw_pii'] # Number of draws to compute leverage
        self.ndraw_trace = self.params['ndraw_tr'] # Number of draws to compute heteroskedastic correction
        self.compute_he = self.params['he']

        # Store some parameters in results dictionary
        self.res['cores'] = self.ncore
        self.res['ndp'] = self.ndraw_pii
        self.res['ndt'] = self.ndraw_trace

        # Create NumPy Generator instance
        self.rng = np.random.default_rng(self.params['seed'])

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
            self.__get_fe_estimates() # Add fixed effect columns
            if not self.params['feonly']: # If running full model
                self.__compute_trace_approximation_ho() # Compute trace approximation

                # If computing heteroskedastic correction
                if self.compute_he:
                    self.__compute_leverages_Pii() # Solve heteroskedastic model
                    self.__compute_trace_approximation_he() # Compute trace approximation

                self.__collect_res() # Collect all results

        end_time = time.time()

        self.res['total_time'] = end_time - self.start_time
        del self.start_time

        self.__save_res() # Save results to json

        self.__drop_cols() # Drop irrelevant columns

        self.logger.info('------ DONE -------')

    def __prep_vars(self):
        '''
        Generate some initial class attributes and results.
        '''
        self.logger.info('preparing the data')

        self.nf = self.adata.n_firms() # Number of firms
        self.nw = self.adata.n_workers() # Number of workers
        self.nn = len(self.adata) # Number of observations
        self.logger.info('data firms={} workers={} observations={}'.format(self.nf, self.nw, self.nn))

        self.res['n_firms'] = self.nf
        self.res['n_workers'] = self.nw
        self.res['n_movers'] = len(np.unique(self.adata[self.adata['m'] == 1]['i']))
        self.res['n_stayers'] = self.res['n_workers'] - self.res['n_movers']
        self.logger.info('data movers={} stayers={}'.format(self.res['n_movers'], self.res['n_stayers']))

        # Prepare 'cs' column (0 if observation is first for a worker, 1 if intermediate, 2 if last for a worker)
        worker_first_obs = (self.adata['i'] != self.adata['i'].shift(1))
        worker_last_obs = (self.adata['i'] != self.adata['i'].shift(-1))
        self.adata['cs'] = 1
        self.adata.loc[(worker_first_obs) & ~(worker_last_obs), 'cs'] = 0
        self.adata.loc[(worker_last_obs) & ~(worker_first_obs), 'cs'] = 2

        #res['year_max'] = int(sdata['year'].max())
        #res['year_min'] = int(sdata['year'].min())

    def __prep_JWM(self):
        '''
        Generate J, W, and M matrices.
        '''
        # Matrices for the cross-section
        J = csc_matrix((np.ones(self.nn), (self.adata.index, self.adata['j'])), shape=(self.nn, self.nf)) # Firms
        J = J[:, range(self.nf - 1)]  # Normalize one firm to 0
        self.J = J
        W = csc_matrix((np.ones(self.nn), (self.adata.index, self.adata['i'])), shape=(self.nn, self.nw)) # Workers
        self.W = W
        if 'w' in self.adata.columns:
            # Diagonal weight matrix
            Dp = diags(self.adata['w'])
            # Dwinv = diags(1.0 / ((W.T @ Dp @ W).diagonal())) # linalg.inv(csc_matrix(W.T @ Dp @ W))
        else:
            # Diagonal weight matrix - all weight one
            Dp = diags(np.ones(len(self.adata)))
        Dwinv = diags(1.0 / ((W.T @ Dp @ W).diagonal()))
        self.Dp = Dp
        self.Dp_sqrt = np.sqrt(Dp)
        self.Dwinv = Dwinv

        self.logger.info('Prepare linear solver')

        # Finally create M
        M = J.T @ Dp @ J - J.T @ Dp @ W @ Dwinv @ W.T @ Dp @ J
        self.M = M
        self.ml = pyamg.ruge_stuben_solver(M)

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
        Compute weighted variance.

        Arguments:
            v: vector to weight
            w: weights

        Returns:
            v0: weighted variance
        '''
        m0 = np.sum(w * v) / np.sum(w)
        v0 = np.sum(w * (v - m0) ** 2) / np.sum(w)

        return v0

    def __weighted_cov(self, v1, v2, w): # FIXME was formerly a function outside the class
        '''
        Compute weighted covariance.

        Arguments:
            v1: vector to weight
            v2: vector to weight
            w: weights

        Returns:
            v0: weighted variance
        '''
        m1 = np.sum(w * v1) / np.sum(w)
        m2 = np.sum(w * v2) / np.sum(w)
        v0 = np.sum(w * (v1 - m1) * (v2 - m2)) / np.sum(w)

        return v0

    def __compute_early_stats(self):
        '''
        Compute some early statistics.
        '''
        fdata = self.adata.groupby('j').agg({'m': 'sum', 'y': 'mean', 'i': 'count'})
        self.res['mover_quantiles'] = self.__weighted_quantile(fdata['m'], np.linspace(0, 1, 11), fdata['i']).tolist()
        self.res['size_quantiles'] = self.__weighted_quantile(fdata['i'], np.linspace(0, 1, 11), fdata['i']).tolist()
        self.res['between_firm_var'] = self.__weighted_var(fdata['y'], fdata['i'])
        self.res['var_y'] = self.__weighted_var(self.adata['y'], self.Dp)
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
            self.adata['Jq'] = 1
            self.adata['Wq'] = 1
            # Rows for csc_matrix
            self.adata['Jq_row'] = self.adata['Jq'].cumsum() - 1
            self.adata['Wq_row'] = self.adata['Wq'].cumsum() - 1
            # Columns for csc_matrix
            self.adata['Jq_col'] = self.adata['j']
            self.adata['Wq_col'] = self.adata['i']

        elif self.params['Q'] == 'cov(psi_t, psi_{t+1})':
            self.adata['Jq'] = (self.adata['m'] == 1) & ((self.adata['cs'] == 0) | (self.adata['cs'] == 1))
            self.adata['Jq_row'] = self.adata['Jq'].cumsum() - 1
            self.adata['Jq_col'] = self.adata['j']
            self.adata['Wq'] = (self.adata['m'] == 1) & ((self.adata['cs'] == 1) | (self.adata['cs'] == 2))
            self.adata['Wq_row'] = self.adata['Wq'].cumsum() - 1
            self.adata['Wq_col'] = self.adata['j']

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
        self.Yq = Jq['y']
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
        # if nW_col == self.nf: # If looking at firms, normalize one to 0
        #     Wq = Wq[:, range(self.nf - 1)]

        return Jq, Wq

    def __create_fe_solver(self):
        '''
        Solve FE model.
        '''
        self.Y = self.adata['y']

        # try to pickle the object to see its size
        # self.save('tmp.pkl') # FIXME should we delete these 2 lines?

        self.logger.info('extract firm effects')

        self.psi_hat, self.alpha_hat = self.__solve(self.Y)

        self.logger.info('solver time {:2.4f} seconds'.format(self.last_invert_time))
        self.logger.info('expected total time {:2.4f} minutes'.format( (self.ndraw_trace * (1 + self.compute_he) + self.ndraw_pii * self.compute_he) * self.last_invert_time / 60))

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

        self.var_e_pi = np.var(self.E) # Plug-in variance
        if 'w' in self.adata.columns:
            self.__compute_trace_approximation_sigma_2()
            trace_approximation = np.mean(self.tr_sigma_ho_all)
            self.var_e = (self.nn * self.var_e_pi) / (np.sum(1 / self.Dp.data[0]) - trace_approximation)
        else:
            self.var_e = (self.nn * self.var_e_pi) / (self.nn - (self.nw + self.nf - 1))
        self.logger.info('[ho] variance of residuals {:2.4f}'.format(self.var_e))

    def __compute_trace_approximation_ho(self):
        '''
        Compute weighted HO trace approximation for arbitrary Q.
        '''
        self.logger.info('Starting plug-in estimation')

        Jq, Wq = self.__construct_Jq_Wq()

        # Compute plug-in (biased) estimators
        self.tot_var = np.var(self.Y)
        if 'w' in self.adata.columns:
            self.logger.info('[weighted fe]')
            self.var_fe = self.__weighted_var(Jq * self.psi_hat, self.adata['w'])
            self.cov_fe = self.__weighted_cov(Jq * self.psi_hat, Wq * self.alpha_hat, self.adata['w'])
        else:
            self.logger.info('[fe]')
            vcv = np.cov(Jq * self.psi_hat, Wq * self.alpha_hat, ddof=0) # Set ddof=0 is necessary, otherwise takes 1 / (N - 1) by default instead of 1 / N
            self.var_fe = vcv[0, 0]
            self.cov_fe = vcv[0, 1]
        self.logger.info('var_psi={:2.4f}'.format(self.var_fe))
        self.logger.info('cov={:2.4f} tot={:2.4f}'.format(self.cov_fe, self.tot_var))

        ##### Start full Trace without collapse operator ######
        # # Begin trace approximation
        # self.tr_var_ho_all = np.zeros(self.ndraw_trace)
        # self.tr_cov_ho_all = np.zeros(self.ndraw_trace)

        # for r in trange(self.ndraw_trace):
        #     # Generate -1 or 1 - in this case length nn
        #     Z = 2 * self.rng.binomial(1, 0.5, self.nn) - 1

        #     # Compute either side of the Trace
        #     R_psi, R_alpha = self.__solve(Z)

        #     # Applying the Qcov and Qpsi implied by Jq and Wq
        #     Rq_psi = Jq @ R_psi
        #     Rq_alpha = Wq @ R_alpha

        #     self.tr_var_ho_all[r] = np.cov(Rq_psi, Rq_psi)[0][1]
        #     self.tr_cov_ho_all[r] = np.cov(Rq_psi, Rq_alpha)[0][1]

        #     self.logger.debug('FE [traces] step {}/{} done.'.format(r, self.ndraw_trace))
        ##### End full Trace without collapse operator ######

        self.logger.info('Starting weighted homoskedastic trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))

        # Begin trace approximation
        self.tr_var_ho_all = np.zeros(self.ndraw_trace)
        self.tr_cov_ho_all = np.zeros(self.ndraw_trace)

        for r in trange(self.ndraw_trace):
            # Generate -1 or 1
            Zpsi = 2 * self.rng.binomial(1, 0.5, self.nf - 1) - 1
            Zalpha = 2 * self.rng.binomial(1, 0.5, self.nw) - 1

            R1 = Jq @ Zpsi
            psi1, alpha1 = self.__mult_AAinv(Zpsi, Zalpha)
            # Trace correction - var(psi)
            R2_psi = Jq @ psi1
            self.tr_var_ho_all[r] = np.cov(R1, R2_psi)[0][1]
            # Trace correction - cov(psi, alpha)
            R2_alpha = Wq @ alpha1
            self.tr_cov_ho_all[r] = np.cov(R1, R2_alpha)[0][1]

            self.logger.debug('homoskedastic [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    # def __compute_trace_approximation_fe(self):
    #     '''
    #     Compute FE trace approximation for arbitrary Q.
    #     '''
    #     self.logger.info('Starting FE trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))

    #     Jq, Wq = self.__construct_Jq_Wq()

    #     # Compute some stats
    #     # FIXME Need to figure out when this section can be run
    #     self.tot_var = np.var(self.Y)
    #     self.logger.info('[fe]')
    #     try:
    #         # print('psi', self.psi_hat)
    #         self.var_fe = np.var(Jq * self.psi_hat)
    #         self.logger.info('var_psi={:2.4f}'.format(self.var_fe))
    #     except ValueError: # If dimension mismatch
    #         pass
    #     try:
    #         self.cov_fe = np.cov(Jq * self.psi_hat, Wq * self.alpha_hat)[0][1]
    #         self.logger.info('cov={:2.4f} tot={:2.4f}'.format(self.cov_fe, self.tot_var))
    #     except ValueError: # If dimension mismatch
    #         pass
    #     # FIXME Section ends here

    #     # Begin trace approximation
    #     self.tr_var_ho_all = np.zeros(self.ndraw_trace)
    #     self.tr_cov_ho_all = np.zeros(self.ndraw_trace)

    #     for r in trange(self.ndraw_trace):
    #         # Generate -1 or 1
    #         Zpsi = 2 * self.rng.binomial(1, 0.5, self.nf - 1) - 1
    #         Zalpha = 2 * self.rng.binomial(1, 0.5, self.nw) - 1

    #         R1 = Jq * Zpsi
    #         psi1, alpha1 = self.__mult_AAinv(Zpsi, Zalpha)
    #         try:
    #             R2_psi = Jq * psi1
    #             # Trace correction
    #             self.tr_var_ho_all[r] = np.cov(R1, R2_psi)[0][1]
    #         except ValueError: # If dimension mismatch
    #             try:
    #                 del self.tr_var_ho_all
    #             except AttributeError: # Once deleted
    #                 pass
    #         try:
    #             R2_alpha = Wq * alpha1
    #             # Trace correction
    #             self.tr_cov_ho_all[r] = np.cov(R1, R2_alpha)[0][1]
    #         except ValueError: # If dimension mismatch
    #             try:
    #                 del self.tr_cov_ho_all
    #             except AttributeError: # Once deleted
    #                 pass

    #         self.logger.debug('FE [traces] step {}/{} done.'.format(r, self.ndraw_trace))

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
    #         Zpsi = 2 * self.rng.binomial(1, 0.5, self.nf - 1) - 1
    #         Zalpha = 2 * self.rng.binomial(1, 0.5, self.nw) - 1

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
    #         Zpsi = 2 * self.rng.binomial(1, 0.5, self.nf - 1) - 1
    #         Zalpha = 2 * self.rng.binomial(1, 0.5, self.nw) - 1

    #         R1 = self.J1 * Zpsi
    #         psi1, _ = self.__mult_AAinv(Zpsi, Zalpha)
    #         R2_psi = self.J2 * psi1

    #         # Trace corrections
    #         self.tr_var_ho_all[r] = np.cov(R1, R2_psi)[0][1]
    #         self.logger.debug('FE [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    def __compute_trace_approximation_he(self):
        '''
        Compute heteroskedastic trace approximation.
        '''
        self.logger.info('Starting heteroskedastic trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))
        self.tr_var_he_all = np.zeros(self.ndraw_trace)
        self.tr_cov_he_all = np.zeros(self.ndraw_trace)

        Jq, Wq = self.__construct_Jq_Wq()

        for r in trange(self.ndraw_trace):
            # Generate -1 or 1
            Zpsi = 2 * self.rng.binomial(1, 0.5, self.nf - 1) - 1
            Zalpha = 2 * self.rng.binomial(1, 0.5, self.nw) - 1

            psi1, alpha1 = self.__mult_AAinv(Zpsi, Zalpha)
            R2_psi = Jq * psi1
            R2_alpha = Wq * alpha1

            psi2, alpha2 = self.__mult_AAinv(*self.__mult_Atranspose(self.Sii * self.__mult_A(Zpsi, Zalpha, weighted=True)))
            R3_psi = Jq * psi2

            # Trace corrections
            self.tr_var_he_all[r] = np.cov(R2_psi, R3_psi)[0][1]
            self.tr_cov_he_all[r] = np.cov(R2_alpha, R3_psi)[0][1]

            self.logger.debug('heteroskedastic [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    def __compute_trace_approximation_sigma_2(self):
        '''
        Compute weighted sigma^2 trace approximation.

        Solving Tr[A'A(A'DA)^{-1}] = Tr[A(A'DA)^{-1}A']. This is for the case where E[epsilon epsilon'|A] = sigma^2 * D^{-1}.
        
        Commented out, complex case: solving Tr[A(A'DA)^{-1}A'DD'A(A'DA)^{-1}A'] = Tr[D'A(A'DA)^{-1}A'A(A'DA)^{-1}A'D] by multiplying the right half by Z, then transposing that to get the left half.
        '''
        self.logger.info('Starting weighted sigma^2 trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))

        # Begin trace approximation
        self.tr_sigma_ho_all = np.zeros(self.ndraw_trace)

        for r in trange(self.ndraw_trace):
            # Generate -1 or 1 - in this case length nn
            Z = 2 * self.rng.binomial(1, 0.5, self.nn) - 1

            # Compute Trace
            R_psi, R_alpha = self.__solve(Z, Dp2=False)
            R_y = self.__mult_A(R_psi, R_alpha)

            self.tr_sigma_ho_all[r] = Z.T @ R_y

            # Trace when not using collapse operator:
            # # Compute either side of the Trace
            # R_y = self.__proj(Z)

            # self.tr_sigma_ho_all[r] = np.sum(R_y ** 2)

            self.logger.debug('sigma^2 [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    def __collect_res(self):
        '''
        Collect all results.
        '''
        self.res['tot_var'] = self.tot_var
        self.res['eps_var_ho'] = self.var_e
        self.res['eps_var_fe'] = np.var(self.E)
        # self.res['var_y'] = self.__weighted_var(self.Yq, self.Dp)

        # FE results
        # Plug-in variance
        self.res['var_fe'] = self.var_fe
        self.logger.info('[ho] VAR fe={:2.4f}'.format(self.var_fe))
        # Plug-in covariance
        self.logger.info('[ho] COV fe={:2.4f}'.format(self.cov_fe))
        self.res['cov_fe'] = self.cov_fe

        # Homoskedastic results
        # Trace approximation: variance
        self.res['tr_var_ho'] = np.mean(self.tr_var_ho_all)
        self.logger.info('[ho] VAR tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_var_ho'], np.std(self.tr_var_ho_all)))
        # Trace approximation: covariance
        self.res['tr_cov_ho'] = np.mean(self.tr_cov_ho_all)
        self.logger.info('[ho] COV tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_cov_ho'], np.std(self.tr_cov_ho_all)))
        # Bias-corrected variance
        self.res['var_ho'] = self.var_fe - self.var_e * self.res['tr_var_ho']
        self.logger.info('[ho] VAR bc={:2.4f}'.format(self.res['var_ho']))
        # Bias-corrected covariance
        self.res['cov_ho'] = self.cov_fe - self.var_e * self.res['tr_cov_ho']
        self.logger.info('[ho] COV bc={:2.4f}'.format(self.res['cov_ho']))

        for res in ['var_y', 'var_fe', 'cov_fe', 'var_ho', 'cov_ho']:
            self.summary[res] = self.res[res]

        # Heteroskedastic results
        if self.compute_he:
            self.res['eps_var_he'] = self.Sii.mean()
            self.res['min_lev'] = self.adata.query('m == 1').Pii.min()
            self.res['max_lev'] = self.res['max_lev'] # self.adata.query('m == 1').Pii.max() # Already computed, this just reorders the dictionary
            self.res['tr_var_he'] = np.mean(self.tr_var_he_all)
            self.res['tr_cov_he'] = np.mean(self.tr_cov_he_all)
            self.res['tr_var_ho_sd'] = np.std(self.tr_var_ho_all)
            self.res['tr_cov_ho_sd'] = np.std(self.tr_cov_ho_all)
            self.res['tr_var_he_sd'] = np.std(self.tr_var_he_all)
            self.res['tr_cov_he_sd'] = np.std(self.tr_cov_he_all)
            self.logger.info('[he] VAR tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_var_he'], np.std(self.tr_var_he_all)))
            self.logger.info('[he] COV tr={:2.4f} (sd={:2.4e})'.format(self.res['tr_cov_he'], np.std(self.tr_cov_he_all)))
            # ----- FINAL ------
            self.res['var_he'] = self.var_fe - self.res['tr_var_he']
            self.logger.info('[he] VAR fe={:2.4f} bc={:2.4f}'.format(self.var_fe, self.res['var_he']))
            self.res['cov_he'] = self.cov_fe - self.res['tr_cov_he']
            self.logger.info('[he] COV fe={:2.4f} bc={:2.4f}'.format(self.cov_fe, self.res['cov_he']))

            for res in ['var_he', 'cov_he']:
                self.summary[res] = self.res[res]

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

    def __get_fe_estimates(self):
        '''
        Add the estimated psi_hats and alpha_hats to the dataframe.
        '''
        j_vals = np.arange(self.nf) # np.arange(self.adata.j1.max()) + 1
        i_vals = np.arange(self.nw) # np.arange(self.adata.i.max()) + 1
        psi_hat_dict = dict(zip(j_vals, np.concatenate([self.psi_hat, np.array([0])]))) # Add 0 for normalized firm
        alpha_hat_dict = dict(zip(i_vals, self.alpha_hat))

        # Attach columns
        self.adata['psi_hat'] = self.adata['j'].map(psi_hat_dict)
        self.adata['alpha_hat'] = self.adata['i'].map(alpha_hat_dict)

    def __solve(self, Y, Dp1=True, Dp2=True):
        '''
        Compute (A'D_1A)^-1 A'D_2Y, the least squares estimate of A [psi_hat, alpha_hat] = Y.

        Arguments:
            Y (Pandas DataFrame): labor data
            Dp1 (bool): include first weight
            Dp2 (bool or str): include second weight, if Dp2='sqrt' then use square root of weights

        Returns:
            psi_hat: estimated firm fixed effects @ FIXME correct datatype
            alpha_hat: estimated worker fixed effects @ FIXME correct datatype
        '''
        J_transpose_Y, W_transpose_Y = self.__mult_Atranspose(Y, Dp2) # This gives A'Y
        psi_hat, alpha_hat = self.__mult_AAinv(J_transpose_Y, W_transpose_Y, Dp1)

        return psi_hat, alpha_hat

    def __mult_A(self, psi, alpha, weighted=False):
        '''
        Multiplies A = [J W] stored in the object by psi and alpha (used, for example, to compute estimated outcomes and sample errors).

        Arguments:
            psi: firm fixed effects @ FIXME correct datatype
            alpha: worker fixed effects @ FIXME correct datatype
            weighted (bool or str): include weights, if weighted='sqrt' then use square root of weights

        Returns:
            J_psi + W_alpha (CSC Matrix): firms * firm fixed effects + workers * worker fixed effects
        '''
        # J_psi = self.J * psi
        # W_alpha = self.W * alpha

        if weighted:
            if weighted == 'sqrt':
                return self.Dp_sqrt @ (self.J @ psi + self.W @ alpha)
            return self.Dp @ (self.J @ psi + self.W @ alpha)
        return self.J @ psi + self.W @ alpha # J_psi + W_alpha

    def __mult_Atranspose(self, v, weighted=True):
        '''
        Multiplies the transpose of A = [J W] stored in the object by v.

        Arguments:
            v: what to multiply by @ FIXME correct datatype
            weighted (bool or str): include weights, if weighted='sqrt' then use square root of weights

        Returns:
            J_transpose_V (CSC Matrix): firms * v
            W_transpose_V (CSC Matrix): workers * v
        '''
        if weighted:
            if weighted == 'sqrt':
                return self.J.T @ self.Dp_sqrt @ v, self.W.T @ self.Dp_sqrt @ v
            return self.J.T @ self.Dp @ v, self.W.T @ self.Dp @ v

        return self.J.T @ v, self.W.T @ v # J_transpose_V, W_transpose_V

    def __mult_AAinv(self, psi, alpha, weighted=True):
        '''
        Multiplies gamma = [psi alpha] by (A'A)^(-1) where A = [J W] stored in the object (i.e. takes (A'DA)^{-1}gamma).

        Arguments:
            psi: firm fixed effects @ FIXME correct datatype
            alpha: worker fixed effects @ FIXME correct datatype
            weighted (bool): include weights

        Returns:
            psi_out: estimated firm fixed effects @ FIXME correct datatype
            alpha_out: estimated worker fixed effects @ FIXME correct datatype
        '''
        # inter1 = self.ml.solve( psi , tol=1e-10 )
        # inter2 = self.ml.solve(  , tol=1e-10 )
        # psi_out = inter1 - inter2

        start = timer()
        if weighted:
            psi_out = self.ml.solve(psi - self.J.T * (self.Dp * (self.W * (self.Dwinv * alpha))), tol=1e-10)
            self.last_invert_time = timer() - start

            alpha_out = - self.Dwinv * (self.W.T * (self.Dp * (self.J * psi_out))) + self.Dwinv * alpha
        else:
            psi_out = self.ml.solve(psi - self.J.T * (self.W * (self.Dwinv * alpha)), tol=1e-10)
            self.last_invert_time = timer() - start

            alpha_out = - self.Dwinv * (self.W.T * (self.J * psi_out)) + self.Dwinv * alpha

        return psi_out, alpha_out

    def __proj(self, Y, Dp0=False, Dp1=True, Dp2=True):
        '''
        Solve Y, then project onto X space of data stored in the object. Essentially solves A(A'A)^{-1}A'Y

        Details:
            Solve computes (A'D_1A)^-1 A'D_2Y, the least squares estimate of A [psi_hat \\ alpha_hat] = Y.
            __mult_A computes A @ gamma = [J & W] @ [psi \\ alpha]

        Arguments:
            y (Pandas DataFrame): labor data
            Dp0 (bool or str): include weights in __mult_A, if Dp0='sqrt' then use square root of weights
            Dp1 (bool): include first weight in __solve()
            Dp2 (bool or str): include second weight in __solve(), if Dp2='sqrt' then use square root of weights

        Returns:
            Projection of psi, alpha solved from Y onto X space
        '''
        return self.__mult_A(*self.__solve(Y, Dp1, Dp2), Dp0)

    def __compute_leverages_Pii(self):
        '''
        Compute leverages for heteroskedastic correction.
        '''
        Pii = np.zeros(self.nn)

        if len(self.params['levfile']) > 1:
            self.logger.info('[he] starting heteroskedastic correction, loading precomputed files')

            files = glob.glob('{}*'.format(self.params['levfile']))
            self.logger.info('[he] found {} files to get leverages from'.format(len(files)))
            self.res['lev_file_count'] = len(files)
            assert len(files) > 0, "Didn't find any leverage files!"

            for f in files:
                pp = np.load(f)
                Pii += pp / len(files)

        elif self.ncore > 1:
            self.logger.info('[he] starting heteroskedastic correction p2={}, using {} cores, batch size {}'.format(self.ndraw_pii, self.ncore, self.params['batch']))
            set_start_method('spawn')
            with Pool(processes=self.ncore) as pool:
                Pii_all = pool.starmap(self.__leverage_approx, [self.params['batch'] for _ in range(self.ndraw_pii // self.params['batch'])])

            for pp in Pii_all:
                Pii += pp / len(Pii_all)

        else:
            Pii_all = list(itertools.starmap(self.__leverage_approx, [[self.params['batch']] for _ in range(self.ndraw_pii // self.params['batch'])]))

            for pp in Pii_all:
                Pii += pp / len(Pii_all)

        I = 1.0 * self.adata.eval('m == 1')
        self.res['max_lev'] = (I * Pii).max()

        if self.res['max_lev'] >= 1:
            leverage_warning = 'Max P_ii is {} which is >= 1. This should not happen - increase your value of ndraw_pii until this warning is no longer raised (ndraw_pii is currently set to {}).'.format(self.res['max_lev'], self.params['ndraw_pii'])
            warnings.warn(leverage_warning)
            self.logger.info(leverage_warning)

        # Attach the computed Pii to the dataframe
        self.adata['Pii'] = Pii
        self.logger.info('[he] Leverage range {:2.4f} to {:2.4f}'.format(self.adata.query('m == 1').Pii.min(), self.adata.query('m == 1').Pii.max()))
        # print('Observation with max leverage:', self.adata[self.adata['Pii'] == self.res['max_lev']])

        # Give stayers the variance estimate at the firm level
        self.adata['Sii'] = self.Y * self.E / (1 - Pii)
        S_j = pd.DataFrame(self.adata).query('m == 1').rename(columns={'Sii': 'Sii_j'}).groupby('j')['Sii_j'].agg('mean')

        Sii_j = pd.merge(self.adata['j'], S_j, on='j')['Sii_j']
        self.adata['Sii'] = np.where(self.adata['m'] == 1, self.adata['Sii'], Sii_j)
        self.Sii = self.adata['Sii']

        self.logger.info('[he] variance of residuals in heteroskedastic case: {:2.4f}'.format(self.Sii.mean()))

    def __leverage_approx(self, ndraw_pii):
        '''
        Compute an approximate leverage using ndraw_pii.

        Arguments:
            ndraw_pii (int): number of draws

        Returns:
            Pii (NumPy Array): Pii array
        '''
        Pii = np.zeros(self.nn)

        # Compute the different draws
        for r in trange(ndraw_pii):
            R2 = 2 * self.rng.binomial(1, 0.5, self.nn) - 1
            Pii += 1 / ndraw_pii * np.power(self.__proj(R2, Dp0='sqrt', Dp2='sqrt'), 2.0)

        self.logger.info('done with batch')

        return Pii

    def __drop_cols(self):
        '''
        Drop irrelevant columns (['Jq', 'Wq', 'Jq_row', 'Wq_row', 'Jq_col', 'Wq_col']).
        '''
        for col in ['Jq', 'Wq', 'Jq_row', 'Wq_row', 'Jq_col', 'Wq_col', 'Pii', 'Sii']:
            if col in self.adata.columns:
                self.adata.drop(col, inplace=True)
