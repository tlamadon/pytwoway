'''
Defines class FEEstimator, which uses multigrid and partialing out to solve two way fixed effect models. This includes AKM, the Andrews et al. homoskedastic correction, and the Kline et al. heteroskedastic correction.
'''
import warnings
from pathlib import Path
import pyamg
import numpy as np
import pandas as pd
from bipartitepandas import ParamsDict, logger_init
from scipy.sparse import csc_matrix, coo_matrix, diags, linalg, hstack, eye
import time
# import pyreadr
import os
from multiprocessing import Pool, TimeoutError, Value, set_start_method
from timeit import default_timer as timer
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

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq1(a):
    return a >= 1
def _0to1(a):
    return 0 <= a <= 1

# Define default parameter dictionary
_fe_params_default = ParamsDict({
    'ncore': (1, 'type_constrained', (int, _gteq1),
        '''
            (default=1) Number of cores to use.
        ''', '>= 1'),
    'weighted': (True, 'type', bool,
        '''
            (default=True) If True, use weighted estimators.
        ''', None),
    'statsonly': (False, 'type', bool,
        '''
            (default=False) If True, return only basic statistics.
        ''', None),
    'feonly': (False, 'type', bool,
        '''
            (default=False) If True, estimate only fixed effects and not variances.
        ''', None),
    'Q': ('cov(alpha, psi)', 'set', ['cov(alpha, psi)', 'cov(psi_t, psi_{t+1})'],
        '''
            (default='cov(alpha, psi)') Which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'.
        ''', None),
    'ndraw_trace': (5, 'type_constrained', (int, _gteq1),
        '''
            (default=5) Number of draws to use in trace approximations.
        ''', '>= 1'),
    # 'trace_analytical': (False, 'type', bool, # FIXME not used
    #     '''
    #         (default=False) If True, estimate trace analytically.
    #     ''', None)
    'he': (False, 'type', bool,
        '''
            (default=False) If True, estimate heteroskedastic correction.
        ''', None),
    'he_analytical': (False, 'type', bool,
        '''
            (default=False) If True, estimate heteroskedastic correction using analytical formula; if False, use JL approxmation.
        ''', None),
    'lev_batchsize': (50, 'type_constrained', (int, _gteq1),
        '''
            (default=50) Number of draws to use for each batch in approximation of leverages for heteroskedastic correction.
        ''', '>= 1'),
    'lev_batchsize_multiprocessing': (10, 'type_constrained', (int, _gteq1),
        '''
            (default=10) Batch size to send in parallel. Should evenly divide 'lev_batchsize'.
        ''', '>= 1'),
    'lev_nbatches': (5, 'type_constrained', (int, _gteq1),
        '''
            (default=5) Maximum number of batches to run in approximation of leverages for heteroskedastic correction.
        ''', '>= 1'),
    'lev_threshold_obs': (100, 'type_constrained', (int, _gteq1),
        '''
            (default=100) Minimum number of observations with Pii >= threshold where batches will keep running in approximation of leverages for heteroskedastic correction. Once this threshold is met, remaining Pii above threshold will be recomputed analytically.
        ''', '>= 1'),
    'lev_threshold_pii': (0.98, 'type_constrained', (float, _0to1),
        '''
            (default=0.98) Threshold Pii value for computing threshold number of Pii observations in approximation of leverages for heteroskedastic correction.
        ''', 'in [0, 1]'),
    'levfile': ('', 'type', str,
        '''
            (default='') File to load precomputed leverages for heteroskedastic correction.
        ''', None),
    # 'con': (False, 'type', bool, # FIXME not used
    #     '''
    #         (default=False) Computes the smallest eigen values, this is the filepath where these results are saved.
    #     ''', None),
    'out': ('res_fe.json', 'type', str,
        '''
            (default='res_fe.json') Outputfile where results are saved.
        ''', None)
})

def fe_params(update_dict={}):
    '''
    Dictionary of default fe_params. Run tw.fe_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of fe_params
    '''
    new_dict = _fe_params_default.copy()
    new_dict.update(update_dict)
    return new_dict

def _weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
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

def _weighted_var(v, w):
    '''
    Compute weighted variance.

    Arguments:
        v (NumPy Array): vector to weight
        w (NumPy Array): weights

    Returns:
        v0 (NumPy Array): weighted variance
    '''
    m0 = np.sum(w * v) / np.sum(w)
    v0 = np.sum(w * (v - m0) ** 2) / np.sum(w)

    return v0

def _weighted_cov(v1, v2, w):
    '''
    Compute weighted covariance.

    Arguments:
        v1 (NumPy Array): vector to weight
        v2 (NumPy Array): vector to weight
        w (NumPy Array): weights

    Returns:
        v0 (NumPy Array): weighted variance
    '''
    m1 = np.sum(w * v1) / np.sum(w)
    m2 = np.sum(w * v2) / np.sum(w)
    v0 = np.sum(w * (v1 - m1) * (v2 - m2)) / np.sum(w)

    return v0

class FEEstimator:
    '''
    Uses multigrid and partialing out to solve two way fixed effect models. This includes AKM, the Andrews et al. homoskedastic correction, and the Kline et al. heteroskedastic correction.
    '''

    def __init__(self, data, params=fe_params()):
        '''
        Arguments:
            data (BipartitePandas DataFrame): (collapsed) long format labor data. Data contains the following columns:

                i (worker id)

                j (firm id)

                y (compensation)

                t (period) if long

                t1 (first period of observation) if collapsed long

                t2 (last period of observation) if collapsed long

                w (weight)

                m (0 if stayer, 1 if mover)
            params (ParamsDict): dictionary of parameters for FE estimation. Run tw.fe_params().describe_all() for descriptions of all valid parameters.
        '''
        # Start logger
        logger_init(self)
        # self.logger.info('initializing FEEstimator object')

        self.adata = data

        self.params = params
        # Results dictionary
        self.res = {}
        # Summary results dictionary
        self.summary = {}

        ## Save some commonly used parameters as attributes
        # Number of cores to use
        self.ncore = self.params['ncore']
        # Number of draws to compute leverage for heteroskedastic correction
        self.lev_batchsize = self.params['lev_batchsize']
        # Number of draws to use in trace approximations
        self.ndraw_trace = self.params['ndraw_trace']
        self.compute_he = self.params['he']

        ## Store some parameters in results dictionary
        self.res['cores'] = self.ncore
        self.res['ndp'] = self.lev_batchsize
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
        # Make d the attribute dictionary
        self.__dict__ = d
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

    def fit(self, rng=np.random.default_rng(None)):
        '''
        Run FE solver.

        Arguments:
            rng (np.random.Generator): NumPy random number generator
        '''
        self.fit_1()
        self.construct_Q()
        self.fit_2(rng)

    def fit_1(self):
        '''
        Run FE solver, part 1. Before fit_2(), modify adata to allow creation of Q matrix.
        '''
        self.start_time = time.time()

        # Begin cleaning and analysis
        self._prep_vars() # Prepare data
        self._prep_JWM() # Use cleaned adata to generate some attributes
        self._compute_early_stats() # Use cleaned data to compute some statistics

    def fit_2(self, rng=np.random.default_rng(None)):
        '''
        Run FE solver, part 2.

        Arguments:
            rng (np.random.Generator): NumPy random number generator
        '''
        if self.params['statsonly']:
            # If only returning early statistics
            self._save_early_stats()

        else:
            ## If running analysis
            # Solve FE model
            self._create_fe_solver(rng)
            # Add fixed effect columns
            self._get_fe_estimates()

            if not self.params['feonly']:
                ## If running full model
                # Compute trace approximation
                self._compute_trace_approximation_ho(rng)

                if self.compute_he:
                    ## If computing heteroskedastic correction
                    # Solve heteroskedastic model
                    self._compute_leverages_Pii(rng)
                    # Compute trace approximation
                    self._compute_trace_approximation_he(rng)

                # Collect all results
                self._collect_res()

        end_time = time.time()

        self.res['total_time'] = end_time - self.start_time
        del self.start_time

        # Save results to json
        self._save_res()

        # Drop irrelevant columns
        self._drop_cols()

        self.logger.info('------ DONE -------')

    def _prep_vars(self):
        '''
        Generate some initial class attributes and results.
        '''
        self.logger.info('preparing the data')
        # self.adata.sort_values(['i', to_list(self.adata.reference_dict['t'])[0]], inplace=True)

        # Number of firms
        self.nf = self.adata.n_firms()
        # Number of workers
        self.nw = self.adata.n_workers()
        # Number of observations
        self.nn = len(self.adata)
        self.logger.info('data firms={} workers={} observations={}'.format(self.nf, self.nw, self.nn))

        self.res['n_firms'] = self.nf
        self.res['n_workers'] = self.nw
        self.res['n_movers'] = self.adata.loc[self.adata['m'].to_numpy() > 0, :].n_workers()
        self.res['n_stayers'] = self.res['n_workers'] - self.res['n_movers']
        self.logger.info('data movers={} stayers={}'.format(self.res['n_movers'], self.res['n_stayers']))

        # Generate 'worker_m' indicating whether a worker is a mover or a stayer
        self.adata.loc[:, 'worker_m'] = (self.adata.groupby('i')['m'].transform('max') > 0).astype(int, copy=False)

        # # Prepare 'cs' column (0 if observation is first for a worker, 1 if intermediate, 2 if last for a worker)
        # worker_first_obs = (self.adata['i'].to_numpy() != np.roll(self.adata['i'].to_numpy(), 1))
        # worker_last_obs = (self.adata['i'].to_numpy() != np.roll(self.adata['i'].to_numpy(), -1))
        # self.adata['cs'] = 1
        # self.adata.loc[(worker_first_obs) & ~(worker_last_obs), 'cs'] = 0
        # self.adata.loc[(worker_last_obs) & ~(worker_first_obs), 'cs'] = 2

        #res['year_max'] = int(sdata['year'].max())
        #res['year_min'] = int(sdata['year'].min())

    def _prep_JWM(self):
        '''
        Generate J, W, and M matrices.
        '''
        ### Matrices for the cross-section
        ## Firms
        J = csc_matrix((np.ones(self.nn), (self.adata.index.to_numpy(), self.adata['j'].to_numpy())), shape=(self.nn, self.nf))
        # Normalize one firm to 0
        J = J[:, range(self.nf - 1)]
        self.J = J
        ## Workers
        W = csc_matrix((np.ones(self.nn), (self.adata.index.to_numpy(), self.adata['i'].to_numpy())), shape=(self.nn, self.nw))
        self.W = W
        if self.params['weighted'] and ('w' in self.adata.columns):
            # Diagonal weight matrix
            Dp = diags(self.adata['w'].to_numpy())
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

    def _compute_early_stats(self):
        '''
        Compute some early statistics.
        '''
        fdata = self.adata.groupby('j').agg({'worker_m': 'sum', 'y': 'mean', 'i': 'count'})
        fm, fy, fi = fdata.loc[:, 'worker_m'].to_numpy(), fdata.loc[:, 'y'].to_numpy(), fdata.loc[:, 'i'].to_numpy()
        ls = np.linspace(0, 1, 11)
        self.res['mover_quantiles'] = _weighted_quantile(fm, ls, fi).tolist()
        self.res['size_quantiles'] = _weighted_quantile(fi, ls, fi).tolist()
        # self.res['movers_per_firm'] = self.adata.loc[self.adata.loc[:, 'm'] > 0, :].groupby('j')['i'].nunique().mean()
        self.res['between_firm_var'] = _weighted_var(fy, fi)
        self.res['var_y'] = _weighted_var(self.adata.loc[:, 'y'].to_numpy(), self.Dp)
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

    def _save_early_stats(self):
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
            # self.adata['Jq'] = 1
            # self.adata['Wq'] = 1
            # Rows for csc_matrix
            self.adata.loc[:, 'Jq_row'] = np.arange(self.nn) # self.adata['Jq'].cumsum() - 1
            self.adata.loc[:, 'Wq_row'] = np.arange(self.nn) # self.adata['Wq'].cumsum() - 1
            # Columns for csc_matrix
            self.adata.loc[:, 'Jq_col'] = self.adata.loc[:, 'j']
            self.adata.loc[:, 'Wq_col'] = self.adata.loc[:, 'i']

        elif self.params['Q'] in ['cov(psi_t, psi_{t+1})', 'cov(psi_i, psi_j)']:
            warnings.warn('These Q options are not yet implemented.')

        # elif self.params['Q'] == 'cov(psi_t, psi_{t+1})':
        #     self.adata['Jq'] = (self.adata['worker_m'] > 0) & ((self.adata['cs'] == 0) | (self.adata['cs'] == 1))
        #     self.adata['Jq_row'] = self.adata['Jq'].cumsum() - 1
        #     self.adata['Jq_col'] = self.adata['j']
        #     self.adata['Wq'] = (self.adata['worker_m'] > 0) & ((self.adata['cs'] == 1) | (self.adata['cs'] == 2))
        #     self.adata['Wq_row'] = self.adata['Wq'].cumsum() - 1
        #     self.adata['Wq_col'] = self.adata['j']

        # elif self.params['Q'] == 'cov(psi_i, psi_j)': # Code doesn't work
        #     self.adata['Jq'] = (self.adata['worker_m'] > 0) & (self.adata['cs'] == 1)
        #     self.adata['Jq_row'] = self.adata['j1']
        #     self.adata['Jq_col'] = self.adata['j1']
        #     self.adata['Wq'] = (self.adata['worker_m'] > 0) & (self.adata['cs'] == 0)
        #     # Recall j1, j2 swapped for m==1 and cs==0
        #     self.adata['Wq_row'] = self.adata['j2']
        #     self.adata['Wq_col'] = self.adata['j1']

    def __construct_Jq_Wq(self):
        '''
        Construct Jq and Wq matrices.

        Returns:
            Jq (Pandas DataFrame): left matrix for computing Q
            Wq (Pandas DataFrame): right matrix for computing Q
        '''
        # FIXME this method is irrelevant at the moment
        return self.J, self.W

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

    def _create_fe_solver(self, rng=np.random.default_rng(None)):
        '''
        Solve FE model.

        Arguments:
            rng (np.random.Generator): NumPy random number generator
        '''
        self.Y = self.adata.loc[:, 'y'].to_numpy()

        # try to pickle the object to see its size
        # self.save('tmp.pkl') # FIXME should we delete these 2 lines?

        self.logger.info('extract firm effects')

        self.psi_hat, self.alpha_hat = self.__solve(self.Y)

        self.logger.info('solver time {:2.4f} seconds'.format(self.last_invert_time))
        self.logger.info('expected total time {:2.4f} minutes'.format( (self.ndraw_trace * (1 + self.compute_he) + self.lev_batchsize * self.params['lev_nbatches'] * self.compute_he) * self.last_invert_time / 60))

        self.E = self.Y - self.__mult_A(self.psi_hat, self.alpha_hat)

        self.res['solver_time'] = self.last_invert_time

        fe_rsq = 1 - np.power(self.E, 2).mean() / np.power(self.Y, 2).mean()
        self.logger.info('fixed effect R-square {:2.4f}'.format(fe_rsq))

        # Plug-in variance
        self.var_e_pi = np.var(self.E)
        if self.params['weighted'] and ('w' in self.adata.columns):
            self._compute_trace_approximation_sigma_2(rng)
            trace_approximation = np.mean(self.tr_sigma_ho_all)
            self.var_e = (self.nn * self.var_e_pi) / (np.sum(1 / self.Dp.data[0]) - trace_approximation)
        else:
            self.var_e = (self.nn * self.var_e_pi) / (self.nn - (self.nw + self.nf - 1))
        self.logger.info('[ho] variance of residuals {:2.4f}'.format(self.var_e))

    def _compute_trace_approximation_ho(self, rng=np.random.default_rng(None)):
        '''
        Compute weighted HO trace approximation for arbitrary Q.

        Arguments:
            rng (np.random.Generator): NumPy random number generator
        '''
        self.logger.info('Starting plug-in estimation')

        Jq, Wq = self.__construct_Jq_Wq()

        # Compute plug-in (biased) estimators
        self.tot_var = np.var(self.Y)
        if 'w' in self.adata.columns:
            self.logger.info('[weighted fe]')
            self.var_fe = _weighted_var(Jq * self.psi_hat, self.Dp)
            self.cov_fe = _weighted_cov(Jq * self.psi_hat, Wq * self.alpha_hat, self.Dp)
        else:
            self.logger.info('[fe]')
            # Set ddof=0 is necessary, otherwise takes 1 / (N - 1) by default instead of 1 / N
            vcv = np.cov(Jq * self.psi_hat, Wq * self.alpha_hat, ddof=0)
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
        #     Z = 2 * rng.binomial(1, 0.5, self.nn) - 1

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
            Zpsi = 2 * rng.binomial(1, 0.5, self.nf - 1) - 1
            Zalpha = 2 * rng.binomial(1, 0.5, self.nw) - 1

            R1 = Jq @ Zpsi
            psi1, alpha1 = self.__mult_AAinv(Zpsi, Zalpha)
            # Trace correction - var(psi)
            R2_psi = Jq @ psi1
            self.tr_var_ho_all[r] = np.cov(R1, R2_psi)[0][1]
            # Trace correction - cov(psi, alpha)
            R2_alpha = Wq @ alpha1
            self.tr_cov_ho_all[r] = np.cov(R1, R2_alpha)[0][1]

            self.logger.debug('homoskedastic [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    # def __compute_trace_approximation_fe(self, rng=np.random.default_rng(None)):
    #     '''
    #     Compute FE trace approximation for arbitrary Q.

    #     Arguments:
    #         rng (np.random.Generator): NumPy random number generator
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
    #         Zpsi = 2 * rng.binomial(1, 0.5, self.nf - 1) - 1
    #         Zalpha = 2 * rng.binomial(1, 0.5, self.nw) - 1

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

    # def compute_trace_approximation_fe(self, rng=np.random.default_rng(None)):
    #     '''
    #     Purpose:
    #         Compute FE trace approximation.

    #     Arguments:
    #         rng (np.random.Generator): NumPy random number generator
    #     '''
    #     self.logger.info('Starting FE trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))
    #     self.tr_var_ho_all = np.zeros(self.ndraw_trace)
    #     self.tr_cov_ho_all = np.zeros(self.ndraw_trace)

    #     for r in trange(self.ndraw_trace):
    #         # Generate -1 or 1
    #         Zpsi = 2 * rng.binomial(1, 0.5, self.nf - 1) - 1
    #         Zalpha = 2 * rng.binomial(1, 0.5, self.nw) - 1

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
    #         Zpsi = 2 * rng.binomial(1, 0.5, self.nf - 1) - 1
    #         Zalpha = 2 * rng.binomial(1, 0.5, self.nw) - 1

    #         R1 = self.J1 * Zpsi
    #         psi1, _ = self.__mult_AAinv(Zpsi, Zalpha)
    #         R2_psi = self.J2 * psi1

    #         # Trace corrections
    #         self.tr_var_ho_all[r] = np.cov(R1, R2_psi)[0][1]
    #         self.logger.debug('FE [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    def _compute_trace_approximation_he(self, rng=np.random.default_rng(None)):
        '''
        Compute heteroskedastic trace approximation.

        Arguments:
            rng (np.random.Generator): NumPy random number generator
        '''
        self.logger.info('Starting heteroskedastic trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))
        self.tr_var_he_all = np.zeros(self.ndraw_trace)
        self.tr_cov_he_all = np.zeros(self.ndraw_trace)

        Jq, Wq = self.__construct_Jq_Wq()

        for r in trange(self.ndraw_trace):
            # Generate -1 or 1
            Zpsi = 2 * rng.binomial(1, 0.5, self.nf - 1) - 1
            Zalpha = 2 * rng.binomial(1, 0.5, self.nw) - 1

            psi1, alpha1 = self.__mult_AAinv(Zpsi, Zalpha)
            R2_psi = Jq * psi1
            R2_alpha = Wq * alpha1

            psi2, alpha2 = self.__mult_AAinv(*self.__mult_Atranspose(self.Sii * self.__mult_A(Zpsi, Zalpha, weighted=True)))
            R3_psi = Jq * psi2

            # Trace corrections
            self.tr_var_he_all[r] = np.cov(R2_psi, R3_psi)[0][1]
            self.tr_cov_he_all[r] = np.cov(R2_alpha, R3_psi)[0][1]

            self.logger.debug('heteroskedastic [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    def _compute_trace_approximation_sigma_2(self, rng=np.random.default_rng(None)):
        '''
        Compute weighted sigma^2 trace approximation.

        Solving Tr[A'A(A'DA)^{-1}] = Tr[A(A'DA)^{-1}A']. This is for the case where E[epsilon epsilon'|A] = sigma^2 * D^{-1}.
        
        Commented out, complex case: solving Tr[A(A'DA)^{-1}A'DD'A(A'DA)^{-1}A'] = Tr[D'A(A'DA)^{-1}A'A(A'DA)^{-1}A'D] by multiplying the right half by Z, then transposing that to get the left half.

        Arguments:
            rng (np.random.Generator): NumPy random number generator
        '''
        self.logger.info('Starting weighted sigma^2 trace correction ndraws={}, using {} cores'.format(self.ndraw_trace, self.ncore))

        # Begin trace approximation
        self.tr_sigma_ho_all = np.zeros(self.ndraw_trace)

        for r in trange(self.ndraw_trace):
            # Generate -1 or 1 - in this case length nn
            Z = 2 * rng.binomial(1, 0.5, self.nn) - 1

            # Compute Trace
            R_psi, R_alpha = self.__solve(Z, Dp2=False)
            R_y = self.__mult_A(R_psi, R_alpha)

            self.tr_sigma_ho_all[r] = Z.T @ R_y

            # Trace when not using collapse operator:
            # # Compute either side of the Trace
            # R_y = self.__proj(Z)

            # self.tr_sigma_ho_all[r] = np.sum(R_y ** 2)

            self.logger.debug('sigma^2 [traces] step {}/{} done.'.format(r, self.ndraw_trace))

    def _collect_res(self):
        '''
        Collect all results.
        '''
        self.res['tot_var'] = self.tot_var
        self.res['eps_var_ho'] = self.var_e
        self.res['eps_var_fe'] = np.var(self.E)
        # self.res['var_y'] = _weighted_var(self.Yq, self.Dp)

        ## FE results ##
        # Plug-in variance
        self.res['var_fe'] = self.var_fe
        self.logger.info('[ho] VAR fe={:2.4f}'.format(self.var_fe))
        # Plug-in covariance
        self.logger.info('[ho] COV fe={:2.4f}'.format(self.cov_fe))
        self.res['cov_fe'] = self.cov_fe

        ## Homoskedastic results ##
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

        ## Heteroskedastic results ##
        if self.compute_he:
            ## Already computed, this just reorders the dictionary
            self.res['eps_var_he'] = self.res['eps_var_he']
            self.res['min_lev'] = self.res['min_lev']
            self.res['max_lev'] = self.res['max_lev']
            ## New results
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

    def _save_res(self):
        '''
        Save results as json.
        '''
        # Convert results into strings to prevent JSON errors
        for key, val in self.res.items():
            self.res[key] = str(val)

        with open(self.params['out'], 'w') as outfile:
            json.dump(self.res, outfile)

        self.logger.info('saved results to {}'.format(self.params['out']))

    def _get_fe_estimates(self):
        '''
        Add the estimated psi_hats and alpha_hats to the dataframe.
        '''
        j_vals = np.arange(self.nf)
        i_vals = np.arange(self.nw)

        # Add 0 for normalized firm
        psi_hat_dict = dict(zip(j_vals, np.concatenate([self.psi_hat, np.array([0])])))
        alpha_hat_dict = dict(zip(i_vals, self.alpha_hat))

        # Attach columns
        self.adata.loc[:, 'psi_hat'] = self.adata.loc[:, 'j'].map(psi_hat_dict)
        self.adata.loc[:, 'alpha_hat'] = self.adata.loc[:, 'i'].map(alpha_hat_dict)

    def __solve(self, Y, Dp1=True, Dp2=True):
        '''
        Compute (A' * Dp1 * A)^{-1} * A' * Dp2 * Y, the least squares estimate of Y = A * [psi_hat' alpha_hat']', where A = [J W] (J is firm indicators and W is worker indicators) and Dp gives weights.

        Arguments:
            Y (NumPy Array): wage data
            Dp1 (bool): if True, include first weight
            Dp2 (bool or str): if True, include second weight; if 'sqrt', use square root of weights

        Returns:
            (tuple of CSC Matrices): (estimated firm fixed effects, estimated worker fixed effects)
        '''
        # This gives A' * Dp2 * Y
        J_transpose_Y, W_transpose_Y = self.__mult_Atranspose(Y, Dp2)
        # This gives (A' * Dp1 * A)^{-1} * A' * Dp2 * Y
        psi_hat, alpha_hat = self.__mult_AAinv(J_transpose_Y, W_transpose_Y, Dp1)

        return psi_hat, alpha_hat

    def __mult_A(self, psi, alpha, weighted=False):
        '''
        Computes Dp * A * [psi' alpha']', where A = [J W] (J is firm indicators and W is worker indicators) and Dp gives weights (used, for example, to compute estimated outcomes and sample errors).

        Arguments:
            psi (NumPy Array): firm part to multiply
            alpha (NumPy Array): worker part to multiply
            weighted (bool or str): if True, include weights; if 'sqrt', use square root of weights

        Returns:
            (CSC Matrix): result of Dp * A * [psi' alpha']'
        '''
        if weighted:
            if weighted == 'sqrt':
                return self.Dp_sqrt @ (self.J @ psi + self.W @ alpha)
            return self.Dp @ (self.J @ psi + self.W @ alpha)
        return self.J @ psi + self.W @ alpha

    def __mult_Atranspose(self, v, weighted=True):
        '''
        Computes A' * Dp * v, where A = [J W] (J is firm indicators and W is worker indicators) and Dp gives weights.

        Arguments:
            v (NumPy Array): what to multiply by
            weighted (bool or str): if True, include weights; if 'sqrt', use square root of weights

        Returns:
            (tuple of CSC Matrices): (firm part of result, worker part of result)
        '''
        if weighted:
            if weighted == 'sqrt':
                return self.J.T @ self.Dp_sqrt @ v, self.W.T @ self.Dp_sqrt @ v
            return self.J.T @ self.Dp @ v, self.W.T @ self.Dp @ v

        return self.J.T @ v, self.W.T @ v

    def __mult_AAinv(self, psi, alpha, weighted=True):
        '''
        Computes (A' * Dp * A)^{-1} * [psi' alpha']', where A = [J W] (J is firm indicators and W is worker indicators) and Dp gives weights.

        Arguments:
            psi (NumPy Array): firm part to multiply
            alpha (NumPy Array): worker part to multiply
            weighted (bool): if True, include weights

        Returns:
            (tuple of NumPy Arrays): (firm part of result, worker part of result)
        '''
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
        Compute Dp0 * A * (A' * Dp1 * A)^{-1} * A' * Dp2 * Y, where A = [J W] (J is firm indicators and W is worker indicators) and Dp gives weights (essentially projects Y onto A space).
        Solve Y, then project onto X space of data stored in the object. Essentially solves A(A'A)^{-1}A'Y

        Arguments:
            Y (NumPy Array): wage data
            Dp0 (bool or str): if True, include weights in __mult_A(); if 'sqrt', use square root of weights
            Dp1 (bool): if True, include first weight in __solve()
            Dp2 (bool or str): if True, include second weight in __solve(); if 'sqrt', use square root of weights

        Returns:
            (CSC Matrix): result of Dp0 * A * (A' * Dp1 * A)^{-1} * A' * Dp2 * Y (essentially the projection of Y onto A space)
        '''
        return self.__mult_A(*self.__solve(Y, Dp1, Dp2), Dp0)

    def __construct_M_inv(self):
        '''
        Construct (A' * Dp * A)^{-1} block matrix components where M^{-1} is computed explicitly.
        '''
        # Define variables
        J = self.J
        W = self.W
        Minv = np.linalg.inv(self.M.todense())
        Dp = self.Dp
        Dwinv = self.Dwinv

        # Construct blocks
        self.AA_inv_A = Minv
        self.AA_inv_B = - Minv @ J.T @ Dp @ W @ Dwinv
        self.AA_inv_C = - Dwinv @ W.T @ Dp @ J @ Minv
        self.AA_inv_D = Dwinv + self.AA_inv_C @ self.M @ self.AA_inv_B # Dwinv @ (eye(nw) + W.T @ Dp @ J @ M @ J.T @ Dp @ W @ Dwinv)

    def __construct_AAinv_components(self):
        '''
        Construct (A' * Dp * A)^{-1} block matrix components. Use this for computing a small number of individual Pii.
        '''
        # Define variables
        J = self.J
        W = self.W

        Dp = self.Dp
        Dwinv = self.Dwinv

        # Construct blocks
        self.AA_inv_A = None
        self.AA_inv_B = J.T @ Dp @ W @ Dwinv
        self.AA_inv_C = - Dwinv @ W.T @ Dp @ J
        self.AA_inv_D = None

    def __compute_Pii(self, DpJ_i, DpW_i):
        '''
        Compute Pii for a single observation for heteroskedastic correction.

        Arguments:
            DpJ_i (NumPy Array): weighted J matrix
            DpW_i (NumPy Array): weighted W matrix

        Returns:
            (float): estimate for Pii
        '''
        if self.AA_inv_A is not None:
            # M^{-1} has been explicitly computed
            A = DpJ_i @ self.AA_inv_A @ DpJ_i
            B = DpJ_i @ self.AA_inv_B @ DpW_i
            C = DpW_i @ self.AA_inv_C @ DpJ_i
            D = DpW_i @ self.AA_inv_D @ DpW_i
        else:
            # M^{-1} has not been explicitly computed
            M_DpJ_i = self.ml.solve(DpJ_i)
            M_B = self.ml.solve(self.AA_inv_B @ DpW_i)

            # Construct blocks
            A = DpJ_i @ M_DpJ_i
            B = - DpJ_i @ M_B
            C = DpW_i @ self.AA_inv_C @ M_DpJ_i
            D = DpW_i @ (self.Dwinv @ DpW_i - self.AA_inv_C @ M_B)

        return A + B + C + D

    def _compute_leverages_Pii(self, rng=np.random.default_rng(None)):
        '''
        Compute leverages for heteroskedastic correction.

        Arguments:
            rng (np.random.Generator): NumPy random number generator
        '''
        Pii = np.zeros(self.nn)
        # Indices to compute Pii analytically
        analytical_indices = []
        worker_m = (self.adata.loc[:, 'worker_m'].to_numpy() > 0)

        if self.params['he_analytical']:
            analytical_indices = self.adata.loc[worker_m, :].index
            self.__construct_M_inv()

        else:
            if len(self.params['levfile']) > 1:
                self.logger.info('[he] starting heteroskedastic correction, loading precomputed files')

                files = glob.glob('{}*'.format(self.params['levfile']))
                self.logger.info('[he] found {} files to get leverages from'.format(len(files)))
                self.res['lev_file_count'] = len(files)
                assert len(files) > 0, "Didn't find any leverage files!"

                for f in files:
                    pp = np.load(f)
                    Pii += pp / len(files)

            else:
                self.logger.info('[he] starting heteroskedastic correction lev_batchsize={}, lev_nbatches={}, using {} cores'.format(self.params['lev_batchsize'], self.params['lev_nbatches'], self.ncore))
                for batch_i in range(self.params['lev_nbatches']):
                    if self.ncore > 1:
                        # Multiprocessing
                        ndraw_seeds = self.lev_batchsize // self.params['lev_batchsize_multiprocessing']
                        if np.round(ndraw_seeds * self.params['lev_batchsize_multiprocessing']) != self.lev_batchsize:
                            # 'lev_batchsize_multiprocessing' must evenly divide 'lev_batchsize'
                            raise ValueError("'lev_batchsize_multiprocessing' (currently {}) should evenly divide 'lev_batchsize' (currently {}).".format(self.params['lev_batchsize_multiprocessing'], self.lev_batchsize))
                        # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
                        seeds = rng.bit_generator._seed_seq.spawn(ndraw_seeds)
                        set_start_method('spawn')
                        with Pool(processes=self.ncore) as pool:
                            Pii_all = pool.starmap(self._leverage_approx, [(self.params['lev_batchsize_multiprocessing'], np.random.default_rng(seed)) for seed in seeds])

                        # Take mean over draws
                        Pii_i = sum(Pii_all) / len(Pii_all)
                    else:
                        # Single core
                        Pii_i = self._leverage_approx(self.lev_batchsize, rng)

                    # Take weighted average over all Pii draws
                    Pii = (batch_i * Pii + Pii_i) / (batch_i + 1)

                    # Compute number of bad draws
                    n_bad_draws = sum(worker_m & (Pii >= self.params['lev_threshold_pii']))

                    # If few enough bad draws, compute them analytically
                    if n_bad_draws < self.params['lev_threshold_obs']:
                        leverage_warning = 'Threshold for max Pii is {}, with {} draw(s) per batch and a maximum of {} batch(es) being drawn. There is/are {} observation(s) with Pii above this threshold. These will be recomputed analytically. It took {} batch(es) to get below the threshold of {} bad observations.'.format(self.params['lev_threshold_pii'], self.lev_batchsize, self.params['lev_nbatches'], n_bad_draws, batch_i + 1, self.params['lev_threshold_obs'])
                        warnings.warn(leverage_warning)
                        break
                    elif batch_i == self.params['lev_nbatches'] - 1:
                        leverage_warning = 'Threshold for max Pii is {}, with {} draw(s) per batch and a maximum of {} batch(es) being drawn. After exhausting the maximum number of batches, there is/are still {} draw(s) with Pii above this threshold. These will be recomputed analytically.'.format(self.params['lev_threshold_pii'], self.lev_batchsize, self.params['lev_nbatches'], n_bad_draws)
                        warnings.warn(leverage_warning)

            # Compute Pii analytically for observations with Pii approximation above threshold value
            analytical_indices = self.adata.loc[worker_m & (Pii >= self.params['lev_threshold_pii']), :].index
            if len(analytical_indices) > 0:
                self.__construct_AAinv_components()

        # Compute analytical Pii
        if len(analytical_indices) > 0:
            # Construct weighted J and W
            DpJ = np.asarray((self.Dp_sqrt @ self.J).todense())
            DpW = np.asarray((self.Dp_sqrt @ self.W).todense())

            for i in analytical_indices:
                DpJ_i = DpJ[i, :]
                DpW_i = DpW[i, :]
                Pii[i] = self.__compute_Pii(DpJ_i, DpW_i)

        self.res['min_lev'] = Pii[worker_m].min()
        self.res['max_lev'] = Pii[worker_m].max()

        if self.res['max_lev'] >= 1:
            leverage_warning = "Max P_ii is {} which is >= 1. This means your data is not leave-one-observation-out connected. The HE estimator requires leave-one-observation-out connected data to work properly. When cleaning your data, please set clean_params['connectedness'] = 'leave_one_observation_out' to correct this.".format(self.res['max_lev'])
            self.logger.info(leverage_warning)
            raise ValueError(leverage_warning)
            # self.adata['Pii'] = Pii
            # self.adata.to_feather('pii_data.ftr')
            # raise NotImplementedError

        self.logger.info('[he] Leverage range {:2.4f} to {:2.4f}'.format(self.res['min_lev'], self.res['max_lev']))
        # print('Observation with max leverage:', self.adata[self.adata['Pii'] == self.res['max_lev']])

        ## Give stayers the variance estimate at the firm level ##
        # Temporarily set Pii = 0 for stayers to avoid divide-by-zero warning
        Pii[~worker_m] = 0
        # Compute Sii for movers
        self.adata.loc[:, 'Sii'] = self.Y * self.E / (1 - Pii)
        # Link firms to average Sii of movers
        S_j = self.adata.loc[worker_m, :].groupby('j')['Sii'].mean().to_dict()
        Sii_j = self.adata.loc[:, 'j'].map(S_j)
        self.Sii = np.where(worker_m, self.adata.loc[:, 'Sii'], Sii_j)
        # No longer need Sii column
        self.adata.drop('Sii', axis=1, inplace=True)

        self.res['eps_var_he'] = self.Sii.mean()
        self.logger.info('[he] variance of residuals in heteroskedastic case: {:2.4f}'.format(self.res['eps_var_he']))

    def _leverage_approx(self, ndraw_pii, rng=np.random.default_rng(None)):
        '''
        Draw Pii estimates for use in JL approximation of leverage.

        Arguments:
            ndraw_pii (int): number of Pii draws
            rng (np.random.Generator): NumPy random number generator

        Returns:
            Pii (NumPy Array): Pii array
        '''
        Pii = np.zeros(self.nn)

        # Compute the different draws
        for _ in range(ndraw_pii):
            R2 = 2 * rng.binomial(1, 0.5, self.nn) - 1
            Pii += np.power(self.__proj(R2, Dp0='sqrt', Dp2='sqrt'), 2.0)

        # Take mean over draws
        Pii /= ndraw_pii

        self.logger.info('done with batch')

        return Pii

    def _drop_cols(self):
        '''
        Drop irrelevant columns (['worker_m', 'Jq', 'Wq', 'Jq_row', 'Wq_row', 'Jq_col', 'Wq_col']).
        '''
        for col in ['worker_m', 'Jq', 'Wq', 'Jq_row', 'Wq_row', 'Jq_col', 'Wq_col']:
            if col in self.adata.columns:
                self.adata.drop(col, axis=1, inplace=True)
