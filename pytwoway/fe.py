'''
Defines class FEEstimator, which uses multigrid and partialing out to estimate weighted two way fixed effect models. This includes AKM, the Andrews et al. homoskedastic correction, and the Kline et al. heteroskedastic correction.
'''
from tqdm.auto import tqdm, trange
import time, pickle, json, glob # warnings
from timeit import default_timer as timer
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, diags, eye
import pyamg
from bipartitepandas.util import ParamsDict, logger_init
from pytwoway import Q

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
    'attach_fe_estimates': (False, 'type', bool,
        '''
            (default=False) If True, attach the estimated psi_hat and alpha_hat as columns to the input dataframe.
        ''', None),
    'he': (False, 'type', bool,
        '''
            (default=False) If True, estimate heteroskedastic correction.
        ''', None),
    'Q_var': (None, 'type_none', (Q.VarPsi, Q.VarAlpha),
        '''
            (default=None) Which Q matrix to use when estimating variance term; None is equivalent to tw.Q.VarPsi().
        ''', None),
    'Q_cov': (None, 'type_none', (Q.CovPsiAlpha, Q.CovPsiPrevPsiNext),
        '''
            (default=None) Which Q matrix to use when estimating covariance term; None is equivalent to tw.Q.CovPsiAlpha().
        ''', None),
    'ndraw_trace_sigma_2': (5, 'type_constrained', (int, _gteq1),
        '''
            (default=5) Number of draws to use when estimating trace approximation for sigma^2.
        ''', '>= 1'),
    'ndraw_trace_ho': (5, 'type_constrained', (int, _gteq1),
        '''
            (default=5) Number of draws to use when estimating trace approximation for homoskedastic correction.
        ''', '>= 1'),
    'ndraw_trace_he': (5, 'type_constrained', (int, _gteq1),
        '''
            (default=5) Number of draws to use when estimating trace approximation for heteroskedastic correction.
        ''', '>= 1'),
    'exact_trace_sigma_2': (False, 'type', bool,
        '''
            (default=False) If True, estimate trace analytically for sigma^2; if False, use a trace approximation.
        ''', None),
    'exact_trace_ho': (False, 'type', bool,
        '''
            (default=False) If True, estimate trace analytically for homoskedastic correction; if False, use a trace approximation.
        ''', None),
    'exact_trace_he': (False, 'type', bool,
        '''
            (default=False) If True, estimate trace analytically for heteroskedastic correction; if False, use a trace approximation.
        ''', None),
    'exact_lev_he': (False, 'type', bool,
        '''
            (default=False) If True, estimate leverages analytically for heteroskedastic correction; if False, use the JL approximation.
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
    #         (default=False) Computes the smallest eigenvalues, this is the filepath where these results are saved.
    #     ''', None),
    'outputfile': (None, 'type_none', str,
        '''
            (default=None) Outputfile where results will be saved in json format. If None, results will not be saved.
        ''', None),
    'progress_bars': (True, 'type', bool,
        '''
            (default=True) If True, display progress bars.
        ''', None),
    'verbose': (True, 'type', bool,
        '''
            (default=True) If True, print warnings during HE estimation.
        ''', None)
})

def fe_params(update_dict=None):
    '''
    Dictionary of default fe_params. Run tw.fe_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict or None): user parameter values; None is equivalent to {}

    Returns:
        (ParamsDict) dictionary of fe_params
    '''
    new_dict = _fe_params_default.copy()
    if update_dict is not None:
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

def _weighted_var(v, w, dof=0):
    '''
    Compute weighted variance.

    Arguments:
        v (NumPy Array): vector to weight
        w (NumPy Array): weights
        dof (int): degrees of freedom

    Returns:
        (NumPy Array): weighted variance
    '''
    m0 = np.sum(w * v) / np.sum(w)
    v0 = np.sum(w * (v - m0) ** 2) / (np.sum(w) - dof)

    return v0

def _weighted_cov(v1, v2, w1, w2, dof=0):
    '''
    Compute weighted covariance.

    Arguments:
        v1 (NumPy Array): vector to weight
        v2 (NumPy Array): vector to weight
        w1 (NumPy Array): weights for v1
        w2 (NumPy Array): weights for v2
        dof (int): degrees of freedom

    Returns:
        (NumPy Array): weighted covariance
    '''
    m1 = np.sum(w1 * v1) / np.sum(w1)
    m2 = np.sum(w2 * v2) / np.sum(w2)
    w3 = np.sqrt(w1 * w2)
    v0 = np.sum(w3 * (v1 - m1) * (v2 - m2)) / (np.sum(w3) - dof)

    return v0

class FEEstimator:
    '''
    Uses multigrid and partialing out to solve two way fixed effect models. This includes AKM, the Andrews et al. homoskedastic correction, and the Kline et al. heteroskedastic correction.
    '''

    def __init__(self, data, params=None):
        '''
        Arguments:
            data (BipartiteDataFrame): long or collapsed long format labor data
            fe_params (ParamsDict or None)): dictionary of parameters for FE estimation. Run tw.fe_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.fe_params().
        '''
        # Start logger
        logger_init(self)
        # self.logger.info('initializing FEEstimator object')

        if params is None:
            params = fe_params()

        self.adata = data

        self.params = params
        # Results dictionary
        self.res = {}
        # Summary results dictionary
        self.summary = {}

        ## Save some commonly used parameters as attributes ##
        # Number of cores to use
        self.ncore = params['ncore']
        # Number of draws to compute leverage for heteroskedastic correction
        self.lev_batchsize = params['lev_batchsize']
        # Number of draws to use in sigma^2 trace approximation
        self.ndraw_trace_sigma_2 = params['ndraw_trace_sigma_2']
        # Number of draws to use in homoskedastic trace approximation
        self.ndraw_trace_ho = params['ndraw_trace_ho']
        # Number of draws to use in heteroskedastic trace approximation
        self.ndraw_trace_he = params['ndraw_trace_he']
        # Whether to compute heteroskedastic correction
        self.compute_he = params['he']
        # Whether data is weighted
        self.weighted = (params['weighted'] and ('w' in data.columns))
        # Progress bars
        self.no_pbars = not params['progress_bars']
        # Verbose
        self.verbose = params['verbose']

        ## Store some parameters in results dictionary ##
        self.res['cores'] = self.ncore
        self.res['ndp'] = self.lev_batchsize
        self.res['ndraw_trace_sigma_2'] = self.ndraw_trace_sigma_2
        self.res['ndraw_trace_ho'] = self.ndraw_trace_ho
        self.res['ndraw_trace_he'] = self.ndraw_trace_he

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
        self.ml = pyamg.ruge_stuben_solver(self.Minv)

    @staticmethod
    def __load(filename):
        '''
        Load files for heteroskedastic correction.

        Arguments:
            filename (string): file to load

        Returns:
            (file): loaded file
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

    def fit(self, rng=None):
        '''
        Estimate FE model.

        Arguments:
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        self.logger.info('----- STARTING FE ESTIMATION -----')
        start_time = time.time()

        ## Begin cleaning and analysis #
        # Store commonly use variables
        self._prep_vars()
        # Generate commonly used matrices
        self._prep_matrices()
        # Compute basic statistics
        self._compute_early_stats()

        if self.params['statsonly']:
            ## Compute basic statistics ##
            self.logger.info('statsonly=True, so we skip the full estimation')

        else:
            ## Estimate full model ##
            # Estimate fixed effects using OLS
            self._estimate_ols()
            if self.params['attach_fe_estimates']:
                # Attach fixed effect columns
                self._attach_fe_estimates()

            if not self.params['feonly']:
                ## Full estimation ##
                # Estimate trace for sigma^2 (variance of residuals)
                if self.params['exact_trace_sigma_2']:
                    # Analytical trace
                    self._construct_AAinv_components_full()
                    self._estimate_exact_trace_sigma_2()
                else:
                    # Approximate trace
                    self._estimate_approximate_trace_sigma_2(rng)
                # Estimate sigma^2 (variance of residuals)
                self._estimate_sigma_2()
                # Construct Q matrix
                Q_params = self._construct_Q()
                # Estimate plug-in (biased) FE model
                self._estimate_fe(Q_params)

                ## HO/HE corrections ##
                # Estimate trace for HO correction
                if self.params['exact_trace_ho']:
                    # Analytical trace
                    if not self.params['exact_trace_sigma_2']:
                        self._construct_AAinv_components_full()
                    self._estimate_exact_trace_ho(Q_params)
                else:
                    # Approximate trace
                    self._estimate_approximate_trace_ho(Q_params, rng)

                if self.compute_he:
                    ## HE correction ##
                    if (self.params['exact_lev_he'] or self.params['exact_trace_he']) and not (self.params['exact_trace_sigma_2'] or self.params['exact_trace_ho']):
                        self._construct_AAinv_components_full()
                    # Estimate leverages for HE correction
                    if len(self.params['levfile']) > 0:
                        # Precomputed leverages
                        Pii, p = self._extract_precomputed_leverages()
                    elif self.params['exact_lev_he']:
                        # Analytical leverages
                        Pii, p = self._estimate_exact_leverages()
                    else:
                        # Approximate leverages
                        Pii, p = self._estimate_approximate_leverages(rng)
                    # Estimate Sii (sigma^2) for HE correction
                    Sii = self._estimate_Sii_he(Pii, p)
                    del Pii
                    # Estimate trace for HE correction
                    if self.params['exact_trace_he']:
                        # Analytical trace
                        self._estimate_exact_trace_he(Q_params, Sii)
                    else:
                        # Approximate trace
                        self._estimate_approximate_trace_he(Q_params, Sii, rng)
                    del Sii

                # Collect all results
                self._collect_res()

        # Clear attributes
        del self.Y, self.J, self.W, self.Dp, self.Dp_sqrt, self.Dwinv, self.Minv, self.ml,

        # Drop 'worker_m' column
        self.adata.drop('worker_m', axis=1, inplace=True)

        # Total estimation time
        end_time = time.time()
        self.res['total_time'] = end_time - start_time

        # Save results to json
        self._save_res()

        self.logger.info('----- DONE WITH FE ESTIMATION -----')

    def _prep_vars(self):
        '''
        Generate some initial class attributes and results.
        '''
        self.logger.info('preparing data')

        # Number of firms
        self.nf = self.adata.n_firms()
        # Number of workers
        self.nw = self.adata.n_workers()
        # Number of observations
        self.nn = len(self.adata)
        self.logger.info(f'data firms={self.nf} workers={self.nw} observations={self.nn}')

        self.res['n_firms'] = self.nf
        self.res['n_workers'] = self.nw
        self.res['n_movers'] = self.adata.loc[self.adata['m'].to_numpy() > 0, :].n_workers()
        self.res['n_stayers'] = self.res['n_workers'] - self.res['n_movers']
        self.logger.info(f"data movers={self.res['n_movers']} stayers={self.res['n_stayers']}")

        # Generate 'worker_m' indicating whether a worker is a mover or a stayer
        self.adata.loc[:, 'worker_m'] = self.adata.get_worker_m(is_sorted=True).astype(int, copy=False)

        # # Prepare 'cs' column (0 if observation is first for a worker, 1 if intermediate, 2 if last for a worker)
        # worker_first_obs = (self.adata['i'].to_numpy() != np.roll(self.adata['i'].to_numpy(), 1))
        # worker_last_obs = (self.adata['i'].to_numpy() != np.roll(self.adata['i'].to_numpy(), -1))
        # self.adata['cs'] = 1
        # self.adata.loc[(worker_first_obs) & ~(worker_last_obs), 'cs'] = 0
        # self.adata.loc[(worker_last_obs) & ~(worker_first_obs), 'cs'] = 2

        #res['year_max'] = int(sdata['year'].max())
        #res['year_min'] = int(sdata['year'].min())

    def _prep_matrices(self):
        '''
        Generate J, W, Dp, Dwinv, and M matrices. Convert Y to NumPy vector.
        '''
        ## J (firms) ##
        J = csc_matrix((np.ones(self.nn), (self.adata.index.to_numpy(), self.adata.loc[:, 'j'].to_numpy())), shape=(self.nn, self.nf))

        # Normalize one firm to 0
        J = J[:, range(self.nf - 1)]

        ## W (workers) ##
        W = csc_matrix((np.ones(self.nn), (self.adata.index.to_numpy(), self.adata.loc[:, 'i'].to_numpy())), shape=(self.nn, self.nw))

        ## Dp (weight) ##
        if self.weighted:
            # Diagonal weight matrix
            Dp = diags(self.adata.loc[:, 'w'].to_numpy())
        else:
            # Diagonal weight matrix - all weight one
            Dp = diags(np.ones(self.nn).astype(int, copy=False))

        ## Dwinv ##
        Dwinv = diags(1 / ((W.T @ Dp @ W).diagonal()))

        ## Minv ##
        Minv = J.T @ Dp @ J - J.T @ Dp @ W @ Dwinv @ W.T @ Dp @ J

        ## Store matrices ##
        self.Y = self.adata.loc[:, 'y'].to_numpy()
        self.J = J
        self.W = W
        self.Dp = Dp
        self.Dp_sqrt = np.sqrt(Dp)
        self.Dwinv = Dwinv
        self.Minv = Minv

        self.logger.info('preparing linear solver')
        self.ml = pyamg.ruge_stuben_solver(Minv)

        # Save time variable
        self.last_invert_time = 0

    def _compute_early_stats(self):
        '''
        Compute some early statistics.
        '''
        if self.weighted:
            self.adata.loc[:, 'weighted_m'] = self.Dp * self.adata.loc[:, 'worker_m'].to_numpy()
            self.adata.loc[:, 'weighted_y'] = self.Dp * self.Y
            fdata = self.adata.groupby('j')[['weighted_m', 'weighted_y', 'w']].sum()
            fm, fy, fi = fdata.loc[:, 'weighted_m'].to_numpy(), fdata.loc[:, 'weighted_y'].to_numpy(), fdata.loc[:, 'w'].to_numpy()
            fy /= fi
            self.adata.drop(['weighted_m', 'weighted_y'], axis=1, inplace=True)
        else:
            fdata = self.adata.groupby('j').agg({'worker_m': 'sum', 'y': 'mean', 'i': 'count'})
            fm, fy, fi = fdata.loc[:, 'worker_m'].to_numpy(), fdata.loc[:, 'y'].to_numpy(), fdata.loc[:, 'i'].to_numpy()
        ls = np.linspace(0, 1, 11)
        self.res['mover_quantiles'] = _weighted_quantile(fm, ls, fi).tolist()
        self.res['size_quantiles'] = _weighted_quantile(fi, ls, fi).tolist()
        # self.res['movers_per_firm'] = self.adata.loc[self.adata.loc[:, 'm'] > 0, :].groupby('j')['i'].nunique().mean()
        self.res['between_firm_var'] = _weighted_var(fy, fi)
        self.res['var_y'] = _weighted_var(self.Y, self.Dp)
        self.logger.info(f"total variance: {self.res['var_y']:2.4f}")

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

    def _estimate_ols(self):
        '''
        Estimate fixed effects using OLS.
        '''
        ## Estimate psi and alpha ##
        self.logger.info('estimating firm and worker effects')

        self.psi_hat, self.alpha_hat = self._solve(self.Y)

        self.res['solver_time'] = self.last_invert_time
        self.logger.info(f'solver time {self.last_invert_time:2.4f} seconds')
        n_draws_sigma_2 = self.ndraw_trace_sigma_2
        n_draws_ho = (not self.params['feonly']) * self.ndraw_trace_ho
        n_draws_he = (not self.params['feonly']) * self.compute_he * (self.ndraw_trace_he + self.params['lev_nbatches'] * self.lev_batchsize)
        expected_time = (self.last_invert_time / 60) * (n_draws_sigma_2 + n_draws_ho + n_draws_he)
        self.logger.info(f'expected total time {expected_time:2.4f} minutes')

    def _attach_fe_estimates(self):
        '''
        Attach the estimated psi_hat and alpha_hat as columns to the input dataframe.
        '''
        j_vals = np.arange(self.nf)
        i_vals = np.arange(self.nw)

        # Add 0 for normalized firm
        psi_hat_dict = dict(zip(j_vals, np.concatenate([self.psi_hat, np.array([0])])))
        alpha_hat_dict = dict(zip(i_vals, self.alpha_hat))

        # Attach columns
        self.adata.loc[:, 'psi_hat'] = self.adata.loc[:, 'j'].map(psi_hat_dict)
        self.adata.loc[:, 'alpha_hat'] = self.adata.loc[:, 'i'].map(alpha_hat_dict)

    def _estimate_exact_trace_sigma_2(self):
        '''
        Estimate analytical trace of sigma^2.
        '''
        self.logger.info(f'[sigma^2] [analytical trace]')

        # Unpack attributes
        J, W = self.J, self.W
        A, B, C, D = self.AA_inv_A, self.AA_inv_B, self.AA_inv_C, self.AA_inv_D

        ## Compute Tr[A @ (A'D_pA)^{-1} @ A'] ##
        self.tr_sigma_ho_all = np.trace((J @ A + W @ C) @ J.T + (J @ B + W @ D) @ W.T)

        self.logger.debug(f'[sigma^2] [analytical traces] done')

    def _estimate_approximate_trace_sigma_2(self, rng=None):
        '''
        Estimate trace approximation of sigma^2.

        Arguments:
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        self.logger.info(f'[sigma^2] [approximate trace] ndraws={self.ndraw_trace_sigma_2}')

        if rng is None:
            rng = np.random.default_rng(None)

        # Begin trace approximation
        self.tr_sigma_ho_all = np.zeros(self.ndraw_trace_sigma_2)

        pbar = trange(self.ndraw_trace_sigma_2, disable=self.no_pbars)
        pbar.set_description('sigma^2')
        for r in pbar:
            ## Compute Tr[A @ (A'D_pA)^{-1} @ A'] ##
            # Generate -1 or 1
            Z = 2 * rng.binomial(1, 0.5, self.nn) - 1

            # Compute Z.T @ A @ (A'D_pA)^{-1} @ A' @ Z
            self.tr_sigma_ho_all[r] = Z.T @ self._mult_A( # Z.T @ A @
                *self._solve(Z, Dp1=True, Dp2=False), weighted=False # (A'D_pA)^{-1} @ A' @ Z
            )

            self.logger.debug(f'[sigma^2] [approximate trace] step {r}/{self.ndraw_trace_sigma_2} done')

    def _estimate_sigma_2(self):
        '''
        Estimate residuals and sigma^2 (variance of residuals).
        '''
        ## Estimate residuals ##
        self.E = self.Y - self._mult_A(self.psi_hat, self.alpha_hat)

        fe_rsq = 1 - (self.Dp * (self.E ** 2)).sum() / (self.Dp * (self.Y ** 2)).sum()
        self.logger.info(f'fixed effect R-square {fe_rsq:2.4f}')

        ## Estimate variance of residuals ##
        # Plug-in
        self.sigma_2_pi = np.var(self.E, ddof=0) # _weighted_var(self.E, self.Dp)
        # Bias-corrected
        if self.weighted:
            trace_approximation = np.mean(self.tr_sigma_ho_all)
            self.sigma_2_ho = (self.nn * self.sigma_2_pi) / (np.sum(1 / self.Dp.data[0]) - trace_approximation)
        else:
            self.sigma_2_ho = (self.nn * self.sigma_2_pi) / (self.nn - (self.nw + self.nf - 1))
        self.logger.info(f'[ho] variance of residuals {self.sigma_2_ho:2.4f}')

    def _estimate_fe(self, Q_params):
        '''
        Estimate plug-in (biased) FE model.

        Arguments:
            Q_params (tuple): (Q variance parameters, Q left covariance parameters, Q right covariance parameters)
        '''
        self.logger.info('starting plug-in estimation')

        Q_var, Ql_cov, Qr_cov = Q_params
        Q_var_matrix, Q_var_psialpha, Q_var_weights, Q_var_dof = Q_var
        Ql_cov_matrix, Ql_cov_psialpha, Ql_cov_weights, Q_cov_dof = Ql_cov
        Qr_cov_matrix, Qr_cov_psialpha, Qr_cov_weights = Qr_cov

        psialpha_dict = {
            'psi': self.psi_hat,
            'alpha': self.alpha_hat
        }

        self.var_fe = _weighted_var(Q_var_matrix @ psialpha_dict[Q_var_psialpha], Q_var_weights, dof=0)
        self.cov_fe = _weighted_cov(Ql_cov_matrix @ psialpha_dict[Ql_cov_psialpha], Qr_cov_matrix @ psialpha_dict[Qr_cov_psialpha], Ql_cov_weights, Qr_cov_weights, dof=0)

        self.logger.info('[fe]')
        self.logger.info(f'var_psi={self.var_fe:2.4f}')
        self.logger.info(f"cov={self.cov_fe:2.4f} tot={self.res['var_y']:2.4f}")

    def _estimate_exact_trace_ho(self, Q_params):
        '''
        Estimate analytical trace of HO-corrected model.

        Arguments:
            Q_params (tuple): (Q variance parameters, Q left covariance parameters, Q right covariance parameters)
        '''
        self.logger.info(f'[ho] [analytical trace]')

        Q_var, Ql_cov, Qr_cov = Q_params
        Q_var_matrix, Q_var_psialpha, Q_var_weights, Q_var_dof = Q_var
        Ql_cov_matrix, Ql_cov_psialpha, Ql_cov_weights, Q_cov_dof = Ql_cov
        Qr_cov_matrix, Qr_cov_psialpha, Qr_cov_weights = Qr_cov
        Q_var_weights_sq = np.sqrt(Q_var_weights)
        Ql_cov_weights_sq = np.sqrt(Ql_cov_weights)
        Qr_cov_weights_sq = np.sqrt(Qr_cov_weights)
        Q_cov_weights = Ql_cov_weights_sq * Qr_cov_weights_sq

        ## Compute Tr[Q @ (A'D_pA)^{-1}] ##
        right_dict = {
            'psi': (self.AA_inv_A, self.AA_inv_B),
            'alpha': (self.AA_inv_C, self.AA_inv_D)
        }
        left_dict = {
            'psi': 0,
            'alpha': 1
        }

        ## Trace - variance ##
        # Q = Ql.T @ Ql_weight.T @ U @ Qr_weight @ Qr
        n = np.sum(Q_var_weights)
        # Construct U (demeaning matrix)
        U_dim = Q_var_matrix.shape[0]
        U = eye(U_dim) - (1 / n) * np.tile(Q_var_weights.diagonal(), (U_dim, 1)).T
        # Compute Q @ (A'D_pA)^{-1} using block matrix components
        Q_var = (1 / n) * Q_var_matrix.T @ Q_var_weights_sq.T @ U @ Q_var_weights_sq @ Q_var_matrix
        # Compute trace by considering correct quadrant
        self.tr_var_ho_all = np.trace(Q_var @ right_dict[Q_var_psialpha][left_dict[Q_var_psialpha]])

        ## Trace - covariance ##
        # Q = Ql.T @ Ql_weight.T @ U @ Qr_weight @ Qr
        n = np.sum(Q_cov_weights)
        # Construct U (demeaning matrix)
        U_dim = Ql_cov_matrix.shape[0]
        U = eye(U_dim) - (1 / n) * np.tile(Q_cov_weights.diagonal(), (U_dim, 1)).T
        # Compute Q @ (A'D_pA)^{-1} using block matrix components
        Q_cov = (1 / n) * Ql_cov_matrix.T @ Ql_cov_weights_sq.T @ U @ Qr_cov_weights_sq @ Qr_cov_matrix
        # Compute trace by considering correct quadrant
        self.tr_cov_ho_all = np.trace(Q_cov @ right_dict[Qr_cov_psialpha][left_dict[Ql_cov_psialpha]])

    def _estimate_approximate_trace_ho(self, Q_params, rng=None):
        '''
        Estimate trace approximation of HO-corrected model.

        Arguments:
            Q_params (tuple): (Q variance parameters, Q left covariance parameters, Q right covariance parameters)
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        self.logger.info(f'[ho] [approximate trace] ndraws={self.ndraw_trace_ho}')

        Q_var, Ql_cov, Qr_cov = Q_params
        Q_var_matrix, Q_var_psialpha, Q_var_weights, Q_var_dof = Q_var
        Ql_cov_matrix, Ql_cov_psialpha, Ql_cov_weights, Q_cov_dof = Ql_cov
        Qr_cov_matrix, Qr_cov_psialpha, Qr_cov_weights = Qr_cov

        self.tr_var_ho_all = np.zeros(self.ndraw_trace_ho)
        self.tr_cov_ho_all = np.zeros(self.ndraw_trace_ho)

        pbar = trange(self.ndraw_trace_ho, disable=self.no_pbars)
        pbar.set_description('ho')
        for r in pbar:
            ## Compute Tr[Q @ (A'D_pA)^{-1}] ##
            # Generate -1 or 1
            Zpsi = 2 * rng.binomial(1, 0.5, self.nf - 1) - 1
            Zalpha = 2 * rng.binomial(1, 0.5, self.nw) - 1
            Z_dict = {
                'psi': Zpsi,
                'alpha': Zalpha
            }

            # Compute (A'D_pA)^{-1} @ Z
            psi1, alpha1 = self._mult_AAinv(Zpsi, Zalpha)
            AAinv_dict = {
                'psi': psi1,
                'alpha': alpha1
            }
            del Zpsi, Zalpha, psi1, alpha1

            ## Trace correction - variance ##
            # Left term of Q matrix
            L_var = Z_dict[Q_var_psialpha] @ Q_var_matrix.T
            # Right term of Q matrix
            R_var = Q_var_matrix @ AAinv_dict[Q_var_psialpha]
            self.tr_var_ho_all[r] = _weighted_cov(L_var, R_var, Q_var_weights, Q_var_weights)
            del L_var, R_var
            ## Trace correction - covariance ##
            # Left term of Q matrix
            L_cov = Z_dict[Ql_cov_psialpha] @ Ql_cov_matrix.T
            # Right term of Q matrix
            R_cov = Qr_cov_matrix @ AAinv_dict[Qr_cov_psialpha]
            self.tr_cov_ho_all[r] = _weighted_cov(L_cov, R_cov, Ql_cov_weights, Qr_cov_weights)
            del L_cov, R_cov, Z_dict, AAinv_dict

            self.logger.debug(f'[ho] [approximate trace] step {r + 1}/{self.ndraw_trace_ho} done')

    def _estimate_exact_trace_he(self, Q_params, Sii):
        '''
        Estimate analytical trace of HE-corrected model.

        Arguments:
            Q_params (tuple): (Q variance parameters, Q left covariance parameters, Q right covariance parameters)
            Sii (NumPy Array): Sii (sigma^2) for heteroskedastic correction
        '''
        self.logger.info(f'[he] [analytical trace]')

        Q_var, Ql_cov, Qr_cov = Q_params
        Q_var_matrix, Q_var_psialpha, Q_var_weights, Q_var_dof = Q_var
        Ql_cov_matrix, Ql_cov_psialpha, Ql_cov_weights, Q_cov_dof = Ql_cov
        Qr_cov_matrix, Qr_cov_psialpha, Qr_cov_weights = Qr_cov
        Q_var_weights_sq = np.sqrt(Q_var_weights)
        Ql_cov_weights_sq = np.sqrt(Ql_cov_weights)
        Qr_cov_weights_sq = np.sqrt(Qr_cov_weights)
        Q_cov_weights = Ql_cov_weights_sq * Qr_cov_weights_sq

        # Unpack attributes
        J, W = self.J, self.W
        Dp = self.Dp
        A, B, C, D = self.AA_inv_A, self.AA_inv_B, self.AA_inv_C, self.AA_inv_D
        Sii = diags(Sii)

        ## Compute Tr[Q @ (A'D_pA)^{-1} @ (D_pA)' @ Omega @ (D_pA) @ (A'D_pA)^{-1}] ##
        ## Left ##
        # [A & B \\ C & D] @ [J' \\ W'] @ D_p
        # = [A & B \\ C & D] @ [J' @ D_p \\ W' @ D_p]
        # = [A @ J' @ D_p + B @ W' @ D_p \\ C @ J' @ D_p + D @ W' @ D_p]
        ## Right ##
        # D_p @ [J & W] @ [A & B \\ C & D]
        # = [D_p @ J & D_p @ W] @ [A & B \\ C & D]
        # = [D_p @ J @ A + D_p @ W @ C & D_p @ J @ B + D_p @ W @ D]
        ## Total ##
        # [A @ J' @ D_p + B @ W' @ D_p \\ C @ J' @ D_p + D @ W' @ D_p] @ Omega @ [D_p @ J @ A + D_p @ W @ C & D_p @ J @ B + D_p @ W @ D]
        # = [(A @ J' + B @ W') @ D_p @ Omega \\ (C @ J' + D @ W') @ D_p @ Omega] @ [D_p @ (J @ A + W @ C) & D_p @ (J @ B + W @ D)]
        # = [(A @ J' + B @ W') @ D_p @ Omega @ D_p @ (J @ A + W @ C) & (A @ J' + B @ W') @ D_p @ Omega @ D_p @ (J @ B + W @ D) \\ (C @ J' + D @ W') @ D_p @ Omega @ D_p @ (J @ A + W @ C) & (C @ J' + D @ W') @ D_p @ Omega @ D_p @ (J @ B + W @ D)]

        ## Trace - variance ##
        # Q = Ql.T @ Ql_weight.T @ U @ Qr_weight @ Qr
        n = np.sum(Q_var_weights)
        # Construct U (demeaning matrix)
        U_dim = Q_var_matrix.shape[0]
        U = eye(U_dim) - (1 / n) * np.tile(Q_var_weights.diagonal(), (U_dim, 1)).T
        # Compute Q @ (A'D_pA)^{-1} @ (D_pA)' @ Omega @ (D_pA) @ (A'D_pA)^{-1} using block matrix components
        if Q_var_psialpha == 'psi':
            AA_inv = Dp @ (J @ A + W @ C)
            AA_inv = AA_inv.T @ Sii @ AA_inv
        elif Q_var_psialpha == 'alpha':
            AA_inv = Dp @ (J @ B + W @ D)
            AA_inv = AA_inv.T @ Sii @ AA_inv
        Q_var = (1 / n) * Q_var_matrix.T @ Q_var_weights_sq.T @ U @ Q_var_weights_sq @ Q_var_matrix
        # Compute trace by considering correct quadrant
        self.tr_var_he_all = np.trace(Q_var @ AA_inv)

        ## Trace - covariance ##
        # Q = Ql.T @ Ql_weight.T @ U @ Qr_weight @ Qr
        n = np.sum(Q_cov_weights)
        # Construct U (demeaning matrix)
        U_dim = Ql_cov_matrix.shape[0]
        U = eye(U_dim) - (1 / n) * np.tile(Q_cov_weights.diagonal(), (U_dim, 1)).T
        # Compute Q @ (A'D_pA)^{-1} @ (D_pA)' @ Omega @ (D_pA) @ (A'D_pA)^{-1} using block matrix components
        if Qr_cov_psialpha == 'psi':
            if Ql_cov_psialpha == 'psi':
                if Q_var_psialpha != 'psi':
                    # No need to recompute if var is the same
                    AA_inv = Dp @ (J @ A + W @ C)
                    AA_inv = AA_inv.T @ Sii @ AA_inv
            elif Ql_cov_psialpha == 'alpha':
                AA_inv = (A @ J.T + B @ W.T) @ Dp @ Sii @ Dp @ (J @ B + W @ D)
        elif Qr_cov_psialpha == 'alpha':
            if Ql_cov_psialpha == 'psi':
                AA_inv = (C @ J.T + D @ W.T) @ Dp @ Sii @ Dp @ (J @ A + W @ C)
            elif Ql_cov_psialpha == 'alpha':
                if Q_var_psialpha != 'alpha':
                    # No need to recompute if var is the same
                    AA_inv = Dp @ (J @ B + W @ D)
                    AA_inv = AA_inv.T @ Sii @ AA_inv
        Q_cov = (1 / n) * Ql_cov_matrix.T @ Ql_cov_weights_sq.T @ U @ Qr_cov_weights_sq @ Qr_cov_matrix
        # Compute trace by considering correct quadrant
        self.tr_cov_he_all = np.trace(Q_cov @ AA_inv)

    def _estimate_approximate_trace_he(self, Q_params, Sii, rng=None):
        '''
        Estimate trace approximation of HE-corrected model.

        Arguments:
            Q_params (tuple): (Q variance parameters, Q left covariance parameters, Q right covariance parameters)
            Sii (NumPy Array): Sii (sigma^2) for heteroskedastic correction
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        self.logger.info(f'[he] [approximate trace] ndraws={self.ndraw_trace_he}')

        Q_var, Ql_cov, Qr_cov = Q_params
        Q_var_matrix, Q_var_psialpha, Q_var_weights, Q_var_dof = Q_var
        Ql_cov_matrix, Ql_cov_psialpha, Ql_cov_weights, Q_cov_dof = Ql_cov
        Qr_cov_matrix, Qr_cov_psialpha, Qr_cov_weights = Qr_cov

        self.tr_var_he_all = np.zeros(self.ndraw_trace_he)
        self.tr_cov_he_all = np.zeros(self.ndraw_trace_he)

        pbar = trange(self.ndraw_trace_he, disable=self.no_pbars)
        pbar.set_description('he')
        for r in pbar:
            ## Compute Tr[Q @ (A'D_pA)^{-1} @ (D_pA)' @ Omega @ (D_pA) @ (A'D_pA)^{-1}] ##
            # Generate -1 or 1
            Zpsi = 2 * rng.binomial(1, 0.5, self.nf - 1) - 1
            Zalpha = 2 * rng.binomial(1, 0.5, self.nw) - 1
            Z_dict = {
                'psi': Zpsi,
                'alpha': Zalpha
            }

            # Compute (A'D_pA)^{-1} @ (D_pA)' @ Omega @ (D_pA) @ (A'D_pA)^{-1} @ Z
            psi1, alpha1 = self._mult_AAinv( # (A'D_pA)^{-1} @
                *self._mult_Atranspose( # (D_pA)' @
                    Sii * self._mult_A( # Omega @ (D_pA) @
                        *self._mult_AAinv( # (A'D_pA)^{-1} @ Z
                            Zpsi, Zalpha, weighted=True
                        ), weighted=True
                    ), weighted=True
                ), weighted=True
            )
            AAinv_dict = {
                'psi': psi1,
                'alpha': alpha1
            }
            del Zpsi, Zalpha, psi1, alpha1

            ## Trace correction - variance ##
            # Left term of Q matrix
            L_var = Z_dict[Q_var_psialpha] @ Q_var_matrix.T
            # Right term of Q matrix
            R_var = Q_var_matrix @ AAinv_dict[Q_var_psialpha]
            self.tr_var_he_all[r] = _weighted_cov(L_var, R_var, Q_var_weights, Q_var_weights)
            del L_var, R_var
            ## Trace correction - covariance ##
            # Left term of Q matrix
            L_cov = Z_dict[Ql_cov_psialpha] @ Ql_cov_matrix.T
            # Right term of Q matrix
            R_cov = Qr_cov_matrix @ AAinv_dict[Qr_cov_psialpha]
            self.tr_cov_he_all[r] = _weighted_cov(L_cov, R_cov, Ql_cov_weights, Qr_cov_weights)
            del L_cov, R_cov, Z_dict, AAinv_dict

            self.logger.debug(f'[he] [approximate trace] step {r + 1}/{self.ndraw_trace_he} done')

    def _collect_res(self):
        '''
        Collect all results.
        '''
        # Already computed, this just reorders the dictionary
        self.res['var_y'] = self.res['var_y']

        ## FE results ##
        # Plug-in sigma^2
        self.res['eps_var_fe'] = self.sigma_2_pi
        # Plug-in variance
        self.res['var_fe'] = self.var_fe
        self.logger.info(f'[ho] VAR fe={self.var_fe:2.4f}')
        # Plug-in covariance
        self.logger.info(f'[ho] COV fe={self.cov_fe:2.4f}')
        self.res['cov_fe'] = self.cov_fe

        ## Homoskedastic results ##
        # Bias-corrected sigma^2
        self.res['eps_var_ho'] = self.sigma_2_ho
        # Trace approximation: variance
        self.res['tr_var_ho'] = np.mean(self.tr_var_ho_all)
        self.res['tr_var_ho_sd'] = np.std(self.tr_var_ho_all)
        self.logger.info(f"[ho] VAR tr={self.res['tr_var_ho']:2.4f} (sd={self.res['tr_var_ho_sd']:2.4e})")
        # Trace approximation: covariance
        self.res['tr_cov_ho'] = np.mean(self.tr_cov_ho_all)
        self.res['tr_cov_ho_sd'] = np.std(self.tr_cov_ho_all)
        self.logger.info(f"[ho] COV tr={self.res['tr_cov_ho']:2.4f} (sd={self.res['tr_cov_ho_sd']:2.4e})")
        # Bias-corrected variance
        self.res['var_ho'] = self.var_fe - self.sigma_2_ho * self.res['tr_var_ho']
        self.logger.info(f"[ho] VAR bc={self.res['var_ho']:2.4f}")
        # Bias-corrected covariance
        self.res['cov_ho'] = self.cov_fe - self.sigma_2_ho * self.res['tr_cov_ho']
        self.logger.info(f"[ho] COV bc={self.res['cov_ho']:2.4f}")

        for res in ['var_y', 'var_fe', 'cov_fe', 'var_ho', 'cov_ho']:
            self.summary[res] = self.res[res]

        ## Heteroskedastic results ##
        if self.compute_he:
            ## Already computed, this just reorders the dictionary ##
            # Bias-corrected sigma^2
            self.res['eps_var_he'] = self.res['eps_var_he']
            self.res['min_lev'] = self.res['min_lev']
            self.res['max_lev'] = self.res['max_lev']
            ## New results ##
            # Trace approximation: variance
            self.res['tr_var_he'] = np.mean(self.tr_var_he_all)
            self.res['tr_var_he_sd'] = np.std(self.tr_var_he_all)
            self.logger.info(f"[he] VAR tr={self.res['tr_var_he']:2.4f} (sd={self.res['tr_var_he_sd']:2.4e})")
            # Trace approximation: covariance
            self.res['tr_cov_he'] = np.mean(self.tr_cov_he_all)
            self.res['tr_cov_he_sd'] = np.std(self.tr_cov_he_all)
            self.logger.info(f"[he] COV tr={self.res['tr_cov_he']:2.4f} (sd={self.res['tr_cov_he_sd']:2.4e})")
            # Bias-corrected variance
            self.res['var_he'] = self.var_fe - self.res['tr_var_he']
            self.logger.info(f"[he] VAR fe={self.var_fe:2.4f} bc={self.res['var_he']:2.4f}")
            # Bias-corrected covariance
            self.res['cov_he'] = self.cov_fe - self.res['tr_cov_he']
            self.logger.info(f"[he] COV fe={self.cov_fe:2.4f} bc={self.res['cov_he']:2.4f}")

            for res in ['var_he', 'cov_he']:
                self.summary[res] = self.res[res]

    def _save_res(self):
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

    def _construct_Q(self):
        '''
        Construct Q matrix and generate related parameters.

        Returns:
            (tuple): (Q variance parameters, Q left covariance parameters, Q right covariance parameters)
        '''
        Q_var = self.params['Q_var']
        Q_cov = self.params['Q_cov']
        if Q_var is None:
            Q_var = Q.VarPsi()
        if Q_cov is None:
            Q_cov = Q.CovPsiAlpha()
        Q_params = self.adata, self.nf, self.nw, self.J, self.W, self.Dp
        return (Q_var._get_Q(*Q_params), Q_cov._get_Ql(*Q_params), Q_cov._get_Qr(*Q_params))

    def _construct_AAinv_components_full(self):
        '''
        Construct (A' * Dp * A)^{-1} block matrix components by explicitly computing M.
        '''
        # Define variables
        J = self.J
        W = self.W
        M = np.linalg.inv(self.Minv.todense())
        Dp = self.Dp
        Dwinv = self.Dwinv

        # Construct blocks
        self.AA_inv_A = M
        self.AA_inv_B = - M @ J.T @ Dp @ W @ Dwinv
        self.AA_inv_C = - Dwinv @ W.T @ Dp @ J @ M
        self.AA_inv_D = Dwinv + self.AA_inv_C @ self.Minv @ self.AA_inv_B

    def _construct_AAinv_components_partial(self):
        '''
        Construct (A' * Dp * A)^{-1} block matrix components without explicitly computing M. Use this for computing a small number of individual Pii analytically.
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

    def _solve(self, Y, Dp1=True, Dp2=True):
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
        J_transpose_Y, W_transpose_Y = self._mult_Atranspose(Y, weighted=Dp2)
        # This gives (A' * Dp1 * A)^{-1} * A' * Dp2 * Y
        psi_hat, alpha_hat = self._mult_AAinv(J_transpose_Y, W_transpose_Y, weighted=Dp1)

        return psi_hat, alpha_hat

    def _mult_A(self, psi, alpha, weighted=False):
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

    def _mult_Atranspose(self, v, weighted=True):
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

    def _mult_AAinv(self, psi, alpha, weighted=True):
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

    def _proj(self, Y, Dp0=False, Dp1=True, Dp2=True):
        '''
        Compute Dp0 * A * (A' * Dp1 * A)^{-1} * A' * Dp2 * Y, where A = [J W] (J is firm indicators and W is worker indicators) and Dp gives weights (essentially projects Y onto A space).
        Solve Y, then project onto X space of data stored in the object. Essentially solves A(A'A)^{-1}A'Y

        Arguments:
            Y (NumPy Array): wage data
            Dp0 (bool or str): if True, include weights in _mult_A(); if 'sqrt', use square root of weights
            Dp1 (bool): if True, include first weight in _solve()
            Dp2 (bool or str): if True, include second weight in _solve(); if 'sqrt', use square root of weights

        Returns:
            (CSC Matrix): result of Dp0 * A * (A' * Dp1 * A)^{-1} * A' * Dp2 * Y (essentially the projection of Y onto A space)
        '''
        return self._mult_A(*self._solve(Y, Dp1, Dp2), Dp0)

    def _compute_Pii(self, DpJ_i, DpW_i):
        '''
        Compute Pii for a single observation for heteroskedastic correction.

        Arguments:
            DpJ_i (NumPy Array): weighted J matrix
            DpW_i (NumPy Array): weighted W matrix

        Returns:
            (float): estimate for Pii
        '''
        # if self.params['exact_trace_sigma_2'] or self.params['exact_trace_ho'] or self.params['exact_trace_he'] or self.params['exact_lev_he']:
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

    def _extract_precomputed_leverages(self):
        '''
        Extract precomputed leverages (Pii) for heteroskedastic correction.

        Returns:
            (tuple of (NumPy Array, int)): (Pii --> estimated leverages, p --> number of draws used in leverage approximation)
        '''
        Pii = np.zeros(self.nn)
        worker_m = (self.adata.loc[:, 'worker_m'].to_numpy() > 0)

        self.logger.info('[he] starting heteroskedastic correction, loading precomputed files')

        files = glob.glob('{}*'.format(self.params['levfile']))
        self.logger.info(f'[he] found {len(files)} files to get leverages from')
        self.res['lev_file_count'] = len(files)
        assert len(files) > 0, "Didn't find any leverage files!"

        for f in files:
            pp = np.load(f)
            Pii += pp / len(files)

        self.res['min_lev'] = Pii[worker_m].min()
        self.res['max_lev'] = Pii[worker_m].max()

        if self.res['max_lev'] >= 1:
            leverage_warning = f"Max P_ii is {self.res['max_lev']} which is >= 1. This means your data is not leave-one-observation-out connected. The HE estimator requires leave-one-observation-out connected data to work properly. When cleaning your data, please set clean_params['connectedness'] = 'leave_out_observation' to correct this."
            self.logger.info(leverage_warning)
            raise ValueError(leverage_warning)

        self.logger.info(f"[he] leverage range {self.res['min_lev']:2.4f} to {self.res['max_lev']:2.4f}")

        return (Pii, len(files))

    def _estimate_exact_leverages(self):
        '''
        Estimate analytical leverages (Pii) for heteroskedastic correction.

        Returns:
            (tuple of (NumPy Array, None)): (Pii --> estimated leverages, p --> number of draws used in leverage approximation (None in this case))
        '''
        Pii = np.zeros(self.nn)
        worker_m = (self.adata.loc[:, 'worker_m'].to_numpy() > 0)

        self.logger.info('[he] [analytical Pii]')

        # Construct weighted J and W
        DpJ = np.asarray((self.Dp_sqrt @ self.J).todense())
        DpW = np.asarray((self.Dp_sqrt @ self.W).todense())

        pbar = tqdm(self.adata.loc[worker_m, :].index, disable=self.no_pbars)
        pbar.set_description('leverages')
        for i in pbar:
            DpJ_i = DpJ[i, :]
            DpW_i = DpW[i, :]
            # Compute analytical Pii
            Pii[i] = self._compute_Pii(DpJ_i, DpW_i)
        del pbar

        self.res['min_lev'] = Pii[worker_m].min()
        self.res['max_lev'] = Pii[worker_m].max()

        if self.res['max_lev'] >= 1:
            leverage_warning = f"Max P_ii is {self.res['max_lev']} which is >= 1. This means your data is not leave-one-observation-out connected. The HE estimator requires leave-one-observation-out connected data to work properly. When cleaning your data, please set clean_params['connectedness'] = 'leave_out_observation' to correct this."
            self.logger.info(leverage_warning)
            raise ValueError(leverage_warning)

        self.logger.info(f"[he] leverage range {self.res['min_lev']:2.4f} to {self.res['max_lev']:2.4f}")

        return (Pii, None)

    def _estimate_approximate_leverages(self, rng=None):
        '''
        Estimate approximate leverages (Pii) for heteroskedastic correction.

        Arguments:
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (tuple of (NumPy Array, int)): (Pii --> estimated leverages, p --> number of draws used in leverage approximation)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        Pii = np.zeros(self.nn)
        p = 0
        worker_m = (self.adata.loc[:, 'worker_m'].to_numpy() > 0)

        self.logger.info(f"[he] [approximate pii] lev_batchsize={self.params['lev_batchsize']}, lev_nbatches={self.params['lev_nbatches']}, using {self.ncore} core(s)")

        pbar = trange(self.params['lev_nbatches'], disable=self.no_pbars)
        pbar.set_description('leverages 1')
        for batch_i in pbar:
            if self.ncore > 1:
                # Multiprocessing
                ndraw_seeds = self.lev_batchsize // self.params['lev_batchsize_multiprocessing']
                if np.round(ndraw_seeds * self.params['lev_batchsize_multiprocessing']) != self.lev_batchsize:
                    # 'lev_batchsize_multiprocessing' must evenly divide 'lev_batchsize'
                    raise ValueError(f"'lev_batchsize_multiprocessing' (currently {self.params['lev_batchsize_multiprocessing']}) should evenly divide 'lev_batchsize' (currently {self.lev_batchsize}).")
                # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
                seeds = rng.bit_generator._seed_seq.spawn(ndraw_seeds)
                with Pool(processes=self.ncore) as pool:
                    pbar2 = tqdm([(self.params['lev_batchsize_multiprocessing'], np.random.default_rng(seed)) for seed in seeds], total=ndraw_seeds, disable=self.no_pbars)
                    pbar2.set_description('leverages 1.5')
                    Pii_all = pool.starmap(self._leverage_approx, pbar2)
                    del pbar2

                # Take mean over draws
                Pii_i = sum(Pii_all) / len(Pii_all)
            else:
                # Single core
                Pii_i = self._leverage_approx(self.lev_batchsize, rng)

            self.logger.debug(f"[he] [approximate pii] step {batch_i + 1}/{self.params['lev_nbatches']} done")

            # lev_batchsize more draws
            p += self.lev_batchsize

            # Take weighted average over all Pii draws
            Pii = (batch_i * Pii + Pii_i) / (batch_i + 1)

            # Compute number of bad draws
            n_bad_draws = sum(worker_m & (Pii >= self.params['lev_threshold_pii']))

            # If few enough bad draws, compute them analytically
            if n_bad_draws < self.params['lev_threshold_obs']:
                if n_bad_draws > 0:
                    leverage_warning = f"Breaking loop - threshold for max Pii is {self.params['lev_threshold_pii']}, with {self.lev_batchsize} draw(s) per batch and a maximum of {self.params['lev_nbatches']} batch(es) being drawn. There is/are {n_bad_draws} observation(s) with Pii above this threshold. This/these will be recomputed analytically. It took {batch_i + 1} batch(es) to get below the threshold of {self.params['lev_threshold_obs']} bad observation(s)."
                else:
                    leverage_warning = f"Breaking loop - threshold for max Pii is {self.params['lev_threshold_pii']}, with {self.lev_batchsize} draw(s) per batch and a maximum of {self.params['lev_nbatches']} batch(es) being drawn. No observations have Pii above this threshold. It took {batch_i + 1} batch(es) to get below the threshold of {self.params['lev_threshold_obs']} bad observation(s)."
                self.logger.debug(leverage_warning)
                if self.verbose:
                    tqdm.write(leverage_warning) # warnings.warn(leverage_warning)
                break
            elif batch_i == self.params['lev_nbatches'] - 1:
                leverage_warning = f"Threshold for max Pii is {self.params['lev_threshold_pii']}, with {self.lev_batchsize} draw(s) per batch and a maximum of {self.params['lev_nbatches']} batch(es) being drawn. After exhausting the maximum number of batches, there is/are still {n_bad_draws} draw(s) with Pii above this threshold. This/these will be recomputed analytically."
                self.logger.debug(leverage_warning)
                if self.verbose:
                    tqdm.write(leverage_warning) # warnings.warn(leverage_warning)
        del pbar

        # Compute Pii analytically for observations with Pii approximation above threshold value
        analytical_indices = self.adata.loc[worker_m & (Pii >= self.params['lev_threshold_pii']), :].index
        if (len(analytical_indices) > 0) and not (self.params['exact_trace_sigma_2'] or self.params['exact_trace_ho'] or self.params['exact_trace_he']):
            self._construct_AAinv_components_partial()

        # Compute analytical Pii
        if len(analytical_indices) > 0:
            # Construct weighted J and W
            DpJ = np.asarray((self.Dp_sqrt @ self.J).todense())
            DpW = np.asarray((self.Dp_sqrt @ self.W).todense())

            pbar = tqdm(analytical_indices, disable=self.no_pbars)
            pbar.set_description('leverages 2')
            for i in pbar:
                DpJ_i = DpJ[i, :]
                DpW_i = DpW[i, :]
                # Compute analytical Pii
                Pii[i] = self._compute_Pii(DpJ_i, DpW_i)
            del pbar

        self.res['min_lev'] = Pii[worker_m].min()
        self.res['max_lev'] = Pii[worker_m].max()

        if self.res['max_lev'] >= 1:
            leverage_warning = f"Max P_ii is {self.res['max_lev']} which is >= 1. This means your data is not leave-one-observation-out connected. The HE estimator requires leave-one-observation-out connected data to work properly. When cleaning your data, please set clean_params['connectedness'] = 'leave_out_observation' to correct this."
            self.logger.info(leverage_warning)
            raise ValueError(leverage_warning)

        self.logger.info(f"[he] leverage range {self.res['min_lev']:2.4f} to {self.res['max_lev']:2.4f}")

        return (Pii, p)

    def _estimate_Sii_he(self, Pii, p):
        '''
        Estimate Sii (sigma^2) for heteroskedastic correction.

        Arguments:
            Pii (NumPy Array): estimated leverages
            p (int): number of draws used in leverage approximation

        Returns:
            (NumPy Array): Sii (sigma^2) for heteroskedastic correction
        '''
        worker_m = (self.adata.loc[:, 'worker_m'].to_numpy() > 0)
        if self.weighted:
            w = self.Dp.data[0]

        ## Compute Sii for movers ##
        Sii_m = self.Y[worker_m] * self.E[worker_m] / (1 - Pii[worker_m])
        if not self.params['exact_lev_he']:
            # Non-linearity bias correction
            Sii_m *= (1 - (1 / p) * (3 * (Pii[worker_m] ** 3) + (Pii[worker_m] ** 2)) / (1 - Pii[worker_m]))

        ## Compute Sii for stayers ##
        # Source: https://github.com/rsaggio87/LeaveOutTwoWay/blob/master/codes/sigma_for_stayers.m
        if self.weighted:
            # Weighted
            Pii_s = 1 / self.adata.loc[~worker_m, ['i', 'w']].groupby('i', sort=False)['w'].transform('sum').to_numpy()
            Sii_s = (self.Y[~worker_m] - np.sum(w[~worker_m] * self.Y[~worker_m]) / np.sum(w[~worker_m])) * (self.E[~worker_m] / (1 - Pii_s))
        else:
            # Unweighted
            Pii_s = 1 / self.adata.loc[~worker_m, ['i', 'j']].groupby('i', sort=False)['j'].transform('size').to_numpy()
            Sii_s = (self.Y[~worker_m] - np.mean(self.Y[~worker_m])) * (self.E[~worker_m] / (1 - Pii_s))

        ## Combine Sii for all workers ##
        Sii = np.zeros(self.nn)
        Sii[worker_m] = Sii_m
        Sii[~worker_m] = Sii_s

        ## Take mean Sii_s at the firm level ##
        self.adata.loc[:, 'Sii'] = Sii
        if self.weighted:
            # Weighted average
            # How it works: imagine two observations in a firm, i=0 has w=1 and i=1 has w=5. Then Sii_0 is correct, but Sii_1 represents 1/5 of the variance of a single observation. We multiply Sii_1 by 5 to make it representative of a single observation, but then also need to weight it properly as 5 observations.
            self.adata.loc[:, 'weighted_Sii'] = (w ** 2) * Sii
            groupby_j = self.adata.loc[~worker_m, ['j', 'weighted_Sii', 'w']].groupby('j')
            Sii[~worker_m] = groupby_j['weighted_Sii'].transform('sum') / groupby_j['w'].transform('sum')
            # Divide by weight to re-adjust variance to account for more observations
            Sii[~worker_m] /= w[~worker_m]
            # No longer need weighted_Sii column
            self.adata.drop('weighted_Sii', axis=1, inplace=True)
        else:
            # Unweighted average
            Sii[~worker_m] = self.adata.loc[~worker_m, ['j', 'Sii']].groupby('j')['Sii'].transform('mean')
        # No longer need Sii column
        self.adata.drop('Sii', axis=1, inplace=True)

        # Compute sigma^2 HE
        if self.weighted:
            # Weighted average
            # Multiply by w, because each observation's variance is divided by w and we need to undo that
            self.res['eps_var_he'] = np.average(w * Sii, weights=w)
        else:
            # Unweighted average
            self.res['eps_var_he'] = np.mean(Sii)
        self.logger.info(f"[he] variance of residuals in heteroskedastic case: {self.res['eps_var_he']:2.4f}")

        return Sii

    def _leverage_approx(self, ndraw_pii, rng=None):
        '''
        Draw Pii estimates for use in JL approximation of leverage.

        Arguments:
            ndraw_pii (int): number of Pii draws
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (NumPy Array): Pii array
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        Pii = np.zeros(self.nn)

        # Compute the different draws
        for _ in range(ndraw_pii):
            R2 = 2 * rng.binomial(1, 0.5, self.nn) - 1
            Pii += (self._proj(R2, Dp0='sqrt', Dp2='sqrt') ** 2)

        # Take mean over draws
        Pii /= ndraw_pii

        self.logger.info('[he] [approximate pii] done with pii batch')

        return Pii
