'''
Defines class FEEstimator, which uses multigrid and partialing out to estimate weighted two way fixed effect models. This includes AKM, the Andrews et al. homoskedastic correction, and the Kline et al. heteroskedastic correction.
'''
'''
TODO:
    -leave-out-worker
    -Q with exact trace for more than psi and alpha
    -control variables
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
    'ho': (True, 'type', bool,
        '''
            (default=True) If True, estimate homoskedastic correction.
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
    'Sii_stayers': ('firm_mean', 'set', ['firm_mean', 'upper_bound'],
        '''
            (default='firm_mean') How to compute variance of worker effects for stayers for heteroskedastic correction. 'firm_mean' gives stayers the average variance estimate for movers at their firm. 'upper_bound' gives the upper bound variance estimate for stayers for worker effects by assuming the variance matrix is diagonal (please see page 17 of https://github.com/rsaggio87/LeaveOutTwoWay/blob/master/doc/VIGNETTE.pdf for more details).
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
    'ndraw_lev_he': (50, 'type_constrained', (int, _gteq1),
        '''
            (default=50) Number of draws to use when estimating leverage approximation for heteroskedastic correction.
        ''', '>= 1'),
    'lev_batchsize_he': (10, 'type_constrained', (int, _gteq1),
        '''
            (default=10) Batch size to send in parallel. Should evenly divide 'ndraw_lev_he'.
        ''', '>= 1'),
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
            params (ParamsDict or None): dictionary of parameters for FE estimation. Run tw.fe_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.fe_params().
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

        ### Save some commonly used parameters as attributes ###
        ## All ##
        # Whether data is weighted
        self.weighted = (params['weighted'] and ('w' in data.columns))
        # Progress bars
        self.no_pbars = not params['progress_bars']
        # Verbose
        self.verbose = params['verbose']
        # Number of cores to use
        self.ncore = params['ncore']
        ## HO/HE ##
        # Whether to compute homoskedastic correction
        self.compute_ho = params['ho']
        # Whether to compute heteroskedastic correction
        self.compute_he = params['he']
        # Number of draws to use in sigma^2 trace approximation
        self.ndraw_trace_sigma_2 = params['ndraw_trace_sigma_2']
        # Number of draws to use in homoskedastic trace approximation
        self.ndraw_trace_ho = params['ndraw_trace_ho']
        # Number of draws to use in heteroskedastic trace approximation
        self.ndraw_trace_he = params['ndraw_trace_he']
        # Number of draws to compute leverage for heteroskedastic correction
        self.ndraw_lev_he = params['ndraw_lev_he']


        ## Store some parameters in results dictionary ##
        self.res['cores'] = self.ncore
        self.res['ndraw_trace_ho'] = self.ndraw_trace_ho
        self.res['ndraw_trace_he'] = self.ndraw_trace_he
        self.res['ndraw_lev_he'] = self.ndraw_lev_he

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
                # Keep track of whether (A.T @ A)^{-1} has been computed
                aainv = False
                # Estimate sigma^2 (variance of residuals)
                self._estimate_sigma_2_fe()
                # Construct Q matrix
                Q_params = self._construct_Q()
                # Estimate plug-in (biased) FE model
                self._estimate_fe(Q_params)

                ## HO/HE corrections ##
                if self.compute_ho:
                    ## HO correction ##
                    if self.weighted:
                        # Estimate trace for sigma^2 (variance of residuals) for HO correction
                        if self.params['exact_trace_sigma_2']:
                            # Analytical trace
                            if not aainv:
                                self._construct_AAinv_components_full()
                                aainv = True
                            self._estimate_exact_trace_sigma_2()
                        else:
                            # Approximate trace
                            self._estimate_approximate_trace_sigma_2(rng)
                    # Estimate sigma^2 (variance of residuals) for HO correction
                    self._estimate_sigma_2_ho()
                    # Estimate trace for HO correction
                    if self.params['exact_trace_ho']:
                        # Analytical trace
                        if not aainv:
                            self._construct_AAinv_components_full()
                            aainv = True
                        self._estimate_exact_trace_ho(Q_params)
                    else:
                        # Approximate trace
                        self._estimate_approximate_trace_ho(Q_params, rng)

                if self.compute_he:
                    ## HE correction ##
                    if (self.params['exact_lev_he'] or self.params['exact_trace_he']) and not aainv:
                        self._construct_AAinv_components_full()
                        aainv = True
                    # Estimate leverages for HE correction
                    if len(self.params['levfile']) > 0:
                        # Precomputed leverages
                        Pii, jla_factor = self._extract_precomputed_leverages()
                    elif self.params['exact_lev_he']:
                        # Analytical leverages
                        Pii, jla_factor = self._estimate_exact_leverages()
                    else:
                        # Approximate leverages
                        Pii, jla_factor = self._estimate_approximate_leverages(rng)
                    # Estimate Sii (variance of residuals) for HE correction
                    Sii = self._estimate_Sii_he(Pii, jla_factor)
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
        del self.worker_m, self.Y, self.J, self.W, self.Dp, self.Dp_sqrt, self.Dwinv, self.Minv, self.ml
        try:
            del self.AA_inv_A, self.AA_inv_B, self.AA_inv_C, self.AA_inv_D
        except AttributeError:
            pass

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

        # Generate worker_m, indicating whether a worker is a mover or a stayer
        self.worker_m = self.adata.get_worker_m(is_sorted=True)

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
            self.adata.loc[:, 'weighted_m'] = self.Dp * self.worker_m
            self.adata.loc[:, 'weighted_y'] = self.Dp * self.Y
            fdata = self.adata.groupby('j')[['weighted_m', 'weighted_y', 'w']].sum()
            fm, fy, fi = fdata.loc[:, 'weighted_m'].to_numpy(), fdata.loc[:, 'weighted_y'].to_numpy(), fdata.loc[:, 'w'].to_numpy()
            fy /= fi
            self.adata.drop(['weighted_m', 'weighted_y'], axis=1, inplace=True)
        else:
            self.adata.loc[:, 'worker_m'] = self.worker_m
            fdata = self.adata.groupby('j').agg({'worker_m': 'sum', 'y': 'mean', 'i': 'count'})
            fm, fy, fi = fdata.loc[:, 'worker_m'].to_numpy(), fdata.loc[:, 'y'].to_numpy(), fdata.loc[:, 'i'].to_numpy()
            self.adata.drop('worker_m', axis=1, inplace=True)
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
        self.logger.info(f'solver time: {self.last_invert_time:2.4f} seconds')

        if not self.params['feonly']:
            n_draws_ho = self.compute_ho * self.ndraw_trace_ho
            n_draws_he = self.compute_he * (self.ndraw_trace_he + self.ndraw_lev_he / self.ncore)
            expected_time = (self.last_invert_time / 60) * (n_draws_ho + n_draws_he)

            self.logger.info(f'expected total time: {expected_time:2.4f} minutes')

        else:
            self.logger.info(f'expected total time: 0 minutes')

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

    def _estimate_sigma_2_fe(self):
        '''
        Estimate residuals and sigma^2 (variance of residuals) for plug-in (biased) FE model.
        '''
        Dp = self.Dp

        ## Estimate residuals ##
        self.E = self.Y - self._mult_A(self.psi_hat, self.alpha_hat)

        ## Estimate R^2 ##
        fe_rsq = 1 - (Dp * (self.E ** 2)).sum() / (Dp * (self.Y ** 2)).sum()
        self.logger.info(f'fixed effect R-square {fe_rsq:2.4f}')

        ## Estimate variance of residuals (DON'T DEMEAN) ##
        # NOTE: multiply by Dp, because each observation's variance is divided by Dp and we need to undo that
        self.sigma_2_pi = np.average(Dp * self.E ** 2, weights=Dp.data[0])

        self.logger.info(f'[fe] variance of residuals {self.sigma_2_pi:2.4f}')

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

        psi = self.psi_hat
        alpha = self.alpha_hat

        self.var_fe = _weighted_var(self.Q_var._Q_mult(Q_var_matrix, psi, alpha), Q_var_weights, dof=0)
        self.cov_fe = _weighted_cov(self.Q_cov._Ql_mult(Ql_cov_matrix, psi, alpha), self.Q_cov._Qr_mult(Qr_cov_matrix, psi, alpha), Ql_cov_weights, Qr_cov_weights, dof=0)

        self.logger.info('[fe]')
        self.logger.info(f'var_psi={self.var_fe:2.4f}')
        self.logger.info(f"cov={self.cov_fe:2.4f} tot={self.res['var_y']:2.4f}")

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

    def _estimate_sigma_2_ho(self):
        '''
        Estimate sigma^2 (variance of residuals) for HO-corrected model.
        '''

        if self.weighted:
            # Must use unweighted sigma^2 for numerator (weighting will make the estimator biased)
            sigma_2_unweighted = np.mean(self.E ** 2)
            trace_approximation = np.mean(self.tr_sigma_ho_all)
            self.sigma_2_ho = (self.nn * sigma_2_unweighted) / (np.sum(1 / self.Dp.data[0]) - trace_approximation)
        else:
            self.sigma_2_ho = (self.nn * self.sigma_2_pi) / (self.nn - (self.nw + self.nf - 1))

        self.logger.info(f'[ho] variance of residuals {self.sigma_2_ho:2.4f}')

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
        Q_var_weights_sqrt = np.sqrt(Q_var_weights)
        Ql_cov_weights_sqrt = np.sqrt(Ql_cov_weights)
        Qr_cov_weights_sqrt = np.sqrt(Qr_cov_weights)
        Q_cov_weights = Ql_cov_weights_sqrt * Qr_cov_weights_sqrt

        if Q_var_psialpha not in ['psi', 'alpha']:
            raise NotImplementedError(f'Exact HO correction is compatible with only Q matrices that interact solely with psi or alpha components, but the selected Q-variance interacts with {Q_var_psialpha!r}.')
        if (Ql_cov_psialpha not in ['psi', 'alpha']) or (Qr_cov_psialpha not in ['psi', 'alpha']):
            raise NotImplementedError(f'Exact HO correction is compatible with only Q matrices that interact solely with psi or alpha components, but the selected Ql/Qr-covariances interact with {Ql_cov_psialpha!r}/{Qr_cov_psialpha!r}.')

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
        # Q = Ql.T @ U.T @ Ql_weight.T @ Qr_weight @ U @ Qr
        n = np.sum(Q_var_weights)
        # Construct U (demeaning matrix)
        U_dim = Q_var_matrix.shape[0]
        U = eye(U_dim) - (1 / n) * np.tile(Q_var_weights.diagonal(), (U_dim, 1))
        # Compute Q @ (A'D_pA)^{-1} using block matrix components
        Q_var = Q_var_weights_sqrt @ U @ Q_var_matrix
        Q_var = (1 / n) * Q_var.T @ Q_var
        # Compute trace by considering correct quadrant
        self.tr_var_ho_all = np.trace(Q_var @ right_dict[Q_var_psialpha][left_dict[Q_var_psialpha]])

        ## Trace - covariance ##
        # Q = Ql.T @ Ul.T @ Ql_weight.T @ Qr_weight @ Ur @ Qr
        n = np.sum(Q_cov_weights)
        nl = np.sum(Ql_cov_weights)
        nr = np.sum(Qr_cov_weights)
        # Construct U (demeaning matrix)
        Ul_dim = Ql_cov_matrix.shape[0]
        Ur_dim = Qr_cov_matrix.shape[0]
        Ul = eye(Ul_dim) - (1 / nl) * np.tile(Ql_cov_weights.diagonal(), (Ul_dim, 1))
        Ur = eye(Ur_dim) - (1 / nr) * np.tile(Qr_cov_weights.diagonal(), (Ur_dim, 1))
        # Compute Q @ (A'D_pA)^{-1} using block matrix components
        Q_cov = (1 / n) * Ql_cov_matrix.T @ Ul.T @ Ql_cov_weights_sqrt.T @ Qr_cov_weights_sqrt @ Ur @ Qr_cov_matrix
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

            # Compute (A'D_pA)^{-1} @ Z
            psi1, alpha1 = self._mult_AAinv(Zpsi, Zalpha)

            ## Trace correction - variance ##
            # Left term of Q matrix
            L_var = self.Q_var._Q_mult(Q_var_matrix, Zpsi, Zalpha).T
            # Right term of Q matrix
            R_var = self.Q_var._Q_mult(Q_var_matrix, psi1, alpha1)
            self.tr_var_ho_all[r] = _weighted_cov(L_var, R_var, Q_var_weights, Q_var_weights)
            del L_var, R_var

            ## Trace correction - covariance ##
            # Left term of Q matrix
            L_cov = self.Q_cov._Ql_mult(Ql_cov_matrix, Zpsi, Zalpha).T
            # Right term of Q matrix
            R_cov = self.Q_cov._Qr_mult(Qr_cov_matrix, psi1, alpha1)
            self.tr_cov_ho_all[r] = _weighted_cov(L_cov, R_cov, Ql_cov_weights, Qr_cov_weights)
            del L_cov, R_cov

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
        Q_var_weights_sqrt = np.sqrt(Q_var_weights)
        Ql_cov_weights_sqrt = np.sqrt(Ql_cov_weights)
        Qr_cov_weights_sqrt = np.sqrt(Qr_cov_weights)
        Q_cov_weights = Ql_cov_weights_sqrt * Qr_cov_weights_sqrt

        if Q_var_psialpha not in ['psi', 'alpha']:
            raise NotImplementedError(f'Exact HE correction is compatible with only Q matrices that interact solely with psi or alpha components, but the selected Q-variance interacts with {Q_var_psialpha!r}.')
        if (Ql_cov_psialpha not in ['psi', 'alpha']) or (Qr_cov_psialpha not in ['psi', 'alpha']):
            raise NotImplementedError(f'Exact HE correction is compatible with only Q matrices that interact solely with psi or alpha components, but the selected Ql/Qr-covariances interact with {Ql_cov_psialpha!r}/{Qr_cov_psialpha!r}.')

        # Unpack attributes
        J, W = self.J, self.W
        Dp_sqrt = self.Dp_sqrt
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
        # Q = Ql.T @ U.T @ Ql_weight.T @ Qr_weight @ U @ Qr
        n = np.sum(Q_var_weights)
        # Construct U (demeaning matrix)
        U_dim = Q_var_matrix.shape[0]
        U = eye(U_dim) - (1 / n) * np.tile(Q_var_weights.diagonal(), (U_dim, 1))
        ## Compute Q @ (A'D_pA)^{-1} @ (D_pA)' @ Omega @ (D_pA) @ (A'D_pA)^{-1} using block matrix components ##
        # Construct Q
        Q_var = Q_var_weights_sqrt @ U @ Q_var_matrix
        Q_var = (1 / n) * Q_var.T @ Q_var
        # Construct AA_inv
        if Q_var_psialpha == 'psi':
            AA_inv = Dp_sqrt @ (J @ A + W @ C)
            AA_inv = AA_inv.T @ Sii @ AA_inv
        elif Q_var_psialpha == 'alpha':
            AA_inv = Dp_sqrt @ (J @ B + W @ D)
            AA_inv = AA_inv.T @ Sii @ AA_inv
        
        # Compute trace by considering correct quadrant
        self.tr_var_he_all = np.trace(Q_var @ AA_inv)

        ## Trace - covariance ##
        # Q = Ql.T @ Ul.T @ Ql_weight.T @ Qr_weight @ Ur @ Qr
        n = np.sum(Q_cov_weights)
        nl = np.sum(Ql_cov_weights)
        nr = np.sum(Qr_cov_weights)
        # Construct U (demeaning matrix)
        Ul_dim = Ql_cov_matrix.shape[0]
        Ur_dim = Qr_cov_matrix.shape[0]
        Ul = eye(Ul_dim) - (1 / nl) * np.tile(Ql_cov_weights.diagonal(), (Ul_dim, 1))
        Ur = eye(Ur_dim) - (1 / nr) * np.tile(Qr_cov_weights.diagonal(), (Ur_dim, 1))
        ## Compute Q @ (A'D_pA)^{-1} @ (D_pA)' @ Omega @ (D_pA) @ (A'D_pA)^{-1} using block matrix components ##
        # Construct Q
        Q_cov = (1 / n) * Ql_cov_matrix.T @ Ul.T @ Ql_cov_weights_sqrt.T @ Qr_cov_weights_sqrt @ Ur @ Qr_cov_matrix
        # Construct AA_inv
        if Qr_cov_psialpha == 'psi':
            if Ql_cov_psialpha == 'psi':
                if Q_var_psialpha != 'psi':
                    # No need to recompute if var is the same
                    AA_inv = Dp_sqrt @ (J @ A + W @ C)
                    AA_inv = AA_inv.T @ Sii @ AA_inv
            elif Ql_cov_psialpha == 'alpha':
                AA_inv = (A @ J.T + B @ W.T) @ Dp_sqrt @ Sii @ Dp_sqrt @ (J @ B + W @ D)
        elif Qr_cov_psialpha == 'alpha':
            if Ql_cov_psialpha == 'psi':
                AA_inv = (C @ J.T + D @ W.T) @ Dp_sqrt @ Sii @ Dp_sqrt @ (J @ A + W @ C)
            elif Ql_cov_psialpha == 'alpha':
                if Q_var_psialpha != 'alpha':
                    # No need to recompute if var is the same
                    AA_inv = Dp_sqrt @ (J @ B + W @ D)
                    AA_inv = AA_inv.T @ Sii @ AA_inv
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

            ## Trace correction - variance ##
            # Left term of Q matrix
            L_var = self.Q_var._Q_mult(Q_var_matrix, Zpsi, Zalpha).T
            # Right term of Q matrix
            R_var = self.Q_var._Q_mult(Q_var_matrix, psi1, alpha1)
            self.tr_var_he_all[r] = _weighted_cov(L_var, R_var, Q_var_weights, Q_var_weights)
            del L_var, R_var

            ## Trace correction - covariance ##
            # Left term of Q matrix
            L_cov = self.Q_cov._Ql_mult(Ql_cov_matrix, Zpsi, Zalpha).T
            # Right term of Q matrix
            R_cov = self.Q_cov._Qr_mult(Qr_cov_matrix, psi1, alpha1)
            self.tr_cov_he_all[r] = _weighted_cov(L_cov, R_cov, Ql_cov_weights, Qr_cov_weights)
            del L_cov, R_cov

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
        self.logger.info(f'[fe] VAR fe={self.var_fe:2.4f}')
        # Plug-in covariance
        self.res['cov_fe'] = self.cov_fe
        self.logger.info(f'[fe] COV fe={self.cov_fe:2.4f}')

        for res in ['var_y', 'var_fe', 'cov_fe']:
            self.summary[res] = self.res[res]

        ## Homoskedastic results ##
        if self.compute_ho:
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

            for res in ['var_ho', 'cov_ho']:
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
            self.logger.info(f"[he] VAR bc={self.var_fe:2.4f} bc={self.res['var_he']:2.4f}")
            # Bias-corrected covariance
            self.res['cov_he'] = self.cov_fe - self.res['tr_cov_he']
            self.logger.info(f"[he] COV bc={self.cov_fe:2.4f} bc={self.res['cov_he']:2.4f}")

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
        Construct Q (variance/covariance) matrices, store Q classes as attributes, and generate related parameters.

        Returns:
            (tuple): (Q variance parameters, Q left covariance parameters, Q right covariance parameters)
        '''
        Q_var = self.params['Q_var']
        Q_cov = self.params['Q_cov']
        if Q_var is None:
            Q_var = Q.VarPsi()
        if Q_cov is None:
            Q_cov = Q.CovPsiAlpha()
        self.Q_var = Q_var
        self.Q_cov = Q_cov
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
        # if (self.params['exact_trace_sigma_2'] and self.weighted) or self.params['exact_trace_ho'] or self.params['exact_trace_he'] or self.params['exact_lev_he']:
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
            (tuple of NumPy Arrays): (Pii --> estimated leverages, jla_factor --> JLA non-linearity bias correction (this method always sets jla_factor=None))
        '''
        Pii = np.zeros(self.nn)

        self.logger.info('[he] starting heteroskedastic correction, loading precomputed files')

        files = glob.glob('{}*'.format(self.params['levfile']))
        self.logger.info(f'[he] found {len(files)} files to get leverages from')
        self.res['lev_file_count'] = len(files)
        assert len(files) > 0, "Didn't find any leverage files!"

        for f in files:
            pp = np.load(f)
            Pii += pp / len(files)

        if self.weighted or (self.params['Sii_stayers'] == 'firm_mean'):
            self.res['min_lev'] = Pii[self.worker_m].min()
            self.res['max_lev'] = Pii[self.worker_m].max()
        else:
            self.res['min_lev'] = Pii.min()
            self.res['max_lev'] = Pii.max()

        if self.res['max_lev'] >= 1:
            leverage_warning = f"Max P_ii is {self.res['max_lev']} which is >= 1. This means your data is not leave-one-observation-out connected. The HE estimator requires leave-one-observation-out connected data to work properly. When cleaning your data, please set clean_params['connectedness'] = 'leave_out_observation' to correct this."
            self.logger.info(leverage_warning)
            raise ValueError(leverage_warning)

        self.logger.info(f"[he] leverage range {self.res['min_lev']:2.4f} to {self.res['max_lev']:2.4f}")

        return (Pii, None)

    def _estimate_exact_leverages(self):
        '''
        Estimate analytical leverages (Pii) for heteroskedastic correction.

        Returns:
            (tuple of NumPy Arrays): (Pii --> estimated leverages, jla_factor --> JLA non-linearity bias correction (this method always sets jla_factor=None))
        '''
        self.logger.info('[he] [analytical Pii]')

        Pii = np.zeros(self.nn)

        # Construct weighted J and W
        DpJ = self.Dp_sqrt @ self.J
        DpW = self.Dp_sqrt @ self.W

        if self.weighted:
            pbar = tqdm(self.adata.loc[self.worker_m, :].index, disable=self.no_pbars)
        else:
            pbar = tqdm(self.adata.index, disable=self.no_pbars)
        pbar.set_description('leverages')
        for i in pbar:
            DpJ_i = np.asarray(DpJ[i, :].todense())[0, :]
            DpW_i = np.asarray(DpW[i, :].todense())[0, :]

            # Compute analytical Pii
            Pii[i] = self._compute_Pii(DpJ_i, DpW_i)
        del pbar

        if self.weighted or (self.params['Sii_stayers'] == 'firm_mean'):
            self.res['min_lev'] = Pii[self.worker_m].min()
            self.res['max_lev'] = Pii[self.worker_m].max()
        else:
            self.res['min_lev'] = Pii.min()
            self.res['max_lev'] = Pii.max()

        if self.res['max_lev'] >= 1:
            leverage_warning = f"Max P_ii is {self.res['max_lev']} which is >= 1. This means your data is not leave-one-observation-out connected. The HE estimator requires leave-one-observation-out connected data to work properly. When cleaning your data, please set clean_params['connectedness'] = 'leave_out_observation' to correct this."
            self.logger.info(leverage_warning)
            raise ValueError(leverage_warning)

        self.logger.info(f"[he] leverage range {self.res['min_lev']:2.4f} to {self.res['max_lev']:2.4f}")

        return (Pii, None)

    def _estimate_approximate_leverages(self, rng=None):
        '''
        Estimate approximate leverages (Pii) for heteroskedastic correction (source: https://github.com/rsaggio87/LeaveOutTwoWay/blob/master/doc/improved_JLA.pdf).

        Arguments:
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (tuple of NumPy Arrays): (Pii --> estimated leverages, jla_factor --> JLA non-linearity bias correction)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        Pii = np.zeros(self.nn)
        worker_m = self.worker_m

        self.logger.info(f"[he] [approximate pii] ndraw_lev_he={self.ndraw_lev_he}, lev_batchsize_he={self.params['lev_batchsize_he']}, using {self.ncore} core(s)")

        if self.ncore > 1:
            # Multiprocessing
            ndraw_seeds = self.ndraw_lev_he // self.params['lev_batchsize_he']
            if np.round(ndraw_seeds * self.params['lev_batchsize_he']) != self.ndraw_lev_he:
                # 'lev_batchsize_he' must evenly divide 'ndraw_lev_he'
                raise ValueError(f"'lev_batchsize_he' (currently {self.params['lev_batchsize_he']}) should evenly divide 'ndraw_lev_he' (currently {self.ndraw_lev_he}).")
            # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
            seeds = rng.bit_generator._seed_seq.spawn(ndraw_seeds)
            with Pool(processes=self.ncore) as pool:
                pbar2 = tqdm([(self.params['lev_batchsize_he'], np.random.default_rng(seed)) for seed in seeds], total=ndraw_seeds, disable=self.no_pbars)
                pbar2.set_description('leverages batch')
                Pii_all, Pii_sq_all, Mii_all, Mii_sq_all, Pii_Mii_all = pool.starmap(self._leverage_approx, pbar2)
                del pbar2

            # Take mean over draws
            Pii = sum(Pii_all) / ndraw_seeds
            Pii_sq = sum(Pii_sq_all) / ndraw_seeds
            Mii = sum(Mii_all) / ndraw_seeds
            Mii_sq = sum(Mii_sq_all) / ndraw_seeds
            Pii_Mii = sum(Pii_Mii_all) / ndraw_seeds
        else:
            # Single core
            Pii, Pii_sq, Mii, Mii_sq, Pii_Mii = self._leverage_approx(self.ndraw_lev_he, rng)

        # Normalize Pii
        if self.weighted:
            Pii[worker_m] = Pii[worker_m] / (Pii[worker_m] + Mii[worker_m])
            self.res['min_lev'] = Pii[worker_m].min()
            self.res['max_lev'] = Pii[worker_m].max()
        else:
            Pii = Pii / (Pii + Mii)
            self.res['min_lev'] = Pii.min()
            self.res['max_lev'] = Pii.max()

        if self.res['max_lev'] >= 1:
            leverage_warning = f"Max P_ii is {self.res['max_lev']} which is >= 1. This means your data is not leave-one-observation-out connected. The HE estimator requires leave-one-observation-out connected data to work properly. When cleaning your data, please set clean_params['connectedness'] = 'leave_out_observation' to correct this."
            self.logger.info(leverage_warning)
            raise ValueError(leverage_warning)

        self.logger.info(f"[he] leverage range {self.res['min_lev']:2.4f} to {self.res['max_lev']:2.4f}")

        ## JLA non-linearity bias correction ##
        if self.weighted or (self.params['Sii_stayers'] == 'firm_mean'):
            # Compute for movers
            Pii_m = Pii[worker_m]
            Mii_m = 1 - Pii_m
            Vi = (Mii_m ** 2) * Pii_sq[worker_m] + (Pii_m ** 2) * Mii_sq[worker_m] - 2 * Pii_m * Mii_m * Pii_Mii[worker_m]
            Bi = Mii_m * Pii_sq[worker_m] - Pii_m * Mii_sq[worker_m] + 2 * (Mii_m - Pii_m) * Pii_Mii[worker_m]
            # Compute bias correction factor
            jla_factor = (1 - (1 / self.ndraw_lev_he) * ((Vi / Mii_m + Bi) / Mii_m))
        else:
            # Compute for movers and stayers
            Mii = 1 - Pii
            Vi = (Mii ** 2) * Pii_sq + (Pii ** 2) * Mii_sq - 2 * Pii * Mii * Pii_Mii
            Bi = Mii * Pii_sq - Pii * Mii_sq + 2 * (Mii - Pii) * Pii_Mii
            # Compute bias correction factor
            jla_factor = (1 - (1 / self.ndraw_lev_he) * ((Vi / Mii + Bi) / Mii))

        return (Pii, jla_factor)

    def _estimate_Sii_he(self, Pii, jla_factor=None):
        '''
        Estimate Sii (sigma^2) for heteroskedastic correction, and the non-linearity bias correction if using approximate Pii (source: https://github.com/rsaggio87/LeaveOutTwoWay/blob/master/doc/improved_JLA.pdf).

        Arguments:
            Pii (NumPy Array): estimated leverages
            jla_factor (NumPy Array or None): JLA non-linearity bias correction; None provides no correction

        Returns:
            (NumPy Array): Sii (sigma^2) for heteroskedastic correction
        '''
        worker_m = self.worker_m
        w = self.Dp.data[0]

        # Mean wage
        Y_bar = np.average(self.Y, weights=w)

        if self.params['Sii_stayers'] == 'firm_mean':
            ### Give stayers the average variance estimate at the firm level ###
            ## Compute Sii for movers ##
            Pii_m = Pii[worker_m]
            Sii_m = (self.Y[worker_m] - Y_bar) * (self.E[worker_m] / (1 - Pii_m))
            if (not self.params['exact_lev_he']) and (jla_factor is not None):
                # Multiply by bias correction factor
                Sii_m *= jla_factor
            del Pii_m

            ## Compute Sii for stayers ##
            if self.weighted:
                ### Weighted ###
                # Add Sii for movers to dataframe (multiply by weight)
                self.adata.loc[worker_m, 'weighted_Sii'] = (w[worker_m] ** 2) * Sii_m
                # Link firms to average Sii of movers
                groupby_j = self.adata.loc[worker_m, ['j', 'w', 'weighted_Sii']].groupby('j', sort=True)
                Sii_j = (groupby_j['weighted_Sii'].sum() / groupby_j['w'].sum()).to_dict()
                # Compute Sii for stayers (divide by weight)
                Sii_s = self.adata.loc[~worker_m, 'j'].map(Sii_j).to_numpy() / w[~worker_m]
                # No longer need Sii column or groupby_j
                self.adata.drop('weighted_Sii', axis=1, inplace=True)
                del groupby_j
            else:
                ### Unweighted ###
                # Add Sii for movers to dataframe
                self.adata.loc[worker_m, 'Sii'] = Sii_m
                # Link firms to average Sii of movers
                Sii_j = self.adata.loc[worker_m, ['j', 'Sii']].groupby('j', sort=True)['Sii'].mean().to_dict()
                # Compute Sii for stayers
                Sii_s = self.adata.loc[~worker_m, 'j'].map(Sii_j).to_numpy()
                # No longer need Sii column
                self.adata.drop('Sii', axis=1, inplace=True)
            # No longer need Sii_j
            del Sii_j

            ## Combine Sii for all workers ##
            Sii = np.zeros(self.nn)
            Sii[worker_m] = Sii_m
            Sii[~worker_m] = Sii_s
            del Sii_m, Sii_s

        elif self.params['Sii_stayers'] == 'upper_bound':
            ### Give stayers the upper bound variance estimate for worker effects ###
            # Source: https://github.com/rsaggio87/LeaveOutTwoWay/blob/master/codes/sigma_for_stayers.m
            if self.weighted:
                ### Weighted ###
                ## Compute Sii for movers ##
                Pii_m = Pii[worker_m]
                Sii_m = (self.Y[worker_m] - Y_bar) * (self.E[worker_m] / (1 - Pii_m))
                if (not self.params['exact_lev_he']) and (jla_factor is not None):
                    # Multiply by bias correction factor
                    Sii_m *= jla_factor
                del Pii_m

                ## Compute Sii for stayers ##
                Pii_s = 1 / self.adata.loc[~worker_m, ['i', 'w']].groupby('i', sort=False)['w'].transform('sum').to_numpy()
                if np.any(Pii_s == 1):
                    raise ValueError("At least one stayer has observation-weight 1, which prevents computing Pii and Sii for those stayers. Please set 'drop_single_stayers'=True during data cleaning to avoid this error.")
                # NOTE: divide by weight
                Sii_s = (self.Y[~worker_m] - Y_bar) * (self.E[~worker_m] / (1 - Pii_s)) / w[~worker_m]
                del Pii_s

                ## Combine Sii for all workers ##
                Sii = np.zeros(self.nn)
                Sii[worker_m] = Sii_m
                Sii[~worker_m] = Sii_s
                del Sii_m, Sii_s
            else:
                ### Unweighted ###
                if np.any(self.adata.loc[~worker_m, ['i', 'j']].groupby('i', sort=False)['j'].transform('size').to_numpy() == 1):
                    raise ValueError("At least one stayer has only one observation, which prevents computing Pii and Sii for those stayers. Please set 'drop_single_stayers'=True during data cleaning to avoid this error.")
                Sii = (self.Y - Y_bar) * self.E / (1 - Pii)
                if (not self.params['exact_lev_he']) and (jla_factor is not None):
                    # Multiply by bias correction factor
                    Sii *= jla_factor

        # Compute sigma^2 HE (multiply by Dp, because each observation's variance is divided by Dp and we need to undo that)
        self.res['eps_var_he'] = np.average(w * Sii, weights=w)

        self.logger.info(f"[he] variance of residuals {self.res['eps_var_he']:2.4f}")

        return Sii

    def _leverage_approx(self, ndraw_pii, rng=None):
        '''
        Draw Pii estimates for use in JL approximation of leverage (source: https://github.com/rsaggio87/LeaveOutTwoWay/blob/master/doc/improved_JLA.pdf).

        Arguments:
            ndraw_pii (int): number of Pii draws
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (tuple of NumPy Arrays): (Pii --> estimated leverages, Pii_sq --> estimated square of leverages, Mii --> estimated (1 - leverages), Mii_sq --> estimated square of (1 - leverages), Pii_Mii --> estimated product of leverages and (1 - leverages))
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        Pii = np.zeros(self.nn)
        Pii_sq = np.zeros(self.nn)
        Mii = np.zeros(self.nn)
        Mii_sq = np.zeros(self.nn)
        Pii_Mii = np.zeros(self.nn)

        # Compute the different draws
        pbar = trange(ndraw_pii, disable=self.no_pbars)
        pbar.set_description('leverages')
        for _ in pbar:
            R2 = 2 * rng.binomial(1, 0.5, self.nn) - 1
            # NOTE: set Dp0='sqrt' and Dp2='sqrt'
            q = self._proj(R2, Dp0='sqrt', Dp2='sqrt')
            # Compute Pii and Mii for this i
            Pi_i = (q ** 2)
            Mi_i = (R2 - q) ** 2
            # Update Pii, Pii_sq, Mii, Mii_sq, and Pii_Mii
            Pii += Pi_i
            Pii_sq += (Pi_i ** 2)
            Mii += Mi_i
            Mii_sq += (Mi_i ** 2)
            Pii_Mii += Pi_i * Mi_i

        # Take mean over draws
        Pii /= ndraw_pii
        Pii_sq /= ndraw_pii
        Mii /= ndraw_pii
        Mii_sq /= ndraw_pii
        Pii_Mii /= ndraw_pii

        self.logger.info('[he] [approximate pii] done with pii batch')

        return (Pii, Pii_sq, Mii, Mii_sq, Pii_Mii)
