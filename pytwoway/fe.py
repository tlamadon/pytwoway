'''
Defines class FEEstimator, which uses multigrid and partialing out to estimate weighted two way fixed effect models. This includes AKM, the Andrews et al. homoskedastic correction, and the Kline et al. heteroskedastic correction.
'''
'''
TODO:
    -leave-out-worker
    -Q with exact trace for more than psi and alpha
    -Hutch++ for HO and HE trace approximations
'''
from tqdm.auto import tqdm, trange
import time, pickle, json, glob
from timeit import default_timer as timer
try:
    from multiprocess import Pool
except ImportError:
    from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import bicg, bicgstab, cg, cgs, gmres, minres, qmr, LinearOperator, spilu
solver_dict = {
    'bicg': bicg,
    'bicgstab': bicgstab,
    'cg': cg,
    'cgs': cgs,
    'gmres': gmres,
    'minres': minres,
    'qmr': qmr
}
from pyamg import ruge_stuben_solver as rss
from bipartitepandas.util import ParamsDict, to_list, logger_init
from pytwoway import Q
from pytwoway import preconditioners as pcd
from pytwoway.util import weighted_mean, weighted_var, weighted_cov, weighted_quantile, DxSP, SPxD, DxM, MxD, diag_of_sp_prod, diag_of_prod

# def pipe_qcov(df, e1, e2): # FIXME I moved this from above, also this is used only in commented out code
#     v1 = df.eval(e1)
#     v2 = df.eval(e2)
#     return np.cov(v1, v2)[0][1]

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq1(a):
    return a >= 1
def _gteq0(a):
    return a >= 0

# Define default parameter dictionary
_fe_params_default = ParamsDict({
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
    'Q_var': (None, 'list_of_type_none', (Q.VarPsi, Q.VarAlpha, Q.VarPsiPlusAlpha),
        '''
            (default=None) List of Q matrices to use when estimating variance term; None is equivalent to tw.Q.VarPsi().
        ''', None),
    'Q_cov': (None, 'list_of_type_none', (Q.CovPsiAlpha, Q.CovPsiPrevPsiNext),
        '''
            (default=None) List of Q matrices to use when estimating covariance term; None is equivalent to tw.Q.CovPsiAlpha().
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
    'ndraw_lev_he': (200, 'type_constrained', (int, _gteq1),
        '''
            (default=200) Number of draws to use when estimating leverage approximation for heteroskedastic correction. Should be a multiple of 'ncore'.
        ''', '>= 1'),
    'levfile': ('', 'type', str,
        '''
            (default='') File to load precomputed leverages for heteroskedastic correction.
        ''', None),
    # 'con': (False, 'type', bool, # FIXME not used
    #     '''
    #         (default=False) Computes the smallest eigenvalues, this is the filepath where these results are saved.
    #     ''', None),
    'ncore': (1, 'type_constrained', (int, _gteq1),
        '''
            (default=1) Number of cores to use.
        ''', '>= 1'),
    'solver': ('minres', 'set', ['bicg', 'bicgstab', 'cg', 'cgs', 'gmres', 'minres', 'qmr', 'amg'],
        '''
            (default='minres') Solver to use for solving linear systems. The recommended solver is 'minres', unless you are working with very large datasets (e.g. 100 million observations or more), in which case it is recommended to use 'amg'. Options are:
                bicg: BIConjugate Gradient
                bicgstab: BIConjugate Gradient STABilized
                cg: Conjugate Gradient
                cgs: Conjugate Gradient Squared
                gmres: Generalized Minimal RESidual
                minres: MINimum RESidual
                qmr: Quasi-Minimal Residual
                amg: Algebraic Multi-Grid
        ''', None),
    'solver_tol': (1e-10, 'type_constrained', (int, _gteq0),
        '''
            (default=1e-10) Tolerance for convergence of linear solver (Ax=b), iterations stop when norm(residual) <= tol * norm(b). A lower tolerance will achieve better estimates at the cost of comp1e-10utation time.
        ''', '>= 0'),
    'preconditioner': ('ichol', 'set', (None, 'jacobi', 'vcycle', 'ichol', 'ilu'),
        '''
            (default='ichol') Preconditioner for linear solver. 'jacobi' uses Jacobi preconditioner; 'vcycle' uses V-Cycle preconditioner; 'ichol' uses incomplete Cholesky decomposition preconditioner; 'ilu' uses incomplete LU decomposition preconditioner. For large datasets, it is recommended to switch to the Jacobi preconditioner. If None, do not precondition linear solver. Not used for 'amg' solver.
        ''', None),
    'preconditioner_options': (None, 'type_none', dict,
        '''
            (default=None) Dictionary of preconditioner options. If None, sets discard threshold to 0.05 for 'ichol' and 'ilu' preconditioners, but uses default values for all other parameters. Options for the Jacobi, iCholesky, and V-Cycle preconditioners can be found here: https://pymatting.github.io/pymatting.preconditioner.html. Options for the iLU preconditioner can be found here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spilu.html.
        ''', None),
    'tr_method_sigma_2': ('hutchinson', 'set', ('hutchinson', 'hutch++'),
        '''
            (default='hutchinson') Algorithm to use to approximate trace for sigma^2. Note that hutch++ should require 1/3 as many trace draws for equivalent approximation error.
        ''', None),
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

class FEEstimator:
    '''
    Uses multigrid and partialing out to solve two way fixed effect models. This includes AKM, the Andrews et al. homoskedastic correction, and the Kline et al. heteroskedastic correction.

    Arguments:
        adata (BipartiteDataFrame): long or collapsed long format labor data
        params (ParamsDict or None): dictionary of parameters for FE estimation. Run tw.fe_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.fe_params().
    '''

    def __init__(self, adata, params=None):
        # Start logger
        logger_init(self)
        # self.logger.info('initializing FEEstimator object')

        if params is None:
            params = fe_params()

        self.adata = adata

        self.params = params
        # Results dictionary
        self.res = {}
        # Summary results dictionary
        self.summary = {}

        ### Save some commonly used parameters as attributes ###
        ## All ##
        # Whether data is weighted
        self.weighted = (params['weighted'] and ('w' in adata.columns))
        # Progress bars
        self.no_pbars = (not params['progress_bars'])
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

        ## Check that 'ndraw_lev_he' is a multiple of 'ncore' ##
        if self.compute_he and (self.ndraw_lev_he % self.ncore != 0):
            raise ValueError(f"'ndraw_lev_he' (currently {self.ndraw_lev_he}) should be a multiple of 'ncore' (currently {self.ncore}).")
        self.batchsize_he = self.ndraw_lev_he // self.ncore


        ## Store some parameters in results dictionary ##
        self.res['cores'] = self.ncore
        self.res['ndraw_trace_ho'] = self.ndraw_trace_ho
        self.res['ndraw_trace_he'] = self.ndraw_trace_he
        self.res['ndraw_lev_he'] = self.ndraw_lev_he

        # self.logger.info('FEEstimator object initialized')

    # def __getstate__(self):
    #     '''
    #     Defines how the model is pickled.
    #     '''
    #     odict = {k: self.__dict__[k] for k in self.__dict__.keys() - {'ml'}}
    #     return odict

    # def __setstate__(self, d):
    #     '''
    #     Defines how the model is unpickled.

    #     Arguments:
    #         d (dict): attribute dictionary
    #     '''
    #     # Need to recreate the simple model and the search representation
    #     # Make d the attribute dictionary
    #     self.__dict__ = d
    #     self.Minv_solver = rss(self.Minv)

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
                    if not self.params['levfile']:
                        del self.sqrt_DpJ, self.sqrt_DpW

                del self.Q_var, self.Q_cov
                if aainv:
                    del self.AA_inv_A, self.AA_inv_B, self.AA_inv_C, self.AA_inv_D

                # Collect all results
                self._collect_res()

        # Clear attributes
        del self.worker_m, self.Y, self.J, self.DpJ, self.W, self.DpW, self.Dp, self.Dwinv, self.DwinvWtDpJ, self.Minv
        if not self.params['statsonly']:
            if self.params['solver'] == 'amg':
                del self.Minv_solver
            else:
                del self.preconditioner

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

        if self.weighted:
            ### Weighted ###
            ## Dp (weight) ##
            Dp = self.adata.loc[:, 'w'].to_numpy()

            ## Weighted J and W ##
            DpJ = DxSP(Dp, J)
            DpW = DxSP(Dp, W)

            if self.params['he'] and (not self.params['levfile']):
                Dp_sqrt = np.sqrt(Dp)
                sqrt_DpJ = DxSP(Dp_sqrt, J)
                sqrt_DpW = DxSP(Dp_sqrt, W)
        else:
            ## Unweighted ##
            ## Dp (weight) ##
            Dp = 1

            ## Weighted J and W ##
            DpJ = J
            DpW = W

            if self.params['he'] and (not self.params['levfile']):
                sqrt_DpJ = J
                sqrt_DpW = W

        ## Dwinv ##
        Dwinv = 1 / diag_of_sp_prod(W.T, DpW)

        ## Dwinv @ W.T @ Dp @ J ##
        WtDpJ = W.T @ DpJ
        DwinvWtDpJ = DxSP(Dwinv, WtDpJ.tocsc())

        ## Minv ##
        Minv = J.T @ DpJ - WtDpJ.T @ DwinvWtDpJ
        # # Force symmetry
        # Minv = (Minv + Minv.T) / 2

        ## Store matrices ##
        self.Y = self.adata.loc[:, 'y'].to_numpy()
        self.J, self.DpJ = J, DpJ
        self.W, self.DpW = W, DpW
        if self.params['he'] and (not self.params['levfile']):
            self.sqrt_DpJ = sqrt_DpJ
            self.sqrt_DpW = sqrt_DpW
        self.Dp = Dp
        self.Dwinv = Dwinv
        self.DwinvWtDpJ = DwinvWtDpJ
        self.Minv = Minv

        if not self.params['statsonly']:
            self.logger.info('preparing linear solver')

            solver = self.params['solver']
            if solver == 'amg':
                # Prepare AMG solver
                self.Minv_solver = rss(Minv)
            else:
                # Prepare SciPy linear solver
                pcdr = self.params['preconditioner']
                pcd_options = self.params['preconditioner_options']
                if pcdr is None:
                    self.preconditioner = None
                else:
                    pcdT_operator = None
                    if pcdr == 'jacobi':
                        pcd_operator = pcd.jacobi(Minv.tocsc()).precondition
                        if solver in ['bicg', 'qmr']:
                            # These solvers need the preconditioner for M.T
                            pcdT_operator = pcd.jacobi(Minv.T).precondition
                    elif pcdr == 'vcycle':
                        if pcd_options is None:
                            pcd_options = {}
                        # Always set 'direct_solve_size' = 0, otherwise it runs scipy.sparse.linalg.spsolve, which we don't want
                        pcd_options['direct_solve_size'] = 0
                        pcd_operator = pcd.vcycle(Minv.tocsc(), (Minv.shape[0], 1), **pcd_options).precondition
                        if solver in ['bicg', 'qmr']:
                            # These solvers need the preconditioner for M.T
                            pcdT_operator = pcd.vcycle(Minv.T, (Minv.shape[0], 1), **pcd_options).precondition
                    elif pcdr == 'ichol':
                        if pcd_options is None:
                            pcd_options = {'discard_threshold': 0.05}
                        pcd_operator = pcd.ichol(Minv.tocsc(), **pcd_options)
                        if solver in ['bicg', 'qmr']:
                            # These solvers need the preconditioner for M.T
                            pcdT_operator = pcd.ichol(Minv.T, **pcd_options)
                    elif pcdr == 'ilu':
                        if pcd_options is None:
                            pcd_options = {'drop_tol': 0.05}
                        pcd_operator = spilu(Minv.tocsc(), **pcd_options).solve
                        if solver in ['bicg', 'qmr']:
                            # These solvers need the preconditioner for M.T
                            pcdT_operator = spilu(Minv.T, **pcd_options).solve
                    self.preconditioner = LinearOperator(Minv.shape, matvec=pcd_operator, rmatvec=pcdT_operator)

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
        self.res['mover_quantiles'] = weighted_quantile(fm, ls, fi).tolist()
        self.res['size_quantiles'] = weighted_quantile(fi, ls, fi).tolist()
        # self.res['movers_per_firm'] = self.adata.loc[self.adata.loc[:, 'm'] > 0, :].groupby('j')['i'].nunique().mean()
        self.res['between_firm_var'] = weighted_var(fy, fi)
        self.res['var(y)'] = weighted_var(self.Y, self.Dp)
        self.logger.info(f"total variance: {self.res['var(y)']:2.4f}")

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

        self.psi_hat, self.alpha_hat = self._solve(self.Y, Dp2=True)

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
        # Add 0 for normalized firm
        psi_hat = np.append(self.psi_hat, 0)
        alpha_hat = self.alpha_hat

        # Attach columns
        self.adata.loc[:, 'psi_hat'] = psi_hat[self.adata.loc[:, 'j']]
        self.adata.loc[:, 'alpha_hat'] = alpha_hat[self.adata.loc[:, 'i']]

    def _estimate_sigma_2_fe(self):
        '''
        Estimate residuals and sigma^2 (variance of residuals) for plug-in (biased) FE model.
        '''
        Dp = self.Dp

        ## Estimate residuals ##
        self.E = self.Y - self._mult_A(self.psi_hat, self.alpha_hat, weighted=False)

        ## Estimate R^2 ##
        fe_rsq = 1 - (Dp * (self.E ** 2)).sum() / (Dp * (self.Y ** 2)).sum()
        self.logger.info(f'fixed effect R-square {fe_rsq:2.4f}')

        ## Estimate variance of residuals (DON'T DEMEAN) ##
        # NOTE: multiply by Dp, because each observation's variance is divided by Dp and we need to undo that
        self.sigma_2_pi = weighted_mean(Dp * (self.E ** 2), Dp)

        self.logger.info(f'[fe] variance of residuals {self.sigma_2_pi:2.4f}')

    def _construct_Q(self):
        '''
        Construct Q (variance/covariance) matrices, store Q classes as attributes, and generate related parameters.

        Returns:
            (tuple of dicts): (dict of Q variance parameters, dict of Q covariance parameters)
        '''
        Q_var = self.params['Q_var']
        Q_cov = self.params['Q_cov']
        if Q_var is None:
            Q_var = Q.VarPsi()
        if Q_cov is None:
            Q_cov = Q.CovPsiAlpha()

        Q_var = {Q_subvar.name(): Q_subvar for Q_subvar in to_list(Q_var)}
        Q_cov = {Q_subcov.name(): Q_subcov for Q_subcov in to_list(Q_cov)}

        # Store as attributes
        self.Q_var = Q_var
        self.Q_cov = Q_cov

        # Construct Q matrices
        Q_params = self.adata, self.J, self.W, self.Dp
        Q_vars = {var_name: Q_subvar._get_Q(*Q_params) for var_name, Q_subvar in Q_var.items()}
        Q_covs = {cov_name: (Q_subcov._get_Ql(*Q_params), Q_subcov._get_Qr(*Q_params)) for cov_name, Q_subcov in Q_cov.items()}
        return (Q_vars, Q_covs)

    def _estimate_fe(self, Q_params):
        '''
        Estimate plug-in (biased) FE model.

        Arguments:
            Q_params (tuple): (Q variance parameters, Q left covariance parameters, Q right covariance parameters)
        '''
        self.logger.info('starting plug-in estimation')
        self.logger.info('[fe]')

        Q_vars, Q_covs = Q_params
        psi, alpha = self.psi_hat, self.alpha_hat
        var_fe = {}
        cov_fe = {}

        ## Variances ##
        for var_name, Q_subvar in Q_vars.items():
            Q_var_matrix, Q_var_weights, Q_var_psialpha = Q_subvar

            var_fe[var_name] = weighted_var(self.Q_var[var_name]._Q_mult(Q_var_matrix, psi, alpha), Q_var_weights, dof=0)

            self.logger.info(f'{var_name}_fe={var_fe[var_name]:2.4f}')

        ## Covariances ##
        for cov_name, Q_subcov in Q_covs.items():
            Ql_cov, Qr_cov = Q_subcov
            Ql_cov_matrix, Ql_cov_weights, Ql_cov_psialpha = Ql_cov
            Qr_cov_matrix, Qr_cov_weights, Qr_cov_psialpha = Qr_cov

            cov_fe[cov_name] = weighted_cov(self.Q_cov[cov_name]._Ql_mult(Ql_cov_matrix, psi, alpha), self.Q_cov[cov_name]._Qr_mult(Qr_cov_matrix, psi, alpha), Ql_cov_weights, Qr_cov_weights, dof=0)

            self.logger.info(f"{cov_name}_fe={cov_fe[cov_name]:2.4f}")

        self.var_fe = var_fe
        self.cov_fe = cov_fe

    def _estimate_exact_trace_sigma_2(self):
        '''
        Estimate analytical trace of sigma^2.
        '''
        self.logger.info(f'[sigma^2] [analytical trace]')

        # Unpack attributes
        J, W = self.J, self.W
        A, B, C, D = self.AA_inv_A, self.AA_inv_B, self.AA_inv_C, self.AA_inv_D

        ## Compute Tr[A @ (A'D_pA)^{-1} @ A'] ##
        # np.trace((J @ A + W @ C) @ J.T + (J @ B + W @ D) @ W.T)
        self.tr_sigma_ho_all = np.sum(diag_of_sp_prod(J, (J @ A + W @ C).T) + diag_of_sp_prod(W, (J @ B + W @ D).T))

        self.logger.debug(f'[sigma^2] [analytical traces] done')

    def _estimate_approximate_trace_sigma_2(self, rng=None):
        '''
        Estimate trace approximation of sigma^2.

        Arguments:
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        n_draws = self.ndraw_trace_sigma_2
        self.logger.info(f'[sigma^2] [approximate trace] ndraws={n_draws}')

        # Begin trace approximation
        if self.params['tr_method_sigma_2'] == 'hutchinson':
            self.tr_sigma_ho_all = np.zeros(n_draws)
        elif self.params['tr_method_sigma_2'] == 'hutch++':
            Az_lst = []

        pbar = trange(n_draws, disable=self.no_pbars)
        pbar.set_description('sigma^2')
        for r in pbar:
            ## Compute Tr[A @ (A'D_pA)^{-1} @ A'] ##
            # Generate -1 or 1
            Z = 2 * rng.binomial(1, 0.5, self.nn) - 1

            # Compute A @ (A'D_pA)^{-1} @ A' @ Z
            Az = self._mult_A( # A @
                *self._solve(Z, Dp2=False), weighted=False # (A'D_pA)^{-1} @ A' @ Z
            )
            if self.params['tr_method_sigma_2'] == 'hutchinson':
                # Compute Z.T @ A @ (A'D_pA)^{-1} @ A' @ Z
                self.tr_sigma_ho_all[r] = Z.T @ Az
            elif self.params['tr_method_sigma_2'] == 'hutch++':
                # Store Az
                Az_lst.append(Az)

            self.logger.debug(f'[sigma^2] [approximate trace] step {r}/{n_draws} done')

        if self.params['tr_method_sigma_2'] == 'hutch++':
            ## Compute Hutch++ ##
            G = 2 * rng.binomial(1, 0.5, size=(self.nn, n_draws)) - 1
            Q = np.linalg.qr(np.stack(Az_lst, axis=1))[0]
            G = G - Q @ (Q.T @ G)
            Aq_lst = []
            Ag_lst = []
            pbar = trange(n_draws, disable=self.no_pbars)
            pbar.set_description('sigma^2 part 2')
            for r in pbar:
                ## (A) @ Q and (A) @ G ##
                # where (A) = A @ (A'D_pA)^{-1} @ A'
                AQ_r = self._mult_A( # A @
                    *self._solve(Q[:, r], Dp2=False), weighted=False # (A'D_pA)^{-1} @ A' @ Q[:, r]
                )
                AG_r = self._mult_A( # A @
                    *self._solve(G[:, r], Dp2=False), weighted=False # (A'D_pA)^{-1} @ A' @ G[:, r]
                )
                Aq_lst.append(AQ_r)
                Ag_lst.append(AG_r)
            ## tr[Q.T @ (A) @ Q] + (1 / n_draws) * tr[G.T @ (A) @ G] ##
            # where (A) = A @ (A'D_pA)^{-1} @ A'
            self.tr_sigma_ho_all = np.sum(diag_of_prod(Q.T, np.stack(Aq_lst, axis=1))) + (1 / n_draws) * np.sum(diag_of_prod(G.T, np.stack(Ag_lst, axis=1)))

    def _estimate_sigma_2_ho(self):
        '''
        Estimate sigma^2 (variance of residuals) for HO-corrected model.
        '''
        if self.weighted:
            # Must use unweighted sigma^2 for numerator (weighting will make the estimator biased)
            sigma_2_unweighted = np.mean(self.E ** 2)
            trace_approximation = np.mean(self.tr_sigma_ho_all)
            self.sigma_2_ho = (self.nn * sigma_2_unweighted) / (np.sum(1 / self.Dp) - trace_approximation)
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

        Q_vars, Q_covs = Q_params

        ## Prepare dictionaries of results ##
        tr_var_ho_all = {}
        tr_cov_ho_all = {}

        ## Compute Tr[Q @ (A'D_pA)^{-1}] ##
        right_dict = {
            'psi': (self.AA_inv_A, self.AA_inv_B),
            'alpha': (self.AA_inv_C, self.AA_inv_D)
        }
        left_dict = {
            'psi': 0,
            'alpha': 1
        }

        ## Trace correction - variances ##
        for var_name, Q_subvar in Q_vars.items():
            Q_var_matrix, Q_var_weights, Q_var_psialpha = Q_subvar

            if isinstance(Q_var_weights, (float, int)):
                Q_var_weights = np.ones(Q_var_matrix.shape[0])
            Q_var_weights_sqrt = np.sqrt(Q_var_weights)

            if Q_var_psialpha not in ['psi', 'alpha']:
                raise NotImplementedError(f'Exact HO correction is compatible with only Q matrices that interact solely with psi or alpha components, but the selected Q-variance interacts with {Q_var_psialpha!r}.')

            # Q = Ql.T @ U.T @ Ql_weight.T @ Qr_weight @ U @ Qr
            n = np.sum(Q_var_weights)
            # Construct U (demeaning matrix)
            U_dim = Q_var_matrix.shape[0]
            U = np.eye(U_dim) - (1 / n) * np.tile(Q_var_weights, (U_dim, 1))
            # Compute Q @ (A'D_pA)^{-1} using block matrix components
            Q_var = DxM(Q_var_weights_sqrt, U) @ Q_var_matrix
            Q_var = (1 / n) * Q_var.T @ Q_var
            # Compute trace by considering correct quadrant
            tr_var_ho_all[var_name] = np.sum(diag_of_prod(Q_var, right_dict[Q_var_psialpha][left_dict[Q_var_psialpha]]))

        ## Trace correction - covariances ##
        for cov_name, Q_subcov in Q_covs.items():
            Ql_cov, Qr_cov = Q_subcov
            Ql_cov_matrix, Ql_cov_weights, Ql_cov_psialpha = Ql_cov
            Qr_cov_matrix, Qr_cov_weights, Qr_cov_psialpha = Qr_cov

            if isinstance(Ql_cov_weights, (float, int)):
                Ql_cov_weights = np.ones(Ql_cov_matrix.shape[0])
            if isinstance(Qr_cov_weights, (float, int)):
                Qr_cov_weights = np.ones(Qr_cov_matrix.shape[0])
            Ql_cov_weights_sqrt = np.sqrt(Ql_cov_weights)
            Qr_cov_weights_sqrt = np.sqrt(Qr_cov_weights)
            Q_cov_weights = Ql_cov_weights_sqrt * Qr_cov_weights_sqrt

            if (Ql_cov_psialpha not in ['psi', 'alpha']) or (Qr_cov_psialpha not in ['psi', 'alpha']):
                raise NotImplementedError(f'Exact HO correction is compatible with only Q matrices that interact solely with psi or alpha components, but the selected Ql/Qr-covariances interact with {Ql_cov_psialpha!r}/{Qr_cov_psialpha!r}.')

            # Q = Ql.T @ Ul.T @ Ql_weight.T @ Qr_weight @ Ur @ Qr
            n = np.sum(Q_cov_weights)
            nl = np.sum(Ql_cov_weights)
            nr = np.sum(Qr_cov_weights)
            # Construct U (demeaning matrix)
            Ul_dim = Ql_cov_matrix.shape[0]
            Ur_dim = Qr_cov_matrix.shape[0]
            Ul = np.eye(Ul_dim) - (1 / nl) * np.tile(Ql_cov_weights, (Ul_dim, 1))
            Ur = np.eye(Ur_dim) - (1 / nr) * np.tile(Qr_cov_weights, (Ur_dim, 1))
            # Compute Q @ (A'D_pA)^{-1} using block matrix components
            Q_cov = (1 / n) * Ql_cov_matrix.T @ MxD(Ul.T, Ql_cov_weights_sqrt.T) @ DxM(Qr_cov_weights_sqrt, Ur) @ Qr_cov_matrix
            # Compute trace by considering correct quadrant
            tr_cov_ho_all[cov_name] = np.sum(diag_of_prod(Q_cov, right_dict[Qr_cov_psialpha][left_dict[Ql_cov_psialpha]]))

        self.tr_var_ho_all = tr_var_ho_all
        self.tr_cov_ho_all = tr_cov_ho_all

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

        Q_vars, Q_covs = Q_params

        ## Prepare vectors of results ##
        tr_var_ho_all = {}
        tr_cov_ho_all = {}
        # Variances #
        for var_name in Q_vars.keys():
            tr_var_ho_all[var_name] = np.zeros(self.ndraw_trace_ho)
        # Covariances #
        for cov_name in Q_covs.keys():
            tr_cov_ho_all[cov_name] = np.zeros(self.ndraw_trace_ho)

        pbar = trange(self.ndraw_trace_ho, disable=self.no_pbars)
        pbar.set_description('ho')
        for r in pbar:
            ## Compute Tr[Q @ (A'D_pA)^{-1}] ##
            # Generate -1 or 1
            Zpsi = 2 * rng.binomial(1, 0.5, self.nf - 1) - 1
            Zalpha = 2 * rng.binomial(1, 0.5, self.nw) - 1

            # Compute (A'D_pA)^{-1} @ Z
            psi1, alpha1 = self._mult_AAinv(Zpsi, Zalpha)

            ## Trace correction - variances ##
            for var_name, Q_subvar in Q_vars.items():
                Q_var_matrix, Q_var_weights, Q_var_psialpha = Q_subvar

                # Left term of Q matrix
                L_var = self.Q_var[var_name]._Q_mult(Q_var_matrix, Zpsi, Zalpha).T
                # Right term of Q matrix
                R_var = self.Q_var[var_name]._Q_mult(Q_var_matrix, psi1, alpha1)
                tr_var_ho_all[var_name][r] = weighted_cov(L_var, R_var, Q_var_weights, Q_var_weights)
            del L_var, R_var

            ## Trace correction - covariances ##
            for cov_name, Q_subcov in Q_covs.items():
                Ql_cov, Qr_cov = Q_subcov
                Ql_cov_matrix, Ql_cov_weights, Ql_cov_psialpha = Ql_cov
                Qr_cov_matrix, Qr_cov_weights, Qr_cov_psialpha = Qr_cov

                # Left term of Q matrix
                L_cov = self.Q_cov[cov_name]._Ql_mult(Ql_cov_matrix, Zpsi, Zalpha).T
                # Right term of Q matrix
                R_cov = self.Q_cov[cov_name]._Qr_mult(Qr_cov_matrix, psi1, alpha1)
                tr_cov_ho_all[cov_name][r] = weighted_cov(L_cov, R_cov, Ql_cov_weights, Qr_cov_weights)
            del L_cov, R_cov

            self.logger.debug(f'[ho] [approximate trace] step {r + 1}/{self.ndraw_trace_ho} done')

        self.tr_var_ho_all = tr_var_ho_all
        self.tr_cov_ho_all = tr_cov_ho_all

    def _estimate_exact_trace_he(self, Q_params, Sii):
        '''
        Estimate analytical trace of HE-corrected model.

        Arguments:
            Q_params (tuple): (Q variance parameters, Q left covariance parameters, Q right covariance parameters)
            Sii (NumPy Array): Sii (sigma^2) for heteroskedastic correction
        '''
        self.logger.info(f'[he] [analytical trace]')

        # Unpack attributes
        Q_vars, Q_covs = Q_params
        J, W = self.J, self.W
        sqrt_DpJ, sqrt_DpW = self.sqrt_DpJ, self.sqrt_DpW
        Dp = self.Dp
        A, B, C, D = self.AA_inv_A, self.AA_inv_B, self.AA_inv_C, self.AA_inv_D

        ## Prepare vectors of results ##
        tr_var_he_all = {}
        tr_cov_he_all = {}

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

        ## Trace correction - variances ##
        for var_name, Q_subvar in Q_vars.items():
            Q_var_matrix, Q_var_weights, Q_var_psialpha = Q_subvar

            if isinstance(Q_var_weights, (float, int)):
                Q_var_weights = np.ones(Q_var_matrix.shape[0])
            Q_var_weights_sqrt = np.sqrt(Q_var_weights)

            if Q_var_psialpha not in ['psi', 'alpha']:
                raise NotImplementedError(f'Exact HO correction is compatible with only Q matrices that interact solely with psi or alpha components, but the selected Q-variance interacts with {Q_var_psialpha!r}.')

            # Q = Ql.T @ U.T @ Ql_weight.T @ Qr_weight @ U @ Qr
            n = np.sum(Q_var_weights)
            # Construct U (demeaning matrix)
            U_dim = Q_var_matrix.shape[0]
            U = np.eye(U_dim) - (1 / n) * np.tile(Q_var_weights, (U_dim, 1))
            ## Compute Q @ (A'D_pA)^{-1} @ (D_pA)' @ Omega @ (D_pA) @ (A'D_pA)^{-1} using block matrix components ##
            # Construct Q
            Q_var = (Q_var_weights_sqrt * U.T).T @ Q_var_matrix
            Q_var = (1 / n) * Q_var.T @ Q_var
            # Construct AA_inv
            if Q_var_psialpha == 'psi':
                AA_inv = np.asarray(sqrt_DpJ @ A + sqrt_DpW @ C)
                AA_inv = AA_inv.T @ DxM(Sii, AA_inv)
            elif Q_var_psialpha == 'alpha':
                AA_inv = np.asarray(sqrt_DpJ @ B + sqrt_DpW @ D)
                AA_inv = AA_inv.T @ DxM(Sii, AA_inv)

            # Compute trace by considering correct quadrant
            tr_var_he_all[var_name] = np.sum(diag_of_prod(Q_var, AA_inv))

        ## Trace correction - covariances ##
        for cov_name, Q_subcov in Q_covs.items():
            Ql_cov, Qr_cov = Q_subcov
            Ql_cov_matrix, Ql_cov_weights, Ql_cov_psialpha = Ql_cov
            Qr_cov_matrix, Qr_cov_weights, Qr_cov_psialpha = Qr_cov

            if isinstance(Ql_cov_weights, (float, int)):
                Ql_cov_weights = np.ones(Ql_cov_matrix.shape[0])
            if isinstance(Qr_cov_weights, (float, int)):
                Qr_cov_weights = np.ones(Qr_cov_matrix.shape[0])
            Ql_cov_weights_sqrt = np.sqrt(Ql_cov_weights)
            Qr_cov_weights_sqrt = np.sqrt(Qr_cov_weights)
            Q_cov_weights = Ql_cov_weights_sqrt * Qr_cov_weights_sqrt

            if (Ql_cov_psialpha not in ['psi', 'alpha']) or (Qr_cov_psialpha not in ['psi', 'alpha']):
                raise NotImplementedError(f'Exact HO correction is compatible with only Q matrices that interact solely with psi or alpha components, but the selected Ql/Qr-covariances interact with {Ql_cov_psialpha!r}/{Qr_cov_psialpha!r}.')

            # Q = Ql.T @ Ul.T @ Ql_weight.T @ Qr_weight @ Ur @ Qr
            n = np.sum(Q_cov_weights)
            nl = np.sum(Ql_cov_weights)
            nr = np.sum(Qr_cov_weights)
            # Construct U (demeaning matrix)
            Ul_dim = Ql_cov_matrix.shape[0]
            Ur_dim = Qr_cov_matrix.shape[0]
            Ul = np.eye(Ul_dim) - (1 / nl) * np.tile(Ql_cov_weights, (Ul_dim, 1))
            Ur = np.eye(Ur_dim) - (1 / nr) * np.tile(Qr_cov_weights, (Ur_dim, 1))
            ## Compute Q @ (A'D_pA)^{-1} @ (D_pA)' @ Omega @ (D_pA) @ (A'D_pA)^{-1} using block matrix components ##
            # Construct Q
            Q_cov = (1 / n) * Ql_cov_matrix.T @ MxD(Ul.T, Ql_cov_weights_sqrt.T) @ DxM(Qr_cov_weights_sqrt, Ur) @ Qr_cov_matrix
            # Construct AA_inv
            if Qr_cov_psialpha == 'psi':
                if Ql_cov_psialpha == 'psi':
                    # if Q_var_psialpha != 'psi':
                    #     # No need to recompute if var is the same
                    AA_inv = np.asarray(sqrt_DpJ @ A + sqrt_DpW @ C)
                    AA_inv = AA_inv.T @ DxM(Sii, AA_inv)
                elif Ql_cov_psialpha == 'alpha':
                    AA_inv = (A @ J.T + B @ W.T) @ DxM(Dp * Sii, np.asarray(J @ B + W @ D))
            elif Qr_cov_psialpha == 'alpha':
                if Ql_cov_psialpha == 'psi':
                    AA_inv = (C @ J.T + D @ W.T) @ DxM(Dp * Sii, np.asarray(J @ A + W @ C))
                elif Ql_cov_psialpha == 'alpha':
                    # if Q_var_psialpha != 'alpha':
                    #     # No need to recompute if var is the same
                    AA_inv = np.asarray(sqrt_DpJ @ B + sqrt_DpW @ D)
                    AA_inv = AA_inv.T @ DxM(Sii, AA_inv)
            # Compute trace by considering correct quadrant
            tr_cov_he_all[cov_name] = np.sum(diag_of_prod(Q_cov, AA_inv))

        self.tr_var_he_all = tr_var_he_all
        self.tr_cov_he_all = tr_cov_he_all

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

        Q_vars, Q_covs = Q_params

        ## Prepare vectors of results ##
        tr_var_he_all = {}
        tr_cov_he_all = {}
        # Variances #
        for var_name in Q_vars.keys():
            tr_var_he_all[var_name] = np.zeros(self.ndraw_trace_he)
        # Covariances #
        for cov_name in Q_covs.keys():
            tr_cov_he_all[cov_name] = np.zeros(self.ndraw_trace_he)

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
                            Zpsi, Zalpha
                        ), weighted=True
                    ), weighted=True
                )
            )

            ## Trace correction - variances ##
            for var_name, Q_subvar in Q_vars.items():
                Q_var_matrix, Q_var_weights, Q_var_psialpha = Q_subvar

                # Left term of Q matrix
                L_var = self.Q_var[var_name]._Q_mult(Q_var_matrix, Zpsi, Zalpha).T
                # Right term of Q matrix
                R_var = self.Q_var[var_name]._Q_mult(Q_var_matrix, psi1, alpha1)
                tr_var_he_all[var_name][r] = weighted_cov(L_var, R_var, Q_var_weights, Q_var_weights)
            del L_var, R_var

            ## Trace correction - covariances ##
            for cov_name, Q_subcov in Q_covs.items():
                Ql_cov, Qr_cov = Q_subcov
                Ql_cov_matrix, Ql_cov_weights, Ql_cov_psialpha = Ql_cov
                Qr_cov_matrix, Qr_cov_weights, Qr_cov_psialpha = Qr_cov

                # Left term of Q matrix
                L_cov = self.Q_cov[cov_name]._Ql_mult(Ql_cov_matrix, Zpsi, Zalpha).T
                # Right term of Q matrix
                R_cov = self.Q_cov[cov_name]._Qr_mult(Qr_cov_matrix, psi1, alpha1)
                tr_cov_he_all[cov_name][r] = weighted_cov(L_cov, R_cov, Ql_cov_weights, Qr_cov_weights)
            del L_cov, R_cov

            self.logger.debug(f'[he] [approximate trace] step {r + 1}/{self.ndraw_trace_he} done')

        self.tr_var_he_all = tr_var_he_all
        self.tr_cov_he_all = tr_cov_he_all

    def _collect_res(self):
        '''
        Collect all results.
        '''
        # Already computed, this just reorders the dictionary
        self.res['var(y)'] = self.res['var(y)']

        # Names of variance and covariances
        var_names = sorted(self.var_fe.keys())
        cov_names = sorted(self.cov_fe.keys())

        ## FE results ##
        # Plug-in sigma^2
        self.res['var(eps)_fe'] = self.sigma_2_pi
        # Plug-in variances
        self.logger.info('[fe] VARIANCES')
        for var_name in var_names:
            self.res[f'{var_name}_fe'] = self.var_fe[var_name]
            self.logger.info(f'{var_name}_fe={self.var_fe[var_name]:2.4f}')
        # Plug-in covariances
        self.logger.info('[fe] COVARIANCES')
        for cov_name in cov_names:
            self.res[f'{cov_name}_fe'] = self.cov_fe[cov_name]
            self.logger.info(f'{cov_name}_fe={self.cov_fe[cov_name]:2.4f}')

        ## Homoskedastic results ##
        if self.compute_ho:
            # Bias-corrected sigma^2
            self.res['var(eps)_ho'] = self.sigma_2_ho
            # Trace approximation: variances
            self.logger.info('[ho] VARIANCE TRACES')
            for var_name in var_names:
                self.res[f'tr_{var_name}_ho'] = np.mean(self.tr_var_ho_all[var_name])
                self.res[f'tr_{var_name}_ho_sd'] = np.std(self.tr_var_ho_all[var_name])
                self.logger.info(f"tr_{var_name}_ho={self.res[f'tr_{var_name}_ho']:2.4f} (sd={self.res[f'tr_{var_name}_ho_sd']:2.4e})")
            # Trace approximation: covariances
            self.logger.info('[ho] COVARIANCE TRACES')
            for cov_name in cov_names:
                self.res[f'tr_{cov_name}_ho'] = np.mean(self.tr_cov_ho_all[cov_name])
                self.res[f'tr_{cov_name}_ho_sd'] = np.std(self.tr_cov_ho_all[cov_name])
                self.logger.info(f"tr_{cov_name}_ho={self.res[f'tr_{cov_name}_ho']:2.4f} (sd={self.res[f'tr_{cov_name}_ho_sd']:2.4e})")
            # Bias-corrected variances
            self.logger.info('[ho] VARIANCES')
            for var_name in var_names:
                self.res[f'{var_name}_ho'] = self.var_fe[var_name] - self.sigma_2_ho * self.res[f'tr_{var_name}_ho']
                self.logger.info(f"{var_name}_ho={self.res[f'{var_name}_ho']:2.4f}")
            # Bias-corrected covariances
            self.logger.info('[ho] COVARIANCES')
            for cov_name in cov_names:
                self.res[f'{cov_name}_ho'] = self.cov_fe[cov_name] - self.sigma_2_ho * self.res[f'tr_{cov_name}_ho']
                self.logger.info(f"{cov_name}_ho={self.res[f'{cov_name}_ho']:2.4f}")

        ## Heteroskedastic results ##
        if self.compute_he:
            ## Already computed, this just reorders the dictionary ##
            # Bias-corrected sigma^2
            self.res['var(eps)_he'] = self.res['var(eps)_he']
            self.res['min_lev'] = self.res['min_lev']
            self.res['max_lev'] = self.res['max_lev']
            ## New results ##
            # Trace approximation: variances
            self.logger.info('[he] VARIANCE TRACES')
            for var_name in var_names:
                self.res[f'tr_{var_name}_he'] = np.mean(self.tr_var_he_all[var_name])
                self.res[f'tr_{var_name}_he_sd'] = np.std(self.tr_var_he_all[var_name])
                self.logger.info(f"tr_{var_name}_he={self.res[f'tr_{var_name}_he']:2.4f} (sd={self.res[f'tr_{var_name}_he_sd']:2.4e})")
            # Trace approximation: covariances
            self.logger.info('[he] COVARIANCE TRACES')
            for cov_name in cov_names:
                self.res[f'tr_{cov_name}_he'] = np.mean(self.tr_cov_he_all[cov_name])
                self.res[f'tr_{cov_name}_he_sd'] = np.std(self.tr_cov_he_all[cov_name])
                self.logger.info(f"tr_{cov_name}_he={self.res[f'tr_{cov_name}_he']:2.4f} (sd={self.res[f'tr_{cov_name}_he_sd']:2.4e})")
            # Bias-corrected variances
            self.logger.info('[he] VARIANCES')
            for var_name in var_names:
                self.res[f'{var_name}_he'] = self.var_fe[var_name] - self.res[f'tr_{var_name}_he']
                self.logger.info(f"{var_name}_he={self.res[f'{var_name}_he']:2.4f}")
            # Bias-corrected covariances
            self.logger.info('[he] COVARIANCES')
            for cov_name in cov_names:
                self.res[f'{cov_name}_he'] = self.cov_fe[cov_name] - self.res[f'tr_{cov_name}_he']
                self.logger.info(f"{cov_name}_he={self.res[f'{cov_name}_he']:2.4f}")

        ### Summary ###
        ## General ##
        self.summary['var(y)'] = self.res['var(y)']
        self.summary['var(eps)_fe'] = self.res['var(eps)_fe']
        if self.compute_ho:
            # HO #
            self.summary['var(eps)_ho'] = self.res['var(eps)_ho']
        if self.compute_he:
            # HE #
            self.summary['var(eps)_he'] = self.res['var(eps)_he']

        ## Variances ##
        for var_name in var_names:
            # FE #
            self.summary[f'{var_name}_fe'] = self.res[f'{var_name}_fe']
            if self.compute_ho:
                # HO #
                self.summary[f'{var_name}_ho'] = self.res[f'{var_name}_ho']
            if self.compute_he:
                # HE #
                self.summary[f'{var_name}_he'] = self.res[f'{var_name}_he']

        ## Covariances ##
        for cov_name in cov_names:
            # FE #
            self.summary[f'{cov_name}_fe'] = self.res[f'{cov_name}_fe']
            if self.compute_ho:
                # HO #
                self.summary[f'{cov_name}_ho'] = self.res[f'{cov_name}_ho']
            if self.compute_he:
                # HE #
                self.summary[f'{cov_name}_he'] = self.res[f'{cov_name}_he']

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

    def _construct_AAinv_components_full(self):
        '''
        Construct (A' @ Dp @ A)^{-1} block matrix components by explicitly computing M.
        '''
        # Define variables
        J = self.J
        DpW = self.DpW
        M = np.linalg.inv(self.Minv.todense())
        Dwinv = self.Dwinv

        # Construct blocks
        self.AA_inv_A = M
        self.AA_inv_B = - M @ J.T @ SPxD(DpW, Dwinv)
        self.AA_inv_C = self.AA_inv_B.T
        self.AA_inv_D = Dwinv + self.AA_inv_C @ self.Minv @ self.AA_inv_B

    def _construct_AAinv_components_partial(self):
        '''
        Construct (A' @ Dp @ A)^{-1} block matrix components without explicitly computing M. Use this for computing a small number of individual Pii analytically.
        '''
        # Define variables
        J = self.J
        DpW = self.DpW
        Dwinv = self.Dwinv

        # Construct blocks
        self.AA_inv_A = None
        self.AA_inv_B = - J.T @ SPxD(DpW, Dwinv)
        self.AA_inv_C = self.AA_inv_B.T
        self.AA_inv_D = None

    def _mult_A(self, psi, alpha, weighted=False):
        '''
        Compute Dp @ A @ [psi' alpha']', where A = [J W] (J is firm indicators and W is worker indicators) and Dp gives weights.

        Arguments:
            psi (NumPy Array): firm part of vector to multiply
            alpha (NumPy Array): worker part of vector to multiply
            weighted (bool or str): if True, include weights; if 'sqrt', use square root of weights

        Returns:
            (NumPy Array): vector result of Dp @ A @ [psi' alpha']'
        '''
        if weighted == 'sqrt':
            J, W = self.sqrt_DpJ, self.sqrt_DpW
        else:
            J, W = {
                True: (self.DpJ, self.DpW),
                False: (self.J, self.W)
            }[weighted]

        return J @ psi + W @ alpha

    def _mult_Atranspose(self, v, weighted=True):
        '''
        Compute A' @ Dp @ v, where A = [J W] (J is firm indicators and W is worker indicators) and Dp gives weights.

        Arguments:
            v (NumPy Array): vector to multiply
            weighted (bool or str): if True, include weights; if 'sqrt', use square root of weights

        Returns:
            (tuple of NumPy Arrays): (firm part of vector result, worker part of vector result) where vector is result of Dp @ A @ [psi' alpha']'
        '''
        if weighted == 'sqrt':
            J, W = self.sqrt_DpJ, self.sqrt_DpW
        else:
            J, W = {
                True: (self.DpJ, self.DpW),
                False: (self.J, self.W)
            }[weighted]

        return J.T @ v, W.T @ v

    def _mult_AAinv(self, psi, alpha):
        '''
        Compute (A' @ Dp @ A)^{-1} @ [psi' alpha']', where A = [J W] (J is firm indicators and W is worker indicators) and Dp gives weights.

        Arguments:
            psi (NumPy Array): firm part of vector to multiply
            alpha (NumPy Array): worker part of vector to multiply

        Returns:
            (tuple of NumPy Arrays): (firm part of vector result, worker part of vector result) where vector is result of (A' @ Dp @ A)^{-1} @ [psi' alpha']'
        '''
        start = timer()

        solver_name = self.params['solver']
        tol = self.params['solver_tol']
        if solver_name == 'amg':
            # Use AMG solver
            psi_out = self.Minv_solver.solve(psi - self.DwinvWtDpJ.T @ alpha, tol=tol)
        else:
            # Use SciPy sparse iterative solver
            solver = solver_dict[solver_name]
            if solver_name == 'minres':
                psi_out = solver(self.Minv, psi - self.DwinvWtDpJ.T @ alpha, M=self.preconditioner, tol=tol)[0]
            elif solver_name == 'qmr':
                psi_out = solver(self.Minv, psi - self.DwinvWtDpJ.T @ alpha, tol=tol, atol=0)[0]
            else:
                psi_out = solver(self.Minv, psi - self.DwinvWtDpJ.T @ alpha, M=self.preconditioner, tol=tol, atol=0)[0]
        self.last_invert_time = timer() - start

        alpha_out = self.Dwinv * alpha - self.DwinvWtDpJ @ psi_out

        return psi_out, alpha_out

    def _solve(self, Y, Dp2=True):
        '''
        Compute (A' @ Dp1 @ A)^{-1} @ A' @ Dp2 @ Y, the least squares estimate of Y = A @ [psi_hat' alpha_hat']', where A = [J W] (J is firm indicators and W is worker indicators) and Dp gives weights.

        Arguments:
            Y (NumPy Array): outcome vector
            Dp2 (bool or str): if True, include weight Dp2; if 'sqrt', use square root of Dp2

        Returns:
            (tuple of NumPy Arrays): (firm part of vector result, worker part of vector result) where vector is result of (A' @ Dp1 @ A)^{-1} @ A' @ Dp2 @ Y
        '''
        # This gives A' * Dp2 * Y
        J_transpose_Y, W_transpose_Y = self._mult_Atranspose(Y, weighted=Dp2)
        # This gives (A' * Dp1 * A)^{-1} * A' * Dp2 * Y
        psi_hat, alpha_hat = self._mult_AAinv(J_transpose_Y, W_transpose_Y)

        return psi_hat, alpha_hat

    def _proj(self, Y, Dp0=False, Dp2=True):
        '''
        Compute Dp0 @ A @ (A' @ Dp1 @ A)^{-1} @ A' @ Dp2 @ Y (i.e. projects Y onto A space), where A = [J W] (J is firm indicators and W is worker indicators) and Dp gives weights.

        Arguments:
            Y (NumPy Array): outcome vector
            Dp0 (bool or str): if True, include weight Dp0; if 'sqrt', use square root of Dp0
            Dp2 (bool or str): if True, include weight Dp2; if 'sqrt', use square root of Dp2

        Returns:
            (NumPy Array): vector result of Dp0 @ A @ (A' @ Dp1 @ A)^{-1} @ A' @ Dp2 @ Y
        '''
        return self._mult_A(*self._solve(Y, Dp2), Dp0)

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
            solver_name = self.params['solver']
            tol = self.params['solver_tol']
            if solver_name == 'amg':
                # Use AMG solver
                M_DpJ_i = self.Minv_solver.solve(DpJ_i)
                M_B = self.Minv_solver.solve(self.AA_inv_B @ DpW_i)
            else:
                # Use SciPy sparse iterative solver
                solver = solver_dict[solver_name]
                if solver_name == 'minres':
                    M_DpJ_i = solver(self.Minv, DpJ_i, tol=tol)[0]
                    M_B = solver(self.Minv, self.AA_inv_B @ DpW_i, tol=tol)[0]
                else:
                    M_DpJ_i = solver(self.Minv, DpJ_i, tol=tol, atol=0)[0]
                    M_B = solver(self.Minv, self.AA_inv_B @ DpW_i, tol=tol, atol=0)[0]

            # Construct blocks
            A = DpJ_i @ M_DpJ_i
            B = DpJ_i @ M_B
            C = DpW_i @ self.AA_inv_C @ M_DpJ_i
            D = DpW_i @ (self.Dwinv @ DpW_i + self.AA_inv_C @ M_B)

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
        DpJ = self.sqrt_DpJ
        DpW = self.sqrt_DpW

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
        ncore = self.ncore

        self.logger.info(f"[he] [approximate pii] ndraw_lev_he={self.ndraw_lev_he}, using {ncore} core(s)")

        if ncore > 1:
            ## Multiprocessing ##
            # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
            seeds = rng.bit_generator._seed_seq.spawn(ncore)
            with Pool(processes=ncore) as pool:
                pbar2 = tqdm([(self.batchsize_he, np.random.default_rng(seed)) for seed in seeds], total=ncore, disable=self.no_pbars)
                pbar2.set_description('leverages batch')
                V = pool.starmap(self._leverage_approx, pbar2)
                del pbar2

            # Extract results
            Pii_all = [subV[0] for subV in V]
            Pii_sq_all = [subV[1] for subV in V]
            Mii_all = [subV[2] for subV in V]
            Mii_sq_all = [subV[3] for subV in V]
            Pii_Mii_all = [subV[4] for subV in V]

            # Take mean over draws
            Pii = sum(Pii_all) / ncore
            Pii_sq = sum(Pii_sq_all) / ncore
            Mii = sum(Mii_all) / ncore
            Mii_sq = sum(Mii_sq_all) / ncore
            Pii_Mii = sum(Pii_Mii_all) / ncore
        else:
            # Single core
            Pii, Pii_sq, Mii, Mii_sq, Pii_Mii = self._leverage_approx(self.ndraw_lev_he, rng)

        # Normalize Pii
        if (not self.weighted) and (self.params['Sii_stayers'] == 'upper_bound'):
            Pii = Pii / (Pii + Mii)
            self.res['min_lev'] = Pii.min()
            self.res['max_lev'] = Pii.max()
        else:
            Pii[worker_m] = Pii[worker_m] / (Pii[worker_m] + Mii[worker_m])
            self.res['min_lev'] = Pii[worker_m].min()
            self.res['max_lev'] = Pii[worker_m].max()

        if self.res['max_lev'] >= 1:
            if self.params['Sii_stayers'] == 'upper_bound':
                if self.weighted:
                    ### Weighted ###
                    Pii_s = 1 / self.adata.loc[~worker_m, ['i', 'w']].groupby('i', sort=False)['w'].transform('sum').to_numpy()
                    if np.any(Pii_s == 1):
                        raise ValueError("At least one stayer has observation-weight 1, which prevents computing Pii and Sii for those stayers. Please set 'drop_single_stayers'=True during data cleaning to avoid this error.")
                else:
                    ### Unweighted ###
                    if np.any(self.adata.loc[~worker_m, ['i', 'j']].groupby('i', sort=False)['j'].transform('size').to_numpy() == 1):
                        raise ValueError("At least one stayer has only one observation, which prevents computing Pii and Sii for those stayers. Please set 'drop_single_stayers'=True during data cleaning to avoid this error.")

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
        w = self.Dp
        if isinstance(w, (float, int)):
            # FIXME this is a temporary workaround
            w = np.ones(self.nn)

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
        self.res['var(eps)_he'] = np.average(w * Sii, weights=w)

        self.logger.info(f"[he] variance of residuals {self.res['var(eps)_he']:2.4f}")

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
