'''
Class for running Monte Carlo estimations on simulated bipartite networks
'''
from tqdm.auto import tqdm, trange
import warnings
try:
    from multiprocess import Pool
except ImportError:
    from multiprocessing import Pool
import numpy as np
from matplotlib import pyplot as plt
import bipartitepandas as bpd
import pytwoway as tw

class MonteCarlo:
    '''
    Class of MonteCarlo, where MonteCarlo runs a Monte Carlo estimation by simulating bipartite networks of firms and workers.

    Arguments:
        sim_params (ParamsDict or None): dictionary of parameters for simulating data. Run bpd.sim_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.sim_params().
        fe_params (ParamsDict or None): dictionary of parameters for FE estimation. Run tw.fe_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.fe_params().
        cre_params (ParamsDict or None): dictionary of parameters for CRE estimation. Run tw.cre_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.cre_params().
        estimate_bs (bool): if True, estimate Borovickova-Shimer model
        cluster_params (ParamsDict or None): dictionary of parameters for clustering in CRE estimation. Run bpd.cluster_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.cluster_params().
        clean_params (ParamsDict or None): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.clean_params().
        collapse (str or None): if None, run estimators on full dataset; if 'spell', run estimators on data collapsed at the worker-firm spell level; if 'spell', run estimators on data collapsed at the worker-firm match level
        move_to_worker (bool): if True, each move is treated as a new worker
        log (bool): if True, will create log file(s)
    '''

    def __init__(self, sim_params=None, fe_params=None, cre_params=None, estimate_bs=False, cluster_params=None, clean_params=None, collapse=None, move_to_worker=False, log=False):
        # Start logger
        # logger_init(self)
        # self.logger.info('initializing MonteCarlo object')

        if sim_params is None:
            sim_params = bpd.sim_params()
        if fe_params is None:
            fe_params = tw.fe_params()
        if cre_params is None:
            cre_params = tw.cre_params()
        if cluster_params is None:
            cluster_params = bpd.cluster_params()
        if clean_params is None:
            clean_params = bpd.clean_params()

        ## Save attributes ##
        # Parameter dictionaries
        self.fe_params = fe_params.copy()
        self.cre_params = cre_params.copy()
        self.cluster_params = cluster_params.copy()
        self.clean_params = clean_params.copy()
        # Other attributes
        self.sim_network = bpd.SimBipartite(sim_params)
        self.estimate_bs = estimate_bs
        self.collapse = collapse
        self.move_to_worker = move_to_worker
        self.log = log

        ## Update parameter dictionaries ##
        # FE params
        self.fe_params['ho'] = True
        self.fe_params['he'] = True
        self.fe_params['progress_bars'] = False
        self.fe_params['verbose'] = False
        # Clean parameters
        self.clean_params['collapse_at_connectedness_measure'] = True
        self.clean_params['drop_single_stayers'] = True
        self.clean_params['is_sorted'] = True
        self.clean_params['force'] = True
        self.clean_params['verbose'] = False
        self.clean_params['copy'] = False
        if collapse is None:
            # FIXME at the moment, CRE requires collapsed data
            warnings.warn("CRE currently requires data that is collapsed at the spell or match level; to avoid an error, collapse=None changed to collapse='spell'.")
            self.clean_params['connectedness'] = 'leave_out_spell'
            # self.clean_params['connectedness'] = 'leave_out_observation'
        elif collapse == 'spell':
            self.clean_params['connectedness'] = 'leave_out_spell'
        elif collapse == 'match':
            self.clean_params['connectedness'] = 'leave_out_match'
        else:
            raise ValueError(f"`collapse` must one of None, 'spell', or 'match', but input specifies invalid input {collapse!r}.")
        # Cluster parameters
        self.cluster_params['clean_params'] = self.clean_params.copy()
        self.cluster_params['is_sorted'] = True
        self.cluster_params['copy'] = False

        # No results yet
        self.res = None

        # self.logger.info('MonteCarlo object initialized')

    # Cannot include two underscores because isn't compatible with starmap for multiprocessing
    # Source: https://stackoverflow.com/questions/27054963/python-attribute-error-object-has-no-attribute
    def _monte_carlo_interior(self, rng=None):
        '''
        Run Monte Carlo simulations of two way fixed effect models to see the distribution of the true vs. estimated variance of psi and covariance between psi and alpha. This is the interior function to monte_carlo.

        Arguments:
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (dict): true, plug-in FE, FE-HO, FE-HE, CRE, and Borovickova-Shimer estimates for var(psi) and cov(psi, alpha), where psi gives firm effects and alpha gives worker effects
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        nk, nl = self.sim_network.params.get_multiple(('nk', 'nl'))

        ## Simulate data ##
        sim_data = self.sim_network.simulate(rng)

        ## Compute true sample variance of psi and covariance of psi and alpha ##
        var_psi = np.var(sim_data.loc[:, 'psi'].to_numpy(), ddof=0)
        cov_psi_alpha = np.cov(sim_data.loc[:, 'psi'].to_numpy(), sim_data.loc[:, 'alpha'].to_numpy(), ddof=0)[0, 1]

        ## Convert into BipartiteDataFrame ##
        sim_data = bpd.BipartiteLong(sim_data.loc[:, ['i', 'j', 'y', 't']], log=self.log)
        if self.move_to_worker:
            ## Set moves to worker ids ##
            sim_data = sim_data.to_eventstudy(move_to_worker=True, is_sorted=True, copy=False).to_long(is_sorted=True, copy=False)

        ## Clean data ##
        sim_data = sim_data.clean(self.clean_params)

        ## Estimate FE model ##
        fe_estimator = tw.FEEstimator(sim_data, params=self.fe_params)
        fe_estimator.fit(rng)
        # Save results
        fe_res = fe_estimator.res

        ## Estimate CRE model ##
        # Cluster
        sim_data = sim_data.cluster(self.cluster_params, rng=rng)
        # Estimate
        cre_estimator = tw.CREEstimator(sim_data.to_eventstudy(move_to_worker=False, is_sorted=True, copy=False).get_cs(copy=False), params=self.cre_params)
        cre_estimator.fit(rng)
        # Save results
        cre_res = cre_estimator.res

        res = {
            'var(psi)_true': var_psi,
            'var(psi)_fe': fe_res['var(psi)_fe'],
            'var(psi)_ho': fe_res['var(psi)_ho'],
            'var(psi)_he': fe_res['var(psi)_he'],
            'var(psi)_cre': cre_res['tot_var'],
            'cov(psi, alpha)_true': cov_psi_alpha,
            'cov(psi, alpha)_fe': fe_res['cov(psi, alpha)_fe'],
            'cov(psi, alpha)_ho': fe_res['cov(psi, alpha)_ho'],
            'cov(psi, alpha)_he': fe_res['cov(psi, alpha)_he'],
            'cov(psi, alpha)_cre': cre_res['tot_cov']
        }

        if self.estimate_bs:
            ## Estimate Borovickova-Shimer model ##
            # Make sure there are no returners and that all workers and firms have at least 2 observations

            # Clean parameters
            clean_params_1 = self.clean_params.copy()
            clean_params_2 = self.clean_params.copy()
            clean_params_1['connectedness'] = None
            clean_params_1['drop_returns'] = 'returns'
            clean_params_2['connectedness'] = None

            # Clean
            sim_data = sim_data.clean(clean_params_1).min_joint_obs_frame(is_sorted=True, copy=False).clean(clean_params_2)

            ### Estimate ###
            bs_estimator = tw.BSEstimator()

            ## Standard ##
            bs_estimator.fit(sim_data, alternative_estimator=False)
            # Save results
            bs_res = bs_estimator.res

            res['var(psi)_bs1'] = bs_res['var(mu)']
            res['cov(psi, alpha)_bs1'] = bs_res['cov(lambda, mu)']

            ## Alternative ##
            bs_estimator.fit(sim_data, alternative_estimator=True)
            # Save results
            bs_res = bs_estimator.res

            res['var(psi)_bs2'] = bs_res['var(mu)']
            res['cov(psi, alpha)_bs2'] = bs_res['cov(lambda, mu)']

        return res

    def monte_carlo(self, N=10, ncore=1, rng=None):
        '''
        Run Monte Carlo simulations of two way fixed effect models to see the distribution of the true vs. estimated variance of psi and covariance between psi and alpha. Saves the following results in the dictionary self.res: true, plug-in FE, FE-HO, FE-HE, CRE, and Borovickova-Shimer estimates for var(psi) and cov(psi, alpha), where psi gives firm effects and alpha gives worker effects.

        Arguments:
            N (int): number of simulations
            ncore (int): number of cores to use
            rng (np.random.Generator or None): NumPy random number generator. This overrides the random number generators for simulating, FE, and CRE. None is equivalent to np.random.default_rng(None).
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Simulate networks
        if ncore > 1:
            ## Multiprocessing
            # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
            seeds = rng.bit_generator._seed_seq.spawn(N)
            with Pool(processes=ncore) as pool:
                # Multiprocessing tqdm source: https://stackoverflow.com/a/45276885/17333120
                all_res = list(tqdm(pool.imap(self._monte_carlo_interior, [np.random.default_rng(seed) for seed in seeds]), total=N))
                # all_res = pool.starmap(self._monte_carlo_interior, tqdm([(np.random.default_rng(seed)) for seed in seeds], total=N))
        else:
            # Single core
            all_res = [self._monte_carlo_interior(rng) for _ in trange(N)]

        # Extract results
        estimators = [
            'var(psi)_true', 'var(psi)_fe', 'var(psi)_ho',
            'var(psi)_he', 'var(psi)_cre',
            'cov(psi, alpha)_true', 'cov(psi, alpha)_fe', 'cov(psi, alpha)_ho',
            'cov(psi, alpha)_he', 'cov(psi, alpha)_cre'
        ]
        if self.estimate_bs:
            estimators += ['var(psi)_bs1', 'var(psi)_bs2', 'cov(psi, alpha)_bs1', 'cov(psi, alpha)_bs2']
        res = {estimator: np.zeros(N) for estimator in estimators}

        for i, res_i in enumerate(all_res):
            for estimator in estimators:
                res[estimator][i] = res_i[estimator]

        self.res = res

    def hist(self, fe=True, ho=True, he=True, cre=True, bs1=True, bs2=True, density=False):
        '''
        Plot histogram of how Monte Carlo simulation results differ from truth.

        Arguments:
            fe (bool): if True, plot FE results
            ho (bool): if True, plot FE-HO results
            he (bool): if True, plot FE-HE results
            cre (bool): if True, plot CRE results
            bs1 (bool): if True, plot Borovickova-Shimer results for the standard estimator
            bs2 (bool): if True, plot Borovickova-Shimer results for the alternative estimator
            density (bool): if True, plot density; if False, plot count
        '''
        if self.res is None:
            raise AttributeError('Attribute .res is None - must run Monte Carlo simulations before histogram can be generated. This can be done by running .fit()')

        if not self.estimate_bs:
            bs1 = False
            bs2 = False

        # Extract results
        res = self.res

        ## Plotting dictionaries ##
        to_plot_dict = {
            'fe': fe,
            'ho': ho,
            'he': he,
            'cre': cre,
            'bs1': bs1,
            'bs2': bs2
        }
        plot_options_dict = {
            'fe': {
                'label': 'FE',
                'color': 'C0'
            },
            'ho': {
                'label': 'FE-HO',
                'color': 'C1'
            },
            'he': {
                'label': 'FE-HE',
                'color': 'C2'
            },
            'cre': {
                'label': 'CRE',
                'color': 'C3'
            },
            'bs1': {
                'label': 'BS-1',
                'color': 'C4'
            },
            'bs2': {
                'label': 'BS-2',
                'color': 'C5'
            }
        }

        # Define differences
        var_diffs = {estimator: np.sort(res[f'var(psi)_{estimator}'] - res['var(psi)_true']) for estimator, to_plot in to_plot_dict.items() if to_plot}
        cov_diffs = {estimator: np.sort(res[f'cov(psi, alpha)_{estimator}'] - res['cov(psi, alpha)_true']) for estimator, to_plot in to_plot_dict.items() if to_plot}

        ### Plot histograms ###
        fig, axs = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, dpi=150)

        ## Define plot bounds (source: https://stackoverflow.com/a/50864765) ##
        min_err_var = np.inf
        min_err_cov = np.inf
        max_err_var = -np.inf
        max_err_cov = -np.inf
        for estimator, to_plot in to_plot_dict.items():
            if to_plot:
                min_err_var = min(min_err_var, np.min(var_diffs[estimator]))
                min_err_cov = min(min_err_cov, np.min(cov_diffs[estimator]))
                max_err_var = max(max_err_var, np.max(var_diffs[estimator]))
                max_err_cov = max(max_err_cov, np.max(cov_diffs[estimator]))
        min_err_var *= ((min_err_var <= 0) * 1.05 + (min_err_var > 0) * 0.95)
        min_err_cov *= ((min_err_cov <= 0) * 1.05 + (min_err_cov > 0) * 0.95)
        max_err_var *= ((max_err_var <= 0) * 0.95 + (max_err_var > 0) * 1.05)
        max_err_cov *= ((max_err_cov <= 0) * 0.95 + (max_err_cov > 0) * 1.05)

        # Plot bounds
        plt_range_var = (min_err_var, max_err_var)
        plt_range_cov = (min_err_cov, max_err_cov)

        ## var(psi) ##
        axs[0].axvline(x=0, color='purple', linestyle='--', label=r'$\Delta$truth=0')
        for estimator, plot_options in plot_options_dict.items():
            if to_plot_dict[estimator]:
                # Plot if to_plot=True
                axs[0].hist(var_diffs[estimator], bins=50, range=plt_range_var, density=density, color=plot_options['color'], label=plot_options['label'])
        # axs[0].legend()
        axs[0].set_title(r'var($\psi$)')
        axs[0].set_xlabel(r'$\Delta$truth')
        if density:
            axs[0].set_ylabel('density')
        else:
            axs[0].set_ylabel('frequency')

        ## cov(psi, alpha) ##
        axs[1].axvline(x=0, color='purple', linestyle='--', label=r'$\Delta$truth=0')
        for estimator, plot_options in plot_options_dict.items():
            if to_plot_dict[estimator]:
                # Plot if to_plot=True
                axs[1].hist(cov_diffs[estimator], bins=50, range=plt_range_cov, density=density, color=plot_options['color'], label=plot_options['label'])
        # axs[1].legend()
        axs[1].set_title(r'cov($\psi$, $\alpha$)')
        axs[1].set_xlabel(r'$\Delta$truth')

        # Shared legend (source: https://stackoverflow.com/a/46921590/17333120)
        handles, labels = axs[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.75, 0.375))
        fig.tight_layout()
        # Make space for legend (source: https://stackoverflow.com/a/43439132/17333120)
        fig.subplots_adjust(right=0.75)
        plt.show()
