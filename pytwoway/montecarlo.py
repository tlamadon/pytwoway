'''
Class for running Monte Carlo estimations on simulated bipartite networks
'''
import numpy as np
import pandas as pd
from multiprocessing import Pool
from matplotlib import pyplot as plt
import bipartitepandas as bpd
import pytwoway as tw
from tqdm import tqdm, trange
import warnings

class MonteCarlo:
    '''
    Class of MonteCarlo, where MonteCarlo runs a Monte Carlo estimation by simulating bipartite networks of firms and workers.

    Arguments:
        sim_params (ParamsDict or None): dictionary of parameters for simulating data. Run bpd.sim_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.sim_params().
        fe_params (ParamsDict or None): dictionary of parameters for FE estimation. Run tw.fe_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.fe_params().
        cre_params (ParamsDict or None): dictionary of parameters for CRE estimation. Run tw.cre_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.cre_params().
        cluster_params (ParamsDict or None): dictionary of parameters for clustering in CRE estimation. Run bpd.cluster_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.cluster_params().
        clean_params (ParamsDict or None): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.clean_params().
        collapse (bool): if True, run estimators on data collapsed at the worker-firm spell level
        move_to_worker (bool): if True, each move is treated as a new worker
        log (bool): if True, will create log file(s)
    '''

    def __init__(self, sim_params=None, fe_params=None, cre_params=None, cluster_params=None, clean_params=None, collapse=True, move_to_worker=False, log=False):
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
        # Start logger
        # logger_init(self)
        # self.logger.info('initializing MonteCarlo object')

        ## Save attributes
        # Parameter dictionaries
        self.fe_params = fe_params.copy()
        self.cre_params = cre_params.copy()
        self.cluster_params = cluster_params.copy()
        self.clean_params = clean_params.copy()
        # Other attributes
        self.sim_network = bpd.SimBipartite(sim_params)
        self.collapse = collapse
        self.move_to_worker = move_to_worker
        self.log = log

        # Update parameter dictionaries
        self.fe_params['he'] = True
        # Clean parameters
        self.clean_params['connectedness'] = 'leave_out_observation'
        self.clean_params['is_sorted'] = True
        self.clean_params['force'] = True
        self.clean_params['copy'] = False
        if collapse:
            self.clean_params_one = self.clean_params.copy()
            self.clean_params_one['connectedness'] = None
            self.clean_params_two = self.clean_params.copy()
            self.clean_params_two['force'] = False
        # Cluster parameters
        self.cluster_params['clean_params'] = self.clean_params.copy()
        self.cluster_params['is_sorted'] = True
        self.cluster_params['copy'] = False

        # Prevent plotting until results exist
        self.monte_carlo_res = False

        # self.logger.info('MonteCarlo object initialized')

    # Cannot include two underscores because isn't compatible with starmap for multiprocessing
    # Source: https://stackoverflow.com/questions/27054963/python-attribute-error-object-has-no-attribute
    def _monte_carlo_interior(self, rng=None):
        '''
        Run Monte Carlo simulations of two way fixed effect models to see the distribution of the true vs. estimated variance of psi and covariance between psi and alpha. This is the interior function to monte_carlo.

        Arguments:
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            true_psi_var (float): true simulated sample variance of psi
            true_psi_alpha_cov (float): true simulated sample covariance of psi and alpha
            cre_psi_var (float): CRE estimate of variance of psi
            cre_psi_alpha_cov (float): CRE estimate of covariance of psi and alpha
            fe_psi_var (float): FE estimate of variance of psi
            fe_psi_alpha_cov (float): FE estimate of covariance of psi and alpha
            ho_psi_var (float): homoskedastic-corrected estimate of variance of psi
            ho_psi_alpha_cov (float): homoskedastic-corrected estimate of covariance of psi and alpha
            he_psi_var (float): heteroskedastic-corrected estimate of variance of psi
            he_psi_alpha_cov (float): heteroskedastic-corrected estimate of covariance of psi and alpha
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        ## Simulate data
        sim_data = self.sim_network.simulate(rng)
        ## Compute true sample variance of psi and covariance of psi and alpha
        psi_var = np.var(sim_data.loc[:, 'psi'].to_numpy(), ddof=1)
        psi_alpha_cov = np.cov(sim_data.loc[:, 'psi'].to_numpy(), sim_data.loc[:, 'alpha'].to_numpy(), ddof=1)[0, 1]
        ## Convert into BipartitePandas dataframe
        sim_data = bpd.BipartiteLong(sim_data.loc[:, ['i', 'j', 'y', 't']], log=self.log)
        ## Clean data
        if self.collapse or self.move_to_worker:
            ## Collapsing or setting moves to worker ids
            # Initial clean without connectedness
            sim_data = sim_data.clean(self.clean_params_one)
            if self.move_to_worker:
                # Move-to-Worker
                sim_data = sim_data.to_eventstudy(move_to_worker=True, is_sorted=True, copy=False).to_long(is_sorted=True, copy=False)
            if self.collapse:
                # Collapse
                sim_data = sim_data.collapse(is_sorted=True, copy=False)
            # Final clean with connectedness
            sim_data = sim_data.clean(self.clean_params_two)
        else:
            # Standard
            sim_data = sim_data.clean(self.clean_params)
        ## Estimate FE model
        fe_estimator = tw.FEEstimator(sim_data, self.fe_params)
        fe_estimator.fit(rng)
        # Save results
        fe_res = fe_estimator.res
        ## Estimate CRE model
        # Cluster
        sim_data = sim_data.cluster(self.cluster_params)
        # Estimate
        cre_estimator = tw.CREEstimator(sim_data.to_eventstudy(move_to_worker=False, is_sorted=True, copy=False).get_cs(copy=False), self.cre_params)
        cre_estimator.fit(rng)
        # Save results
        cre_res = cre_estimator.res

        return psi_var, psi_alpha_cov, \
                cre_res['tot_var'], cre_res['tot_cov'], \
                fe_res['var_fe'], fe_res['cov_fe'], \
                fe_res['var_ho'], fe_res['cov_ho'], \
                fe_res['var_he'], fe_res['cov_he']

    def monte_carlo(self, N=10, ncore=1, rng=None):
        '''
        Run Monte Carlo simulations of two way fixed effect models to see the distribution of the true vs. estimated variance of psi and covariance between psi and alpha. Saves the following results in the dictionary self.res:

            true_psi_var (NumPy Array): true simulated sample variance of psi

            true_psi_alpha_cov (NumPy Array): true simulated sample covariance of psi and alpha

            cre_psi_var (NumPy Array): CRE estimate of variance of psi

            cre_psi_alpha_cov (NumPy Array): CRE estimate of covariance of psi and alpha

            fe_psi_var (NumPy Array): AKM estimate of variance of psi

            fe_psi_alpha_cov (NumPy Array): AKM estimate of covariance of psi and alpha

            ho_psi_var (NumPy Array): homoskedastic-corrected AKM estimate of variance of psi

            ho_psi_alpha_cov (NumPy Array): homoskedastic-corrected AKM estimate of covariance of psi and alpha

            he_psi_var (NumPy Array): heteroskedastic-corrected AKM estimate of variance of psi

            he_psi_alpha_cov (NumPy Array): heteroskedastic-corrected AKM estimate of covariance of psi and alpha

        Arguments:
            N (int): number of simulations
            ncore (int): number of cores to use
            rng (np.random.Generator or None): NumPy random number generator. This overrides the random number generators for simulating, FE, and CRE. None is equivalent to np.random.default_rng(None).
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Initialize NumPy arrays to store results
        true_psi_var = np.zeros(N)
        true_psi_alpha_cov = np.zeros(N)
        cre_psi_var = np.zeros(N)
        cre_psi_alpha_cov = np.zeros(N)
        fe_psi_var = np.zeros(N)
        fe_psi_alpha_cov = np.zeros(N)
        ho_psi_var = np.zeros(N)
        ho_psi_alpha_cov = np.zeros(N)
        he_psi_var = np.zeros(N)
        he_psi_alpha_cov = np.zeros(N)

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
        for i, res_i in enumerate(all_res):
            true_psi_var[i], true_psi_alpha_cov[i], cre_psi_var[i], cre_psi_alpha_cov[i], fe_psi_var[i], fe_psi_alpha_cov[i], ho_psi_var[i], ho_psi_alpha_cov[i], he_psi_var[i], he_psi_alpha_cov[i] = res_i

        res = {}

        res['true_psi_var'] = true_psi_var
        res['true_psi_alpha_cov'] = true_psi_alpha_cov
        res['cre_psi_var'] = cre_psi_var
        res['cre_psi_alpha_cov'] = cre_psi_alpha_cov
        res['fe_psi_var'] = fe_psi_var
        res['fe_psi_alpha_cov'] = fe_psi_alpha_cov
        res['ho_psi_var'] = ho_psi_var
        res['ho_psi_alpha_cov'] = ho_psi_alpha_cov
        res['he_psi_var'] = he_psi_var
        res['he_psi_alpha_cov'] = he_psi_alpha_cov

        self.res = res
        self.monte_carlo_res = True

    def plot_monte_carlo(self, density=False, fe=True, ho=True, he=True, cre=True):
        '''
        Plot results from Monte Carlo simulations.

        Arguments:
            density (bool): if True, plot density; if False, plot count
            fe (bool): if True, plot FE results
            ho (bool): if True, plot homoskedastic correction results
            he (bool): if True, plot heteroskedastic correction results
            cre (bool): if True, plot CRE results
        '''
        if not self.monte_carlo_res:
            warnings.warn('Must run Monte Carlo simulations before results can be plotted. This can be done by running .monte_carlo().')

        else:
            # Extract results
            true_psi_var = self.res['true_psi_var']
            true_psi_alpha_cov = self.res['true_psi_alpha_cov']
            cre_psi_var = self.res['cre_psi_var']
            cre_psi_alpha_cov = self.res['cre_psi_alpha_cov']
            fe_psi_var = self.res['fe_psi_var']
            fe_psi_alpha_cov = self.res['fe_psi_alpha_cov']
            ho_psi_var = self.res['ho_psi_var']
            ho_psi_alpha_cov = self.res['ho_psi_alpha_cov']
            he_psi_var = self.res['he_psi_var']
            he_psi_alpha_cov = self.res['he_psi_alpha_cov']

            # Define differences
            cre_psi_diff = sorted(cre_psi_var - true_psi_var)
            cre_psi_alpha_diff = sorted(cre_psi_alpha_cov - true_psi_alpha_cov)
            fe_psi_diff = sorted(fe_psi_var - true_psi_var)
            fe_psi_alpha_diff = sorted(fe_psi_alpha_cov - true_psi_alpha_cov)
            ho_psi_diff = sorted(ho_psi_var - true_psi_var)
            ho_psi_alpha_diff = sorted(ho_psi_alpha_cov - true_psi_alpha_cov)
            he_psi_diff = sorted(he_psi_var - true_psi_var)
            he_psi_alpha_diff = sorted(he_psi_alpha_cov - true_psi_alpha_cov)

            # Plot histograms
            # First, var(psi)
            # Source for fixing bin size:
            # https://stackoverflow.com/a/50864765
            min_err = np.inf
            if fe:
                min_err = min(min_err, np.min(fe_psi_diff))
            if ho:
                min_err = min(min_err, np.min(ho_psi_diff))
            if he:
                min_err = min(min_err, np.min(he_psi_diff))
            if cre:
                min_err = min(min_err, np.min(cre_psi_diff))
            max_err = -np.inf
            if fe:
                max_err = max(max_err, np.max(fe_psi_diff))
            if ho:
                max_err = max(max_err, np.max(ho_psi_diff))
            if he:
                max_err = max(max_err, np.max(he_psi_diff))
            if cre:
                max_err = max(max_err, np.max(cre_psi_diff))
            if min_err > 0:
                min_err *= 0.95
            else:
                min_err *= 1.05
            if max_err > 0:
                max_err *= 1.05
            else:
                max_err *= 0.95
            plt_range = (min_err, max_err)
            plt.axvline(x=0, color='purple', linestyle='--', label=r'$\Delta$Truth=0')
            if fe:
                plt.hist(fe_psi_diff, bins=50, range=plt_range, density=density, label='FE var(psi)')
            if ho:
                plt.hist(ho_psi_diff, bins=50, range=plt_range, density=density, label='HO-corrected var(psi)')
            if he:
                plt.hist(he_psi_diff, bins=50, range=plt_range, density=density, label='HE-corrected var(psi)')
            if cre:
                plt.hist(cre_psi_diff, bins=50, range=plt_range, density=density, label='CRE var(psi)')
            plt.legend()
            plt.show()

            # Second, cov(psi, alpha)
            min_err = np.inf
            if fe:
                min_err = min(min_err, np.min(fe_psi_alpha_diff))
            if ho:
                min_err = min(min_err, np.min(ho_psi_alpha_diff))
            if he:
                min_err = min(min_err, np.min(he_psi_alpha_diff))
            if cre:
                min_err = min(min_err, np.min(cre_psi_alpha_diff))
            max_err = -np.inf
            if fe:
                max_err = max(max_err, np.max(fe_psi_alpha_diff))
            if ho:
                max_err = max(max_err, np.max(ho_psi_alpha_diff))
            if he:
                max_err = max(max_err, np.max(he_psi_alpha_diff))
            if cre:
                max_err = max(max_err, np.max(cre_psi_alpha_diff))
            if min_err > 0:
                min_err *= 0.95
            else:
                min_err *= 1.05
            if max_err > 0:
                max_err *= 1.05
            else:
                max_err *= 0.95
            plt_range = (min_err, max_err)
            plt.axvline(x=0, color='purple', linestyle='--', label=r'$\Delta$Truth=0')
            if fe:
                plt.hist(fe_psi_alpha_diff, bins=50, range=plt_range, density=density, label='FE cov(psi, alpha)')
            if ho:
                plt.hist(ho_psi_alpha_diff, bins=50, range=plt_range, density=density, label='HO-corrected cov(psi, alpha)')
            if he:
                plt.hist(he_psi_alpha_diff, bins=50, range=plt_range, density=density, label='HE-corrected cov(psi, alpha)')
            if cre:
                plt.hist(cre_psi_alpha_diff, bins=50, range=plt_range, density=density, label='CRE cov(psi, alpha)')
            plt.legend()
            plt.show()
