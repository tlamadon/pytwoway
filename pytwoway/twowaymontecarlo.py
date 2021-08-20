'''
Class for running Monte Carlo estimations on simulated bipartite networks
'''
from multiprocessing import Pool
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import bipartitepandas as bpd
import pytwoway as tw
from tqdm import trange
import warnings

class TwoWayMonteCarlo:
    '''
    Class of TwoWayMonteCarlo, where TwoWayMonteCarlo runs a Monte Carlo estimation by simulating bipartite networks of firms and workers.

    Arguments:
        sim_params (dict): parameters for simulated data

            Dictionary parameters:

                num_ind (int, default=10000): number of workers

                num_time (int, default=5): time length of panel

                firm_size (int, default=50): max number of individuals per firm

                nk (int, default=10): number of firm types

                nl (int, default=5): number of worker types

                alpha_sig (float, default=1): standard error of individual fixed effect (volatility of worker effects)

                psi_sig (float, default=1): standard error of firm fixed effect (volatility of firm effects)

                w_sig (float, default=1): standard error of residual in AKM wage equation (volatility of wage shocks)

                csort (float, default=1): sorting effect

                cnetw (float, default=1): network effect

                csig (float, default=1): standard error of sorting/network effects

                p_move (float, default=0.5): probability a worker moves firms in each period

                seed (int, default=None): NumPy RandomState seed
    '''

    def __init__(self, sim_params={}):
        # Start logger
        # logger_init(self)
        # self.logger.info('initializing TwoWayMonteCarlo object')

        self.sbp_net = bpd.SimBipartite(sim_params)

        # Prevent plotting unless results exist
        self.monte_carlo_res = False

        # self.logger.info('TwoWayMonteCarlo object initialized')

    # Cannot include two underscores because isn't compatible with starmap for multiprocessing
    # Source: https://stackoverflow.com/questions/27054963/python-attribute-error-object-has-no-attribute
    def _twfe_monte_carlo_interior(self, fe_params={}, cre_params={}, cluster_params={}, collapsed=True, clean_params={}):
        '''
        Run Monte Carlo simulations of TwoWay to see the distribution of the true vs. estimated variance of psi and covariance between psi and alpha. This is the interior function to twfe_monte_carlo.

        Arguments:
            fe_params (dict): dictionary of parameters for FE estimation

                Dictionary parameters:

                    ncore (int, default=1): number of cores to use

                    batch (int, default=1): batch size to send in parallel

                    ndraw_pii (int, default=50): number of draws to use in approximation for leverages

                    levfile (str, default=''): file to load precomputed leverages

                    ndraw_tr (int, default=5): number of draws to use in approximation for traces

                    he (bool, default=False): if True, compute heteroskedastic correction

                    out (str, default='res_fe.json'): outputfile where results are saved

                    statsonly (bool, default=False): if True, return only basic statistics

                    feonly (bool, default=False): if True, compute only fixed effects and not variances

                    Q (str, default='cov(alpha, psi)'): which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'

                    seed (int, default=None): NumPy RandomState seed

            cre_params (dict): dictionary of parameters for CRE estimation

                Dictionary parameters:

                    ncore (int, default=1): number of cores to use

                    ndraw_tr (int, default=5): number of draws to use in approximation for traces

                    ndp (int, default=50): number of draw to use in approximation for leverages

                    out (str, default='res_cre.json'): outputfile where results are saved

                    posterior (bool, default=False): if True, compute posterior variance

                    wo_btw (bool, default=False): if True, sets between variation to 0, pure RE

            cluster_params (dict): dictionary of parameters for clustering in CRE estimation

                Dictionary parameters:

                    measures (function or list of functions): how to compute measures for clustering. Options can be seen in bipartitepandas.measures.

                    grouping (function): how to group firms based on measures. Options can be seen in bipartitepandas.grouping.

                    stayers_movers (str or None, default=None): if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers

                    t (int or None, default=None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data)

                    weighted (bool, default=True): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)

                    dropna (bool, default=False): if True, drop observations where firms aren't clustered; if False, keep all observations

            collapsed (bool): if True, run estimators on collapsed data
            clean_params (dict): dictionary of parameters for cleaning

                Dictionary parameters:

                    connectedness (str or None, default='connected'): if 'connected', keep observations in the largest connected set of firms; if 'biconnected', keep observations in the largest biconnected set of firms; if None, keep all observations

                    i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids

                    data_validity (bool, default=True): if True, run data validity checks; much faster if set to False

                    copy (bool, default=False): if False, avoid copy

        Returns:
            true_psi_var (float): true simulated sample variance of psi
            true_psi_alpha_cov (float): true simulated sample covariance of psi and alpha
            cre_psi_var (float): CRE estimate of variance of psi
            cre_psi_alpha_cov (float): CRE estimate of covariance of psi and alpha
            fe_psi_var (float): AKM estimate of variance of psi
            fe_psi_alpha_cov (float): AKM estimate of covariance of psi and alpha
            fe_ho_psi_var (float): homoskedastic-corrected AKM estimate of variance of psi
            fe_ho_psi_alpha_cov (float): homoskedastic-corrected AKM estimate of covariance of psi and alpha
            fe_he_psi_var (float): heteroskedastic-corrected AKM estimate of variance of psi
            fe_he_psi_alpha_cov (float): heteroskedastic-corrected AKM estimate of covariance of psi and alpha
        '''
        # Simulate data
        sim_data = self.sbp_net.sim_network()
        # Compute true sample variance of psi and covariance of psi and alpha
        psi_var = np.var(sim_data['psi'])
        psi_alpha_cov = np.cov(sim_data['psi'], sim_data['alpha'])[0, 1]
        # Use data to create TwoWay object
        tw_net = tw.TwoWay(sim_data)
        # Prepare data
        tw_net.prep_data(collapsed=collapsed, user_clean=clean_params, he=False)
        # Cluster data
        tw_net.cluster(**cluster_params)
        # Estimate CRE model
        tw_net.fit_cre(user_cre=cre_params)
        # Save results
        cre_res = tw_net.cre_res
        # Prepare data for heteroskedastic correction
        tw_net.prep_data(collapsed=collapsed, user_clean=clean_params, he=True)
        # Estimate FE model
        tw_net.fit_fe(user_fe=fe_params)
        # Save results
        fe_res = tw_net.fe_res

        return psi_var, psi_alpha_cov, \
                cre_res['tot_var'], cre_res['tot_cov'], \
                fe_res['var_fe'], fe_res['cov_fe'], \
                fe_res['var_ho'], fe_res['cov_ho'], \
                fe_res['var_he'], fe_res['cov_he']

    def twfe_monte_carlo(self, N=10, ncore=1, fe_params={}, cre_params={}, cluster_params={}, collapsed=True, clean_params={}):
        '''
        Run Monte Carlo simulations of TwoWay to see the distribution of the true vs. estimated variance of psi and covariance between psi and alpha. Saves the following results in the dictionary self.res:

            true_psi_var (NumPy Array): true simulated sample variance of psi

            true_psi_alpha_cov (NumPy Array): true simulated sample covariance of psi and alpha

            cre_psi_var (NumPy Array): CRE estimate of variance of psi

            cre_psi_alpha_cov (NumPy Array): CRE estimate of covariance of psi and alpha

            fe_psi_var (NumPy Array): AKM estimate of variance of psi

            fe_psi_alpha_cov (NumPy Array): AKM estimate of covariance of psi and alpha

            fe_ho_psi_var (NumPy Array): homoskedastic-corrected AKM estimate of variance of psi

            fe_ho_psi_alpha_cov (NumPy Array): homoskedastic-corrected AKM estimate of covariance of psi and alpha

            fe_he_psi_var (NumPy Array): heteroskedastic-corrected AKM estimate of variance of psi

            fe_he_psi_alpha_cov (NumPy Array): heteroskedastic-corrected AKM estimate of covariance of psi and alpha

        Arguments:
            N (int): number of simulations
            ncore (int): how many cores to use
            fe_params (dict): dictionary of parameters for FE estimation

                Dictionary parameters:

                    ncore (int, default=1): number of cores to use

                    batch (int, default=1): batch size to send in parallel

                    ndraw_pii (int, default=50): number of draws to use in approximation for leverages

                    levfile (str, default=''): file to load precomputed leverages

                    ndraw_tr (int, default=5): number of draws to use in approximation for traces

                    he (bool, default=False): if True, compute heteroskedastic correction

                    out (str, default='res_fe.json'): outputfile where results are saved

                    statsonly (bool, default=False): if True, return only basic statistics

                    feonly (bool, default=False): if True, compute only fixed effects and not variances

                    Q (str, default='cov(alpha, psi)'): which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'

                    seed (int, default=None): NumPy RandomState seed

            cre_params (dict): dictionary of parameters for CRE estimation

                Dictionary parameters:

                    ncore (int, default=1): number of cores to use

                    ndraw_tr (int, default=5): number of draws to use in approximation for traces

                    ndp (int, default=50): number of draw to use in approximation for leverages

                    out (str, default='res_cre.json'): outputfile where results are saved

                    posterior (bool, default=False): if True, compute posterior variance

                    wo_btw (bool, default=False): if True, sets between variation to 0, pure RE

            cluster_params (dict): dictionary of parameters for clustering in CRE estimation

                Dictionary parameters:

                    measures (function or list of functions): how to compute measures for clustering. Options can be seen in bipartitepandas.measures.

                    grouping (function): how to group firms based on measures. Options can be seen in bipartitepandas.grouping.

                    stayers_movers (str or None, default=None): if None, clusters on entire dataset; if 'stayers', clusters on only stayers; if 'movers', clusters on only movers

                    t (int or None, default=None): if None, clusters on entire dataset; if int, gives period in data to consider (only valid for non-collapsed data)

                    weighted (bool, default=True): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)

                    dropna (bool, default=False): if True, drop observations where firms aren't clustered; if False, keep all observations

            collapsed (bool): if True, run estimators on data collapsed by worker-firm spells
            clean_params (dict): dictionary of parameters for cleaning

                Dictionary parameters:

                    connectedness (str or None, default='connected'): if 'connected', keep observations in the largest connected set of firms; if 'biconnected', keep observations in the largest biconnected set of firms; if None, keep all observations

                    i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids

                    data_validity (bool, default=True): if True, run data validity checks; much faster if set to False

                    copy (bool, default=False): if False, avoid copy
        '''
        # Initialize NumPy arrays to store results
        true_psi_var = np.zeros(N)
        true_psi_alpha_cov = np.zeros(N)
        cre_psi_var = np.zeros(N)
        cre_psi_alpha_cov = np.zeros(N)
        fe_psi_var = np.zeros(N)
        fe_psi_alpha_cov = np.zeros(N)
        fe_ho_psi_var = np.zeros(N)
        fe_ho_psi_alpha_cov = np.zeros(N)
        fe_he_psi_var = np.zeros(N)
        fe_he_psi_alpha_cov = np.zeros(N)

        # Set fe_params to always have he=True
        fe_params['he'] = True

        # Use multi-processing
        if ncore > 1:
            # Simulate networks
            with Pool(processes=ncore) as pool:
                V = pool.starmap(self._twfe_monte_carlo_interior, [(fe_params, cre_params, cluster_params, collapsed, clean_params) for _ in range(N)])
            for i, res in enumerate(V):
                true_psi_var[i], true_psi_alpha_cov[i], cre_psi_var[i], cre_psi_alpha_cov[i], fe_psi_var[i], fe_psi_alpha_cov[i], fe_ho_psi_var[i], fe_ho_psi_alpha_cov[i], fe_he_psi_var[i], fe_he_psi_alpha_cov[i] = res
        else:
            for i in trange(N):
                # Simulate a network
                true_psi_var[i], true_psi_alpha_cov[i], cre_psi_var[i], cre_psi_alpha_cov[i], fe_psi_var[i], fe_psi_alpha_cov[i], fe_ho_psi_var[i], fe_ho_psi_alpha_cov[i], fe_he_psi_var[i], fe_he_psi_alpha_cov[i] = self._twfe_monte_carlo_interior(fe_params=fe_params, cre_params=cre_params, cluster_params=cluster_params, collapsed=collapsed, clean_params=clean_params)

        res = {}

        res['true_psi_var'] = true_psi_var
        res['true_psi_alpha_cov'] = true_psi_alpha_cov
        res['cre_psi_var'] = cre_psi_var
        res['cre_psi_alpha_cov'] = cre_psi_alpha_cov
        res['fe_psi_var'] = fe_psi_var
        res['fe_psi_alpha_cov'] = fe_psi_alpha_cov
        res['fe_ho_psi_var'] = fe_ho_psi_var
        res['fe_ho_psi_alpha_cov'] = fe_ho_psi_alpha_cov
        res['fe_he_psi_var'] = fe_he_psi_var
        res['fe_he_psi_alpha_cov'] = fe_he_psi_alpha_cov

        self.res = res
        self.monte_carlo_res = True

    def plot_monte_carlo(self):
        '''
        Plot results from Monte Carlo simulations.
        '''
        if not self.monte_carlo_res:
            warnings.warn('Must run Monte Carlo simulations before results can be plotted. This can be done by running .twfe_monte_carlo(self, N=10, ncore=1, fe_params={}, cre_params={}, cluster_params={})')

        else:
            # Extract results
            true_psi_var = self.res['true_psi_var']
            true_psi_alpha_cov = self.res['true_psi_alpha_cov']
            cre_psi_var = self.res['cre_psi_var']
            cre_psi_alpha_cov = self.res['cre_psi_alpha_cov']
            fe_psi_var = self.res['fe_psi_var']
            fe_psi_alpha_cov = self.res['fe_psi_alpha_cov']
            fe_ho_psi_var = self.res['fe_ho_psi_var']
            fe_ho_psi_alpha_cov = self.res['fe_ho_psi_alpha_cov']
            fe_he_psi_var = self.res['fe_he_psi_var']
            fe_he_psi_alpha_cov = self.res['fe_he_psi_alpha_cov']

            # Define differences
            cre_psi_diff = sorted(cre_psi_var - true_psi_var)
            cre_psi_alpha_diff = sorted(cre_psi_alpha_cov - true_psi_alpha_cov)
            fe_psi_diff = sorted(fe_psi_var - true_psi_var)
            fe_psi_alpha_diff = sorted(fe_psi_alpha_cov - true_psi_alpha_cov)
            fe_ho_psi_diff = sorted(fe_ho_psi_var - true_psi_var)
            fe_ho_psi_alpha_diff = sorted(fe_ho_psi_alpha_cov - true_psi_alpha_cov)
            fe_he_psi_diff = sorted(fe_he_psi_var - true_psi_var)
            fe_he_psi_alpha_diff = sorted(fe_he_psi_alpha_cov - true_psi_alpha_cov)

            # Plot histograms
            # First, var(psi)
            # Source for fixing bin size:
            # https://stackoverflow.com/a/50864765
            min_err = np.min([np.min(fe_psi_diff), np.min(fe_ho_psi_diff), np.min(fe_he_psi_diff), np.min(cre_psi_diff)])
            max_err = np.max([np.max(fe_psi_diff), np.max(fe_ho_psi_diff), np.max(fe_he_psi_diff), np.max(cre_psi_diff)])
            plt_range = (0.95 * min_err, 1.05 * max_err)
            plt.axvline(x=0, color='purple', linestyle='--', label=r'$\Delta$Truth=0')
            plt.hist(fe_psi_diff, bins=50, range=plt_range, label='AKM var(psi)')
            plt.hist(fe_ho_psi_diff, bins=50, range=plt_range, label='Ho-corrected AKM var(psi)')
            plt.hist(fe_he_psi_diff, bins=50, range=plt_range, label='He-corrected AKM var(psi)')
            plt.hist(cre_psi_diff, bins=50, range=plt_range, label='CRE var(psi)')
            plt.legend()
            plt.show()

            # Second, cov(psi, alpha)
            min_err = np.min([np.min(fe_psi_alpha_diff), np.min(fe_ho_psi_alpha_diff), np.min(fe_he_psi_alpha_diff), np.min(cre_psi_alpha_diff)])
            max_err = np.max([np.max(fe_psi_alpha_diff), np.max(fe_ho_psi_alpha_diff), np.max(fe_he_psi_alpha_diff), np.max(cre_psi_alpha_diff)])
            plt_range = (0.95 * min_err, 1.05 * max_err)
            plt.axvline(x=0, color='purple', linestyle='--', label=r'$\Delta$Truth=0')
            plt.hist(fe_psi_alpha_diff, bins=50, range=plt_range, label='AKM cov(psi, alpha)')
            plt.hist(fe_ho_psi_alpha_diff, bins=50, range=plt_range, label='Ho-corrected AKM cov(psi, alpha)')
            plt.hist(fe_he_psi_alpha_diff, bins=50, range=plt_range, label='He-corrected AKM cov(psi, alpha)')
            plt.hist(cre_psi_alpha_diff, bins=50, range=plt_range, label='CRE cov(psi, alpha)')
            plt.legend()
            plt.show()
