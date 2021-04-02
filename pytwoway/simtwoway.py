'''
Class for a simulated two-way fixed effect network
'''

import logging
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from numpy import matlib
import pandas as pd
from random import choices
from scipy.stats import mode, norm
from scipy.linalg import eig
ax = np.newaxis
from matplotlib import pyplot as plt
import pytwoway as tw
from bipartitepandas import update_dict, logger_init
from tqdm import trange

class SimTwoWay:
    '''
    Class of SimTwoWay, where SimTwoWay simulates a network of firms and workers.

    Arguments:
        sim_params (dict): parameters for simulated data

            Dictionary parameters:

                num_ind (int): number of workers

                num_time (int): time length of panel

                firm_size (int): max number of individuals per firm

                nk (int): number of firm types

                nl (int): number of worker types

                alpha_sig (float): standard error of individual fixed effect (volatility of worker effects)

                psi_sig (float): standard error of firm fixed effect (volatility of firm effects)

                w_sig (float): standard error of residual in AKM wage equation (volatility of wage shocks)

                csort (float): sorting effect

                cnetw (float): network effect

                csig (float): standard error of sorting/network effects

                p_move (float): probability a worker moves firms in any period
    '''

    def __init__(self, sim_params={}):
        # Start logger
        logger_init(self)
        # self.logger.info('initializing SimTwoWay object')

        # Define default parameter dictionaries
        self.default_sim_params = {'num_ind': 10000, 'num_time': 5, 'firm_size': 50, 'nk': 10, 'nl': 5, 'alpha_sig': 1, 'psi_sig': 1, 'w_sig': 1, 'csort': 1, 'cnetw': 1, 'csig': 1, 'p_move': 0.5}

        # Update parameters to include user parameters
        self.sim_params = update_dict(self.default_sim_params, sim_params)

        # Prevent plotting unless results exist
        self.monte_carlo_res = False

    def __sim_network_gen_fe(self, sim_params):
        '''
        Generate fixed effects values for simulated panel data corresponding to the calibrated model.

        Arguments:
            sim_params (dict): parameters for simulated data

                Dictionary parameters:

                    num_ind (int): number of workers

                    num_time (int): time length of panel

                    firm_size (int): max number of individuals per firm

                    nk (int): number of firm types

                    nl (int): number of worker types

                    alpha_sig (float): standard error of individual fixed effect (volatility of worker effects)

                    psi_sig (float): standard error of firm fixed effect (volatility of firm effects)

                    w_sig (float): standard error of residual in AKM wage equation (volatility of wage shocks)

                    csort (float): sorting effect

                    cnetw (float): network effect

                    csig (float): standard error of sorting/network effects

                    p_move (float): probability a worker moves firms in any period

        Returns:
            psi (NumPy Array): array of firm fixed effects
            alpha (NumPy Array): array of individual fixed effects
            G (NumPy Array): transition matrices
            H (NumPy Array): stationary distribution
        '''
        # Extract parameters
        nk, nl, alpha_sig, psi_sig = sim_params['nk'], sim_params['nl'], sim_params['alpha_sig'], sim_params['psi_sig']
        csort, cnetw, csig = sim_params['csort'], sim_params['cnetw'], sim_params['csig']

        # Draw fixed effects
        psi = norm.ppf(np.linspace(1, nk, nk) / (nk + 1)) * psi_sig
        alpha = norm.ppf(np.linspace(1, nl, nl) / (nl + 1)) * alpha_sig

        # Generate transition matrices
        G = norm.pdf((psi[ax, ax, :] - cnetw * psi[ax, :, ax] - csort * alpha[:, ax, ax]) / csig)
        G = np.divide(G, G.sum(axis=2)[:, :, ax])

        # Generate empty stationary distributions
        H = np.ones((nl, nk)) / nl

        # Solve stationary distributions
        for l in range(0, nl):
            # Solve eigenvectors
            # Source: https://stackoverflow.com/questions/31791728/python-code-explanation-for-stationary-distribution-of-a-markov-chain
            S, U = eig(G[l, :, :].T)
            stationary = np.array(U[:, np.where(np.abs(S-1.) < 1e-8)[0][0]].flat)
            stationary = stationary / np.sum(stationary)
            H[l, :] = stationary

        return psi, alpha, G, H

    def __sim_network_draw_fids(self, freq, num_time, firm_size):
        '''
        Draw firm ids for individual, given data that is grouped by worker id, spell id, and firm type.

        Arguments:
            freq (NumPy Array): size of groups (groups by worker id, spell id, and firm type)
            num_time (int): time length of panel
            firm_size (int): max number of individuals per firm

        Returns:
            (NumPy Array): random firms for each group
        '''
        max_int = int(np.maximum(1, freq.sum() / (firm_size * num_time)))
        return np.array(np.random.choice(max_int, size=freq.count()) + 1)

    def sim_network(self):
        '''
        Simulate panel data corresponding to the calibrated model.

        Returns:
            data (Pandas DataFrame): simulated network
        '''
        # Generate fixed effects
        psi, alpha, G, H = self.__sim_network_gen_fe(self.sim_params)

        # Extract parameters
        num_ind, num_time, firm_size = self.sim_params['num_ind'], self.sim_params['num_time'], self.sim_params['firm_size']
        nk, nl, w_sig, p_move = self.sim_params['nk'], self.sim_params['nl'], self.sim_params['w_sig'], self.sim_params['p_move']

        # Generate empty NumPy arrays
        network = np.zeros((num_ind, num_time), dtype=int)
        spellcount = np.ones((num_ind, num_time))

        # Random draws of worker types for all individuals in panel
        sim_worker_types = np.random.randint(low=1, high=nl, size=num_ind)

        for i in range(0, num_ind):
            l = sim_worker_types[i]
            # At time 1, we draw from H for initial firm
            network[i, 0] = choices(range(0, nk), H[l, :])[0]

            for t in range(1, num_time):
                # Hit moving shock
                if np.random.rand() < p_move:
                    network[i, t] = choices(range(0, nk), G[l, network[i, t - 1], :])[0]
                    spellcount[i, t] = spellcount[i, t - 1] + 1
                else:
                    network[i, t] = network[i, t - 1]
                    spellcount[i, t] = spellcount[i, t - 1]

        # Compiling IDs and timestamps
        ids = np.reshape(np.outer(range(1, num_ind + 1), np.ones(num_time)), (num_time * num_ind, 1))
        ids = ids.astype(int)[:, 0]
        ts = np.reshape(np.matlib.repmat(range(1, num_time + 1), num_ind, 1), (num_time * num_ind, 1))
        ts = ts.astype(int)[:, 0]

        # Compiling worker types
        types = np.reshape(np.outer(sim_worker_types, np.ones(num_time)), (num_time * num_ind, 1))
        alpha_data = alpha[types.astype(int)][:, 0]

        # Compiling firm types
        psi_data = psi[np.reshape(network, (num_time * num_ind, 1))][:, 0]
        k_data = np.reshape(network, (num_time * num_ind, 1))[:, 0]

        # Compiling spell data
        spell_data = np.reshape(spellcount, (num_time * num_ind, 1))[:, 0]

        # Merging all columns into a dataframe
        data = pd.DataFrame(data={'i': ids, 't': ts, 'k': k_data,
                                'alpha': alpha_data, 'psi': psi_data,
                                'spell': spell_data.astype(int)})

        # Generate size of spells
        dspell = data.groupby(['i', 'spell', 'k']).size().to_frame(name='freq').reset_index()
        # Draw firm ids
        dspell['j'] = dspell.groupby(['k'])['freq'].transform(self.__sim_network_draw_fids, *[num_time, firm_size])
        # Make firm ids contiguous (and have them start at 1)
        dspell['j'] = dspell.groupby(['k', 'j'])['freq'].ngroup() + 1

        # Merge spells into panel
        data = data.merge(dspell, on=['i', 'spell', 'k'])

        data['move'] = (data['j'] != data['j'].shift(1)) & (data['i'] == data['i'].shift(1))

        # Compute wages through the AKM formula
        data['y'] = data['alpha'] + data['psi'] + w_sig * norm.rvs(size=num_ind * num_time)

        data['i'] -= 1 # Start at 0
        data['j'] -= 1 # Start at 0

        return data

class TwoWayMonteCarlo:
    '''
    Class of TwoWayMonteCarlo, where TwoWayMonteCarlo runs a Monte Carlo estimate by simulating networks of firms and workers.

    Arguments:
        sim_params (dict): parameters for simulated data

            Dictionary parameters:

                num_ind (int): number of workers

                num_time (int): time length of panel

                firm_size (int): max number of individuals per firm

                nk (int): number of firm types

                nl (int): number of worker types

                alpha_sig (float): standard error of individual fixed effect (volatility of worker effects)

                psi_sig (float): standard error of firm fixed effect (volatility of firm effects)

                w_sig (float): standard error of residual in AKM wage equation (volatility of wage shocks)

                csort (float): sorting effect

                cnetw (float): network effect

                csig (float): standard error of sorting/network effects

                p_move (float): probability a worker moves firms in any period
    '''

    def __init__(self, sim_params={}):
        # Start logger
        # logger_init(self)
        # self.logger.info('initializing TwoWayMonteCarlo object')

        self.stw_net = SimTwoWay(sim_params)

        # Prevent plotting unless results exist
        self.monte_carlo_res = False

        # self.logger.info('TwoWayMonteCarlo object initialized')

    # Cannot include two underscores because isn't compatible with starmap for multiprocessing
    # Source: https://stackoverflow.com/questions/27054963/python-attribute-error-object-has-no-attribute
    def _twfe_monte_carlo_interior(self, fe_params={}, cre_params={}, cluster_params={}):
        '''
        Run Monte Carlo simulations of TwoWay to see the distribution of the true vs. estimated variance of psi and covariance between psi and alpha. This is the interior function to twfe_monte_carlo.

        Arguments:
            fe_params (dict): dictionary of parameters for FE estimation

                Dictionary parameters:

                    ncore (int): number of cores to use

                    batch (int): batch size to send in parallel

                    ndraw_pii (int): number of draws to use in approximation for leverages

                    levfile (str): file to load precomputed leverages

                    ndraw_tr (int): number of draws to use in approximation for traces

                    h2 (bool): if True, compute h2 correction

                    out (str): outputfile where results are saved

                    statsonly (bool): if True, return only basic statistics

                    Q (str): which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'

            cre_params (dict): dictionary of parameters for CRE estimation

                Dictionary parameters:

                    ncore (int): number of cores to use

                    ndraw_tr (int): number of draws to use in approximation for traces

                    ndp (int): number of draw to use in approximation for leverages

                    out (string): outputfile

                    posterior (bool): compute posterior variance

                    wo_btw (bool): sets between variation to 0, pure RE

            cluster_params (dict): dictionary of parameters for clustering in CRE estimation

                Dictionary parameters:

                    cdf_resolution (int): how many values to use to approximate the cdf

                    grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)

                    stayers_movers (str or None): if None, uses entire dataset; if 'stayers', uses only stayers; if 'movers', uses only movers

                    t (int or None): if None, uses entire dataset; if int, gives time in data to consider (only valid for non-collapsed data)

                    weighted (bool): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)

                    dropna (bool): if True, drop observations where firms aren't clustered; if False, keep all observations

                    user_KMeans (dict): parameters for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

        Returns:
            true_psi_var (float): true simulated sample variance of psi
            true_psi_alpha_cov (float): true simulated sample covariance of psi and alpha
            fe_psi_var (float): AKM estimate of variance of psi
            fe_psi_alpha_cov (float): AKM estimate of covariance of psi and alpha
            fe_corr_psi_var (float): bias-corrected AKM estimate of variance of psi
            fe_corr_psi_alpha_cov (float): bias-corrected AKM estimate of covariance of psi and alpha
            cre_psi_var (float): CRE estimate of variance of psi
            cre_psi_alpha_cov (float): CRE estimate of covariance of psi and alpha
        '''
        # Simulate data
        sim_data = self.stw_net.sim_network()
        # Compute true sample variance of psi and covariance of psi and alpha
        psi_var = np.var(sim_data['psi'])
        psi_alpha_cov = np.cov(sim_data['psi'], sim_data['alpha'])[0, 1]
        # Use data to create TwoWay object
        tw_net = tw.TwoWay(sim_data)
        # Estimate FE model
        tw_net.fit_fe(user_fe=fe_params)
        # Save results
        fe_res = tw_net.fe_res
        # Estimate CRE model
        tw_net.fit_cre(user_cre=cre_params, user_cluster=cluster_params)
        # Save results
        cre_res = tw_net.cre_res

        return psi_var, psi_alpha_cov, \
                fe_res['var_fe'], fe_res['cov_fe'], \
                fe_res['var_ho'], fe_res['cov_ho'], \
                cre_res['tot_var'], cre_res['tot_cov']

    def twfe_monte_carlo(self, N=10, ncore=1, fe_params={}, cre_params={}, cluster_params={}):
        '''
        Run Monte Carlo simulations of TwoWay to see the distribution of the true vs. estimated variance of psi and covariance between psi and alpha. Saves the following results in the dictionary self.res:

            true_psi_var (NumPy Array): true simulated sample variance of psi

            true_psi_alpha_cov (NumPy Array): true simulated sample covariance of psi and alpha

            fe_psi_var (NumPy Array): AKM estimate of variance of psi

            fe_psi_alpha_cov (NumPy Array): AKM estimate of covariance of psi and alpha

            fe_corr_psi_var (NumPy Array): bias-corrected AKM estimate of variance of psi

            fe_corr_psi_alpha_cov (NumPy Array): bias-corrected AKM estimate of covariance of psi and alpha

            cre_psi_var (NumPy Array): CRE estimate of variance of psi

            cre_psi_alpha_cov (NumPy Array): CRE estimate of covariance of psi and alpha

        Arguments:
            N (int): number of simulations
            ncore (int): how many cores to use
            fe_params (dict): dictionary of parameters for FE estimation

                Dictionary parameters:

                    ncore (int): number of cores to use

                    batch (int): batch size to send in parallel

                    ndraw_pii (int): number of draws to use in approximation for leverages

                    levfile (str): file to load precomputed leverages

                    ndraw_tr (int): number of draws to use in approximation for traces

                    h2 (bool): if True, compute h2 correction

                    out (str): outputfile where results are saved

                    statsonly (bool): if True, return only basic statistics

                    Q (str): which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'

            cre_params (dict): dictionary of parameters for CRE estimation

                Dictionary parameters:

                    ncore (int): number of cores to use

                    ndraw_tr (int): number of draws to use in approximation for traces

                    ndp (int): number of draw to use in approximation for leverages

                    out (string): outputfile

                    posterior (bool): compute posterior variance

                    wo_btw (bool): sets between variation to 0, pure RE

            cluster_params (dict): dictionary of parameters for clustering in CRE estimation

                Dictionary parameters:

                    cdf_resolution (int): how many values to use to approximate the cdf

                    grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)

                    stayers_movers (str or None): if None, uses entire dataset; if 'stayers', uses only stayers; if 'movers', uses only movers

                    t (int or None): if None, uses entire dataset; if int, gives time in data to consider (only valid for non-collapsed data)

                    weighted (bool): if True, weight firm clusters by firm size (if a weight column is included, firm weight is computed using this column; otherwise, each observation has weight 1)

                    dropna (bool): if True, drop observations where firms aren't clustered; if False, keep all observations

                    user_KMeans (dict): parameters for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
        '''
        # Initialize NumPy arrays to store results
        true_psi_var = np.zeros(N)
        true_psi_alpha_cov = np.zeros(N)
        fe_psi_var = np.zeros(N)
        fe_psi_alpha_cov = np.zeros(N)
        fe_corr_psi_var = np.zeros(N)
        fe_corr_psi_alpha_cov = np.zeros(N)
        cre_psi_var = np.zeros(N)
        cre_psi_alpha_cov = np.zeros(N)

        # Use multi-processing
        if ncore > 1:
            # Simulate networks
            with Pool(processes=ncore) as pool:
                V = pool.starmap(self._twfe_monte_carlo_interior, [(fe_params, cre_params, cluster_params) for _ in range(N)])
            for i, res in enumerate(V):
                true_psi_var[i], true_psi_alpha_cov[i], fe_psi_var[i], fe_psi_alpha_cov[i], fe_corr_psi_var[i], fe_corr_psi_alpha_cov[i], cre_psi_var[i], cre_psi_alpha_cov[i] = res
        else:
            for i in trange(N):
                # Simulate a network
                true_psi_var[i], true_psi_alpha_cov[i], fe_psi_var[i], fe_psi_alpha_cov[i], fe_corr_psi_var[i], fe_corr_psi_alpha_cov[i], cre_psi_var[i], cre_psi_alpha_cov[i] = self._twfe_monte_carlo_interior(fe_params=fe_params, cre_params=cre_params, cluster_params=cluster_params)

        res = {}

        res['true_psi_var'] = true_psi_var
        res['true_psi_alpha_cov'] = true_psi_alpha_cov
        res['fe_psi_var'] = fe_psi_var
        res['fe_psi_alpha_cov'] = fe_psi_alpha_cov
        res['fe_corr_psi_var'] = fe_corr_psi_var
        res['fe_corr_psi_alpha_cov'] = fe_corr_psi_alpha_cov
        res['cre_psi_var'] = cre_psi_var
        res['cre_psi_alpha_cov'] = cre_psi_alpha_cov

        self.res = res
        self.monte_carlo_res = True

    def plot_monte_carlo(self):
        '''
        Plot results from Monte Carlo simulations.
        '''
        if not self.monte_carlo_res:
            print('Must run Monte Carlo simulations before results can be plotted. This can be done by running network_name.twfe_monte_carlo(self, N=10, ncore=1, fe_params={}, cre_params={}, cluster_params={})')

        else:
            # Extract results
            true_psi_var = self.res['true_psi_var']
            true_psi_alpha_cov = self.res['true_psi_alpha_cov']
            fe_psi_var = self.res['fe_psi_var']
            fe_psi_alpha_cov = self.res['fe_psi_alpha_cov']
            fe_corr_psi_var = self.res['fe_corr_psi_var']
            fe_corr_psi_alpha_cov = self.res['fe_corr_psi_alpha_cov']
            cre_psi_var = self.res['cre_psi_var']
            cre_psi_alpha_cov = self.res['cre_psi_alpha_cov']

            # Define differences
            fe_psi_diff = sorted(fe_psi_var - true_psi_var)
            fe_psi_alpha_diff = sorted(fe_psi_alpha_cov - true_psi_alpha_cov)
            fe_corr_psi_diff = sorted(fe_corr_psi_var - true_psi_var)
            fe_corr_psi_alpha_diff = sorted(fe_corr_psi_alpha_cov - true_psi_alpha_cov)
            cre_psi_diff = sorted(cre_psi_var - true_psi_var)
            cre_psi_alpha_diff = sorted(cre_psi_alpha_cov - true_psi_alpha_cov)

            # Plot histograms
            # First, var(psi)
            plt.axvline(x=0, color='purple', linestyle='--', label=r'$\Delta$Truth=0')
            plt.hist(fe_psi_diff, bins=50, label='AKM var(psi)')
            plt.hist(fe_corr_psi_diff, bins=50, label='Bias-corrected AKM var(psi)')
            plt.hist(cre_psi_diff, bins=50, label='CRE var(psi)')
            plt.legend()
            plt.show()

            # Second, cov(psi, alpha)
            plt.axvline(x=0, color='purple', linestyle='--', label=r'$\Delta$Truth=0')
            plt.hist(fe_psi_alpha_diff, bins=50, label='AKM cov(psi, alpha)')
            plt.hist(fe_corr_psi_alpha_diff, bins=50, label='Bias-corrected AKM cov(psi, alpha)')
            plt.hist(cre_psi_alpha_diff, bins=50, label='CRE cov(psi, alpha)')
            plt.legend()
            plt.show()
