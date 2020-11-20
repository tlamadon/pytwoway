'''
Class for a simulated two-way fixed effect network
'''

import logging
from multiprocessing import Pool
import numpy as np
from numpy import matlib
import pandas as pd
from random import choices
from scipy.stats import mode, norm
from scipy.linalg import eig
inv, ax, randint = np.linalg.inv, np.newaxis, np.random.randint
repmat, rand = np.matlib.repmat, np.random.rand
from matplotlib import pyplot as plt
from twfe_network import twfe_network

class sim_twfe_network:
    '''
    Class of sim_twfe_network, where sim_twfe_network simulates a network of firms and workers. This class has the following functions:<br/>
        __init__(): initialize<br/>
        update_dict(): update values in parameter dictionaries (this function is similar to, but different from dict.update())
        sim_network_gen_fe(): generate fixed effects values for simulated panel data corresponding to the calibrated model (only for simulated data)<br/>
        sim_network_draw_fids(): draw firm ids for individual, given data that is grouped by worker id, spell id, and firm type (only for simulated data)<br/>
        sim_network(): simulate panel data corresponding to the calibrated model (only for simulated data)<br/>
        twfe_monte_carlo_interior(): interior function for twfe_monte_carlo()<br/>
        twfe_monte_carlo(): run Monte Carlo simulations of sim_twfe_network<br/>
        plot_monte_carlo(): plot results from Monte Carlo simulations
    '''

    def __init__(self, sim_params={}):
        '''
        Purpose:
            Initialize sim_twfe_network object.

        Arguments:
            sim_params (dict): parameters for simulated data<br/>
                <ul>
                <li>Dictionary parameters:</li>
                    <ul><ul>
                    <li>num_ind (int): number of workers</li>
                    <li>num_time (int): time length of panel</li>
                    <li>firm_size (int): max number of individuals per firm</li>
                    <li>nk (int): number of firm types</li>
                    <li>nl (int): number of worker types</li>
                    <li>alpha_sig (float): standard error of individual fixed effect (volatility of worker effects)</li>
                    <li>psi_sig (float): standard error of firm fixed effect (volatility of firm effects)</li>
                    <li>w_sig (float): standard error of residual in AKM wage equation (volatility of wage shocks)</li>
                    <li>csort (float): sorting effect</li>
                    <li>cnetw (float): network effect</li>
                    <li>csig (float): standard error of sorting/network effects</li>
                    <li>p_move (float): probability a worker moves firms in any period</li>
                    </ul></ul>
                </ul>
        '''
        logger.info('initializing sim_twfe_network object')

        # Define default parameter dictionaries
        self.default_sim_params = {'num_ind': 10000, 'num_time': 5, 'firm_size': 50, 'nk': 10, 'nl': 5, 'alpha_sig': 1, 'psi_sig': 1, 'w_sig': 1, 'csort': 1, 'cnetw': 1, 'csig': 1, 'p_move': 0.5}

        # Update parameters to include user parameters
        self.sim_params = self.update_dict(self.default_sim_params, sim_params)

        # Prevent plotting unless results exist
        self.monte_carlo_res = False

    def update_dict(self, default_params, user_params):
        '''
        Purpose:
            Replace entries in default_params with values in user_params. This function allows user_params to include only a subset of the required parameters in the dictionary.

        Arguments:
            default_params (dict): default parameter values
            user_params (dict): user selected parameter values

        Returns:
            params (dict): default_params updated with parameter values in user_params
        '''
        params = default_params.copy()

        params.update(user_params)

        return params

    def sim_network_gen_fe(self, sim_params):
        '''
        Purpose:
            Generate fixed effects values for simulated panel data corresponding to the calibrated model.

        Arguments:
            sim_params (dict): parameters for simulated data<br/>
                <ul>
                <li>Dictionary parameters:</li>
                    <ul><ul>
                    <li>num_ind (int): number of workers</li>
                    <li>num_time (int): time length of panel</li>
                    <li>firm_size (int): max number of individuals per firm</li>
                    <li>nk (int): number of firm types</li>
                    <li>nl (int): number of worker types</li>
                    <li>alpha_sig (float): standard error of individual fixed effect (volatility of worker effects)</li>
                    <li>psi_sig (float): standard error of firm fixed effect (volatility of firm effects)</li>
                    <li>w_sig (float): standard error of residual in AKM wage equation (volatility of wage shocks)</li>
                    <li>csort (float): sorting effect</li>
                    <li>cnetw (float): network effect</li>
                    <li>csig (float): standard error of sorting/network effects</li>
                    <li>p_move (float): probability a worker moves firms in any period</li>
                    </ul></ul>
                </ul>

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

    def sim_network_draw_fids(self, freq, num_time, firm_size):
        '''
        Purpose:
            Draw firm ids for individual, given data that is grouped by worker id, spell id, and firm type.

        Arguments:
            freq (NumPy Array): size of groups (groups by worker id, spell id, and firm type)
            num_time (int): time length of panel
            firm_size (int): max number of individuals per firm

        Returns:
            (NumPy Array): random firms for each group
        '''
        max_int = np.int(np.maximum(1, freq.sum() / (firm_size * num_time)))
        return np.array(np.random.choice(max_int, size=freq.count()) + 1)

    def sim_network(self):
        '''
        Purpose:
            Simulate panel data corresponding to the calibrated model.

        Returns:
            data (Pandas DataFrame): simulated network
        '''
        # Generate fixed effects
        psi, alpha, G, H = self.sim_network_gen_fe(self.sim_params)

        # Extract parameters
        num_ind, num_time, firm_size = self.sim_params['num_ind'], self.sim_params['num_time'], self.sim_params['firm_size']
        nk, nl, w_sig, p_move = self.sim_params['nk'], self.sim_params['nl'], self.sim_params['w_sig'], self.sim_params['p_move']

        # Generate empty NumPy arrays
        network = np.zeros((num_ind, num_time), dtype=int)
        spellcount = np.ones((num_ind, num_time))

        # Random draws of worker types for all individuals in panel
        sim_worker_types = randint(low=1, high=nl, size=num_ind)

        for i in range(0, num_ind):
            l = sim_worker_types[i]
            # At time 1, we draw from H for initial firm
            network[i, 0] = choices(range(0, nk), H[l, :])[0]

            for t in range(1, num_time):
                # Hit moving shock
                if rand() < p_move:
                    network[i, t] = choices(range(0, nk), G[l, network[i, t - 1], :])[0]
                    spellcount[i, t] = spellcount[i, t - 1] + 1
                else:
                    network[i, t] = network[i, t - 1]
                    spellcount[i, t] = spellcount[i, t - 1]

        # Compiling IDs and timestamps
        ids = np.reshape(np.outer(range(1, num_ind + 1), np.ones(num_time)), (num_time * num_ind, 1))
        ids = ids.astype(int)[:, 0]
        ts = np.reshape(repmat(range(1, num_time + 1), num_ind, 1), (num_time * num_ind, 1))
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
        data = pd.DataFrame(data={'wid': ids, 'year': ts, 'k': k_data,
                                'alpha': alpha_data, 'psi': psi_data,
                                'spell': spell_data.astype(int)})

        # Generate size of spells
        dspell = data.groupby(['wid', 'spell', 'k']).size().to_frame(name='freq').reset_index()
        # Draw firm ids
        dspell['fid'] = dspell.groupby(['k'])['freq'].transform(self.sim_network_draw_fids, *[num_time, firm_size])
        # Make firm ids contiguous (and have them start at 1)
        dspell['fid'] = dspell.groupby(['k', 'fid'])['freq'].ngroup() + 1

        # Merge spells into panel
        data = data.merge(dspell, on=['wid', 'spell', 'k'])

        data['move'] = (data['fid'] != data['fid'].shift(1)) & (data['wid'] == data['wid'].shift(1))

        # Compute wages through the AKM formula
        data['comp'] = data['alpha'] + data['psi'] + w_sig * norm.rvs(size=num_ind * num_time)

        return data

    def twfe_monte_carlo_interior(self, akm_params={}, cre_params={}, cluster_params={}):
        '''
        Purpose:
            Run Monte Carlo simulations of twfe_network to see the distribution of the true vs. estimated variance of psi and covariance between psi and alpha. This is the interior function to twfe_monte_carlo.

        Arguments:
            akm_params (dict): dictionary of parameters for bias-corrected AKM estimation
                <ul>
                <li>Dictionary parameters:</li>
                    <ul><ul>
                    <li>ncore (int): number of cores to use</li>
                    <li>batch (int): batch size to send in parallel</li>
                    <li>ndraw_pii (int): number of draw to use in approximation for leverages</li>
                    <li>ndraw_tr (int): number of draws to use in approximation for traces</li>
                    <li>check (bool): whether to compute the non-approximated estimates as well</li>
                    <li>hetero (bool): whether to compute the heteroskedastic estimates</li>
                    <li>out (string): outputfile</li>
                    <li>con (string): computes the smallest eigen values, this is the filepath where these results are saved</li>
                    <li>logfile (string): log output to a logfile</li>
                    <li>levfile (string): file to load precomputed leverages</li>
                    <li>statsonly (bool): save data statistics only</li>
                    </ul></ul>
                </ul>
            cre_params (dict): dictionary of parameters for CRE estimation
                <ul>
                <li>Dictionary parameters:</li>
                    <ul><ul>
                    <li>ncore (int): number of cores to use</li>
                    <li>ndraw_tr (int): number of draws to use in approximation for traces</li>
                    <li>ndp (int): number of draw to use in approximation for leverages</li>
                    <li>out (string): outputfile</li>
                    <li>posterior (bool): compute posterior variance</li>
                    <li>wobtw (bool): sets between variation to 0, pure RE</li>
                    </ul></ul>
                </ul>
            cluster_params (dict): dictionary of parameters for clustering in CRE estimation
                <ul>
                <li>Dictionary parameters:</li>
                    <ul><ul>
                    <li>cdf_resolution (int): how many values to use to approximate the cdf</li>
                    <li>grouping (string): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)</li>
                    <li>year (int): if None, uses entire dataset; if int, gives year of data to consider</li>
                    <li>KMeans_params (dict): use parameters defined in KMeans_dict for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and use default parameters defined in class attribute default_KMeans for any parameters not specified</li>
                    </ul></ul>
                </ul>

        Returns:
            true_psi_var (float): true simulated sample variance of psi
            true_psi_alpha_cov (float): true simulated sample covariance of psi and alpha
            akm_psi_var (float): AKM estimate of variance of psi
            akm_psi_alpha_cov (float): AKM estimate of covariance of psi and alpha
            akm_corr_psi_var (float): bias-corrected AKM estimate of variance of psi
            akm_corr_psi_alpha_cov (float): bias-corrected AKM estimate of covariance of psi and alpha
            cre_psi_var (float): CRE estimate of variance of psi
            cre_psi_alpha_cov (float): CRE estimate of covariance of psi and alpha
        '''
        # Simulate data
        sim_data = self.sim_network()
        # Compute true sample variance of psi and covariance of psi and alpha
        psi_var = np.var(sim_data['psi'])
        psi_alpha_cov = np.cov(sim_data['psi'], sim_data['alpha'])[0, 1]
        # Use data to create twfe_network object
        nw = twfe_network(data=sim_data)
        # Convert into event study
        nw.refactor_es()
        # Estimate AKM model
        akm_res = nw.run_akm_corrected(user_akm=akm_params)
        # Cluster for CRE model
        nw.cluster(user_cluster=cluster_params)
        # Estimate CRE model
        cre_res = nw.run_cre(user_cre=cre_params)

        return psi_var, psi_alpha_cov, \
                akm_res['var_fe'], akm_res['cov_fe'], \
                akm_res['var_ho'], akm_res['cov_ho'], \
                cre_res['var_wt'] + cre_res['var_bw'], cre_res['cov_wt'] + cre_res['cov_bw']
        

    def twfe_monte_carlo(self, N=10, ncore=1, akm_params={}, cre_params={}, cluster_params={}):
        '''
        Purpose:
            Run Monte Carlo simulations of twfe_network to see the distribution of the true vs. estimated variance of psi and covariance between psi and alpha. Saves the following results in the dictionary self.res:
                <ul>
                <li>true_psi_var (NumPy Array): true simulated sample variance of psi</li>
                <li>true_psi_alpha_cov (NumPy Array): true simulated sample covariance of psi and alpha</li>
                <li>akm_psi_var (NumPy Array): AKM estimate of variance of psi</li>
                <li>akm_psi_alpha_cov (NumPy Array): AKM estimate of covariance of psi and alpha</li>
                <li>akm_corr_psi_var (NumPy Array): bias-corrected AKM estimate of variance of psi</li>
                <li>akm_corr_psi_alpha_cov (NumPy Array): bias-corrected AKM estimate of covariance of psi and alpha</li>
                <li>cre_psi_var (NumPy Array): CRE estimate of variance of psi</li>
                <li>cre_psi_alpha_cov (NumPy Array): CRE estimate of covariance of psi and alpha</li>
                </ul>

        Arguments:
            N (int): number of simulations
            ncore (int): how many cores to use
            akm_params (dict): dictionary of parameters for bias-corrected AKM estimation
                <ul>
                <li>Dictionary parameters:</li>
                    <ul><ul>
                    <li>ncore (int): number of cores to use</li>
                    <li>batch (int): batch size to send in parallel</li>
                    <li>ndraw_pii (int): number of draw to use in approximation for leverages</li>
                    <li>ndraw_tr (int): number of draws to use in approximation for traces</li>
                    <li>check (bool): whether to compute the non-approximated estimates as well</li>
                    <li>hetero (bool): whether to compute the heteroskedastic estimates</li>
                    <li>out (string): outputfile</li>
                    <li>con (string): computes the smallest eigen values, this is the filepath where these results are saved</li>
                    <li>logfile (string): log output to a logfile</li>
                    <li>levfile (string): file to load precomputed leverages</li>
                    <li>statsonly (bool): save data statistics only</li>
                    </ul></ul>
                </ul>
            cre_params (dict): dictionary of parameters for CRE estimation
                <ul>
                <li>Dictionary parameters:</li>
                    <ul><ul>
                    <li>ncore (int): number of cores to use</li>
                    <li>ndraw_tr (int): number of draws to use in approximation for traces</li>
                    <li>ndp (int): number of draw to use in approximation for leverages</li>
                    <li>out (string): outputfile</li>
                    <li>posterior (bool): compute posterior variance</li>
                    <li>wobtw (bool): sets between variation to 0, pure RE</li>
                    </ul></ul>
                </ul>
            cluster_params (dict): dictionary of parameters for clustering in CRE estimation
                <ul>
                <li>Dictionary parameters:</li>
                    <ul><ul>
                    <li>cdf_resolution (int): how many values to use to approximate the cdf</li>
                    <li>grouping (string): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)</li>
                    <li>year (int): if None, uses entire dataset; if int, gives year of data to consider</li>
                    <li>KMeans_params (dict): use parameters defined in KMeans_dict for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and use default parameters defined in class attribute default_KMeans for any parameters not specified</li>
                    </ul></ul>
                </ul>
        '''
        # Initialize NumPy arrays to store results
        true_psi_var = np.zeros(N)
        true_psi_alpha_cov = np.zeros(N)
        akm_psi_var = np.zeros(N)
        akm_psi_alpha_cov = np.zeros(N)
        akm_corr_psi_var = np.zeros(N)
        akm_corr_psi_alpha_cov = np.zeros(N)
        cre_psi_var = np.zeros(N)
        cre_psi_alpha_cov = np.zeros(N)

        # Use multi-processing
        if ncore > 1:
            V = []
            # Simulate networks
            with Pool(processes=ncore) as pool:
                V = pool.starmap(self.twfe_monte_carlo_interior, [[akm_params, cre_params, cluster_params] for _ in range(N)])
            for i, res in enumerate(V):
                true_psi_var[i], true_psi_alpha_cov[i], akm_psi_var[i], akm_psi_alpha_cov[i], akm_corr_psi_var[i], akm_corr_psi_alpha_cov[i], cre_psi_var[i], cre_psi_alpha_cov[i] = res
        else:
            for i in range(N):
                # Simulate a network
                true_psi_var[i], true_psi_alpha_cov[i], akm_psi_var[i], akm_psi_alpha_cov[i], akm_corr_psi_var[i], akm_corr_psi_alpha_cov[i], cre_psi_var[i], cre_psi_alpha_cov[i] = self.twfe_monte_carlo_interior(akm_params=akm_params, cre_params=cre_params, cluster_params=cluster_params)

        res = {}

        res['true_psi_var'] = true_psi_var
        res['true_psi_alpha_cov'] = true_psi_alpha_cov
        res['akm_psi_var'] = akm_psi_var
        res['akm_psi_alpha_cov'] = akm_psi_alpha_cov
        res['akm_corr_psi_var'] = akm_corr_psi_var
        res['akm_corr_psi_alpha_cov'] = akm_corr_psi_alpha_cov
        res['cre_psi_var'] = cre_psi_var
        res['cre_psi_alpha_cov'] = cre_psi_alpha_cov

        self.res = res
        self.monte_carlo_res = True

    def plot_monte_carlo(self):
        '''
        Purpose:
            Plot results from Monte Carlo simulations.
        '''
        if not self.monte_carlo_res:
            print('Must run Monte Carlo simulations before results can be plotted. This can be done by running network_name.twfe_monte_carlo(self, N=10, ncore=1, akm_params={}, cre_params={}, cluster_params={})')

        else:
            # Extract results
            true_psi_var = self.res['true_psi_var']
            true_psi_alpha_cov = self.res['true_psi_alpha_cov']
            akm_psi_var = self.res['akm_psi_var']
            akm_psi_alpha_cov = self.res['akm_psi_alpha_cov']
            akm_corr_psi_var = self.res['akm_corr_psi_var']
            akm_corr_psi_alpha_cov = self.res['akm_corr_psi_alpha_cov']
            cre_psi_var = self.res['cre_psi_var']
            cre_psi_alpha_cov = self.res['cre_psi_alpha_cov']

            # Define differences
            akm_psi_diff = sorted(akm_psi_var - true_psi_var)
            akm_psi_alpha_diff = sorted(akm_psi_alpha_cov - true_psi_alpha_cov)
            akm_corr_psi_diff = sorted(akm_corr_psi_var - true_psi_var)
            akm_corr_psi_alpha_diff = sorted(akm_corr_psi_alpha_cov - true_psi_alpha_cov)
            cre_psi_diff = sorted(cre_psi_var - true_psi_var)
            cre_psi_alpha_diff = sorted(cre_psi_alpha_cov - true_psi_alpha_cov)

            # Plot histograms
            # First, var(psi)
            plt.hist(akm_psi_diff, bins=50, label='AKM var(psi)')
            plt.hist(akm_corr_psi_diff, bins=50, label='Bias-corrected AKM var(psi)')
            plt.hist(cre_psi_diff, bins=50, label='CRE var(psi)')
            plt.legend()
            plt.show()

            # Second, cov(psi, alpha)
            plt.hist(akm_psi_alpha_diff, bins=50, label='AKM cov(psi, alpha)')
            plt.hist(akm_corr_psi_alpha_diff, bins=50, label='Bias-corrected AKM cov(psi, alpha)')
            plt.hist(cre_psi_alpha_diff, bins=50, label='CRE cov(psi, alpha)')
            plt.legend()
            plt.show()

# Begin logging
logger = logging.getLogger('sim_twfe')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('sim_twfe_spam.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)
