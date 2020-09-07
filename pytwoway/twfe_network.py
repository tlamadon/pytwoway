'''
Class for a two-way fixed effect network
'''

import logging
from multiprocessing import Pool
from tqdm.auto import tqdm
import numpy as np
from numpy import matlib
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from random import choices
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import mode, norm
from scipy.linalg import eig
inv, ax, randint = np.linalg.inv, np.newaxis, np.random.randint
repmat, rand = np.matlib.repmat, np.random.rand
from pytwoway import fe_approximate_correction_full as feacf
from pytwoway import cre

# Testing
# data = pd.read_feather('../../Google Drive File Stream/.shortcut-targets-by-id/1iN9LApqNxHmVCOV4IUISMwPS7KeZcRhz/ra-adam/data/English/worker_cleaned.ftr')
# col_dict = {'fid': 'codf', 'wid': 'codf_w', 'year': 'year', 'comp': 'comp_current'}
# d_net for data_network
# d_net = twfe_network(data=data, formatting='long', col_dict=col_dict)
# d_net.refactor_es()
# d_net.data_validity()
# d_net.cluster()
# akm_res = d_net.run_akm_corrected()
# cre_res = d_net.run_cre()

# d_net = twfe_network()
# cdfs_1 = d_net.approx_cdfs()
# d_net.refactor_es()
# cdfs_2 = d_net.approx_cdfs()

# from matplotlib import pyplot as plt
# true_psi_var, true_psi_alpha_cov, akm_psi_var, akm_psi_alpha_cov, cre_psi_var, cre_psi_alpha_cov = twfe_monte_carlo(N=100, ncore=4)
# akm_psi_diff = sorted(akm_psi_var - true_psi_var)
# akm_psi_alpha_diff = sorted(akm_psi_alpha_cov - true_psi_alpha_cov)
# cre_psi_diff = sorted(cre_psi_var - true_psi_var)
# cre_psi_alpha_diff = sorted(cre_psi_alpha_cov - true_psi_alpha_cov)
# plt.hist(akm_psi_diff, label='AKM var(psi)')
# plt.hist(cre_psi_diff, label='CRE var(psi)')
# plt.legend()
# plt.show()
# plt.hist(akm_psi_alpha_diff, label='AKM cov(psi, alpha)')
# plt.hist(cre_psi_alpha_diff, label='CRE cov(psi, alpha)')
# plt.legend()
# plt.show()

def twfe_monte_carlo_interior(params={'num_ind': 10000, 'num_time': 5, 'firm_size': 50, 'nk': 200, 'nl': 50, 'alpha_sig': 1, 'psi_sig': 1, 'w_sig': 1, 'csort': 1, 'cnetw': 1, 'csig': 1, 'p_move': 0.5}):
    '''
    Purpose:
        Run Monte Carlo simulations of twfe_network to see the distribution of the true vs. estimated variance of psi and covariance between psi and alpha. This is the interior function to twfe_monte_carlo

    Inputs:
        params (dictionary): parameters for simulated data
            Dictionary parameters:
                num_ind: number of workers
                num_time: time length of panel
                firm_size: max number of individuals per firm
                nk: number of firm types
                nl: number of worker types
                alpha_sig: standard error of individual fixed effect (volatility of worker effects)
                psi_sig: standard error of firm fixed effect (volatility of firm effects)
                w_sig: standard error of residual in AKM wage equation (volatility of wage shocks)
                csort: sorting effect
                cnetw: network effect
                csig: standard error of sorting/network effects
                p_move: probability a worker moves firms in any period

    Returns:
        true_psi_var (float): true simulated sample variance of psi
        true_psi_alpha_cov (float): true simulated sample covariance of psi and alpha
        akm_psi_var (float): bias-corrected AKM estimate of variance of psi
        akm_psi_alpha_cov (float): bias-corrected AKM estimate of covariance of psi and alpha
        cre_psi_var (float): CRE estimate of variance of psi
        cre_psi_alpha_cov (float): CRE estimate of covariance of psi and alpha
    '''
    # Simulate network
    nw = twfe_network(data=params)
    # Compute true sample variance of psi and covariance of psi and alpha
    psi_var = np.var(nw.data['psi'])
    psi_alpha_cov = np.cov(nw.data['psi'], nw.data['alpha'])[0, 1]
    # Convert into event study
    nw.refactor_es()
    # Estimate bias-corrected AKM model
    akm_res = nw.run_akm_corrected()
    # Cluster for CRE model
    nw.cluster()
    # Estimate CRE model
    cre_res = nw.run_cre()

    return psi_var, psi_alpha_cov, akm_res['var_ho'], akm_res['cov_ho'], cre_res['var_bw'], cre_res['cov_bw']
    

def twfe_monte_carlo(params={'num_ind': 10000, 'num_time': 5, 'firm_size': 50, 'nk': 200, 'nl': 50, 'alpha_sig': 1, 'psi_sig': 1, 'w_sig': 1, 'csort': 1, 'cnetw': 1, 'csig': 1, 'p_move': 0.5}, N=500, ncore=1):
    '''
    Purpose:
        Run Monte Carlo simulations of twfe_network to see the distribution of the true vs. estimated variance of psi and covariance between psi and alpha

    Inputs:
        params (dictionary): parameters for simulated data
            Dictionary parameters:
                num_ind: number of workers
                num_time: time length of panel
                firm_size: max number of individuals per firm
                nk: number of firm types
                nl: number of worker types
                alpha_sig: standard error of individual fixed effect (volatility of worker effects)
                psi_sig: standard error of firm fixed effect (volatility of firm effects)
                w_sig: standard error of residual in AKM wage equation (volatility of wage shocks)
                csort: sorting effect
                cnetw: network effect
                csig: standard error of sorting/network effects
                p_move: probability a worker moves firms in any period
        N (int): number of simulations
        ncore (int): how many cores to use

    Returns:
        true_psi_var (NumPy array): true simulated sample variance of psi
        true_psi_alpha_cov (NumPy array): true simulated sample covariance of psi and alpha
        akm_psi_var (NumPy array): bias-corrected AKM estimate of variance of psi
        akm_psi_alpha_cov (NumPy array): bias-corrected AKM estimate of covariance of psi and alpha
        cre_psi_var (NumPy array): CRE estimate of variance of psi
        cre_psi_alpha_cov (NumPy array): CRE estimate of covariance of psi and alpha
    '''
    # Initialize NumPy arrays to store results
    true_psi_var = np.zeros(N)
    true_psi_alpha_cov = np.zeros(N)
    akm_psi_var = np.zeros(N)
    akm_psi_alpha_cov = np.zeros(N)
    cre_psi_var = np.zeros(N)
    cre_psi_alpha_cov = np.zeros(N)

    # Use multi-processing
    if ncore > 1:
        V = []
        # Simulate networks
        with Pool(processes=ncore) as pool:
            V = pool.starmap(twfe_monte_carlo_interior, [[params] for _ in range(N)])
        for i, res in enumerate(V):
            true_psi_var[i], true_psi_alpha_cov[i], akm_psi_var[i], akm_psi_alpha_cov[i], cre_psi_var[i], cre_psi_alpha_cov[i] = res
    else:
        for i in range(N):
            # Simulate a network
            true_psi_var[i], true_psi_alpha_cov[i], akm_psi_var[i], akm_psi_alpha_cov[i], cre_psi_var[i], cre_psi_alpha_cov[i] = twfe_monte_carlo_interior(params)

    return true_psi_var, true_psi_alpha_cov, akm_psi_var, akm_psi_alpha_cov, cre_psi_var, cre_psi_alpha_cov

class twfe_network:
    '''
    Class of twfe_network, where twfe_network gives a network of firms and workers. This class has the following functions:
        __init__(): initialize
        update_cols(): rename columns and keep only relevant columns
        n_workers(): get the number of unique workers
        n_firms(): get the number of unique firms
        data_validity(): check that data is formatted correctly
        conset(): update data to include only the largest connected set of movers, and if firm ids are contiguous, also return the NetworkX Graph
        contiguous_fids(): make firm ids contiguous
        refactor_es(): refactor long form data into event study data
        approx_cdfs(): generate cdfs of compensation for firms
        cluster(): cluster data and assign a new column giving the cluster for each firm
        run_akm_corrected(): run bias-corrected AKM estimator
        run_cre(): run CRE estimator
        sim_network_gen_fe(): generate fixed effects values for simulated panel data corresponding to the calibrated model (only for simulated data)
        sim_network_draw_fids(): draw firm ids for individual, given data that is grouped by worker id, spell id, and firm type (only for simulated data)
        sim_network(): simulate panel data corresponding to the calibrated model (only for simulated data)
    '''

    def __init__(self, data={'num_ind': 10000, 'num_time': 5, 'firm_size': 50, 'nk': 200, 'nl': 50, 'alpha_sig': 1, 'psi_sig': 1, 'w_sig': 1, 'csort': 1, 'cnetw': 1, 'csig': 1, 'p_move': 0.5}, formatting='long', col_dict=False):
        '''
        Purpose:
            Initialize twfe_network object

        Inputs:
            data (dict or Pandas DataFrame): if dict, simulate network of firms and workers using parameter values in dictionary; if Pandas DataFrame, then real data giving firms and workers
                Dictionary parameters:
                    num_ind: number of workers
                    num_time: time length of panel
                    firm_size: max number of individuals per firm
                    nk: number of firm types
                    nl: number of worker types
                    alpha_sig: standard error of individual fixed effect (volatility of worker effects)
                    psi_sig: standard error of firm fixed effect (volatility of firm effects)
                    w_sig: standard error of residual in AKM wage equation (volatility of wage shocks)
                    csort: sorting effect
                    cnetw: network effect
                    csig: standard error of sorting/network effects
                    p_move: probability a worker moves firms in any period
            formatting (string): if 'long', then data in long format; if 'es', then data in event study format. If simulating data, keep default value of 'long'
            col_dict (dictionary): make data columns readable (requires: wid (worker id), comp (compensation), fid (firm id), year if long; wid (worker id), y1 (compensation 1), y2 (compensation 2), f1i (firm id 1), f2i (firm id 2), m(0 if stayer, 1 if mover) if event study)

        Returns:
            Nothing
        '''
        logger.info('initializing twfe_network object')

        # Define some variables
        self.connected = False
        self.contiguous = False
        self.formatting = formatting
        self.col_dict = col_dict

        # Simulate data
        if isinstance(data, dict):
            self.data = self.sim_network(data)
        # Use given data
        else:
            self.data = data.dropna()
            # Make sure data is valid
            # Note that column names are corrected in this function if all columns are in the data
            self.data_validity()
            # Drop na values
            self.data = self.data.dropna()

        # Generate largest connected set
        self.conset()

        # Make firm ids contiguous
        self.contiguous_fids()

        # Using contiguous fids, get NetworkX Graph of largest connected set
        self.G = self.conset()

        # Check data validity after initial cleaning
        if isinstance(col_dict, dict):
            self.data_validity()

    def update_cols(self):
        '''
        Purpose:
            Rename columns and keep only relevant columns

        Inputs:
            Nothing

        Returns:
            Nothing
        '''
        if isinstance(self.col_dict, dict):
            if self.formatting == 'long':
                self.data = self.data.rename({self.col_dict['wid']: 'wid', self.col_dict['comp']: 'comp', self.col_dict['fid']: 'fid', self.col_dict['year']: 'year'}, axis=1)
                self.data = self.data[['wid', 'comp', 'fid', 'year']]
                self.col_dict = {'wid': 'wid', 'comp': 'comp', 'fid': 'fid', 'year': 'year'}
            elif self.formatting == 'es':
                self.data = self.data.rename({self.col_dict['wid']: 'wid', self.col_dict['y1']: 'y1', self.col_dict['y2']: 'y2', self.col_dict['f1i']: 'f1i', self.col_dict['f2i']: 'f2i', self.col_dict['m']: 'm'}, axis=1)
                self.data = self.data[['wid', 'y1', 'y2', 'f1i', 'f2i', 'm']]
                self.col_dict = {'wid': 'wid', 'y1': 'y1', 'y2': 'y2', 'f1i': 'f1i', 'f2i': 'f2i', 'm': 'm'}

    def n_workers(self):
        '''
        Purpose:
            Get the number of unique workers

        Inputs:
            Nothing

        Returns:
            (int): number of unique workers
        '''
        return len(self.data['wid'].unique())

    def n_firms(self):
        '''
        Purpose:
            Get the number of unique firms

        Inputs:
            Nothing

        Returns:
            (int): number of unique firms
        '''
        if self.formatting == 'long':
            return len(self.data['fid'].unique())
        elif self.formatting == 'es':
            return len(set(list(self.data['f1i'].unique()) + list(self.data['f2i'].unique())))

    def data_validity(self):
        '''
        Purpose:
            Check that data is formatted correctly. Results are logged

        Inputs:
            Nothing

        Returns:
            Nothing
        '''
        if self.formatting == 'long':
            success = True

            logger.info('--- checking columns ---')
            cols = True
            for col in ['wid', 'comp', 'fid', 'year']:
                if self.col_dict[col] not in self.data.columns:
                    logger.info(col, 'missing from data')
                    cols = False
                else:
                    if col == 'year':
                        if self.data[self.col_dict[col]].dtype != 'int':
                            logger.info(self.col_dict[col], 'has wrong dtype, should be int but is', self.data[self.col_dict[col]].dtype)
                            cols = False
                    elif col == 'comp':
                        if self.data[self.col_dict[col]].dtype != 'float64':
                            logger.info(self.col_dict[col], 'has wrong dtype, should be float64 but is', self.data[self.col_dict[col]].dtype)
                            cols = False

            logger.info('columns correct:' + str(cols))
            if not cols:
                success = False
                raise ValueError('Your data does not include the correct columns. The twfe_network object cannot be generated with your data.')
            else:
                # Correct column names
                self.update_cols()

            logger.info('--- checking worker-year observations ---')

            max_obs = self.data.groupby(['wid', 'year']).size().max()

            logger.info('max number of worker-year observations (should be 1):' + str(max_obs))
            if max_obs > 1:
                success = False

            logger.info('--- checking nan data ---')

            nan = self.data.shape[0] - self.data.dropna().shape[0]

            logger.info('data nan rows (should be 0):' + str(nan))
            if nan > 0:
                success = False

            logger.info('--- checking connected set ---')
            self.data['fid_max'] = self.data.groupby(['wid'])['fid'].transform(max)
            G = nx.from_pandas_edgelist(self.data, 'fid', 'fid_max')
            largest_cc = max(nx.connected_components(G), key=len)
            self.data = self.data.drop(['fid_max'], axis=1)

            outside_cc = self.data[(~self.data['fid'].isin(largest_cc))].shape[0]

            logger.info('observations outside connected set (should be 0):' + str(outside_cc))
            if outside_cc > 0:
                success = False

            logger.info('Overall success:' + str(success))

        elif self.formatting == 'es':
                success_stayers = True
                success_movers = True
                
                logger.info('--- checking columns ---')
                cols = True
                for col in ['wid', 'y1', 'y2', 'f1i', 'f2i', 'm']:
                    if self.col_dict[col] not in self.data.columns:
                        logger.info(col, 'missing from stayers')
                        cols = False
                    else:
                        if col in ['y1', 'y2']:
                            if self.data[self.col_dict[col]].dtype != 'float64':
                                logger.info(col, 'has wrong dtype, should be float64 but is', self.data[self.col_dict[col]].dtype)
                                cols = False
                        elif col == 'm':
                            if self.data[self.col_dict[col]].dtype != 'int':
                                logger.info(col, 'has wrong dtype, should be int but is', self.data[self.col_dict[col]].dtype)
                                cols = False

                logger.info('columns correct:' + str(cols))
                if not cols:
                    success_stayers = False
                    success_movers = False
                    raise ValueError('Your data does not include the correct columns. The twfe_network object cannot be generated with your data.')
                else:
                    # Correct column names
                    self.update_cols()

                stayers = self.data[self.data['m'] == 0]
                movers = self.data[self.data['m'] == 1]

                logger.info('--- checking rows ---')
                na_stayers = stayers.shape[0] - stayers.dropna().shape[0]
                na_movers = movers.shape[0] - movers.dropna().shape[0]

                logger.info('stayers nan rows (should be 0):' + str(na_stayers))
                logger.info('movers nan rows (should be 0):' + str(na_movers))
                if na_stayers > 0:
                    success_stayers = False
                if na_movers > 0:
                    success_movers = False

                logger.info('--- checking firms ---')
                firms_stayers = (stayers['f1i'] != stayers['f2i']).sum()
                firms_movers = (movers['f1i'] == movers['f2i']).sum()

                logger.info('stayers with different firms (should be 0):' + str(firms_stayers))
                logger.info('movers with same firm (should be 0):' + str(firms_movers))
                if firms_stayers > 0:
                    success_stayers = False
                if firms_movers > 0:
                    success_movers = False

                logger.info('--- checking income ---')
                income_stayers = (stayers['y1'] != stayers['y2']).sum()

                logger.info('stayers with different income (should be 0):' + str(income_stayers))
                if income_stayers > 0:
                    success_stayers = False

                logger.info('--- checking connected set ---')
                G = nx.from_pandas_edgelist(movers, 'f1i', 'f2i')
                largest_cc = max(nx.connected_components(G), key=len)

                cc_stayers = stayers[(~stayers['f1i'].isin(largest_cc)) | (~stayers['f2i'].isin(largest_cc))].shape[0]
                cc_movers = movers[(~movers['f1i'].isin(largest_cc)) | (~movers['f2i'].isin(largest_cc))].shape[0]

                logger.info('stayers outside connected set (should be 0):' + str(cc_stayers))
                logger.info('movers outside connected set (should be 0):' + str(cc_movers))
                if cc_stayers > 0:
                    success_stayers = False
                if cc_movers > 0:
                    success_movers = False

                logger.info('Overall success for stayers:' + str(success_stayers))
                logger.info('Overall success for movers:' + str(success_movers))

    def conset(self):
        '''
        Purpose:
            Update data to include only the largest connected set of movers, and if firm ids are contiguous, also return the NetworkX Graph

        Inputs:
            Nothing

        Returns:
            G (NetworkX Graph): largest connected set of movers (only returns if firm ids are contiguous, otherwise returns None)
        '''
        prev_workers = self.n_workers()
        if self.formatting == 'long':
            # Add max firm id per worker to serve as a central node for the worker
            # self.data['fid_f1'] = self.data.groupby('wid')['fid'].transform(lambda a: a.shift(-1)) # FIXME - this is directed but is much slower
            self.data['fid_max'] = self.data.groupby(['wid'])['fid'].transform(max) # FIXME - this is undirected but is much faster

            # Find largest connected set
            # Source: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html
            G = nx.from_pandas_edgelist(self.data, 'fid', 'fid_max')
            # Drop fid_max
            self.data = self.data.drop(['fid_max'], axis=1)
            # Update data if not connected
            if not self.connected:
                largest_cc = max(nx.connected_components(G), key=len)
                # Keep largest connected set of firms
                self.data = self.data[self.data['fid'].isin(largest_cc)]
        elif self.formatting == 'es':
            # Find largest connected set
            # Source: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html
            G = nx.from_pandas_edgelist(self.data, 'f1i', 'f2i')
            # Update data if not connected
            if not self.connected:
                largest_cc = max(nx.connected_components(G), key=len)
                # Keep largest connected set of firms
                self.data = self.data[(self.data['f1i'].isin(largest_cc)) & (self.data['f2i'].isin(largest_cc))]

        # Data is now connected
        self.connected = True

        # If connected data != full data, set contiguous to False
        if prev_workers != self.n_workers():
            self.contiguous = False
    
        # Return G if firm ids are contiguous (if they're not contiguous, they have to be updated first)
        if self.contiguous:
            return G

        return None

    def contiguous_fids(self):
        '''
        Purpose:
            Make firm ids contiguous

        Inputs:
            Nothing

        Returns:
            Nothing
        '''
        # Generate fid_list (note that all columns listed in fid_list are included in the set of firm ids, and all columns are adjusted to have the new, contiguous firm ids)
        if self.formatting == 'long':
            fid_list = ['fid']
        elif self.formatting == 'es':
            fid_list = ['f1i', 'f2i']
        # Create sorted set of unique fids
        fids = []
        for fid in fid_list:
            fids += list(self.data[fid].unique())
        fids = sorted(list(set(fids)))

        # Create list of adjusted fids
        adjusted_fids = np.linspace(0, len(fids) - 1, len(fids)).astype(int) + 1

        # Update each fid one at a time
        for fid in fid_list:
            # Create dictionary linking current to new fids, then convert into a dataframe for merging
            fids_dict = {fid: fids, 'adj_' + fid: adjusted_fids}
            fids_df = pd.DataFrame(fids_dict, index=adjusted_fids)

            # Merge new, contiguous fids into event study data
            self.data = self.data.merge(fids_df, how='left', on=fid)

            # Drop old fid column and rename contiguous fid column
            self.data = self.data.drop([fid], axis=1)
            self.data = self.data.rename({'adj_' + fid: fid}, axis=1)

        # Firm ids are now contiguous
        self.contiguous = True

    def refactor_es(self):
        '''
        Purpose:
            Refactor long form data into event study data

        Inputs:
            Nothing

        Returns:
            Nothing
        '''
        if self.formatting == 'long':
            # Sort data by wid and year
            self.data = self.data.sort_values(['wid', 'year'])
            logger.info('data sorted by wid and year')

            # Introduce lagged fid and wid
            self.data['fid_l1'] = self.data['fid'].shift(periods=1)
            self.data['wid_l1'] = self.data['wid'].shift(periods=1)
            logger.info('lagged fid introduced')

            # Generate spell ids
            # Source: https://stackoverflow.com/questions/59778744/pandas-grouping-and-aggregating-consecutive-rows-with-same-value-in-column
            new_spell = (self.data['fid'] != self.data['fid_l1']) | (self.data['wid'] != self.data['wid_l1']) # Allow for wid != wid_l1 to ensure that consecutive workers at the same firm get counted as different spells
            self.data['spell_id'] = new_spell.cumsum()
            logger.info('spell ids generated')

            # Aggregate at the spell level
            spell = self.data.groupby(['spell_id'])
            data_spell = spell.agg(
                fid=pd.NamedAgg(column='fid', aggfunc='first'),
                wid=pd.NamedAgg(column='wid', aggfunc='first'),
                comp=pd.NamedAgg(column='comp', aggfunc='mean'),
                year_start=pd.NamedAgg(column='year', aggfunc='min'),
                year_end=pd.NamedAgg(column='fid', aggfunc='max')
            )
            logger.info('data aggregated at the spell level')

            ## Format as event study ##
            # Split workers by spell count
            spell_count = data_spell.groupby(['wid']).size()
            single_spell = spell_count[spell_count == 1].index
            stayers = data_spell[data_spell['wid'].isin(single_spell)]
            mult_spell = spell_count[spell_count > 1].index
            movers = data_spell[data_spell['wid'].isin(mult_spell)]
            logger.info('workers split by spell count')

            # Add lagged values
            movers = movers.sort_values(['wid', 'year_start'])
            movers['fid_l1'] = movers['fid'].shift(periods=1)
            movers['wid_l1'] = movers['wid'].shift(periods=1)
            movers['comp_l1'] = movers['comp'].shift(periods=1)
            movers = movers[movers['wid'] == movers['wid_l1']]

            # Update columns
            stayers = stayers.rename({'fid': 'f1i', 'comp': 'y1'}, axis=1)
            stayers['f2i'] = stayers['f1i']
            stayers['y2'] = stayers['y1']
            stayers['m'] = 0
            movers = movers.rename({'fid_l1': 'f1i', 'fid': 'f2i', 'comp_l1': 'y1', 'comp': 'y2'}, axis=1)
            movers['f1i'] = movers['f1i'].astype(int)
            movers['m'] = 1

            # Keep only relevant columns
            stayers = stayers[['wid', 'y1', 'y2', 'f1i', 'f2i', 'm']]
            movers = movers[['wid', 'y1', 'y2', 'f1i', 'f2i', 'm']]
            logger.info('columns updated')

            # Merge stayers and movers
            self.data = pd.concat([stayers, movers])

            # Update col_dict
            self.col_dict = {'wid': 'wid', 'y1': 'y1', 'y2': 'y2', 'f1i': 'f1i', 'f2i': 'f2i', 'm': 'm'}

            logger.info('data reformatted as event study')

            # Data is now formatted as event study
            self.formatting = 'es'

    def approx_cdfs(self, cdf_resolution=10, grouping='quantile_all', year=None):
        '''
        Purpose:
            Generate cdfs of compensation for firms

        Inputs:
            cdf_resolution (int): how many values to use to approximate the cdf
            grouping (string): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)
            year (int): if None, uses entire dataset; if int, gives year of data to consider

        Returns:
            cdf_df (numpy array): numpy array of firm cdfs
        '''
        # If year-level, then only use data for that particular year
        if isinstance(year, int) and (self.formatting == 'long'):
            data = data[data['year'] == year]

        # Create empty numpy array to fill with the cdfs
        n_firms = self.n_firms()
        cdfs = np.zeros([n_firms, cdf_resolution])

        # Create quantiles of interest
        quantiles = np.linspace(1 / cdf_resolution, 1, cdf_resolution)

        # Re-arrange event study data to be in long format (temporarily)
        if self.formatting == 'es':
            self.data = self.data.rename({'f1i': 'fid', 'y1': 'comp'}, axis=1)
            self.data = pd.concat([self.data, self.data[['f2i', 'y2']].rename({'f2i': 'fid', 'y2': 'comp'}, axis=1)], axis=0)

        if grouping == 'quantile_all':
            # Get quantiles from all data
            quantile_groups = self.data['comp'].quantile(quantiles)

            # Generate firm-level cdfs
            for i, quant in enumerate(quantile_groups):
                cdfs[:, i] = self.data.assign(firm_quant=lambda d: d['comp'] <= quant).groupby('fid')['firm_quant'].agg(sum).to_numpy()

            # Normalize by firm size (convert to cdf)
            fsize = self.data.groupby('fid').size().to_numpy()
            cdfs /= np.expand_dims(fsize, 1)

        elif grouping in ['quantile_firm_small', 'quantile_firm_large']:
            # Sort data by compensation (do this once now, so that don't need to do it again later) (also note it is faster to sort then manually compute quantiles than to use built-in quantile functions)
            self.data = self.data.sort_values(['comp'])

            if grouping == 'quantile_firm_small':
                # Convert pandas dataframe into a dictionary to access data faster
                # Source for idea: https://stackoverflow.com/questions/57208997/looking-for-the-fastest-way-to-slice-a-row-in-a-huge-pandas-dataframe
                # Source for how to actually format data correctly: https://stackoverflow.com/questions/56064677/pandas-series-to-dict-with-repeated-indices-make-dict-with-list-values
                data_dict = self.data['comp'].groupby(level=0).agg(list).to_dict()

            # Generate the cdfs
            for row in tqdm(range(n_firms)):
                fid = row + 1 # fids start at 1
                # Get the firm-level compensation data (don't need to sort because already sorted)
                if grouping == 'quantile_firm_small':
                    comp = data_dict[fid]
                elif grouping == 'quantile_firm_large':
                    comp = self.data.loc[self.data['fid'] == fid, 'comp']
                # Generate the firm-level cdf
                # Note: update numpy array element by element
                # Source: https://stackoverflow.com/questions/30012362/faster-way-to-convert-list-of-objects-to-numpy-array/30012403
                for i in range(cdf_resolution):
                    index = max(len(comp) * (i + 1) // cdf_resolution - 1, 0) # Don't want negative index
                    # Update cdfs with the firm-level cdf
                    cdfs[row, i] = comp[index]

        # Drop rows that were appended earlier and rename columns
        if self.formatting == 'es':
            self.data = self.data.dropna()
            self.data = self.data.rename({'fid': 'f1i', 'comp': 'y1'}, axis=1)

        return cdfs

    def cluster(self, n_clusters=5, cdf_resolution=10, grouping='quantile_all', year=None):
        '''
        Purpose:
            Cluster data and assign a new column giving the cluster for each firm

        Inputs:
            n_clusters (int): how many clusters to consider
            cdf_resolution (int): how many values to use to approximate the cdf
            grouping (string): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)
            year (int): if None, uses entire dataset; if int, gives year of data to consider

        Returns:
            Nothing
        '''
        if self.formatting == 'es':
            # Compute cdfs
            cdfs = self.approx_cdfs(cdf_resolution=cdf_resolution, grouping=grouping, year=year)
            logger.info('firm cdfs computed')

            # Compute firm clusters
            clusters = KMeans(n_clusters=n_clusters, random_state=0).fit(cdfs).labels_ + 1 # Need +1 because need > 0
            logger.info('firm clusters computed')

            # Create Pandas dataframe linking fid to firm cluster
            n_firms = cdfs.shape[0]
            fids = np.linspace(0, n_firms - 1, n_firms) + 1
            clusters_dict_1 = {'f1i': fids, 'j1': clusters}
            clusters_dict_2 = {'f2i': fids, 'j2': clusters}
            clusters_df_1 = pd.DataFrame(clusters_dict_1, index=fids)
            clusters_df_2 = pd.DataFrame(clusters_dict_2, index=fids)
            logger.info('dataframes linked fids to clusters generated')

            # Merge into event study data
            self.data = self.data.merge(clusters_df_1, how='left', on='f1i')
            self.data = self.data.merge(clusters_df_2, how='left', on='f2i')
            logger.info('clusters merged into event study data')

            # Correct datatypes
            self.data[['f1i', 'f2i', 'm']] = self.data[['f1i', 'f2i', 'm']].astype(int)
            logger.info('datatypes of clusters corrected')

    def run_akm_corrected(self, params={}):
        '''
        Purpose:
            Run bias-corrected AKM estimator

        Inputs:
            params (dictionary): dictionary of parameters
                ncore (int): number of cores to use
                batch (int): batch size to send in parallel
                ndraw_pii (int): number of draw to use in approximation for leverages
                ndraw_tr (int): number of draws to use in approximation for traces
                check (bool): whether to compute the non-approximated estimates as well
                hetero (bool): whether to compute the heteroskedastic estimates
                out (string): outputfile
                con (string): computes the smallest eigen values, this is the filepath where these results are saved
                logfile (string): log output to a logfile
                levfile (string): file to load precomputed leverages
                statsonly (bool): save data statistics only

        Returns:
            akm_res (dictionary): dictionary of results
        '''
        default_params = {'ncore': 1, 'batch': 1, 'ndraw_pii': 50, 'ndraw_tr': 5, 'check': False, 'hetero': False, 'out': 'res_akm.json', 'con': False, 'logfile': '', 'levfile': '', 'statsonly': False, 'data': self.data}

        for key, val in params.items():
            default_params[key] = val
        
        akm_res = feacf.main(default_params)

        return akm_res

    def run_cre(self, params={}):
        '''
        Purpose:
            Run CRE estimator

        Inputs:
            params (dictionary): dictionary of parameters
                ncore (int): number of cores to use
                ndraw_tr (int): number of draws to use in approximation for traces
                ndp (int): number of draw to use in approximation for leverages
                out (string): outputfile
                posterior (bool): compute posterior variance
                wobtw (bool): sets between variation to 0, pure RE

        Returns:
            cre_res (dictionary): dictionary of results
        '''
        default_params = {'ncore': 1, 'ndraw_tr': 5, 'ndp': 50, 'out': 'res_cre.json', 'posterior': False, 'wobtw': False, 'data': self.data}

        for key, val in params.items():
            default_params[key] = val
        
        cre_res = cre.main(default_params)

        return cre_res

    def sim_network_gen_fe(self, params):
        '''
        Purpose:
            Generate fixed effects values for simulated panel data corresponding to the calibrated model

        Inputs:
            params (dictionary): dictionary linking parameters to values
                Dictionary parameters:
                    num_ind: number of workers
                    num_time: time length of panel
                    firm_size: max number of individuals per firm
                    nk: number of firm types
                    nl: number of worker types
                    alpha_sig: standard error of individual fixed effect (volatility of worker effects)
                    psi_sig: standard error of firm fixed effect (volatility of firm effects)
                    w_sig: standard error of residual in AKM wage equation (volatility of wage shocks)
                    csort: sorting effect
                    cnetw: network effect
                    csig: standard error of sorting/network effects
                    p_move: probability a worker moves firms in any period

        Returns:
            psi (NumPy array): array of firm fixed effects
            alpha (NumPy array): array of individual fixed effects
            G (NumPy array): transition matrices
            H (NumPy array): stationary distribution
        '''
        # Extract parameters
        nk, nl, alpha_sig, psi_sig = params['nk'], params['nl'], params['alpha_sig'], params['psi_sig']
        csort, cnetw, csig = params['csort'], params['cnetw'], params['csig']

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
            Draw firm ids for individual, given data that is grouped by worker id, spell id, and firm type

        Inputs:
            freq (NumPy array): size of groups (groups by worker id, spell id, and firm type)
            num_time (int): time length of panel
            firm_size (int): max number of individuals per firm

        Returns:
            (NumPy array): random firms for each group
        '''
        max_int = np.int(np.maximum(1, freq.sum() / (firm_size * num_time)))
        return np.array(np.random.choice(max_int, size=freq.count()) + 1)

    def sim_network(self, params):
        '''
        Purpose:
            Simulate panel data corresponding to the calibrated model

        Inputs:
            params (dictionary): dictionary linking parameters to values
                Dictionary parameters:
                    num_ind: number of workers
                    num_time: time length of panel
                    firm_size: max number of individuals per firm
                    nk: number of firm types
                    nl: number of worker types
                    alpha_sig: standard error of individual fixed effect (volatility of worker effects)
                    psi_sig: standard error of firm fixed effect (volatility of firm effects)
                    w_sig: standard error of residual in AKM wage equation (volatility of wage shocks)
                    csort: sorting effect
                    cnetw: network effect
                    csig: standard error of sorting/network effects
                    p_move: probability a worker moves firms in any period

        Returns:
            data (Pandas DataFrame): simulated network
        '''
        # Generate fixed effects
        psi, alpha, G, H = self.sim_network_gen_fe(params)

        # Extract parameters
        num_ind, num_time, firm_size = params['num_ind'], params['num_time'], params['firm_size']
        nk, nl, w_sig, p_move = params['nk'], params['nl'], params['w_sig'], params['p_move']

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

# Begin logging
logger = logging.getLogger('twfe')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('twfe_spam.log')
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
