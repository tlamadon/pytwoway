'''
Class for a two-way fixed effect network
'''

import logging
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from random import choices
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pytwoway import fe_approximate_correction_full as feacf
from pytwoway import cre

class twfe_network:
    '''
    Class of twfe_network, where twfe_network gives a network of firms and workers.

    Arguments:
        data (Pandas DataFrame): data giving firms, workers, and compensation
        formatting (str): if 'long', then data in long format; if 'es', then data in event study format. If simulating data, keep default value of 'long'
        col_dict (dict): make data columns readable (requires: wid (worker id), comp (compensation), fid (firm id), year if long; wid (worker id), y1 (compensation 1), y2 (compensation 2), f1i (firm id 1), f2i (firm id 2), m (0 if stayer, 1 if mover) if event study). Keep None if column names already correct
    '''

    def __init__(self, data, formatting='long', col_dict=None):
        logger.info('initializing twfe_network object')

        # Define some attributes
        self.data = data
        self.formatting = formatting
        self.connected = False # If True, all firms are connected by movers
        self.contiguous = False # If True, firm ids are contiguous
        self.no_na = False # If True, no NaN observations in the data

        if col_dict is None: # If columns already correct
            self.col_dict = {'fid': 'fid', 'wid': 'wid', 'year': 'year', 'comp': 'comp'}
        else:
            self.col_dict = col_dict

        # Define default parameter dictionaries
        self.default_KMeans = {'n_clusters': 10, 'init': 'k-means++', 'n_init': 500, 'max_iter': 300, 'tol': 0.0001, 'precompute_distances': 'deprecated', 'verbose': 0, 'random_state': None, 'copy_x': True, 'n_jobs': 'deprecated', 'algorithm': 'auto'}

        self.default_cluster = {'cdf_resolution': 10, 'grouping': 'quantile_all', 'year': None, 'user_KMeans': self.default_KMeans}

        self.default_akm = {'ncore': 1, 'batch': 1, 'ndraw_pii': 50, 'ndraw_tr': 5, 'check': False, 'hetero': False, 'out': 'res_akm.json', 'con': False, 'logfile': '', 'levfile': '', 'statsonly': False, 'Q': 'cov(alpha, psi)'} # Do not define 'data' because will be updated later

        self.default_cre = {'ncore': 1, 'ndraw_tr': 5, 'ndp': 50, 'out': 'res_cre.json', 'posterior': False, 'wobtw': False} # Do not define 'data' because will be updated later

        logger.info('twfe_network object initialized')

    def update_dict(self, default_params, user_params):
        '''
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

    def update_cols(self):
        '''
        Rename columns and keep only relevant columns.
        '''
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
        Get the number of unique workers.

        Returns:
            (int): number of unique workers
        '''
        return len(self.data['wid'].unique())

    def n_firms(self):
        '''
        Get the number of unique firms.

        Returns:
            (int): number of unique firms
        '''
        if self.formatting == 'long':
            return len(self.data['fid'].unique())
        elif self.formatting == 'es':
            return len(set(list(self.data['f1i'].unique()) + list(self.data['f2i'].unique())))

    def clean_data(self):
        '''
        Clean data to make sure there are no NaN observations, firms are connected by movers and firm ids are contiguous.
        '''
        logger.info('beginning data cleaning')
        logger.info('checking quality of data')
        # Make sure data is valid - computes no_na, connected, and contiguous, along with other checks (note that column names are corrected in data_validity() if all columns are in the data)
        self.data_validity()

        # Next, drop NaN observations
        if not self.no_na:
            logger.info('dropping NaN observations')
            self.data = self.data.dropna()

            # Update no_na
            self.no_na = True

        # Next, find largest set of firms connected by movers
        if not self.connected:
            # Generate largest connected set
            logger.info('generating largest connected set')
            self.conset()

        # Next, make firm ids contiguous
        if not self.contiguous:
            # Make firm ids contiguous
            logger.info('making firm ids contiguous')
            self.contiguous_fids()

        # Using contiguous fids, get NetworkX Graph of largest connected set (note that this must be done even if firms already connected and contiguous)
        logger.info('generating NetworkX Graph of largest connected set')
        self.G = self.conset()

        logger.info('data cleaning complete')

    def data_validity(self):
        '''
        Checks that data is formatted correctly and updates relevant attributes.
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
                        if self.data[self.col_dict[col]].dtype not in ['float', 'float16', 'float32', 'float64', 'float128', 'int', 'int16', 'int32', 'int64']:
                            logger.info(self.col_dict[col], 'has wrong dtype, should be float or int but is', self.data[self.col_dict[col]].dtype)
                            cols = False

            logger.info('columns correct:' + str(cols))
            if not cols:
                success = False
                raise ValueError('Your data does not include the correct columns. The twfe_network object cannot be generated with your data.')
            else:
                # Correct column names
                logger.info('correcting column names')
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
            else:
                self.no_na = True

            logger.info('--- checking connected set ---')
            self.data['fid_max'] = self.data.groupby(['wid'])['fid'].transform(max)
            G = nx.from_pandas_edgelist(self.data, 'fid', 'fid_max')
            largest_cc = max(nx.connected_components(G), key=len)
            self.data = self.data.drop(['fid_max'], axis=1)

            outside_cc = self.data[(~self.data['fid'].isin(largest_cc))].shape[0]

            logger.info('observations outside connected set (should be 0):' + str(outside_cc))
            if outside_cc > 0:
                success = False
            else:
                self.connected = True

            logger.info('--- checking contiguous firm ids ---')
            fid_max = self.data['fid'].max()
            n_firms = self.n_firms()

            contig = (fid_max == n_firms)

            logger.info('contiguous firm ids (should be True):' + str(contig))
            if not contig:
                success = False
            else:
                self.contiguous = True

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
                            if self.data[self.col_dict[col]].dtype not in ['float', 'float16', 'float32', 'float64', 'float128', 'int', 'int16', 'int32', 'int64']:
                                logger.info(col, 'has wrong dtype, should be float or int but is', self.data[self.col_dict[col]].dtype)
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
                    logger.info('correcting column names')
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
                if (na_stayers == 0) and (na_movers == 0):
                    self.no_na = True

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
                if (cc_stayers == 0) and (cc_movers == 0):
                    self.connected = True

                logger.info('--- checking contiguous firm ids ---')
                fid_max = max(stayers['f1i'].max(), max(movers['f1i'].max(), movers['f2i'].max()))
                n_firms = self.n_firms()

                contig = (fid_max == n_firms)
                if not contig:
                    success = False
                else:
                    self.contiguous = True

                logger.info('Overall success for stayers:' + str(success_stayers))
                logger.info('Overall success for movers:' + str(success_movers))

    def conset(self):
        '''
        Update data to include only the largest connected set of movers, and if firm ids are contiguous, also return the NetworkX Graph.

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
        Make firm ids contiguous.
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
        Refactor long form data into event study data.
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
        Generate cdfs of compensation for firms.

        Arguments:
            cdf_resolution (int): how many values to use to approximate the cdf
            grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)
            year (int or None): if None, uses entire dataset; if int, gives year of data to consider

        Returns:
            cdf_df (NumPy Array): NumPy array of firm cdfs
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

    def cluster(self, user_cluster={}):
        '''
        Cluster data and assign a new column giving the cluster for each firm.

        Arguments:
            user_cluster (dict): dictionary of parameters for clustering

                Dictionary parameters:

                    cdf_resolution (int): how many values to use to approximate the cdf

                    grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)

                    year (int or None): if None, uses entire dataset; if int, gives year of data to consider

                    user_KMeans (dict): use parameters defined in KMeans_dict for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and use default parameters defined in class attribute default_KMeans for any parameters not specified
        '''
        if self.formatting == 'es':
            # Update dictionary
            cluster_params = self.update_dict(self.default_cluster, user_cluster)

            # Unpack dictionary
            cdf_resolution = cluster_params['cdf_resolution']
            grouping = cluster_params['grouping']
            year = cluster_params['year']
            user_KMeans = cluster_params['user_KMeans']

            # Compute cdfs
            cdfs = self.approx_cdfs(cdf_resolution=cdf_resolution, grouping=grouping, year=year)
            logger.info('firm cdfs computed')

            # Compute firm clusters
            KMeans_params = self.update_dict(self.default_KMeans, user_KMeans)
            clusters = KMeans(**KMeans_params).fit(cdfs).labels_ + 1 # Need +1 because need > 0
            logger.info('firm clusters computed')

            # Create Pandas dataframe linking fid to firm cluster
            n_firms = cdfs.shape[0]
            fids = np.linspace(0, n_firms - 1, n_firms) + 1
            clusters_dict_1 = {'f1i': fids, 'j1': clusters}
            clusters_dict_2 = {'f2i': fids, 'j2': clusters}
            clusters_df_1 = pd.DataFrame(clusters_dict_1, index=fids)
            clusters_df_2 = pd.DataFrame(clusters_dict_2, index=fids)
            logger.info('dataframes linking fids to clusters generated')

            # Merge into event study data
            self.data = self.data.merge(clusters_df_1, how='left', on='f1i')
            self.data = self.data.merge(clusters_df_2, how='left', on='f2i')
            logger.info('clusters merged into event study data')

            # Correct datatypes
            self.data[['f1i', 'f2i', 'm']] = self.data[['f1i', 'f2i', 'm']].astype(int)
            logger.info('datatypes of clusters corrected')

    def run_akm_corrected(self, user_akm={}):
        '''
        Run bias-corrected AKM estimator.

        Arguments:
            user_akm (dict): dictionary of parameters for bias-corrected AKM estimation

                Dictionary parameters:

                    ncore (int): number of cores to use

                    batch (int): batch size to send in parallel

                    ndraw_pii (int): number of draw to use in approximation for leverages

                    ndraw_tr (int): number of draws to use in approximation for traces

                    check (bool): whether to compute the non-approximated estimates as well

                    hetero (bool): whether to compute the heteroskedastic estimates

                    out (str): outputfile

                    con (str): computes the smallest eigen values, this is the filepath where these results are saved

                    logfile (str): log output to a logfile

                    levfile (str): file to load precomputed leverages

                    statsonly (bool): save data statistics only

                    Q (str): which Q matrix to consider. Options include 'cov(alpha, psi)' and 'cov(psi_t, psi_{t+1})'

        Returns:
            akm_res (dict): dictionary of results
        '''
        akm_params = self.update_dict(self.default_akm, user_akm)

        akm_params['data'] = self.data # Make sure to use up-to-date data

        akm_solver = feacf.FEsolver(akm_params)
        akm_solver.run_1()
        akm_solver.construct_Q() # Comment out this line and manually create Q if you want a custom Q matrix
        akm_solver.run_2()

        akm_res = akm_solver.res

        return akm_res

    def run_cre(self, user_cre={}):
        '''
        Run CRE estimator.

        Arguments:
            user_cre (dict): dictionary of parameters for CRE estimation

                Dictionary parameters:

                    ncore (int): number of cores to use

                    ndraw_tr (int): number of draws to use in approximation for traces

                    ndp (int): number of draw to use in approximation for leverages

                    out (str): outputfile

                    posterior (bool): compute posterior variance

                    wobtw (bool): sets between variation to 0, pure RE

        Returns:
            cre_res (dict): dictionary of results
        '''
        cre_params = self.update_dict(self.default_cre, user_cre)

        cre_params['data'] = self.data # Make sure to use up-to-date data

        cre_res = cre.main(cre_params)

        return cre_res

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
