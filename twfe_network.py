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
import fe_approximate_correction_full as feacf
import cre

# # Testing
# data = pd.read_feather('../../Google Drive File Stream/.shortcut-targets-by-id/1iN9LApqNxHmVCOV4IUISMwPS7KeZcRhz/ra-adam/data/English/worker_cleaned.ftr')
# col_dict = {'fid': 'codf', 'wid': 'codf_w', 'year': 'year', 'comp': 'comp_current'}
# # d_net for data_network
# d_net = twfe_network(data=data, formatting='long', col_dict=col_dict)
# d_net.refactor_es()
# d_net.data_validity()
# d_net.cluster()
# akm_res = d_net.run_akm_corrected()
# cre_res = d_net.run_cre()

# # Simulate data
# sim_params = 
# d_net = twfe_network()
# cdfs_1 = d_net.approx_cdfs()
# d_net.refactor_es()
# cdfs_2 = d_net.approx_cdfs()

class twfe_network:
    '''
    Class of twfe_network, where twfe_network gives a network of firms and workers. This class has the following functions:<br/>
        __init__(): initialize<br/>
        update_dict(): update values in parameter dictionaries (this function is similar to, but different from dict.update())
        update_cols(): rename columns and keep only relevant columns<br/>
        n_workers(): get the number of unique workers<br/>
        n_firms(): get the number of unique firms<br/>
        data_validity(): check that data is formatted correctly<br/>
        conset(): update data to include only the largest connected set of movers, and if firm ids are contiguous, also return the NetworkX Graph<br/>
        contiguous_fids(): make firm ids contiguous<br/>
        refactor_es(): refactor long form data into event study data<br/>
        approx_cdfs(): generate cdfs of compensation for firms<br/>
        cluster(): cluster data and assign a new column giving the cluster for each firm<br/>
        run_akm_corrected(): run bias-corrected AKM estimator<br/>
        run_cre(): run CRE estimator
    '''

    def __init__(self, data, formatting='long', col_dict=False):
        '''
        Purpose:
            Initialize twfe_network object.

        Arguments:
            data (Pandas DataFrame): data giving firms, workers, and compensation
            formatting (str): if 'long', then data in long format; if 'es', then data in event study format. If simulating data, keep default value of 'long'
            col_dict (dict): make data columns readable (requires: wid (worker id), comp (compensation), fid (firm id), year if long; wid (worker id), y1 (compensation 1), y2 (compensation 2), f1i (firm id 1), f2i (firm id 2), m (0 if stayer, 1 if mover) if event study). Keep False if column names already correct
        '''
        logger.info('initializing twfe_network object')

        # Define some attributes
        self.connected = False
        self.contiguous = False
        self.formatting = formatting

        if isinstance(col_dict, bool): # If columns already correct
            self.col_dict = {'fid': 'fid', 'wid': 'wid', 'year': 'year', 'comp': 'comp'}
        else:
            self.col_dict = col_dict
        
        # Define default parameter dictionaries
        self.default_KMeans = {'n_clusters': 10, 'init': 'k-means++', 'n_init': 500, 'max_iter': 300, 'tol': 0.0001, 'precompute_distances': 'deprecated', 'verbose': 0, 'random_state': None, 'copy_x': True, 'n_jobs': 'deprecated', 'algorithm': 'auto'}

        self.default_cluster = {'cdf_resolution': 10, 'grouping': 'quantile_all', 'year': None, 'user_KMeans': self.default_KMeans}

        self.default_akm = {'ncore': 1, 'batch': 1, 'ndraw_pii': 50, 'ndraw_tr': 5, 'check': False, 'hetero': False, 'out': 'res_akm.json', 'con': False, 'logfile': '', 'levfile': '', 'statsonly': False} # Do not define 'data' because will be updated later

        self.default_cre = {'ncore': 1, 'ndraw_tr': 5, 'ndp': 50, 'out': 'res_cre.json', 'posterior': False, 'wobtw': False} # Do not define 'data' because will be updated later

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

        logger.info('twfe_network object initialized')

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

    def update_cols(self):
        '''
        Purpose:
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
        Purpose:
            Get the number of unique workers.

        Returns:
            (int): number of unique workers
        '''
        return len(self.data['wid'].unique())

    def n_firms(self):
        '''
        Purpose:
            Get the number of unique firms.

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
            Check that data is formatted correctly. Results are logged.
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
        Purpose:
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
        Purpose:
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
        Purpose:
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
        Purpose:
            Cluster data and assign a new column giving the cluster for each firm.

        Arguments:
            user_cluster (dict): dictionary of parameters for clustering<br/>
                <ul>
                <li>Dictionary parameters:</li>
                    <ul><ul>
                    <li>cdf_resolution (int): how many values to use to approximate the cdf</li>
                    <li>grouping (string): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)</li>
                    <li>year (int or None): if None, uses entire dataset; if int, gives year of data to consider</li>
                    <li>user_KMeans (dict): use parameters defined in KMeans_dict for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and use default parameters defined in class attribute default_KMeans for any parameters not specified</li>
                    </ul></ul>
                </ul>
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
            clusters = KMeans(n_clusters=KMeans_params['n_clusters'], init=KMeans_params['init'], n_init=KMeans_params['n_init'], max_iter=KMeans_params['max_iter'], tol=KMeans_params['tol'], precompute_distances=KMeans_params['precompute_distances'], verbose=KMeans_params['verbose'], random_state=KMeans_params['random_state'], copy_x=KMeans_params['copy_x'], n_jobs=KMeans_params['n_jobs'], algorithm=KMeans_params['algorithm']).fit(cdfs).labels_ + 1 # Need +1 because need > 0
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

    def run_akm_corrected(self, user_akm={}):
        '''
        Purpose:
            Run bias-corrected AKM estimator.

        Arguments:
            user_akm (dict): dictionary of parameters for bias-corrected AKM estimation<br/>
                <ul>
                <li>Dictionary parameters:</li>
                    <ul><ul>
                    <li>ncore (int): number of cores to use</li>
                    <li>batch (int): batch size to send in parallel</li>
                    <li>ndraw_pii (int): number of draw to use in approximation for leverages</li>
                    <li>ndraw_tr (int): number of draws to use in approximation for traces</li>
                    <li>check (bool): whether to compute the non-approximated estimates as well</li>
                    <li>hetero (bool): whether to compute the heteroskedastic estimates</li>
                    <li>out (str): outputfile</li>
                    <li>con (str): computes the smallest eigen values, this is the filepath where these results are saved</li>
                    <li>logfile (str): log output to a logfile</li>
                    <li>levfile (str): file to load precomputed leverages</li>
                    <li>statsonly (bool): save data statistics only</li>
                    </ul></ul>
                </ul>

        Returns:
            akm_res (dict): dictionary of results
        '''
        akm_params = self.update_dict(self.default_akm, user_akm)

        akm_params['data'] = self.data # Make sure to use up-to-date data

        akm_res = feacf.FEsolver(akm_params).res # feacf.main(akm_params) # FIXME corrected for new feacf file using class structure

        return akm_res

    def run_cre(self, user_cre={}):
        '''
        Purpose:
            Run CRE estimator.

        Arguments:
            user_cre (dict): dictionary of parameters for CRE estimation<br/>
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
