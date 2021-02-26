'''
Class for a bipartite network
'''
import logging
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import connected_components
import warnings
from pytwoway import update_dict

class BipartiteData:
    '''
    Class of BipartiteData, where BipartiteData gives a bipartite network of firms and workers. Subclasses include BipartiteLong which gives a bipartite network of firms and workers in long form, BipartiteLongCollapsed which gives a bipartite network of firms and workers in collapsed long form (i.e. employment spells are collapsed into a single observation), and BipartiteEventStudy which gives a bipartite network of firms and workers in event study form.

    Arguments:
        data (Pandas DataFrame): bipartite network data
        formatting (str): if 'long', then data in long format; if 'long_collapsed', then data in collapsed long format; if 'es', then data in event study format.
        col_dict (dict): make data columns readable (requires:
            if long: wid (worker id), comp (compensation), fid (firm id), year;
            if collapsed long: wid (worker id), comp (compensation), fid (firm id), year_start (first year of employment spell), year_end (last year of employment spell);
            if event study: wid (worker id), y1 (compensation 1), y2 (compensation 2), f1i (firm id 1), f2i (firm id 2), m (0 if stayer, 1 if mover);
                optionally include: year_end_1 (last year of observation 1), year_end_2 (last year of observation 2), w1 (weight 1), w2 (weight 2)).
            Keep None if column names already correct
    '''

    def __init__(self, data, formatting='long', col_dict=None):
        # Begin logging
        self.logger = logging.getLogger('bipartite')
        self.logger.setLevel(logging.DEBUG)
        # Create logs folder
        Path('twoway_logs').mkdir(parents=True, exist_ok=True)
        # Create file handler which logs even debug messages
        fh = logging.FileHandler('twoway_logs/bipartite_spam.log')
        fh.setLevel(logging.DEBUG)
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info('initializing BipartiteData object')

        # Define some attributes
        self.data = data
        self.formatting = formatting
        self.col_dict = col_dict

        # Create subclass
        if self.formatting == 'long':
            self.bd = BipartiteLong(self.data, self.col_dict)
        if self.formatting == 'long_collapsed':
            self.bd = BipartiteLongCollapsed(self.data, self.col_dict)
        elif self.formatting == 'es':
            self.bd = BipartiteEventStudy(self.data, self.col_dict)

        self.logger.info('BipartiteData object initialized')

    def copy(self):
        '''
        Copy the current instance of BipartiteData.

        Returns:
            bd_copy (BipartiteData): copy of the current instance of BipartiteData.
        '''
        bd_copy = BipartiteData(self.data.copy(), self.formatting, self.col_dict.copy())
        bd_copy.bd = self.bd.copy()

        return bd_copy

    def clean_data(self):
        '''
        Clean data.
        '''
        self.bd.clean_data()
        self.data = self.bd.data
        self.col_dict = self.bd.col_dict

    def long_to_collapsed_long(self):
        '''
        Refactor long data into collapsed long data.
        '''
        if self.formatting == 'long':
            # Save attributes
            connected = self.bd.connected
            contiguous = self.bd.contiguous
            no_na = self.bd.no_na
            no_duplicates = self.bd.no_duplicates
            # Update bipartite network
            self.bd = BipartiteLongCollapsed(self.bd.get_collapsed_long())
            # Use attributes from before
            self.bd.connected = connected
            self.bd.contiguous = contiguous
            self.bd.no_na = no_na
            self.bd.no_duplicates = no_duplicates
            # Update superclass attributes
            self.data = self.bd.data
            self.col_dict = self.bd.col_dict
            self.formatting = 'long_collapsed'
        elif self.formatting == 'es':
            warnings.warn('Data is formatted as event study. Please run es_to_collapsed_long() to convert your data into collapsed long form.')
        elif self.formatting == 'long_collapsed':
            warnings.warn('Data is already formatted as collapsed long.')
        else:
            warnings.warn('Data cannot be refactored into event study as its current formatting, {}, is invalid.'.format(self.formatting))

    def long_to_es(self):
        '''
        Refactor long data into event study data.
        '''
        if self.formatting == 'long':
            # Save attributes
            connected = self.bd.connected
            contiguous = self.bd.contiguous
            no_na = self.bd.no_na
            no_duplicates = self.bd.no_duplicates
            # Update bipartite network
            self.bd = BipartiteEventStudy(self.bd.get_es(), collapsed=False)
            # Use attributes from before
            self.bd.connected = connected
            self.bd.contiguous = contiguous
            self.bd.no_na = no_na
            self.bd.no_duplicates = no_duplicates
            # Update superclass attributes
            self.data = self.bd.data
            self.col_dict = self.bd.col_dict
            self.formatting = 'es'
        elif self.formatting == 'long_collapsed':
            warnings.warn('Data is formatted as collapsed long. Please run collapsed_long_to_es() to convert your data into event study form.')
        elif self.formatting == 'es':
            warnings.warn('Data is already formatted as event study.')
        else:
            warnings.warn('Data cannot be refactored into event study as its current formatting, {}, is invalid.'.format(self.formatting))

    def collapsed_long_to_es(self):
        '''
        Refactor collapsed long data into event study data.
        '''
        if self.formatting == 'long_collapsed':
            # Save attributes
            connected = self.bd.connected
            contiguous = self.bd.contiguous
            no_na = self.bd.no_na
            no_duplicates = self.bd.no_duplicates
            # Update bipartite network
            self.bd = BipartiteEventStudy(self.bd.get_es(), collapsed=True)
            # Use attributes from before
            self.bd.connected = connected
            self.bd.contiguous = contiguous
            self.bd.no_na = no_na
            self.bd.no_duplicates = no_duplicates
            # Update superclass attributes
            self.data = self.bd.data
            self.col_dict = self.bd.col_dict
            self.formatting = 'es'
        elif self.formatting == 'long':
            warnings.warn('Data is formatted as long. Please run long_to_collapsed_long() to convert your data into collapsed long form.')
        elif self.formatting == 'es':
            warnings.warn('Data is already formatted as event study.')
        else:
            warnings.warn('Data cannot be refactored into event study as its current formatting, {}, is invalid.'.format(self.formatting))

    def es_to_collapsed_long(self):
        '''
        Refactor event study data into collapsed long data.
        '''
        if self.formatting == 'es':
            if self.bd.collapsed:
                # Save attributes
                connected = self.bd.connected
                contiguous = self.bd.contiguous
                no_na = self.bd.no_na
                no_duplicates = self.bd.no_duplicates
                # Update bipartite network
                self.bd = BipartiteLongCollapsed(self.bd.get_collapsed_long())
                # Use attributes from before
                self.bd.connected = connected
                self.bd.contiguous = contiguous
                self.bd.no_na = no_na
                self.bd.no_duplicates = no_duplicates
                # Update superclass attributes
                self.data = self.bd.data
                self.col_dict = self.bd.col_dict
                self.formatting = 'long_collapsed'
            else:
                warnings.warn('Event study data is not collapsed, cannot refactor into collapsed long form. Try running es_to_long(), then long_to_collapsed_long() to convert your data into collapsed long form.')
        elif self.formatting == 'long':
            warnings.warn('Data is formatted as long. Try running long_to_collapsed_long() to convert your data into collapsed long form.')
        elif self.formatting == 'long_collapsed':
            warnings.warn('Data is already formatted as collapsed long.')
        else:
            warnings.warn('Data cannot be refactored into collapsed long as its current formatting, {}, is invalid.'.format(self.formatting))

    def es_to_long(self):
        '''
        Refactor event study data into long data.
        '''
        if self.formatting == 'es':
            if not self.bd.collapsed:
                # Save attributes
                connected = self.bd.connected
                contiguous = self.bd.contiguous
                no_na = self.bd.no_na
                no_duplicates = self.bd.no_duplicates
                # Update bipartite network
                self.bd = BipartiteLong(self.bd.get_long())
                # Use attributes from before
                self.bd.connected = connected
                self.bd.contiguous = contiguous
                self.bd.no_na = no_na
                self.bd.no_duplicates = no_duplicates
                # Update superclass attributes
                self.data = self.bd.data
                self.col_dict = self.bd.col_dict
                self.formatting = 'long'
            else:
                warnings.warn('Data is formatted as collapsed long, cannot un-collapse data. Try running es_to_collapsed_long() to convert your data into collapsed long form.')
        elif self.formatting == 'long_collapsed':
            warnings.warn('Data is formatted as collapsed long, cannot un-collapse data.')
        elif self.formatting == 'long':
            warnings.warn('Data is already formatted as long.')
        else:
            warnings.warn('Data cannot be refactored into collapsed long as its current formatting, {}, is invalid.'.format(self.formatting))

    def get_cs(self):
        '''
        Return event study data reformatted into cross section data.
        '''
        if self.formatting == 'es':
            return self.bd.get_cs()
        elif self.formatting == 'long':
            warnings.warn('Cross section cannot be constructed from long data. Run long_to_es() first to convert data into event study formatting.')
        elif self.formatting == 'long_collapsed':
            warnings.warn('Cross section cannot be constructed from collapsed long data. Run collapsed_long_to_es() first to convert data into event study formatting.')
        else:
            warnings.warn('Data cannot be refactored into cross section as its current formatting, {}, is invalid.'.format(self.formatting))

    def cluster(self, user_cluster={}):
        '''
        Cluster data and assign a new column giving the cluster for each firm. Only works if data is formatted as event study.

        Arguments:
            user_cluster (dict): dictionary of parameters for clustering

                Dictionary parameters:

                    cdf_resolution (int): how many values to use to approximate the cdf

                    grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)

                    year (int or None): if None, uses entire dataset; if int, gives year of data to consider

                    user_KMeans (dict): use parameters defined in KMeans_dict for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and use default parameters defined in class attribute default_KMeans for any parameters not specified
        '''
        self.bd.cluster(user_cluster=user_cluster)
        self.data = self.bd.data

class BipartiteLong:
    '''
    Class of BipartiteLong, where BipartiteLong gives a bipartite network of firms and workers in long form.

    Arguments:
        data (Pandas DataFrame): bipartite network data
        col_dict (dict): make data columns readable (requires: wid (worker id), comp (compensation), fid (firm id), year). Keep None if column names already correct
    '''

    def __init__(self, data, col_dict=None):
        # Begin logging
        self.logger = logging.getLogger('bipartite')
        self.logger.setLevel(logging.DEBUG)
        # Create logs folder
        Path('twoway_logs').mkdir(parents=True, exist_ok=True)
        # Create file handler which logs even debug messages
        fh = logging.FileHandler('twoway_logs/bipartite_spam.log')
        fh.setLevel(logging.DEBUG)
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info('initializing BipartiteLong object')

        # Define some attributes
        self.data = data
        self.connected = False # If True, all firms are connected by movers
        self.contiguous = False # If True, firm ids are contiguous
        self.no_na = False # If True, no NaN observations in the data
        self.no_duplicates = False # If True, no duplicate rows in the data

        if col_dict is None: # If columns already correct
            self.col_dict = {
                'fid': 'fid',
                'wid': 'wid',
                'year': 'year',
                'comp': 'comp'
            }
            if 'j' in data.columns:
                self.col_dict['j'] = 'j'
            else:
                self.col_dict['j'] = None
        else:
            self.col_dict = col_dict
            if 'j' not in col_dict.keys():
                self.col_dict['j'] = None

        # Define default parameter dictionaries
        self.default_KMeans = {
            'n_clusters': 10,
            'init': 'k-means++',
            'n_init': 500,
            'max_iter': 300,
            'tol': 0.0001,
            'precompute_distances': 'deprecated',
            'verbose': 0,
            'random_state': None,
            'copy_x': True,
            'n_jobs': 'deprecated',
            'algorithm': 'auto'
        }

        self.default_cluster = {
            'cdf_resolution': 10,
            'grouping': 'quantile_all',
            'year': None,
            'user_KMeans': self.default_KMeans
        }

        self.logger.info('BipartiteLong object initialized')

    def copy(self):
        '''
        Copy the current instance of BipartiteLong.

        Returns:
            bd_copy (BipartiteLong): copy of the current instance of BipartiteLong.
        '''
        bd_copy = BipartiteLong(self.data, self.col_dict)
        bd_copy.connected = self.connected
        bd_copy.contiguous = self.contiguous
        bd_copy.no_na = self.no_na
        bd_copy.no_duplicates = self.no_duplicates

        return bd_copy

    def update_cols(self):
        '''
        Rename columns and keep only relevant columns.
        '''
        self.data = self.data.rename({
            self.col_dict['wid']: 'wid',
            self.col_dict['comp']: 'comp',
            self.col_dict['fid']: 'fid',
            self.col_dict['year']: 'year'
        }, axis=1)
        if self.col_dict['j'] is None:
            self.data = self.data[['wid', 'comp', 'fid', 'year']]
            self.col_dict = {
                'wid': 'wid',
                'comp': 'comp',
                'fid': 'fid',
                'year': 'year'
            }
        else:
            self.data = self.data.rename({self.col_dict['j']: 'j'}, axis=1)
            self.data = self.data[['wid', 'comp', 'fid', 'year', 'j']]
            self.col_dict = {
                'wid': 'wid',
                'comp': 'comp',
                'fid': 'fid',
                'year': 'year',
                'j': 'j'
            }

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
        return len(self.data['fid'].unique())

    def clean_data(self):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.
        '''
        self.logger.info('beginning data cleaning')
        self.logger.info('checking quality of data')
        # Make sure data is valid - computes no_na, no_duplicates, connected, and contiguous, along with other checks (note that column names are corrected in data_validity() if all columns are in the data)
        self.data_validity()

        # Next, drop NaN observations
        if not self.no_na:
            self.logger.info('dropping NaN observations')
            self.data = self.data.dropna()

            # Update no_na
            self.no_na = True

        # Next, drop duplicate observations
        if not self.no_duplicates:
            self.logger.info('dropping duplicate observations')
            self.data = self.data.drop_duplicates()

            # Update no_duplicates
            self.no_duplicates = True

        # Next, find largest set of firms connected by movers
        if not self.connected:
            # Generate largest connected set
            self.logger.info('generating largest connected set')
            self.conset()

        # Next, make firm and worker ids contiguous
        if not self.contiguous:
            # Make firm ids contiguous
            self.logger.info('making firm ids contiguous')
            self.contiguous_ids('fid')
            # Make worker ids contiguous
            self.logger.info('making worker ids contiguous')
            self.contiguous_ids('wid')

        # Using contiguous fids, get NetworkX Graph of largest connected set (note that this must be done even if firms already connected and contiguous)
        self.logger.info('generating NetworkX Graph of largest connected set')
        # self.G = self.conset() # FIXME currently not used

        self.logger.info('data cleaning complete')

    def data_validity(self):
        '''
        Checks that data is formatted correctly and updates relevant attributes.
        '''
        success = True

        all_cols = ['wid', 'comp', 'fid', 'year']

        clustered = self.col_dict['j'] is not None

        if clustered:
            all_cols += ['j']

        self.logger.info('--- checking columns ---')
        cols = True
        for col in all_cols:
            if self.col_dict[col] not in self.data.columns:
                self.logger.info(col, 'missing from data')
                cols = False
            else:
                if col in ['year', 'j']:
                    if self.data[self.col_dict[col]].dtype not in ['int', 'int16', 'int32', 'int64']:
                        self.logger.info(self.col_dict[col], 'has wrong dtype, should be int but is', self.data[self.col_dict[col]].dtype)
                        cols = False
                elif col == 'comp':
                    if self.data[self.col_dict[col]].dtype not in ['float', 'float16', 'float32', 'float64', 'float128', 'int', 'int16', 'int32', 'int64']:
                        self.logger.info(self.col_dict[col], 'has wrong dtype, should be float or int but is', self.data[self.col_dict[col]].dtype)
                        cols = False

        self.logger.info('columns correct:' + str(cols))
        if not cols:
            success = False
            raise ValueError('Your data does not include the correct columns. The TwoWay object cannot be generated with your data.')
        else:
            # Correct column names
            self.logger.info('correcting column names')
            self.update_cols()

        self.logger.info('--- checking worker-year observations ---')
        max_obs = self.data.groupby(['wid', 'year']).size().max()

        self.logger.info('max number of worker-year observations (should be 1):' + str(max_obs))
        if max_obs > 1:
            success = False

        self.logger.info('--- checking nan data ---')
        nans = self.data.shape[0] - self.data.dropna().shape[0]

        self.logger.info('data nans (should be 0):' + str(nans))
        if nans > 0:
            success = False
        else:
            self.no_na = True

        self.logger.info('--- checking duplicates ---')
        duplicates = self.data.shape[0] - self.data.drop_duplicates().shape[0]

        self.logger.info('duplicates (should be 0):' + str(duplicates))
        if duplicates > 0:
            success = False
        else:
            self.no_duplicates = True

        self.logger.info('--- checking connected set ---')
        self.data['fid_max'] = self.data.groupby(['wid'])['fid'].transform(max)
        G = nx.from_pandas_edgelist(self.data, 'fid', 'fid_max')
        largest_cc = max(nx.connected_components(G), key=len)
        self.data = self.data.drop(['fid_max'], axis=1)

        outside_cc = self.data[(~self.data['fid'].isin(largest_cc))].shape[0]

        self.logger.info('observations outside connected set (should be 0):' + str(outside_cc))
        if outside_cc > 0:
            success = False
        else:
            self.connected = True

        self.logger.info('--- checking contiguous firm ids ---')
        fid_max = self.data['fid'].max()
        n_firms = self.n_firms()

        contig = (fid_max == n_firms)

        self.logger.info('contiguous firm ids (should be True):' + str(contig))
        if not contig:
            success = False
        else:
            self.contiguous = True
        
        if clustered:
            self.logger.info('--- checking contiguous cluster ids ---')
            cid_max = self.data['j'].max()
            n_cids = len(self.data['j'].unique())

            contig_cids = (cid_max == n_cids)

            self.logger.info('contiguous cluster ids (should be True):' + str(contig_cids))
            if not contig_cids:
                success = False

        self.logger.info('Overall success:' + str(success))

    def conset(self):
        '''
        Update data to include only the largest connected set of movers, and if firm ids are contiguous, also return the NetworkX Graph.

        Returns:
            G (NetworkX Graph): largest connected set of movers (only returns if firm ids are contiguous, otherwise returns None)
        '''
        prev_workers = self.n_workers()
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

        # Data is now connected
        self.connected = True

        # If connected data != full data, set contiguous to False
        if prev_workers != self.n_workers():
            self.contiguous = False

        # Return G if firm ids are contiguous (if they're not contiguous, they have to be updated first)
        if self.contiguous:
            return G

        return None

    def contiguous_ids(self, id_col):
        '''
        Make column of ids contiguous.
        '''
        # Create sorted set of unique ids
        ids = sorted(list(self.data[id_col].unique()))

        # Create list of adjusted (contiguous) ids
        adjusted_ids = np.arange(len(ids)).astype(int)

        # Create dictionary linking current to new ids, then convert into a dataframe for merging
        ids_dict = {id_col: ids, 'adj_' + id_col: adjusted_ids}
        ids_df = pd.DataFrame(ids_dict, index=adjusted_ids)

        # Merge new, contiguous fids into event study data
        self.data = self.data.merge(ids_df, how='left', on=id_col)

        # Drop old fid column and rename contiguous fid column
        self.data = self.data.drop([id_col], axis=1)
        self.data = self.data.rename({'adj_' + id_col: id_col}, axis=1)

        if id_col == 'fid':
            # Firm ids are now contiguous
            self.contiguous = True

    def get_collapsed_long(self):
        '''
        Collapse long data by job spells (so each spell for a particular worker at a particular firm is one observation).

        Returns:
            data (Pandas DataFrame): long data collapsed by job spells
        '''
        # Copy data
        data = self.data.copy()
        # Sort data by wid and year
        data = data.sort_values(['wid', 'year'])
        self.logger.info('copied data sorted by wid and year')
        # Determine whether clustered
        clustered = 'j' in self.data.columns

        # Introduce lagged fid and wid
        data['fid_l1'] = data['fid'].shift(periods=1)
        data['wid_l1'] = data['wid'].shift(periods=1)
        self.logger.info('lagged fid introduced')

        # Generate spell ids
        # Source: https://stackoverflow.com/questions/59778744/pandas-grouping-and-aggregating-consecutive-rows-with-same-value-in-column
        new_spell = (data['fid'] != data['fid_l1']) | (data['wid'] != data['wid_l1']) # Allow for wid != wid_l1 to ensure that consecutive workers at the same firm get counted as different spells
        data['spell_id'] = new_spell.cumsum()
        self.logger.info('spell ids generated')

        # Aggregate at the spell level
        spell = data.groupby(['spell_id'])
        if clustered:
            data_spell = spell.agg(
                wid=pd.NamedAgg(column='wid', aggfunc='first'),
                comp=pd.NamedAgg(column='comp', aggfunc='mean'),
                fid=pd.NamedAgg(column='fid', aggfunc='first'),
                year_start=pd.NamedAgg(column='year', aggfunc='min'),
                year_end=pd.NamedAgg(column='year', aggfunc='max'),
                weight=pd.NamedAgg(column='wid', aggfunc='size'),
                j=pd.NamedAgg(column='j', aggfunc='first')
            )
        else:
            data_spell = spell.agg(
                wid=pd.NamedAgg(column='wid', aggfunc='first'),
                comp=pd.NamedAgg(column='comp', aggfunc='mean'),
                fid=pd.NamedAgg(column='fid', aggfunc='first'),
                year_start=pd.NamedAgg(column='year', aggfunc='min'),
                year_end=pd.NamedAgg(column='year', aggfunc='max'),
                weight=pd.NamedAgg(column='wid', aggfunc='size')
            )
        # Classify movers and stayers
        spell_count = data_spell.groupby(['wid']).transform('count')['fid'] # Choice of fid arbitrary
        data_spell['m'] = (spell_count > 1).astype(int)
        data = data_spell.reset_index(drop=True)

        self.logger.info('data aggregated at the spell level')

        return data

    def get_es(self):
        '''
        Return long form data reformatted into event study data.

        Returns:
            data_es (Pandas DataFrame): event study data
        '''
        # Compute movers
        m = self.data.groupby('wid')['fid'].transform(lambda x: len(np.unique(x)) > 1)
        # Split movers and stayers
        stayers = self.data[m == 0]
        movers = self.data[m == 1]
        self.logger.info('workers split by movers and stayers')

        # Determine whether clustered
        clustered = self.col_dict['j'] is not None

        # Rename year to year_start
        movers = movers.rename({'year': 'year_start'}, axis=1)
        stayers = stayers.rename({'year': 'year_start'}, axis=1)

        # Add lagged values
        movers = movers.sort_values(['wid', 'year_start'])
        movers['fid_l1'] = movers['fid'].shift(periods=1)
        movers['wid_l1'] = movers['wid'].shift(periods=1) # Used to mark consecutive observations as being for the same worker
        movers['comp_l1'] = movers['comp'].shift(periods=1)
        movers['year_start_l1'] = movers['year_start'].shift(periods=1)
        if clustered:
            movers['j_l1'] = movers['j'].shift(periods=1)
        movers = movers[movers['wid'] == movers['wid_l1']]
        movers[['fid_l1', 'year_start_l1']] = movers[['fid_l1', 'year_start_l1']].astype(int) # Shifting adds nans which converts columns into float, but want int            

        # Update columns
        stayers = stayers.rename({
            'fid': 'f1i',
            'comp': 'y1',
            'year_start': 'year_start_1'
        }, axis=1)
        stayers['f2i'] = stayers['f1i']
        stayers['y2'] = stayers['y1']
        # Since all data is 1 year, define year_end as year_start and weight as 1
        movers['year_end'] = movers['year_start']
        movers['year_end_l1'] = movers['year_start_l1']
        movers['w1'] = 1
        movers['w2'] = 1
        stayers['year_start_2'] = stayers['year_start_1']
        stayers['year_end_1'] = stayers['year_start_1']
        stayers['year_end_2'] = stayers['year_start_1']
        stayers['w1'] = 1
        stayers['w2'] = 1
        
        movers = movers.rename({
            'fid_l1': 'f1i',
            'fid': 'f2i',
            'comp_l1': 'y1',
            'comp': 'y2',
            'year_start_l1': 'year_start_1',
            'year_start': 'year_start_2',
            'year_end_l1': 'year_end_1',
            'year_end': 'year_end_2',
            'weight_l1': 'w1',
            'weight': 'w2'
        }, axis=1)

        # Mark movers and stayers
        stayers['m'] = 0
        movers['m'] = 1

        keep_cols = ['wid', 'y1', 'y2', 'f1i', 'f2i', 'year_start_1', 'year_start_2', 'year_end_1', 'year_end_2', 'w1', 'w2', 'm']

        if clustered:
            stayers = stayers.rename({'j': 'j1'}, axis=1)
            stayers['j2'] = stayers['j1']
            movers['j_l1'] = movers['j_l1'].astype(int)
            movers = movers.rename({'j': 'j2', 'j_l1': 'j1'}, axis=1)
            keep_cols += ['j']

        # Reorder and keep only relevant columns
        stayers = stayers[keep_cols]
        movers = movers[keep_cols]
        self.logger.info('columns updated')

        # Merge stayers and movers
        data_es = pd.concat([stayers, movers])

        self.logger.info('data reformatted as event study')

        return data_es

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
        if isinstance(year, int):
            data = self.data[self.data['year'] == year]

        # Create empty numpy array to fill with the cdfs
        n_firms = self.n_firms()
        cdfs = np.zeros([n_firms, cdf_resolution])

        # Create quantiles of interest
        quantiles = np.linspace(1 / cdf_resolution, 1, cdf_resolution)

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
            for fid in tqdm(range(n_firms)):
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
                    cdfs[fid, i] = comp[index]

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
        # Update dictionary
        cluster_params = update_dict(self.default_cluster, user_cluster)

        # Unpack dictionary
        cdf_resolution = cluster_params['cdf_resolution']
        grouping = cluster_params['grouping']
        user_KMeans = cluster_params['user_KMeans']

        # Compute cdfs
        cdfs = self.approx_cdfs(cdf_resolution=cdf_resolution, grouping=grouping)
        self.logger.info('firm cdfs computed')

        # Compute firm clusters
        KMeans_params = update_dict(self.default_KMeans, user_KMeans)
        clusters = KMeans(**KMeans_params).fit(cdfs).labels_
        self.logger.info('firm clusters computed')

        # Create Pandas dataframe linking fid to firm cluster
        fids = np.arange(self.n_firms())
        clusters_dict = {'fid': fids, 'j': clusters}
        clusters_df = pd.DataFrame(clusters_dict, index=fids)
        self.logger.info('dataframe linking fids to clusters generated')

        # Merge into event study data
        self.data = self.data.merge(clusters_df, how='left', on='fid')
        self.col_dict['j'] = 'j'
        self.logger.info('clusters merged into event study data')

class BipartiteLongCollapsed:
    '''
    Class of BipartiteLongCollapsed, where BipartiteLongCollapsed gives a bipartite network of firms and workers in collapsed long form (i.e. employment spells are collapsed into a single observation).

    Arguments:
        data (Pandas DataFrame): bipartite network data
        col_dict (dict): make data columns readable (requires: wid (worker id), comp (compensation), fid (firm id), year). Keep None if column names already correct
    '''

    def __init__(self, data, col_dict=None):
        # Begin logging
        self.logger = logging.getLogger('bipartite')
        self.logger.setLevel(logging.DEBUG)
        # Create logs folder
        Path('twoway_logs').mkdir(parents=True, exist_ok=True)
        # Create file handler which logs even debug messages
        fh = logging.FileHandler('twoway_logs/bipartite_spam.log')
        fh.setLevel(logging.DEBUG)
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info('initializing BipartiteLongCollapsed object')

        # Define some attributes
        self.data = data
        self.connected = False # If True, all firms are connected by movers
        self.contiguous = False # If True, firm ids are contiguous
        self.no_na = False # If True, no NaN observations in the data
        self.no_duplicates = False # If True, no duplicate rows in the data

        if col_dict is None: # If columns already correct
            self.col_dict = {
                'wid': 'wid',
                'comp': 'comp',
                'fid': 'fid',
                'year_start': 'year_start',
                'year_end': 'year_end',
                'weight': 'weight',
                'm': 'm'
            }
            if 'j' in data.columns:
                self.col_dict['j'] = 'j'
            else:
                self.col_dict['j'] = None
        else:
            self.col_dict = col_dict
            if 'j' not in col_dict.keys():
                self.col_dict['j'] = None

        # Define default parameter dictionaries
        self.default_KMeans = {
            'n_clusters': 10,
            'init': 'k-means++',
            'n_init': 500,
            'max_iter': 300,
            'tol': 0.0001,
            'precompute_distances': 'deprecated',
            'verbose': 0,
            'random_state': None,
            'copy_x': True,
            'n_jobs': 'deprecated',
            'algorithm': 'auto'
        }

        self.default_cluster = {
            'cdf_resolution': 10,
            'grouping': 'quantile_all',
            'year': None,
            'user_KMeans': self.default_KMeans
        }

        self.logger.info('BipartiteLongCollapsed object initialized')

    def copy(self):
        '''
        Copy the current instance of BipartiteLongCollapsed.

        Returns:
            bd_copy (BipartiteLongCollapsed): copy of the current instance of BipartiteLongCollapsed.
        '''
        bd_copy = BipartiteLongCollapsed(self.data, self.col_dict)
        bd_copy.connected = self.connected
        bd_copy.contiguous = self.contiguous
        bd_copy.no_na = self.no_na
        bd_copy.no_duplicates = self.no_duplicates

        return bd_copy

    def update_cols(self):
        '''
        Rename columns and keep only relevant columns.
        '''
        self.data = self.data.rename({
            self.col_dict['wid']: 'wid',
            self.col_dict['comp']: 'comp',
            self.col_dict['fid']: 'fid',
            self.col_dict['year_start']: 'year_start',
            self.col_dict['year_end']: 'year_end',
            self.col_dict['weight']: 'weight',
            self.col_dict['m']: 'm'
        }, axis=1)
        if self.col_dict['j'] is None:
            self.data = self.data[['wid', 'comp', 'fid', 'year']]
            self.col_dict = {
                'wid': 'wid',
                'comp': 'comp',
                'fid': 'fid',
                'year_start': 'year_start',
                'year_end': 'year_end',
                'weight': 'weight',
                'm': 'm'
            }
        else:
            self.data = self.data.rename({self.col_dict['j']: 'j'}, axis=1)
            self.data = self.data[['wid', 'comp', 'fid', 'year', 'j']]
            self.col_dict = {
                'wid': 'wid',
                'comp': 'comp',
                'fid': 'fid',
                'year_start': 'year_start',
                'year_end': 'year_end',
                'weight': 'weight',
                'm': 'm',
                'j': 'j'
            }

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
        return len(self.data['fid'].unique())

    def clean_data(self):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.
        '''
        self.logger.info('beginning data cleaning')
        self.logger.info('checking quality of data')
        # Make sure data is valid - computes no_na, no_duplicates, connected, and contiguous, along with other checks (note that column names are corrected in data_validity() if all columns are in the data)
        self.data_validity()

        # Next, drop NaN observations
        if not self.no_na:
            self.logger.info('dropping NaN observations')
            self.data = self.data.dropna()

            # Update no_na
            self.no_na = True

        # Next, drop duplicate observations
        if not self.no_duplicates:
            self.logger.info('dropping duplicate observations')
            self.data = self.data.drop_duplicates()

            # Update no_duplicates
            self.no_duplicates = True

        # Next, find largest set of firms connected by movers
        if not self.connected:
            # Generate largest connected set
            self.logger.info('generating largest connected set')
            self.conset()

        # Next, make firm and worker ids contiguous
        if not self.contiguous:
            # Make firm ids contiguous
            self.logger.info('making firm ids contiguous')
            self.contiguous_ids('fid')
            # Make worker ids contiguous
            self.logger.info('making worker ids contiguous')
            self.contiguous_ids('wid')

        # Using contiguous fids, get NetworkX Graph of largest connected set (note that this must be done even if firms already connected and contiguous)
        self.logger.info('generating NetworkX Graph of largest connected set')
        # self.G = self.conset() # FIXME currently not used

        self.logger.info('data cleaning complete')

    def data_validity(self):
        '''
        Checks that data is formatted correctly and updates relevant attributes.
        '''
        success = True

        all_cols = ['wid', 'comp', 'fid', 'year_start', 'year_end', 'weight', 'm']

        clustered = self.col_dict['j'] is not None

        if clustered:
            all_cols += ['j']

        self.logger.info('--- checking columns ---')
        cols = True
        for col in all_cols:
            if self.col_dict[col] not in self.data.columns:
                self.logger.info(col, 'missing from data')
                cols = False
            else:
                if col in ['year_start', 'year_end', 'm', 'j']:
                    if self.data[self.col_dict[col]].dtype not in ['int', 'int16', 'int32', 'int64']:
                        self.logger.info(self.col_dict[col], 'has wrong dtype, should be int but is', self.data[self.col_dict[col]].dtype)
                        cols = False
                elif col in ['comp', 'weight']:
                    if self.data[self.col_dict[col]].dtype not in ['float', 'float16', 'float32', 'float64', 'float128', 'int', 'int16', 'int32', 'int64']:
                        self.logger.info(self.col_dict[col], 'has wrong dtype, should be float or int but is', self.data[self.col_dict[col]].dtype)
                        cols = False

        self.logger.info('columns correct:' + str(cols))
        if not cols:
            success = False
            raise ValueError('Your data does not include the correct columns. The TwoWay object cannot be generated with your data.')
        else:
            # Correct column names
            self.logger.info('correcting column names')
            self.update_cols()

        self.logger.info('--- checking worker-year observations ---')
        max_obs_start = self.data.groupby(['wid', 'year_start']).size().max()
        max_obs_end = self.data.groupby(['wid', 'year_end']).size().max()
        max_obs = max(max_obs_start, max_obs_end)

        self.logger.info('max number of worker-year observations (should be 1):' + str(max_obs))
        if max_obs > 1:
            success = False

        self.logger.info('--- checking nan data ---')
        nans = self.data.shape[0] - self.data.dropna().shape[0]

        self.logger.info('data nans (should be 0):' + str(nans))
        if nans > 0:
            success = False
        else:
            self.no_na = True

        self.logger.info('--- checking duplicates ---')
        duplicates = self.data.shape[0] - self.data.drop_duplicates().shape[0]

        self.logger.info('duplicates (should be 0):' + str(duplicates))
        if duplicates > 0:
            success = False
        else:
            self.no_duplicates = True

        self.logger.info('--- checking connected set ---')
        self.data['fid_max'] = self.data.groupby(['wid'])['fid'].transform(max)
        G = nx.from_pandas_edgelist(self.data, 'fid', 'fid_max')
        largest_cc = max(nx.connected_components(G), key=len)
        self.data = self.data.drop(['fid_max'], axis=1)

        outside_cc = self.data[(~self.data['fid'].isin(largest_cc))].shape[0]

        self.logger.info('observations outside connected set (should be 0):' + str(outside_cc))
        if outside_cc > 0:
            success = False
        else:
            self.connected = True

        self.logger.info('--- checking contiguous firm ids ---')
        fid_max = self.data['fid'].max()
        n_firms = self.n_firms()

        contig_fids = (fid_max == n_firms)

        self.logger.info('contiguous firm ids (should be True):' + str(contig_fids))
        if not contig_fids:
            success = False
        else:
            self.contiguous = True

        if clustered:
            self.logger.info('--- checking contiguous cluster ids ---')
            cid_max = self.data['j'].max()
            n_cids = len(self.data['j'].unique())

            contig_cids = (cid_max == n_cids)

            self.logger.info('contiguous cluster ids (should be True):' + str(contig_cids))
            if not contig_cids:
                success = False

        self.logger.info('Overall success:' + str(success))

    def conset(self):
        '''
        Update data to include only the largest connected set of movers, and if firm ids are contiguous, also return the NetworkX Graph.

        Returns:
            G (NetworkX Graph): largest connected set of movers (only returns if firm ids are contiguous, otherwise returns None)
        '''
        prev_workers = self.n_workers()
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

        # Data is now connected
        self.connected = True

        # If connected data != full data, set contiguous to False
        if prev_workers != self.n_workers():
            self.contiguous = False

        # Return G if firm ids are contiguous (if they're not contiguous, they have to be updated first)
        if self.contiguous:
            return G

        return None

    def contiguous_ids(self, id_col):
        '''
        Make column of ids contiguous.
        '''
        # Create sorted set of unique ids
        ids = sorted(list(self.data[id_col].unique()))

        # Create list of adjusted (contiguous) ids
        adjusted_ids = np.arange(len(ids)).astype(int)

        # Create dictionary linking current to new ids, then convert into a dataframe for merging
        ids_dict = {id_col: ids, 'adj_' + id_col: adjusted_ids}
        ids_df = pd.DataFrame(ids_dict, index=adjusted_ids)

        # Merge new, contiguous fids into event study data
        self.data = self.data.merge(ids_df, how='left', on=id_col)

        # Drop old fid column and rename contiguous fid column
        self.data = self.data.drop([id_col], axis=1)
        self.data = self.data.rename({'adj_' + id_col: id_col}, axis=1)

        if id_col == 'fid':
            # Firm ids are now contiguous
            self.contiguous = True

    def get_es(self):
        '''
        Return collapsed long form data reformatted into event study data.

        Returns:
            data_es (Pandas DataFrame): event study data
        '''
        # Split workers by movers and stayers
        stayers = self.data[self.data['m'] == 0]
        movers = self.data[self.data['m'] == 1]
        self.logger.info('workers split by movers and stayers')

        # Determine whether clustered
        clustered = self.col_dict['j'] is not None

        # Add lagged values
        movers = movers.sort_values(['wid', 'year_start'])
        movers['fid_l1'] = movers['fid'].shift(periods=1)
        movers['wid_l1'] = movers['wid'].shift(periods=1) # Used to mark consecutive observations as being for the same worker
        movers['comp_l1'] = movers['comp'].shift(periods=1)
        movers['year_start_l1'] = movers['year_start'].shift(periods=1)
        movers['year_end_l1'] = movers['year_end'].shift(periods=1)
        movers['weight_l1'] = movers['weight'].shift(periods=1)
        if clustered:
            movers['j_l1'] = movers['j'].shift(periods=1)
        movers = movers[movers['wid'] == movers['wid_l1']]
        movers[['fid_l1', 'year_start_l1', 'year_end_l1', 'weight_l1']] = movers[['fid_l1', 'year_start_l1', 'year_end_l1', 'weight_l1']].astype(int) # Shifting adds nans which converts columns into float, but want int

        # Update columns
        stayers = stayers.rename({
            'fid': 'f1i',
            'comp': 'y1',
            'year_start': 'year_start_1',
            'year_end': 'year_end_1',
            'weight': 'w1'
        }, axis=1)
        stayers['f2i'] = stayers['f1i']
        stayers['y2'] = stayers['y1']
        stayers['year_start_2'] = stayers['year_start_1']
        stayers['year_end_2'] = stayers['year_end_1']
        stayers['w2'] = stayers['w1']
        
        movers = movers.rename({
            'fid_l1': 'f1i',
            'fid': 'f2i',
            'comp_l1': 'y1',
            'comp': 'y2',
            'year_start_l1': 'year_start_1',
            'year_start': 'year_start_2',
            'year_end_l1': 'year_end_1',
            'year_end': 'year_end_2',
            'weight_l1': 'w1',
            'weight': 'w2'
        }, axis=1)

        # Mark movers and stayers
        stayers['m'] = 0
        movers['m'] = 1

        keep_cols = ['wid', 'y1', 'y2', 'f1i', 'f2i', 'year_start_1', 'year_start_2', 'year_end_1', 'year_end_2', 'w1', 'w2', 'm']

        if clustered:
            stayers = stayers.rename({'j': 'j1'}, axis=1)
            stayers['j2'] = stayers['j1']
            movers['j_l1'] = movers['j_l1'].astype(int)
            movers = movers.rename({'j_l1': 'j1', 'j': 'j2'}, axis=1)
            keep_cols += ['j']

        # Reorder and keep only relevant columns
        stayers = stayers[keep_cols]
        movers = movers[keep_cols]
        self.logger.info('columns updated')

        # Merge stayers and movers
        data_es = pd.concat([stayers, movers]).reset_index(drop=True)

        self.logger.info('data reformatted as event study')

        return data_es

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
        if isinstance(year, int):
            data = self.data[self.data['year_start'] == year]

        # Create empty numpy array to fill with the cdfs
        n_firms = self.n_firms()
        cdfs = np.zeros([n_firms, cdf_resolution])

        # Create quantiles of interest
        quantiles = np.linspace(1 / cdf_resolution, 1, cdf_resolution)

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
            for fid in tqdm(range(n_firms)):
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
                    cdfs[fid, i] = comp[index]

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
        # Update dictionary
        cluster_params = update_dict(self.default_cluster, user_cluster)

        # Unpack dictionary
        cdf_resolution = cluster_params['cdf_resolution']
        grouping = cluster_params['grouping']
        user_KMeans = cluster_params['user_KMeans']

        # Compute cdfs
        cdfs = self.approx_cdfs(cdf_resolution=cdf_resolution, grouping=grouping)
        self.logger.info('firm cdfs computed')

        # Compute firm clusters
        KMeans_params = update_dict(self.default_KMeans, user_KMeans)
        clusters = KMeans(**KMeans_params).fit(cdfs).labels_
        self.logger.info('firm clusters computed')

        # Create Pandas dataframe linking fid to firm cluster
        fids = np.arange(self.n_firms())
        clusters_dict = {'fid': fids, 'j': clusters}
        clusters_df = pd.DataFrame(clusters_dict, index=fids)
        self.logger.info('dataframe linking fids to clusters generated')

        # Merge into event study data
        self.data = self.data.merge(clusters_df, how='left', on='fid')
        self.col_dict['j'] = 'j'
        self.logger.info('clusters merged into event study data')

class BipartiteEventStudy:
    '''
    Class of BipartiteEventStudy, where BipartiteEventStudy gives a bipartite network of firms and workers in long form.

    Arguments:
        data (Pandas DataFrame): bipartite network data
        collapsed (bool): if True, data collapsed by employment spells
        col_dict (dict): make data columns readable (requires: wid (worker id), y1 (compensation 1), y2 (compensation 2), f1i (firm id 1), f2i (firm id 2), m (0 if stayer, 1 if mover); optionally include: year_end_1 (last year of observation 1), year_end_2 (last year of observation 2), w1 (weight 1), w2 (weight 2)). Keep None if column names already correct
    '''

    def __init__(self, data, collapsed, col_dict=None):
        # Begin logging
        self.logger = logging.getLogger('bipartite')
        self.logger.setLevel(logging.DEBUG)
        # Create logs folder
        Path('twoway_logs').mkdir(parents=True, exist_ok=True)
        # Create file handler which logs even debug messages
        fh = logging.FileHandler('twoway_logs/bipartite_spam.log')
        fh.setLevel(logging.DEBUG)
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info('initializing BipartiteEventStudy object')

        # Define some attributes
        self.data = data
        self.connected = False # If True, all firms are connected by movers
        self.contiguous = False # If True, firm ids are contiguous
        self.no_na = False # If True, no NaN observations in the data
        self.no_duplicates = False # If True, no duplicate rows in the data
        self.collapsed = collapsed

        if col_dict is None: # If columns already correct
            self.col_dict = {
                'wid': 'wid',
                'f1i': 'f1i',
                'f2i': 'f2i',
                'y1': 'y1',
                'y2': 'y2',
                'year_start_1': 'year_start_1',
                'year_start_2': 'year_start_2',
                'year_end_1': 'year_end_1',
                'year_end_2': 'year_end_2',
                'w1': 'w1',
                'w2': 'w2',
                'm': 'm',
            }
            if 'j1' in data.columns and 'j2' in data.columns:
                self.col_dict['j1'] = 'j1'
                self.col_dict['j2'] = 'j2'
            else:
                self.col_dict['j1'] = None
                self.col_dict['j2'] = None
        else:
            self.col_dict = col_dict
            if 'j1' not in col_dict.keys() or 'j2' not in col_dict.keys():
                self.col_dict['j1'] = None
                self.col_dict['j2'] = None

        # Define default parameter dictionaries
        self.default_KMeans = {
            'n_clusters': 10,
            'init': 'k-means++',
            'n_init': 500,
            'max_iter': 300,
            'tol': 0.0001,
            'precompute_distances': 'deprecated',
            'verbose': 0,
            'random_state': None,
            'copy_x': True,
            'n_jobs': 'deprecated',
            'algorithm': 'auto'
        }

        self.default_cluster = {
            'cdf_resolution': 10,
            'grouping': 'quantile_all',
            'year': None,
            'user_KMeans': self.default_KMeans
        }

        self.logger.info('BipartiteEventStudy object initialized')

    def copy(self):
        '''
        Copy the current instance of BipartiteEventStudy.

        Returns:
            bd_copy (BipartiteEventStudy): copy of the current instance of BipartiteEventStudy.
        '''
        bd_copy = BipartiteEventStudy(self.data, self.col_dict)
        bd_copy.connected = self.connected
        bd_copy.contiguous = self.contiguous
        bd_copy.no_na = self.no_na
        bd_copy.no_duplicates = self.no_duplicates

        return bd_copy

    def update_cols(self):
        '''
        Rename columns and keep only relevant columns.
        '''
        self.data = self.data.rename({
            self.col_dict['wid']: 'wid',
            self.col_dict['f1i']: 'f1i',
            self.col_dict['f2i']: 'f2i',
            self.col_dict['y1']: 'y1',
            self.col_dict['y2']: 'y2',
            self.col_dict['year_start_1']: 'year_start_1',
            self.col_dict['year_start_2']: 'year_start_2',
            self.col_dict['year_end_1']: 'year_end_1',
            self.col_dict['year_end_2']: 'year_end_2',
            self.col_dict['w1']: 'w1',
            self.col_dict['w2']: 'w2',
            self.col_dict['m']: 'm'
        }, axis=1)
        if self.col_dict['j1'] is None or self.col_dict['j2'] is None:
            self.data = self.data[['wid', 'f1i', 'f2i', 'y1', 'y2', 'year_start_1', 'year_start_2', 'year_end_1', 'year_end_2', 'w1', 'w2', 'm']]
            self.col_dict = {
                'wid': 'wid',
                'f1i': 'f1i',
                'f2i': 'f2i',
                'y1': 'y1',
                'y2': 'y2',
                'year_start_1': 'year_start_1',
                'year_start_2': 'year_start_2',
                'year_end_1': 'year_end_1',
                'year_end_2': 'year_end_2',
                'w1': 'w1',
                'w2': 'w2',
                'm': 'm',
                'j1': None,
                'j2': None
            }
        else:
            self.data = self.data.rename({self.col_dict['j1']: 'j1', self.col_dict['j2']: 'j2'}, axis=1)
            self.data = self.data[['wid', 'f1i', 'f2i', 'y1', 'y2', 'year_start_1', 'year_start_2', 'year_end_1', 'year_end_2', 'w1', 'w2', 'm', 'j1', 'j2']]
            self.col_dict = {
                'wid': 'wid',
                'f1i': 'f1i',
                'f2i': 'f2i',
                'y1': 'y1',
                'y2': 'y2',
                'year_start_1': 'year_start_1',
                'year_start_2': 'year_start_2',
                'year_end_1': 'year_end_1',
                'year_end_2': 'year_end_2',
                'w1': 'w1',
                'w2': 'w2',
                'm': 'm',
                'j1': 'j1',
                'j2': 'j2'
            }

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
        return len(set(list(self.data['f1i'].unique()) + list(self.data['f2i'].unique())))

    def clean_data(self):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.
        '''
        self.logger.info('beginning data cleaning')
        self.logger.info('checking quality of data')
        # Make sure data is valid - computes no_na, no_duplicates, connected, and contiguous, along with other checks (note that column names are corrected in data_validity() if all columns are in the data)
        self.data_validity()

        # Next, drop NaN observations
        if not self.no_na:
            self.logger.info('dropping NaN observations')
            self.data = self.data.dropna()

            # Update no_na
            self.no_na = True

        # Next, drop duplicate observations
        if not self.no_duplicates:
            self.logger.info('dropping duplicate observations')
            self.data = self.data.drop_duplicates()

            # Update no_duplicates
            self.no_duplicates = True

        # Next, find largest set of firms connected by movers
        if not self.connected:
            # Generate largest connected set
            self.logger.info('generating largest connected set')
            self.conset()

        # Next, make firm and worker ids contiguous
        if not self.contiguous:
            # Make firm ids contiguous
            self.logger.info('making firm ids contiguous')
            self.contiguous_ids('fid')
            # Make worker ids contiguous
            self.logger.info('making worker ids contiguous')
            self.contiguous_ids('wid')

        # Using contiguous fids, get NetworkX Graph of largest connected set (note that this must be done even if firms already connected and contiguous)
        self.logger.info('generating NetworkX Graph of largest connected set')
        # self.G = self.conset() # FIXME currently not used

        self.logger.info('data cleaning complete')

    def data_validity(self):
        '''
        Checks that data is formatted correctly and updates relevant attributes.
        '''
        success_stayers = True
        success_movers = True

        all_cols = ['wid', 'y1', 'y2', 'f1i', 'f2i', 'year_start_1', 'year_start_2', 'year_end_1', 'year_end_2', 'w1', 'w2', 'm']

        clustered = self.col_dict['j1'] is not None and self.col_dict['j2'] is not None

        if clustered:
            all_cols += ['j1', 'j2']

        self.logger.info('--- checking columns ---')
        cols = True
        for col in all_cols:
            if self.col_dict[col] not in self.data.columns:
                self.logger.info(col, 'column missing from data')
                cols = False
            else:
                if col in ['y1', 'y2', 'w1', 'w2']:
                    if self.data[self.col_dict[col]].dtype not in ['float', 'float16', 'float32', 'float64', 'float128', 'int', 'int16', 'int32', 'int64']:
                        self.logger.info(col, 'column has wrong dtype, should be float or int but is', self.data[self.col_dict[col]].dtype)
                        cols = False
                elif col in ['year_start_1', 'year_start_2', 'year_end_1', 'year_end_2', 'm', 'j1', 'j2']:
                    if self.data[self.col_dict[col]].dtype not in ['int', 'int16', 'int32', 'int64']:
                        self.logger.info(col, 'column has wrong dtype, should be int but is', self.data[self.col_dict[col]].dtype)
                        cols = False

        self.logger.info('columns correct:' + str(cols))
        if not cols:
            success_stayers = False
            success_movers = False
            raise ValueError('Your data does not include the correct columns. The TwoWay object cannot be generated with your data.')
        else:
            # Correct column names
            self.logger.info('correcting column names')
            self.update_cols()

        stayers = self.data[self.data['m'] == 0]
        movers = self.data[self.data['m'] == 1]

        self.logger.info('--- checking nan data ---')
        na_stayers = stayers.shape[0] - stayers.dropna().shape[0]
        na_movers = movers.shape[0] - movers.dropna().shape[0]

        self.logger.info('stayers nans (should be 0):' + str(na_stayers))
        self.logger.info('movers nans (should be 0):' + str(na_movers))
        if na_stayers > 0:
            success_stayers = False
        if na_movers > 0:
            success_movers = False
        if (na_stayers == 0) and (na_movers == 0):
            self.no_na = True

        self.logger.info('--- checking duplicates ---')
        duplicates_stayers = stayers.shape[0] - stayers.drop_duplicates().shape[0]
        duplicates_movers = movers.shape[0] - movers.drop_duplicates().shape[0]

        self.logger.info('stayers duplicates (should be 0):' + str(duplicates_stayers))
        self.logger.info('movers duplicates (should be 0):' + str(duplicates_movers))
        if duplicates_stayers > 0:
            success_stayers = False
        if duplicates_movers > 0:
            success_movers = False
        if (duplicates_stayers == 0) and (duplicates_movers == 0):
            self.no_duplicates = True

        self.logger.info('--- checking firms ---')
        firms_stayers = (stayers['f1i'] != stayers['f2i']).sum()
        firms_movers = (movers['f1i'] == movers['f2i']).sum()

        self.logger.info('stayers with different firms (should be 0):' + str(firms_stayers))
        self.logger.info('movers with same firm (should be 0):' + str(firms_movers))
        if firms_stayers > 0:
            success_stayers = False
        if firms_movers > 0:
            success_movers = False

        self.logger.info('--- checking income ---')
        income_stayers = (stayers['y1'] != stayers['y2']).sum()

        self.logger.info('stayers with different income (should be 0):' + str(income_stayers))
        if income_stayers > 0:
            success_stayers = False

        self.logger.info('--- checking connected set ---')
        G = nx.from_pandas_edgelist(movers, 'f1i', 'f2i')
        largest_cc = max(nx.connected_components(G), key=len)

        cc_stayers = stayers[(~stayers['f1i'].isin(largest_cc)) | (~stayers['f2i'].isin(largest_cc))].shape[0]
        cc_movers = movers[(~movers['f1i'].isin(largest_cc)) | (~movers['f2i'].isin(largest_cc))].shape[0]

        self.logger.info('stayers outside connected set (should be 0):' + str(cc_stayers))
        self.logger.info('movers outside connected set (should be 0):' + str(cc_movers))
        if cc_stayers > 0:
            success_stayers = False
        if cc_movers > 0:
            success_movers = False
        if (cc_stayers == 0) and (cc_movers == 0):
            self.connected = True

        self.logger.info('--- checking contiguous firm ids ---')
        fid_max = max(self.data['f1i'].max(), self.data['f2i'].max())
        n_firms = self.n_firms()

        contig_fids = (fid_max + 1 == n_firms)
        self.logger.info('contiguous firm ids (should be True):' + str(contig_fids))
        self.contiguous = contig_fids
        if not contig_fids:
            success_stayers = False
            success_movers = False

        if clustered:
            self.logger.info('--- checking contiguous cluster ids ---')
            cid_max = max(self.data['j1'].max(), self.data['j2'].max())
            n_cids = len(set(list(self.data['j1'].unique()) + list(self.data['j2'].unique())))

            contig_cids = (cid_max + 1 == n_cids)
            self.logger.info('contiguous cluster ids (should be True):' + str(contig_cids))

            if not contig_cids:
                success_stayers = False
                success_movers = False

        self.logger.info('Overall success for stayers:' + str(success_stayers))
        self.logger.info('Overall success for movers:' + str(success_movers))

    def conset(self):
        '''
        Update data to include only the largest connected set of movers, and if firm ids are contiguous, also return the NetworkX Graph.

        Returns:
            G (NetworkX Graph): largest connected set of movers (only returns if firm ids are contiguous, otherwise returns None)
        '''
        prev_workers = self.n_workers()
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

    def contiguous_ids(self, id_col):
        '''
        Make column of ids contiguous.

        Parameters:
            id_col (str): column to make contiguous
        '''
        # Generate id_list (note that all columns listed in id_list are included in the set of ids, and all columns are adjusted to have the new, contiguous ids)
        if id_col == 'fid':
            id_list = ['f1i', 'f2i']
        elif id_col == 'j':
            id_list = ['j1', 'j2']
        elif id_col == 'wid':
            id_list = ['wid']
        # Create sorted set of unique ids
        ids = []
        for id in id_list:
            ids += list(self.data[id].unique())
        ids = sorted(list(set(ids)))

        # Create list of adjusted ids
        adjusted_ids = np.arange(len(ids)).astype(int)

        # Update each fid one at a time
        for id in id_list:
            # Create dictionary linking current to new ids, then convert into a dataframe for merging
            ids_dict = {id: ids, 'adj_' + id: adjusted_ids}
            ids_df = pd.DataFrame(ids_dict, index=adjusted_ids)

            # Merge new, contiguous ids into event study data
            self.data = self.data.merge(ids_df, how='left', on=id)

            # Drop old id column and rename contiguous id column
            self.data = self.data.drop([id], axis=1)
            self.data = self.data.rename({'adj_' + id: id}, axis=1)

        if id_col == 'fid':
            # Firm ids are now contiguous
            self.contiguous = True

    def get_cs(self):
        '''
        Return event study data reformatted into cross section data.

        Returns:
            data_cs (Pandas DataFrame): cross section data
        '''
        sdata = self.data[self.data['m'] == 0]
        jdata = self.data[self.data['m'] == 1]

        # Assign some values
        ns = len(sdata)
        nm = len(jdata)

        # Determine whether clustered
        clustered = self.col_dict['j1'] is not None and self.col_dict['j2'] is not None

        # # Reset index
        # sdata.set_index(np.arange(ns) + 1 + nm)
        # jdata.set_index(np.arange(nm) + 1)

        # Combine the 2 data-sets
        if clustered: # If clustering, include j1 and j2
            data_cs = pd.concat([
                sdata[['wid', 'f1i', 'f2i', 'y1', 'y2', 'j1', 'j2', 'year_start_1', 'year_start_2', 'year_end_1', 'year_end_2', 'w1', 'w2', 'm']].assign(cs=1), jdata[['wid', 'f1i', 'f2i', 'y1', 'y2', 'j1', 'j2', 'year_start_1', 'year_start_2', 'year_end_1', 'year_end_2', 'w1', 'w2', 'm']].assign(cs=1), jdata[['wid', 'f1i', 'f2i', 'y2', 'y1', 'j1', 'j2', 'year_start_2', 'year_start_1', 'year_end_2', 'year_end_1', 'w2', 'w1', 'm']].rename({'f1i': 'f2i', 'f2i': 'f1i', 'y1': 'y2', 'y2': 'y1', 'j1': 'j2', 'j2': 'j1', 'year_start_1': 'year_start_2', 'year_start_2': 'year_start_1', 'year_end_1': 'year_end_2', 'year_end_2': 'year_end_1', 'w1': 'w2', 'w2': 'w1'}, axis=1).assign(cs=0)], ignore_index=True)
        else: # If not clustering, no j1 or j2
            data_cs = pd.concat([
                sdata[['wid', 'f1i', 'f2i', 'y1', 'y2', 'year_start_1', 'year_start_2', 'year_end_1', 'year_end_2', 'w1', 'w2', 'm']].assign(cs=1), jdata[['wid', 'f1i', 'f2i', 'y1', 'y2', 'year_start_1', 'year_start_2', 'year_end_1', 'year_end_2', 'w1', 'w2', 'm']].assign(cs=1), jdata[['wid', 'f1i', 'f2i', 'y2', 'y1', 'year_start_2', 'year_start_1', 'year_end_2', 'year_end_1', 'w2', 'w1', 'm']].rename({'f1i': 'f2i', 'f2i': 'f1i', 'y1': 'y2', 'y2': 'y1', 'year_start_1': 'year_start_2', 'year_start_2': 'year_start_1', 'year_end_1': 'year_end_2', 'year_end_2': 'year_end_1', 'w1': 'w2', 'w2': 'w1'}, axis=1).assign(cs=0)], ignore_index=True)
        # self.data = self.data.reset_index(drop=True)
        # self.data['wid'] = self.data['wid'].astype('category').cat.codes + 1
        self.logger.info('mover and stayer long form datasets combined into cross section')

        return data_cs

    def get_collapsed_long(self):
        '''
        Return event study data reformatted into collapsed long form.

        Returns:
            (Pandas DataFrame): collapsed long data
        '''
        # Determine whether clustered
        clustered = self.col_dict['j1'] is not None and self.col_dict['j2'] is not None
        # Append the last row if a mover (this is because the last observation is only given as an f2i, never as an f1i)
        if clustered:
            return self.data.groupby('wid').apply(lambda a: a.append(a.iloc[-1].rename({'f1i': 'f2i', 'f2i': 'f1i', 'y1': 'y2', 'y2': 'y1', 'year_start_1': 'year_start_2', 'year_start_2': 'year_start_1', 'year_end_1': 'year_end_2', 'year_end_2': 'year_end_1', 'w1': 'w2', 'w2': 'w1', 'j1': 'j2', 'j2': 'j1'}, axis=1)) if a.iloc[0]['m'] == 1 else a) \
                .reset_index(drop=True) \
                .drop(['f2i', 'y2', 'year_start_2', 'year_end_2', 'w2', 'j2'], axis=1) \
                .rename({'f1i': 'fid', 'y1': 'comp', 'year_start_1': 'year_start', 'year_end_1': 'year_end', 'w1': 'weight', 'j1': 'j'}, axis=1) \
                .astype({'wid': int, 'fid': int, 'year_start': int, 'year_end': int, 'weight': int, 'm': int})
        else:
            return self.data.groupby('wid').apply(lambda a: a.append(a.iloc[-1].rename({'f1i': 'f2i', 'f2i': 'f1i', 'y1': 'y2', 'y2': 'y1', 'year_start_1': 'year_start_2', 'year_start_2': 'year_start_1', 'year_end_1': 'year_end_2', 'year_end_2': 'year_end_1', 'w1': 'w2', 'w2': 'w1'}, axis=1)) if a.iloc[0]['m'] == 1 else a) \
                .reset_index(drop=True) \
                .drop(['f2i', 'y2', 'year_start_2', 'year_end_2', 'w2'], axis=1) \
                .rename({'f1i': 'fid', 'y1': 'comp', 'year_start_1': 'year_start', 'year_end_1': 'year_end', 'w1': 'weight'}, axis=1) \
                .astype({'wid': int, 'fid': int, 'year_start': int, 'year_end': int, 'weight': int, 'm': int})

    def get_long(self):
        '''
        Return event study data reformatted into long form.

        Returns:
            (Pandas DataFrame): long data
        '''
        # Determine whether clustered
        clustered = self.col_dict['j1'] is not None and self.col_dict['j2'] is not None
        # Append the last row if a mover (this is because the last observation is only given as an f2i, never as an f1i)
        if clustered:
            return self.data.groupby('wid').apply(lambda a: a.append(a.iloc[-1].rename({'f1i': 'f2i', 'f2i': 'f1i', 'y1': 'y2', 'y2': 'y1', 'year_start_1': 'year_start_2', 'year_start_2': 'year_start_1', 'year_end_1': 'year_end_2', 'year_end_2': 'year_end_1', 'w1': 'w2', 'w2': 'w1', 'j1': 'j2', 'j2': 'j1'}, axis=1)) if a.iloc[0]['m'] == 1 else a) \
                .reset_index(drop=True) \
                .drop(['f2i', 'y2', 'year_start_2', 'year_end_1', 'year_end_2', 'w1', 'w2', 'm', 'j2'], axis=1) \
                .rename({'f1i': 'fid', 'y1': 'comp', 'year_start_1': 'year', 'j1': 'j'}, axis=1) \
                .astype({'wid': int, 'fid': int, 'year': int})
        else:
            return self.data.groupby('wid').apply(lambda a: a.append(a.iloc[-1].rename({'f1i': 'f2i', 'f2i': 'f1i', 'y1': 'y2', 'y2': 'y1', 'year_start_1': 'year_start_2', 'year_start_2': 'year_start_1', 'year_end_1': 'year_end_2', 'year_end_2': 'year_end_1', 'w1': 'w2', 'w2': 'w1'}, axis=1)) if a.iloc[0]['m'] == 1 else a) \
                .reset_index(drop=True) \
                .drop(['f2i', 'y2', 'year_start_2', 'year_end_1', 'year_end_2', 'w1', 'w2', 'm'], axis=1) \
                .rename({'f1i': 'fid', 'y1': 'comp', 'year_start_1': 'year'}, axis=1) \
                .astype({'wid': int, 'fid': int, 'year': int})

    def approx_cdfs(self, cdf_resolution=10, grouping='quantile_all'):
        '''
        Generate cdfs of compensation for firms.

        Arguments:
            cdf_resolution (int): how many values to use to approximate the cdf
            grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)

        Returns:
            cdf_df (NumPy Array): NumPy array of firm cdfs
        '''
        # Create empty numpy array to fill with the cdfs
        n_firms = self.n_firms()
        cdfs = np.zeros([n_firms, cdf_resolution])

        # Create quantiles of interest
        quantiles = np.linspace(1 / cdf_resolution, 1, cdf_resolution)

        # Re-arrange event study data to be in long format (temporarily)
        self.data = self.data.rename({'f1i': 'fid', 'y1': 'comp'}, axis=1)
        self.data = pd.concat([self.data, self.data.rename({'f2i': 'fid', 'y2': 'comp', 'fid': 'f2i', 'comp': 'y2'}, axis=1).assign(f2i = - 1)], axis=0) # Include irrelevant columns and rename f1i to f2i to prevent nans, which convert columns from int into float # FIXME duplicating both movers and stayers, should probably only be duplicating movers

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
            for fid in tqdm(range(n_firms)):
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
                    cdfs[fid, i] = comp[index]

        # Drop rows that were appended earlier and rename columns
        self.data = self.data[self.data['f2i'] >= 0]
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
        # Update dictionary
        cluster_params = update_dict(self.default_cluster, user_cluster)

        # Unpack dictionary
        cdf_resolution = cluster_params['cdf_resolution']
        grouping = cluster_params['grouping']
        user_KMeans = cluster_params['user_KMeans']

        # Compute cdfs
        cdfs = self.approx_cdfs(cdf_resolution=cdf_resolution, grouping=grouping)
        self.logger.info('firm cdfs computed')

        # Compute firm clusters
        KMeans_params = update_dict(self.default_KMeans, user_KMeans)
        clusters = KMeans(**KMeans_params).fit(cdfs).labels_
        self.logger.info('firm clusters computed')

        # Create Pandas dataframe linking fid to firm cluster
        fids = np.arange(self.n_firms())
        clusters_dict_1 = {'f1i': fids, 'j1': clusters}
        clusters_dict_2 = {'f2i': fids, 'j2': clusters}
        clusters_df_1 = pd.DataFrame(clusters_dict_1, index=fids)
        clusters_df_2 = pd.DataFrame(clusters_dict_2, index=fids)
        self.logger.info('dataframes linking fids to clusters generated')

        # Merge into event study data
        self.data = self.data.merge(clusters_df_1, how='left', on='f1i')
        self.data = self.data.merge(clusters_df_2, how='left', on='f2i')
        self.col_dict['j1'] = 'j1'
        self.col_dict['j2'] = 'j2'
        self.logger.info('clusters merged into event study data')
