'''
Class for a bipartite network
'''
import logging
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from pandas import DataFrame
import networkx as nx
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import connected_components
import warnings
from pytwoway import update_dict

col_order = ['wid', 'fid', 'f1i', 'f2i', 'comp', 'y1', 'y2', 'year', 'year_1', 'year_2', 'year_start', 'year_end', 'year_start_1', 'year_end_1', 'year_start_2', 'year_end_2', 'weight', 'w1', 'w2', 'j', 'j1', 'j2', 'm', 'cs'].index

def _col_dict_optional_cols(default_col_dict, user_col_dict, data_cols, optional_cols=()):
    '''
    Update col_dict to account for whether certain optional columns are included.

    Arguments:
        default_col_dict (dict): default col_dict values
        user_col_dict (dict): user col_dict
        data_cols (list): columns from user data
        optional_cols (list of lists): optional columns to check if included in user data. If sub-list has multiple columns, all columns must be included in the data for them to be added to new_col_dict.

    Returns:
        new_col_dict (dict): updated col_dict
    '''
    if user_col_dict is None: # If columns already correct
        new_col_dict = default_col_dict
    else:
        new_col_dict = update_dict(default_col_dict, user_col_dict)
    # Add columns in 
    for col_list in optional_cols:
        include = True
        for col in col_list:
            exists_assigned = new_col_dict[col] is not None
            exists_not_assigned = (col in data_cols) and (new_col_dict[col] is None) and (col not in new_col_dict.values()) # Last condition checks whether data has a different column with same name
            if not exists_assigned and not exists_not_assigned:
                include = False
        if include:
            for col in col_list:
                if new_col_dict[col] is None:
                    new_col_dict[col] = col
        else: # Reset column names to None if not all essential columns included
            for col in col_list:
                new_col_dict[col] = None
    return new_col_dict

def _update_cols(bipartite, inplace=True):
    '''
    Rename columns and keep only relevant columns.

    Arguments:
        bipartite (BipartiteData): bipartite object
        inplace (bool): if True, modify in-place

    Returns:
        frame (BipartiteData): bipartite object with updated columns
    '''
    if inplace:
        frame = bipartite
    else:
        frame = bipartite.copy()

    new_col_dict = {}
    rename_dict = {} # For renaming columns in data
    keep_cols = []

    for key, val in frame.col_dict.items():
        if val is not None:
            rename_dict[val] = key
            new_col_dict[key] = key
            keep_cols.append(key)
        else:
            new_col_dict[key] = None
    frame.col_dict = new_col_dict
    keep_cols = sorted(keep_cols, key=col_order) # Sort columns
    DataFrame.rename(frame, rename_dict, axis=1, inplace=True)
    frame = frame[keep_cols]

    return frame

def _to_list(data):
    '''
    Convert data into a list if it isn't already.

    Arguments:
        data (obj): data to check if it's a list

    Returns:
        (list): data as a list
    '''
    if not isinstance(data, (list, tuple)):
        return [data]
    return data

# class BipartiteBase(DataFrame):
#     '''
#     Base class for BipartitePandas. Contains generalized methods.

#     Arguments:
#         columns_req (list): required columns
#         columns_opt (list): optional columns
#     '''

#     def __init__(self, *args, columns_req=[], columns_opt=[], **kwargs):
#         super().__init__(*args, **kwargs)

# def __repr__(self):
#     '''
#     Print statement.
#     '''
#     if self.formatting == 'long':
#         collapsed = False
#     elif self.formatting == 'long_collapsed':
#         collapsed = True
#     else:
#         collapsed = self.bd.collapsed
#     if self.formatting in ['long', 'long_collapsed']:
#         mean_wage = np.mean(self.data['comp'])
#         max_wage = self.data['comp'].max()
#         min_wage = self.data['comp'].min()
#     elif self.formatting == 'es':
#         mean_wage = np.mean(self.data[['y1', 'y2']].to_numpy().flatten())
#         max_wage = self.data[['y1', 'y2']].to_numpy().flatten().max()
#         min_wage = self.data[['y1', 'y2']].to_numpy().flatten().min()
#     ret_str = 'format: ' + self.formatting + '\n'
#     ret_str += 'number of workers: ' + str(self.bd.n_workers()) + '\n'
#     ret_str += 'number of firms: ' + str(self.bd.n_firms()) + '\n'
#     ret_str += 'number of observations: ' + str(len(self.bd.data)) + '\n'
#     ret_str += 'mean wage: ' + str(mean_wage) + '\n'
#     ret_str += 'max wage: ' + str(max_wage) + '\n'
#     ret_str += 'min wage: ' + str(min_wage) + '\n'
#     ret_str += 'collapsed by spell: ' + str(collapsed) + '\n'
#     ret_str += 'connected: ' + str(self.bd.connected) + '\n'
#     ret_str += 'contiguous firm ids: ' + str(self.bd.contiguous_fids) + '\n'
#     ret_str += 'contiguous worker ids: ' + str(self.bd.contiguous_wids) + '\n'
#     ret_str += 'contiguous cluster ids (None if not clustered): ' + str(self.bd.contiguous_cids) + '\n'
#     ret_str += 'no nans: ' + str(self.bd.no_na) + '\n'
#     ret_str += 'no duplicates: ' + str(self.bd.no_duplicates) + '\n'

#     return ret_str

class BipartitePandas(DataFrame):
    '''
    Class of BipartitePandas, where BipartitePandas gives a bipartite network of firms and workers in long form.

    Arguments:
        *args: arguments for Pandas DataFrame
        col_dict (dict): make data columns readable (requires: wid (worker id), comp (compensation), fid (firm id), year). Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, **kwargs):
        # Initialize DataFrame
        super().__init__(*args, **kwargs)

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
        self.logger.info('initializing BipartitePandas object')

        # Define some attributes
        self.connected = False # If True, all firms are connected by movers
        self.contiguous_fids = False # If True, firm ids are contiguous
        self.contiguous_wids = False # If True, worker ids are contiguous
        self.contiguous_cids = None # If True, cluster ids are contiguous; if None, data not clustered
        self.no_na = False # If True, no NaN observations in the data
        self.no_duplicates = False # If True, no duplicate rows in the data

        # Define default parameter dictionaries
        default_col_dict = {
            'wid': 'wid',
            'fid': 'fid',
            'comp': 'comp',
            'year': 'year',
            'm': None,
            'j': None
        }
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
            'stayers_movers': None,
            'year': None,
            'dropna': False,
            'user_KMeans': self.default_KMeans
        }

        # Create self.col_dict
        optional_cols = [['m'], ['j']]
        self.col_dict = _col_dict_optional_cols(default_col_dict, col_dict, self.columns, optional_cols=optional_cols)

        if self.col_dict['j'] is not None:
            self.contiguous_cids = False

        self.logger.info('BipartitePandas object initialized')

    def copy(self):
        '''
        Copy the current instance of BipartitePandas.

        Returns:
            bd_copy (BipartitePandas): copy of the current instance of BipartitePandas.
        '''
        bd_copy = BipartitePandas(DataFrame.copy(self), col_dict=self.col_dict)
        bd_copy.connected = self.connected
        bd_copy.contiguous_fids = self.contiguous_fids
        bd_copy.contiguous_wids = self.contiguous_wids
        bd_copy.contiguous_cids = self.contiguous_cids
        bd_copy.no_na = self.no_na
        bd_copy.no_duplicates = self.no_duplicates

        return bd_copy

    def n_workers(self):
        '''
        Get the number of unique workers.

        Returns:
            (int): number of unique workers
        '''
        return len(self['wid'].unique())

    def n_firms(self):
        '''
        Get the number of unique firms.

        Returns:
            (int): number of unique firms
        '''
        return len(self['fid'].unique())

    def n_clusters(self):
        '''
        Get the number of unique clusters.

        Returns:
            (int): number of unique clusters
        '''
        if self.col_dict['j'] is not None:
            return len(self['j'].unique())
        return 0

    def drop(self, indices, axis, inplace=True):
        '''
        Drop indices along axis.

        Arguments:
            indices (int or str, optionally as a list): row(s) or column(s) to drop
            axis (int): 0 to drop rows, 1 to drop columns
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartitePandas): BipartitePandas with dropped indices
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        if axis == 1:
            for col in _to_list(indices):
                if col in frame.columns:
                    if col in ['m', 'j']:
                        DataFrame.drop(frame, col, axis=1, inplace=True)
                        frame.col_dict[col] = None
                        if col == 'j':
                            frame.contiguous_cids = None
                    else:
                        warnings.warn('{} is an essential column and cannot be dropped')
                else:
                    warnings.warn('{} is not in data columns')
        elif axis == 0:
            DataFrame.drop(frame, indices, axis=0, inplace=True)
            frame.connected = False
            frame.contiguous_fids = False
            frame.contiguous_wids = False
            if frame.contiguous_cids is not None:
                frame.contiguous_cids = False
            frame.no_na = False
            frame.no_duplicates = False
            frame.clean_data()

        return frame

    def rename(self, rename_dict, inplace=True):
        '''
        Rename a column.

        Arguments:
            rename_dict (dict): key is current column name, value is new column name
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartitePandas): BipartitePandas with renamed columns
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        for col_cur, col_new in rename_dict.items():
            if col_cur in frame.columns:
                if col_cur in ['m', 'j']:
                    frame.rename({col_cur: col_new}, axis=1, inplace=True)
                    frame.col_dict[col_cur] = None
                    if col_cur == 'j':
                        frame.contiguous_cids = None
                else:
                    warnings.warn('{} is an essential column and cannot be renamed')
            else:
                warnings.warn('{} is not in data columns')

        return frame

    def clean_data(self, inplace=True):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartitePandas): BipartitePandas with cleaned data
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        frame.logger.info('beginning data cleaning')
        frame.logger.info('checking quality of data')
        # Make sure data is valid - computes no_na, no_duplicates, connected, and contiguous, along with other checks (note that column names are corrected in data_validity() if all columns are in the data)
        frame.data_validity()

        # Next, drop NaN observations
        if not frame.no_na:
            frame.logger.info('dropping NaN observations')
            frame.dropna(inplace=True)

            # Update no_na
            frame.no_na = True

        # Next, drop duplicate observations
        if not frame.no_duplicates:
            frame.logger.info('dropping duplicate observations')
            frame.drop_duplicates(inplace=True)

            # Update no_duplicates
            frame.no_duplicates = True

        # Next, find largest set of firms connected by movers
        if not frame.connected:
            # Generate largest connected set
            frame.logger.info('generating largest connected set')
            frame.conset()

        # Next, make firm ids contiguous
        if not frame.contiguous_fids:
            frame.logger.info('making firm ids contiguous')
            frame.contiguous_ids('fid')

        # Next, make worker ids contiguous
        if not frame.contiguous_wids:
            frame.logger.info('making firm ids contiguous')
            frame.contiguous_ids('wid')

        # Next, make cluster ids contiguous
        if frame.contiguous_cids is not None and not frame.contiguous_cids:
            frame.logger.info('making cluster ids contiguous')
            frame.contiguous_ids('j')

        # Using contiguous fids, get NetworkX Graph of largest connected set (note that this must be done even if firms already connected and contiguous)
        frame.logger.info('generating NetworkX Graph of largest connected set')
        # frame.G = frame.conset() # FIXME currently not used

        frame.logger.info('data cleaning complete')

        return frame

    def data_validity(self, inplace=True):
        '''
        Checks that data is formatted correctly and updates relevant attributes.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartitePandas): BipartitePandas with corrected columns and attributes
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        success = True

        all_cols = ['wid', 'comp', 'fid', 'year']

        # Determine whether m, cluster columns exist
        m = frame.col_dict['m'] is not None
        clustered = frame.col_dict['j'] is not None

        if m:
            all_cols += ['m']
        if clustered:
            all_cols += ['j']

        frame.logger.info('--- checking columns ---')
        cols = True
        for col in all_cols:
            if frame.col_dict[col] not in frame.columns:
                frame.logger.info(col, 'missing from data')
                cols = False
            else:
                if col in ['year', 'm', 'j']:
                    if frame[self.col_dict[col]].dtype not in ['int', 'int16', 'int32', 'int64', 'Int64']:
                        frame.logger.info(frame.col_dict[col], 'has wrong dtype, should be int but is', frame[frame.col_dict[col]].dtype)
                        cols = False
                elif col == 'comp':
                    if frame[frame.col_dict[col]].dtype not in ['float', 'float16', 'float32', 'float64', 'float128', 'int', 'int16', 'int32', 'int64', 'Int64']:
                        frame.logger.info(frame.col_dict[col], 'has wrong dtype, should be float or int but is', frame[frame.col_dict[col]].dtype)
                        cols = False

        frame.logger.info('columns correct:' + str(cols))
        if not cols:
            success = False
            raise ValueError('Your data does not include the correct columns. The TwoWay object cannot be generated with your data.')
        else:
            # Correct column names
            frame.logger.info('correcting column names')
            _update_cols(frame)

        frame.logger.info('--- checking worker-year observations ---')
        max_obs = frame.groupby(['wid', 'year']).size().max()

        frame.logger.info('max number of worker-year observations (should be 1):' + str(max_obs))
        if max_obs > 1:
            success = False

        frame.logger.info('--- checking nan data ---')
        nans = frame.shape[0] - frame.dropna().shape[0]

        frame.logger.info('data nans (should be 0):' + str(nans))
        if nans > 0:
            frame.no_na = False
            success = False
        else:
            frame.no_na = True

        frame.logger.info('--- checking duplicates ---')
        duplicates = frame.shape[0] - frame.drop_duplicates().shape[0]

        frame.logger.info('duplicates (should be 0):' + str(duplicates))
        if duplicates > 0:
            frame.no_duplicates = False
            success = False
        else:
            frame.no_duplicates = True

        frame.logger.info('--- checking connected set ---')
        frame['fid_max'] = frame.groupby(['wid'])['fid'].transform(max)
        G = nx.from_pandas_edgelist(frame, 'fid', 'fid_max')
        largest_cc = max(nx.connected_components(G), key=len)
        DataFrame.drop(frame, ['fid_max'], axis=1, inplace=True)

        outside_cc = frame[(~frame['fid'].isin(largest_cc))].shape[0]

        frame.logger.info('observations outside connected set (should be 0):' + str(outside_cc))
        if outside_cc > 0:
            frame.connected = False
            success = False
        else:
            frame.connected = True

        frame.logger.info('--- checking contiguous firm ids ---')
        fid_max = frame['fid'].max()
        n_firms = frame.n_firms()

        contig_fids = (fid_max == n_firms - 1)
        frame.contiguous_fids = contig_fids

        frame.logger.info('contiguous firm ids (should be True):' + str(contig_fids))
        if not contig_fids:
            success = False

        frame.logger.info('--- checking contiguous worker ids ---')
        wid_max = frame['wid'].max()
        n_workers = frame.n_workers()

        contig_wids = (wid_max == n_workers - 1)
        frame.contiguous_wids = contig_wids

        frame.logger.info('contiguous worker ids (should be True):' + str(contig_wids))
        if not contig_wids:
            success = False
        
        if clustered:
            frame.logger.info('--- checking contiguous cluster ids ---')
            cid_max = frame['j'].max()
            n_cids = len(frame['j'].unique())

            contig_cids = (cid_max == n_cids)
            frame.contiguous_cids = contig_cids

            frame.logger.info('contiguous cluster ids (should be True):' + str(contig_cids))
            if not contig_cids:
                success = False

        frame.logger.info('Overall success:' + str(success))

        return frame

    def conset(self, inplace=True):
        '''
        Update data to include only the largest connected set of movers, and if firm ids are contiguous, also return the NetworkX Graph.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            (tuple):
                frame (BipartitePandas): BipartitePandas with connected set of movers
                G (NetworkX Graph): largest connected set of movers (only returns if firm ids are contiguous, otherwise returns None)
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        prev_workers = frame.n_workers()
        prev_firms = frame.n_firms()
        prev_clusters = frame.n_clusters()
        # Add max firm id per worker to serve as a central node for the worker
        # frame['fid_f1'] = frame.groupby('wid')['fid'].transform(lambda a: a.shift(-1)) # FIXME - this is directed but is much slower
        frame['fid_max'] = frame.groupby(['wid'])['fid'].transform(max) # FIXME - this is undirected but is much faster

        # Find largest connected set
        # Source: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html
        G = nx.from_pandas_edgelist(frame, 'fid', 'fid_max')
        # Drop fid_max
        DataFrame.drop(frame, ['fid_max'], axis=1, inplace=True)
        # Update data if not connected
        if not frame.connected:
            largest_cc = max(nx.connected_components(G), key=len)
            # Keep largest connected set of firms
            frame = frame[frame['fid'].isin(largest_cc)]

        # Data is now connected
        frame.connected = True

        # If connected data != full data, set contiguous to False
        if prev_firms != frame.n_firms():
            frame.contiguous_fids = False
        if prev_workers != frame.n_workers():
            frame.contiguous_wids = False
        if prev_clusters != frame.n_clusters():
            if frame.col_dict['j'] is not None:
                frame.contiguous_cids = False

        # Return G if all ids are contiguous (if they're not contiguous, they have to be updated first)
        if frame.contiguous_fids and frame.contiguous_wids and (frame.col_dict['j'] is None or frame.contiguous_cids):
            return frame, G

        return frame, None

    def contiguous_ids(self, id_col, inplace=True):
        '''
        Make column of ids contiguous.

        Arguments:
            id_col (str): column to make contiguous ('fid', 'wid', or 'j')
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartitePandas): BipartitePandas with contiguous ids
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        # Create sorted set of unique ids
        ids = sorted(list(frame[id_col].unique()))

        # Create list of adjusted (contiguous) ids
        adjusted_ids = np.arange(len(ids)).astype(int)

        # Create dictionary linking current to new ids, then convert into a dataframe for merging
        ids_dict = {id_col: ids, 'adj_' + id_col: adjusted_ids}
        ids_df = pd.DataFrame(ids_dict, index=adjusted_ids)

        # Merge new, contiguous fids into event study data
        frame = frame.merge(ids_df, how='left', on=id_col)

        # Drop old fid column and rename contiguous fid column
        DataFrame.drop(frame, [id_col], axis=1, inplace=True)
        DataFrame.rename(frame, {'adj_' + id_col: id_col}, axis=1, inplace=True)

        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        if id_col == 'fid':
            # Firm ids are now contiguous
            frame.contiguous_fids = True
        elif id_col == 'wid':
            # Worker ids are now contiguous
            frame.contiguous_wids = True
        elif id_col == 'j':
            # Cluster ids are now contiguous
            frame.contiguous_cids = True

        return frame

    def gen_m(self, inplace=True):
        '''
        Generate m column for data (m == 0 if stayer, m == 1 if mover).

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartitePandas): BipartitePandas with m column
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        frame['m'] = frame.groupby('wid')['fid'].transform(lambda x: len(np.unique(x)) > 1).astype(int)
        frame.col_dict['m'] = 'm'
        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        return frame

    def get_collapsed_long(self):
        '''
        Collapse long data by job spells (so each spell for a particular worker at a particular firm is one observation).

        Returns:
            collapsed_frame (BipartiteCollapsed): BipartiteCollapsed object generated from long data collapsed by job spells
        '''
        # Copy data
        data = DataFrame(self, copy=True)
        # Sort data by wid and year
        data = data.sort_values(['wid', 'year'])
        self.logger.info('copied data sorted by wid and year')
        # Determine whether m, cluster columns exist
        m = self.col_dict['m'] is not None
        clustered = self.col_dict['j'] is not None

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
        if m and clustered:
            data_spell = spell.agg(
                wid=pd.NamedAgg(column='wid', aggfunc='first'),
                comp=pd.NamedAgg(column='comp', aggfunc='mean'),
                fid=pd.NamedAgg(column='fid', aggfunc='first'),
                year_start=pd.NamedAgg(column='year', aggfunc='min'),
                year_end=pd.NamedAgg(column='year', aggfunc='max'),
                weight=pd.NamedAgg(column='wid', aggfunc='size'),
                m=pd.NamedAgg(column='m', aggfunc='first'),
                j=pd.NamedAgg(column='j', aggfunc='first')
            )
        elif m:
            data_spell = spell.agg(
                wid=pd.NamedAgg(column='wid', aggfunc='first'),
                comp=pd.NamedAgg(column='comp', aggfunc='mean'),
                fid=pd.NamedAgg(column='fid', aggfunc='first'),
                year_start=pd.NamedAgg(column='year', aggfunc='min'),
                year_end=pd.NamedAgg(column='year', aggfunc='max'),
                weight=pd.NamedAgg(column='wid', aggfunc='size'),
                m=pd.NamedAgg(column='m', aggfunc='first')
            )
        elif clustered:
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
        if not m:
            spell_count = data_spell.groupby(['wid']).transform('count')['fid'] # Choice of fid arbitrary
            data_spell['m'] = (spell_count > 1).astype(int)
        data = data_spell.reset_index(drop=True)

        # Sort columns
        sorted_cols = sorted(data.columns, key=col_order)
        data = data[sorted_cols]

        self.logger.info('data aggregated at the spell level')

        collapsed_frame = BipartiteCollapsed(data)
        collapsed_frame.connected = self.connected
        collapsed_frame.contiguous_fids = self.contiguous_fids
        collapsed_frame.contiguous_wids = self.contiguous_wids
        collapsed_frame.contiguous_cids = self.contiguous_cids
        collapsed_frame.no_na = self.no_na
        collapsed_frame.no_duplicates = self.no_duplicates

        return collapsed_frame

    def get_es(self):
        '''
        Return long form data reformatted into event study data.

        Returns:
            es_frame (BipartiteEventStudy): BipartiteEventStudy object generated from long data
        '''
        frame = self.copy()
        # Determine whether m, cluster columns exist
        m = frame.col_dict['m'] is not None
        clustered = frame.col_dict['j'] is not None
        if not m:
            # Generate m column
            frame.gen_m()
        # Split workers by movers and stayers
        stayers = frame[frame['m'] == 0]
        movers = frame[frame['m'] == 1]
        frame.logger.info('workers split by movers and stayers')

        # Add lagged values
        movers = movers.sort_values(['wid', 'year'])
        movers['fid_l1'] = movers['fid'].shift(periods=1)
        movers['wid_l1'] = movers['wid'].shift(periods=1) # Used to mark consecutive observations as being for the same worker
        movers['comp_l1'] = movers['comp'].shift(periods=1)
        movers['year_l1'] = movers['year'].shift(periods=1)
        if clustered:
            movers['j_l1'] = movers['j'].shift(periods=1)
        movers = movers[movers['wid'] == movers['wid_l1']]
        movers[['fid_l1', 'year_l1']] = movers[['fid_l1', 'year_l1']].astype(int) # Shifting adds nans which converts columns into float, but want int

        # Update columns
        stayers = stayers.rename({
            'fid': 'f1i',
            'comp': 'y1',
            'year': 'year_1'
        }, axis=1)
        stayers['f2i'] = stayers['f1i']
        stayers['y2'] = stayers['y1']
        stayers['year_2'] = stayers['year_1']
        
        movers = movers.rename({
            'fid_l1': 'f1i',
            'fid': 'f2i',
            'comp_l1': 'y1',
            'comp': 'y2',
            'year': 'year_2',
            'year_l1': 'year_1',
        }, axis=1)

        keep_cols = ['wid', 'y1', 'y2', 'f1i', 'f2i', 'year_1', 'year_2', 'm']

        if clustered:
            stayers = stayers.rename({'j': 'j1'}, axis=1)
            stayers['j2'] = stayers['j1']
            movers['j_l1'] = movers['j_l1'].astype(int)
            movers = movers.rename({'j': 'j2', 'j_l1': 'j1'}, axis=1)
            keep_cols += ['j1', 'j2']

        # Keep only relevant columns
        stayers = stayers[keep_cols]
        movers = movers[keep_cols]
        frame.logger.info('columns updated')

        # Merge stayers and movers
        data_es = pd.concat([stayers, movers]).reset_index(drop=True)

        # Sort columns
        sorted_cols = sorted(data_es.columns, key=col_order)
        data_es = data_es[sorted_cols]

        frame.logger.info('data reformatted as event study')

        es_frame = BipartiteEventStudy(data_es, collapsed=False)
        es_frame.connected = self.connected
        es_frame.contiguous_fids = self.contiguous_fids
        es_frame.contiguous_wids = self.contiguous_wids
        es_frame.contiguous_cids = self.contiguous_cids
        es_frame.no_na = self.no_na
        es_frame.no_duplicates = self.no_duplicates

        return es_frame

    def approx_cdfs(self, cdf_resolution=10, grouping='quantile_all', stayers_movers=None, year=None):
        '''
        Generate cdfs of compensation for firms.

        Arguments:
            cdf_resolution (int): how many values to use to approximate the cdf
            grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)
            stayers_movers (str or None): if None, uses entire dataset; if 'stayers', uses only stayers; if 'movers', uses only movers
            year (int or None): if None, uses entire dataset; if int, gives year of data to consider

        Returns:
            cdf_df (NumPy Array): NumPy array of firm cdfs
            n_firms (int): number of firms in subset of data used to cluster
        '''
        # Determine whether m column exists
        m = self.col_dict['m'] is not None

        if not m:
            self.gen_m()

        if stayers_movers == 'stayers':
            data = self[self['m'] == 0]
        elif stayers_movers == 'movers':
            data = self[self['m'] == 1]
        else:
            data = self

        # If year-level, then only use data for that particular year
        if isinstance(year, int):
            data = data[data['year'] == year]

        # Create empty numpy array to fill with the cdfs
        # n_firms = self.n_firms()
        n_firms = len(data['fid'].unique()) # Can't use self.n_firms() since data could be a subset of self.data
        cdfs = np.zeros([n_firms, cdf_resolution])

        # Create quantiles of interest
        quantiles = np.linspace(1 / cdf_resolution, 1, cdf_resolution)

        if grouping == 'quantile_all':
            # Get quantiles from all data
            quantile_groups = data['comp'].quantile(quantiles)

            # Generate firm-level cdfs
            for i, quant in enumerate(quantile_groups):
                cdfs[:, i] = data.assign(firm_quant=lambda d: d['comp'] <= quant).groupby('fid')['firm_quant'].agg(sum).to_numpy()

            # Normalize by firm size (convert to cdf)
            fsize = data.groupby('fid').size().to_numpy()
            cdfs /= np.expand_dims(fsize, 1)

        elif grouping in ['quantile_firm_small', 'quantile_firm_large']:
            # Sort data by compensation (do this once now, so that don't need to do it again later) (also note it is faster to sort then manually compute quantiles than to use built-in quantile functions)
            data = data.sort_values(['comp'])

            if grouping == 'quantile_firm_small':
                # Convert pandas dataframe into a dictionary to access data faster
                # Source for idea: https://stackoverflow.com/questions/57208997/looking-for-the-fastest-way-to-slice-a-row-in-a-huge-pandas-dataframe
                # Source for how to actually format data correctly: https://stackoverflow.com/questions/56064677/pandas-series-to-dict-with-repeated-indices-make-dict-with-list-values
                data_dict = data['comp'].groupby(level=0).agg(list).to_dict()

            # Generate the cdfs
            for fid in tqdm(range(n_firms)):
                # Get the firm-level compensation data (don't need to sort because already sorted)
                if grouping == 'quantile_firm_small':
                    comp = data_dict[fid]
                elif grouping == 'quantile_firm_large':
                    comp = data.loc[data['fid'] == fid, 'comp']
                # Generate the firm-level cdf
                # Note: update numpy array element by element
                # Source: https://stackoverflow.com/questions/30012362/faster-way-to-convert-list-of-objects-to-numpy-array/30012403
                for i in range(cdf_resolution):
                    index = max(len(comp) * (i + 1) // cdf_resolution - 1, 0) # Don't want negative index
                    # Update cdfs with the firm-level cdf
                    cdfs[fid, i] = comp[index]

        return cdfs, n_firms

    def cluster(self, user_cluster={}, inplace=True):
        '''
        Cluster data and assign a new column giving the cluster for each firm.

        Arguments:
            user_cluster (dict): dictionary of parameters for clustering

                Dictionary parameters:

                    cdf_resolution (int): how many values to use to approximate the cdf

                    grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)

                    stayers_movers (str or None): if None, uses entire dataset; if 'stayers', uses only stayers; if 'movers', uses only movers

                    year (int or None): if None, uses entire dataset; if int, gives year of data to consider

                    dropna (bool): if True, drop observations where firms aren't clustered; if False, keep all observations

                    user_KMeans (dict): use parameters defined in KMeans_dict for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and use default parameters defined in class attribute default_KMeans for any parameters not specified

            inplace (bool): if True, modify in-place
        Returns:
            frame (BipartitePandas): BipartitePandas with clusters
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        # Update dictionary
        cluster_params = update_dict(frame.default_cluster, user_cluster)

        # Unpack dictionary
        cdf_resolution = cluster_params['cdf_resolution']
        grouping = cluster_params['grouping']
        year = cluster_params['year']
        stayers_movers = cluster_params['stayers_movers']
        user_KMeans = cluster_params['user_KMeans']

        # Compute cdfs
        cdfs, n_firms = frame.approx_cdfs(cdf_resolution=cdf_resolution, grouping=grouping, stayers_movers=stayers_movers, year=year)
        frame.logger.info('firm cdfs computed')

        # Compute firm clusters
        KMeans_params = update_dict(frame.default_KMeans, user_KMeans)
        clusters = KMeans(**KMeans_params).fit(cdfs).labels_
        frame.logger.info('firm clusters computed')

        # Create Pandas dataframe linking fid to firm cluster
        fids = np.arange(n_firms)
        clusters_dict = {'fid': fids, 'j': clusters}
        clusters_df = pd.DataFrame(clusters_dict, index=fids)
        frame.logger.info('dataframe linking fids to clusters generated')

        # Drop j column from data if it already exists
        if frame.col_dict['j'] is not None:
            DataFrame.drop(frame, 'j', axis=1, inplace=True)

        # Merge into event study data
        print(type(frame))
        frame = frame.merge(clusters_df, how='left', on='fid')
        print(type(frame))
        stop
        # Keep column as int even with nans
        frame['j'] = frame['j'].astype('Int64')
        frame.col_dict['j'] = 'j'

        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        if cluster_params['dropna']:
            # Drop firms that don't get clustered
            frame = frame.dropna().reset_index(drop=True)
            frame['j'] = frame['j'].astype(int)
            frame.clean_data()

        frame.logger.info('clusters merged into event study data')

        return frame

class BipartiteCollapsed(DataFrame):
    '''
    Class of BipartiteCollapsed, where BipartiteCollapsed gives a bipartite network of firms and workers in collapsed long form (i.e. employment spells are collapsed into a single observation).

    Arguments:
        *args: arguments for Pandas DataFrame
        col_dict (dict): make data columns readable (requires: wid (worker id), comp (compensation), fid (firm id), year). Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, **kwargs):
        # Initialize DataFrame
        super().__init__(*args, **kwargs)

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
        self.logger.info('initializing BipartiteCollapsed object')

        # Define some attributes
        self.connected = False # If True, all firms are connected by movers
        self.contiguous_fids = False # If True, firm ids are contiguous
        self.contiguous_wids = False # If True, worker ids are contiguous
        self.contiguous_cids = None # If True, cluster ids are contiguous; if None, data not clustered
        self.no_na = False # If True, no NaN observations in the data
        self.no_duplicates = False # If True, no duplicate rows in the data

        # Define default parameter dictionaries
        default_col_dict = {
            'wid': 'wid',
            'fid': 'fid',
            'comp': 'comp',
            'year_start': 'year_start',
            'year_end': 'year_end',
            'm': None,
            'weight': None,
            'j': None
        }

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
            'stayers_movers': None,
            'year': None,
            'dropna': False,
            'user_KMeans': self.default_KMeans
        }

        # Create self.col_dict
        optional_cols = [['m'], ['weight'], ['j']]
        self.col_dict = _col_dict_optional_cols(default_col_dict, col_dict, self.columns, optional_cols=optional_cols)

        if self.col_dict['j'] is not None:
            self.contiguous_cids = False

        self.logger.info('BipartiteCollapsed object initialized')

    def copy(self):
        '''
        Copy the current instance of BipartiteCollapsed.

        Returns:
            bd_copy (BipartiteCollapsed): copy of the current instance of BipartiteCollapsed.
        '''
        bd_copy = BipartiteCollapsed(DataFrame.copy(self), col_dict=self.col_dict)
        bd_copy.connected = self.connected
        bd_copy.contiguous_fids = self.contiguous_fids
        bd_copy.contiguous_wids = self.contiguous_wids
        bd_copy.contiguous_cids = self.contiguous_cids
        bd_copy.no_na = self.no_na
        bd_copy.no_duplicates = self.no_duplicates

        return bd_copy

    def n_workers(self):
        '''
        Get the number of unique workers.

        Returns:
            (int): number of unique workers
        '''
        return len(self['wid'].unique())

    def n_firms(self):
        '''
        Get the number of unique firms.

        Returns:
            (int): number of unique firms
        '''
        return len(self['fid'].unique())

    def n_clusters(self):
        '''
        Get the number of unique clusters.

        Returns:
            (int): number of unique clusters
        '''
        if self.col_dict['j'] is not None:
            return len(self['j'].unique())
        return 0

    def drop(self, indices, axis, inplace=True):
        '''
        Drop indices along axis.

        Arguments:
            indices (int or str, optionally as a list): row(s) or column(s) to drop
            axis (int): 0 to drop rows, 1 to drop columns
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteCollapsed): BipartiteCollapsed with dropped indices
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        if axis == 1:
            for col in _to_list(indices):
                if col in frame.columns:
                    if col in ['m', 'weight', 'j']:
                        DataFrame.drop(frame, col, axis=1, inplace=True)
                        frame.col_dict[col] = None
                        if col == 'j':
                            frame.contiguous_cids = None
                    else:
                        warnings.warn('{} is an essential column and cannot be dropped')
                else:
                    warnings.warn('{} is not in data columns')
        elif axis == 0:
            DataFrame.drop(frame, indices, axis=0, inplace=True)
            frame.connected = False
            frame.contiguous_fids = False
            frame.contiguous_wids = False
            if frame.contiguous_cids is not None:
                frame.contiguous_cids = False
            frame.no_na = False
            frame.no_duplicates = False
            frame.clean_data()

        return frame

    def rename(self, rename_dict, inplace=True):
        '''
        Rename a column.

        Arguments:
            rename_dict (dict): key is current column name, value is new column name
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteCollapsed): BipartiteCollapsed with renamed columns
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        for col_cur, col_new in rename_dict.items():
            if col_cur in frame.columns:
                if col_cur in ['m', 'weight', 'j']:
                    frame.rename({col_cur: col_new}, axis=1, inplace=True)
                    frame.col_dict[col_cur] = None
                    if col_cur == 'j':
                        frame.contiguous_cids = None
                else:
                    warnings.warn('{} is an essential column and cannot be renamed')
            else:
                warnings.warn('{} is not in data columns')

        return frame

    def clean_data(self, inplace=True):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteCollapsed): BipartiteCollapsed with cleaned data
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        frame.logger.info('beginning data cleaning')
        frame.logger.info('checking quality of data')
        # Make sure data is valid - computes no_na, no_duplicates, connected, and contiguous, along with other checks (note that column names are corrected in data_validity() if all columns are in the data)
        frame.data_validity()

        # Next, drop NaN observations
        if not frame.no_na:
            frame.logger.info('dropping NaN observations')
            frame.dropna(inplace=True)

            # Update no_na
            frame.no_na = True

        # Next, drop duplicate observations
        if not frame.no_duplicates:
            frame.logger.info('dropping duplicate observations')
            frame.drop_duplicates(inplace=True)

            # Update no_duplicates
            frame.no_duplicates = True

        # Next, find largest set of firms connected by movers
        if not frame.connected:
            # Generate largest connected set
            frame.logger.info('generating largest connected set')
            frame.conset()

        # Next, make firm ids contiguous
        if not self.contiguous_fids:
            self.logger.info('making firm ids contiguous')
            self.contiguous_ids('fid')

        # Next, make worker ids contiguous
        if not frame.contiguous_wids:
            frame.logger.info('making firm ids contiguous')
            frame.contiguous_ids('wid')

        # Next, make cluster ids contiguous
        if frame.contiguous_cids is not None and not frame.contiguous_cids:
            frame.logger.info('making cluster ids contiguous')
            frame.contiguous_ids('j')

        # Using contiguous fids, get NetworkX Graph of largest connected set (note that this must be done even if firms already connected and contiguous)
        frame.logger.info('generating NetworkX Graph of largest connected set')
        # frame.G = frame.conset() # FIXME currently not used

        frame.logger.info('data cleaning complete')

        return frame

    def data_validity(self, inplace=True):
        '''
        Checks that data is formatted correctly and updates relevant attributes.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteCollapsed): BipartiteCollapsed with corrected columns and attributes
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        success = True

        all_cols = ['wid', 'comp', 'fid', 'year_start', 'year_end']

        # Determine whether weight, m, cluster columns exist
        weighted = frame.col_dict['weight'] is not None
        m = frame.col_dict['m'] is not None
        clustered = frame.col_dict['j'] is not None

        if weighted:
            all_cols += ['weight']
        if m:
            all_cols += ['m']
        if clustered:
            all_cols += ['j']

        frame.logger.info('--- checking columns ---')
        cols = True
        for col in all_cols:
            if frame.col_dict[col] not in frame.columns:
                frame.logger.info(col, 'missing from data')
                cols = False
            else:
                if col in ['year_start', 'year_end', 'm', 'j']:
                    if frame[frame.col_dict[col]].dtype not in ['int', 'int16', 'int32', 'int64', 'Int64']:
                        frame.logger.info(frame.col_dict[col], 'has wrong dtype, should be int but is', frame[frame.col_dict[col]].dtype)
                        cols = False
                elif col in ['comp', 'weight']:
                    if frame[frame.col_dict[col]].dtype not in ['float', 'float16', 'float32', 'float64', 'float128', 'int', 'int16', 'int32', 'int64', 'Int64']:
                        frame.logger.info(frame.col_dict[col], 'has wrong dtype, should be float or int but is', frame[frame.col_dict[col]].dtype)
                        cols = False

        frame.logger.info('columns correct:' + str(cols))
        if not cols:
            success = False
            raise ValueError('Your data does not include the correct columns. The TwoWay object cannot be generated with your data.')
        else:
            # Correct column names
            frame.logger.info('correcting column names')
            _update_cols(frame)

        frame.logger.info('--- checking worker-year observations ---')
        max_obs_start = frame.groupby(['wid', 'year_start']).size().max()
        max_obs_end = frame.groupby(['wid', 'year_end']).size().max()
        max_obs = max(max_obs_start, max_obs_end)

        frame.logger.info('max number of worker-year observations (should be 1):' + str(max_obs))
        if max_obs > 1:
            success = False

        frame.logger.info('--- checking nan data ---')
        nans = frame.shape[0] - frame.dropna().shape[0]

        frame.logger.info('data nans (should be 0):' + str(nans))
        if nans > 0:
            frame.no_na = False
            success = False
        else:
            frame.no_na = True

        frame.logger.info('--- checking duplicates ---')
        duplicates = frame.shape[0] - frame.drop_duplicates().shape[0]

        frame.logger.info('duplicates (should be 0):' + str(duplicates))
        if duplicates > 0:
            frame.no_duplicates = False
            success = False
        else:
            frame.no_duplicates = True

        frame.logger.info('--- checking connected set ---')
        frame['fid_max'] = frame.groupby(['wid'])['fid'].transform(max)
        G = nx.from_pandas_edgelist(frame, 'fid', 'fid_max')
        largest_cc = max(nx.connected_components(G), key=len)
        DataFrame.drop(frame, ['fid_max'], axis=1, inplace=True)

        outside_cc = frame[(~frame['fid'].isin(largest_cc))].shape[0]

        frame.logger.info('observations outside connected set (should be 0):' + str(outside_cc))
        if outside_cc > 0:
            frame.connected = False
            success = False
        else:
            frame.connected = True

        frame.logger.info('--- checking contiguous firm ids ---')
        fid_max = frame['fid'].max()
        n_firms = frame.n_firms()

        contig_fids = (fid_max == n_firms - 1)
        frame.contiguous_fids = contig_fids

        frame.logger.info('contiguous firm ids (should be True):' + str(contig_fids))
        if not contig_fids:
            success = False

        frame.logger.info('--- checking contiguous worker ids ---')
        wid_max = max(frame['wid'].max())
        n_workers = frame.n_workers()

        contig_wids = (wid_max == n_workers - 1)
        frame.contiguous_wids = contig_wids

        frame.logger.info('contiguous worker ids (should be True):' + str(contig_wids))
        if not contig_wids:
            success = False

        if clustered:
            frame.logger.info('--- checking contiguous cluster ids ---')
            cid_max = frame['j'].max()
            n_cids = len(frame['j'].unique())

            contig_cids = (cid_max == n_cids)
            frame.contiguous_cids = contig_cids

            frame.logger.info('contiguous cluster ids (should be True):' + str(contig_cids))
            if not contig_cids:
                success = False

        frame.logger.info('Overall success:' + str(success))

        return frame

    def conset(self, inplace=True):
        '''
        Update data to include only the largest connected set of movers, and if firm ids are contiguous, also return the NetworkX Graph.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            (tuple):
                frame (BipartiteCollapsed): BipartiteCollapsed with connected set of movers
                G (NetworkX Graph): largest connected set of movers (only returns if firm ids are contiguous, otherwise returns None)
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        prev_workers = frame.n_workers()
        prev_firms = frame.n_firms()
        prev_clusters = frame.n_clusters()
        # Add max firm id per worker to serve as a central node for the worker
        # frame['fid_f1'] = frame.groupby('wid')['fid'].transform(lambda a: a.shift(-1)) # FIXME - this is directed but is much slower
        frame['fid_max'] = frame.groupby(['wid'])['fid'].transform(max) # FIXME - this is undirected but is much faster

        # Find largest connected set
        # Source: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html
        G = nx.from_pandas_edgelist(frame, 'fid', 'fid_max')
        # Drop fid_max
        DataFrame.drop(frame, ['fid_max'], axis=1, inplace=True)
        # Update data if not connected
        if not frame.connected:
            largest_cc = max(nx.connected_components(G), key=len)
            # Keep largest connected set of firms
            frame = frame[frame['fid'].isin(largest_cc)]

        # Data is now connected
        frame.connected = True

        # If connected data != full data, set contiguous to False
        if prev_firms != frame.n_firms():
            frame.contiguous_fids = False
        if prev_workers != frame.n_workers():
            frame.contiguous_wids = False
        if prev_clusters != frame.n_clusters():
            if frame.col_dict['j'] is not None:
                frame.contiguous_cids = False

        # Return G if all ids are contiguous (if they're not contiguous, they have to be updated first)
        if frame.contiguous_fids and frame.contiguous_wids and (frame.col_dict['j'] is None or frame.contiguous_cids):
            return frame, G

        return frame, None

    def contiguous_ids(self, id_col, inplace=True):
        '''
        Make column of ids contiguous.

        Arguments:
            id_col (str): column to make contiguous ('fid', 'wid', or 'j')
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteCollapsed): BipartiteCollapsed with contiguous ids
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        # Create sorted set of unique ids
        ids = sorted(list(frame[id_col].unique()))

        # Create list of adjusted (contiguous) ids
        adjusted_ids = np.arange(len(ids)).astype(int)

        # Create dictionary linking current to new ids, then convert into a dataframe for merging
        ids_dict = {id_col: ids, 'adj_' + id_col: adjusted_ids}
        ids_df = pd.DataFrame(ids_dict, index=adjusted_ids)

        # Merge new, contiguous fids into event study data
        frame = frame.merge(ids_df, how='left', on=id_col)

        # Drop old fid column and rename contiguous fid column
        DataFrame.drop(frame, [id_col], axis=1, inplace=True)
        DataFrame.rename(frame, {'adj_' + id_col: id_col}, axis=1, inplace=True)

        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        if id_col == 'fid':
            # Firm ids are now contiguous
            frame.contiguous_fids = True
        elif id_col == 'wid':
            # Worker ids are now contiguous
            frame.contiguous_wids = True
        elif id_col == 'j':
            # Cluster ids are now contiguous
            frame.contiguous_cids = True

        return frame

    def gen_m(self, inplace=True):
        '''
        Generate m column for data (m == 0 if stayer, m == 1 if mover).

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteCollapsed): BipartiteCollapsed with m column
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        frame['m'] = frame.groupby('wid')['fid'].transform(lambda x: len(np.unique(x)) > 1).astype(int)
        frame.col_dict['m'] = 'm'
        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        return frame

    def get_es(self):
        '''
        Return collapsed long form data reformatted into event study data.

        Returns:
            es_frame (BipartiteEventStudy): BipartiteEventStudy object generated from collapsed long data
        '''
        # Determine whether m, cluster columns exist
        weighted = self.col_dict['weight'] is not None
        m = self.col_dict['m'] is not None
        clustered = self.col_dict['j'] is not None

        if not m:
            # Generate m column
            self.gen_m()

        # Split workers by movers and stayers
        stayers = self[self['m'] == 0]
        movers = self[self['m'] == 1]
        self.logger.info('workers split by movers and stayers')

        # Add lagged values
        movers = movers.sort_values(['wid', 'year_start'])
        movers['fid_l1'] = movers['fid'].shift(periods=1)
        movers['wid_l1'] = movers['wid'].shift(periods=1) # Used to mark consecutive observations as being for the same worker
        movers['comp_l1'] = movers['comp'].shift(periods=1)
        movers['year_start_l1'] = movers['year_start'].shift(periods=1)
        movers['year_end_l1'] = movers['year_end'].shift(periods=1)
        if weighted:
            movers['weight_l1'] = movers['weight'].shift(periods=1)
        if clustered:
            movers['j_l1'] = movers['j'].shift(periods=1)
        movers = movers[movers['wid'] == movers['wid_l1']]
        movers[['fid_l1', 'year_start_l1', 'year_end_l1']] = movers[['fid_l1', 'year_start_l1', 'year_end_l1']].astype(int) # Shifting adds nans which converts columns into float, but want int

        # Update columns
        stayers = stayers.rename({
            'fid': 'f1i',
            'comp': 'y1',
            'year_start': 'year_start_1',
            'year_end': 'year_end_1',
            'weight': 'w1', # Optional
            'j': 'j1' # Optional
        }, axis=1)
        stayers['f2i'] = stayers['f1i']
        stayers['y2'] = stayers['y1']
        stayers['year_start_2'] = stayers['year_start_1']
        stayers['year_end_2'] = stayers['year_end_1']

        movers = movers.rename({
            'fid_l1': 'f1i',
            'fid': 'f2i',
            'comp_l1': 'y1',
            'comp': 'y2',
            'year_start_l1': 'year_start_1',
            'year_start': 'year_start_2',
            'year_end_l1': 'year_end_1',
            'year_end': 'year_end_2',
            'weight_l1': 'w1', # Optional
            'weight': 'w2', # Optional
            'j_l1': 'j1', # Optional
            'j': 'j2' # Optional
        }, axis=1)

        keep_cols = ['wid', 'y1', 'y2', 'f1i', 'f2i', 'year_start_1', 'year_start_2', 'year_end_1', 'year_end_2', 'm']

        if weighted:
            stayers['w2'] = stayers['w1']
            movers['w1'] = movers['w1'].astype(int)
            keep_cols += ['w1', 'w2']
        if clustered:
            stayers['j2'] = stayers['j1']
            movers['j1'] = movers['j1'].astype(int)
            keep_cols += ['j1', 'j2']

        # Keep only relevant columns
        stayers = stayers[keep_cols]
        movers = movers[keep_cols]
        self.logger.info('columns updated')

        # Merge stayers and movers
        data_es = pd.concat([stayers, movers]).reset_index(drop=True)

        # Sort columns
        sorted_cols = sorted(data_es.columns, key=col_order)
        data_es = data_es[sorted_cols]

        self.logger.info('data reformatted as event study')

        es_frame = BipartiteEventStudy(data_es, collapsed=True)
        es_frame.connected = self.connected
        es_frame.contiguous_fids = self.contiguous_fids
        es_frame.contiguous_wids = self.contiguous_wids
        es_frame.contiguous_cids = self.contiguous_cids
        es_frame.no_na = self.no_na
        es_frame.no_duplicates = self.no_duplicates

        return es_frame

    def approx_cdfs(self, cdf_resolution=10, grouping='quantile_all', stayers_movers=None):
        '''
        Generate cdfs of compensation for firms.

        Arguments:
            cdf_resolution (int): how many values to use to approximate the cdf
            grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)
            stayers_movers (str or None): if None, uses entire dataset; if 'stayers', uses only stayers; if 'movers', uses only movers

        Returns:
            cdf_df (NumPy Array): NumPy array of firm cdfs
            n_firms (int): number of firms in subset of data used to cluster
        '''
        # Determine whether m column exists
        m = self.col_dict['m'] is not None

        if not m:
            self.gen_m()

        if stayers_movers == 'stayers':
            data = self[self['m'] == 0]
        elif stayers_movers == 'movers':
            data = self[self['m'] == 1]
        else:
            data = self

        # Create empty numpy array to fill with the cdfs
        # n_firms = self.n_firms()
        n_firms = len(data['fid'].unique()) # Can't use self.n_firms() since data could be a subset of self.data
        cdfs = np.zeros([n_firms, cdf_resolution])

        # Create quantiles of interest
        quantiles = np.linspace(1 / cdf_resolution, 1, cdf_resolution)

        if grouping == 'quantile_all':
            # Get quantiles from all data
            quantile_groups = data['comp'].quantile(quantiles)

            # Generate firm-level cdfs
            for i, quant in enumerate(quantile_groups):
                cdfs[:, i] = data.assign(firm_quant=lambda d: d['comp'] <= quant).groupby('fid')['firm_quant'].agg(sum).to_numpy()

            # Normalize by firm size (convert to cdf)
            fsize = data.groupby('fid').size().to_numpy()
            cdfs /= np.expand_dims(fsize, 1)

        elif grouping in ['quantile_firm_small', 'quantile_firm_large']:
            # Sort data by compensation (do this once now, so that don't need to do it again later) (also note it is faster to sort then manually compute quantiles than to use built-in quantile functions)
            data = data.sort_values(['comp'])

            if grouping == 'quantile_firm_small':
                # Convert pandas dataframe into a dictionary to access data faster
                # Source for idea: https://stackoverflow.com/questions/57208997/looking-for-the-fastest-way-to-slice-a-row-in-a-huge-pandas-dataframe
                # Source for how to actually format data correctly: https://stackoverflow.com/questions/56064677/pandas-series-to-dict-with-repeated-indices-make-dict-with-list-values
                data_dict = data['comp'].groupby(level=0).agg(list).to_dict()

            # Generate the cdfs
            for fid in tqdm(range(n_firms)):
                # Get the firm-level compensation data (don't need to sort because already sorted)
                if grouping == 'quantile_firm_small':
                    comp = data_dict[fid]
                elif grouping == 'quantile_firm_large':
                    comp = data.loc[data['fid'] == fid, 'comp']
                # Generate the firm-level cdf
                # Note: update numpy array element by element
                # Source: https://stackoverflow.com/questions/30012362/faster-way-to-convert-list-of-objects-to-numpy-array/30012403
                for i in range(cdf_resolution):
                    index = max(len(comp) * (i + 1) // cdf_resolution - 1, 0) # Don't want negative index
                    # Update cdfs with the firm-level cdf
                    cdfs[fid, i] = comp[index]

        return cdfs, n_firms

    def cluster(self, user_cluster={}, inplace=True):
        '''
        Cluster data and assign a new column giving the cluster for each firm.

        Arguments:
            user_cluster (dict): dictionary of parameters for clustering

                Dictionary parameters:

                    cdf_resolution (int): how many values to use to approximate the cdf

                    grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)

                    stayers_movers (str or None): if None, uses entire dataset; if 'stayers', uses only stayers; if 'movers', uses only movers

                    dropna (bool): if True, drop observations where firms aren't clustered; if False, keep all observations

                    user_KMeans (dict): use parameters defined in KMeans_dict for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and use default parameters defined in class attribute default_KMeans for any parameters not specified

                inplace (bool): if True, modify in-place
        Returns:
            frame (BipartitePandas): BipartitePandas with clusters
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        # Warn about selecting a year with event study data
        if user_cluster['year'] is not None:
            warnings.warn('Can only cluster on a particular year with long form data, but data is currently event study form.')

        # Update dictionary
        cluster_params = update_dict(frame.default_cluster, user_cluster)

        # Unpack dictionary
        cdf_resolution = cluster_params['cdf_resolution']
        grouping = cluster_params['grouping']
        stayers_movers = cluster_params['stayers_movers']
        user_KMeans = cluster_params['user_KMeans']

        # Compute cdfs
        cdfs, n_firms = frame.approx_cdfs(cdf_resolution=cdf_resolution, grouping=grouping, stayers_movers=stayers_movers)
        frame.logger.info('firm cdfs computed')

        # Compute firm clusters
        KMeans_params = update_dict(frame.default_KMeans, user_KMeans)
        clusters = KMeans(**KMeans_params).fit(cdfs).labels_
        frame.logger.info('firm clusters computed')

        # Create Pandas dataframe linking fid to firm cluster
        fids = np.arange(n_firms)
        clusters_dict = {'fid': fids, 'j': clusters}
        clusters_df = pd.DataFrame(clusters_dict, index=fids)
        frame.logger.info('dataframe linking fids to clusters generated')

        # Merge into event study data
        frame = frame.merge(clusters_df, how='left', on='fid')
        # Keep column as int even with nans
        frame['j'] = frame['j'].astype('Int64')
        frame.col_dict['j'] = 'j'

        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        if cluster_params['dropna']:
            # Drop firms that don't get clustered
            frame = frame.dropna().reset_index(drop=True)
            frame['j'] = frame['j'].astype(int)
            frame.clean_data()

        frame.logger.info('clusters merged into event study data')

        return frame

class BipartiteEventStudy(DataFrame):
    '''
    Class of BipartiteEventStudy, where BipartiteEventStudy gives a bipartite network of firms and workers in long form.

    Arguments:
        *args: arguments for Pandas DataFrame
        collapsed (bool): if True, data collapsed by employment spells
        col_dict (dict): make data columns readable (requires: wid (worker id), y1 (compensation 1), y2 (compensation 2), f1i (firm id 1), f2i (firm id 2), m (0 if stayer, 1 if mover); optionally include: year_end_1 (last year of observation 1), year_end_2 (last year of observation 2), w1 (weight 1), w2 (weight 2)). Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, collapsed, col_dict=None, **kwargs):
        # Initialize DataFrame
        super().__init__(*args, **kwargs)

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
        self.connected = False # If True, all firms are connected by movers
        self.contiguous_fids = False # If True, firm ids are contiguous
        self.contiguous_wids = False # If True, worker ids are contiguous
        self.contiguous_cids = None # If True, cluster ids are contiguous; if None, data not clustered
        self.no_na = False # If True, no NaN observations in the data
        self.no_duplicates = False # If True, no duplicate rows in the data
        self.collapsed = collapsed

        # Define default parameter dictionaries
        default_col_dict = {
            'wid': 'wid',
            'f1i': 'f1i',
            'f2i': 'f2i',
            'y1': 'y1',
            'y2': 'y2',
            'w1': None,
            'w2': None,
            'm': None,
            'j1': None,
            'j2': None
        }
        if collapsed:
            default_col_dict['year_start_1'] = 'year_start_1'
            default_col_dict['year_start_2'] = 'year_start_2'
            default_col_dict['year_end_1'] = 'year_end_1'
            default_col_dict['year_end_2'] = 'year_end_2'
            
        else:
            default_col_dict['year_1'] = 'year_1'
            default_col_dict['year_2'] = 'year_2'

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
            'stayers_movers': None,
            'year': None,
            'dropna': False,
            'user_KMeans': self.default_KMeans
        }

        # Create self.col_dict
        optional_cols = [['w1', 'w2'], ['m'], ['j1', 'j2']]
        self.col_dict = _col_dict_optional_cols(default_col_dict, col_dict, self.columns, optional_cols=optional_cols)

        if self.col_dict['j1'] is not None and self.col_dict['j2'] is not None:
            self.contiguous_cids = False

        self.logger.info('BipartiteEventStudy object initialized')

    def copy(self):
        '''
        Copy the current instance of BipartiteEventStudy.

        Returns:
            bd_copy (BipartiteEventStudy): copy of the current instance of BipartiteEventStudy.
        '''
        bd_copy = BipartiteEventStudy(DataFrame.copy(self), col_dict=self.col_dict)
        bd_copy.connected = self.connected
        bd_copy.contiguous_fids = self.contiguous_fids
        bd_copy.contiguous_wids = self.contiguous_wids
        bd_copy.contiguous_cids = self.contiguous_cids
        bd_copy.no_na = self.no_na
        bd_copy.no_duplicates = self.no_duplicates

        return bd_copy

    def n_workers(self):
        '''
        Get the number of unique workers.

        Returns:
            (int): number of unique workers
        '''
        return len(self['wid'].unique())

    def n_firms(self):
        '''
        Get the number of unique firms.

        Returns:
            (int): number of unique firms
        '''
        return len(set(list(self['f1i'].unique()) + list(self['f2i'].unique())))

    def n_clusters(self):
        '''
        Get the number of unique clusters.

        Returns:
            (int): number of unique clusters
        '''
        if self.col_dict['j1'] is not None and self.col_dict['j2'] is not None:
            return len(set(list(self['j1'].unique()) + list(self['j2'].unique())))
        return None

    def drop(self, indices, axis, inplace=True):
        '''
        Drop indices along axis. To drop weights or cluster ids, specify 'weight' or 'j', respectively.

        Arguments:
            indices (int or str, optionally as a list): row(s) or column(s) to drop
            axis (int): 0 to drop rows, 1 to drop columns
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteEventStudy): BipartiteEventStudy with dropped indices
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        if axis == 1:
            for col in _to_list(indices):
                if col in frame.columns or (col == 'weight' and frame.col_dict['w1'] is not None and frame.col_dict['w2'] is not None) or (col == 'j' and frame.col_dict['j1'] is not None and frame.col_dict['j2'] is not None):
                    if col == 'm':
                        DataFrame.drop(frame, col, axis=1, inplace=True)
                        frame.col_dict[col] = None
                    elif col == 'weight':
                        DataFrame.drop(frame, ['w1', 'w2'], axis=1, inplace=True)
                        frame.col_dict['w1'] = None
                        frame.col_dict['w2'] = None
                    elif col == 'j':
                        DataFrame.drop(frame, ['j1', 'j2'], axis=1, inplace=True)
                        frame.col_dict['j1'] = None
                        frame.col_dict['j2'] = None
                        frame.contiguous_cids = None
                    else:
                        warnings.warn("{} is either (a) an essential column and cannot be dropped or (b) you specified 'w1'/'w2' instead of 'weight' or 'j1'/'j2' instead of 'j'")
                else:
                    warnings.warn('{} is not in data columns')
        elif axis == 0:
            DataFrame.drop(frame, indices, axis=0, inplace=True)
            frame.connected = False
            frame.contiguous_fids = False
            frame.contiguous_wids = False
            if frame.contiguous_cids is not None:
                frame.contiguous_cids = False
            frame.no_na = False
            frame.no_duplicates = False
            frame.clean_data()

        return frame

    def rename(self, rename_dict, inplace=True):
        '''
        Rename a column.

        Arguments:
            rename_dict (dict): key is current column name, value is new column name
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteEventStudy): BipartiteEventStudy with renamed columns
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        for col_cur, col_new in rename_dict.items():
            if col_cur in frame.columns or (col_cur == 'weight' and frame.col_dict['w1'] is not None and frame.col_dict['w2'] is not None) or (col_cur == 'j' and frame.col_dict['j1'] is not None and frame.col_dict['j2'] is not None):
                if col_cur == 'm':
                    frame.rename({col_cur: col_new}, axis=1, inplace=True)
                    frame.col_dict[col_cur] = None
                elif col_cur == 'weight':
                    frame.rename({'w1': col_new + '1', 'w2': col_new + '2'}, axis=1, inplace=True)
                    frame.col_dict['w1'] = None
                    frame.col_dict['w2'] = None
                elif col_cur == 'j':
                    frame.rename({'j1': col_new + '1', 'j2': col_new + '2'}, axis=1, inplace=True)
                    frame.col_dict['j1'] = None
                    frame.col_dict['j2'] = None
                    frame.contiguous_cids = None
                else:
                    warnings.warn("{} is either (a) an essential column and cannot be dropped or (b) you specified 'w1'/'w2' instead of 'weight' or 'j1'/'j2' instead of 'j'")
            else:
                warnings.warn('{} is not in data columns')

    def clean_data(self, inplace=True):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteEventStudy): BipartiteEventStudy with cleaned data
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        frame.logger.info('beginning data cleaning')
        frame.logger.info('checking quality of data')
        # Make sure data is valid - computes no_na, no_duplicates, connected, and contiguous, along with other checks (note that column names are corrected in data_validity() if all columns are in the data)
        frame.data_validity()

        # Next, drop NaN observations
        if not frame.no_na:
            frame.logger.info('dropping NaN observations')
            frame.dropna(inplace=True)

            # Update no_na
            frame.no_na = True

        # Next, drop duplicate observations
        if not frame.no_duplicates:
            frame.logger.info('dropping duplicate observations')
            frame.drop_duplicates(inplace=True)

            # Update no_duplicates
            frame.no_duplicates = True

        # Next, find largest set of firms connected by movers
        if not frame.connected:
            # Generate largest connected set
            frame.logger.info('generating largest connected set')
            frame.conset()

        # Next, make firm ids contiguous
        if not frame.contiguous_fids:
            frame.logger.info('making firm ids contiguous')
            frame.contiguous_ids('fid')

        # Next, make worker ids contiguous
        if not frame.contiguous_wids:
            frame.logger.info('making firm ids contiguous')
            frame.contiguous_ids('wid')

        # Next, make cluster ids contiguous
        if frame.contiguous_cids is not None and not frame.contiguous_cids:
            frame.logger.info('making cluster ids contiguous')
            frame.contiguous_ids('j')

        # Using contiguous fids, get NetworkX Graph of largest connected set (note that this must be done even if firms already connected and contiguous)
        frame.logger.info('generating NetworkX Graph of largest connected set')
        # frame.G = frame.conset() # FIXME currently not used

        frame.logger.info('data cleaning complete')

        return frame

    def data_validity(self, inplace=True):
        '''
        Checks that data is formatted correctly and updates relevant attributes.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteEventStudy): BipartiteEventStudy with corrected columns and attributes
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        success_stayers = True
        success_movers = True

        all_cols = ['wid', 'y1', 'y2', 'f1i', 'f2i']

        # Determine whether weight, m, cluster columns exist
        weighted = frame.col_dict['w1'] is not None and frame.col_dict['w2'] is not None
        m = frame.col_dict['m'] is not None
        clustered = frame.col_dict['j1'] is not None and frame.col_dict['j2'] is not None

        if 'year_1' in frame.col_dict.keys(): # From long
            all_cols += ['year_1', 'year_2']
        else: # From collapsed long
            all_cols += ['year_start_1', 'year_start_2', 'year_end_1', 'year_end_2']

        if weighted:
            all_cols += ['w1', 'w2']
        if m:
            all_cols += ['m']
        if clustered:
            all_cols += ['j1', 'j2']

        frame.logger.info('--- checking columns ---')
        cols = True
        for col in all_cols:
            if frame.col_dict[col] not in frame.columns:
                frame.logger.info(col, 'column missing from data')
                cols = False
            else:
                if col in ['y1', 'y2', 'w1', 'w2']:
                    if frame[frame.col_dict[col]].dtype not in ['float', 'float16', 'float32', 'float64', 'float128', 'int', 'int16', 'int32', 'int64', 'Int64']:
                        frame.logger.info(col, 'column has wrong dtype, should be float or int but is', frame[frame.col_dict[col]].dtype)
                        cols = False
                elif col in ['year_1', 'year_2', 'year_start_1', 'year_start_2', 'year_end_1', 'year_end_2', 'm', 'j1', 'j2']:
                    if frame[frame.col_dict[col]].dtype not in ['int', 'int16', 'int32', 'int64', 'Int64']:
                        frame.logger.info(col, 'column has wrong dtype, should be int but is', frame[frame.col_dict[col]].dtype)
                        cols = False

        frame.logger.info('columns correct:' + str(cols))
        if not cols:
            success_stayers = False
            success_movers = False
            raise ValueError('Your data does not include the correct columns. The TwoWay object cannot be generated with your data.')
        else:
            # Correct column names
            frame.logger.info('correcting column names')
            _update_cols(frame)

        stayers = frame[frame['m'] == 0]
        movers = frame[frame['m'] == 1]

        frame.logger.info('--- checking nan data ---')
        na_stayers = stayers.shape[0] - stayers.dropna().shape[0]
        na_movers = movers.shape[0] - movers.dropna().shape[0]

        frame.logger.info('stayers nans (should be 0):' + str(na_stayers))
        frame.logger.info('movers nans (should be 0):' + str(na_movers))
        if na_stayers > 0:
            frame.no_na = False
            success_stayers = False
        if na_movers > 0:
            frame.no_na = False
            success_movers = False
        if (na_stayers == 0) and (na_movers == 0):
            frame.no_na = True

        frame.logger.info('--- checking duplicates ---')
        duplicates_stayers = stayers.shape[0] - stayers.drop_duplicates().shape[0]
        duplicates_movers = movers.shape[0] - movers.drop_duplicates().shape[0]

        frame.logger.info('stayers duplicates (should be 0):' + str(duplicates_stayers))
        frame.logger.info('movers duplicates (should be 0):' + str(duplicates_movers))
        if duplicates_stayers > 0:
            frame.no_duplicates = False
            success_stayers = False
        if duplicates_movers > 0:
            frame.no_duplicates = False
            success_movers = False
        if (duplicates_stayers == 0) and (duplicates_movers == 0):
            frame.no_duplicates = True

        frame.logger.info('--- checking firms ---')
        firms_stayers = (stayers['f1i'] != stayers['f2i']).sum()
        firms_movers = (movers['f1i'] == movers['f2i']).sum()

        frame.logger.info('stayers with different firms (should be 0):' + str(firms_stayers))
        frame.logger.info('movers with same firm (should be 0):' + str(firms_movers))
        if firms_stayers > 0:
            success_stayers = False
        if firms_movers > 0:
            success_movers = False

        frame.logger.info('--- checking income ---')
        income_stayers = (stayers['y1'] != stayers['y2']).sum()

        frame.logger.info('stayers with different income (should be 0):' + str(income_stayers))
        if income_stayers > 0:
            success_stayers = False

        frame.logger.info('--- checking connected set ---')
        G = nx.from_pandas_edgelist(movers, 'f1i', 'f2i')
        largest_cc = max(nx.connected_components(G), key=len)

        cc_stayers = stayers[(~stayers['f1i'].isin(largest_cc)) | (~stayers['f2i'].isin(largest_cc))].shape[0]
        cc_movers = movers[(~movers['f1i'].isin(largest_cc)) | (~movers['f2i'].isin(largest_cc))].shape[0]

        frame.logger.info('stayers outside connected set (should be 0):' + str(cc_stayers))
        frame.logger.info('movers outside connected set (should be 0):' + str(cc_movers))
        if cc_stayers > 0:
            frame.connected = False
            success_stayers = False
        if cc_movers > 0:
            frame.connected = False
            success_movers = False
        if (cc_stayers == 0) and (cc_movers == 0):
            frame.connected = True

        frame.logger.info('--- checking contiguous firm ids ---')
        fid_max = max(frame['f1i'].max(), frame['f2i'].max())
        n_firms = frame.n_firms()

        contig_fids = (fid_max == n_firms - 1)
        frame.contiguous_fids = contig_fids

        frame.logger.info('contiguous firm ids (should be True):' + str(contig_fids))
        if not contig_fids:
            success_stayers = False
            success_movers = False

        frame.logger.info('--- checking contiguous worker ids ---')
        wid_max = max(frame['wid'].max())
        n_workers = frame.n_workers()

        contig_wids = (wid_max == n_workers - 1)
        frame.contiguous_wids = contig_wids

        frame.logger.info('contiguous worker ids (should be True):' + str(contig_wids))
        if not contig_wids:
            success_stayers = False
            success_movers = False

        if clustered:
            frame.logger.info('--- checking contiguous cluster ids ---')
            cid_max = max(frame['j1'].max(), frame['j2'].max())
            n_cids = len(set(list(frame['j1'].unique()) + list(frame['j2'].unique())))

            contig_cids = (cid_max + 1 == n_cids)
            frame.contiguous_cids = contig_cids

            frame.logger.info('contiguous cluster ids (should be True):' + str(contig_cids))

            if not contig_cids:
                success_stayers = False
                success_movers = False

        frame.logger.info('Overall success for stayers:' + str(success_stayers))
        frame.logger.info('Overall success for movers:' + str(success_movers))

        return frame

    def conset(self, inplace=True):
        '''
        Update data to include only the largest connected set of movers, and if firm ids are contiguous, also return the NetworkX Graph.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            (tuple):
                frame (BipartiteCollapsed): BipartiteCollapsed with connected set of movers
                G (NetworkX Graph): largest connected set of movers (only returns if firm ids are contiguous, otherwise returns None)
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        prev_workers = frame.n_workers()
        prev_firms = frame.n_firms()
        prev_clusters = frame.n_clusters()
        # Find largest connected set
        # Source: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html
        G = nx.from_pandas_edgelist(frame, 'f1i', 'f2i')
        # Update data if not connected
        if not frame.connected:
            largest_cc = max(nx.connected_components(G), key=len)
            # Keep largest connected set of firms
            frame = frame[(frame['f1i'].isin(largest_cc)) & (frame['f2i'].isin(largest_cc))]

        # Data is now connected
        frame.connected = True

        # If connected data != full data, set contiguous to False
        if prev_firms != frame.n_firms():
            frame.contiguous_fids = False
        if prev_workers != frame.n_workers():
            frame.contiguous_wids = False
        if prev_clusters != frame.n_clusters():
            if frame.col_dict['j'] is not None:
                frame.contiguous_cids = False

        # Return G if firm ids are contiguous (if they're not contiguous, they have to be updated first)
        if frame.contiguous_fids and frame.contiguous_wids and (frame.col_dict['j1'] is None or frame.col_dict['j2'] is None or frame.contiguous_cids):
            return G

        return None

    def contiguous_ids(self, id_col, inplace=True):
        '''
        Make column of ids contiguous.

        Arguments:
            id_col (str): column to make contiguous ('fid', 'wid', or 'j')
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteEventStudy): BipartiteEventStudy with contiguous ids
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        # Generate id_list (note that all columns listed in id_list are included in the set of ids, and all columns are adjusted to have the new, contiguous ids)
        if id_col == 'fid':
            id_list = ['f1i', 'f2i']
        elif id_col == 'wid':
            id_list = ['wid']
        elif id_col == 'j':
            id_list = ['j1', 'j2']
        # Create sorted set of unique ids
        ids = []
        for id in id_list:
            ids += list(frame[id].unique())
        ids = sorted(list(set(ids)))

        # Create list of adjusted ids
        adjusted_ids = np.arange(len(ids)).astype(int)

        # Update each fid one at a time
        for id in id_list:
            # Create dictionary linking current to new ids, then convert into a dataframe for merging
            ids_dict = {id: ids, 'adj_' + id: adjusted_ids}
            ids_df = pd.DataFrame(ids_dict, index=adjusted_ids)

            # Merge new, contiguous ids into event study data
            frame = frame.merge(ids_df, how='left', on=id)

            # Drop old id column and rename contiguous id column
            DataFrame.drop(frame, [id], axis=1, inplace=True)
            DataFrame.rename(frame, {'adj_' + id: id}, axis=1, inplace=True)

        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        if id_col == 'fid':
            # Firm ids are now contiguous
            frame.contiguous_fids = True
        elif id_col == 'wid':
            # Worker ids are now contiguous
            frame.contiguous_wids = True
        elif id_col == 'j':
            # Cluster ids are now contiguous
            frame.contiguous_cids = True

        return frame

    def gen_m(self, inplace=True):
        '''
        Generate m column for data (m == 0 if stayer, m == 1 if mover).

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteEventStudy): BipartiteEventStudy with m column
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        frame['m'] = (frame['f1i'] != frame['f2i']).astype(int)
        frame.col_dict['m'] = 'm'
        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

    def get_cs(self):
        '''
        Return event study data reformatted into cross section data.

        Returns:
            data_cs (Pandas DataFrame): cross section data
        '''
        # Determine whether weight, m, cluster columns exist
        weighted = self.col_dict['w1'] is not None and self.col_dict['w2'] is not None
        m = self.col_dict['m'] is not None
        clustered = self.col_dict['j1'] is not None and self.col_dict['j2'] is not None

        if not m:
            self.gen_m()

        sdata = self[self['m'] == 0]
        jdata = self[self['m'] == 1]

        # Assign some values
        ns = len(sdata)
        nm = len(jdata)

        # # Reset index
        # sdata.set_index(np.arange(ns) + 1 + nm)
        # jdata.set_index(np.arange(nm) + 1)

        # Columns used for constructing cross section
        cs_cols = ['wid', 'f1i', 'f2i', 'y1', 'y2']

        if clustered:
            cs_cols += ['j1', 'j2']
        if 'year_1' in self.col_dict.keys():
            cs_cols += ['year_1', 'year_2']
        else:
            cs_cols += ['year_start_1', 'year_start_2', 'year_end_1', 'year_end_2']
        if weighted:
            cs_cols += ['w1', 'w2']
        cs_cols += ['m']

        rename_dict = {
            'f1i': 'f2i',
            'f2i': 'f1i',
            'y1': 'y2',
            'y2': 'y1',
            'year_1': 'year_2',
            'year_2': 'year_1',
            'year_start_1': 'year_start_2',
            'year_start_2': 'year_start_1',
            'year_end_1': 'year_end_2',
            'year_end_2': 'year_end_1',
            'w1': 'w2',
            'w2': 'w1',
            'j1': 'j2',
            'j2': 'j1'
        }

        # Combine the 2 data-sets
        data_cs = pd.concat([
            sdata[cs_cols].assign(cs=1),
            jdata[cs_cols].assign(cs=1),
            jdata[cs_cols].rename(rename_dict, axis=1).assign(cs=0)
        ], ignore_index=True)

        # Sort columns
        sorted_cols = sorted(data_cs.columns, key=col_order)
        data_cs = data_cs[sorted_cols]

        self.logger.info('mover and stayer event study datasets combined into cross section')

        return data_cs

    def get_long(self):
        '''
        Return event study data reformatted into long form.

        Returns:
            long_frame (BipartitePandas): BipartitePandas object generated from event study data
        '''
        if not self.collapsed:
            # Determine whether weight, m, cluster columns exist
            weighted = self.col_dict['w1'] is not None and self.col_dict['w2'] is not None
            m = self.col_dict['m'] is not None
            clustered = self.col_dict['j1'] is not None and self.col_dict['j2'] is not None

            if not m:
                self.gen_m()

            # Columns to drop
            drops = ['f2i', 'y2', 'year_2']

            rename_dict_1 = {
                'f1i': 'f2i',
                'f2i': 'f1i',
                'y1': 'y2',
                'y2': 'y1',
                'year_1': 'year_2',
                'year_2': 'year_1',
                'w1': 'w2',
                'w2': 'w1',
                'j1': 'j2',
                'j2': 'j1'
            }

            rename_dict_2 = {
                'f1i': 'fid',
                'y1': 'comp',
                'year_1': 'year',
                'w1': 'weight',
                'j1': 'j'
            }

            astype_dict = {
                'wid': int,
                'fid': int,
                'year': int,
                'm': int
            }

            if clustered:
                drops += ['j2']
                astype_dict['j'] = int
            if weighted:
                drops += ['w2']
                astype_dict['weight'] = int

            # Append the last row if a mover (this is because the last observation is only given as an f2i, never as an f1i)
            data_long = self.groupby('wid').apply(lambda a: a.append(a.iloc[-1].rename(rename_dict_1, axis=1)) if a.iloc[0]['m'] == 1 else a) \
                .reset_index(drop=True) \
                .drop(drops, axis=1) \
                .rename(rename_dict_2, axis=1) \
                .astype(astype_dict)

            # Sort columns
            sorted_cols = sorted(data_long.columns, key=col_order)
            data_long = data_long[sorted_cols]

            long_frame = BipartitePandas(data_long)
            long_frame.connected = self.connected
            long_frame.contiguous_fids = self.contiguous_fids
            long_frame.contiguous_wids = self.contiguous_wids
            long_frame.contiguous_cids = self.contiguous_cids
            long_frame.no_na = self.no_na
            long_frame.no_duplicates = self.no_duplicates

            return long_frame
        else:
            warnings.warn('Data is formatted as collapsed long, cannot un-collapse data. Try running es_to_collapsed_long() to convert your data into collapsed long form.')

    def get_collapsed_long(self):
        '''
        Return event study data reformatted into collapsed long form.

        Returns:
            collapsedlong_frame (BipartiteCollapsed): BipartiteCollapsed object generated from event study data
        '''
        if self.collapsed:
            # Determine whether weight, m, cluster columns exist
            weighted = self.col_dict['w1'] is not None and self.col_dict['w2'] is not None
            m = self.col_dict['m'] is not None
            clustered = self.col_dict['j1'] is not None and self.col_dict['j2'] is not None

            if not m:
                self.gen_m()

            # Columns to drop
            drops = ['f2i', 'y2', 'year_start_2', 'year_end_2']

            rename_dict_1 = {
                'f1i': 'f2i',
                'f2i': 'f1i',
                'y1': 'y2',
                'y2': 'y1',
                'year_start_1': 'year_start_2',
                'year_start_2': 'year_start_1',
                'year_end_1': 'year_end_2',
                'year_end_2': 'year_end_1',
                'w1': 'w2',
                'w2': 'w1',
                'j1': 'j2',
                'j2': 'j1'
            }

            rename_dict_2 = {
                'f1i': 'fid',
                'y1': 'comp',
                'year_start_1': 'year_start',
                'year_end_1': 'year_end',
                'w1': 'weight',
                'j1': 'j'
            }

            astype_dict = {
                'wid': int,
                'fid': int,
                'year_start': int,
                'year_end': int,
                'm': int
            }

            if clustered:
                drops += ['j2']
                astype_dict['j'] = int
            if weighted:
                drops += ['w2']
                astype_dict['weight'] = int

            # Append the last row if a mover (this is because the last observation is only given as an f2i, never as an f1i)
            data_collapsed_long = self.groupby('wid').apply(lambda a: a.append(a.iloc[-1].rename(rename_dict_1, axis=1)) if a.iloc[0]['m'] == 1 else a) \
                .reset_index(drop=True) \
                .drop(drops, axis=1) \
                .rename(rename_dict_2, axis=1) \
                .astype(astype_dict)

            # Sort columns
            sorted_cols = sorted(data_collapsed_long.columns, key=col_order)
            data_collapsed_long = data_collapsed_long[sorted_cols]

            collapsedlong_frame = BipartiteCollapsed(data_collapsed_long)
            collapsedlong_frame.connected = self.connected
            collapsedlong_frame.contiguous_fids = self.contiguous_fids
            collapsedlong_frame.contiguous_wids = self.contiguous_wids
            collapsedlong_frame.contiguous_cids = self.contiguous_cids
            collapsedlong_frame.no_na = self.no_na
            collapsedlong_frame.no_duplicates = self.no_duplicates

            return collapsedlong_frame
        else:
            warnings.warn('Event study data is not collapsed, cannot refactor into collapsed long form. Try running es_to_long(), then long_to_collapsed_long() to convert your data into collapsed long form.')

    def approx_cdfs(self, cdf_resolution=10, grouping='quantile_all', stayers_movers=None):
        '''
        Generate cdfs of compensation for firms.

        Arguments:
            cdf_resolution (int): how many values to use to approximate the cdf
            grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)
            stayers_movers (str or None): if None, uses entire dataset; if 'stayers', uses only stayers; if 'movers', uses only movers

        Returns:
            cdf_df (NumPy Array): NumPy array of firm cdfs
            n_firms (int): number of firms in subset of data used to cluster
        '''
        # Determine whether m column exists
        m = self.col_dict['m'] is not None

        if not m:
            self.gen_m()

        if stayers_movers == 'stayers':
            data = self[self['m'] == 0]
        elif stayers_movers == 'movers':
            data = self[self['m'] == 1]
        else:
            data = self

        # Create empty numpy array to fill with the cdfs
        # n_firms = self.n_firms()
        n_firms = len(set(list(data['f1i'].unique()) + list(data['f2i'].unique()))) # Can't use self.n_firms() since data could be a subset of self.data
        cdfs = np.zeros([n_firms, cdf_resolution])

        # Create quantiles of interest
        quantiles = np.linspace(1 / cdf_resolution, 1, cdf_resolution)

        # Re-arrange event study data to be in long format (temporarily)
        data = data.rename({'f1i': 'fid', 'y1': 'comp'}, axis=1)
        data = pd.concat([data, data.rename({'f2i': 'fid', 'y2': 'comp', 'fid': 'f2i', 'comp': 'y2'}, axis=1).assign(f2i = - 1)], axis=0) # Include irrelevant columns and rename f1i to f2i to prevent nans, which convert columns from int into float # FIXME duplicating both movers and stayers, should probably only be duplicating movers

        if grouping == 'quantile_all':
            # Get quantiles from all data
            quantile_groups = data['comp'].quantile(quantiles)

            # Generate firm-level cdfs
            for i, quant in enumerate(quantile_groups):
                cdfs[:, i] = data.assign(firm_quant=lambda d: d['comp'] <= quant).groupby('fid')['firm_quant'].agg(sum).to_numpy()

            # Normalize by firm size (convert to cdf)
            fsize = data.groupby('fid').size().to_numpy()
            cdfs /= np.expand_dims(fsize, 1)

        elif grouping in ['quantile_firm_small', 'quantile_firm_large']:
            # Sort data by compensation (do this once now, so that don't need to do it again later) (also note it is faster to sort then manually compute quantiles than to use built-in quantile functions)
            data = data.sort_values(['comp'])

            if grouping == 'quantile_firm_small':
                # Convert pandas dataframe into a dictionary to access data faster
                # Source for idea: https://stackoverflow.com/questions/57208997/looking-for-the-fastest-way-to-slice-a-row-in-a-huge-pandas-dataframe
                # Source for how to actually format data correctly: https://stackoverflow.com/questions/56064677/pandas-series-to-dict-with-repeated-indices-make-dict-with-list-values
                data_dict = data['comp'].groupby(level=0).agg(list).to_dict()

            # Generate the cdfs
            for fid in tqdm(range(n_firms)):
                # Get the firm-level compensation data (don't need to sort because already sorted)
                if grouping == 'quantile_firm_small':
                    comp = data_dict[fid]
                elif grouping == 'quantile_firm_large':
                    comp = data.loc[data['fid'] == fid, 'comp']
                # Generate the firm-level cdf
                # Note: update numpy array element by element
                # Source: https://stackoverflow.com/questions/30012362/faster-way-to-convert-list-of-objects-to-numpy-array/30012403
                for i in range(cdf_resolution):
                    index = max(len(comp) * (i + 1) // cdf_resolution - 1, 0) # Don't want negative index
                    # Update cdfs with the firm-level cdf
                    cdfs[fid, i] = comp[index]

        # Drop rows that were appended earlier and rename columns
        data = data[data['f2i'] >= 0]
        data = data.rename({'fid': 'f1i', 'comp': 'y1'}, axis=1)

        return cdfs, n_firms

    def cluster(self, user_cluster={}, inplace=True):
        '''
        Cluster data and assign a new column giving the cluster for each firm.

        Arguments:
            user_cluster (dict): dictionary of parameters for clustering

                Dictionary parameters:

                    cdf_resolution (int): how many values to use to approximate the cdf

                    grouping (str): how to group the cdfs ('quantile_all' to get quantiles from entire set of data, then have firm-level values between 0 and 1; 'quantile_firm_small' to get quantiles at the firm-level and have values be compensations if small data; 'quantile_firm_large' to get quantiles at the firm-level and have values be compensations if large data, note that this is up to 50 times slower than 'quantile_firm_small' and should only be used if the dataset is too large to copy into a dictionary)

                    stayers_movers (str or None): if None, uses entire dataset; if 'stayers', uses only stayers; if 'movers', uses only movers

                    dropna (bool): if True, drop observations where firms aren't clustered; if False, keep all observations

                    user_KMeans (dict): use parameters defined in KMeans_dict for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and use default parameters defined in class attribute default_KMeans for any parameters not specified

                inplace (bool): if True, modify in-place
        Returns:
            frame (BipartiteEventStudy): BipartiteEventStudy with clusters
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        # Warn about selecting a year with event study data
        if user_cluster['year'] is not None:
            warnings.warn('Can only cluster on a particular year with long form data, but data is currently event study form.')

        # Update dictionary
        cluster_params = update_dict(frame.default_cluster, user_cluster)

        # Unpack dictionary
        cdf_resolution = cluster_params['cdf_resolution']
        grouping = cluster_params['grouping']
        stayers_movers = cluster_params['stayers_movers']
        user_KMeans = cluster_params['user_KMeans']

        # Compute cdfs
        cdfs, n_firms = frame.approx_cdfs(cdf_resolution=cdf_resolution, grouping=grouping, stayers_movers=stayers_movers)
        frame.logger.info('firm cdfs computed')

        # Compute firm clusters
        KMeans_params = update_dict(frame.default_KMeans, user_KMeans)
        clusters = KMeans(**KMeans_params).fit(cdfs).labels_
        frame.logger.info('firm clusters computed')

        # Create Pandas dataframe linking fid to firm cluster
        fids = np.arange(n_firms)
        clusters_dict_1 = {'f1i': fids, 'j1': clusters}
        clusters_dict_2 = {'f2i': fids, 'j2': clusters}
        clusters_df_1 = pd.DataFrame(clusters_dict_1, index=fids)
        clusters_df_2 = pd.DataFrame(clusters_dict_2, index=fids)
        frame.logger.info('dataframes linking fids to clusters generated')

        # Merge into event study data
        frame = frame.merge(clusters_df_1, how='left', on='f1i')
        frame = frame.merge(clusters_df_2, how='left', on='f2i')
        # Keep column as int even with nans
        frame[['j1', 'j2']] = frame[['j1', 'j2']].astype('Int64')
        frame.col_dict['j1'] = 'j1'
        frame.col_dict['j2'] = 'j2'

        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        if cluster_params['dropna']:
            # Drop firms that don't get clustered
            frame = frame.dropna().reset_index(drop=True)
            frame['j1'] = frame['j1'].astype(int)
            frame['j2'] = frame['j2'].astype(int)
            frame.clean_data()

        frame.logger.info('clusters merged into event study data')
