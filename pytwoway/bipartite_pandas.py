'''
Class for a bipartite network
'''
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from pandas import DataFrame
import networkx as nx
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import connected_components
import warnings
from pytwoway import update_dict, logger_init

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
    for col_list in _to_list(optional_cols):
        include = True
        for col in _to_list(col_list):
            exists_assigned = new_col_dict[col] is not None
            exists_not_assigned = (col in data_cols) and (new_col_dict[col] is None) and (col not in new_col_dict.values()) # Last condition checks whether data has a different column with same name
            if not exists_assigned and not exists_not_assigned:
                include = False
        if include:
            for col in _to_list(col_list):
                if new_col_dict[col] is None:
                    new_col_dict[col] = col
        else: # Reset column names to None if not all essential columns included
            for col in _to_list(col_list):
                new_col_dict[col] = None
    return new_col_dict

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

class BipartiteBase(DataFrame):
    '''
    Base class for BipartitePandas, where BipartitePandas gives a bipartite network of firms and workers. Contains generalized methods. Inherits from DataFrame.

    Arguments:
        *args: arguments for Pandas DataFrame
        columns_req (list): required columns (only put general column names for joint columns, e.g. put 'fid' instead of 'f1i', 'f2i'; then put the joint columns in reference_dict)
        columns_opt (list): optional columns (only put general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'; then put the joint columns in reference_dict)
        reference_dict (dict): clarify which columns are associated with a general column name, e.g. {'wid': 'wid', 'j': ['j1', 'j2']}
        col_dtype_dict (dict): link column to datatype
        col_dict (dict or None): make data columns readable. Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''
    _metadata = ['col_dict', 'reference_dict', 'col_dtype_dict', 'columns_req', 'columns_opt', 'default_KMeans', 'default_cluster', 'dtype_dict', 'connected', 'contiguous_fids', 'contiguous_wids', 'contiguous_cids', 'no_na', 'no_duplicates'] # Attributes, required for Pandas inheritance

    def __init__(self, *args, columns_req=[], columns_opt=[], reference_dict={}, col_dtype_dict={}, col_dict=None, **kwargs):
        # Initialize DataFrame
        super().__init__(*args, **kwargs)

        # Start logger
        logger_init(self)
        self.logger.info('initializing BipartiteBase object')

        if len(args) > 0 and isinstance(args[0], (BipartiteBase, BipartiteLongBase)): # FIXME add all subclasses to this tuple
            self.set_attributes(args[0])
        else:
            self.columns_req = columns_req + ['wid', 'fid', 'comp']
            self.columns_opt = columns_opt + ['m', 'j']
            self.reference_dict = update_dict({'wid': 'wid', 'm': 'm'}, reference_dict)
            self.col_dtype_dict = update_dict({'wid': 'int', 'fid': 'int', 'comp': 'float', 'year': 'int', 'm': 'int', 'j': 'int'}, col_dtype_dict)
            default_col_dict = {}
            for col in _to_list(columns_req):
                for subcol in _to_list(reference_dict[col]):
                    default_col_dict[subcol] = subcol
            for col in _to_list(columns_opt):
                for subcol in _to_list(reference_dict[col]):
                    default_col_dict[subcol] = None

            # Create self.col_dict
            self.col_dict = _col_dict_optional_cols(default_col_dict, col_dict, self.columns, optional_cols=columns_opt)

            # Set attributes
            self.reset_attributes()

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
            'stayers_movers': None,
            'year': None,
            'dropna': False,
            'user_KMeans': self.default_KMeans
        }

        self.dtype_dict = {
            'int': ['int', 'int16', 'int32', 'int64', 'Int64'],
            'float': ['float', 'float16', 'float32', 'float64', 'float128', 'int', 'int16', 'int32', 'int64', 'Int64'],
            'str': 'str'
        }

        self.logger.info('BipartiteBase object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteBase

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
        fid_lst = []
        for fid_col in self.reference_dict['fid']:
            fid_lst += list(self[fid_col].unique())
        return len(set(fid_lst))

    def n_clusters(self):
        '''
        Get the number of unique clusters.

        Returns:
            (int or None): number of unique clusters, None if not clustered
        '''
        if not self.col_included('j'): # If cluster column not in dataframe
            return None
        cid_lst = []
        for j_col in _to_list(self.reference_dict['j']):
            cid_lst += list(self[j_col].unique())
        return len(set(cid_lst))

    def set_attributes(self, frame, deep=False, no_dict=False):
        '''
        Set class attributes to equal those of another BipartitePandas object.

        Arguments:
            frame (BipartitePandas): BipartitePandas object whose attributes to use
            deep (bool): if True, also copy dictionaries
            no_dict (bool): if True, only set booleans, no dictionaries
        '''
        # Dictionaries
        if not no_dict:
            if deep:
                self.columns_req = frame.columns_req.copy()
                self.columns_opt = frame.columns_opt.copy()
                self.reference_dict = frame.reference_dict.copy()
                self.col_dtype_dict = frame.col_dtype_dict.copy()
                self.col_dict = frame.col_dict.copy()
            else:
                self.columns_req = frame.columns_req
                self.columns_opt = frame.columns_opt
                self.reference_dict = frame.reference_dict
                self.col_dtype_dict = frame.col_dtype_dict
                self.col_dict = frame.col_dict
        # Booleans
        self.connected = frame.connected # If True, all firms are connected by movers
        self.contiguous_fids = frame.contiguous_fids # If True, firm ids are contiguous
        self.contiguous_wids = frame.contiguous_wids # If True, worker ids are contiguous
        self.contiguous_cids = frame.contiguous_cids # If True, cluster ids are contiguous; if None, data not clustered (set to False later in __init__ if clusters included)
        self.correct_cols = frame.correct_cols # If True, column names are correct
        self.no_na = frame.no_na # If True, no NaN observations in the data
        self.no_duplicates = frame.no_duplicates # If True, no duplicate rows in the data

    def reset_attributes(self):
        '''
        Reset class attributes conditions to be False/None.
        '''
        self.connected = False # If True, all firms are connected by movers
        self.contiguous_fids = False # If True, firm ids are contiguous
        self.contiguous_wids = False # If True, worker ids are contiguous
        self.contiguous_cids = None # If True, cluster ids are contiguous; if None, data not clustered (set to False later in __init__ if clusters included)
        self.correct_cols = False # If True, column names are correct
        self.no_na = False # If True, no NaN observations in the data
        self.no_duplicates = False # If True, no duplicate rows in the data

        # Verify whether clusters included
        if self.col_included('j'):
            self.contiguous_cids = False

    def col_included(self, col):
        '''
        Check whether a column from the pre-established required/optional lists is included.

        Arguments:
            col (str): column to check. Use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'

        Returns:
            (bool): if True, column is included
        '''
        if col in self.columns_req + self.columns_opt:
            for subcol in _to_list(self.reference_dict[col]):
                if self.col_dict[subcol] is None:
                    return False
            return True
        return False

    def included_cols(self):
        '''
        Get all columns included from the pre-established required/optional lists. Uses general column names for joint columns, e.g. returns 'j' instead of 'j1', 'j2'.

        Returns:
            all_cols (list): included columns
        '''
        all_cols = []
        for col in self.columns_req + self.columns_opt:
            include = True
            for subcol in _to_list(self.reference_dict[col]):
                if self.col_dict[subcol] is None:
                    include = False
                    break
            if include:
                all_cols.append(col)
        return all_cols

    def drop(self, indices, axis=1, inplace=True):
        '''
        Drop indices along axis.

        Arguments:
            indices (int or str, optionally as a list): row(s) or column(s) to drop. For columns, use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'. Only optional columns may be dropped
            axis (int): 0 to drop rows, 1 to drop columns
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase with dropped indices
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        if axis == 1:
            for col in _to_list(indices):
                if col in frame.columns:
                    if col in self.columns_opt: # If column optional
                        for subcol in _to_list(self.reference_dict[col]):
                            DataFrame.drop(frame, subcol, axis=1, inplace=True)
                            frame.col_dict[subcol] = None
                        if col == 'j':
                            frame.contiguous_cids = None
                    elif col not in self.columns_req and col not in self.columns_opt: # If column is not pre-established
                        DataFrame.drop(frame, col, axis=1, inplace=True)
                    else:
                        warnings.warn('{} is a required column and cannot be dropped')
                else:
                    warnings.warn('{} is not in data columns')
        elif axis == 0:
            DataFrame.drop(frame, indices, axis=0, inplace=True)
            frame.reset_attributes()
            frame.clean_data()

        return frame

    def rename(self, rename_dict, inplace=True):
        '''
        Rename a column.

        Arguments:
            rename_dict (dict): key is current column name, value is new column name. Use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'. Only optional columns may be renamed
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase with renamed columns
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        for col_cur, col_new in rename_dict.items():
            if col_cur in frame.columns:
                if col_cur in self.columns_opt: # If column optional
                    if len(_to_list(self.reference_dict[col_cur])) > 1:
                        for i, subcol in enumerate(_to_list(self.reference_dict[col_cur])):
                            DataFrame.rename(frame, {subcol: col_new + str(i + 1)}, axis=1, inplace=True)
                            frame.col_dict[subcol] = None
                    else:
                        DataFrame.rename(frame, {col_cur: col_new}, axis=1, inplace=True)
                        frame.col_dict[col_cur] = None
                    if col_cur == 'j':
                        frame.contiguous_cids = None
                elif col_cur not in self.columns_req and col_cur not in self.columns_opt: # If column is not pre-established
                        DataFrame.rename(frame, {col_cur: col_new}, axis=1, inplace=True)
                else:
                    warnings.warn('{} is a required column and cannot be renamed')
            else:
                warnings.warn('{} is not in data columns')

        return frame

    def merge(self, *args, **kwargs):
        '''
        Merge two BipartiteBase objects.

        Arguments:
            *args: arguments for Pandas merge
            **kwargs: keyword arguments for Pandas merge

        Returns:
            frame (BipartiteBase): merged dataframe
        '''
        frame = DataFrame.merge(self, *args, **kwargs)
        frame = BipartiteBase(frame) # Use correct constructor
        if kwargs['how'] == 'left':
            frame.set_attributes(self)
        else: # Non-left merge could cause issues with data
            frame.reset_attributes()
        return frame

    def contiguous_ids(self, id_col, inplace=True):
        '''
        Make column of ids contiguous.

        Arguments:
            id_col (str): column to make contiguous ('fid', 'wid', or 'j'). Use general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'. Only optional columns may be renamed
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase with contiguous ids
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        # Create sorted set of unique ids
        ids = []
        for id in _to_list(self.reference_dict[id_col]):
            ids += list(frame[id].unique())
        ids = sorted(list(set(ids)))

        # Create list of adjusted ids
        adjusted_ids = np.arange(len(ids)).astype(int)

        # Update each fid one at a time
        for id in _to_list(self.reference_dict[id_col]):
            # Create dictionary linking current to new ids, then convert into a dataframe for merging
            ids_dict = {id: ids, 'adj_' + id: adjusted_ids}
            ids_df = pd.DataFrame(ids_dict, index=adjusted_ids)

            # Merge new, contiguous ids into event study data
            frame = frame.merge(ids_df, how='left', on=id)

            # Drop old id column and rename contiguous id column
            frame.drop(id, axis=1, inplace=True)
            frame.rename({'adj_' + id: id}, axis=1, inplace=True)

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

    def update_cols(self, inplace=True):
        '''
        Rename columns and keep only relevant columns.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase with updated columns
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

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

    def clean_data(self, inplace=True):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase with cleaned data
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        frame.logger.info('beginning BipartiteBase data cleaning')
        frame.logger.info('checking quality of data')
        # Make sure data is valid - computes correct_cols, no_na, no_duplicates, connected, and contiguous, along with other checks (note that column names are corrected in data_validity() if all columns are in the data)
        frame.data_validity()

        # Next, correct column names
        if not frame.correct_cols:
            frame.logger.info('correcting column names')
            frame.update_cols()

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
        # frame.logger.info('generating NetworkX Graph of largest connected set')
        # frame.G = frame.conset() # FIXME currently not used

        frame.logger.info('BipartiteBase data cleaning complete')

        return frame

    def data_validity(self, inplace=True):
        '''
        Checks that data is formatted correctly and updates relevant attributes.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase with corrected attributes
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        success = True

        frame.logger.info('--- checking columns ---')
        all_cols = self.included_cols()
        frame.logger.info('--- checking column datatypes ---')
        col_dtypes = True
        for col in all_cols:
            for subcol in _to_list(self.reference_dict[col]):
                if frame.col_dict[subcol] not in frame.columns:
                    frame.logger.info('{} missing from data'.format(frame.col_dict[subcol]))
                    col_dtypes = False
                else:
                    col_type = frame[frame.col_dict[subcol]].dtype
                    valid_types = _to_list(frame.dtype_dict[frame.col_dtype_dict[col]])
                    if col_type not in valid_types:
                        frame.logger.info('{} has wrong dtype, should be {} but is {}'.format(frame.col_dict[subcol], frame.col_dtype_dict[col], col_type))
                        col_dtypes = False

        frame.logger.info('column datatypes correct:' + str(col_dtypes))
        if not col_dtypes:
            success = False
            raise ValueError('Your data does not include the correct columns. The BipartitePandas object cannot be generated with your data.')

        frame.logger.info('--- checking column names ---')
        col_names = True
        for col in all_cols:
            for subcol in _to_list(self.reference_dict[col]):
                if frame.col_dict[subcol] != subcol:
                    col_names = False
                    break
        frame.logger.info('column names correct:' + str(col_names))
        if col_names:
            self.correct_cols = True
        if not col_names:
            self.correct_cols = False
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
        if self.reference_dict['fid'] == 'fid':
            frame['fid_max'] = frame.groupby(['wid'])['fid'].transform(max)
            G = nx.from_pandas_edgelist(frame, 'fid', 'fid_max')
            # Drop fid_max
            frame.drop('fid_max', axis=1)
            largest_cc = max(nx.connected_components(G), key=len)
            outside_cc = frame[(~frame['fid'].isin(largest_cc))].shape[0]
        else:
            G = nx.from_pandas_edgelist(frame, 'f1i', 'f2i')
            largest_cc = max(nx.connected_components(G), key=len)
            outside_cc = frame[(~frame['f1i'].isin(largest_cc)) | (~frame['f2i'].isin(largest_cc))].shape[0]

        frame.logger.info('observations outside connected set (should be 0):' + str(outside_cc))
        if outside_cc > 0:
            frame.connected = False
            success = False
        else:
            frame.connected = True

        frame.logger.info('--- checking contiguous firm ids ---')
        fid_max = - np.inf
        for fid_col in _to_list(self.reference_dict['fid']):
            fid_max = max(frame[fid_col].max(), fid_max)
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

        if self.col_included('j'):
            frame.logger.info('--- checking contiguous cluster ids ---')
            cid_max = - np.inf
            for cid_col in _to_list(self.reference_dict['j']):
                cid_max = max(frame[cid_col].max(), cid_max)
            n_cids = frame.n_clusters()

            contig_cids = (cid_max == n_cids - 1)
            frame.contiguous_cids = contig_cids

            frame.logger.info('contiguous cluster ids (should be True):' + str(contig_cids))
            if not contig_cids:
                success = False

        frame.logger.info('BipartiteBase success:' + str(success))

        return frame

    def conset(self, return_G=False, inplace=True):
        '''
        Update data to include only the largest connected set of movers, and if firm ids are contiguous, also return the NetworkX Graph.

        Arguments:
            return_G (bool): if True, return a tuple of (frame, G)
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase with connected set of movers
            ALTERNATIVELY
            (tuple):
                frame (BipartiteBase): BipartiteBase with connected set of movers
                G (NetworkX Graph): largest connected set of movers (only returns if firm ids are contiguous, otherwise returns None)
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        prev_workers = frame.n_workers()
        prev_firms = frame.n_firms()
        prev_clusters = frame.n_clusters()
        if self.reference_dict['fid'] == 'fid':
            # Add max firm id per worker to serve as a central node for the worker
            # frame['fid_f1'] = frame.groupby('wid')['fid'].transform(lambda a: a.shift(-1)) # FIXME - this is directed but is much slower
            frame['fid_max'] = frame.groupby(['wid'])['fid'].transform(max) # FIXME - this is undirected but is much faster

            # Find largest connected set
            # Source: https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html
            G = nx.from_pandas_edgelist(frame, 'fid', 'fid_max')
            # Drop fid_max
            frame.drop('fid_max', axis=1)
        else:
            G = nx.from_pandas_edgelist(frame, 'f1i', 'f2i')
        # Update data if not connected
        if not frame.connected:
            largest_cc = max(nx.connected_components(G), key=len)
            # Keep largest connected set of firms
            if self.reference_dict['fid'] == 'fid':
                frame = frame[frame['fid'].isin(largest_cc)]
            else:
                frame = frame[(frame['f1i'].isin(largest_cc)) & (frame['f2i'].isin(largest_cc))]

        # Data is now connected
        frame.connected = True

        # If connected data != full data, set contiguous to False
        if prev_firms != frame.n_firms():
            frame.contiguous_fids = False
        if prev_workers != frame.n_workers():
            frame.contiguous_wids = False
        if prev_clusters is not None and prev_clusters != frame.n_clusters():
            frame.contiguous_cids = False

        if return_G:
            # Return G if all ids are contiguous (if they're not contiguous, they have to be updated first)
            if frame.contiguous_fids and frame.contiguous_wids and (frame.col_dict['j'] is None or frame.contiguous_cids):
                return frame, G
            return frame, None
        return frame

    def gen_m(self, inplace=True):
        '''
        Generate m column for data (m == 0 if stayer, m == 1 if mover).

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteBase): BipartiteBase with m column
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        if not self.col_included('m'):
            if self.reference_dict['fid'] == 'fid':
                frame['m'] = frame.groupby('wid')['fid'].transform(lambda x: len(np.unique(x)) > 1).astype(int)
            else:
                frame['m'] = (frame['f1i'] != frame['f2i']).astype(int)
            frame.col_dict['m'] = 'm'
            # Sort columns
            sorted_cols = sorted(frame.columns, key=col_order)
            frame = frame[sorted_cols]

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

        if stayers_movers is not None:
            # Determine whether m column exists
            if not self.col_included('m'):
                self.gen_m()
            if stayers_movers == 'stayers':
                data = pd.DataFrame(self[self['m'] == 0])
            elif stayers_movers == 'movers':
                data = pd.DataFrame(self[self['m'] == 1])
        else:
            data = pd.DataFrame(self)

        # If year-level, then only use data for that particular year
        if year is not None:
            if self.reference_dict['year'] == 'year':
                data = data[data['year'] == year]
            else:
                warnings.warn('Cannot use data from a particular year on non-BipartiteLong data. Convert into BipartiteLong to cluster only on a particular year')

        # Create empty numpy array to fill with the cdfs
        if self.reference_dict['fid'] == 'fid': # If Long
            n_firms = len(data['fid'].unique()) # Can't use self.n_firms() since data could be a subset of self.data
        else: # If Event Study
            n_firms = len(set(list(data['f1i'].unique()) + list(data['f2i'].unique()))) # Can't use self.n_firms() since data could be a subset of self.data
            data = data.rename({'f1i': 'fid', 'y1': 'comp'}, axis=1)
            data = pd.concat([data, data.rename({'f2i': 'fid', 'y2': 'comp', 'fid': 'f2i', 'comp': 'y2'}, axis=1).assign(f2i = - 1)], axis=0) # Include irrelevant columns and rename f1i to f2i to prevent nans, which convert columns from int into float # FIXME duplicating both movers and stayers, should probably only be duplicating movers
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

        if self.reference_dict['fid'] == ['f1i', 'f2i']: # Unstack Event Study
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

                    year (int or None): if None, uses entire dataset; if int, gives year of data to consider. Works only if data formatted as BipartiteLong

                    dropna (bool): if True, drop observations where firms aren't clustered; if False, keep all observations

                    user_KMeans (dict): use parameters defined in KMeans_dict for KMeans estimation (for more information on what parameters can be used, visit https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html), and use default parameters defined in class attribute default_KMeans for any parameters not specified

                inplace (bool): if True, modify in-place
        Returns:
            frame (BipartiteLong): BipartiteLong with clusters
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
        for i, fid_col in enumerate(_to_list(self.reference_dict['fid'])):
            if self.reference_dict['fid'] == 'fid':
                j_col = 'j'
            else:
                j_col = 'j' + str(i)
            clusters_dict = {fid_col: fids, j_col: clusters}
            clusters_df = pd.DataFrame(clusters_dict, index=fids)
            frame.logger.info('dataframe linking fids to clusters generated')

            # Merge into event study data
            frame = frame.merge(clusters_df, how='left', on='fid')
            # Keep column as int even with nans
            frame[j_col] = frame[j_col].astype('Int64')
            frame.col_dict[j_col] = j_col

        # Sort columns
        sorted_cols = sorted(frame.columns, key=col_order)
        frame = frame[sorted_cols]

        if cluster_params['dropna']:
            # Drop firms that don't get clustered
            frame = frame.dropna().reset_index(drop=True)
            frame[self.reference_dict['j']] = frame[self.reference_dict['j']].astype(int)
            frame.clean_data()

        frame.logger.info('clusters merged into event study data')

        return frame

class BipartiteLongBase(BipartiteBase):
    '''
    Base class for BipartiteLong and BipartiteLongCollapsed, where BipartiteLong and BipartiteLongCollapsed give a bipartite network of firms and workers in long and collapsed long form, respectively. Contains generalized methods. Inherits from BipartiteBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        columns_req (list): required columns (only put general column names for joint columns, e.g. put 'fid' instead of 'f1i', 'f2i'; then put the joint columns in reference_dict)
        columns_opt (list): optional columns (only put general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'; then put the joint columns in reference_dict)
        reference_dict (dict): clarify which columns are associated with a general column name, e.g. {'wid': 'wid', 'j': ['j1', 'j2']}
        col_dtype_dict (dict): link column to datatype
        col_dict (dict or None): make data columns readable. Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, columns_req=[], columns_opt=[], reference_dict={}, col_dtype_dict={}, col_dict=None, **kwargs):
        columns_req += ['year']
        reference_dict = update_dict({'fid': 'fid', 'comp': 'comp', 'j': 'j'}, reference_dict)
        # Initialize DataFrame
        super().__init__(*args, columns_req=columns_req, columns_opt=columns_opt, reference_dict=reference_dict, col_dtype_dict=col_dtype_dict, col_dict=col_dict, **kwargs)

        self.logger.info('BipartiteLongBase object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteLongBase

class BipartiteLong(BipartiteLongBase):
    '''
    Class for bipartite networks of firms and workers in long form. Inherits from BipartiteLongBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        col_dict (dict or None): make data columns readable (requires: wid (worker id), comp (compensation), fid (firm id), year). Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, **kwargs):
        # Initialize DataFrame
        reference_dict = {'year': 'year'}
        super().__init__(*args, reference_dict=reference_dict, col_dict=col_dict, **kwargs)
        self.logger.info('BipartiteLong object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteLong

    def clean_data(self, inplace=True):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteLong): BipartiteLong with cleaned data
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        frame.logger.info('beginning BipartiteLong data cleaning')
        frame.logger.info('checking quality of data')
        frame.data_validity()

        frame.logger.info('BipartiteLong data cleaning complete')

        BipartiteBase.clean_data(self)

        return frame

    def data_validity(self, inplace=True):
        '''
        Checks that data is formatted correctly and updates relevant attributes.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteLong): BipartiteLong with corrected columns and attributes
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        success = True

        frame.logger.info('--- checking worker-year observations ---')
        max_obs = frame.groupby(['wid', 'year']).size().max()

        frame.logger.info('max number of worker-year observations (should be 1):' + str(max_obs))
        if max_obs > 1:
            success = False

        frame.logger.info('BipartiteLongBase success:' + str(success))

        return frame

    def get_collapsed_long(self):
        '''
        Collapse long data by job spells (so each spell for a particular worker at a particular firm is one observation).

        Returns:
            collapsed_frame (BipartiteLongCollapsed): BipartiteLongCollapsed object generated from long data collapsed by job spells
        '''
        # Copy data
        data = pd.DataFrame(self, copy=True)
        # Sort data by wid and year
        data = data.sort_values(['wid', 'year'])
        self.logger.info('copied data sorted by wid and year')
        # Determine whether m, cluster columns exist
        m = self.col_included('m')
        clustered = self.col_included('j')

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
        collapsed_data = data_spell.reset_index(drop=True)

        # Sort columns
        sorted_cols = sorted(collapsed_data.columns, key=col_order)
        collapsed_data = collapsed_data[sorted_cols]

        self.logger.info('data aggregated at the spell level')

        collapsed_frame = BipartiteLongCollapsed(collapsed_data)
        collapsed_frame.set_attributes(self, no_dict=True)

        return collapsed_frame

    def get_es(self):
        '''
        Return long form data reformatted into event study data.

        Returns:
            es_frame (BipartiteEventStudy): BipartiteEventStudy object generated from long data
        '''
        # Determine whether m, cluster columns exist
        m = self.col_included('m')
        clustered = self.col_included('j')

        if not m:
            # Generate m column
            self.gen_m()

        # Split workers by movers and stayers
        stayers = pd.DataFrame(self[self['m'] == 0])
        movers = pd.DataFrame(self[self['m'] == 1])
        self.logger.info('workers split by movers and stayers')

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
        self.logger.info('columns updated')

        # Merge stayers and movers
        data_es = pd.concat([stayers, movers]).reset_index(drop=True)

        # Sort columns
        sorted_cols = sorted(data_es.columns, key=col_order)
        data_es = data_es[sorted_cols]

        self.logger.info('data reformatted as event study')

        es_frame = BipartiteEventStudy(data_es)
        es_frame.set_attributes(self, no_dict=True)

        return es_frame

class BipartiteLongCollapsed(BipartiteLongBase):
    '''
    Class for bipartite networks of firms and workers in collapsed long form (i.e. employment spells are collapsed into a single observation). Inherits from BipartiteLongBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        col_dict (dict): make data columns readable (requires: wid (worker id), comp (compensation), fid (firm id), year). Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, **kwargs):
        columns_opt = ['weight']
        reference_dict = {'year': ['year_start', 'year_end'], 'weight': 'weight'}
        col_dtype_dict = {'weight': 'float'}
        # Initialize DataFrame
        super().__init__(*args, columns_opt=columns_opt, reference_dict=reference_dict, **kwargs)

        self.logger.info('BipartiteLongCollapsed object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteLongCollapsed

    def clean_data(self, inplace=True):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteLongCollapsed): BipartiteLongCollapsed with cleaned data
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        frame.logger.info('beginning BipartiteLongCollapsed data cleaning')
        frame.logger.info('checking quality of data')
        frame.data_validity()

        frame.logger.info('BipartiteLongCollapsed data cleaning complete')

        BipartiteBase.clean_data(self)

        return frame

    def data_validity(self, inplace=True):
        '''
        Checks that data is formatted correctly and updates relevant attributes.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteLongCollapsed): BipartiteLongCollapsed with corrected columns and attributes
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        success = True

        frame.logger.info('--- checking worker-year observations ---')
        max_obs_start = frame.groupby(['wid', 'year_start']).size().max()
        max_obs_end = frame.groupby(['wid', 'year_end']).size().max()
        max_obs = max(max_obs_start, max_obs_end)

        frame.logger.info('max number of worker-year observations (should be 1):' + str(max_obs))
        if max_obs > 1:
            success = False

        frame.logger.info('BipartiteLongCollapsed success:' + str(success))

        return frame

    def get_es(self):
        '''
        Return collapsed long form data reformatted into event study data.

        Returns:
            es_frame (BipartiteEventStudyCollapsed): BipartiteEventStudyCollapsed object generated from collapsed long data
        '''
        # Determine whether m, cluster columns exist
        weighted = self.col_included('weight')
        m = self.col_included('m')
        clustered = self.col_included('j')

        if not m:
            # Generate m column
            self.gen_m()

        # Split workers by movers and stayers
        stayers = pd.DataFrame(self[self['m'] == 0])
        movers = pd.DataFrame(self[self['m'] == 1])
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

        es_frame = BipartiteEventStudyCollapsed(data_es)
        es_frame.set_attributes(self, no_dict=True)

        return es_frame

class BipartiteEventStudyBase(BipartiteBase):
    '''
    Base class for BipartiteEventStudy and BipartiteEventStudyCollapsed, where BipartiteEventStudy and BipartiteEventStudyCollapsed give a bipartite network of firms and workers in event study and collapsed event study form, respectively. Contains generalized methods. Inherits from BipartiteBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        columns_req (list): required columns (only put general column names for joint columns, e.g. put 'fid' instead of 'f1i', 'f2i'; then put the joint columns in reference_dict)
        columns_opt (list): optional columns (only put general column names for joint columns, e.g. put 'j' instead of 'j1', 'j2'; then put the joint columns in reference_dict)
        reference_dict (dict): clarify which columns are associated with a general column name, e.g. {'wid': 'wid', 'j': ['j1', 'j2']}
        col_dtype_dict (dict): link column to datatype
        col_dict (dict or None): make data columns readable. Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, columns_req=[], columns_opt=[], reference_dict={}, col_dtype_dict={}, col_dict=None, **kwargs):
        columns_opt += ['year']
        reference_dict = update_dict({'fid': ['f1i', 'f2i'], 'comp': ['y1', 'y2'], 'j': ['j1', 'j2']}, reference_dict)
        # Initialize DataFrame
        super().__init__(*args, columns_req=columns_req, columns_opt=columns_opt, reference_dict=reference_dict, col_dtype_dict=col_dtype_dict, col_dict=col_dict, **kwargs)

        self.logger.info('BipartiteEventStudy object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteEventStudyBase

    def clean_data(self, inplace=True):
        '''
        Clean data to make sure there are no NaN or duplicate observations, firms are connected by movers and firm ids are contiguous.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteEventStudyBase): BipartiteEventStudyBase with cleaned data
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        frame.logger.info('beginning BipartiteEventStudyBase data cleaning')
        frame.logger.info('checking quality of data')
        frame.data_validity()

        frame.logger.info('BipartiteEventStudyBase data cleaning complete')

        BipartiteBase.clean_data(self)

        return frame

    def data_validity(self, inplace=True):
        '''
        Checks that data is formatted correctly and updates relevant attributes.

        Arguments:
            inplace (bool): if True, modify in-place

        Returns:
            frame (BipartiteEventStudyBase): BipartiteEventStudyBase with corrected columns and attributes
        '''
        if inplace:
            frame = self
        else:
            frame = self.copy()

        success_stayers = True
        success_movers = True

        stayers = frame[frame['m'] == 0]
        movers = frame[frame['m'] == 1]

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

        frame.logger.info('Overall success for stayers:' + str(success_stayers))
        frame.logger.info('Overall success for movers:' + str(success_movers))

        return frame

    def get_cs(self):
        '''
        Return event study data reformatted into cross section data.

        Returns:
            data_cs (Pandas DataFrame): cross section data
        '''
        # Determine whether m column exists
        if not self.col_included('m'):
            self.gen_m()

        sdata = self[self['m'] == 0]
        jdata = self[self['m'] == 1]

        # # Assign some values
        # ns = len(sdata)
        # nm = len(jdata)

        # # Reset index
        # sdata.set_index(np.arange(ns) + 1 + nm)
        # jdata.set_index(np.arange(nm) + 1)

        # Columns used for constructing cross section
        cs_cols = self.included_cols()

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

class BipartiteEventStudy(BipartiteEventStudyBase):
    '''
    Class for bipartite networks of firms and workers in event study form. Inherits from BipartiteEventStudyBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        col_dict (dict): make data columns readable (requires: wid (worker id), y1 (compensation 1), y2 (compensation 2), f1i (firm id 1), f2i (firm id 2), m (0 if stayer, 1 if mover); optionally include: year_1 (year of observation 1), year_2 (year of observation 2)). Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, **kwargs):
        reference_dict = {'year': ['year_1', 'year_2']}
        # Initialize DataFrame
        super().__init__(*args, reference_dict=reference_dict, col_dict=col_dict, **kwargs)

        self.logger.info('BipartiteEventStudy object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteEventStudy

    def get_long(self):
        '''
        Return event study data reformatted into long form.

        Returns:
            long_frame (BipartiteLong): BipartiteLong object generated from event study data
        '''
        # Determine whether weight, m, cluster, year columns exist
        weighted = self.col_included('weight')
        m = self.col_included('m')
        clustered = self.col_included('j')
        years = self.col_included('year')

        if not m:
            self.gen_m()

        # Columns to drop
        drops = ['f2i', 'y2']

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
            'm': int
        }

        if clustered:
            drops += ['j2']
            astype_dict['j'] = int
        if weighted:
            drops += ['w2']
            astype_dict['weight'] = int
        if years:
            drops += ['year_2']
            astype_dict['year'] = int

        # Append the last row if a mover (this is because the last observation is only given as an f2i, never as an f1i)
        data_long = pd.DataFrame(self).groupby('wid').apply(lambda a: a.append(a.iloc[-1].rename(rename_dict_1, axis=1)) if a.iloc[0]['m'] == 1 else a) \
            .reset_index(drop=True) \
            .drop(drops, axis=1) \
            .rename(rename_dict_2, axis=1) \
            .astype(astype_dict)

        # Sort columns
        sorted_cols = sorted(data_long.columns, key=col_order)
        data_long = data_long[sorted_cols]

        long_frame = BipartiteLong(data_long)
        long_frame.set_attributes(self, no_dict=True)

        return long_frame

class BipartiteEventStudyCollapsed(BipartiteEventStudyBase):
    '''
    Class for bipartite networks of firms and workers in collapsed event study form (i.e. employment spells are collapsed into a single observation). Inherits from BipartiteEventStudyBase.

    Arguments:
        *args: arguments for Pandas DataFrame
        col_dict (dict): make data columns readable (requires: wid (worker id), y1 (compensation 1), y2 (compensation 2), f1i (firm id 1), f2i (firm id 2), m (0 if stayer, 1 if mover); optionally include: year_start_1 (first year of observation 1 spell), year_end_1 (last year of observation 1 spell), year_start_2 (first year of observation 2 spell), year_end_2 (last year of observation 2 spell)). Keep None if column names already correct
        **kwargs: keyword arguments for Pandas DataFrame
    '''

    def __init__(self, *args, col_dict=None, **kwargs):
        columns_opt = ['weight']
        reference_dict = {'year': ['year_start_1', 'year_end_1', 'year_start_2', 'year_end_2']}
        # Initialize DataFrame
        super().__init__(*args, columns_opt=columns_opt, reference_dict=reference_dict, col_dict=col_dict, **kwargs)

        self.logger.info('BipartiteEventStudyCollapsed object initialized')

    @property
    def _constructor(self):
        '''
        For inheritance from Pandas.
        '''
        return BipartiteEventStudyCollapsed

    def get_collapsed_long(self):
        '''
        Return collapsed event study data reformatted into collapsed long form.

        Returns:
            collapsedlong_frame (BipartiteCollapsed): BipartiteCollapsed object generated from event study data
        '''
        # Determine whether weight, m, cluster, year columns exist
        weighted = self.col_included('weight')
        m = self.col_included('m')
        clustered = self.col_included('j')
        years = self.col_included('year')

        if not m:
            self.gen_m()

        # Columns to drop
        drops = ['f2i', 'y2']

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
            'm': int
        }

        if clustered:
            drops += ['j2']
            astype_dict['j'] = int
        if weighted:
            drops += ['w2']
            astype_dict['weight'] = int
        if years:
            drops += ['year_start_2', 'year_end_2']
            astype_dict['year_start'] = int
            astype_dict['year_end'] = int

        # Append the last row if a mover (this is because the last observation is only given as an f2i, never as an f1i)
        data_collapsed_long = pd.DataFrame(self).groupby('wid').apply(lambda a: a.append(a.iloc[-1].rename(rename_dict_1, axis=1)) if a.iloc[0]['m'] == 1 else a) \
            .reset_index(drop=True) \
            .drop(drops, axis=1) \
            .rename(rename_dict_2, axis=1) \
            .astype(astype_dict)

        # Sort columns
        sorted_cols = sorted(data_collapsed_long.columns, key=col_order)
        data_collapsed_long = data_collapsed_long[sorted_cols]

        collapsedlong_frame = BipartiteLongCollapsed(data_collapsed_long)
        collapsedlong_frame.set_attributes(self, no_dict=True)

        return collapsedlong_frame
