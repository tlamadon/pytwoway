'''
Class for Attrition plots
'''
from multiprocessing import Pool
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import bipartitepandas as bpd
import pytwoway as tw
from tqdm import trange
import warnings

def attrition_increasing(bdf, subsets=np.linspace(0.1, 0.5, 5), threshold=15, user_clean={}, rng=np.random.default_rng()):
    '''
    First, keep only firms that have at minimum `threshold` many movers. Then take a random subset of subsets[0] percent of remaining movers. Constructively rebuild the data to reach each subsequent value of subsets. Return these subsets as an iterator.

    Arguments:
        bdf (BipartiteBase): data
        subsets (list): percents of movers to keep (must be weakly increasing)
        threshold (int): minimum number of movers required to keep a firm
        user_clean (dict): dictionary of parameters for cleaning

            Dictionary parameters:

                connectedness (str or None, default='connected'): if 'connected', keep observations in the largest connected set of firms; if 'biconnected', keep observations in the largest biconnected set of firms; if None, keep all observations

                i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then data is converted to long, cleaned, then reconverted to its original format

                data_validity (bool, default=True): if True, run data validity checks; much faster if set to False

                copy (bool, default=False): if False, avoid copy

        rng (NumPy RandomState): NumPy RandomState object

    Returns:
        subset (iterator of BipartiteBase): subset of data
    '''
    bdf = bdf.copy()
    bpd.logger_init(bdf) # This stops a weird logging bug that stops multiprocessing from working
    # Update clean_params to make computations faster
    clean_params = user_clean.copy()
    clean_params['data_validity'] = False
    clean_params['copy'] = False

    # Make sure subsets are weakly increasing
    if np.min(np.diff(np.array(subsets))) < 0:
        warnings.warn('Subsets must be weakly increasing.')
        return None

    # Take subset of firms that meet threshold
    threshold_firms = bdf.min_movers(threshold=threshold, copy=False)
    subset_init = bdf.keep_ids('j', threshold_firms, copy=False)

    # Worker ids in base subset
    wids_init = subset_init.loc[subset_init['m'] == 1, 'i'].unique()

    # Draw first subset
    n_wid_drops_1 = int(np.floor((1 - subsets[0]) * len(wids_init))) # Number of wids to drop
    wid_drops_1 = set(rng.choice(wids_init, size=n_wid_drops_1, replace=False)) # Draw wids to drop
    ##### Disable Pandas warning #####
    pd.options.mode.chained_assignment = None
    subset_1 = subset_init.drop_ids('i', wid_drops_1, copy=False).copy()._reset_attributes().clean_data(clean_params).gen_m()
    ##### Re-enable Pandas warning #####
    pd.options.mode.chained_assignment = 'warn'
    subset_1_orig_ids = subset_1.original_ids(copy=False)

    yield subset_1

    # Get list of all valid firms
    valid_firms = []
    for j_subcol in bpd.to_list(bdf.reference_dict['j']):
        original_j = 'original_' + j_subcol
        if original_j not in subset_1_orig_ids.columns:
            # If ids didn't change for subset
            original_j = j_subcol
        valid_firms += list(subset_1_orig_ids[original_j].unique())
    valid_firms = set(valid_firms)

    # Take all data for list of firms in smallest subset
    subset_init = subset_init.keep_ids('j', valid_firms, copy=False)

    # Determine which wids (for movers) can still be drawn
    all_valid_wids = set(subset_init.loc[subset_init['m'] == 1, 'i'].unique())
    dropped_wids = all_valid_wids.difference(set(subset_1_orig_ids.loc[subset_1_orig_ids['m'] == 1, 'original_i'].unique()))
    subset_prev = subset_1
    del subset_1, subset_1_orig_ids

    for i, subset_pct in enumerate(subsets[1:]): # Each step, drop fewer wids
        n_wid_draws_i = min(int(np.ceil((1 - subset_pct) * len(all_valid_wids))), len(dropped_wids))
        if n_wid_draws_i <= 0:
            warnings.warn('Attrition plot does not change at iteration {}'.format(i))
            yield subset_prev
        else:
            wid_draws_i = set(rng.choice(list(dropped_wids), size=n_wid_draws_i, replace=False))
            dropped_wids = wid_draws_i

            ##### Disable Pandas warning #####
            pd.options.mode.chained_assignment = None
            subset_i = subset_init.drop_ids('i', dropped_wids, copy=False).copy()._reset_attributes().clean_data(clean_params).gen_m()
            ##### Re-enable Pandas warning #####
            pd.options.mode.chained_assignment = 'warn'

            yield subset_i

            subset_prev = subset_i

def attrition_decreasing(bdf, subsets=np.linspace(0.5, 0.1, 5), threshold=15, user_clean={}, rng=np.random.default_rng()):
    '''
    First, keep only firms that have at minimum `threshold` many movers. Then take a random subset of subsets[0] percent of remaining movers. Deconstruct the data to reach each subsequent value of subsets. Return these subsets as an iterator.

    Arguments:
        bdf (BipartiteBase): data
        subsets (list): percents of movers to keep percents of movers to keep (must be weakly decreasing)
        threshold (int): minimum number of movers required to keep a firm
        user_clean (dict): dictionary of parameters for cleaning

            Dictionary parameters:

                connectedness (str or None, default='connected'): if 'connected', keep observations in the largest connected set of firms; if 'biconnected', keep observations in the largest biconnected set of firms; if None, keep all observations

                i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then data is converted to long, cleaned, then reconverted to its original format

                data_validity (bool, default=True): if True, run data validity checks; much faster if set to False

                copy (bool, default=False): if False, avoid copy

            rng (NumPy RandomState): NumPy RandomState object

    Returns:
        subset (iterator of BipartiteBase): subset of data
    '''
    bdf = bdf.copy()
    bpd.logger_init(bdf) # This stops a weird logging bug that stops multiprocessing from working
    # Update clean_params to make computations faster
    clean_params = user_clean.copy()
    clean_params['data_validity'] = False
    clean_params['copy'] = False

    # Make sure subsets are weakly decreasing
    if np.min(np.diff(np.array(subsets))) > 0:
        warnings.warn('Subsets must be weakly decreasing.')
        return None

    # Take subset of firms that meet threshold
    threshold_firms = bdf.min_movers(threshold=threshold, copy=False)
    subset_init = bdf.keep_ids('j', threshold_firms, copy=False)

    # Worker ids in base subset
    wids_movers = subset_init.loc[subset_init['m'] == 1, 'i'].unique()
    wids_stayers = list(subset_init.loc[subset_init['m'] == 0, 'i'].unique())

    relative_fraction = subsets / (np.concatenate([[1], subsets]))[:-1]

    for frac in relative_fraction:
        n_draws = int(np.ceil(frac * len(wids_movers)))
        wids_movers = list(rng.choice(wids_movers, size=n_draws, replace=False))

        ##### Disable Pandas warning #####
        pd.options.mode.chained_assignment = None
        subset_i = subset_init.keep_ids('i', wids_movers + wids_stayers, copy=False).copy()._reset_attributes().clean_data(clean_params).gen_m()
        ##### Re-enable Pandas warning #####
        pd.options.mode.chained_assignment = 'warn'

        yield subset_i

class TwoWayAttrition:
    '''
    Class of TwoWayAttrition, which generates attrition plots using bipartite labor data.

    Arguments:
        bdf (BipartiteBase): bipartite dataframe
    '''

    def __init__(self, bdf):
        self.bdf = type(bdf)(bdf)._reset_id_reference_dict(include=True) # This overwrites the id_reference_dict without altering the original dataframe, and without copying the underlying data

        # Prevent plotting until results exist
        self.res = False

        self.default_attrition = {
            'type_and_subsets': ('increasing', np.linspace(0.1, 0.5, 5)), # How to attrition data (either 'increasing' or 'decreasing'), and subsets to consider (both are required because switching type requires swapping the order of the subsets)
            'threshold': 15, # Minimum number of movers required to keep a firm
        }

    # Cannot include two underscores because isn't compatible with starmap for multiprocessing
    # Source: https://stackoverflow.com/questions/27054963/python-attribute-error-object-has-no-attribute
    def _attrition_interior(self, bdf, fe_params={}, cre_params={}, cluster_params={}):
        '''
        Estimate all parameters of interest. This is the interior function to attrition_single.

        Arguments:
            bdf (BipartiteBase): bipartite dataframe
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

        Returns:
            {'fe': tw_net.fe_res, 'cre': tw_net.cre_res} (dict): FE results, CRE results
        '''
        # Use data to create TwoWay object (note that data is already clean)
        tw_net = tw.TwoWay(bdf)
        # Estimate FE model
        tw_net.fit_fe(user_fe=fe_params)
        # Cluster data
        tw_net.cluster(**cluster_params)
        # Estimate CRE model
        tw_net.fit_cre(user_cre=cre_params)

        return {'fe': tw_net.fe_res, 'cre': tw_net.cre_res}

    def _attrition_single(self, ncore=1, attrition_params={}, fe_params={}, cre_params={}, cluster_params={}, clean_params={}, rng=np.random.default_rng()):
        '''
        Run attrition estimations of TwoWay to estimate parameters given fraction of movers remaining. This is the interior function to attrition.

        Arguments:
            ncore (int): how many cores to use
            attrition_params (dict): dictionary of parameters for attrition

                Dictionary parameters:

                    type_and_subsets (tuple of (str, list), default=('increasing', np.linspace(0.1, 0.5, 5))): how to attrition data (either 'increasing' or 'decreasing'), and subsets to consider (both are required because switching type requires swapping the order of the subsets)

                    threshold (int, default=15): minimum number of movers required to keep a firm

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

            clean_params (dict): dictionary of parameters for cleaning

                Dictionary parameters:

                    connectedness (str or None, default='connected'): if 'connected', keep observations in the largest connected set of firms; if 'biconnected', keep observations in the largest biconnected set of firms; if None, keep all observations

                    i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids

                    data_validity (bool, default=True): if True, run data validity checks; much faster if set to False

                    copy (bool, default=False): if False, avoid copy

            rng (NumPy RandomState): NumPy RandomState object

        Returns:
            res_all (dict of dicts of lists): in the first dictionary we choose 'connected' or 'biconnected'; in the second dictionary we choose 'fe' or 'cre'; and finally, we are given a list of results for each attrition percentage.
        '''
        # Create lists to save results
        res_connected = {'fe': [], 'cre': []} # For non-HE
        res_biconnected = {'fe': [], 'cre': []} # For HE
        res_all = [res_connected, res_biconnected]
        # FE params
        fe_params_connected = fe_params.copy()
        fe_params_connected['he'] = False
        fe_params_biconnected = fe_params.copy()
        fe_params_biconnected['he'] = True
        fe_params_all = [fe_params_connected, fe_params_biconnected]
        # Clean params
        clean_params_connected = clean_params.copy()
        clean_params_connected['connectedness'] = 'connected'
        clean_params_biconnected = clean_params.copy()
        clean_params_biconnected['connectedness'] = 'biconnected'
        clean_params_all = [clean_params_connected, clean_params_biconnected]

        attrition_fn = {
            'increasing': attrition_increasing,
            'decreasing': attrition_decreasing
        }
        attrition_fn = attrition_fn[attrition_params['type_and_subsets'][0]]

        for i in range(2):
            # Create iterator
            attrition_iterator = attrition_fn(self.bdf, subsets=attrition_params['type_and_subsets'][1], threshold=attrition_params['threshold'], user_clean=clean_params_all[i], rng=rng)

            # Use multi-processing
            if False: # ncore > 1:
                # Estimate
                with Pool(processes=ncore) as pool:
                    V = pool.starmap(self._attrition_interior, [(attrition_subset, fe_params_all[i], cre_params, cluster_params) for attrition_subset in attrition_iterator])
                for res in enumerate(V):
                    res_all[i]['fe'].append(res[1]['fe'])
                    res_all[i]['cre'].append(res[1]['cre'])
            else:
                for attrition_subset in attrition_iterator:
                    # Estimate
                    res = self._attrition_interior(attrition_subset, fe_params=fe_params_all[i], cre_params=cre_params, cluster_params=cluster_params)
                    res_all[i]['fe'].append(res['fe'])
                    res_all[i]['cre'].append(res['cre'])

        res_all = {'connected': res_all[0], 'biconnected': res_all[1]}

        return res_all

    def attrition(self, N=10, ncore=1, attrition_params={}, fe_params={}, cre_params={}, cluster_params={}, clean_params={}, seed=None):
        '''
        Run Monte Carlo on attrition estimations of TwoWay to estimate variance of parameter estimates given fraction of movers remaining.

        Arguments:
            N (int): number of simulations
            ncore (int): how many cores to use
            attrition_params (dict): dictionary of parameters for attrition

                Dictionary parameters:

                    type_and_subsets (tuple of (str, list), default=('increasing', np.linspace(0.1, 0.5, 5))): how to attrition data (either 'increasing' or 'decreasing'), and subsets to consider (both are required because switching type requires swapping the order of the subsets)

                    threshold (int, default=15): minimum number of movers required to keep a firm

                    copy (bool, default=True): if False, avoid copy

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

            clean_params (dict): dictionary of parameters for cleaning

                Dictionary parameters:

                    connectedness (str or None, default='connected'): if 'connected', keep observations in the largest connected set of firms; if 'biconnected', keep observations in the largest biconnected set of firms; if None, keep all observations

                    i_t_how (str, default='max'): if 'max', keep max paying job; if 'sum', sum over duplicate worker-firm-year observations, then take the highest paying worker-firm sum; if 'mean', average over duplicate worker-firm-year observations, then take the highest paying worker-firm average. Note that if multiple time and/or firm columns are included (as in event study format), then duplicates are cleaned in order of earlier time columns to later time columns, and earlier firm ids to later firm ids

                    data_validity (bool, default=True): if True, run data validity checks; much faster if set to False

                    copy (bool, default=False): if False, avoid copy

            seed (int): NumPy RandomState seed

        Returns:
            res_all (dict of dicts of lists of lists): in the first dictionary we choose 'connected' or 'biconnected'; in the second dictionary we choose 'fe' or 'cre'; then, we are given a list of results for each Monte Carlo simulation; and finally, for a particular Monte Carlo simulation, we are given a list of results for each attrition percentage.
        '''
        rng = np.random.default_rng(seed)
        # Create lists to save results
        res_connected = {'fe': [], 'cre': []} # For non-HE
        res_biconnected = {'fe': [], 'cre': []} # For HE

        attrition_params_full = bpd.util.update_dict(self.default_attrition, attrition_params)

        # Use multi-processing
        if ncore > 1:
            # Estimate
            with Pool(processes=ncore) as pool:
                V = pool.starmap(self._attrition_single, [(ncore, attrition_params_full, fe_params, cre_params, cluster_params, clean_params) for _ in range(N)])
            for res in enumerate(V):
                res_connected['fe'].append(res[1]['connected']['fe'])
                res_connected['cre'].append(res[1]['connected']['cre'])
                res_biconnected['fe'].append(res[1]['biconnected']['fe'])
                res_biconnected['cre'].append(res[1]['biconnected']['cre'])
        else:
            for _ in trange(N):
                # Estimate
                res = self._attrition_single(ncore=ncore, attrition_params=attrition_params_full, fe_params=fe_params, cre_params=cre_params, cluster_params=cluster_params, clean_params=clean_params, rng=rng)
                res_connected['fe'].append(res['connected']['fe'])
                res_connected['cre'].append(res['connected']['cre'])
                res_biconnected['fe'].append(res['biconnected']['fe'])
                res_biconnected['cre'].append(res['biconnected']['cre'])

        res_all = {'connected': res_connected, 'biconnected': res_biconnected}

        self.type_and_subsets = attrition_params_full['type_and_subsets']
        self.res = res_all

    def plot_attrition(self):
        '''
        Plot results from Monte Carlo simulations.
        '''
        if not self.res:
            warnings.warn('Must generate attrition data before results can be plotted. This can be done by running .attrition()')

        else:
            # Extract attrition type and subsets
            attrition_type, subsets = self.type_and_subsets
            # Get N, M
            N = len(self.res['connected']['fe']) # Number of estimations
            M = len(self.res['connected']['fe'][0]) # Number of attritions per estimation
            # Extract results
            # Connected set
            conset_var_psi_pct = np.zeros(shape=[N, M, 3])
            conset_cov_psi_alpha_pct = np.zeros(shape=[N, M, 3])
            # Biconnected set
            biconset_var_psi_pct = np.zeros(shape=[N, M, 4])
            biconset_cov_psi_alpha_pct = np.zeros(shape=[N, M, 4])
            for i in range(N):
                for j in range(M):
                    # Connected set
                    con_res_fe_dict = self.res['connected']['fe'][i][j]
                    con_res_cre_dict = self.res['connected']['cre'][i][j]
                    # Var(psi)
                    conset_var_psi_pct[i, j, 0] = float(con_res_fe_dict['var_fe']) / float(con_res_fe_dict['var_y'])
                    conset_var_psi_pct[i, j, 1] = float(con_res_fe_dict['var_ho']) / float(con_res_fe_dict['var_y'])
                    conset_var_psi_pct[i, j, 2] = float(con_res_cre_dict['tot_var']) / float(con_res_cre_dict['var_y'])
                    # Cov(psi, alpha)
                    conset_cov_psi_alpha_pct[i, j, 0] = 2 * float(con_res_fe_dict['cov_fe']) / float(con_res_fe_dict['var_y'])
                    conset_cov_psi_alpha_pct[i, j, 1] = 2 * float(con_res_fe_dict['cov_ho']) / float(con_res_fe_dict['var_y'])
                    conset_cov_psi_alpha_pct[i, j, 2] = 2 * float(con_res_cre_dict['tot_cov']) / float(con_res_cre_dict['var_y'])

                    # Biconnected set
                    bicon_res_fe_dict = self.res['biconnected']['fe'][i][j]
                    bicon_res_cre_dict = self.res['biconnected']['cre'][i][j]
                    # Var(psi)
                    biconset_var_psi_pct[i, j, 0] = float(bicon_res_fe_dict['var_fe']) / float(bicon_res_fe_dict['var_y'])
                    biconset_var_psi_pct[i, j, 1] = float(bicon_res_fe_dict['var_ho']) / float(bicon_res_fe_dict['var_y'])
                    biconset_var_psi_pct[i, j, 2] = float(bicon_res_cre_dict['tot_var']) / float(bicon_res_cre_dict['var_y'])
                    biconset_var_psi_pct[i, j, 3] = float(bicon_res_fe_dict['var_he']) / float(bicon_res_fe_dict['var_y'])

                    # Cov(psi, alpha)
                    biconset_cov_psi_alpha_pct[i, j, 0] = 2 * float(bicon_res_fe_dict['cov_fe']) / float(bicon_res_fe_dict['var_y'])
                    biconset_cov_psi_alpha_pct[i, j, 1] = 2 * float(bicon_res_fe_dict['cov_ho']) / float(bicon_res_fe_dict['var_y'])
                    biconset_cov_psi_alpha_pct[i, j, 2] = 2 * float(bicon_res_cre_dict['tot_cov']) / float(bicon_res_cre_dict['var_y'])
                    biconset_cov_psi_alpha_pct[i, j, 3] = 2 * float(bicon_res_fe_dict['cov_he']) / float(bicon_res_fe_dict['var_y'])

            # x-axis
            x_axis = (100 * subsets).astype(int)
            conset_var_psi_pct = 100 * conset_var_psi_pct
            conset_cov_psi_alpha_pct = 100 * conset_cov_psi_alpha_pct
            biconset_var_psi_pct = 100 * biconset_var_psi_pct
            biconset_cov_psi_alpha_pct = 100 * biconset_cov_psi_alpha_pct

            # Flip along 1st axis so both increasing and decreasing have the same order
            if attrition_type == 'decreasing':
                x_axis = np.flip(x_axis)
                conset_var_psi_pct = np.flip(conset_var_psi_pct, axis=1)
                conset_cov_psi_alpha_pct = np.flip(conset_cov_psi_alpha_pct, axis=1)
                biconset_var_psi_pct = np.flip(biconset_var_psi_pct, axis=1)
                biconset_cov_psi_alpha_pct = np.flip(biconset_cov_psi_alpha_pct, axis=1)

            # Plot figures
            # Firm effects (connected set)
            plt.plot(x_axis, conset_var_psi_pct[:, :, 0].mean(axis=0), label='FE')
            plt.plot(x_axis, conset_var_psi_pct[:, :, 1].mean(axis=0), label='HO')
            plt.plot(x_axis, conset_var_psi_pct[:, :, 2].mean(axis=0), label='CRE')
            plt.title('Firm effects (connected set)')
            plt.xlabel('Share of Movers Kept (%)')
            plt.ylabel('Firm Effects: Share of Variance (%)')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()
            # Firm effects (biconnected set)
            plt.plot(x_axis, biconset_var_psi_pct[:, :, 0].mean(axis=0), label='FE')
            plt.plot(x_axis, biconset_var_psi_pct[:, :, 1].mean(axis=0), label='HO')
            plt.plot(x_axis, biconset_var_psi_pct[:, :, 2].mean(axis=0), label='CRE')
            plt.plot(x_axis, biconset_var_psi_pct[:, :, 3].mean(axis=0), label='HE')
            plt.title('Firm effects (biconnected set)')
            plt.xlabel('Share of Movers Kept (%)')
            plt.ylabel('Firm Effects: Share of Variance (%)')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

            # Sorting (connected set)
            plt.plot(x_axis, conset_cov_psi_alpha_pct[:, :, 0].mean(axis=0), label='FE')
            plt.plot(x_axis, conset_cov_psi_alpha_pct[:, :, 1].mean(axis=0), label='HO')
            plt.plot(x_axis, conset_cov_psi_alpha_pct[:, :, 2].mean(axis=0), label='CRE')
            plt.title('Sorting (connected set)')
            plt.xlabel('Share of Movers Kept (%)')
            plt.ylabel('Sorting: Share of Variance (%)')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()
            # Sorting (biconnected set)
            plt.plot(x_axis, biconset_cov_psi_alpha_pct[:, :, 0].mean(axis=0), label='FE')
            plt.plot(x_axis, biconset_cov_psi_alpha_pct[:, :, 1].mean(axis=0), label='HO')
            plt.plot(x_axis, biconset_cov_psi_alpha_pct[:, :, 2].mean(axis=0), label='CRE')
            plt.plot(x_axis, biconset_cov_psi_alpha_pct[:, :, 3].mean(axis=0), label='HE')
            plt.title('Sorting (biconnected set)')
            plt.xlabel('Share of Movers Kept (%)')
            plt.ylabel('Sorting: Share of Variance (%)')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

            # Plot boxplots
            # Firm effects (connected set)
            fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
            subtitles = ['FE', 'FE-HO', 'CRE']

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(conset_var_psi_pct[:, :, i], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[i])
            fig.suptitle('Firm effects (connected set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Firm Effects: Share of Variance (%)')
            fig.tight_layout()
            fig.show()
            # Firm effects (biconnected set)
            fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True)
            subtitles = ['FE', 'FE-HO', 'CRE', 'FE-HE']
            order = [0, 1, 3, 2] # Because data is FE, FE-HO, CRE, FE-HE

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(biconset_var_psi_pct[:, :, order[i]], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[order[i]])
            fig.suptitle('Firm effects (biconnected set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Firm Effects: Share of Variance (%)')
            fig.tight_layout()
            fig.show()

            # Sorting (connected set)
            fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
            subtitles = ['FE', 'FE-HO', 'CRE']

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(conset_cov_psi_alpha_pct[:, :, i], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[i])
            fig.suptitle('Sorting (connected set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Sorting: Share of Variance (%)')
            fig.tight_layout()
            fig.show()
            # Sorting (biconnected set)
            fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True)
            subtitles = ['FE', 'FE-HO', 'CRE', 'FE-HE']
            order = [0, 1, 3, 2] # Because data is FE, FE-HO, CRE, FE-HE

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(biconset_cov_psi_alpha_pct[:, :, order[i]], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[order[i]])
            fig.suptitle('Sorting (biconnected set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Sorting: Share of Variance (%)')
            fig.tight_layout()
            fig.show()
