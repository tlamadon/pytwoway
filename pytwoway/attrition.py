'''
Class for attrition estimation and plotting.
'''
from tqdm import tqdm, trange
import warnings
import itertools
from multiprocessing import Pool
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import bipartitepandas as bpd
# from bipartitepandas.util import to_list, logger_init
import pytwoway as tw

class AttritionIncreasing():
    '''
    Generate increasing subsets of a dataset to estimate the effects of attrition. Do this by first drawing a given fraction of all movers. Fix the set of firms connected by these movers. Then, constructively rebuild the data by adding back successively increasing fractions of the previously dropped movers that belong to the fixed set of firms.

    Arguments:
        subset_fractions (NumPy Array or None): fractions of movers to consider for each subset. Must be (weakly) monotonically increasing. None is equivalent to np.linspace(0.1, 0.5, 5).
        n_subsample_draws (int): maximum number of attempts to draw a valid initial subsample (this is necessary because the random draw may ultimately drop too many observations)
        subsample_min_firms (int): minimum number of firms necessary for an initial subsample to be considered valid. This should be at least the number of clusters used when estimating the CRE model.
    '''

    def __init__(self, subset_fractions=None, n_subsample_draws=5, subsample_min_firms=20):
        if subset_fractions is None:
            subset_fractions = np.linspace(0.1, 0.5, 5)

        if not (np.min(np.diff(subset_fractions)) >= 0):
            raise ValueError('Subset fractions must be weakly increasing for AttritionIncreasing().')

        self.subset_fractions = subset_fractions
        self.n_subsample_draws = n_subsample_draws
        self.subsample_min_firms = subsample_min_firms

    def gen_subsets(self, bdf, clean_params=None, rng=None):
        '''
        Generate attrition subsets.

        Arguments:
            bdf (BipartiteDataFrame): bipartite dataframe
            clean_params (ParamsDict or None): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.clean_params().
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (list of tuples): each entry gives a tuple of (fids_to_drop, wids_to_drop), where fids_to_drop (wids_to_drop) gives a set of firm (worker) ids to drop prior to cleaning the BipartiteDataFrame; if fids_to_drop (wids_to_drop) is None, no firm (worker) ids need to be dropped
        '''
        if clean_params is None:
            clean_params = bpd.clean_params()
        if rng is None:
            rng = np.random.default_rng(None)

        if bdf.n_firms() < self.subsample_min_firms:
            # Make sure there are enough firms in the original dataframe
            raise ValueError("Estimating attrition increasing, and dataframe has fewer firms than the minimum threshold required after dropping firms with too few moves. Considering lowering `min_moves_threshold` for tw.Attrition() or lowering `subsample_min_firms` for tw.AttritionIncreasing().")

        # List to return
        return_lst = []

        # Update clean_params
        clean_params['is_sorted'] = True
        clean_params['copy'] = False

        # Get subset fractions
        fracs = self.subset_fractions

        # Must reset id_reference_dict
        bdf = bdf._reset_id_reference_dict(include=True)

        # Get firm ids in base subset
        fids_init = set(bdf.loc[bdf.loc[:, 'm'].to_numpy() > 0, :].unique_ids('j'))
        # Get worker ids in base subset
        wids_init = bdf.loc[bdf.loc[:, 'm'].to_numpy() > 0, :].unique_ids('i')

        for i in range(self.n_subsample_draws):
            # Sometimes we draw so few firms that it's likely to end up with no observations left, so redraw until success
            ## Draw first subset ##
            # Number of wids to drop
            n_wid_drops_1 = int(np.floor((1 - fracs[0]) * len(wids_init)))
            # Draw wids to drop
            wid_drops_1 = set(rng.choice(wids_init, size=n_wid_drops_1, replace=False))
            try:
                # NOTE: this requires the copy if there are returners
                subset_1 = bdf.drop_ids('i', wid_drops_1, drop_returns_to_stays=clean_params['drop_returns_to_stays'], is_sorted=True, reset_index=True, copy=True)._reset_attributes(columns_contig=True, connected=True, no_na=False, no_duplicates=False, i_t_unique=False, no_returns=False).clean(clean_params)
                n_firms = subset_1.n_firms()
                if n_firms >= self.subsample_min_firms:
                    # If sufficiently many firms, break out of the loop
                    break
                else:
                    raise ValueError(f"Insufficiently many firms remain in attrition draw: minimum threshold is {self.subsample_min_firms} firms, but only {n_firms} remain.")
            except ValueError as v:
                if i == (self.n_subsample_draws - 1):
                    # Don't fail unless it's the last loop
                    raise ValueError(v)

        subset_1_orig_ids = subset_1.original_ids(copy=False)
        del subset_1

        ### Extract firms and workers in initial subset ###
        ## Fixed subset of firms ##
        valid_firms = []
        for j_subcol in bpd.util.to_list(bdf.col_reference_dict['j']):
            original_j = 'original_' + j_subcol
            if original_j not in subset_1_orig_ids.columns:
                # If no changes to this column
                original_j = j_subcol
            valid_firms += list(subset_1_orig_ids.loc[:, original_j].unique())
        valid_firms = set(valid_firms)
        fids_to_drop = fids_init.difference(valid_firms)
        ret_fid = {
            True: fids_to_drop,
            False: None
        }
        ret_wid = {
            True: wid_drops_1,
            False: None
        }
        return_lst.append((ret_fid[len(fids_to_drop) > 0], ret_wid[len(wid_drops_1) > 0]))

        ## Drawn workers ##
        original_i = 'original_i'
        if original_i not in subset_1_orig_ids.columns:
            # If no changes to i column
            original_i = 'i'
        drawn_wids = set(subset_1_orig_ids.loc[subset_1_orig_ids.loc[:, 'm'].to_numpy() > 0, original_i].unique())
        del subset_1_orig_ids

        ## Draw new subsets ##
        # Take all data for list of firms in smallest subset (NOTE: this requires the copy if there are returners)
        subset_2 = bdf.keep_ids('j', valid_firms, drop_returns_to_stays=clean_params['drop_returns_to_stays'], is_sorted=True, reset_index=True, copy=True)
        del bdf
        # Clear id_reference_dict since it is no longer necessary
        subset_2._reset_id_reference_dict()

        # Determine which wids (for movers) can still be drawn
        all_valid_wids = set(subset_2.loc[subset_2.loc[:, 'm'].to_numpy() > 0, :].unique_ids('i'))
        wids_to_drop = all_valid_wids.difference(drawn_wids)

        for i, frac in enumerate(fracs[1:]):
            # Each step, drop fewer wids
            n_wid_draws_i = min(int(np.round((1 - frac) * len(all_valid_wids), 1)), len(wids_to_drop))
            if n_wid_draws_i > 0:
                wids_to_drop = set(rng.choice(list(wids_to_drop), size=n_wid_draws_i, replace=False))
            else:
                warnings.warn(f'Attrition plot does not change at iteration {i}')

            ret_fid = {
                True: fids_to_drop,
                False: None
            }
            ret_wid = {
                True: wids_to_drop,
                False: None
            }
            return_lst.append((ret_fid[len(fids_to_drop) > 0], ret_wid[len(wids_to_drop) > 0]))

        return return_lst

    def update_clean_params(self, clean_params):
        '''
        Update clean_params so 'connectedness' = None (once the initial connectedness has been computed, larger subsets are connected by construction).

        Arguments:
            clean_params (ParamsDict): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters.

        Returns:
            (ParamsDict) updated clean_params
        '''
        clean_params = clean_params.copy()
        clean_params['connectedness'] = None

        return clean_params


class AttritionDecreasing():
    '''
    Generate decreasing subsets of a dataset to estimate the effects of attrition. Do this by first drawing successively decreasing fractions of movers from the original dataset.

    Arguments:
        subset_fractions (NumPy Array or None): fractions of movers to consider for each subset. Must be (weakly) monotonically decreasing. None is equivalent to np.linspace(0.5, 0.1, 5).
    '''

    def __init__(self, subset_fractions=None):
        if subset_fractions is None:
            subset_fractions = np.linspace(0.5, 0.1, 5)

        if not (np.max(np.diff(subset_fractions)) <= 0):
            raise ValueError('Subset fractions must be weakly decreasing for AttritionDecreasing().')

        self.subset_fractions = subset_fractions

    def gen_subsets(self, bdf, clean_params=None, rng=None):
        '''
        Generate attrition subsets.

        Arguments:
            bdf (BipartiteBase): bipartite dataframe
            clean_params (ParamsDict or None): used for AttritionIncreasing(), does nothing for AttritionDecreasing()
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (list of tuples): each entry gives a tuple of (fids_to_drop, wids_to_drop), where fids_to_drop (wids_to_drop) gives a set of firm (worker) ids to drop prior to cleaning the BipartiteDataFrame; if fids_to_drop (wids_to_drop) is None, no firm (worker) ids need to be dropped
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # List to return
        return_lst = []

        # Worker ids in base subset
        wids_movers = set(bdf.loc[bdf.loc[:, 'm'].to_numpy() > 0, :].unique_ids('i'))
        # wids_stayers = list(bdf.loc[bdf.loc[:, 'm'].to_numpy() == 0, :].unique_ids('i'))

        # # Drop m since it can change for leave-one-out components
        # bdf.drop('m')

        relative_drop_fraction = 1 - self.subset_fractions / (np.concatenate([[1], self.subset_fractions]))[:-1]
        wids_to_drop = set()

        for i, drop_frac in enumerate(relative_drop_fraction):
            n_wid_draws_i = min(int(np.round(drop_frac * len(wids_movers), 1)), len(wids_movers))
            if n_wid_draws_i > 0:
                wid_draws_i = set(rng.choice(list(wids_movers), size=n_wid_draws_i, replace=False))
                wids_movers = wids_movers.difference(wid_draws_i)
                wids_to_drop = wids_to_drop.union(wid_draws_i)
            else:
                warnings.warn(f'Attrition plot does not change at iteration {i}')

            if len(wids_to_drop) > 0:
                return_lst.append((None, wids_to_drop))
            else:
                return_lst.append((None, None))

            # subset_i = bdf.keep_ids('i', wids_movers + wids_stayers, copy=True)._reset_id_reference_dict()._reset_attributes(columns_contig=True, connected=True, no_na=False, no_duplicates=False, i_t_unique=False, no_returns=False)

        return return_lst

    def update_clean_params(self, clean_params):
        '''
        Used for AttritionIncreasing(), does nothing for AttritionDecreasing().

        Arguments:
            clean_params (ParamsDict): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters.

        Returns:
            (ParamsDict) clean_params
        '''
        return clean_params

def _scramble(lst):
    '''
    Reorder a list from [a, b, c, d, e] to [a, e, b, d, c]. This is used for attrition with multiprocessing, to ensure memory usage stays relatively constant, by mixing together large and small draws. Scrambled lists can be unscrambled with _unscramble().

    Arguments:
        lst (list): list to scramble

    Returns:
        (list): scrambled list
    '''
    new_lst = []
    for i in range(len(lst)):
        if i % 2 == 0:
            new_lst.append(lst[i // 2])
        else:
            new_lst.append(lst[len(lst) - i // 2 - 1])

    return new_lst

def _unscramble(lst):
    '''
    Reorder a list from [a, e, b, d, c] to [a, b, c, d, e]. This undoes the scrambling done by _scramble().

    Arguments:
        lst (list): list to unscramble

    Returns:
        (list): unscrambled list
    '''
    front_lst = []
    back_lst = []
    for i, element in enumerate(lst):
        if i % 2 == 0:
            front_lst.append(element)
        else:
            back_lst.append(element)

    return front_lst + list(reversed(back_lst))

class Attrition:
    '''
    Class of Attrition, which generates attrition plots using bipartite labor data.

    Arguments:
        min_moves_threshold (int): minimum number of moves required to keep a firm
        attrition_how (AttritionIncreasing() or AttritionDecreasing()): instance of AttritionIncreasing() or AttritionDecreasing(), used to specify if attrition should constructively build up using a fixed set of firms, or randomly drop increasing fractions of movers from the original dataset; None is equivalent to tw.AttritionIncreasing()
        fe_params (ParamsDict or None): dictionary of parameters for FE estimation. Run tw.fe_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.fe_params().
        cre_params (ParamsDict or None): dictionary of parameters for CRE estimation. Run tw.cre_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.cre_params().
        cluster_params (ParamsDict or None): dictionary of parameters for clustering in CRE estimation. Run bpd.cluster_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.cluster_params().
        clean_params (ParamsDict or None): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.clean_params().
    '''

    def __init__(self, min_moves_threshold=15, attrition_how=None, fe_params=None, cre_params=None, cluster_params=None, clean_params=None):
        if attrition_how is None:
            attrition_how = AttritionIncreasing()
        if fe_params is None:
            fe_params = tw.fe_params()
        if cre_params is None:
            cre_params = tw.cre_params()
        if cluster_params is None:
            cluster_params = bpd.cluster_params()
        if clean_params is None:
            clean_params = bpd.clean_params()

        ## Save attributes ##
        # Minimum number of moves required to keep a firm
        self.min_moves_threshold = min_moves_threshold
        # AttritionIncreasing() or AttritionDecreasing()
        self.attrition_how = attrition_how
        # Prevent plotting until results exist
        self.attrition_res = False

        #### Parameter dictionaries ####
        ### Save parameter dictionaries ###
        self.fe_params = fe_params.copy()
        self.cre_params = cre_params.copy()
        self.cluster_params = cluster_params.copy()
        self.clean_params = clean_params.copy()

        ### Update parameter dictionaries ###
        # Clean
        self.clean_params['is_sorted'] = True
        self.clean_params['force'] = False
        self.clean_params['copy'] = False

        # Cluster
        self.cluster_params['clean_params'] = self.clean_params.copy()
        self.cluster_params['is_sorted'] = True
        self.cluster_params['copy'] = False

        ## Non-HE and HE parameter dictionaries ##
        # FE
        self.fe_params_non_he = self.fe_params.copy()
        self.fe_params_non_he['he'] = False
        self.fe_params_he = self.fe_params.copy()
        self.fe_params_he['he'] = True

        # Clean
        self.clean_params_non_he = self.clean_params.copy()
        self.clean_params_non_he['connectedness'] = 'connected'
        self.clean_params_he = self.clean_params.copy()
        self.clean_params_he['connectedness'] = 'leave_out_observation'

        # Cluster
        self.cluster_params_non_he = self.cluster_params.copy()
        self.cluster_params_non_he['clean_params'] = self.clean_params_non_he.copy()
        self.cluster_params_he = self.cluster_params.copy()
        self.cluster_params_he['clean_params'] = self.clean_params_he.copy()

        # Combined
        self.non_he_he_params = {
            'non_he': {
                'fe': self.fe_params_non_he,
                'cluster': self.cluster_params_non_he,
                'clean': self.clean_params_non_he
            },
            'he': {
                'fe': self.fe_params_he,
                'cluster': self.cluster_params_he,
                'clean': self.clean_params_he
            }
        }

    # Cannot include two underscores because isn't compatible with starmap for multiprocessing
    # Source: https://stackoverflow.com/questions/27054963/python-attribute-error-object-has-no-attribute
    def _attrition_interior(self, bdf, fids_to_drop, wids_to_drop, fe_params=None, cre_params=None, cluster_params=None, clean_params=None, rng=None):
        '''
        Estimate all parameters of interest. This is the interior function to attrition_single.

        Arguments:
            bdf (BipartiteDataFrame): bipartite dataframe
            fids_to_drop (set or None): firm ids to drop; if None, no firm ids to drop
            wids_to_drop (set or None): worker ids to drop; if None, no worker ids to drop
            fe_params (ParamsDict or None): dictionary of parameters for FE estimation. Run tw.fe_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.fe_params().
            cre_params (ParamsDict or None): dictionary of parameters for CRE estimation. Run tw.cre_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.cre_params().
            cluster_params (ParamsDict or None): dictionary of parameters for clustering in CRE estimation. Run bpd.cluster_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.cluster_params().
            clean_params (ParamsDict or None): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.clean_params().
            rng (np.random.Generator or None): NumPy random number generator. This overrides the random number generators for FE and CRE. None is equivalent to np.random.default_rng(None).

        Returns:
            (dict): {'fe': FE results, 'cre': CRE results}
        '''
        if fe_params is None:
            fe_params = tw.fe_params()
        if cre_params is None:
            cre_params = tw.cre_params()
        if cluster_params is None:
            cluster_params = bpd.cluster_params()
        if clean_params is None:
            clean_params = bpd.clean_params()
        if rng is None:
            rng = np.random.default_rng(None)

        # logger_init(bdf) # This stops a weird logging bug that stops multiprocessing from working
        ## Drop ids and clean data  ## (NOTE: this does not require a copy)
        if fids_to_drop is not None:
            bdf = bdf.drop_ids('j', fids_to_drop, drop_returns_to_stays=clean_params['drop_returns_to_stays'], is_sorted=True, copy=False)
        if wids_to_drop is not None:
            bdf = bdf.drop_ids('i', wids_to_drop, drop_returns_to_stays=clean_params['drop_returns_to_stays'], is_sorted=True, copy=False)
        bdf = bdf._reset_attributes(columns_contig=True, connected=False, no_na=False, no_duplicates=False, i_t_unique=False, no_returns=False).clean(clean_params)
        ## Estimate FE model ##
        fe_estimator = tw.FEEstimator(bdf, fe_params)
        fe_estimator.fit(rng)
        fe_res = fe_estimator.res
        del fe_estimator
        ## Estimate CRE model ##
        # Cluster
        bdf = bdf.cluster(cluster_params)
        # Estimate
        cre_estimator = tw.CREEstimator(bdf.to_eventstudy(move_to_worker=False, is_sorted=True, copy=False).get_cs(copy=False), cre_params)
        del bdf
        cre_estimator.fit(rng)

        return {'fe': fe_res, 'cre': cre_estimator.res}

    def _attrition_single(self, bdf, ncore=1, rng=None):
        '''
        Run attrition estimations to estimate parameters given fraction of movers remaining. This is the interior function to attrition.

        Arguments:
            bdf (BipartiteDataFrame): bipartite dataframe
            ncore (int): number of cores to use
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            res_all (dict of dicts of lists): in the first dictionary we choose 'non_he' or 'he'; in the second dictionary we choose 'fe' or 'cre'; and finally, we are given a list of results for each attrition percentage.
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        ## Create lists to save results ##
        # For non-HE
        res_non_he = {'fe': [], 'cre': []}
        # For HE
        res_he = {'fe': [], 'cre': []}
        # Combined
        res_all = {
            'non_he': res_non_he,
            'he': res_he
        }

        for non_he_he in ['non_he', 'he']:
            # Get parameters
            non_he_he_params = self.non_he_he_params[non_he_he]
            fe_params = non_he_he_params['fe']
            cluster_params = non_he_he_params['cluster']
            clean_params = non_he_he_params['clean']

            # Generate attrition subsets and worker ids to drop
            subsets = self.attrition_how.gen_subsets(bdf=bdf, clean_params=clean_params, rng=rng)

            # Update clean_params
            clean_params = self.attrition_how.update_clean_params(clean_params)

            ## Construct list of parameters to estimate for each subset ##
            attrition_params = []
            # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
            N = len(subsets)
            seeds = rng.bit_generator._seed_seq.spawn(N)
            for i, subset in enumerate(subsets):
                fids_to_drop_i, wids_to_drop_i = subset
                rng_i = np.random.default_rng(seeds[i])
                attrition_params.append((fids_to_drop_i, wids_to_drop_i, fe_params, self.cre_params, cluster_params, clean_params, rng_i))
            del subsets

            ## Estimate on subset ##
            if ncore > 1:
                # Multiprocessing
                with Pool(processes=ncore) as pool:
                    pbar = tqdm(_scramble((bdf, *attrition_params)), total=N)
                    pbar.set_description(f'attrition, {non_he_he}')
                    V = _unscramble(pool.starmap(self._attrition_interior, pbar))
            else:
                # Single core
                pbar = tqdm(attrition_params, total=N)
                pbar.set_description(f'attrition, {non_he_he}')
                V = []
                for attrition_subparams in pbar:
                    V.append(self._attrition_interior(bdf, *attrition_subparams))
            if non_he_he == 'he':
                del bdf

            del attrition_params, pbar

            ## Extract results ##
            for res in V:
                res_all[non_he_he]['fe'].append(res['fe'])
                res_all[non_he_he]['cre'].append(res['cre'])

        return res_all

    def attrition(self, bdf, N=10, ncore=1, copy=False, rng=None):
        '''
        Run Monte Carlo on attrition estimations of TwoWay to estimate variance of parameter estimates given fraction of movers remaining. Note that this overwrites the stored dataframe, meaning if you want to run attrition with different threshold number of movers, you will have to create multiple Attrition objects, or alternatively, run this method with an increasing threshold for each iteration.

        Arguments:
            bdf (BipartiteBase): bipartite dataframe (NOTE: we need to avoid saving bdf as a class attribute, otherwise multiprocessing will create a separate copy of it for each core used)
            N (int): number of simulations
            ncore (int): number of cores to use
            copy (bool): if False, avoid copy
            rng (np.random.Generator or None): NumPy random number generator. This overrides the random number generators for FE and CRE. None is equivalent to np.random.default_rng(None).

        Returns:
            res_all (dict of dicts of lists of lists): in the first dictionary we choose 'non_he' or 'he'; in the second dictionary we choose 'fe' or 'cre'; then, we are given a list of results for each Monte Carlo simulation; and finally, for a particular Monte Carlo simulation, we are given a list of results for each attrition percentage.
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        if copy:
            # Copy
            bdf = bdf.copy()

        ## Create lists to save results ##
        # For non-HE
        res_non_he = {'fe': [], 'cre': []}
        # For HE
        res_he = {'fe': [], 'cre': []}

        # Save movers per firm (do this before taking subset of firms that meet threshold of sufficiently many moves)
        self.movers_per_firm = bdf.loc[bdf.loc[:, 'm'] > 0, :].n_workers() / bdf.n_firms() # bdf.loc[bdf.loc[:, 'm'] > 0, :].groupby('j')['i'].nunique().mean()

        # Take subset of firms that meet threshold of sufficiently many moves
        bdf = bdf.min_moves_frame(threshold=self.min_moves_threshold, drop_returns_to_stays=self.clean_params['drop_returns_to_stays'], is_sorted=True, reset_index=True, copy=False)

        if len(bdf) == 0:
            raise ValueError("Length of dataframe is 0 after dropping firms with too few moves, consider lowering `min_moves_threshold` for tw.Attrition().")

        if False: # ncore > 1:
            # Estimate with multi-processing
            with Pool(processes=ncore) as pool:
                # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
                # Multiprocessing tqdm source: https://stackoverflow.com/a/45276885/17333120
                V = list(tqdm(pool.starmap(self._attrition_single, [(bdf, ncore, np.random.default_rng(seed)) for seed in rng.bit_generator._seed_seq.spawn(N)]), total=N))
            for res in V:
                res_non_he['fe'].append(res['non_he']['fe'])
                res_non_he['cre'].append(res['non_he']['cre'])
                res_he['fe'].append(res['he']['fe'])
                res_he['cre'].append(res['he']['cre'])
        else:
            # Estimate without multi-processing
            pbar = trange(N)
            pbar.set_description('attrition main')
            for _ in pbar:
                res = self._attrition_single(bdf=bdf, ncore=ncore, rng=rng)
                res_non_he['fe'].append(res['non_he']['fe'])
                res_non_he['cre'].append(res['non_he']['cre'])
                res_he['fe'].append(res['he']['fe'])
                res_he['cre'].append(res['he']['cre'])

        # Combine results
        self.attrition_res = {'non_he': res_non_he, 'he': res_he}

    def plot_attrition(self, line_at_movers_per_firm=True, xticks_round=1):
        '''
        Plot results from Monte Carlo simulations.

        Arguments:
            line_at_movers_per_firm (bool): if True, plot a dashed line where movers per firm in the subsample is approximately the number of movers per firm in the full sample
            xticks_round (int): how many digits to round x ticks
        '''
        if not self.attrition_res:
            warnings.warn('Must generate attrition data before results can be plotted. This can be done by running .attrition()')

        else:
            ## Get N, M ##
            # Number of estimations
            N = len(self.attrition_res['non_he']['fe'])
            # Number of attritions per estimation
            M = len(self.attrition_res['non_he']['fe'][0])
            ## Extract results ##
            # Non-HE
            non_he_var_psi_pct = np.zeros(shape=[N, M, 3])
            non_he_cov_psi_alpha_pct = np.zeros(shape=[N, M, 3])
            # HE
            he_var_psi_pct = np.zeros(shape=[N, M, 4])
            he_cov_psi_alpha_pct = np.zeros(shape=[N, M, 4])
            for i in range(N):
                for j in range(M):
                    # Non-HE
                    non_he_res_fe_dict = self.attrition_res['non_he']['fe'][i][j]
                    non_he_res_cre_dict = self.attrition_res['non_he']['cre'][i][j]
                    # Var(psi)
                    non_he_var_psi_pct[i, j, 0] = float(non_he_res_fe_dict['var_fe']) / float(non_he_res_fe_dict['var_y'])
                    non_he_var_psi_pct[i, j, 1] = float(non_he_res_fe_dict['var_ho']) / float(non_he_res_fe_dict['var_y'])
                    non_he_var_psi_pct[i, j, 2] = float(non_he_res_cre_dict['tot_var']) / float(non_he_res_cre_dict['var_y'])
                    # Cov(psi, alpha)
                    non_he_cov_psi_alpha_pct[i, j, 0] = 2 * float(non_he_res_fe_dict['cov_fe']) / float(non_he_res_fe_dict['var_y'])
                    non_he_cov_psi_alpha_pct[i, j, 1] = 2 * float(non_he_res_fe_dict['cov_ho']) / float(non_he_res_fe_dict['var_y'])
                    non_he_cov_psi_alpha_pct[i, j, 2] = 2 * float(non_he_res_cre_dict['tot_cov']) / float(non_he_res_cre_dict['var_y'])

                    # HE
                    he_res_fe_dict = self.attrition_res['he']['fe'][i][j]
                    he_res_cre_dict = self.attrition_res['he']['cre'][i][j]
                    # Var(psi)
                    he_var_psi_pct[i, j, 0] = float(he_res_fe_dict['var_fe']) / float(he_res_fe_dict['var_y'])
                    he_var_psi_pct[i, j, 1] = float(he_res_fe_dict['var_ho']) / float(he_res_fe_dict['var_y'])
                    he_var_psi_pct[i, j, 2] = float(he_res_cre_dict['tot_var']) / float(he_res_cre_dict['var_y'])
                    he_var_psi_pct[i, j, 3] = float(he_res_fe_dict['var_he']) / float(he_res_fe_dict['var_y'])

                    # Cov(psi, alpha)
                    he_cov_psi_alpha_pct[i, j, 0] = 2 * float(he_res_fe_dict['cov_fe']) / float(he_res_fe_dict['var_y'])
                    he_cov_psi_alpha_pct[i, j, 1] = 2 * float(he_res_fe_dict['cov_ho']) / float(he_res_fe_dict['var_y'])
                    he_cov_psi_alpha_pct[i, j, 2] = 2 * float(he_res_cre_dict['tot_cov']) / float(he_res_cre_dict['var_y'])
                    he_cov_psi_alpha_pct[i, j, 3] = 2 * float(he_res_fe_dict['cov_he']) / float(he_res_fe_dict['var_y'])

            # x-axis
            x_axis = np.round(100 * self.attrition_how.subset_fractions, xticks_round)
            if np.all(x_axis == x_axis.astype(int)):
                # This is necessary for the boxplots, since they don't automatically convert to integers
                x_axis = x_axis.astype(int, copy=False)
            non_he_var_psi_pct = 100 * non_he_var_psi_pct
            non_he_cov_psi_alpha_pct = 100 * non_he_cov_psi_alpha_pct
            he_var_psi_pct = 100 * he_var_psi_pct
            he_cov_psi_alpha_pct = 100 * he_cov_psi_alpha_pct

            # Flip along 1st axis so that both increasing and decreasing have the same order
            if np.max(np.diff(self.attrition_how.subset_fractions)) <= 0:
                # If decreasing
                x_axis = np.flip(x_axis)
                non_he_var_psi_pct = np.flip(non_he_var_psi_pct, axis=1)
                non_he_cov_psi_alpha_pct = np.flip(non_he_cov_psi_alpha_pct, axis=1)
                he_var_psi_pct = np.flip(he_var_psi_pct, axis=1)
                he_cov_psi_alpha_pct = np.flip(he_cov_psi_alpha_pct, axis=1)

            ## Prepare line at movers per firm
            if line_at_movers_per_firm:
                non_he_movers_per_firm = np.zeros(shape=M)
                he_movers_per_firm = np.zeros(shape=M)

                for i in range(N):
                    for j in range(M):
                        # Sum over movers per firm for all iterations
                        non_he_movers_i_j = self.attrition_res['non_he']['fe'][i][j]
                        he_movers_i_j = self.attrition_res['he']['fe'][i][j]
                        non_he_movers_per_firm[j] += (int(non_he_movers_i_j['n_movers']) / int(non_he_movers_i_j['n_firms'])) # float(non_he_movers_i_j['movers_per_firm'])
                        he_movers_per_firm[j] += (int(he_movers_i_j['n_movers']) / int(he_movers_i_j['n_firms'])) # float(he_movers_i_j['movers_per_firm'])
                # Take average
                non_he_movers_per_firm /= N
                he_movers_per_firm /= N
                # Increase by 300%, because we are approximating FIXME this is too much
                non_he_movers_per_firm *= 1.5
                he_movers_per_firm *= 1.5

                # Reverse order so that both increasing and decreasing have the same order
                if np.max(np.diff(self.attrition_how.subset_fractions)) <= 0:
                    # If decreasing
                    non_he_movers_per_firm = np.flip(non_he_movers_per_firm, axis=0)
                    he_movers_per_firm = np.flip(he_movers_per_firm, axis=0)

                if self.movers_per_firm >= np.max(non_he_movers_per_firm):
                    non_he_movers_per_firm_line = np.max(x_axis)
                elif self.movers_per_firm <= np.min(non_he_movers_per_firm):
                    non_he_movers_per_firm_line = np.min(x_axis)
                else:
                    # Find where movers per firm in subset approximates movers per firm in entire dataset
                    for i, non_he_movers_per_firm_i in enumerate(non_he_movers_per_firm[1:]):
                        if self.movers_per_firm < non_he_movers_per_firm_i:
                            frac = (self.movers_per_firm - non_he_movers_per_firm[i]) / (non_he_movers_per_firm_i - non_he_movers_per_firm[i])
                            non_he_movers_per_firm_line = x_axis[i] + frac * (x_axis[i + 1] - x_axis[i])
                            break

                if self.movers_per_firm >= np.max(he_movers_per_firm):
                    he_movers_per_firm_line = np.max(x_axis)
                elif self.movers_per_firm <= np.min(he_movers_per_firm):
                    he_movers_per_firm_line = np.min(x_axis)
                else:
                    # Find where movers per firm in subset approximates movers per firm in entire dataset
                    for i, he_movers_per_firm_i in enumerate(he_movers_per_firm[1:]):
                        if self.movers_per_firm < he_movers_per_firm_i:
                            frac = (self.movers_per_firm - he_movers_per_firm[i]) / (he_movers_per_firm_i - he_movers_per_firm[i])
                            he_movers_per_firm_line = x_axis[i] + frac * (x_axis[i + 1] - x_axis[i])
                            break

            ### Plot figures ###
            # Source: https://stackoverflow.com/a/68209152/17333120
            fig = plt.figure(constrained_layout=True, dpi=150)
            subfigs = fig.subfigures(nrows=2, ncols=1)
            ## Firm effects ##
            subfigs[0].suptitle('Firm effects', x=0.545)
            axs = subfigs[0].subplots(nrows=1, ncols=2)
            # Firm effects (non-HE)
            axs[0].plot(x_axis, non_he_var_psi_pct[:, :, 0].mean(axis=0), color='C0', label='FE')
            axs[0].plot(x_axis, non_he_var_psi_pct[:, :, 1].mean(axis=0), color='C1', label='HO')
            axs[0].plot(x_axis, non_he_var_psi_pct[:, :, 2].mean(axis=0), color='C3', label='CRE')
            if line_at_movers_per_firm:
                axs[0].axvline(non_he_movers_per_firm_line, color='k', linestyle='--')
            axs[0].set_title('Connected set')
            axs[0].set_xlabel('Share of Movers Kept (%)')
            axs[0].set_ylabel('Share of Variance (%)')
            axs[0].grid()

            # Firm effects (HE)
            axs[1].plot(x_axis, he_var_psi_pct[:, :, 0].mean(axis=0), color='C0', label='FE')
            axs[1].plot(x_axis, he_var_psi_pct[:, :, 1].mean(axis=0), color='C1', label='HO')
            axs[1].plot(x_axis, he_var_psi_pct[:, :, 3].mean(axis=0), color='C2', label='HE')
            axs[1].plot(x_axis, he_var_psi_pct[:, :, 2].mean(axis=0), color='C3', label='CRE')
            if line_at_movers_per_firm:
                axs[1].axvline(he_movers_per_firm_line, color='k', linestyle='--')
            axs[1].set_title('Leave-one-out set')
            axs[1].set_xlabel('Share of Movers Kept (%)')
            axs[1].set_ylabel(' ')
            axs[1].grid()

            ## Sorting ##
            subfigs[1].suptitle('Sorting', x=0.545)
            axs = subfigs[1].subplots(nrows=1, ncols=2)
            # Sorting (non-HE)
            axs[0].plot(x_axis, non_he_cov_psi_alpha_pct[:, :, 0].mean(axis=0), color='C0', label='FE')
            axs[0].plot(x_axis, non_he_cov_psi_alpha_pct[:, :, 1].mean(axis=0), color='C1', label='HO')
            axs[0].plot(x_axis, non_he_cov_psi_alpha_pct[:, :, 2].mean(axis=0), color='C3', label='CRE')
            if line_at_movers_per_firm:
                axs[0].axvline(non_he_movers_per_firm_line, color='k', linestyle='--')
            axs[0].set_title('Connected set')
            axs[0].set_xlabel('Share of Movers Kept (%)')
            axs[0].set_ylabel('Share of Variance (%)')
            axs[0].grid()

            # Sorting (HE)
            axs[1].plot(x_axis, he_cov_psi_alpha_pct[:, :, 0].mean(axis=0), color='C0', label='FE')
            axs[1].plot(x_axis, he_cov_psi_alpha_pct[:, :, 1].mean(axis=0), color='C1', label='HO')
            axs[1].plot(x_axis, he_cov_psi_alpha_pct[:, :, 3].mean(axis=0), color='C2', label='HE')
            axs[1].plot(x_axis, he_cov_psi_alpha_pct[:, :, 2].mean(axis=0), color='C3', label='CRE')
            if line_at_movers_per_firm:
                axs[1].axvline(he_movers_per_firm_line, color='k', linestyle='--')
            axs[1].set_title('Leave-one-out set')
            axs[1].set_xlabel('Share of Movers Kept (%)')
            axs[1].set_ylabel(' ')
            axs[1].grid()

            # Shared legend (source: https://stackoverflow.com/a/46921590/17333120)
            handles, labels = axs[1].get_legend_handles_labels()
            subfigs[1].legend(handles, labels, loc=(1.02, 0.78))
            plt.show()

            ### Plot boxplots ###
            ## Firm effects ##
            # Firm effects (non-HE set)
            fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
            subtitles = ['FE', 'FE-HO', 'CRE']

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(non_he_var_psi_pct[:, :, i], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[i])
            fig.suptitle('Firm effects (connected set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Firm Effects: Share of Variance (%)')
            fig.tight_layout()
            # NOTE: must use plt.show(), fig.show() raises a warning in Jupyter Notebook (source: https://stackoverflow.com/a/52827912/17333120)
            plt.show()

            # Firm effects (HE set)
            fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True)
            subtitles = ['FE', 'FE-HO', 'CRE', 'FE-HE']
            # Change order because data is FE, FE-HO, CRE, FE-HE but want FE, FE-HO, FE-HE, CRE
            order = [0, 1, 3, 2]

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(he_var_psi_pct[:, :, order[i]], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[order[i]])
            fig.suptitle('Firm effects (leave-one-out set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Firm Effects: Share of Variance (%)')
            fig.tight_layout()
            plt.show()

            ## Sorting ##
            # Sorting (non-HE set)
            fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)
            subtitles = ['FE', 'FE-HO', 'CRE']

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(non_he_cov_psi_alpha_pct[:, :, i], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[i])
            fig.suptitle('Sorting (connected set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Sorting: Share of Variance (%)')
            fig.tight_layout()
            plt.show()

            # Sorting (HE set)
            fig, ax = plt.subplots(nrows=1, ncols=4, sharey=True)
            subtitles = ['FE', 'FE-HO', 'CRE', 'FE-HE']
            # Change order because data is FE, FE-HO, CRE, FE-HE but want FE, FE-HO, FE-HE, CRE
            order = [0, 1, 3, 2]

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(he_cov_psi_alpha_pct[:, :, order[i]], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[order[i]])
            fig.suptitle('Sorting (leave-one-out set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Sorting: Share of Variance (%)')
            fig.tight_layout()
            plt.show()
