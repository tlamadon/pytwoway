'''
Classes for estimating attrition using increasing (building up from a fixed set of firms) or decreasing (with varying sets of firms) fractions of movers.
'''
'''
NOTE: use classes rather than nested functions because nested functions cannot be pickled (source: https://stackoverflow.com/a/12022055/17333120).
'''
import warnings
import numpy as np
import bipartitepandas as bpd

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

    def _gen_subsets(self, bdf, clean_params=None, rng=None):
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
        drop_single_stayers = clean_params['drop_single_stayers']
        if drop_single_stayers:
            # We don't want to drop single stayers during cleaning, because if we do then we can't keep track of the stayers that need to be dropped
            clean_params['drop_single_stayers'] = False

        # Get subset fractions
        fracs = self.subset_fractions

        # Must reset id_reference_dict
        bdf = bdf._reset_id_reference_dict(include=True)

        # Get firm ids in base subset
        # NOTE: must consider movers and stayers, because some firms might have only stayers
        fids_init = set(bdf.unique_ids('j'))
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

        if drop_single_stayers:
            ## Drop stayers who have <= 1 observation weight ##
            # NOTE: we make sure to use original worker ids
            worker_m = subset_1.get_worker_m(is_sorted=True)
            original_i = 'original_i'
            if original_i not in subset_1_orig_ids.columns:
                # If no changes to this column
                original_i = 'i'
            if subset_1._col_included('w'):
                stayers_weight = subset_1_orig_ids.loc[~worker_m, [original_i, 'w']].groupby(original_i, sort=False)['w'].transform('sum').to_numpy()
            else:
                stayers_weight = subset_1_orig_ids.loc[~worker_m, [original_i, 'j']].groupby(original_i, sort=False)['j'].transform('size').to_numpy()
            stayers_to_drop = set(subset_1_orig_ids.loc[~worker_m, original_i].to_numpy()[stayers_weight <= 1])

            # Update wid_drops_1
            wid_drops_1 = wid_drops_1.union(stayers_to_drop)

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

        if drop_single_stayers:
            ## Drop stayers who have <= 1 observation weight ##
            # NOTE: must recompute this, because some of the movers dropped at the start might have turned into stayers given the reduced set of firms in the connected set
            worker_m = subset_2.get_worker_m(is_sorted=True)

            if subset_2._col_included('w'):
                stayers_weight = subset_2.loc[~worker_m, ['i', 'w']].groupby('i', sort=False)['w'].transform('sum').to_numpy()
            else:
                stayers_weight = subset_2.loc[~worker_m, ['i', 'j']].groupby('i', sort=False)['j'].transform('size').to_numpy()
            stayers_to_drop = set(subset_2.loc[~worker_m, 'i'].to_numpy()[stayers_weight <= 1])

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

            if drop_single_stayers:
                ## Drop stayers who have <= 1 observation weight ##
                wids_to_drop_full = wids_to_drop.union(stayers_to_drop)
            else:
                wids_to_drop_full = wids_to_drop

            ret_wid = {
                True: wids_to_drop_full,
                False: None
            }
            return_lst.append((ret_fid[len(fids_to_drop) > 0], ret_wid[len(wids_to_drop_full) > 0]))

        if drop_single_stayers:
            # Restore clean parameters
            clean_params['drop_single_stayers'] = True

        return return_lst

    def _update_clean_params(self, clean_params):
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

    def _gen_subsets(self, bdf, clean_params=None, rng=None):
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

    def _update_clean_params(self, clean_params):
        '''
        Used for AttritionIncreasing(), does nothing for AttritionDecreasing().

        Arguments:
            clean_params (ParamsDict): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters.

        Returns:
            (ParamsDict) clean_params
        '''
        return clean_params