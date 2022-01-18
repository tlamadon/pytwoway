'''
Class for Attrition plots
'''
from multiprocessing import Pool, Value
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from bipartitepandas import ParamsDict, to_list, logger_init, cluster_params, clean_params
import pytwoway as tw
from tqdm import tqdm
import warnings

# Define default parameter dictionary
_attrition_params_default = ParamsDict({
    'type_and_subsets': (('increasing', np.linspace(0.1, 0.5, 5)), 'type', tuple,
        '''
            (default=('increasing', np.linspace(0.1, 0.5, 5))) How to attrition data (either 'increasing' or 'decreasing'), and subsets to consider (both are required because switching type requires swapping the order of the subsets).
        '''),
    'min_moves_threshold': (15, 'type', int,
        '''
            (default=15) Minimum number of moves required to keep a firm.
        '''),
    'copy': (False, 'type', bool,
        '''
            (default=False) If False, avoid copy.
        ''')
})

def attrition_params(update_dict={}):
    '''
    Dictionary of default attrition_params.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of attrition_params
    '''
    new_dict = _attrition_params_default.copy()
    new_dict.update(update_dict)
    return new_dict

class Attrition:
    '''
    Class of Attrition, which generates attrition plots using bipartite labor data.

    Arguments:
        bdf (BipartiteBase): bipartite dataframe
        attrition_params (ParamsDict): dictionary of parameters for attrition. Run tw.attrition_params().describe_all() for descriptions of all valid parameters.
        fe_params (ParamsDict): dictionary of parameters for FE estimation. Run tw.fe_params().describe_all() for descriptions of all valid parameters.
        cre_params (ParamsDict): dictionary of parameters for CRE estimation. Run tw.cre_params().describe_all() for descriptions of all valid parameters.
        cluster_params (ParamsDict): dictionary of parameters for clustering in CRE estimation. Run bpd.cluster_params().describe_all() for descriptions of all valid parameters.
        clean_params (ParamsDict): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters.
    '''

    def __init__(self, bdf, attrition_params=attrition_params(), fe_params=tw.fe_params(), cre_params=tw.cre_params(), cluster_params=cluster_params(), clean_params=clean_params()):
        ##### Save attributes
        ## Basic attributes
        # Data
        if attrition_params['copy']:
            self.bdf = bdf.copy()
        else:
            self.bdf = bdf

        # Type and subsets
        self.attrition_type = attrition_params['type_and_subsets'][0]
        self.subsets = attrition_params['type_and_subsets'][1]

        # Attrition function (increasing or decreasing)
        attrition_fn_dict = {
            'increasing': self._attrition_increasing,
            'decreasing': self._attrition_decreasing
        }
        self.attrition_fn = attrition_fn_dict[self.attrition_type]

        # Make sure subsets are weakly increasing/decreasing
        if (self.attrition_type == 'increasing') and (np.min(np.diff(np.array(self.subsets))) < 0):
            raise NotImplementedError('Subsets must be weakly increasing for increasing subsets.')
        elif (self.attrition_type == 'decreasing') and (np.min(np.diff(np.array(self.subsets))) > 0):
            raise NotImplementedError('Subsets must be weakly decreasing for decreasing subsets.')

        # Prevent plotting until results exist
        self.attrition_res = False

        #### Parameter dictionaries
        ### Save parameter dictionaries
        self.attrition_params = attrition_params.copy()
        self.fe_params = fe_params.copy()
        self.cre_params = cre_params.copy()
        self.cluster_params = cluster_params.copy()
        self.clean_params = clean_params.copy()

        ### Update parameter dictionaries
        ## Clean
        self.clean_params['is_sorted'] = True
        self.clean_params['force'] = False
        self.clean_params['copy'] = False

        ## Cluster
        self.cluster_params['clean_params'] = self.clean_params.copy()
        self.cluster_params['is_sorted'] = True
        self.cluster_params['copy'] = False

        ## Non-HE and HE parameter dictionaries
        # FE
        self.fe_params_non_he = self.fe_params.copy()
        self.fe_params_non_he['he'] = False
        self.fe_params_he = self.fe_params.copy()
        self.fe_params_he['he'] = True

        # Clean
        self.clean_params_non_he = self.clean_params.copy()
        self.clean_params_non_he['connectedness'] = 'connected'
        self.clean_params_he = self.clean_params.copy()
        self.clean_params_he['connectedness'] = 'leave_one_observation_out'

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
    def _attrition_interior(self, wids_to_drop, fe_params=tw.fe_params(), cre_params=tw.cre_params(), cluster_params=cluster_params(), clean_params=clean_params()):
        '''
        Estimate all parameters of interest. This is the interior function to attrition_single.

        Arguments:
            wids_to_drop (set or None): if set, worker ids to drop from self.subset_2; if None, subset is saved as self.subset_1
            fe_params (ParamsDict): dictionary of parameters for FE estimation. Run tw.fe_params().describe_all() for descriptions of all valid parameters.
            cre_params (ParamsDict): dictionary of parameters for CRE estimation. Run tw.cre_params().describe_all() for descriptions of all valid parameters.
            cluster_params (ParamsDict): dictionary of parameters for clustering in CRE estimation. Run bpd.cluster_params().describe_all() for descriptions of all valid parameters.
            clean_params (ParamsDict): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters.

        Returns:
            (dict): {'fe': FE results, 'cre': CRE results}
        '''
        # logger_init(bdf) # This stops a weird logging bug that stops multiprocessing from working
        if wids_to_drop is not None:
            # Drop ids and clean data (NOTE: this does not require a copy)
            bdf = self.subset_2.drop_ids('i', wids_to_drop, copy=False)._reset_attributes(columns_contig=True, connected=False, no_na=False, no_duplicates=False, i_t_unique=False, no_returns=False).clean_data(clean_params)
        else:
            bdf = self.subset_1
            del self.subset_1
        ## Estimate FE model
        fe_estimator = tw.FEEstimator(bdf, fe_params)
        fe_estimator.fit()
        ## Estimate CRE model
        # Cluster
        bdf = bdf.cluster(cluster_params)
        # Estimate
        cre_estimator = tw.CREEstimator(bdf.get_es(move_to_worker=False, is_sorted=True).get_cs(), cre_params)
        cre_estimator.fit()

        return {'fe': fe_estimator.res, 'cre': cre_estimator.res}

    def _attrition_increasing(self, clean_params=clean_params(), rng=np.random.default_rng()):
        '''
        First, keep only firms that have at minimum `threshold` many movers. Then take a random subset of subsets[0] percent of remaining movers. Constructively rebuild the data to reach each subsequent value of subsets. Return the worker ids to drop to attain these subsets as an iterator.

        Arguments:
            clean_params (ParamsDict): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters.
            rng (np.random.Generator): NumPy random number generator

        Returns:
            (list of (set or None)): for each entry in list, if set, worker ids to drop from self.subset_2; if None, subset is saved as self.subset_1
        '''
        # Get subsets
        subsets = self.subsets

        # Must reset id_reference_dict
        self.bdf = self.bdf._reset_id_reference_dict(include=True)

        # Worker ids in base subset
        wids_init = self.bdf.loc[self.bdf.loc[:, 'm'].to_numpy() > 0, :].unique_ids('i')

        for i in range(10):
            # Sometimes we draw so few firms that it's likely to end up with no observations left, so redraw until success
            ## Draw first subset
            # Number of wids to drop
            n_wid_drops_1 = int(np.floor((1 - subsets[0]) * len(wids_init)))
            # Draw wids to drop
            wid_drops_1 = set(rng.choice(wids_init, size=n_wid_drops_1, replace=False))
            try:
                # NOTE: this requires the copy if there are returners
                self.subset_1 = self.bdf.drop_ids('i', wid_drops_1, drop_returns_to_stays=clean_params['drop_returns_to_stays'], is_sorted=True, reset_index=True, copy=True)._reset_attributes(columns_contig=True, connected=True, no_na=False, no_duplicates=False, i_t_unique=False, no_returns=False).clean_data(clean_params)
                break
            except ValueError as v:
                if i == 9:
                    # Don't fail unless it's the last loop
                    raise ValueError(v)

        ret_lst = [None]

        subset_1_orig_ids = self.subset_1.original_ids(copy=False)

        # Get list of all valid firms
        valid_firms = []
        for j_subcol in to_list(self.bdf.reference_dict['j']):
            original_j = 'original_' + j_subcol
            if original_j not in subset_1_orig_ids.columns:
                # If no changes to this column
                original_j = j_subcol
            valid_firms += list(subset_1_orig_ids[original_j].unique())
        valid_firms = set(valid_firms)

        # Take all data for list of firms in smallest subset (NOTE: this requires the copy if there are returners)
        self.subset_2 = self.bdf.keep_ids('j', valid_firms, drop_returns_to_stays=clean_params['drop_returns_to_stays'], is_sorted=True, reset_index=True, copy=True)
        # Clear id_reference_dict since it is no longer necessary
        self.subset_2._reset_id_reference_dict()

        # Determine which wids (for movers) can still be drawn
        all_valid_wids = set(self.subset_2.loc[self.subset_2.loc[:, 'm'].to_numpy() > 0, :].unique_ids('i'))

        original_i = 'original_i'
        if original_i not in subset_1_orig_ids.columns:
            # If no changes to i column
            original_i = 'i'
        wids_to_drop = all_valid_wids.difference(set(subset_1_orig_ids.loc[subset_1_orig_ids.loc[:, 'm'].to_numpy() > 0, original_i].unique()))
        del subset_1_orig_ids

        for i, subset_pct in enumerate(subsets[1:]):
            # Each step, drop fewer wids
            n_wid_draws_i = min(int(np.round((1 - subset_pct) * len(all_valid_wids), 1)), len(wids_to_drop))
            if n_wid_draws_i > 0:
                wids_to_drop = set(rng.choice(list(wids_to_drop), size=n_wid_draws_i, replace=False))
            else:
                warnings.warn('Attrition plot does not change at iteration {}'.format(i))

            ret_lst.append(wids_to_drop)

        return ret_lst

    def _attrition_decreasing(self, clean_params=clean_params(), rng=np.random.default_rng()):
        '''
        First, keep only firms that have at minimum `threshold` many movers. Then take a random subset of subsets[0] percent of remaining movers. Deconstruct the data to reach each subsequent value of subsets. Return the worker ids to drop to attain these subsets as an iterator.

        Arguments:
            clean_params (ParamsDict): used for _attrition_increasing(), does nothing for _attrition_decreasing()
            rng (np.random.Generator): NumPy random number generator

        Returns:
            (list): list of worker ids to drop from self.subset_2
        '''
        # bdf = bdf.copy()
        # bdf._reset_id_reference_dict() # Clear id_reference_dict since it is not necessary

        # For consistency with _attrition_increasing()
        self.subset_2 = self.bdf

        # Worker ids in base subset
        wids_movers = set(self.bdf.loc[self.bdf.loc[:, 'm'].to_numpy() > 0, :].unique_ids('i'))
        # wids_stayers = list(self.bdf.loc[self.bdf.loc[:, 'm'].to_numpy() == 0, :].unique_ids('i'))

        # # Drop m since it can change for leave-one-out components
        # bdf.drop('m')

        relative_drop_fraction = 1 - self.subsets / (np.concatenate([[1], self.subsets]))[:-1]
        wids_to_drop = set()
        ret_lst = []

        for i, drop_frac in enumerate(relative_drop_fraction):
            n_wid_draws_i = min(int(np.round(drop_frac * len(wids_movers), 1)), len(wids_movers))
            if n_wid_draws_i > 0:
                wid_draws_i = set(rng.choice(list(wids_movers), size=n_wid_draws_i, replace=False))
                wids_movers = wids_movers.difference(wid_draws_i)
                wids_to_drop = wids_to_drop.union(wid_draws_i)
            else:
                warnings.warn('Attrition plot does not change at iteration {}'.format(i))

            ret_lst.append(wids_to_drop)

            # subset_i = bdf.keep_ids('i', wids_movers + wids_stayers, copy=True)._reset_id_reference_dict()._reset_attributes(columns_contig=True, connected=True, no_na=False, no_duplicates=False, i_t_unique=False, no_returns=False)

        return ret_lst

    def _attrition_single(self, ncore=1, rng=np.random.default_rng()):
        '''
        Run attrition estimations of TwoWay to estimate parameters given fraction of movers remaining. This is the interior function to attrition.

        Arguments:
            ncore (int): number of cores to use
            rng (NumPy RandomState): NumPy RandomState object

        Returns:
            res_all (dict of dicts of lists): in the first dictionary we choose 'non_he' or 'he'; in the second dictionary we choose 'fe' or 'cre'; and finally, we are given a list of results for each attrition percentage.
        '''
        ## Create lists to save results
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

            # Create iterator
            attrition_iterator = self.attrition_fn(clean_params=clean_params, rng=rng)

            if self.attrition_type == 'increasing':
                # Once the initial connectedness has been computed, larger subsets are connected by construction
                clean_params = clean_params.copy()
                clean_params['connectedness'] = None

            def _attrition_interior_params():
                N = len(self.subsets)
                seeds = rng.bit_generator._seed_seq.spawn(N)
                ret_lst = []
                for i, wids_to_drop in enumerate(attrition_iterator):
                    # Update seeds
                    rng_i = np.random.default_rng(seeds[i])
                    sub_fe_params = fe_params.copy()
                    sub_fe_params.update({'rng': rng_i})
                    sub_cre_params = self.cre_params.copy()
                    sub_cre_params.update({'rng': rng_i})
                    # Append parameters
                    ret_lst.append((wids_to_drop, sub_fe_params, sub_cre_params, cluster_params, clean_params))
                return ret_lst

            if ncore > 1:
                # Estimate with multi-processing
                with Pool(processes=ncore) as pool:
                    V = pool.starmap(self._attrition_interior, _attrition_interior_params())
                for res in enumerate(V):
                    res_all[non_he_he]['fe'].append(res[1]['fe'])
                    res_all[non_he_he]['cre'].append(res[1]['cre'])
            else:
                # Estimate without multi-processing
                for attrition_subparams in _attrition_interior_params():
                    res = self._attrition_interior(*attrition_subparams)
                    res_all[non_he_he]['fe'].append(res['fe'])
                    res_all[non_he_he]['cre'].append(res['cre'])

            if (self.attrition_type == 'increasing') and (ncore > 1):
                del self.subset_1
            del self.subset_2

        return res_all

    def attrition(self, N=10, ncore=1, rng=np.random.default_rng(None)):
        '''
        Run Monte Carlo on attrition estimations of TwoWay to estimate variance of parameter estimates given fraction of movers remaining. Note that this overwrites the stored dataframe, meaning if you want to run attrition with different threshold number of movers, you will have to create multiple Attrition objects, or alternatively, run this method with an increasing threshold for each iteration.

        Arguments:
            N (int): number of simulations
            ncore (int): number of cores to use
            rng (np.random.Generator): NumPy random number generator. This overrides the random number generators for FE and CRE.

        Returns:
            res_all (dict of dicts of lists of lists): in the first dictionary we choose 'non_he' or 'he'; in the second dictionary we choose 'fe' or 'cre'; then, we are given a list of results for each Monte Carlo simulation; and finally, for a particular Monte Carlo simulation, we are given a list of results for each attrition percentage.
        '''
        ## Create lists to save results
        # For non-HE
        res_non_he = {'fe': [], 'cre': []}
        # For HE
        res_he = {'fe': [], 'cre': []}

        # Take subset of firms that meet threshold
        self.bdf = self.bdf.min_moves_frame(threshold=self.attrition_params['min_moves_threshold'], drop_returns_to_stays=self.clean_params['drop_returns_to_stays'], is_sorted=True, reset_index=True, copy=False)

        if False: # ncore > 1:
            # Estimate with multi-processing
            with Pool(processes=ncore) as pool:
                # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
                # Multiprocessing tqdm source: https://stackoverflow.com/a/45276885/17333120
                V = list(tqdm(pool.starmap(self._attrition_single, [(ncore, np.random.default_rng(seed)) for seed in rng.bit_generator._seed_seq.spawn(N)]), total=N))
            for res in enumerate(V):
                res_non_he['fe'].append(res[1]['non_he']['fe'])
                res_non_he['cre'].append(res[1]['non_he']['cre'])
                res_he['fe'].append(res[1]['he']['fe'])
                res_he['cre'].append(res[1]['he']['cre'])
        else:
            # Estimate without multi-processing
            for seed in tqdm(rng.bit_generator._seed_seq.spawn(N)):
                rng_i = np.random.default_rng(seed)
                res = self._attrition_single(ncore=ncore, rng=rng_i)
                res_non_he['fe'].append(res['non_he']['fe'])
                res_non_he['cre'].append(res['non_he']['cre'])
                res_he['fe'].append(res['he']['fe'])
                res_he['cre'].append(res['he']['cre'])

        # Combine results
        self.attrition_res = {'non_he': res_non_he, 'he': res_he}

    def plot_attrition(self):
        '''
        Plot results from Monte Carlo simulations.
        '''
        if not self.attrition_res:
            warnings.warn('Must generate attrition data before results can be plotted. This can be done by running .attrition()')

        else:
            ## Get N, M
            # Number of estimations
            N = len(self.attrition_res['non_he']['fe'])
            # Number of attritions per estimation
            M = len(self.attrition_res['non_he']['fe'][0])
            ## Extract results
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
            x_axis = np.round(100 * self.subsets).astype(int)
            non_he_var_psi_pct = 100 * non_he_var_psi_pct
            non_he_cov_psi_alpha_pct = 100 * non_he_cov_psi_alpha_pct
            he_var_psi_pct = 100 * he_var_psi_pct
            he_cov_psi_alpha_pct = 100 * he_cov_psi_alpha_pct

            # Flip along 1st axis so both increasing and decreasing have the same order
            if self.attrition_type == 'decreasing':
                x_axis = np.flip(x_axis)
                non_he_var_psi_pct = np.flip(non_he_var_psi_pct, axis=1)
                non_he_cov_psi_alpha_pct = np.flip(non_he_cov_psi_alpha_pct, axis=1)
                he_var_psi_pct = np.flip(he_var_psi_pct, axis=1)
                he_cov_psi_alpha_pct = np.flip(he_cov_psi_alpha_pct, axis=1)

            # Plot figures
            # Firm effects (non-HE)
            plt.plot(x_axis, non_he_var_psi_pct[:, :, 0].mean(axis=0), label='FE')
            plt.plot(x_axis, non_he_var_psi_pct[:, :, 1].mean(axis=0), label='HO')
            plt.plot(x_axis, non_he_var_psi_pct[:, :, 2].mean(axis=0), label='CRE')
            plt.title('Firm effects (connected set)')
            plt.xlabel('Share of Movers Kept (%)')
            plt.ylabel('Firm Effects: Share of Variance (%)')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()
            # Firm effects (HE)
            plt.plot(x_axis, he_var_psi_pct[:, :, 0].mean(axis=0), label='FE')
            plt.plot(x_axis, he_var_psi_pct[:, :, 1].mean(axis=0), label='HO')
            plt.plot(x_axis, he_var_psi_pct[:, :, 2].mean(axis=0), label='CRE')
            plt.plot(x_axis, he_var_psi_pct[:, :, 3].mean(axis=0), label='HE')
            plt.title('Firm effects (leave-one-out connected set)')
            plt.xlabel('Share of Movers Kept (%)')
            plt.ylabel('Firm Effects: Share of Variance (%)')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

            # Sorting (non-HE)
            plt.plot(x_axis, non_he_cov_psi_alpha_pct[:, :, 0].mean(axis=0), label='FE')
            plt.plot(x_axis, non_he_cov_psi_alpha_pct[:, :, 1].mean(axis=0), label='HO')
            plt.plot(x_axis, non_he_cov_psi_alpha_pct[:, :, 2].mean(axis=0), label='CRE')
            plt.title('Sorting (connected set)')
            plt.xlabel('Share of Movers Kept (%)')
            plt.ylabel('Sorting: Share of Variance (%)')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()
            # Sorting (HE)
            plt.plot(x_axis, he_cov_psi_alpha_pct[:, :, 0].mean(axis=0), label='FE')
            plt.plot(x_axis, he_cov_psi_alpha_pct[:, :, 1].mean(axis=0), label='HO')
            plt.plot(x_axis, he_cov_psi_alpha_pct[:, :, 2].mean(axis=0), label='CRE')
            plt.plot(x_axis, he_cov_psi_alpha_pct[:, :, 3].mean(axis=0), label='HE')
            plt.title('Sorting (leave-one-out connected set)')
            plt.xlabel('Share of Movers Kept (%)')
            plt.ylabel('Sorting: Share of Variance (%)')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

            # Plot boxplots
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
            order = [0, 1, 3, 2] # Because data is FE, FE-HO, CRE, FE-HE

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(he_var_psi_pct[:, :, order[i]], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[order[i]])
            fig.suptitle('Firm effects (leave-one-out connected set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Firm Effects: Share of Variance (%)')
            fig.tight_layout()
            plt.show()

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
            order = [0, 1, 3, 2] # Because data is FE, FE-HO, CRE, FE-HE

            for i, row in enumerate(ax):
                # for col in row:
                row.boxplot(he_cov_psi_alpha_pct[:, :, order[i]], labels=x_axis, showfliers=False)
                row.grid()
                row.set_title(subtitles[order[i]])
            fig.suptitle('Sorting (leave-one-out connected set)')
            fig.supxlabel('Share of Movers Kept (%)')
            fig.supylabel('Sorting: Share of Variance (%)')
            fig.tight_layout()
            plt.show()
