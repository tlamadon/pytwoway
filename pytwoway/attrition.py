'''
Class for attrition estimation and plotting.
'''
from tqdm.auto import tqdm, trange
# import itertools
from multiprocessing import Pool
import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
import bipartitepandas as bpd
# from bipartitepandas.util import to_list, logger_init
import pytwoway as tw

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
        attrition_how (tw.attrition_utils.AttritionIncreasing() or tw.attrition_utils.AttritionDecreasing()): instance of AttritionIncreasing() or AttritionDecreasing(), used to specify if attrition should use increasing (building up from a fixed set of firms) or decreasing (with varying sets of firms) fractions of movers; None is equivalent to AttritionIncreasing()
        fe_params (ParamsDict or None): dictionary of parameters for FE estimation. Run tw.fe_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.fe_params().
        cre_params (ParamsDict or None): dictionary of parameters for CRE estimation. Run tw.cre_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.cre_params().
        cluster_params (ParamsDict or None): dictionary of parameters for clustering in CRE estimation. Run bpd.cluster_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.cluster_params().
        clean_params (ParamsDict or None): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.clean_params().
    '''

    def __init__(self, min_moves_threshold=15, attrition_how=None, fe_params=None, cre_params=None, cluster_params=None, clean_params=None):
        if attrition_how is None:
            attrition_how = tw.attrition_utils.AttritionIncreasing()
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
        self.attrition_res = None

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
        self.clean_params_he['drop_single_stayers'] = True

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
        bdf = bdf._reset_attributes(columns_contig=True, connected=True, no_na=False, no_duplicates=False, i_t_unique=False, no_returns=False).clean(clean_params)
        ## Estimate CRE model ##
        # Cluster
        bdf = bdf.cluster(cluster_params, rng=rng)
        # Estimate
        cre_estimator = tw.CREEstimator(bdf.to_eventstudy(move_to_worker=False, is_sorted=True, copy=False).get_cs(copy=False), cre_params)
        cre_estimator.fit(rng)
        cre_res = cre_estimator.res
        del cre_estimator

        # Delete time and cluster column(s)
        bdf = bdf.drop(['t', 'g'], axis=1, inplace=True, allow_optional=True)

        ## Estimate FE model ##
        fe_estimator = tw.FEEstimator(bdf, fe_params)
        fe_estimator.fit(rng)
        fe_res = fe_estimator.res
        del bdf, fe_estimator

        return {'fe': fe_res, 'cre': cre_res}

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
            subsets = self.attrition_how._gen_subsets(bdf=bdf, clean_params=clean_params, rng=rng)

            # Update clean_params
            clean_params = self.attrition_how._update_clean_params(clean_params)

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
                    pbar = tqdm(_scramble([(bdf, *attrition_subparams) for attrition_subparams in attrition_params]), total=N)
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

    def _combine_res(self, line_at_movers_per_firm=True, xticks_round=1):
        '''
        Combine attrition results into NumPy Arrays.

        Arguments:
            line_at_movers_per_firm (bool): if True, plot a dashed line where movers per firm in the subsample is approximately the number of movers per firm in the full sample
            xticks_round (int): how many digits to round x ticks

        Returns:
            (tuple): if line_at_movers_per_firm=True, (non_he_var_psi_pct, he_var_psi_pct, non_he_cov_psi_alpha_pct, he_cov_psi_alpha_pct, non_he_movers_per_firm_line, he_movers_per_firm_line, x_axis); if line_at_movers_per_firm=False, (non_he_var_psi_pct, he_var_psi_pct, non_he_cov_psi_alpha_pct, he_cov_psi_alpha_pct, x_axis)
        '''
        if self.attrition_res is None:
            raise AttributeError('Attribute attrition_res does not exist - must generate attrition data before results can be plotted. This can be done by running .attrition()')

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
            # Increase by 5%, because we are approximating
            non_he_movers_per_firm *= 1.05
            he_movers_per_firm *= 1.05

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

            return (non_he_var_psi_pct, he_var_psi_pct, non_he_cov_psi_alpha_pct, he_cov_psi_alpha_pct, non_he_movers_per_firm_line, he_movers_per_firm_line, x_axis)

        return (non_he_var_psi_pct, he_var_psi_pct, non_he_cov_psi_alpha_pct, he_cov_psi_alpha_pct, x_axis)

    def plots(self, line_at_movers_per_firm=True, xticks_round=1):
        '''
        Generate attrition result plots.

        Arguments:
            line_at_movers_per_firm (bool): if True, plot a dashed line where movers per firm in the subsample is approximately the number of movers per firm in the full sample
            xticks_round (int): how many digits to round x ticks
        '''
        # Extract results
        non_he_var_psi_pct, he_var_psi_pct, non_he_cov_psi_alpha_pct, he_cov_psi_alpha_pct, non_he_movers_per_firm_line, he_movers_per_firm_line, x_axis = self._combine_res(line_at_movers_per_firm=line_at_movers_per_firm, xticks_round=xticks_round)

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
        axs[0].set_title('Connected set', fontsize=9)
        axs[0].set_xlabel('Share of Movers Kept (%)', fontsize=7)
        axs[0].set_ylabel('Share of Variance (%)', fontsize=7)
        axs[0].tick_params(axis='x', labelsize=7)
        axs[0].tick_params(axis='y', labelsize=5)
        axs[0].grid()

        # Firm effects (HE)
        axs[1].plot(x_axis, he_var_psi_pct[:, :, 0].mean(axis=0), color='C0', label='FE')
        axs[1].plot(x_axis, he_var_psi_pct[:, :, 1].mean(axis=0), color='C1', label='HO')
        axs[1].plot(x_axis, he_var_psi_pct[:, :, 3].mean(axis=0), color='C2', label='HE')
        axs[1].plot(x_axis, he_var_psi_pct[:, :, 2].mean(axis=0), color='C3', label='CRE')
        if line_at_movers_per_firm:
            axs[1].axvline(he_movers_per_firm_line, color='k', linestyle='--')
        axs[1].set_title('Leave-one-out set', fontsize=9)
        axs[1].set_xlabel('Share of Movers Kept (%)', fontsize=7)
        axs[1].set_ylabel(' ')
        axs[1].tick_params(axis='x', labelsize=7)
        axs[1].tick_params(axis='y', labelsize=5)
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
        axs[0].set_title('Connected set', fontsize=9)
        axs[0].set_xlabel('Share of Movers Kept (%)', fontsize=7)
        axs[0].set_ylabel('Share of Variance (%)', fontsize=7)
        axs[0].tick_params(axis='x', labelsize=7)
        axs[0].tick_params(axis='y', labelsize=5)
        axs[0].grid()

        # Sorting (HE)
        axs[1].plot(x_axis, he_cov_psi_alpha_pct[:, :, 0].mean(axis=0), color='C0', label='FE')
        axs[1].plot(x_axis, he_cov_psi_alpha_pct[:, :, 1].mean(axis=0), color='C1', label='HO')
        axs[1].plot(x_axis, he_cov_psi_alpha_pct[:, :, 3].mean(axis=0), color='C2', label='HE')
        axs[1].plot(x_axis, he_cov_psi_alpha_pct[:, :, 2].mean(axis=0), color='C3', label='CRE')
        if line_at_movers_per_firm:
            axs[1].axvline(he_movers_per_firm_line, color='k', linestyle='--')
        axs[1].set_title('Leave-one-out set', fontsize=9)
        axs[1].set_xlabel('Share of Movers Kept (%)', fontsize=7)
        axs[1].set_ylabel(' ')
        axs[1].tick_params(axis='x', labelsize=7)
        axs[1].tick_params(axis='y', labelsize=5)
        axs[1].grid()

        # Shared legend (source: https://stackoverflow.com/a/46921590/17333120)
        handles, labels = axs[1].get_legend_handles_labels()
        subfigs[1].legend(handles, labels, loc=(1.02, 0.78))
        plt.show()

    def boxplots(self, xticks_round=1):
        '''
        Generate attrition result boxplots.

        Arguments:
            line_at_movers_per_firm (bool): if True, plot a dashed line where movers per firm in the subsample is approximately the number of movers per firm in the full sample
            xticks_round (int): how many digits to round x ticks
        '''
        # Extract results
        non_he_var_psi_pct, he_var_psi_pct, non_he_cov_psi_alpha_pct, he_cov_psi_alpha_pct, x_axis = self._combine_res(line_at_movers_per_firm=False, xticks_round=xticks_round)

        ### Plot boxplots ###
        # Source: https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subfigures.html
        fig = plt.figure(constrained_layout=True, dpi=150)
        subfigs = fig.subfigures(nrows=2, ncols=1)

        ## Firm effects ##
        subfigs[0].suptitle('Firm effects', x=0.545)
        subsubfigs = subfigs[0].subfigures(nrows=1, ncols=2)

        # Firm effects (non-HE set)
        axs = subsubfigs[0].subplots(nrows=1, ncols=3, sharey=True)
        subtitles = ['FE', 'FE-HO', 'CRE']

        for i, row in enumerate(axs):
            # for col in row:
            row.boxplot(non_he_var_psi_pct[:, :, i], showfliers=False)
            row.set_xticklabels(x_axis, fontsize=5)
            row.tick_params(axis='y', labelsize=5)
            row.grid()
            row.set_title(subtitles[i], fontsize=7)
        subsubfigs[0].suptitle('Connected set', fontsize=9)
        subsubfigs[0].supxlabel('Share of Movers Kept (%)', fontsize=7)
        subsubfigs[0].supylabel('Share of Variance (%)', fontsize=7)

        # Firm effects (HE set)
        axs = subsubfigs[1].subplots(nrows=1, ncols=4, sharey=True)
        subtitles = ['FE', 'FE-HO', 'CRE', 'FE-HE']
        # Change order because data is FE, FE-HO, CRE, FE-HE but want FE, FE-HO, FE-HE, CRE
        order = [0, 1, 3, 2]

        for i, row in enumerate(axs):
            # for col in row:
            row.boxplot(he_var_psi_pct[:, :, order[i]], showfliers=False)
            row.set_xticklabels(x_axis, fontsize=5)
            row.tick_params(axis='y', labelsize=5)
            row.grid()
            row.set_title(subtitles[order[i]], fontsize=7)
        subsubfigs[1].suptitle('Leave-one-out set', fontsize=9)
        subsubfigs[1].supxlabel('Share of Movers Kept (%)', fontsize=7)
        subsubfigs[1].supylabel(' ')

        ## Sorting ##
        subfigs[1].suptitle('Sorting', x=0.545)
        subsubfigs = subfigs[1].subfigures(nrows=1, ncols=2)

        # Sorting (non-HE set)
        axs = subsubfigs[0].subplots(nrows=1, ncols=3, sharey=True)
        subtitles = ['FE', 'FE-HO', 'CRE']

        for i, row in enumerate(axs):
            # for col in row:
            row.boxplot(non_he_cov_psi_alpha_pct[:, :, i], showfliers=False)
            row.set_xticklabels(x_axis, fontsize=5)
            row.tick_params(axis='y', labelsize=5)
            row.grid()
            row.set_title(subtitles[i], fontsize=7)
        subsubfigs[0].suptitle('Connected set', fontsize=9)
        subsubfigs[0].supxlabel('Share of Movers Kept (%)', fontsize=7)
        subsubfigs[0].supylabel('Share of Variance (%)', fontsize=7)

        # Sorting (HE set)
        axs = subsubfigs[1].subplots(nrows=1, ncols=4, sharey=True)
        subtitles = ['FE', 'FE-HO', 'CRE', 'FE-HE']
        # Change order because data is FE, FE-HO, CRE, FE-HE but want FE, FE-HO, FE-HE, CRE
        order = [0, 1, 3, 2]

        for i, row in enumerate(axs):
            # for col in row:
            row.boxplot(he_cov_psi_alpha_pct[:, :, order[i]], showfliers=False)
            row.set_xticklabels(x_axis, fontsize=5)
            row.tick_params(axis='y', labelsize=5)
            row.grid()
            row.set_title(subtitles[order[i]], fontsize=7)
        subsubfigs[1].suptitle('Leave-one-out set', fontsize=9)
        subsubfigs[1].supxlabel('Share of Movers Kept (%)', fontsize=7)
        subsubfigs[1].supylabel(' ')

        # NOTE: must use plt.show(), fig.show() raises a warning in Jupyter Notebook (source: https://stackoverflow.com/a/52827912/17333120)
        plt.show()
