'''
Class for attrition estimation and plotting.
'''
from tqdm.auto import tqdm, trange
# import itertools
try:
    from multiprocess import Pool
except ImportError:
    from multiprocessing import Pool
import numpy as np
# import pandas as pd
from matplotlib import pyplot as plt
import bipartitepandas as bpd
# from bipartitepandas.util import to_list, logger_init
import pytwoway as tw
from pytwoway.util import scramble, unscramble

class Attrition:
    '''
    Class of Attrition, which generates attrition plots using bipartite labor data.

    Arguments:
        min_movers_threshold (int): minimum number of movers required to keep a firm
        attrition_how (tw.attrition_utils.AttritionIncreasing() or tw.attrition_utils.AttritionDecreasing()): instance of AttritionIncreasing() or AttritionDecreasing(), used to specify if attrition should use increasing (building up from a fixed set of firms) or decreasing (with varying sets of firms) fractions of movers; None is equivalent to AttritionIncreasing()
        fe_params (ParamsDict or None): dictionary of parameters for FE estimation. Run tw.fe_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.fe_params().
        cre_params (ParamsDict or None): dictionary of parameters for CRE estimation. Run tw.cre_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.cre_params().
        estimate_bs (bool): if True, estimate Borovickova-Shimer model
        cluster_params (ParamsDict or None): dictionary of parameters for clustering in CRE estimation. Run bpd.cluster_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.cluster_params().
        clean_params (ParamsDict or None): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.clean_params().
    '''

    def __init__(self, min_movers_threshold=15, attrition_how=None, fe_params=None, cre_params=None, estimate_bs=False, cluster_params=None, clean_params=None):
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
        # Minimum number of movers required to keep a firm
        self.min_movers_threshold = min_movers_threshold
        # AttritionIncreasing() or AttritionDecreasing()
        self.attrition_how = attrition_how
        # Whether to estimate Borovickova-Shimer model
        self.estimate_bs = estimate_bs
        # Prevent plotting until results exist
        self.res = None

        ## Models to estimate ##
        models = ['fe', 'cre']
        if estimate_bs:
            models.append('bs')
        self.models = models

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
        Estimate all parameters of interest. This is the interior function to _attrition_single.

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
            (dict): {'fe': FE results, 'cre': CRE results}; if estimating Borovickova-Shimer estimator, then add key {'bs': Borovickova-Shimer results}
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
        del fe_estimator

        res = {'fe': fe_res, 'cre': cre_res}

        if self.estimate_bs:
            ## Estimate Borovickova-Shimer model ##
            # Make sure there are no returners and that all workers and firms have at least 2 observations

            # Clean parameters
            clean_params_1 = clean_params.copy()
            clean_params_2 = clean_params.copy()
            clean_params_1['connectedness'] = None
            clean_params_1['drop_returns'] = 'returns'
            clean_params_2['connectedness'] = None

            # Clean
            bdf = bdf.clean(clean_params_1).min_joint_obs_frame(is_sorted=True, copy=False).clean(clean_params_2)

            ### Estimate ###
            bs_estimator = tw.BSEstimator()

            ## Standard ##
            bs_estimator.fit(bdf, alternative_estimator=False)
            # Save results
            bs1_res = bs_estimator.res

            ## Alternative ##
            bs_estimator.fit(bdf, alternative_estimator=True)
            # Save results
            bs2_res = bs_estimator.res

            ## Combine ##
            bs_res = {k + '_bs1' if k != 'var(y)' else k: v for k, v in bs1_res.items()}
            bs_res.update({k + '_bs2': v for k, v in bs2_res.items() if k != 'var(y)'})

            # Save results
            res['bs'] = bs_res

            del bs_estimator

        return res

    def _attrition_single(self, bdf, ncore=1, rng=None):
        '''
        Run attrition estimations to estimate parameters given fraction of movers remaining. This is the interior function to attrition.

        Arguments:
            bdf (BipartiteDataFrame): bipartite dataframe
            ncore (int): number of cores to use
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (dict of dicts of lists): in the first dictionary we choose 'non_he' or 'he'; in the second dictionary we choose 'fe' or 'cre'; and finally, we are given a list of results for each specified fraction of movers.
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        ## Create lists to save results ##
        # For non-HE
        res_non_he = {model: [] for model in self.models}
        # For HE
        res_he = {model: [] for model in self.models}
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
                    pbar = tqdm(scramble([(bdf, *attrition_subparams) for attrition_subparams in attrition_params]), total=N)
                    pbar.set_description(f'attrition, {non_he_he}')
                    V = unscramble(pool.starmap(self._attrition_interior, pbar))
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
                for model in self.models:
                    res_all[non_he_he][model].append(res[model])

        return res_all

    def attrition(self, bdf, N=10, ncore=1, copy=False, rng=None):
        '''
        Run Monte Carlo on attrition estimations of TwoWay to estimate variance of parameter estimates given fraction of movers remaining. Note that this overwrites the stored dataframe, meaning if you want to run attrition with different threshold number of movers, you will have to create multiple Attrition objects, or alternatively, run this method with an increasing threshold for each iteration. Saves results as a dict of dicts of lists of lists in the class attribute .attrition_res: in the first dictionary we choose 'non_he' or 'he'; in the second dictionary we choose 'fe' or 'cre'; then, we are given a list of results for each Monte Carlo simulation; and finally, for a particular Monte Carlo simulation, we are given a list of results for each specified fraction of movers.

        Arguments:
            bdf (BipartiteBase): bipartite dataframe (NOTE: we need to avoid saving bdf as a class attribute, otherwise multiprocessing will create a separate copy of it for each core used)
            N (int): number of simulations
            ncore (int): number of cores to use
            copy (bool): if False, avoid copy
            rng (np.random.Generator or None): NumPy random number generator. This overrides the random number generators for FE and CRE. None is equivalent to np.random.default_rng(None).
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        if copy:
            # Copy
            bdf = bdf.copy()

        ## Create lists to save results ##
        # For non-HE
        res_non_he = {model: [] for model in self.models}
        # For HE
        res_he = {model: [] for model in self.models}

        # Save movers per firm (do this before taking subset of firms that meet threshold of sufficiently many movers)
        self.movers_per_firm = bdf.loc[bdf.loc[:, 'm'] > 0, :].n_workers() / bdf.n_firms() # bdf.loc[bdf.loc[:, 'm'] > 0, :].groupby('j')['i'].nunique().mean()

        # Take subset of firms that meet threshold of sufficiently many movers
        bdf = bdf.min_movers_frame(threshold=self.min_movers_threshold, drop_returns_to_stays=self.clean_params['drop_returns_to_stays'], is_sorted=True, reset_index=True, copy=False)

        if len(bdf) == 0:
            raise ValueError("Length of dataframe is 0 after dropping firms with too few movers, consider lowering `min_movers_threshold` for tw.Attrition().")

        if False: # ncore > 1:
            # Estimate with multi-processing
            with Pool(processes=ncore) as pool:
                # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
                # Multiprocessing tqdm source: https://stackoverflow.com/a/45276885/17333120
                V = list(tqdm(pool.starmap(self._attrition_single, [(bdf, ncore, np.random.default_rng(seed)) for seed in rng.bit_generator._seed_seq.spawn(N)]), total=N))
            for res in V:
                for model in self.models:
                    res_non_he[model].append(res['non_he'][model])
                    res_he[model].append(res['he'][model])
        else:
            # Estimate without multi-processing
            pbar = trange(N)
            pbar.set_description('attrition main')
            for _ in pbar:
                res = self._attrition_single(bdf=bdf, ncore=ncore, rng=rng)
                for model in self.models:
                    res_non_he[model].append(res['non_he'][model])
                    res_he[model].append(res['he'][model])

        # Combine results
        self.res = {'non_he': res_non_he, 'he': res_he}

    def _combine_res(self, to_plot_dict, line_at_movers_per_firm=True, xticks_round=1):
        '''
        Combine attrition results into NumPy Arrays.

        Arguments:
            to_plot_dict (dict): dictionary linking estimator name to whether it will be plotted
            line_at_movers_per_firm (bool): if True, plot a dashed line where movers per firm in the subsample is approximately the number of movers per firm in the full sample
            xticks_round (int): how many digits to round x ticks

        Returns:
            (tuple): if line_at_movers_per_firm=True, (var_diffs_non_he, var_diffs_he, cov_diffs_non_he, cov_diffs_he, movers_per_firm_line_non_he, movers_per_firm_line_he, x_axis); if line_at_movers_per_firm=False, (var_diffs_non_he, var_diffs_he, cov_diffs_non_he, cov_diffs_he, x_axis)
        '''
        if self.res is None:
            raise AttributeError('Attribute .res is None - must generate attrition data before results can be plotted. This can be done by running .attrition()')

        ## Get N, M ##
        # Number of estimations
        N = len(self.res['non_he']['fe'])
        # Number of attritions per estimation
        M = len(self.res['non_he']['fe'][0])

        ## Data construction dictionaries ##
        model_dict = {
            'fe': 'fe',
            'ho': 'fe',
            'he': 'fe',
            'cre': 'cre',
            'bs1': 'bs',
            'bs2': 'bs'
        }
        var_y_dict = {
            'fe': 'var(y)',
            'cre': 'var_y',
            'bs': 'var(y)'
        }
        var_psi_dict = {
            'fe': 'var(psi)_fe',
            'ho': 'var(psi)_ho',
            'he': 'var(psi)_he',
            'cre': 'tot_var',
            'bs1': 'var(mu)_bs1',
            'bs2': 'var(mu)_bs2'
        }
        cov_dict = {
            'fe': 'cov(psi, alpha)_fe',
            'ho': 'cov(psi, alpha)_ho',
            'he': 'cov(psi, alpha)_he',
            'cre': 'tot_cov',
            'bs1': 'cov(lambda, mu)_bs1',
            'bs2': 'cov(lambda, mu)_bs2'
        }

        ## Extract results ##
        # Non-HE
        var_diffs_non_he = {estimator: np.zeros(shape=[N, M]) for estimator, to_plot in to_plot_dict.items() if (to_plot and estimator != 'he')}
        cov_diffs_non_he = {estimator: np.zeros(shape=[N, M]) for estimator, to_plot in to_plot_dict.items() if (to_plot and estimator != 'he')}
        # HE
        var_diffs_he = {estimator: np.zeros(shape=[N, M]) for estimator, to_plot in to_plot_dict.items() if to_plot}
        cov_diffs_he = {estimator: np.zeros(shape=[N, M]) for estimator, to_plot in to_plot_dict.items() if to_plot}
        for i in range(N):
            for j in range(M):
                for estimator, model in model_dict.items():
                    if to_plot_dict[estimator]:
                        # If plotting this estimator
                        if estimator != 'he':
                            ## Non-HE ##
                            non_he_res_dict = self.res['non_he'][model][i][j]
                            # var(psi)
                            var_diffs_non_he[estimator][i, j] = 100 * (float(non_he_res_dict[var_psi_dict[estimator]]) / float(non_he_res_dict[var_y_dict[model]]))
                            # cov(psi, alpha)
                            cov_diffs_non_he[estimator][i, j] = 100 * (float(non_he_res_dict[cov_dict[estimator]]) / float(non_he_res_dict[var_y_dict[model]]))

                        ## HE ##
                        he_res_dict = self.res['he'][model][i][j]
                        # var(psi)
                        var_diffs_he[estimator][i, j] = 100 * (float(he_res_dict[var_psi_dict[estimator]]) / float(he_res_dict[var_y_dict[model]]))
                        # cov(psi, alpha)
                        cov_diffs_he[estimator][i, j] = 100 * (float(he_res_dict[cov_dict[estimator]]) / float(he_res_dict[var_y_dict[model]]))

        # x-axis
        x_axis = np.round(100 * self.attrition_how.subset_fractions, xticks_round)
        if np.all(x_axis == x_axis.astype(int)):
            # This is necessary for the boxplots, since they don't automatically convert to integers
            x_axis = x_axis.astype(int, copy=False)

        # Flip along 1st axis so that both increasing and decreasing have the same order
        if np.max(np.diff(self.attrition_how.subset_fractions)) <= 0:
            # If decreasing
            x_axis = np.flip(x_axis)
            var_diffs_non_he = np.flip(var_diffs_non_he, axis=1)
            cov_diffs_non_he = np.flip(cov_diffs_non_he, axis=1)
            var_diffs_he = np.flip(var_diffs_he, axis=1)
            cov_diffs_he = np.flip(cov_diffs_he, axis=1)

        ## Prepare line at movers per firm
        if line_at_movers_per_firm:
            movers_per_firm_non_he = np.zeros(shape=M)
            movers_per_firm_he = np.zeros(shape=M)

            for i in range(N):
                for j in range(M):
                    # Sum over movers per firm for all iterations
                    movers_i_j_non_he = self.res['non_he']['fe'][i][j]
                    movers_i_j_he = self.res['he']['fe'][i][j]
                    movers_per_firm_non_he[j] += (int(movers_i_j_non_he['n_movers']) / int(movers_i_j_non_he['n_firms'])) # float(movers_i_j_non_he['movers_per_firm'])
                    movers_per_firm_he[j] += (int(movers_i_j_he['n_movers']) / int(movers_i_j_he['n_firms'])) # float(movers_i_j_he['movers_per_firm'])
            # Take average
            movers_per_firm_non_he /= N
            movers_per_firm_he /= N
            # Increase by 5%, because we are approximating
            movers_per_firm_non_he *= 1.05
            movers_per_firm_he *= 1.05

            # Reverse order so that both increasing and decreasing have the same order
            if np.max(np.diff(self.attrition_how.subset_fractions)) <= 0:
                # If decreasing
                movers_per_firm_non_he = np.flip(movers_per_firm_non_he, axis=0)
                movers_per_firm_he = np.flip(movers_per_firm_he, axis=0)

            if self.movers_per_firm >= np.max(movers_per_firm_non_he):
                movers_per_firm_line_non_he = np.max(x_axis)
            elif self.movers_per_firm <= np.min(movers_per_firm_non_he):
                movers_per_firm_line_non_he = np.min(x_axis)
            else:
                # Find where movers per firm in subset approximates movers per firm in entire dataset
                for i, movers_per_firm_non_he_i in enumerate(movers_per_firm_non_he[1:]):
                    if self.movers_per_firm < movers_per_firm_non_he_i:
                        frac = (self.movers_per_firm - movers_per_firm_non_he[i]) / (movers_per_firm_non_he_i - movers_per_firm_non_he[i])
                        movers_per_firm_line_non_he = x_axis[i] + frac * (x_axis[i + 1] - x_axis[i])
                        break

            if self.movers_per_firm >= np.max(movers_per_firm_he):
                movers_per_firm_line_he = np.max(x_axis)
            elif self.movers_per_firm <= np.min(movers_per_firm_he):
                movers_per_firm_line_he = np.min(x_axis)
            else:
                # Find where movers per firm in subset approximates movers per firm in entire dataset
                for i, movers_per_firm_he_i in enumerate(movers_per_firm_he[1:]):
                    if self.movers_per_firm < movers_per_firm_he_i:
                        frac = (self.movers_per_firm - movers_per_firm_he[i]) / (movers_per_firm_he_i - movers_per_firm_he[i])
                        movers_per_firm_line_he = x_axis[i] + frac * (x_axis[i + 1] - x_axis[i])
                        break

            return (var_diffs_non_he, var_diffs_he, cov_diffs_non_he, cov_diffs_he, movers_per_firm_line_non_he, movers_per_firm_line_he, x_axis)

        return (var_diffs_non_he, var_diffs_he, cov_diffs_non_he, cov_diffs_he, x_axis)

    def plots(self, fe=True, ho=True, he=True, cre=True, bs1=True, bs2=True, line_at_movers_per_firm=True, xticks_round=1):
        '''
        Generate attrition result plots.

        Arguments:
            fe (bool): if True, plot FE results
            ho (bool): if True, plot FE-HO results
            he (bool): if True, plot FE-HE results
            cre (bool): if True, plot CRE results
            bs1 (bool): if True, plot Borovickova-Shimer results for the standard estimator
            bs2 (bool): if True, plot Borovickova-Shimer results for the alternative estimator
            line_at_movers_per_firm (bool): if True, plot a dashed line where movers per firm in the subsample is approximately the number of movers per firm in the full sample
            xticks_round (int): how many digits to round x ticks
        '''
        if self.res is None:
            raise AttributeError('Attribute .res is None - must generate attrition data before results can be plotted. This can be done by running .attrition()')

        if not self.estimate_bs:
            bs1 = False
            bs2 = False

        ## Plotting dictionaries ##
        to_plot_dict = {
            'fe': fe,
            'ho': ho,
            'he': he,
            'cre': cre,
            'bs1': bs1,
            'bs2': bs2
        }
        plot_options_dict = {
            'fe': {
                'label': 'FE',
                'color': 'C0'
            },
            'ho': {
                'label': 'FE-HO',
                'color': 'C1'
            },
            'he': {
                'label': 'FE-HE',
                'color': 'C2'
            },
            'cre': {
                'label': 'CRE',
                'color': 'C3'
            },
            'bs1': {
                'label': 'BS-1',
                'color': 'C4'
            },
            'bs2': {
                'label': 'BS-2',
                'color': 'C5'
            }
        }

        ## Extract results ##
        var_diffs_non_he, var_diffs_he, cov_diffs_non_he, cov_diffs_he, movers_per_firm_line_non_he, movers_per_firm_line_he, x_axis = self._combine_res(to_plot_dict, line_at_movers_per_firm=line_at_movers_per_firm, xticks_round=xticks_round)

        ### Plot figures ###
        ## Information for plots ##
        row_titles = ['Firm effects', 'Sorting']
        col_titles = ['Connected set', 'Leave-one-out set']
        data_array = [[var_diffs_non_he, var_diffs_he], [cov_diffs_non_he, cov_diffs_he]]
        movers_per_firm_lst = [movers_per_firm_line_non_he, movers_per_firm_line_he]

        ## Plot ##
        # Source: https://stackoverflow.com/a/68209152/17333120
        fig = plt.figure(constrained_layout=True, dpi=150)
        subfigs = fig.subfigures(nrows=2, ncols=1)

        for i, subfig in enumerate(subfigs):
            ## Rows of the plot (firm effects vs. sorting) ##
            subfig.suptitle(row_titles[i], x=0.545)
            axs = subfig.subplots(nrows=1, ncols=2)
            for j, subaxs in enumerate(axs):
                ## Columns of the plot (connected vs. leave-one-out) ##
                if line_at_movers_per_firm:
                    subaxs.axvline(movers_per_firm_lst[j], color='k', linestyle='--')
                for estimator, plot_options in plot_options_dict.items():
                    if to_plot_dict[estimator] and not ((estimator == 'he') and (j == 0)):
                        # Plot if to_plot=True
                        subaxs.plot(x_axis, data_array[i][j][estimator].mean(axis=0), color=plot_options['color'], label=plot_options['label'])
                subaxs.set_title(col_titles[j], fontsize=9)
                subaxs.set_xlabel('Share of Movers Kept (%)', fontsize=7)
                if j == 0:
                    subaxs.set_ylabel('Share of Variance (%)', fontsize=7)
                else:
                    subaxs.set_ylabel(' ')
                subaxs.tick_params(axis='x', labelsize=7)
                subaxs.tick_params(axis='y', labelsize=5)
                subaxs.grid()

        # Shared legend (source: https://stackoverflow.com/a/46921590/17333120)
        handles, labels = axs[1].get_legend_handles_labels()
        subfigs[1].legend(handles, labels, loc=(1.02, 0.78))
        plt.show()

    def boxplots(self, fe=True, ho=True, he=True, cre=True, bs1=True, bs2=True, xticks_round=1):
        '''
        Generate attrition result boxplots.

        Arguments:
            fe (bool): if True, plot FE results
            ho (bool): if True, plot FE-HO results
            he (bool): if True, plot FE-HE results
            cre (bool): if True, plot CRE results
            bs1 (bool): if True, plot Borovickova-Shimer results for the standard estimator
            bs2 (bool): if True, plot Borovickova-Shimer results for the alternative estimator
            line_at_movers_per_firm (bool): if True, plot a dashed line where movers per firm in the subsample is approximately the number of movers per firm in the full sample
            xticks_round (int): how many digits to round x ticks
        '''
        if self.res is None:
            raise AttributeError('Attribute .res is None - must generate attrition data before results can be plotted. This can be done by running .attrition()')

        if not self.estimate_bs:
            bs1 = False
            bs2 = False

        ## Plotting dictionaries ##
        to_plot_dict = {
            'fe': fe,
            'ho': ho,
            'he': he,
            'cre': cre,
            'bs1': bs1,
            'bs2': bs2
        }
        plot_options_dict = {
            'fe': {
                'label': 'FE'
            },
            'ho': {
                'label': 'FE-HO'
            },
            'he': {
                'label': 'FE-HE'
            },
            'cre': {
                'label': 'CRE'
            },
            'bs1': {
                'label': 'BS-1'
            },
            'bs2': {
                'label': 'BS-2'
            }
        }
        # Subtract 1 plot if HE is being estimated (since baseline plot doesn't include it)
        n_plots = sum(to_plot_dict.values()) - 1 * he

        ## Extract results ##
        var_diffs_non_he, var_diffs_he, cov_diffs_non_he, cov_diffs_he, x_axis = self._combine_res(to_plot_dict, line_at_movers_per_firm=False, xticks_round=xticks_round)

        ### Plot boxplots ###
        ## Information for plots ##
        row_titles = ['Firm effects', 'Sorting']
        col_titles = ['Connected set', 'Leave-one-out set']
        data_array = [[var_diffs_non_he, var_diffs_he], [cov_diffs_non_he, cov_diffs_he]]

        ## Plot ##
        # Source: https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subfigures.html
        fig = plt.figure(constrained_layout=True, dpi=150)
        subfigs = fig.subfigures(nrows=2, ncols=1)

        for i, subfig in enumerate(subfigs):
            ## Rows of the plot (firm effects vs. sorting) ##
            subfig.suptitle(row_titles[i], x=0.545)
            subsubfigs = subfig.subfigures(nrows=1, ncols=2)
            for j, subsubfig in enumerate(subsubfigs):
                ## Main columns of the plot (connected vs. leave-one-out) ##
                # Column labels
                subsubfig.suptitle(col_titles[j], fontsize=9)
                subsubfig.supxlabel('Share of Movers Kept (%)', fontsize=7)
                if j == 0:
                    subsubfig.supylabel('Share of Variance (%)', fontsize=7)
                else:
                    subsubfig.supylabel(' ')

                # Plots (add 1 column for leave-out-set if HE is being plotted)
                axs = subsubfig.subplots(nrows=1, ncols=n_plots + he * (j == 1), sharey=True)
                k = 0
                for estimator, plot_options in plot_options_dict.items():
                    ## Sub-columns of the plot (FE vs. FE-HO vs. FE-HE vs. CRE) ##
                    if to_plot_dict[estimator] and not ((estimator == 'he') and (j == 0)):
                        # Plot if to_plot=True
                        subaxs = axs[k]
                        subaxs.boxplot(data_array[i][j][estimator], showfliers=False)
                        subaxs.set_title(plot_options['label'], fontsize=7)
                        subaxs.set_xticklabels(x_axis, fontsize=5)
                        subaxs.tick_params(axis='y', labelsize=5)
                        subaxs.grid()
                        # Only iterate k if estimator is plotted
                        k += 1

        # NOTE: must use plt.show(), fig.show() raises a warning in Jupyter Notebook (source: https://stackoverflow.com/a/52827912/17333120)
        plt.show()
