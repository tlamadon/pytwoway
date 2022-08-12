'''
Sorkin (2018) replication code.
'''
from tqdm.auto import tqdm, trange
try:
    from multiprocess import Pool
except ImportError:
    from multiprocessing import Pool
import numpy as np
from scipy.sparse import csc_matrix
# from scipy.sparse.linalg import eigs
import bipartitepandas as bpd
import pytwoway as tw
from pytwoway.util import DxSP, scramble, unscramble
from matplotlib import pyplot as plt

class SorkinEstimator():
    '''
    Class for estimating the fixed-point revealed preference model from Sorkin (2018). Estimated firm values (log(exp(V_EE))) are stored in the attribute .V_EE.
    '''

    def __init__(self):
        self.res = None

    def fit(self, adata, max_iters=500, threshold=1e-5):
        '''
        Estimate the fixed-point revealed preference model from Sorkin (2018).

        Arguments:
            adata (BipartiteDataFrame): event study or collapsed event study format labor data
            max_iters (int): maximum number of iterations for fixed-point estimation
            threshold (float): threshold maximum absolute percent change between iterations to break fixed-point iterations (i.e. the maximum percent change for a particular firm's estimated value)
        '''
        # M0 shows flows between firms: row gives firm 2, column gives firm 1, and the value is movements from firm 1 to firm 2
        # E.g. M0_{kj} is how many workers choose k and M0_{jk} is how many workers choose j
        M0 = adata.loc[adata.get_worker_m(), :].groupby(['j1', 'j2'])['m'].count()
        row = M0.index.get_level_values(1)
        col = M0.index.get_level_values(0)
        M0 = csc_matrix((M0, (row, col)))

        # S0 shows movements from firm k to all other firms
        # S0_kk = sum_j M_jk
        S0 = np.asarray(M0.sum(axis=0))[0, :]

        # Now solve fixed point/eigenvalue-eigenvector problem:
        # S0^{-1} @ M0 @ exp(V^EE) = exp(V^EE)
        lhs = DxSP(1 / S0, M0)
        nf = M0.shape[0]

        ## Fixed point estimation ##
        # eval, evec = eigs(lhs, k=1, sigma=1)
        prev_guess = np.ones(nf) / nf
        for _ in range(max_iters):
            new_guess = np.abs(lhs @ prev_guess)
            new_guess /= np.sum(new_guess)
            if np.max(np.abs((new_guess - prev_guess) / prev_guess)) <= threshold:
                break
            prev_guess = new_guess

        # Flatten and take log
        self.V_EE = np.log(new_guess.flatten())

class SorkinAttrition:
    '''
    Class of SorkinAttrition, which generates attrition plots using bipartite labor data.

    Arguments:
        min_moves_threshold (int): minimum number of moves required to keep a firm
        attrition_how (tw.attrition_utils.AttritionIncreasing() or tw.attrition_utils.AttritionDecreasing()): instance of AttritionIncreasing() or AttritionDecreasing(), used to specify if attrition should use increasing (building up from a fixed set of firms) or decreasing (with varying sets of firms) fractions of movers; None is equivalent to AttritionIncreasing()
        clean_params (ParamsDict or None): dictionary of parameters for cleaning. Run bpd.clean_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.clean_params().
    '''

    def __init__(self, min_moves_threshold=15, attrition_how=None, clean_params=None):
        if attrition_how is None:
            attrition_how = tw.attrition_utils.AttritionIncreasing()
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
        self.clean_params_1 = clean_params.copy()
        self.clean_params_2 = bpd.clean_params({'is_sorted': True, 'copy': False, 'verbose': False})

        ### Update parameter dictionaries ###
        self.clean_params_1['connectedness'] = 'strongly_connected'
        self.clean_params_1['is_sorted'] = True
        self.clean_params_1['force'] = False
        self.clean_params_1['copy'] = False
        self.clean_params_1['verbose'] = False

    # Cannot include two underscores because isn't compatible with starmap for multiprocessing
    # Source: https://stackoverflow.com/questions/27054963/python-attribute-error-object-has-no-attribute
    def _attrition_interior(self, bdf, fids_to_drop, wids_to_drop, rng=None):
        '''
        Estimate all parameters of interest. This is the interior function to _attrition_single.

        Arguments:
            bdf (BipartiteDataFrame): bipartite dataframe
            fids_to_drop (set or None): firm ids to drop; if None, no firm ids to drop
            wids_to_drop (set or None): worker ids to drop; if None, no worker ids to drop
            rng (np.random.Generator or None): NumPy random number generator. This overrides the random number generators for FE and CRE. None is equivalent to np.random.default_rng(None).

        Returns:
            (float): corr(Sorkin, FE psi)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # logger_init(bdf) # This stops a weird logging bug that stops multiprocessing from working
        ## Drop ids and clean data  ## (NOTE: this does not require a copy)
        if fids_to_drop is not None:
            bdf = bdf.drop_ids('j', fids_to_drop, drop_returns_to_stays=self.clean_params_1['drop_returns_to_stays'], is_sorted=True, copy=False)
        if wids_to_drop is not None:
            bdf = bdf.drop_ids('i', wids_to_drop, drop_returns_to_stays=self.clean_params_1['drop_returns_to_stays'], is_sorted=True, copy=False)
        bdf = bdf._reset_attributes(columns_contig=True, connected=True, no_na=False, no_duplicates=False, i_t_unique=False, no_returns=False)

        ## Clean data ##
        bdf = bdf.clean(self.clean_params_1)

        ## Estimate FE model ##
        fe_params = tw.fe_params({'feonly': True})
        fe_estimator = tw.FEEstimator(bdf, fe_params)
        fe_estimator.fit()
        psi_hat = np.concatenate([fe_estimator.psi_hat, np.array([0])])

        ## Estimate Sorkin ##
        # Prepare data
        jdata = bdf.loc[bdf.get_worker_m(), :].clean(self.clean_params_2)
        jdata = jdata.to_eventstudy()
        # Estimate and store results
        sorkin = SorkinEstimator()
        sorkin.fit(jdata)
        V_EE = sorkin.V_EE

        return np.corrcoef(V_EE, psi_hat)[0, 1]

    def _attrition_single(self, bdf, ncore=1, rng=None):
        '''
        Run attrition estimations to estimate parameters given fraction of movers remaining. This is the interior function to attrition.

        Arguments:
            bdf (BipartiteDataFrame): bipartite dataframe
            ncore (int): number of cores to use
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (list): results for each specified fraction of movers
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        ## Generate attrition subsets and worker ids to drop ##
        subsets = self.attrition_how._gen_subsets(bdf=bdf, clean_params=self.clean_params_1, rng=rng)
        N = len(subsets)
        seeds = rng.bit_generator._seed_seq.spawn(N)

        ## Estimate on subset ##
        if ncore > 1:
            # Multiprocessing
            with Pool(processes=ncore) as pool:
                pbar = tqdm(scramble([(bdf, *subsets[i], np.random.default_rng(seeds[i])) for i in range(N)]), total=N)
                pbar.set_description(f'sorkin')
                V = unscramble(pool.starmap(self._attrition_interior, pbar))
        else:
            # Single core
            pbar = tqdm([(bdf, *subsets[i], np.random.default_rng(seeds[i])) for i in range(N)], total=N)
            pbar.set_description(f'sorkin')
            V = []
            for attrition_subparams in pbar:
                V.append(self._attrition_interior(*attrition_subparams))

        return list(V)

    def attrition(self, bdf, N=10, ncore=1, copy=False, rng=None):
        '''
        Run Monte Carlo on attrition estimations of TwoWay to estimate variance of parameter estimates given fraction of movers remaining. Note that this overwrites the stored dataframe, meaning if you want to run attrition with different threshold number of movers, you will have to create multiple Attrition objects, or alternatively, run this method with an increasing threshold for each iteration. Saves results as a NumPy Array in the class attribute .attrition_res: each row gives results for a particular draw with all fractions of movers, and each column gives results for all draws with a particular fraction of movers.

        Arguments:
            bdf (BipartiteBase): bipartite dataframe (NOTE: we need to avoid saving bdf as a class attribute, otherwise multiprocessing will create a separate copy of it for each core used)
            N (int): number of simulations
            ncore (int): number of cores to use
            copy (bool): if False, avoid copy
            rng (np.random.Generator or None): NumPy random number generator. This overrides the random number generators for FE and CRE. None is equivalent to np.random.default_rng(None).

        Returns:
            (dict of dicts of lists of lists): in the first dictionary we choose 'non_he' or 'he'; in the second dictionary we choose 'fe' or 'cre'; then, we are given a list of results for each Monte Carlo simulation; and finally, for a particular Monte Carlo simulation, we are given a list of results for each attrition percentage.
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        if copy:
            # Copy
            bdf = bdf.copy()

        ## Create list to save results ##
        res_all = np.zeros((N, self.attrition_how.n_subsets))

        # Save movers per firm (do this before taking subset of firms that meet threshold of sufficiently many moves)
        self.movers_per_firm = bdf.loc[bdf.loc[:, 'm'] > 0, :].n_workers() / bdf.n_firms() # bdf.loc[bdf.loc[:, 'm'] > 0, :].groupby('j')['i'].nunique().mean()

        # Take subset of firms that meet threshold of sufficiently many moves
        bdf = bdf.min_moves_frame(threshold=self.min_moves_threshold, drop_returns_to_stays=self.clean_params_1['drop_returns_to_stays'], is_sorted=True, reset_index=True, copy=False)

        if len(bdf) == 0:
            raise ValueError("Length of dataframe is 0 after dropping firms with too few moves, consider lowering `min_moves_threshold` for tw.SorkinAttrition().")

        if False: # ncore > 1:
            # Estimate with multi-processing
            with Pool(processes=ncore) as pool:
                # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
                # Multiprocessing tqdm source: https://stackoverflow.com/a/45276885/17333120
                V = list(tqdm(pool.starmap(self._attrition_single, [(bdf, ncore, np.random.default_rng(seed)) for seed in rng.bit_generator._seed_seq.spawn(N)]), total=N))
            for i, res in enumerate(V):
                res_all[i, :] = res
        else:
            # Estimate without multi-processing
            pbar = trange(N)
            pbar.set_description('attrition main')
            for i, _ in enumerate(pbar):
                res = self._attrition_single(bdf=bdf, ncore=ncore, rng=rng)
                res_all[i, :] = res

        # Combine results
        self.attrition_res = res_all

    def plot(self, line_at_movers_per_firm=True, xticks_round=1):
        '''
        Generate attrition result plots.

        Arguments:
            line_at_movers_per_firm (bool): if True, plot a dashed line where movers per firm in the subsample is approximately the number of movers per firm in the full sample
            xticks_round (int): how many digits to round x ticks
        '''
        res = self.attrition_res
        x_axis = self.attrition_how.subset_fractions
        plt.plot(np.round(100 * x_axis, xticks_round), np.mean(res, axis=0))
        plt.xlabel('Share of Movers Kept (%)')
        plt.ylabel('corr(Sorkin, FE psi)')
        plt.grid()
        plt.show()
