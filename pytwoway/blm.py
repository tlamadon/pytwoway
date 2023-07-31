'''
Implement the static, 2-period non-linear estimator from Bonhomme, Lamadon, & Manresa.
'''
from tqdm.auto import tqdm, trange
import copy
import warnings
# import itertools
try:
    from multiprocess import Pool
except ImportError:
    from multiprocessing import Pool
import numpy as np
import pandas as pd
# from scipy.special import logsumexp
from scipy.sparse import csc_matrix
from matplotlib import pyplot as plt
import plotly.graph_objects as go
from paramsdict import ParamsDict, ParamsDictBase
from paramsdict.util import col_type
import bipartitepandas as bpd
from bipartitepandas.util import to_list, HiddenPrints # , _is_subtype
import pytwoway as tw
from pytwoway import constraints as cons
from pytwoway.util import weighted_mean, weighted_quantile, DxSP, DxM, diag_of_sp_prod, jitter_scatter, exp_, log_, logsumexp, lognormpdf, fast_lognormpdf

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq2(a):
    return a >= 2
def _gteq1(a):
    return a >= 1
def _gteq0(a):
    return np.min(a) >= 0
def _gt0(a):
    return a > 0
def _min_gt0(a):
    return np.min(a) > 0

# Define default parameter dictionaries
blm_params = ParamsDict({
    ## Class parameters ##
    'nl': (6, 'type_constrained', (int, _gteq1),
        '''
            (default=6) Number of worker types.
        ''', '>= 1'),
    'nk': (None, 'type_constrained_none', (int, _gteq1),
        '''
            (default=None) Number of firm types. None will raise an error when running the estimator.
        ''', '>= 1'),
    'categorical_controls': (None, 'dict_of_type_none', ParamsDictBase,
        '''
            (default=None) Dictionary linking column names to instances of tw.categorical_control_params(). Each instance specifies a new categorical control variable and how its starting values should be generated. Run tw.categorical_control_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
        ''', None),
    'continuous_controls': (None, 'dict_of_type_none', ParamsDictBase,
        '''
            (default=None) Dictionary linking column names to instances of tw.continuous_control_params(). Each instance specifies a new continuous control variable and how its starting values should be generated. Run tw.continuous_control_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
        ''', None),
    'primary_period': ('first', 'set', ['first', 'second', 'all'],
        '''
            (default='first') Period to normalize and sort over. 'first' uses first period parameters; 'second' uses second period parameters; 'all' uses the average over first and second period parameters.
        ''', None),
    'gpu': (False, 'type', bool,
        '''
            (default=False) If True, utilize the GPU for certain operations. This will only work for CUDA-compatible GPUs, and requires that the package PyTorch is installed.
        ''', None),
    'verbose': (1, 'set', [0, 1, 2, 3],
        '''
            (default=1) If 0, print no output; if 1, print each major step in estimation; if 2, print warnings during estimation; if 3, print likelihoods at each iteration.
        ''', None),
    ## Starting values ##
    'a1_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A1 (mean of fixed effects in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A1 (mean of fixed effects in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A2 (mean of fixed effects in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2 (mean of fixed effects in second period).
        ''', '>= 0'),
    's1_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S1 (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's1_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S1 (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's2_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2 (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    's2_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S2 (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    'pk1_prior': (None, 'array_of_type_constrained_none', (('float', 'int'), _min_gt0),
        '''
            (default=None) Prior for pk1 (probability of being at each combination of firm types for movers). In particular, pk1 now generates as a convex combination of the prior plus a Dirichlet random variable. Must have shape (nk * nk, nl). None puts all weight on the Dirichlet variable.
        ''', 'min > 0'),
    'pk0_prior': (None, 'array_of_type_constrained_none', (('float', 'int'), _min_gt0),
        '''
            (default=None) Prior for pk0 (probability of being at each firm type for stayers). Must have shape (nk, nl). None is equivalent to np.ones((nk, nl)) / nl.
        ''', 'min > 0'),
    ## fit_movers() and fit_stayers() parameters ##
    'weighted': (True, 'type', bool,
        '''
            (default=True) If True, run estimator with weights. These come from data columns 'w1' and 'w2'.
        ''', None),
    'normalize': (True, 'type', bool,
        '''
            (default=True) If True, normalize estimator during estimation if there are categorical controls with constraints. With particular constraints, the estimator may be identified without normalization, in which case this should be set to False.
        ''', None),
    'return_qi': (False, 'type', bool,
        '''
            (default=False) If True, return qi matrix after first loop.
        ''', None),
    ## fit_movers() parameters ##
    'n_iters_movers': (1000, 'type_constrained', (int, _gteq1),
        '''
            (default=1000) Maximum number of EM iterations for movers.
        ''', '>= 1'),
    'threshold_movers': (1e-7, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1e-7) Threshold to break EM loop for movers.
        ''', '>= 0'),
    'update_a': (True, 'type', bool,
        '''
            (default=True) If False, do not update A1 or A2.
        ''', None),
    'update_s': (True, 'type', bool,
        '''
            (default=True) If False, do not update S1 or S2.
        ''', None),
    'update_pk1': (True, 'type', bool,
        '''
            (default=True) If False, do not update pk1.
        ''', None),
    'cons_a': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on A1 and A2. None is equivalent to [].
        ''', None),
    'cons_s': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on S1 and S2. None is equivalent to [].
        ''', None),
    'cons_a_all': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on A1/A2/A1_cat/A2_cat/A1_cts/A2_cts. None is equivalent to [].
        ''', None),
    'cons_s_all': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on S1/S2/S1_cat/S2_cat/S1_cts/S2_cts. None is equivalent to [].
        ''', None),
    's_lower_bound': (1e-7, 'type_constrained', ((float, int), _gt0),
        '''
            (default=1e-7) Lower bound on estimated S1/S2/S1_cat/S2_cat/S1_cts/S2_cts.
        ''', '> 0'),
    'd_prior_movers': (1 + 1e-7, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1 + 1e-7) Account for probabilities being too small by adding (d_prior - 1) to pk1.
        ''', '>= 1'),
    'd_mean_worker_effect': (1e-7, 'type', (float, int),
        '''
            (default=1e-7) When using categorical constraints, force the mean worker type effect over all firm types to increase by at least this amount as worker type increases. Can be set to negative values.
        ''', None),
    'd_mean_firm_effect': (0, 'type', (float, int),
        '''
            (default=0) When setting 'force_min_firm_type'=True, force the mean firm type effect over all worker types for the lowest firm type to be at least this much smaller than for all other firm types. Can be set to negative values.
        ''', None),
    'start_cycle_check_threshold': (200, 'type_constrained', (int, _gteq0),
        '''
            (default=200) When using categorical constraints, the estimator can get stuck cycling through minimum firm types, preventing convergence. This parameter sets the first iteration to start checking if the estimator is stuck in a cycle.
        ''', '>= 0'),
    'cycle_check_n_obs': (20, 'type_constrained', (int, _gteq2),
        '''
            (default=20) When using categorical constraints, the estimator can get stuck cycling through minimum firm types, preventing convergence. This parameter sets the number of consecutive minimum firm types that should be stored at a time to check for cycling.
        ''', '>= 2'),
    'force_min_firm_type': (False, 'type', bool,
        '''
            (default=False) When using categorical constraints, the estimator can get stuck cycling through minimum firm types, preventing convergence. If the estimator is cycling, restart estimation and set this to True to have the estimator iterate over each firm type, constraining it to be the minimum firm type, then store results from the minimum firm type with the highest likelihood.
        ''', None),
    'force_min_firm_type_constraint': (True, 'type', bool,
        '''
            (default=True) If 'force_min_firm_type'=True, add constraint to force minimum firm type to have the lowest average effect out of all firm types (the estimator may work better with this set to False, but the returned parameters may be inconsistent with the given constraints).
        ''', None),
    ## fit_stayers() parameters ##
    'n_iters_stayers': (1000, 'type_constrained', (int, _gteq1),
        '''
            (default=1000) Maximum number of EM iterations for stayers.
        ''', '>= 1'),
    'threshold_stayers': (1e-7, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1e-7) Threshold to break EM loop for stayers.
        ''', '>= 0'),
    'd_prior_stayers': (1 + 1e-7, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1 + 1e-7) Account for probabilities being too small by adding (d_prior - 1) to pk0.
        ''', '>= 1')
})

categorical_control_params = ParamsDict({
    'n': (None, 'type_constrained_none', (int, _gteq2),
        '''
            (default=6) Number of types for the parameter. None will raise an error when running the estimator.
        ''', '>= 2'),
    'a1_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A1_cat (mean of fixed effects in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A1_cat (mean of fixed effects in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A2_cat (mean of fixed effects in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2_cat (mean of fixed effects in second period).
        ''', '>= 0'),
    's1_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S1_cat (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's1_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S1_cat (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's2_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2_cat (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    's2_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2_cat (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    'worker_type_interaction': (False, 'type', bool,
        '''
            (default=False) If True, effect can differ by worker type.
        ''', None),
    'cons_a': (None, 'list_of_type_none', (cons.Linear, cons.LinearAdditive, cons.Monotonic, cons.MonotonicMean, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on A1 and A2. None is equivalent to [].
        ''', None),
    'cons_s': (None, 'list_of_type_none', (cons.Linear, cons.LinearAdditive, cons.Monotonic, cons.MonotonicMean, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on S1 and S2. None is equivalent to [].
        ''', None)
})

continuous_control_params = ParamsDict({
    'a1_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A1_cts (mean of coefficient in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A1_cts (mean of coefficient in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A2_cts (mean of coefficient in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2_cts (mean of coefficient in second period).
        ''', '>= 0'),
    's1_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S1_cts (standard deviation of coefficient in first period).
        ''', '>= 0'),
    's1_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S1_cts (standard deviation of coefficient in first period).
        ''', '>= 0'),
    's2_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2_cts (standard deviation of coefficient in second period).
        ''', '>= 0'),
    's2_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2_cts (standard deviation of coefficient in second period).
        ''', '>= 0'),
    'worker_type_interaction': (False, 'type', bool,
        '''
            (default=False) If True, effect can differ by worker type.
        ''', None),
    'cons_a': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on A1 and A2. None is equivalent to [].
        ''', None),
    'cons_s': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on S1 and S2. None is equivalent to [].
        ''', None)
})

def _optimal_reallocation(model, jdata, sdata, gj, gs, Lm, Ls, method='max', reallocation_scaling_col=None, rng=None):
    '''
    Reallocate workers to firms in order to maximize total expected output.

    Arguments:
        model (BLMModel): BLM model with estimated parameters
        jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
        sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
        gj (NumPy Array or None): firm classes for movers in the first and second periods
        gs (NumPy Array or None): firm classes for stayers
        Lm (NumPy Array): simulated worker types for movers
        Ls (NumPy Array): simulated worker types for stayers
        method (str): reallocate workers to new firms to maximize ('max') or minimize ('min') total output
        reallocation_scaling_col (str or None): specify column to use to scale outcomes when computing optimal reallocation (i.e. multiply outcomes by an observation-level factor); if None, don't scale outcomes
        rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

    Returns:
        (NumPy Array): optimally reallocated firm class for each observation
    '''
    if rng is None:
        rng = np.random.default_rng(None)

    ## Unpack parameters ##
    nl, nk = model.nl, model.nk

    if reallocation_scaling_col is not None:
        # Each observation is allocated
        nl_adj = len(Lm) + len(Ls)
        worker_sizes_1 = np.ones(nl_adj, dtype=int)
    else:
        # Worker-firm match groups are allocated
        nl_adj = nl

    ## Firm sizes ##
    # NOTE: use minlength, since jdata/sdata might be subsets of the full data
    firm_sizes_s = np.bincount(gs, minlength=nk)
    firm_sizes_1 = np.bincount(gj[:, 0], minlength=nk) + firm_sizes_s
    # firm_sizes_2 = np.bincount(gj[:, 1], minlength=nk) + firm_sizes_s
    del firm_sizes_s

    if reallocation_scaling_col is None:
        ## Worker type sizes ##
        worker_sizes_s = np.bincount(Ls)
        worker_sizes_1 = np.bincount(Lm) + worker_sizes_s
        # worker_sizes_2 = np.bincount(Lm) + worker_sizes_s
        del worker_sizes_s

    ### Set up linear programming solver ###
    # NOTE: Force everybody to become a stayer
    ## Y ##
    if method == 'max':
        Y = model.A1
    elif method == 'min':
        Y = -model.A1
    if reallocation_scaling_col is not None:
        # Multiply Y by scaling factor
        scaling_cols = to_list(sdata.col_reference_dict[reallocation_scaling_col])
        scaling_col_1 = scaling_cols[0]
        if len(scaling_cols) == 1:
            scaling_col_2 = scaling_col_1
        elif len(scaling_cols) == 2:
            scaling_col_2 = scaling_cols[1]
        scaling_col_s_1 = sdata.loc[:, scaling_col_1].to_numpy()
        scaling_col_s_2 = sdata.loc[:, scaling_col_2].to_numpy()
        scaling_col_j_1 = jdata.loc[:, scaling_col_1].to_numpy()
        scaling_col_j_2 = jdata.loc[:, scaling_col_2].to_numpy()

        Ys = (scaling_col_s_1 + scaling_col_s_2)[:, None] * Y[Ls, :]
        Ym = (scaling_col_j_1 + scaling_col_j_2)[:, None] * Y[Lm, :]
        Y = np.append(Ym, Ys, axis=0)
    Y = Y.flatten()

    ## Constraints ##
    cons_a = cons.QPConstrained(nl_adj, nk)
    cons_a.add_constraints(cons.FirmSum(b=firm_sizes_1, nt=1))
    cons_a.add_constraints(cons.WorkerSum(b=worker_sizes_1, nt=1))
    # Bound below at 0
    cons_a.add_constraints(cons.BoundedBelow(lb=0, nt=1))

    ### Solve ###
    # NOTE: don't need to constrain to be integers, because will become integers anyway
    cons_a.solve(Y, -Y, solver='linprog') # integrality=np.array([1])

    ## Extract optimal allocations ##
    alloc = np.reshape(np.round(cons_a.res, 0).astype(int, copy=False), (nl_adj, nk))

    ### Apply optimal allocations (make everyone a stayer) ###
    Ls = np.append(Lm, Ls)
    Lm = np.array([], dtype=int)
    if reallocation_scaling_col is None:
        gs = np.zeros(len(Ls), dtype=int)
        # gj = np.zeros((0, 0), dtype=int)
        idx = np.arange(len(Ls))
        for l in range(nl):
            ## Iterate over worker types ##
            idx_l = idx[Ls == l]
            # Use a random index so that the same workers aren't always given the same firms (important with control variables)
            rng.shuffle(idx_l)
            cum_firms_l = 0
            for k in range(nk):
                ## Iterate over firm classes ##
                gs[idx_l[cum_firms_l: cum_firms_l + alloc[l, k]]] = k
                cum_firms_l += alloc[l, k]
    else:
        gs = np.argmax(alloc, axis=1)

    return gs

def _simulate_types_wages(model, jdata, sdata, gj=None, gs=None, pk1=None, pk0=None, qi_j=None, qi_s=None, qi_cum_j=None, qi_cum_s=None, optimal_reallocation=False, reallocation_constraint_category=None, reallocation_scaling_col=None, worker_types_as_ids=True, simulate_wages=True, return_long_df=True, store_worker_types=True, weighted=True, rng=None):
    '''
    Using data and estimated BLM parameters, simulate worker types (and optionally wages).

    Arguments:
        model (BLMModel): BLM model with estimated parameters
        jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
        sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
        gj (NumPy Array or None): firm classes for movers in the first and second periods; if None, extract from jdata
        gs (NumPy Array or None): firm classes for stayers; if None, extract from sdata
        pk1 (NumPy Array or None): (use to assign workers to worker types probabilistically based on dgp-level probabilities) probability of being at each combination of firm types for movers; None if qi_j or qi_cum_j is not None
        pk0 (NumPy Array or None): (use to assign workers to worker types probabilistically based on dgp-level probabilities) probability of being at each firm type for stayers; None if qi_s or qi_cum_s is not None
        qi_j (NumPy Array or None): (use to assign workers to maximum probability worker type based on observation-level probabilities) probabilities for each mover observation to be each worker type; None if pk1 or qi_cum_j is not None
        qi_s (NumPy Array or None): (use to assign workers to maximum probability worker type based on observation-level probabilities) probabilities for each stayer observation to be each worker type; None if pk0 or qi_cum_s is not None
        qi_cum_j (NumPy Array or None): (use to assign workers to worker types probabilistically based on observation-level probabilities) cumulative probabilities for each mover observation to be each worker type; None if pk1 or qi_j is not None
        qi_cum_s (NumPy Array or None): (use to assign workers to worker types probabilistically based on observation-level probabilities) cumulative probabilities for each stayer observation to be each worker type; None if pk0 or qi_s is not None
        optimal_reallocation (bool or str): if not False, reallocate workers to new firms to maximize ('max') or minimize ('min') total output
        reallocation_constraint_category (str or None): specify categorical column to constrain reallocation so that workers must reallocate within their own category; if None, no constraints on how workers can reallocate
        reallocation_scaling_col (str or None): specify column to use to scale outcomes when computing optimal reallocation (i.e. multiply outcomes by an observation-level factor); if None, don't scale outcomes
        worker_types_as_ids (bool): if True, replace worker ids with simulated worker types
        simulate_wages (bool): if True, also simulate wages
        return_long_df (bool): if True, return data as a long-format BipartitePandas DataFrame; otherwise, return tuple of simulated types and wages
        store_worker_types (bool): if True, and return_long_df is True and worker_types_as_ids is False, then stores simulated worker types in the column labeled 'l'
        weighted (bool): if True, simulate using weights
        rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

    Returns:
        (BipartitePandas DataFrame or tuple of NumPy Arrays): if return_long_df is True, return BipartitePandas DataFrame with simulated data; if False, return tuple of (Lm --> vector of mover types; Ls --> vector of stayer types; yj --> tuple of wages for movers, where first element in first period wages and second element is second period wages; ys --> wages for stayers) if simulating wages; otherwise, (Lm, Ls)
    '''
    if optimal_reallocation and ((pk1 is not None) or (pk0 is not None)):
        raise ValueError('Cannot specify `optimal_reallocation` with `pk1` and `pk0`.')
    if optimal_reallocation and weighted:
        raise ValueError('Cannot specify `optimal_reallocation` with `weighted`.')
    if optimal_reallocation and (optimal_reallocation not in ['max', 'min']):
        raise ValueError(f"`optimal_reallocation` must be one of False, 'max', or 'min', but input specifies {optimal_reallocation!r}.")

    if rng is None:
        rng = np.random.default_rng(None)

    ## Unpack parameters ##
    nl, nk = model.nl, model.nk

    ## Firm classes ##
    if gj is None:
        gj = jdata.loc[:, ['g1', 'g2']].to_numpy().astype(int, copy=True)
    if gs is None:
        gs = sdata.loc[:, 'g1'].to_numpy().astype(int, copy=True)

    ## Weights ##
    wj1, wj2 = None, None
    ws = None
    if weighted:
        if jdata._col_included('w'):
            wj1, wj2 = jdata.loc[:, 'w1'].to_numpy(), jdata.loc[:, 'w2'].to_numpy()
        if sdata._col_included('w'):
            ws = sdata.loc[:, 'w1'].to_numpy()

    ## Simulate worker types ##
    Lm = tw.simblm._simulate_worker_types_movers(nl=nl, nk=nk, NNm=None, G1=gj[:, 0], G2=gj[:, 1], pk1=pk1, qi=qi_j, qi_cum=qi_cum_j, simulating_data=False, rng=rng)
    Ls = tw.simblm._simulate_worker_types_stayers(nl=nl, nk=nk, NNs=None, G=gs, pk0=pk0, qi=qi_s, qi_cum=qi_cum_s, simulating_data=False, rng=rng)

    if optimal_reallocation:
        ### Reallocate workers ###
        adata = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
        # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
        adata._set_attributes(jdata)
        if reallocation_constraint_category is not None:
            ## Reallocate within-category ##
            # Categorical column name
            cat_cons_col = to_list(sdata.col_reference_dict[reallocation_constraint_category])[0]

            # Number of categories
            n_cat = adata.n_unique_ids(reallocation_constraint_category)

            # Categorical column data
            cat_cons_col_a = adata.loc[:, cat_cons_col].to_numpy()
            cat_cons_col_s = sdata.loc[:, cat_cons_col].to_numpy()
            cat_cons_col_j = jdata.loc[:, cat_cons_col].to_numpy()

            for i in range(n_cat):
                cat_i_s = (cat_cons_col_s == i)
                cat_i_j = (cat_cons_col_j == i)
                if (cat_i_s.any() or cat_i_j.any()):
                    # Group might not show up in first period
                    gs_i = _optimal_reallocation(model, jdata.loc[cat_i_j, :], sdata.loc[cat_i_s, :], gj[cat_i_j, :], gs[cat_i_s], Lm[cat_i_j], Ls[cat_i_s], method=optimal_reallocation, reallocation_scaling_col=reallocation_scaling_col, rng=rng)
                    # Set G1/G2 and J1/J2
                    cat_i_a = (cat_cons_col_a == i)
                    adata.loc[cat_i_a, 'g1'], adata.loc[cat_i_a, 'g2'] = (gs_i, gs_i)
                    adata.loc[cat_i_a, 'j1'], adata.loc[cat_i_a, 'j2'] = (gs_i, gs_i)
            del cat_cons_col, cat_cons_col_a, cat_cons_col_s, cat_cons_col_j, cat_i_a, cat_i_j, cat_i_s, gs_i
        else:
            gs = _optimal_reallocation(model, jdata, sdata, gj, gs, Lm, Ls, method=optimal_reallocation, reallocation_scaling_col=reallocation_scaling_col, rng=rng)
            # Set G1/G2 and J1/J2
            adata.loc[:, 'g1'], adata.loc[:, 'g2'] = (gs, gs)
            adata.loc[:, 'j1'], adata.loc[:, 'j2'] = (gs, gs)
        ## Convert everyone to stayers ##
        sdata = adata
        # Set m
        sdata.loc[:, 'm'] = 0
        # Clear jdata
        jdata = pd.DataFrame()
        # Update Ls/Lm
        Ls = np.append(Lm, Ls)
        Lm = np.array([], dtype=int)
        # Update gs
        gs = sdata.loc[:, 'g1'].to_numpy().astype(int, copy=True)

    if simulate_wages:
        ## Simulate wages ##
        if len(jdata) > 0:
            yj = tw.simblm._simulate_wages_movers(jdata, Lm, blm_model=model, G1=gj[:, 0], G2=gj[:, 1], w1=wj1, w2=wj2, rng=rng)
        else:
            yj = (np.array([]), np.array([]))
        if len(sdata) > 0:
            ys = tw.simblm._simulate_wages_stayers(sdata, Ls, blm_model=model, G=gs, w=ws, rng=rng)
        else:
            ys = (np.array([]), np.array([]))

    if not return_long_df:
        if simulate_wages:
            return (Lm, Ls, yj, ys)
        return (Lm, Ls)

    ## Convert to BipartitePandas DataFrame ##
    bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
    # Set attributes from sdata, so that conversion to long works (since pd.concat drops attributes)
    bdf._set_attributes(sdata)

    with bpd.util.ChainedAssignment():
        if worker_types_as_ids:
            ## Update worker types ##
            bdf.loc[:, 'i'] = np.append(Lm, Ls)
        elif store_worker_types:
            bdf = bdf.add_column(
                'l',
                [np.append(Lm, Ls)],
                is_categorical=True,
                dtype='categorical',
                long_es_split=False,
                copy=False)
        if simulate_wages:
            ## Update wages ##
            bdf.loc[:, 'y1'] = np.append(yj[0], ys[0])
            bdf.loc[:, 'y2'] = np.append(yj[1], ys[1])
            del yj, ys
        del Lm, Ls

    # If simulating worker types, data is not sorted
    bdf = bdf.to_long(is_sorted=(not worker_types_as_ids), copy=False)

    return bdf

def _plot_worker_types_over_time(bdf, subplot, nl, subplot_title='', weighted=True):
    '''
    Generate a subplot for plot_worker_types_over_time().

    Arguments:
        bdf (BipartitePandas DataFrame): long format data
        subplot (MatPlotLib Subplot): subplot
        nl (int): number of worker types
        subplot_title (str): subplot title
        weighted (bool): if True, use weights
    '''
    weighted = weighted and bdf._col_included('w')
    qi_cols = [f'qi_' + 'i' * (l + 1) for l in range(nl)]

    ## Plot over time ##
    t_col = bdf.loc[:, 't'].to_numpy()
    all_t = np.unique(t_col)
    type_proportions = np.zeros([len(all_t), nl])
    for t_int, t_str in enumerate(all_t):
        bdf_t = bdf.loc[t_col == t_str, :]
        if weighted:
            w_t = bdf_t.loc[:, 'w'].to_numpy()
            # Number of observations per worker type per period
            type_proportions[t_int, :] = np.sum(w_t[:, None] * bdf_t.loc[:, qi_cols].to_numpy(), axis=0)
        else:
            # Number of observations per worker type per period
            type_proportions[t_int, :] = np.sum(bdf_t.loc[:, qi_cols].to_numpy(), axis=0)
        # Normalize to proportions
        type_proportions[t_int, :] /= type_proportions[t_int, :].sum()

    ## Compute cumulative sum ##
    type_props_cumsum = np.cumsum(type_proportions, axis=1)

    ## Plot ##
    x_axis = all_t.astype(str)
    subplot.bar(x_axis, type_proportions[:, 0])
    for l in range(1, nl):
        subplot.bar(x_axis, type_proportions[:, l], bottom=type_props_cumsum[:, l - 1])
    subplot.set_title(subplot_title)

def plot_worker_types_over_time(jdata, sdata, qi_j, qi_s, dynamic=False, breakdown_category=None, n_cols=3, category_labels=None, subset='all', xlabel='year', ylabel='type proportions', title='Worker type proportions over time', subplot_title='', weighted=True, dpi=None):
    '''
    Plot worker type proportions over time.

    Arguments:
        jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
        sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
        qi_j (NumPy Array): probabilities for each mover observation to be each worker type
        qi_s (NumPy Array): probabilities for each stayer observation to be each worker type
        dynamic (bool): if False, plotting estimates from static BLM; if True, plotting estimates from dynamic BLM
        breakdown_category (str or None): str specifies a categorical column, where for each group in the specified category, plot worker type proportions over time within that group; if None, plot worker type proportions over time for the entire dataset
        n_cols (int): (if breakdown_category is specified) number of subplot columns
        category_labels (list or None): (if breakdown_category is specified) specify labels for each category, where label indices should be based on sorted categories; if None, use values stored in data
        subset (str): 'all' plots a weighted average over movers and stayers; 'movers' plots movers; 'stayers' plots stayers
        xlabel (str): label for x-axis
        ylabel (str): label for y-axis
        title (str): plot title
        subplot_title (str): (if breakdown_category is specified) subplot title (subplots will be titled `subplot_title` + category, e.g. if `subplot_title`='k=', then subplots will be titled 'k=1', 'k=2', etc., or if `subplot_title`='', then subplots will be titled '1', '2', etc.)
        weighted (bool): if True, use weights
        dpi (float or None): dpi for plot
    '''
    if (not jdata._col_included('t')) or (not sdata._col_included('t')):
        raise ValueError('jdata and sdata must include time data.')

    ## Unpack parameters ##
    nl = qi_j.shape[1]
    if not dynamic:
        nt = 2
    else:
        nt = 4

    ## Add qi probabilities to dataframes ##
    if subset in ['movers', 'all']:
        for l in range(nl):
            jdata = jdata.add_column('qi_' + 'i' * (l + 1), [qi_j[:, l]] * nt, long_es_split=True, copy=True)
    if subset in ['stayers', 'all']:
        for l in range(nl):
            sdata = sdata.add_column('qi_' + 'i' * (l + 1), [qi_s[:, l]] * nt, long_es_split=True, copy=True)
    ## Convert to BipartitePandas DataFrame ##
    if subset == 'movers':
        bdf = jdata
    elif subset == 'stayers':
        bdf = sdata
    elif subset == 'all':
        bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
        # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
        bdf._set_attributes(jdata)
    bdf = bdf.to_long(is_sorted=True, copy=False)
    if isinstance(bdf, bpd.BipartiteLongCollapsed):
        bdf = bdf.uncollapse(is_sorted=True, copy=False)

    ## Plot ##
    if breakdown_category is None:
        n_rows = 1
        n_cols = 1
    else:
        cat_groups = np.array(sorted(bdf.unique_ids(breakdown_category)))
        if category_labels is None:
            category_labels = cat_groups + 1
        else:
            cat_order = np.argsort(category_labels)
            cat_groups = cat_groups[cat_order]
            category_labels = np.array(category_labels)[cat_order]
        n_cat = len(cat_groups)
        n_rows = n_cat // n_cols
        if n_rows * n_cols < n_cat:
            # If the bottom column won't be filled
            n_rows += 1

    ## Create subplots ##
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=True, dpi=dpi)
    if breakdown_category is None:
        _plot_worker_types_over_time(bdf=bdf, subplot=axs, nl=nl, subplot_title='', weighted=(weighted and (not dynamic)))
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel)
        axs.set_title(title)
    else:
        n_plots = 0
        for i, row in enumerate(axs):
            for j, ax in enumerate(row):
                if i * n_cols + j < n_cat:
                    # Keep category i * n_cols + j
                    cat_ij = cat_groups[i * n_cols + j]
                    subplot_title_ij = subplot_title + str(category_labels[i * n_cols + j])
                    _plot_worker_types_over_time(
                        bdf=bdf.loc[bdf.loc[:, breakdown_category].to_numpy() == cat_ij, :],
                        subplot=ax, nl=nl, subplot_title=subplot_title_ij,
                        weighted=(weighted and (not dynamic))
                    )
                    n_plots += 1
                else:
                    fig.delaxes(ax)

        fig.supxlabel(xlabel)
        fig.supylabel(ylabel)
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_worker_type_proportions_by_category(jdata, sdata, qi_j, qi_s, breakdown_category, category_labels=None, dynamic=False, subset='all', xlabel='category', ylabel='type proportions', title='Worker type proportions by category', dpi=None):
    '''
    Plot worker type proportions broken down by the given category. NOTE: should not use weights.

    Arguments:
        jdata (BipartitePandas DataFrame): event study, collapsed event study, or extended event study format labor data for movers
        sdata (BipartitePandas DataFrame): event study, collapsed event study, or extended event study format labor data for stayers
        qi_j (NumPy Array): probabilities for each mover observation to be each worker type
        qi_s (NumPy Array): probabilities for each stayer observation to be each worker type
        breakdown_category (str): categorical column, where worker type proportions are plotted for each group within the category
        category_labels (list or None): specify labels for each category, where label indices should be based on sorted categories; if None, use values stored in data
        dynamic (bool): if False, plotting estimates from static BLM; if True, plotting estimates from dynamic BLM
        subset (str): 'all' plots a weighted average over movers and stayers; 'movers' plots movers; 'stayers' plots stayers
        xlabel (str): label for x-axis
        ylabel (str): label for y-axis
        title (str): plot title
        dpi (float or None): dpi for plot
    '''
    ## Unpack parameters ##
    nl = qi_j.shape[1]
    if not dynamic:
        nt = 2
    else:
        nt = 4
    cat_groups = np.array(sorted(jdata.unique_ids(breakdown_category)))

    ## Add qi probabilities to dataframes ##
    if subset in ['movers', 'all']:
        for l in range(nl):
            jdata = jdata.add_column('qi_' + 'i' * (l + 1), [qi_j[:, l]] * nt, long_es_split=True, copy=True)
    if subset in ['stayers', 'all']:
        for l in range(nl):
            sdata = sdata.add_column('qi_' + 'i' * (l + 1), [qi_s[:, l]] * nt, long_es_split=True, copy=True)
    ## Convert to BipartitePandas DataFrame ##
    if subset == 'movers':
        bdf = jdata
    elif subset == 'stayers':
        bdf = sdata
    elif subset == 'all':
        bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
        # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
        bdf._set_attributes(jdata)
    bdf = bdf.to_long(is_sorted=True, copy=False)

    ## Compute proportions ##
    qi_cols = [f'qi_' + 'i' * (l + 1) for l in range(nl)]
    cat_col = bdf.loc[:, breakdown_category].to_numpy()
    type_proportions = np.zeros([len(cat_groups), nl])
    for i, cat_group in enumerate(cat_groups):
        bdf_i = bdf.loc[cat_col == cat_group, :]
        # Number of observations per worker type per group
        type_proportions[i, :] = np.sum(bdf_i.loc[:, qi_cols].to_numpy(), axis=0)
        # Normalize to proportions
        type_proportions[i, :] /= type_proportions[i, :].sum()

    if category_labels is not None:
        ## Reorder categories ##
        cat_order = np.argsort(category_labels)
        type_proportions = type_proportions[cat_order, :]

    ## Compute cumulative sum ##
    type_props_cumsum = np.cumsum(type_proportions, axis=1)

    ## Plot ##
    fig, ax = plt.subplots(dpi=dpi)
    if category_labels is None:
        x_axis = cat_groups.astype(str)
    else:
        x_axis = sorted(category_labels)
    ax.bar(x_axis, type_proportions[:, 0])
    for l in range(1, nl):
        ax.bar(x_axis, type_proportions[:, l], bottom=type_props_cumsum[:, l - 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()

def _plot_type_proportions_by_category(bdf, subplot, nl, nk, firm_order=None, subplot_title=''):
    '''
    Generate a subplot for plot_type_proportions_by_category(). NOTE: should not use weights.

    Arguments:
        bdf (BipartitePandas DataFrame): long format data
        subplot (MatPlotLib Subplot): subplot
        nl (int): number of worker types
        nk (int): number of firm classes
        firm_order (NumPy Array or None): sorted firm class order; None keeps the original firm order
        subplot_title (str): subplot title
    '''
    qi_cols = [f'qi_' + 'i' * (l + 1) for l in range(nl)]

    ## Plot type proportions ##
    GG = bdf.loc[:, 'g'].to_numpy()
    type_proportions = np.zeros([nk, nl])
    for k in range(nk):
        bdf_k = bdf.loc[GG == k, :]
        # Number of observations per worker type per firm class
        type_proportions[k, :] = np.sum(bdf_k.loc[:, qi_cols].to_numpy(), axis=0)
        # Normalize to proportions
        type_proportions[k, :] /= type_proportions[k, :].sum()

    if firm_order is not None:
        ## Reorder firms ##
        type_proportions = type_proportions[firm_order, :]

    ## Compute cumulative sum ##
    type_props_cumsum = np.cumsum(type_proportions, axis=1)

    ## Plot ##
    x_axis = np.arange(1, nk + 1).astype(str)
    subplot.bar(x_axis, type_proportions[:, 0])
    for l in range(1, nl):
        subplot.bar(x_axis, type_proportions[:, l], bottom=type_props_cumsum[:, l - 1])
    subplot.set_title(subplot_title)

def plot_type_proportions_by_category(jdata, sdata, qi_j, qi_s, breakdown_category, n_cols=3, category_labels=None, dynamic=False, subset='all', firm_order=None, xlabel='firm class k', ylabel='type proportions', title='Type proportions by category', subplot_title='category ', dpi=None):
    '''
    Plot worker-firm type proportions broken down by the given category. NOTE: should not use weights.

    Arguments:
        jdata (BipartitePandas DataFrame): event study, collapsed event study, or extended event study format labor data for movers
        sdata (BipartitePandas DataFrame): event study, collapsed event study, or extended event study format labor data for stayers
        qi_j (NumPy Array): probabilities for each mover observation to be each worker type
        qi_s (NumPy Array): probabilities for each stayer observation to be each worker type
        breakdown_category (str): categorical column, where worker type proportions are plotted for each group within the category
        n_cols (int): number of subplot columns
        category_labels (list or None): specify labels for each category, where label indices should be based on sorted categories; if None, use values stored in data
        dynamic (bool): if False, plotting estimates from static BLM; if True, plotting estimates from dynamic BLM
        subset (str): 'all' plots a weighted average over movers and stayers; 'movers' plots movers; 'stayers' plots stayers
        firm_order (NumPy Array or None): sorted firm class order; None keeps the original firm order
        xlabel (str): label for x-axis
        ylabel (str): label for y-axis
        title (str): plot title
        subplot_title (str): subplot title (subplots will be titled `subplot_title` + category, e.g. if `subplot_title`='k=', then subplots will be titled 'k=1', 'k=2', etc., or if `subplot_title`='', then subplots will be titled '1', '2', etc.)
        dpi (float or None): dpi for plot
    '''
    ## Unpack parameters ##
    nk = jdata.n_clusters()
    nl = qi_j.shape[1]
    if not dynamic:
        nt = 2
    else:
        nt = 4

    ## Add qi probabilities to dataframes ##
    if subset in ['movers', 'all']:
        for l in range(nl):
            jdata = jdata.add_column('qi_' + 'i' * (l + 1), [qi_j[:, l]] * nt, long_es_split=True, copy=True)
    if subset in ['stayers', 'all']:
        for l in range(nl):
            sdata = sdata.add_column('qi_' + 'i' * (l + 1), [qi_s[:, l]] * nt, long_es_split=True, copy=True)
    ## Convert to BipartitePandas DataFrame ##
    if subset == 'movers':
        bdf = jdata
    elif subset == 'stayers':
        bdf = sdata
    elif subset == 'all':
        bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
        # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
        bdf._set_attributes(jdata)
    bdf = bdf.to_long(is_sorted=True, copy=False)

    ## Plot ##
    cat_groups = np.array(sorted(jdata.unique_ids(breakdown_category)))
    if category_labels is None:
        category_labels = cat_groups + 1
    else:
        cat_order = np.argsort(category_labels)
        cat_groups = cat_groups[cat_order]
        category_labels = np.array(category_labels)[cat_order]
    n_cat = len(cat_groups)
    n_rows = n_cat // n_cols
    if n_rows * n_cols < n_cat:
        # If the bottom column won't be filled
        n_rows += 1

    ## Create subplots ##
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=False, sharey=True, dpi=dpi)
    n_plots = 0
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            if i * n_cols + j < n_cat:
                # Keep category i * n_cols + j
                cat_ij = cat_groups[i * n_cols + j]
                subplot_title_ij = subplot_title + str(category_labels[i * n_cols + j])
                _plot_type_proportions_by_category(
                    bdf=bdf.loc[bdf.loc[:, breakdown_category].to_numpy() == cat_ij, :],
                    subplot=ax, nl=nl, nk=nk, firm_order=firm_order,
                    subplot_title=subplot_title_ij
                )
                n_plots += 1
            else:
                fig.delaxes(ax)

    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_type_flows_between_categories(jdata, qi_j, breakdown_category, method='stacked', category_labels=None, dynamic=False, title='Worker flows', axis_label='category', subplot_title='worker type', n_cols=3, circle_scale=1, dpi=None, opacity=0.4, font_size=15):
    '''
    Plot flows of worker types between each group in a given category.

    Arguments:
        jdata (BipartitePandas DataFrame): event study, collapsed event study, or extended event study format labor data for movers
        qi_j (NumPy Array): probabilities for each mover observation to be each worker type
        breakdown_category (str): categorical column, where worker type proportions are plotted for each group within the category
        method (str): 'stacked' for stacked plot; 'sankey' for Sankey plot
        category_labels (list or None): specify labels for each category, where label indices should be based on sorted categories; if None, use values stored in data
        dynamic (bool): if False, plotting estimates from static BLM; if True, plotting estimates from dynamic BLM
        title (str): plot title
        axis_label (str): label for axes (for stacked)
        subplot_title (str): label for subplots (for stacked)
        n_cols (int): number of subplot columns (for stacked)
        circle_scale (float): size scale for circles (for stacked)
        dpi (float or None): dpi for plot (for stacked)
        opacity (float): opacity of flows (for Sankey)
        font_size (float): font size for plot (for Sankey)
    '''
    if method not in ['stacked', 'sankey']:
        raise ValueError(f"`method` must be one of 'stacked' or 'sankey', but input specifies {method!r}.")

    ## Extract parameters ##
    cat_groups = np.array(sorted(jdata.unique_ids(breakdown_category)))
    n_cat = len(cat_groups)
    nk = jdata.n_clusters()
    nl = qi_j.shape[1]
    g1 = f'{breakdown_category}1'
    g2 = f'{breakdown_category}'
    if not dynamic:
        g2 += '2'
    else:
        g2 += '4'
    G1 = jdata.loc[:, g1].to_numpy().astype(int, copy=False)
    G2 = jdata.loc[:, g2].to_numpy().astype(int, copy=False)
    NNm = jdata.groupby(g1)[g2].value_counts().unstack(fill_value=0).to_numpy()

    ### Compute pk1 ###
    ## Joint firm indicator ##
    KK = G1 + n_cat * G2
    KK2 = np.tile(KK, (nl, 1)).T
    KK3 = KK2 + n_cat ** 2 * np.arange(nl)
    KK2 = KK3.flatten()
    del KK3
    ## pk1 ##
    pk1 = np.bincount(KK2, weights=qi_j.flatten()).reshape(nl, n_cat ** 2).T
    # Normalize rows to sum to 1
    pk1 = DxM(1 / np.sum(pk1, axis=1), pk1)

    ## Compute worker flows ##
    reshaped_pk1 = np.reshape(pk1, (nk, nk, nl))
    mover_flows = (NNm.T * reshaped_pk1.T).T

    if category_labels is None:
        category_labels = cat_groups + 1
    else:
        ## Sort categories ##
        cat_order = np.argsort(category_labels)
        mover_flows = mover_flows[cat_order, :, :][:, cat_order, :]
        category_labels = np.array(category_labels)[cat_order]

    if method == 'stacked':
        ## Compute number of subplot rows ##
        n_rows = nl // n_cols
        if n_rows * n_cols < nl:
            # If the bottom column won't be filled
            n_rows += 1

        ## Create subplots ##
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True, dpi=dpi)

        ## Create axes ##
        x_vals, y_vals = np.meshgrid(np.arange(n_cat) + 1, np.arange(n_cat) + 1, indexing='ij')

        ## Generate plots ##
        l = 0
        for row in axs:
            for ax in row:
                if l < nl:
                    ax.scatter(x_vals, y_vals, s=(circle_scale * mover_flows[:, :, l]))
                    ax.set_title(f'{subplot_title} {l + 1}')
                    ax.grid()
                    l += 1
                else:
                    fig.delaxes(ax)

        plt.setp(axs, xticks=category_labels, yticks=category_labels)
        fig.supxlabel(f'{axis_label}, period 1')
        fig.supylabel(f'{axis_label}, period 2')
        fig.suptitle(f'{title}')
        plt.tight_layout()
        plt.show()
    elif method == 'sankey':
        colors = np.array(
            [
                [31, 119, 180],
                [255, 127, 14],
                [44, 160, 44],
                [214, 39, 40],
                [148, 103, 189],
                [140, 86, 75],
                [227, 119, 194],
                [127, 127, 127],
                [188, 189, 34],
                [23, 190, 207],
                [255, 0, 255]
            ]
        )

        ## Sankey with legend ##
        # Source: https://stackoverflow.com/a/76223740/17333120
        sankey = go.Sankey(
            # Define nodes
            node=dict(
                pad=15,
                thickness=1,
                line=dict(color='white', width=0),
                label=[f'k={k + 1}' for k in range(nk)] + [f'k={k + 1}' for k in range(nk)],
                color='white'
            ),
            link=dict(
                # Source firm
                source=np.repeat(np.arange(nk), nk * nl),
                # Destination firm
                target=np.tile(np.repeat(np.arange(nk), nl), nk) + nk,
                # Worker type
                label=[f'l={l + 1}' for _ in range(nk) for _ in range(nk) for l in range(nl)],
                # Worker flows
                value=mover_flows.flatten(),
                # Color (specify mean for each l, and for each k go from -80 below the mean to +80 above the mean)
                color=[f'rgba({str(list(np.minimum(255, np.maximum(0, colors[l, :] - 80) + 160 * k / (nk - 1))))[1: -1]}, {opacity})' for k in range(nk) for _ in range(nk) for l in range(nl)]
            )
        )

        legend = []
        for l in range(nl):
            legend.append(
                go.Scatter(
                    mode='markers',
                    x=[None],
                    y=[None],
                    marker=dict(color=f'rgba({str(list(colors[l, :]))[1: -1]}, {opacity})', symbol='square'),
                    name=f'$l={l + 1}$',
                )
            )

        traces = [sankey] + legend
        layout = go.Layout(
            showlegend=True,
            plot_bgcolor='rgba(0, 0, 0, 0)',
        )

        fig = go.Figure(data=traces, layout=layout)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title_text=title, font_size=font_size)
        fig.show()

class BLMModel:
    '''
    Class for estimating BLM using a single set of starting values.

    Arguments:
        params (ParamsDict): dictionary of parameters for BLM estimation. Run tw.blm_params().describe_all() for descriptions of all valid parameters.
        rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
    '''
    def __init__(self, params, rng=None):
        if rng is None:
            rng = np.random.default_rng(None)

        # Store parameters
        self.params = params.copy()
        self.rng = rng
        nl, nk = self.params.get_multiple(('nl', 'nk'))
        # Make sure that nk is specified
        if nk is None:
            raise ValueError(f"tw.blm_params() key 'nk' must be changed from the default value of None.")
        self.nl, self.nk = nl, nk

        # GPU
        self.gpu = params['gpu']

        # Log likelihood for movers
        self.lik1 = None
        # Path of log likelihoods for movers
        self.liks1 = np.array([])
        # Log likelihood for stayers
        self.lik0 = None
        # Path of log likelihoods for stayers
        self.liks0 = np.array([])
        # Connectedness measure of data
        self.connectedness = None
        # Parameter dimensions
        dims = (nl, nk)
        self.dims = dims

        ## Unpack control variable parameters ##
        cat_dict = self.params['categorical_controls']
        cts_dict = self.params['continuous_controls']
        ## Check if control variable parameters are None ##
        if cat_dict is None:
            cat_dict = {}
        if cts_dict is None:
            cts_dict = {}
        ## Create dictionary of all control variables ##
        controls_dict = cat_dict.copy()
        controls_dict.update(cts_dict.copy())
        ## Control variable ordering ##
        cat_cols = sorted(cat_dict.keys())
        cts_cols = sorted(cts_dict.keys())
        ## Store control variable attributes ##
        # Dictionaries #
        self.controls_dict = controls_dict
        self.cat_dict = cat_dict
        self.cts_dict = cts_dict
        # Lists #
        self.cat_cols = cat_cols
        self.cts_cols = cts_cols

        # Check that no control variables appear multiple times
        control_cols = cat_cols + cts_cols
        if len(control_cols) > len(set(control_cols)):
            for col in control_cols:
                if control_cols.count(col) > 1:
                    raise ValueError(f'Control variable names must be unique, but {col!r} appears multiple times.')

        # Check that all categorical variables specify 'n'
        for col, col_dict in cat_dict.items():
            if col_dict['n'] is None:
                raise ValueError(f"Categorical control variables must specify 'n', but column {col!r} does not.")

        # Check if there are any control variables
        self.any_controls = (len(control_cols) > 0)
        # Check if any control variables interact with worker type
        self.any_worker_type_interactions = any([col_dict['worker_type_interaction'] for col_dict in controls_dict.values()])
        # Check if any control variables don't interact with worker type
        self.any_non_worker_type_interactions = any([not col_dict['worker_type_interaction'] for col_dict in controls_dict.values()])

        ## Generate starting values ##
        a1_mu, a1_sig, a2_mu, a2_sig, s1_low, s1_high, s2_low, s2_high, pk1_prior, pk0_prior = self.params.get_multiple(('a1_mu', 'a1_sig', 'a2_mu', 'a2_sig', 's1_low', 's1_high', 's2_low', 's2_high', 'pk1_prior', 'pk0_prior'))
        s_lb = params['s_lower_bound']
        # Model for Y1 | Y2, l, k for movers and stayers
        self.A1 = rng.normal(loc=a1_mu, scale=a1_sig, size=dims)
        self.S1 = rng.uniform(low=np.maximum(s1_low, s_lb), high=s1_high, size=dims)
        # Model for Y4 | Y3, l, k for movers and stayers
        self.A2 = rng.normal(loc=a2_mu, scale=a2_sig, size=dims)
        self.S2 = rng.uniform(low=np.maximum(s2_low, s_lb), high=s2_high, size=dims)
        # Model for p(K | l, l') for movers
        self.pk1 = rng.dirichlet(alpha=np.ones(nl), size=nk * nk)
        if pk1_prior is not None:
            self.pk1 = (self.pk1 + pk1_prior) / 2
        # Model for p(K | l, l') for stayers
        if pk0_prior is None:
            self.pk0 = np.ones((nk, nl)) / nl
        else:
            self.pk0 = pk0_prior.copy()

        ### Control variables ###
        ## Categorical ##
        self.A1_cat = {col:
                rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=(nl, controls_dict[col]['n']))
                if controls_dict[col]['worker_type_interaction'] else
                rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=controls_dict[col]['n'])
            for col in cat_cols}
        self.A2_cat = {col:
                rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=(nl, controls_dict[col]['n']))
                if controls_dict[col]['worker_type_interaction'] else
                rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=controls_dict[col]['n'])
            for col in cat_cols}
        self.S1_cat = {col:
                rng.uniform(low=np.maximum(controls_dict[col]['s1_low'], s_lb), high=controls_dict[col]['s1_high'], size=(nl, controls_dict[col]['n']))
                if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=np.maximum(controls_dict[col]['s1_low'], s_lb), high=controls_dict[col]['s1_high'], size=controls_dict[col]['n'])
            for col in cat_cols}
        self.S2_cat = {col:
                rng.uniform(low=np.maximum(controls_dict[col]['s2_low'], s_lb), high=controls_dict[col]['s2_high'], size=(nl, controls_dict[col]['n']))
                if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=np.maximum(controls_dict[col]['s2_low'], s_lb), high=controls_dict[col]['s2_high'], size=controls_dict[col]['n'])
            for col in cat_cols}
        # # Stationary #
        # for col in cat_cols:
        #     if controls_dict[col]['stationary_A']:
        #         self.A2_cat[col] = self.A1_cat[col]
        #     if controls_dict[col]['stationary_S']:
        #         self.S2_cat[col] = self.S1_cat[col]
        ## Continuous ##
        self.A1_cts = {col:
                rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=nl)
                if controls_dict[col]['worker_type_interaction'] else
                rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=1)
            for col in cts_cols}
        self.A2_cts = {col:
                rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=nl)
                if controls_dict[col]['worker_type_interaction'] else
                rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=1)
            for col in cts_cols}
        self.S1_cts = {col:
                rng.uniform(low=np.maximum(controls_dict[col]['s1_low'], s_lb), high=controls_dict[col]['s1_high'], size=nl)
                if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=np.maximum(controls_dict[col]['s1_low'], s_lb), high=controls_dict[col]['s1_high'], size=1)
            for col in cts_cols}
        self.S2_cts = {col:
                rng.uniform(low=np.maximum(controls_dict[col]['s2_low'], s_lb), high=controls_dict[col]['s2_high'], size=nl)
                if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=np.maximum(controls_dict[col]['s2_low'], s_lb), high=controls_dict[col]['s2_high'], size=1)
            for col in cts_cols}
        # # Stationary #
        # for col in cts_cols:
        #     if controls_dict[col]['stationary_A']:
        #         self.A2_cts[col] = self.A1_cts[col]
        #     if controls_dict[col]['stationary_S']:
        #         self.S2_cts[col] = self.S1_cts[col]

        # for l in range(nl):
        #     self.A1[l] = np.sort(self.A1[l], axis=0)
        #     self.A2[l] = np.sort(self.A2[l], axis=0)

        # if self.fixb:
        #     self.A2 = np.mean(self.A2, axis=1) + self.A1 - np.mean(self.A1, axis=1)

        # if self.stationary:
        #     self.A2 = self.A1

        ## NNm and NNs ##
        self.NNm = None
        self.NNs = None

    def _gen_constraints(self, min_firm_type):
        '''
        Generate constraints for estimating A and S in fit_movers().

        Arguments:
            min_firm_type (int): lowest firm type

        Returns:
            (tuple of constraints): (cons_a --> constraints for base A1 and A2, cons_s --> constraints for base S1 and S2, cons_a_dict --> constraints for A1 and A2 for control variables, cons_s_dict --> controls for S1 and S2 for control variables)
        '''
        # Unpack parameters
        params = self.params
        nl, nk = self.nl, self.nk
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        controls_dict = self.controls_dict

        ## General ##
        cons_a = cons.QPConstrained(nl, nk)
        cons_s = cons.QPConstrained(nl, nk)
        cons_s.add_constraints(cons.BoundedBelow(lb=params['s_lower_bound']))

        if params['cons_a'] is not None:
            cons_a.add_constraints(params['cons_a'])
        if params['cons_a_all'] is not None:
            cons_a.add_constraints(params['cons_a_all'])
        if params['cons_s'] is not None:
            cons_s.add_constraints(params['cons_s'])
        if params['cons_s_all'] is not None:
            cons_s.add_constraints(params['cons_s_all'])

        ## Control variables ##
        cons_a_dict = {}
        cons_s_dict = {}
        for col in cat_cols:
            col_dict = controls_dict[col]
            cons_a_dict[col] = cons.QPConstrained(nl, col_dict['n'])
            cons_s_dict[col] = cons.QPConstrained(nl, col_dict['n'])
        for col in cts_cols:
            col_dict = controls_dict[col]
            cons_a_dict[col] = cons.QPConstrained(nl, 1)
            cons_s_dict[col] = cons.QPConstrained(nl, 1)
        for col in cat_cols + cts_cols:
            cons_s_dict[col].add_constraints(cons.BoundedBelow(lb=params['s_lower_bound']))

            if not controls_dict[col]['worker_type_interaction']:
                cons_a_dict[col].add_constraints(cons.NoWorkerTypeInteraction())
                cons_s_dict[col].add_constraints(cons.NoWorkerTypeInteraction())

            if controls_dict[col]['cons_a'] is not None:
                cons_a_dict[col].add_constraints(controls_dict[col]['cons_a'])
            if params['cons_a_all'] is not None:
                cons_a_dict[col].add_constraints(params['cons_a_all'])
            if controls_dict[col]['cons_s'] is not None:
                cons_s_dict[col].add_constraints(controls_dict[col]['cons_s'])
            if params['cons_s_all'] is not None:
                cons_s_dict[col].add_constraints(params['cons_s_all'])

        ## Normalization ##
        if len(cat_cols) > 0:
            # Check if any columns interact with worker type and/or are stationary (tv stands for time-varying, which is equivalent to non-stationary; and wi stands for worker-interaction)
            any_tv_nwi = False
            any_tnv_nwi = False
            any_tv_wi = False
            any_tnv_wi = False
            for col in cat_cols:
                # Check if column is stationary
                is_stationary = False
                if controls_dict[col]['cons_a'] is not None:
                    for subcons_a in to_list(controls_dict[col]['cons_a']):
                        if isinstance(subcons_a, cons.Stationary):
                            is_stationary = True
                            break

                if controls_dict[col]['worker_type_interaction']:
                    # If the column interacts with worker types
                    if is_stationary:
                        any_tnv_wi = True
                    else:
                        any_tv_wi = True
                        break
                else:
                    # If the column doesn't interact with worker types (this requires a constraint)
                    if is_stationary:
                        any_tnv_nwi = True
                    else:
                        any_tv_nwi = True

            ## Determine primary and second periods ##
            primary_period_dict = {
                'first': 0,
                'second': 1,
                'all': range(2)
            }
            secondary_period_dict = {
                'first': 1,
                'second': 0,
                'all': 0
            }
            pp = primary_period_dict[params['primary_period']]
            sp = secondary_period_dict[params['primary_period']]
            ### Add constraints ###
            ## Monotonic worker types ##
            cons_a.add_constraints(cons.MonotonicMean(md=params['d_mean_worker_effect'], cross_period_mean=True, nnt=pp))
            if params['normalize']:
                ## Lowest firm type ##
                if params['force_min_firm_type'] and params['force_min_firm_type_constraint']:
                    cons_a.add_constraints(cons.MinFirmType(min_firm_type=min_firm_type, md=params['d_mean_firm_effect'], is_min=True, cross_period_mean=True, nnt=pp))
                ## Normalize ##
                if any_tv_wi:
                    # Normalize everything
                    cons_a.add_constraints(cons.NormalizeAllWorkerTypes(min_firm_type=min_firm_type, nnt=range(2)))
                else:
                    if any_tnv_wi:
                        # Normalize primary period
                        cons_a.add_constraints(cons.NormalizeAllWorkerTypes(min_firm_type=min_firm_type, cross_period_normalize=True, nnt=pp))
                        if any_tv_nwi:
                            # Normalize lowest type pair from secondary period
                            cons_a.add_constraints(cons.NormalizeLowest(min_firm_type=min_firm_type, nnt=sp))
                    else:
                        if any_tv_nwi:
                            # Normalize lowest type pair in both periods
                            cons_a.add_constraints(cons.NormalizeLowest(min_firm_type=min_firm_type, nnt=range(2)))
                        elif any_tnv_nwi:
                            # Normalize lowest type pair in primary period
                            cons_a.add_constraints(cons.NormalizeLowest(min_firm_type=min_firm_type, cross_period_normalize=True, nnt=pp))

        return (cons_a, cons_s, cons_a_dict, cons_s_dict)

    def sorted_firm_classes(self):
        '''
        Return list of sorted firm classes based on estimated parameters.

        Returns:
            (NumPy Array): new firm class order
        '''
        ## Unpack attributes ##
        params = self.params
        A1, A2 = self.A1, self.A2

        ## Primary period ##
        if params['primary_period'] == 'first':
            A_mean = A1
        elif params['primary_period'] == 'second':
            A_mean = A2
        elif params['primary_period'] == 'all':
            A_mean = (A1 + A2) / 2

        return np.mean(A_mean, axis=0).argsort()

    def _sort_parameters(self, A1, A2, S1=None, S2=None, A1_cat=None, A2_cat=None, S1_cat=None, S2_cat=None, A1_cts=None, A2_cts=None, S1_cts=None, S2_cts=None, pk1=None, pk0=None, NNm=None, NNs=None, sort_firm_types=False, reverse=False):
        '''
        Sort parameters by worker type order (and optionally firm type order).

        Arguments:
            A1 (NumPy Array): mean of fixed effects in the first period
            A2 (NumPy Array): mean of fixed effects in the second period
            S1 (NumPy Array or None): standard deviation of fixed effects in the first period; if None, S1 is not sorted or returned
            S2 (NumPy Array or None): standard deviation of fixed effects in the second period; if None, S2 is not sorted or returned
            A1_cat (dict of NumPy Arrays or None): dictionary linking column names to the mean of fixed effects in the first period for categorical control variables
            ; if None, A1_cat is not sorted or returned
            A2_cat (dict of NumPy Arrays or None): dictionary linking column names to the mean of fixed effects in the second period for categorical control variables; if None, A2_cat is not sorted or returned
            S1_cat (dict of NumPy Arrays or None): dictionary linking column names to the standard deviation of fixed effects in the first period for categorical control variables; if None, S1_cat is not sorted or returned
            S2_cat (dict of NumPy Arrays or None): dictionary linking column names to the standard deviation of fixed effects in the second period for categorical control variables; if None, S2_cat is not sorted or returned
            A1_cts (dict of NumPy Arrays or None): dictionary linking column names to the mean of coefficients in the first period for continuous control variables; if None, A1_cts is not sorted or returned
            A2_cts (dict of NumPy Arrays or None): dictionary linking column names to the mean of coefficients in the second period for continuous control variables; if None, A2_cts is not sorted or returned
            S1_cts (dict of NumPy Arrays or None): dictionary linking column names to the standard deviation of coefficients in the first period for continuous control variables; if None, S1_cts is not sorted or returned
            S2_cts (dict of NumPy Arrays or None): dictionary linking column names to the standard deviation of coefficients in the second period for continuous control variables; if None, S2_cts is not sorted or returned
            pk1 (NumPy Array or None): probability of being at each combination of firm types for movers; if None, pk1 is not sorted or returned
            pk0 (NumPy Array or None): probability of being at each firm type for stayers; if None, pk0 is not sorted or returned
            NNm (NumPy Array or None): the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
            NNs (NumPy Array or None): the number of stayers at each firm type (e.g. entry (1) gives the number of stayers at firm type 1)
            sort_firm_types (bool): if True, also sort by firm type order
            reverse (bool): if True, sort in reverse order

        Returns (tuple of NumPy Arrays and dicts of NumPy Arrays): sorted parameters that are not None (A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, pk0)
        '''
        # Copy parameters
        A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, pk0, NNm, NNs = copy.deepcopy((A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, pk0, NNm, NNs))
        # Unpack attributes
        params = self.params
        nl, nk = self.nl, self.nk
        controls_dict = self.controls_dict
        ## Primary period ##
        if params['primary_period'] == 'first':
            A_mean = A1
        elif params['primary_period'] == 'second':
            A_mean = A2
        elif params['primary_period'] == 'all':
            A_mean = (A1 + A2) / 2
        # ## Compute sum of all effects ##
        # A_sum = self.A1 + self.A2
        # for control_dict in (self.A1_cat, self.A2_cat):
        #     for control_col, control_array in control_dict.items():
        #         if controls_dict[control_col]['worker_type_interaction']:
        #             A_sum = (A_sum.T + np.mean(control_array, axis=1)).T
        ## Sort worker types ##
        worker_type_order = np.mean(A_mean, axis=1).argsort()
        if reverse:
            worker_type_order = list(reversed(worker_type_order))
        if np.any(worker_type_order != np.arange(nl)):
            # Sort if out of order
            A1 = A1[worker_type_order, :]
            A2 = A2[worker_type_order, :]
            if S1 is not None:
                S1 = S1[worker_type_order, :]
            if S2 is not None:
                S2 = S2[worker_type_order, :]
            if pk1 is not None:
                pk1 = pk1[:, worker_type_order]
            if pk0 is not None:
                pk0 = pk0[:, worker_type_order]
            # Sort control variables #
            for control_dict in (A1_cat, A2_cat, S1_cat, S2_cat):
                if control_dict is not None:
                    for control_col, control_array in control_dict.items():
                        if controls_dict[control_col]['worker_type_interaction']:
                            control_dict[control_col] = control_array[worker_type_order, :]
            for control_dict in (A1_cts, A2_cts, S1_cts, S2_cts):
                if control_dict is not None:
                    for control_col, control_array in control_dict.items():
                        if controls_dict[control_col]['worker_type_interaction']:
                            control_dict[control_col] = control_array[worker_type_order]

        if sort_firm_types:
            ## Sort firm types ##
            firm_type_order = np.mean(A_mean, axis=0).argsort()
            if reverse:
                firm_type_order = list(reversed(firm_type_order))
            if np.any(firm_type_order != np.arange(nk)):
                # Sort if out of order
                A1 = A1[:, firm_type_order]
                A2 = A2[:, firm_type_order]
                if S1 is not None:
                    S1 = S1[:, firm_type_order]
                if S2 is not None:
                    S2 = S2[:, firm_type_order]
                if pk0 is not None:
                    pk0 = pk0[firm_type_order, :]
                if pk1 is not None:
                    # # Reorder part 1: e.g. nk=2, and type 0 > type 1, then 0, 1, 2, 3 would reorder to 1, 0, 3, 2 (i.e. reorder within groups)
                    # pk1_order_1 = np.tile(firm_type_order, nk) + nk * np.repeat(range(nk), nk)
                    # pk1 = pk1[pk1_order_1, :]
                    # # Reorder part 2: e.g. nk=2, and type 0 > type 1, then 0, 1, 2, 3 would reorder to 2, 3, 0, 1 (i.e. reorder between groups)
                    # pk1_order_2 = nk * np.repeat(firm_type_order, nk) + np.tile(range(nk), nk)
                    # pk1 = pk1[pk1_order_2, :]
                    adj_pk1 = np.reshape(pk1, (nk, nk, nl))
                    adj_pk1 = adj_pk1[firm_type_order, :, :][:, firm_type_order, :]
                    pk1 = np.reshape(adj_pk1, (nk * nk, nl))
                if NNm is not None:
                    NNm = NNm[firm_type_order, :]
                    NNm = NNm[:, firm_type_order]
                if NNs is not None:
                    NNs = NNs[firm_type_order]

        res = [a for a in (A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, pk0, NNm, NNs) if a is not None]

        if len(res) == 1:
            res = res[0]

        return res

    def _sum_by_non_nl(self, ni, C1, C2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, compute_A=True, compute_S=True):
        '''
        Compute A1_sum/A2_sum/S1_sum_sq/S2_sum_sq for non-worker-interaction terms.

        Arguments:
            ni (int): number of observations
            C1 (dict of NumPy Arrays): dictionary linking column names to control variable data for the first period
            C2 (dict of NumPy Arrays): dictionary linking column names to control variable data for the second period
            A1_cat (dict of NumPy Arrays): dictionary linking column names to the mean of fixed effects in the first period for categorical control variables
            A2_cat (dict of NumPy Arrays): dictionary linking column names to the mean of fixed effects in the second period for categorical control variables
            S1_cat (dict of NumPy Arrays): dictionary linking column names to the standard deviation of fixed effects in the first period for categorical control variables
            S2_cat (dict of NumPy Arrays): dictionary linking column names to the standard deviation of fixed effects in the second period for categorical control variables
            A1_cts (dict of NumPy Arrays): dictionary linking column names to the mean of coefficients in the first period for continuous control variables
            A2_cts (dict of NumPy Arrays): dictionary linking column names to the mean of coefficients in the second period for continuous control variables
            S1_cts (dict of NumPy Arrays): dictionary linking column names to the standard deviation of coefficients in the first period for continuous control variables
            S2_cts (dict of NumPy Arrays): dictionary linking column names to the standard deviation of coefficients in the second period for continuous control variables
            compute_A (bool): if True, compute and return A terms
            compute_S (bool): if True, compute and return S terms

        Returns:
            (tuple of NumPy Arrays): (A1_sum, A2_sum, S1_sum_sq, S2_sum_sq), where each term gives the sum of estimated effects for control variables that do not interact with worker type (A terms are dropped if compute_A=False, and S terms are dropped if compute_S=False)
        '''
        if (not compute_A) and (not compute_S):
            raise ValueError('compute_A=False and compute_S=False. Must specify at least one to be True.')

        if not self.any_non_worker_type_interactions:
            # If all control variables interact with worker type
            if compute_A and compute_S:
                return [0] * 4
            return [0] * 2

        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        controls_dict = self.controls_dict

        if compute_A:
            A1_sum = np.zeros(ni)
            A2_sum = np.zeros(ni)
        if compute_S:
            S1_sum_sq = np.zeros(ni)
            S2_sum_sq = np.zeros(ni)

        ## Categorical ##
        for col in cat_cols:
            if not controls_dict[col]['worker_type_interaction']:
                if compute_A:
                    A1_sum += A1_cat[col][C1[col]]
                    A2_sum += A2_cat[col][C2[col]]
                if compute_S:
                    S1_sum_sq += (S1_cat[col] ** 2)[C1[col]]
                    S2_sum_sq += (S2_cat[col] ** 2)[C2[col]]
        ## Continuous ##
        for col in cts_cols:
            if not controls_dict[col]['worker_type_interaction']:
                if compute_A:
                    A1_sum += A1_cts[col] * C1[col]
                    A2_sum += A2_cts[col] * C2[col]
                if compute_S:
                    S1_sum_sq += S1_cts[col] ** 2
                    S2_sum_sq += S2_cts[col] ** 2

        if compute_A and compute_S:
            return (A1_sum, A2_sum, S1_sum_sq, S2_sum_sq)
        if compute_A:
            return (A1_sum, A2_sum)
        if compute_S:
            return (S1_sum_sq, S2_sum_sq)

    def _sum_by_nl_l(self, ni, l, C1, C2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, compute_A=True, compute_S=True):
        '''
        Compute A1_sum/A2_sum/S1_sum_sq/S2_sum_sq to account for worker-interaction terms for a particular worker type.

        Arguments:
            ni (int): number of observations
            l (int): worker type (must be in range(0, nl))
            C1 (dict of NumPy Arrays): dictionary linking column names to control variable data for the first period
            C2 (dict of NumPy Arrays): dictionary linking column names to control variable data for the second period
            A1_cat (dict of NumPy Arrays): dictionary linking column names to the mean of fixed effects in the first period for categorical control variables
            A2_cat (dict of NumPy Arrays): dictionary linking column names to the mean of fixed effects in the second period for categorical control variables
            S1_cat (dict of NumPy Arrays): dictionary linking column names to the standard deviation of fixed effects in the first period for categorical control variables
            S2_cat (dict of NumPy Arrays): dictionary linking column names to the standard deviation of fixed effects in the second period for categorical control variables
            A1_cts (dict of NumPy Arrays): dictionary linking column names to the mean of coefficients in the first period for continuous control variables
            A2_cts (dict of NumPy Arrays): dictionary linking column names to the mean of coefficients in the second period for continuous control variables
            S1_cts (dict of NumPy Arrays): dictionary linking column names to the standard deviation of coefficients in the first period for continuous control variables
            S2_cts (dict of NumPy Arrays): dictionary linking column names to the standard deviation of coefficients in the second period for continuous control variables
            compute_A (bool): if True, compute and return A terms
            compute_S (bool): if True, compute and return S terms

        Returns:
            (tuple of NumPy Arrays): (A1_sum_l, A2_sum_l, S1_sum_sq_l, S2_sum_sq_l), where each term gives the sum of estimated effects for control variables that interact with worker type, specifically for worker type l (A terms are dropped if compute_A=False, and S terms are dropped if compute_S=False)
        '''
        if (not compute_A) and (not compute_S):
            raise ValueError('compute_A=False and compute_S=False. Must specify at least one to be True.')

        if not self.any_worker_type_interactions:
            # If no control variables interact with worker type
            if compute_A and compute_S:
                return [0] * 4
            return [0] * 2

        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        controls_dict = self.controls_dict

        if compute_A:
            A1_sum_l = np.zeros(ni)
            A2_sum_l = np.zeros(ni)
        if compute_S:
            S1_sum_sq_l = np.zeros(ni)
            S2_sum_sq_l = np.zeros(ni)

        ## Categorical ##
        for col in cat_cols:
            if controls_dict[col]['worker_type_interaction']:
                if compute_A:
                    A1_sum_l += A1_cat[col][l, C1[col]]
                    A2_sum_l += A2_cat[col][l, C2[col]]
                if compute_S:
                    S1_sum_sq_l += (S1_cat[col][l, :] ** 2)[C1[col]]
                    S2_sum_sq_l += (S2_cat[col][l, :] ** 2)[C2[col]]
        ## Continuous ##
        for col in cts_cols:
            if controls_dict[col]['worker_type_interaction']:
                if compute_A:
                    A1_sum_l += A1_cts[col][l] * C1[col]
                    A2_sum_l += A2_cts[col][l] * C2[col]
                if compute_S:
                    S1_sum_sq_l += S1_cts[col][l] ** 2
                    S2_sum_sq_l += S2_cts[col][l] ** 2

        if compute_A and compute_S:
            return (A1_sum_l, A2_sum_l, S1_sum_sq_l, S2_sum_sq_l)
        if compute_A:
            return (A1_sum_l, A2_sum_l)
        if compute_S:
            return (S1_sum_sq_l, S2_sum_sq_l)

    def fit_movers(self, jdata, compute_NNm=True):
        '''
        EM algorithm for movers.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # Unpack parameters
        params = self.params

        if params['force_min_firm_type']:
            ## If forcing minimum firm type ##
            if params['return_qi']:
                raise ValueError("Cannot return qi if 'force_min_firm_type'=True.")
            # Unpack parameters
            nk = self.nk

            # Best model
            best_model = None
            for k in range(nk):
                # Copy initial guesses
                A1, A2, S1, S2 = copy.deepcopy((self.A1, self.A2, self.S1, self.S2))
                A1_cat, A2_cat, S1_cat, S2_cat = copy.deepcopy((self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat))
                A1_cts, A2_cts, S1_cts, S2_cts = copy.deepcopy((self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts))
                pk1 = copy.deepcopy(self.pk1)

                ## Estimate with min_firm_type == k ##
                blm_k = BLMModel(params)
                # Set initial guesses
                blm_k.A1, blm_k.A2, blm_k.S1, blm_k.S2 = A1, A2, S1, S2
                blm_k.A1_cat, blm_k.A2_cat, blm_k.S1_cat, blm_k.S2_cat = A1_cat, A2_cat, S1_cat, S2_cat
                blm_k.A1_cts, blm_k.A2_cts, blm_k.S1_cts, blm_k.S2_cts = A1_cts, A2_cts, S1_cts, S2_cts
                blm_k.pk1 = pk1
                # Fit estimator
                blm_k._fit_movers(jdata=jdata, compute_NNm=False, min_firm_type=k)

                ## Store best estimator ##
                if (best_model is None) or (blm_k.lik1 > best_model.lik1):
                    best_model = blm_k
            ## Update parameters with best model ##
            self.A1, self.A2, self.S1, self.S2 = best_model.A1, best_model.A2, best_model.S1, best_model.S2
            self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat = best_model.A1_cat, best_model.A2_cat, best_model.S1_cat, best_model.S2_cat
            self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts = best_model.A1_cts, best_model.A2_cts, best_model.S1_cts, best_model.S2_cts
            self.pk1 = best_model.pk1
            self.liks1, self.lik1 = best_model.liks1, best_model.lik1

            if compute_NNm:
                # Update NNm
                self.NNm = jdata.groupby('g1')['g2'].value_counts().unstack(fill_value=0).to_numpy()

        else:
            # If not forcing minimum firm type
            if params['return_qi']:
                return self._fit_movers(jdata=jdata, compute_NNm=compute_NNm, min_firm_type=None)
            else:
                self._fit_movers(jdata=jdata, compute_NNm=compute_NNm, min_firm_type=None)

    def _fit_movers(self, jdata, compute_NNm=True, min_firm_type=None):
        '''
        Wrapped EM algorithm for movers.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
            min_firm_type (int or None): if params['force_min_firm_type'] == True, gives the firm type to force as the lowest firm type; if params['force_min_firm_type'] == False, this parameter is not used
        '''
        # Unpack parameters
        params = self.params
        nl, nk, ni = self.nl, self.nk, jdata.shape[0]
        A1, A2, S1, S2 = self.A1, self.A2, self.S1, self.S2
        A1_cat, A2_cat, S1_cat, S2_cat = self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat
        A1_cts, A2_cts, S1_cts, S2_cts = self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        cat_dict, cts_dict = self.cat_dict, self.cts_dict
        controls_dict = self.controls_dict
        any_controls, any_non_worker_type_interactions = self.any_controls, self.any_non_worker_type_interactions

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        G1 = jdata.loc[:, 'g1'].to_numpy().astype(int, copy=False)
        G2 = jdata.loc[:, 'g2'].to_numpy().astype(int, copy=False)

        # Weights
        if params['weighted'] and jdata._col_included('w'):
            W1 = jdata.loc[:, 'w1'].to_numpy()
            W2 = jdata.loc[:, 'w2'].to_numpy()
        else:
            W1 = 1
            W2 = 1
        W = np.sqrt(W1 * W2)

        ## Control variables ##
        C1 = {}
        C2 = {}
        for i, col in enumerate(cat_cols + cts_cols):
            # Get subcolumns associated with col
            subcols = to_list(jdata.col_reference_dict[col])
            n_subcols = len(subcols)
            if n_subcols == 1:
                # If column is constant over time
                subcol_1 = subcols[0]
                subcol_2 = subcols[0]
            elif n_subcols == 2:
                # If column can change over time
                subcol_1 = subcols[0]
                subcol_2 = subcols[1]
            else:
                raise NotImplementedError(f'Column names must have either one or two associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
            if i < len(cat_cols):
                # Categorical
                C1[col] = jdata.loc[:, subcol_1].to_numpy().astype(int, copy=False)
                C2[col] = jdata.loc[:, subcol_2].to_numpy().astype(int, copy=False)
            else:
                # Continuous
                C1[col] = jdata.loc[:, subcol_1].to_numpy()
                C2[col] = jdata.loc[:, subcol_2].to_numpy()

        ## Joint firm indicator ##
        KK = G1 + nk * G2
        KK2 = np.tile(KK, (nl, 1)).T
        KK3 = KK2 + nk * nk * np.arange(nl)
        KK2 = KK3.flatten()
        del KK3

        # Matrix of prior probabilities
        pk1 = self.pk1
        # Matrix of posterior probabilities
        qi = np.ones(shape=(ni, nl))
        # Log pdfs
        lp = np.zeros(shape=(ni, nl))
        # Log likelihood for movers
        lik1 = None
        # Path of log likelihoods for movers
        liks1 = []
        prev_lik = np.inf
        # Fix error from bad initial guesses causing probabilities to be too low
        d_prior = params['d_prior_movers']
        # Track minimum firm type to check whether estimator stuck in a loop
        min_firm_types = []
        # Whether results should be stored
        store_res = True

        ## Sort ##
        A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, self.pk0 = self._sort_parameters(A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, self.pk0)

        ## Constraints ##
        if params['force_min_firm_type']:
            # If forcing minimum firm type
            prev_min_firm_type = min_firm_type
            min_firm_type = min_firm_type
        else:
            # If not forcing minimum firm type
            prev_min_firm_type = tw.simblm._min_firm_type(A1, A2, params['primary_period'])
        cons_a, cons_s, cons_a_dict, cons_s_dict = self._gen_constraints(prev_min_firm_type)

        for iter in range(params['n_iters_movers']):
            # ---------- E-Step ----------
            # We compute the posterior probabilities for each row
            # We iterate over the worker types, should not be be too costly since the vector is quite large within each iteration
            log_pk1 = np.log(pk1)
            if any_controls:
                ## Account for control variables ##
                if iter == 0:
                    A1_sum, A2_sum, S1_sum_sq, S2_sum_sq = self._sum_by_non_nl(ni=ni, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts)
                else:
                    S1_sum_sq, S2_sum_sq = self._sum_by_non_nl(ni=ni, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts, compute_A=False)

                for l in range(nl):
                    # Update A1_sum/A2_sum/S1_sum_sq/S2_sum_sq to account for worker-interaction terms
                    A1_sum_l, A2_sum_l, S1_sum_sq_l, S2_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts)
                    lp1 = lognormpdf(Y1, A1[l, G1] + A1_sum + A1_sum_l, var=(S1[l, :] ** 2)[G1] + S1_sum_sq + S1_sum_sq_l, gpu=self.gpu)
                    lp2 = lognormpdf(Y2, A2[l, G2] + A2_sum + A2_sum_l, var=(S2[l, :] ** 2)[G2] + S2_sum_sq + S2_sum_sq_l, gpu=self.gpu)
                    lp[:, l] = log_pk1[KK, l] + W1 * lp1 + W2 * lp2
            else:
                for l in range(nl):
                    lp1 = fast_lognormpdf(Y1, A1[l, :], S1[l, :], G1, gpu=self.gpu)
                    lp2 = fast_lognormpdf(Y2, A2[l, :], S2[l, :], G2, gpu=self.gpu)
                    lp[:, l] = log_pk1[KK, l] + W1 * lp1 + W2 * lp2
            del log_pk1, lp1, lp2

            # We compute log sum exp to get likelihoods and probabilities
            lse_lp = logsumexp(lp, axis=1, gpu=self.gpu)
            qi = exp_(lp.T - lse_lp, gpu=self.gpu).T
            if params['return_qi']:
                return qi
            lik1 = lse_lp.mean()
            del lse_lp
            if (iter > 0) and params['update_pk1']:
                # Account for Dirichlet prior
                lik_prior = (d_prior - 1) * np.sum(np.log(pk1))
                lik1 += lik_prior
            liks1.append(lik1)
            if params['verbose'] == 3:
                print('loop {}, liks {}'.format(iter, lik1))

            if not params['force_min_firm_type']:
                # If not forcing minimum firm type, compute lowest firm type
                min_firm_type = tw.simblm._min_firm_type(A1, A2, params['primary_period'])

            if ((abs(lik1 - prev_lik) < params['threshold_movers']) and (min_firm_type == prev_min_firm_type)):
                # Break loop
                break
            if iter == (params['n_iters_movers'] - 1):
                print(f"Maximum iterations reached for movers. It is recommended to increase `n_iters_movers` from its current value of {params['n_iters_movers']}.")
            prev_lik = lik1
            prev_min_firm_type = min_firm_type

            if (len(cat_cols) > 0) and (iter >= params['start_cycle_check_threshold']) and (not params['force_min_firm_type']):
                ## Check whether estimator is stuck in a cycle ##
                min_firm_types.append(min_firm_type)
                if len(min_firm_types) >= params['cycle_check_n_obs']:
                    if len(min_firm_types) > params['cycle_check_n_obs']:
                        min_firm_types = min_firm_types[1:]
                    if np.all(np.array(min_firm_types[1:]) != np.array(min_firm_types[: -1])):
                        # Check if estimator is changing minimum firm type every loop
                        warnings.warn("Estimator is stuck in a cycle of minimum firm types! Please set 'force_min_firm_type'=True to have the estimator automatically iterate over each firm type, constraining it to be the minimum firm type, then store results from the minimum firm type with the highest likelihood. Since 'force_min_firm_type' == False, this iteration will start now that the estimator has become stuck in a cycle.")
                        self.params['force_min_firm_type'] = True
                        self.fit_movers(jdata, compute_NNm=compute_NNm)
                        self.params['force_min_firm_type'] = False
                        store_res = False
                        break

            # ---------- Update pk1 ----------
            if params['update_pk1']:
                # NOTE: add dirichlet prior
                # NOTE: this is equivalent to pk1 = GG12.T @ (qi + d_prior - 1)
                # NOTE: don't use weights here, since type probabilities are unrelated to length of spell
                pk1 = np.bincount(KK2, weights=(qi + d_prior - 1).flatten()).reshape(nl, nk * nk).T
                # Normalize rows to sum to 1
                pk1 = DxM(1 / np.sum(pk1, axis=1), pk1)

                if pd.isna(pk1).any():
                    warnings.warn('Estimated pk1 has NaN values. Please try a different set of starting values.')
                    break
                    # raise ValueError('Estimated pk1 has NaN values. Please try a different set of starting values.')

            if params['update_a'] or params['update_s']:
                # ---------- M-step ----------
                # Constrained OLS (source: https://scaron.info/blog/quadratic-programming-in-python.html)

                # The regression has 2 * nl * nk parameters and nl * ni rows
                # We do not necessarily want to construct the duplicated data by nl
                # Instead we will construct X'X and X'Y by looping over nl
                # We also note that X'X is block diagonal with 2*nl matrices of dimensions nk^2
                ## General ##
                # Shift for period 2
                ts = nl * nk
                # Only store the diagonal
                XwX = np.zeros(shape=2 * ts)
                if params['update_a']:
                    XwY = np.zeros(shape=2 * ts)

                ## Categorical ##
                if len(cat_cols) > 0:
                    ts_cat = {col: nl * col_dict['n'] for col, col_dict in cat_dict.items()}
                    XwX_cat = {col: np.zeros(shape=2 * col_ts) for col, col_ts in ts_cat.items()}
                    if params['update_a']:
                        XwY_cat = {col: np.zeros(shape=2 * col_ts) for col, col_ts in ts_cat.items()}
                ### Continuous ###
                if len(cts_cols) > 0:
                    XwX_cts = {col: np.zeros(shape=2 * nl) for col in cts_cols}
                    if params['update_a']:
                        XwY_cts = {col: np.zeros(shape=2 * nl) for col in cts_cols}

                if iter == 0:
                    if any_non_worker_type_interactions:
                        Y1_adj = Y1.copy()
                        Y2_adj = Y2.copy()
                        Y1_adj -= A1_sum
                        Y2_adj -= A2_sum
                    else:
                        Y1_adj = Y1
                        Y2_adj = Y2

                ## Update A ##
                if params['update_s']:
                    # Store weights computed for A for use when computing S
                    weights1 = []
                    weights2 = []
                for l in range(nl):
                    l_index, r_index = l * nk, (l + 1) * nk

                    ## Compute weights ##
                    weights_1 = W1 * qi[:, l] / S1[l, G1]
                    weights_2 = W2 * qi[:, l] / S2[l, G2]
                    if params['update_s']:
                        weights1.append(weights_1)
                        weights2.append(weights_2)

                    ## Compute XwX_l (equivalent to GG1.T @ weights @ GG1) ##
                    # Use np.bincount to perform groupby-sum (source: https://stackoverflow.com/a/7089540/17333120)
                    XwX[l_index: r_index] = np.bincount(G1, weights_1)
                    XwX[l_index + ts: r_index + ts] = np.bincount(G2, weights_2)

                    if params['update_a']:
                        # Update A1_sum and A2_sum to account for worker-interaction terms
                        A1_sum_l, A2_sum_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts, compute_S=False)

                        ## Update weights ##
                        if params['update_s']:
                            weights_1 = weights_1 * (Y1_adj - A1_sum_l)
                            weights_2 = weights_2 * (Y2_adj - A2_sum_l)
                        else:
                            weights_1 *= (Y1_adj - A1_sum_l)
                            weights_2 *= (Y2_adj - A2_sum_l)

                        ## Compute XwY_l (equivalent to GG1.T @ weights @ Y) ##
                        XwY[l_index: r_index] = np.bincount(G1, weights_1)
                        XwY[l_index + ts: r_index + ts] = np.bincount(G2, weights_2)
                        del A1_sum_l, A2_sum_l
                del weights_1, weights_2

                # print('A1 before:')
                # print(A1)
                # print('A2 before:')
                # print(A2)
                # print('S1 before:')
                # print(S1)
                # print('S2 before:')
                # print(S2)
                # print('A1_cat before:')
                # print(A1_cat)
                # print('A2_cat before:')
                # print(A2_cat)
                # print('S1_cat before:')
                # print(S1_cat)
                # print('S2_cat before:')
                # print(S2_cat)
                # print('A1_cts before:')
                # print(A1_cts)
                # print('A2_cts before:')
                # print(A2_cts)
                # print('S1_cts before:')
                # print(S1_cts)
                # print('S2_cts before:')
                # print(S2_cts)

                # We solve the system to get all the parameters (use dense solver)
                XwX = np.diag(XwX)
                if params['update_a']:
                    if iter > 0:
                        ## Constraints ##
                        cons_a, cons_s, cons_a_dict, cons_s_dict = self._gen_constraints(min_firm_type)
                    try:
                        cons_a.solve(XwX, -XwY, solver='quadprog')
                        del XwY
                        if cons_a.res is None:
                            # If constraints inconsistent, keep A1 and A2 the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing A1/A2: estimates are None')
                        else:
                            res_a1, res_a2 = np.split(cons_a.res, 2)
                            # if pd.isna(res_a1).any() or pd.isna(res_a2).any():
                            #     raise ValueError('Estimated A1/A2 has NaN values')
                            A1 = np.reshape(res_a1, self.dims)
                            A2 = np.reshape(res_a2, self.dims)

                    except ValueError as e:
                        # If constraints inconsistent, keep A1 and A2 the same
                        if params['verbose'] in [2, 3]:
                            print(f'Passing A1/A2: {e}')

                ## Categorical ##
                if params['update_s']:
                    # Store weights computed for A_cat for use when computing S_cat
                    weights1_cat = {col: [] for col in cat_cols}
                    weights2_cat = {col: [] for col in cat_cols}
                for col in cat_cols:
                    col_n = cat_dict[col]['n']

                    if not cat_dict[col]['worker_type_interaction']:
                        Y1_adj += A1_cat[col][C1[col]]
                        Y2_adj += A2_cat[col][C2[col]]

                    for l in range(nl):
                        l_index, r_index = l * col_n, (l + 1) * col_n

                        ## Compute variances ##
                        if cat_dict[col]['worker_type_interaction']:
                            S1_cat_l = S1_cat[col][l, C1[col]]
                            S2_cat_l = S2_cat[col][l, C2[col]]
                        else:
                            S1_cat_l = S1_cat[col][C1[col]]
                            S2_cat_l = S2_cat[col][C2[col]]

                        ## Compute weights ##
                        weights_1 = W1 * qi[:, l] / S1_cat_l
                        weights_2 = W2 * qi[:, l] / S2_cat_l
                        del S1_cat_l, S2_cat_l
                        if params['update_s']:
                            weights1_cat[col].append(weights_1)
                            weights2_cat[col].append(weights_2)

                        ## Compute XwX_cat_l (equivalent to CC1_cat.T @ weights @ CC1_cat) ##
                        XwX_cat[col][l_index: r_index] = np.bincount(C1[col], weights_1)
                        XwX_cat[col][l_index + ts_cat[col]: r_index + ts_cat[col]] = np.bincount(C2[col], weights_2)

                        if params['update_a']:
                            # Update A1_sum and A2_sum to account for worker-interaction terms
                            A1_sum_l, A2_sum_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts, compute_S=False)

                            if cat_dict[col]['worker_type_interaction']:
                                A1_sum_l -= A1_cat[col][l, C1[col]]
                                A2_sum_l -= A2_cat[col][l, C2[col]]

                            ## Update weights ##
                            if params['update_s']:
                                weights_1 = weights_1 * (Y1_adj - A1_sum_l - A1[l, G1])
                                weights_2 = weights_2 * (Y2_adj - A2_sum_l - A2[l, G2])
                            else:
                                weights_1 *= (Y1_adj - A1_sum_l - A1[l, G1])
                                weights_2 *= (Y2_adj - A2_sum_l - A2[l, G2])

                            ## Compute XwY_cat_l (equivalent to CC1_cat.T @ weights @ Y) ##
                            XwY_cat[col][l_index: r_index] = np.bincount(C1[col], weights_1)
                            XwY_cat[col][l_index + ts_cat[col]: r_index + ts_cat[col]] = np.bincount(C2[col], weights_2)
                            del A1_sum_l, A2_sum_l
                    del weights_1, weights_2

                    # We solve the system to get all the parameters (use dense solver)
                    XwX_cat[col] = np.diag(XwX_cat[col])
                    if params['update_a']:
                        try:
                            a_solver = cons_a_dict[col]
                            a_solver.solve(XwX_cat[col], -XwY_cat[col], solver='quadprog')
                            del XwY_cat[col]
                            if a_solver.res is None:
                                # If constraints inconsistent, keep A1_cat and A2_cat the same
                                if params['verbose'] in [2, 3]:
                                    print(f'Passing A1_cat/A2_cat for column {col!r}: estimates are None')
                            else:
                                res_a1, res_a2 = np.split(a_solver.res, 2)
                                # if pd.isna(res_a1).any() or pd.isna(res_a2).any():
                                #     raise ValueError(f'Estimated A1_cat/A2_cat has NaN values for column {col!r}')
                                if cat_dict[col]['worker_type_interaction']:
                                    A1_cat[col] = np.reshape(res_a1, (nl, col_n))
                                    A2_cat[col] = np.reshape(res_a2, (nl, col_n))
                                else:
                                    A1_cat[col] = res_a1[: col_n]
                                    A2_cat[col] = res_a2[: col_n]

                        except ValueError as e:
                            # If constraints inconsistent, keep A1_cat and A2_cat the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing A1_cat/A2_cat for column {col!r}: {e}')

                    if not cat_dict[col]['worker_type_interaction']:
                        Y1_adj -= A1_cat[col][C1[col]]
                        Y2_adj -= A2_cat[col][C2[col]]

                ## Continuous ##
                if params['update_s']:
                    # Store weights computed for A_cts for use when computing S_cts
                    Xw1_cts = {col: [] for col in cts_cols}
                    Xw2_cts = {col: [] for col in cts_cols}
                for col in cts_cols:
                    if not cts_dict[col]['worker_type_interaction']:
                        Y1_adj += A1_cts[col] * C1[col]
                        Y2_adj += A2_cts[col] * C2[col]

                    for l in range(nl):
                        ## Compute variances ##
                        if cts_dict[col]['worker_type_interaction']:
                            S1_cts_l = S1_cts[col][l]
                            S2_cts_l = S2_cts[col][l]
                        else:
                            S1_cts_l = S1_cts[col]
                            S2_cts_l = S2_cts[col]

                        ## Compute Xw_cts_l ##
                        Xw1_cts_l = C1[col].T * (W1 * qi[:, l] / S1_cts_l)
                        Xw2_cts_l = C2[col].T * (W2 * qi[:, l] / S2_cts_l)
                        del S1_cts_l, S2_cts_l
                        if params['update_s']:
                            Xw1_cts[col].append(Xw1_cts_l)
                            Xw2_cts[col].append(Xw2_cts_l)

                        ## Compute XwX_cts_l ##
                        XwX_cts[col][l] = (Xw1_cts_l @ C1[col])
                        XwX_cts[col][l + nl] = (Xw2_cts_l @ C2[col])

                        if params['update_a']:
                            # Update A1_sum and A2_sum to account for worker-interaction terms
                            A1_sum_l, A2_sum_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts, compute_S=False)

                            if cts_dict[col]['worker_type_interaction']:
                                A1_sum_l -= A1_cts[col][l] * C1[col]
                                A2_sum_l -= A2_cts[col][l] * C2[col]

                            ## Compute XwY_cts_l ##
                            XwY_cts[col][l] = Xw1_cts_l @ (Y1_adj - A1_sum_l - A1[l, G1])
                            XwY_cts[col][l + nl] = Xw2_cts_l @ (Y2_adj - A2_sum_l - A2[l, G2])
                            del A1_sum_l, A2_sum_l
                    del Xw1_cts_l, Xw2_cts_l

                    # We solve the system to get all the parameters (use dense solver)
                    XwX_cts[col] = np.diag(XwX_cts[col])
                    if params['update_a']:
                        try:
                            a_solver = cons_a_dict[col]
                            a_solver.solve(XwX_cts[col], -XwY_cts[col], solver='quadprog')
                            del XwY_cts[col]
                            if a_solver.res is None:
                                # If constraints inconsistent, keep A1_cts and A2_cts the same
                                if params['verbose'] in [2, 3]:
                                    print(f'Passing A1_cts/A2_cts for column {col!r}: estimates are None')
                            else:
                                res_a1, res_a2 = np.split(a_solver.res, 2)
                                # if pd.isna(res_a1).any() or pd.isna(res_a2).any():
                                #     raise ValueError(f'Estimated A1_cts/A2_cts has NaN values for column {col!r}')
                                if cts_dict[col]['worker_type_interaction']:
                                    A1_cts[col] = res_a1
                                    A2_cts[col] = res_a2
                                else:
                                    A1_cts[col] = res_a1[0]
                                    A2_cts[col] = res_a2[0]

                        except ValueError as e:
                            # If constraints inconsistent, keep A1_cts and A2_cts the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing A1_cts/A2_cts for column {col!r}: {e}')

                    if not cts_dict[col]['worker_type_interaction']:
                        Y1_adj -= A1_cts[col] * C1[col]
                        Y2_adj -= A2_cts[col] * C2[col]

                if any_non_worker_type_interactions:
                    # Update A1_sum and A2_sum
                    A1_sum = Y1 - Y1_adj
                    A2_sum = Y2 - Y2_adj

                if params['update_s']:
                    ## Update the variances ##
                    XwS = np.zeros(shape=2 * ts)

                    ## Categorical ##
                    if len(cat_cols) > 0:
                        XwS_cat = {col: np.zeros(shape=2 * col_ts) for col, col_ts in ts_cat.items()}

                    ## Continuous ##
                    if len(cts_cols) > 0:
                        XwS_cts = {col: np.zeros(shape=2 * nl) for col in cts_cols}

                    ## Residuals ##
                    eps1_sq = []
                    eps2_sq = []

                    ## Update S ##
                    for l in range(nl):
                        # Update A1_sum/A2_sum/S1_sum_sq/S2_sum_sq to account for worker-interaction terms
                        if any_controls:
                            # If controls, calculate S
                            A1_sum_l, A2_sum_l, S1_sum_sq_l, S2_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts, compute_S=True)
                        else:
                            # If no controls, don't calculate S
                            A1_sum_l, A2_sum_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts, compute_S=False)

                        ## Residuals ##
                        eps1_sq.append((Y1_adj - A1_sum_l - A1[l, G1]) ** 2)
                        eps2_sq.append((Y2_adj - A2_sum_l - A2[l, G2]) ** 2)
                        del A1_sum_l, A2_sum_l

                        ## XwS_l ##
                        l_index, r_index = l * nk, (l + 1) * nk

                        ## Update weights ##
                        weights1[l] *= eps1_sq[l]
                        weights2[l] *= eps2_sq[l]
                        if any_controls:
                            ## Account for other variables' contribution to variance ##
                            weights1[l] *= (S1[l, :] ** 2)[G1] / ((S1[l, :] ** 2)[G1] + S1_sum_sq + S1_sum_sq_l)
                            weights2[l] *= (S2[l, :] ** 2)[G2] / ((S2[l, :] ** 2)[G2] + S2_sum_sq + S2_sum_sq_l)
                            del S1_sum_sq_l, S2_sum_sq_l

                        ## Compute wS_l ##
                        XwS[l_index: r_index] = np.bincount(G1, weights1[l])
                        XwS[l_index + ts: r_index + ts] = np.bincount(G2, weights2[l])

                        ## Clear weights[l] ##
                        weights1[l] = 0
                        weights2[l] = 0

                    try:
                        cons_s.solve(XwX, -XwS, solver='quadprog')
                        del XwS
                        if cons_s.res is None:
                            # If constraints inconsistent, keep S1 and S2 the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing S1/S2: estimates are None')
                        else:
                            res_s1, res_s2 = np.split(cons_s.res, 2)
                            # if pd.isna(res_s1).any() or pd.isna(res_s2).any():
                            #     raise ValueError('Estimated S1/S2 has NaN values')
                            S1 = np.sqrt(np.reshape(res_s1, self.dims))
                            S2 = np.sqrt(np.reshape(res_s2, self.dims))

                    except ValueError as e:
                        # If constraints inconsistent, keep S1 and S2 the same
                        if params['verbose'] in [2, 3]:
                            print(f'Passing S1/S2: {e}')

                    ## Categorical ##
                    for col in cat_cols:
                        col_n = cat_dict[col]['n']

                        for l in range(nl):
                            # Update S1_sum_sq and S2_sum_sq to account for worker-interaction terms
                            S1_sum_sq_l, S2_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts, compute_A=False, compute_S=True)

                            l_index, r_index = l * col_n, (l + 1) * col_n

                            ## Compute variances ##
                            if cat_dict[col]['worker_type_interaction']:
                                S1_cat_l = (S1_cat[col][l, :] ** 2)[C1[col]]
                                S2_cat_l = (S2_cat[col][l, :] ** 2)[C2[col]]
                            else:
                                S1_cat_l = (S1_cat[col] ** 2)[C1[col]]
                                S2_cat_l = (S2_cat[col] ** 2)[C2[col]]

                            ## Update weights ##
                            weights1_cat[col][l] *= eps1_sq[l] * S1_cat_l / ((S1[l, :] ** 2)[G1] + S1_sum_sq + S1_sum_sq_l)
                            weights2_cat[col][l] *= eps2_sq[l] * S2_cat_l / ((S2[l, :] ** 2)[G2] + S2_sum_sq + S2_sum_sq_l)
                            del S1_sum_sq_l, S2_sum_sq_l, S1_cat_l, S2_cat_l

                            ## XwS_cat_l ##
                            XwS_cat[col][l_index: r_index] = np.bincount(C1[col], weights1_cat[col][l])
                            XwS_cat[col][l_index + ts_cat[col]: r_index + ts_cat[col]] = np.bincount(C2[col], weights2_cat[col][l])

                            ## Clear weights_cat[col][l] ##
                            weights1_cat[col][l] = 0
                            weights2_cat[col][l] = 0

                        try:
                            s_solver = cons_s_dict[col]
                            s_solver.solve(XwX_cat[col], -XwS_cat[col], solver='quadprog')
                            del XwS_cat[col]
                            if s_solver.res is None:
                                # If constraints inconsistent, keep S1_cat and S2_cat the same
                                if params['verbose'] in [2, 3]:
                                    print(f'Passing S1_cat/S2_cat for column {col!r}: estimates are None')
                            else:
                                res_s1, res_s2 = np.split(s_solver.res, 2)
                                # if pd.isna(res_s1).any() or pd.isna(res_s2).any():
                                #     raise ValueError(f'Estimated S1_cat/S2_cat has NaN values for column {col!r}')

                                if not cat_dict[col]['worker_type_interaction']:
                                    S1_sum_sq -= (S1_cat[col] ** 2)[C1[col]]
                                    S2_sum_sq -= (S2_cat[col] ** 2)[C2[col]]

                                if cat_dict[col]['worker_type_interaction']:
                                    S1_cat[col] = np.sqrt(np.reshape(res_s1, (nl, col_n)))
                                    S2_cat[col] = np.sqrt(np.reshape(res_s2, (nl, col_n)))
                                else:
                                    S1_cat[col] = np.sqrt(res_s1[: col_n])
                                    S2_cat[col] = np.sqrt(res_s2[: col_n])

                                if not cat_dict[col]['worker_type_interaction']:
                                    S1_sum_sq += (S1_cat[col] ** 2)[C1[col]]
                                    S2_sum_sq += (S2_cat[col] ** 2)[C2[col]]

                        except ValueError as e:
                            # If constraints inconsistent, keep S1_cat and S2_cat the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing S1_cat/S2_cat for column {col!r}: {e}')

                    ## Continuous ##
                    for col in cts_cols:
                        for l in range(nl):
                            # Update S1_sum_sq and S2_sum_sq to account for worker-interaction terms
                            S1_sum_sq_l, S2_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts, compute_A=False, compute_S=True)

                            ## Compute variances ##
                            if cts_dict[col]['worker_type_interaction']:
                                S1_cts_l = S1_cts[col][l] ** 2
                                S2_cts_l = S2_cts[col][l] ** 2
                            else:
                                # Already removed from S_sum_sq
                                S1_cts_l = S1_cts[col] ** 2
                                S2_cts_l = S2_cts[col] ** 2

                            ## XwS_cts_l ##
                            # NOTE: take absolute value
                            XwS_cts[col][l] = np.abs(Xw1_cts[col][l] @ (eps1_sq[l] * S1_cts_l / ((S1[l, :] ** 2)[G1] + S1_sum_sq + S1_sum_sq_l)))
                            XwS_cts[col][l + nl] = np.abs(Xw2_cts[col][l] @ (eps2_sq[l] * S2_cts_l / ((S2[l, :] ** 2)[G2] + S2_sum_sq + S2_sum_sq_l)))
                            del S1_sum_sq_l, S2_sum_sq_l, S1_cts_l, S2_cts_l

                            ## Clear Xw_cts[col][l] ##
                            Xw1_cts[col][l] = 0
                            Xw2_cts[col][l] = 0

                        try:
                            s_solver = cons_s_dict[col]
                            s_solver.solve(XwX_cts[col], -XwS_cts[col], solver='quadprog')
                            del XwS_cts[col]
                            if s_solver.res is None:
                                # If constraints inconsistent, keep S1_cts and S2_cts the same
                                if params['verbose'] in [2, 3]:
                                    print(f'Passing S1_cts/S2_cts for column {col!r}: estimates are None')
                            else:
                                res_s1, res_s2 = np.split(s_solver.res, 2)
                                # if pd.isna(res_s1).any() or pd.isna(res_s2).any():
                                #     raise ValueError(f'Estimated S1_cts/S2_cts has NaN values for column {col!r}')

                                if not cts_dict[col]['worker_type_interaction']:
                                    S1_sum_sq -= S1_cts[col] ** 2
                                    S2_sum_sq -= S2_cts[col] ** 2

                                if cts_dict[col]['worker_type_interaction']:
                                    S1_cts[col] = np.sqrt(res_s1)
                                    S2_cts[col] = np.sqrt(res_s2)
                                else:
                                    S1_cts[col] = np.sqrt(res_s1[0])
                                    S2_cts[col] = np.sqrt(res_s2[0])

                                if not cts_dict[col]['worker_type_interaction']:
                                    S1_sum_sq += S1_cts[col] ** 2
                                    S2_sum_sq += S2_cts[col] ** 2

                        except ValueError as e:
                            # If constraints inconsistent, keep S1_cts and S2_cts the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing S1_cts/S2_cts for column {col!r}: {e}')

                    del eps1_sq, eps2_sq

                del XwX, weights1, weights2
                if len(cat_cols) > 0:
                    del XwX_cat, weights1_cat, weights2_cat
                if len(cts_cols) > 0:
                    del XwX_cts, Xw1_cts, Xw2_cts

                # print('A1 after:')
                # print(A1)
                # print('A2 after:')
                # print(A2)
                # print('S1 after:')
                # print(S1)
                # print('S2 after:')
                # print(S2)
                # print('A1_cat after:')
                # print(A1_cat)
                # print('A2_cat after:')
                # print(A2_cat)
                # print('S1_cat after:')
                # print(S1_cat)
                # print('S2_cat after:')
                # print(S2_cat)
                # print('A1_cts after:')
                # print(A1_cts)
                # print('A2_cts after:')
                # print(A2_cts)
                # print('S1_cts after:')
                # print(S1_cts)
                # print('S2_cts after:')
                # print(S2_cts)

        if store_res:
            ## Sort parameters ##
            A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, self.pk0 = self._sort_parameters(A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, self.pk0)

            # Store parameters
            self.A1, self.A2, self.S1, self.S2 = A1, A2, S1, S2
            self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat = A1_cat, A2_cat, S1_cat, S2_cat
            self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts = A1_cts, A2_cts, S1_cts, S2_cts
            self.pk1 = pk1
            self.lik1, self.liks1 = lik1, liks1 # np.concatenate([self.liks1, liks1])

            # Update NNm
            if compute_NNm:
                self.NNm = jdata.groupby('g1')['g2'].value_counts().unstack(fill_value=0).to_numpy()

    def fit_stayers(self, sdata, compute_NNs=True):
        '''
        EM algorithm for stayers.

        Arguments:
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            compute_NNs (bool): if True, compute vector giving the number of stayers at each firm type (e.g. entry (1) gives the number of stayers at firm type 1)
        '''
        # Unpack parameters
        params = self.params
        nl, nk, ni = self.nl, self.nk, sdata.shape[0]
        A1, A2, S1, S2 = self.A1, self.A2, self.S1, self.S2
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        any_controls = self.any_controls

        # Store wage outcomes and groups
        Y1 = sdata['y1'].to_numpy()
        # Y2 = sdata['y2'].to_numpy()
        G1 = sdata['g1'].to_numpy().astype(int, copy=False)
        # G2 = sdata['g2'].to_numpy().astype(int, copy=False)

        # Weights
        if params['weighted'] and sdata._col_included('w'):
            W1 = sdata.loc[:, 'w1'].to_numpy()
            # W2 = sdata.loc[:, 'w2'].to_numpy()
        else:
            W1 = 1
            # W2 = 1
        W = W1 # np.sqrt(W1 * W2)

        ## Control variables ##
        C1 = {}
        C2 = {}
        for i, col in enumerate(cat_cols + cts_cols):
            # Get subcolumns associated with col
            subcols = to_list(sdata.col_reference_dict[col])
            n_subcols = len(subcols)
            if n_subcols == 1:
                # If column is constant over time
                subcol_1 = subcols[0]
                subcol_2 = subcols[0]
            elif n_subcols == 2:
                # If column can change over time
                subcol_1 = subcols[0]
                subcol_2 = subcols[1]
            else:
                raise NotImplementedError(f'Column names must have either one or two associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
            if i < len(cat_cols):
                # Categorical
                C1[col] = sdata.loc[:, subcol_1].to_numpy().astype(int, copy=False)
                C2[col] = sdata.loc[:, subcol_2].to_numpy().astype(int, copy=False)
            else:
                # Continuous
                C1[col] = sdata.loc[:, subcol_1].to_numpy()
                C2[col] = sdata.loc[:, subcol_2].to_numpy()

        # Joint firm indicator
        KK = G1
        KK2 = np.tile(KK, (nl, 1)).T
        KK3 = KK2 + nk * np.arange(nl)
        KK = KK3.flatten()
        del KK2, KK3

        # # Transition probability matrix
        # GG1 = csc_matrix((np.ones(ni), (range(ni), G1)), shape=(ni, nk))

        # Matrix of prior probabilities
        pk0 = self.pk0
        # Matrix of posterior probabilities
        qi = np.ones(shape=(ni, nl))
        # Log pdfs
        lp_stable = np.zeros(shape=(ni, nl))
        lp = np.zeros(shape=(ni, nl))
        # Log likelihood for stayers
        lik0 = None
        # Path of log likelihoods for stayers
        liks0 = []
        prev_lik = np.inf
        # Fix error from bad initial guesses causing probabilities to be too low
        d_prior = params['d_prior_stayers']

        if any_controls:
            ## Account for control variables ##
            A1_sum, A2_sum, S1_sum_sq, S2_sum_sq = self._sum_by_non_nl(ni=ni, C1=C1, C2=C2, A1_cat=self.A1_cat, A2_cat=self.A2_cat, S1_cat=self.S1_cat, S2_cat=self.S2_cat, A1_cts=self.A1_cts, A2_cts=self.A2_cts, S1_cts=self.S1_cts, S2_cts=self.S2_cts)

            for l in range(nl):
                # Update A1_sum/S1_sum_sq to account for worker-interaction terms
                A1_sum_l, A2_sum_l, S1_sum_sq_l, S2_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, A1_cat=self.A1_cat, A2_cat=self.A2_cat, S1_cat=self.S1_cat, S2_cat=self.S2_cat, A1_cts=self.A1_cts, A2_cts=self.A2_cts, S1_cts=self.S1_cts, S2_cts=self.S2_cts)
                lp1 = lognormpdf(Y1, A1[l, G1] + A1_sum + A1_sum_l, var=(S1[l, :] ** 2)[G1] + S1_sum_sq + S1_sum_sq_l, gpu=self.gpu)
                # lp2 = lognormpdf(Y2, A2[l, G2] + A2_sum + A2_sum_l, var=(S2[l, :] ** 2)[G2] + S2_sum_sq + S2_sum_sq_l, gpu=self.gpu)
                lp_stable[:, l] = W1 * lp1 # + W2 * lp2
        else:
            for l in range(nl):
                lp1 = fast_lognormpdf(Y1, A1[l, :], S1[l, :], G1, gpu=self.gpu)
                # lp2 = fast_lognormpdf(Y2, A2[l, :], S2[l, :], G2, gpu=self.gpu)
                lp_stable[:, l] = W1 * lp1 # + W2 * lp2
        del lp1 #, lp2

        for iter in range(params['n_iters_stayers']):

            # ---------- E-Step ----------
            # We compute the posterior probabilities for each row
            # We iterate over the worker types, should not be be
            # too costly since the vector is quite large within each iteration
            for l in range(nl):
                lp[:, l] = lp_stable[:, l] + np.log(pk0)[G1, l]

            # We compute log sum exp to get likelihoods and probabilities
            lse_lp = logsumexp(lp, axis=1, gpu=self.gpu)
            qi = exp_(lp.T - lse_lp, gpu=self.gpu).T
            if params['return_qi']:
                return qi
            lik0 = lse_lp.mean()
            del lse_lp
            if iter > 0:
                # Account for Dirichlet prior
                lik_prior = (d_prior - 1) * np.sum(np.log(pk0))
                lik0 += lik_prior
            liks0.append(lik0)
            if params['verbose'] == 3:
                print('loop {}, liks {}'.format(iter, lik0))

            if abs(lik0 - prev_lik) < params['threshold_stayers']:
                # Break loop
                break
            if iter == (params['n_iters_stayers'] - 1):
                print(f"Maximum iterations reached for stayers. It is recommended to increase `n_iters_stayers` from its current value of {params['n_iters_stayers']}.")
            prev_lik = lik0

            # ---------- M-step ----------
            # NOTE: add dirichlet prior
            # NOTE: this is equivalent to pk0 = GG1.T @ (qi + d_prior - 1)
            # NOTE: don't use weights here, since type probabilities are unrelated to length of spell
            pk0 = np.bincount(KK, (qi + d_prior - 1).flatten()).reshape(nl, nk).T
            # Normalize rows to sum to 1
            pk0 = DxM(1 / np.sum(pk0, axis=1), pk0)

        self.pk0 = pk0
        self.lik0, self.liks0 = lik0, liks0 # np.concatenate([self.liks0, liks0])

        # Update NNs
        if compute_NNs:
            NNs = sdata['g1'].value_counts(sort=False)
            NNs.sort_index(inplace=True)
            self.NNs = NNs.to_numpy()

    def fit_movers_cstr_uncstr(self, jdata, linear_additivity=True, compute_NNm=True):
        '''
        Run fit_movers(), first constrained, then using results as starting values, run unconstrained.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            linear_additivity (bool): if True, include the loop with the linear additivity constraint
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        ## First, simulate parameters but keep A fixed ##
        ## Second, use estimated parameters as starting point to run with A constrained to be linear ##
        ## Finally use estimated parameters as starting point to run without constraints ##
        # Save original parameters
        user_params = self.params.copy()
        ##### Loop 1 #####
        # First fix A but update S and pk
        self.params['update_a'] = False
        self.params['update_s'] = True
        self.params['update_pk1'] = True
        if (self.params['categorical_controls'] is not None) and (len(self.params['categorical_controls']) > 0):
            # Don't normalize until unconstrained, since normalizing can conflict with linear additivity and stationary firm-type variation constraints
            self.params['normalize'] = False
        if self.params['verbose'] in [1, 2, 3]:
            print('Fitting movers with A fixed')
        self.fit_movers(jdata, compute_NNm=False)
        ##### Loop 2 #####
        # Now update A with Linear Additive constraint
        self.params['update_a'] = True
        if linear_additivity and (self.nl > 1):
            # Set constraints
            if user_params['cons_a_all'] is None:
                self.params['cons_a_all'] = cons.LinearAdditive()
            else:
                self.params['cons_a_all'] = to_list(user_params['cons_a_all']) + [cons.LinearAdditive()]
            if self.params['verbose'] in [1, 2, 3]:
                print('Fitting movers with Linear Additive constraint on A')
            self.fit_movers(jdata, compute_NNm=False)
        ##### Loop 3 #####
        # Now update A with Stationary Firm Type Variation constraint
        if self.nl > 1:
            # Set constraints
            if user_params['cons_a_all'] is None:
                self.params['cons_a_all'] = cons.StationaryFirmTypeVariation()
            else:
                self.params['cons_a_all'] = to_list(user_params['cons_a_all']) + [cons.StationaryFirmTypeVariation()]
            if self.params['verbose'] in [1, 2, 3]:
                print('Fitting movers with Stationary Firm Type Variation constraint on A')
            self.fit_movers(jdata, compute_NNm=False)
        ##### Loop 4 #####
        ## Restore user constraints ##
        if (self.params['categorical_controls'] is not None) and (len(self.params['categorical_controls']) > 0):
            # Can normalize again
            self.params['normalize'] = user_params['normalize']
            if self.params['verbose'] in [2, 3]:
                print(f"Restoring normalize to {user_params['normalize']} - it is recommended to turn off normalization if estimator is not converging")
        self.params['cons_a_all'] = user_params['cons_a_all']
        if self.params['verbose'] in [1, 2, 3]:
            print('Fitting unconstrained movers')
        self.fit_movers(jdata, compute_NNm=compute_NNm)
        ##### Compute connectedness #####
        if not pd.isna(self.pk1).any():
            self.compute_connectedness_measure()
        else:
            warnings.warn('Estimated pk1 has NaN values. Please try a different set of starting values.')
        # Restore original parameters
        self.params = user_params

    def fit_A(self, jdata, compute_NNm=True):
        '''
        Run fit_movers() and update A while keeping S and pk1 fixed.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # Save original parameters
        user_params = self.params.copy()
        # Update parameters
        self.params['update_a'] = True
        self.params['update_s'] = False
        self.params['update_pk1'] = False
        # Estimate
        if self.params['verbose'] in [1, 2, 3]:
            print('Running fit_A')
        self.fit_movers(jdata, compute_NNm=compute_NNm)
        # Restore original parameters
        self.params = user_params

    def fit_S(self, jdata, compute_NNm=True):
        '''
        Run fit_movers() and update S while keeping A and pk1 fixed.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # Save original parameters
        user_params = self.params.copy()
        # Update parameters
        self.params['update_a'] = False
        self.params['update_s'] = True
        self.params['update_pk1'] = False
        # Estimate
        if self.params['verbose'] in [1, 2, 3]:
            print('Running fit_S')
        self.fit_movers(jdata, compute_NNm=compute_NNm)
        # Restore original parameters
        self.params = user_params

    def fit_pk(self, jdata, compute_NNm=True):
        '''
        Run fit_movers() and update pk1 while keeping A and S fixed.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # Save original parameters
        user_params = self.params.copy()
        # Update parameters
        self.params['update_a'] = False
        self.params['update_s'] = False
        self.params['update_pk1'] = True
        # Estimate
        if self.params['verbose'] in [1, 2, 3]:
            print('Running fit_pk')
        self.fit_movers(jdata, compute_NNm=compute_NNm)
        # Restore original parameters
        self.params = user_params

    def reclassify_firms(self, jdata, sdata):
        '''
        Reclassify firms.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers

        Returns:
            (NumPy Array): new firm classes (index corresponds to firm id)
        '''
        # Unpack parameters
        params = self.params
        nl, nk, nij, nis = self.nl, self.nk, jdata.shape[0], sdata.shape[0]
        A1, A2, S1, S2 = self.A1, self.A2, self.S1, self.S2
        A1_cat, A2_cat, S1_cat, S2_cat = self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat
        A1_cts, A2_cts, S1_cts, S2_cts = self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts
        pk1, pk0 = self.pk1, self.pk0
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        any_controls = self.any_controls

        # Number of firms (movers and stayers don't necessarily have the same firms)
        nf = max(jdata.loc[:, 'j1'].to_numpy().max(), jdata.loc[:, 'j2'].to_numpy().max(), sdata.loc[:, 'j1'].to_numpy().max()) + 1

        ### Movers ###
        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        J1 = jdata.loc[:, 'j1'].to_numpy()
        J2 = jdata.loc[:, 'j2'].to_numpy()
        G1 = jdata.loc[:, 'g1'].to_numpy().astype(int, copy=False)
        G2 = jdata.loc[:, 'g2'].to_numpy().astype(int, copy=False)

        # Weights
        if params['weighted'] and jdata._col_included('w'):
            W1 = jdata.loc[:, 'w1'].to_numpy()
            W2 = jdata.loc[:, 'w2'].to_numpy()
        else:
            W1 = 1
            W2 = 1

        ## Control variables ##
        C1 = {}
        C2 = {}
        for i, col in enumerate(cat_cols + cts_cols):
            # Get subcolumns associated with col
            subcols = to_list(jdata.col_reference_dict[col])
            n_subcols = len(subcols)
            if n_subcols == 1:
                # If column is constant over time
                subcol_1 = subcols[0]
                subcol_2 = subcols[0]
            elif n_subcols == 2:
                # If column can change over time
                subcol_1 = subcols[0]
                subcol_2 = subcols[1]
            else:
                raise NotImplementedError(f'Column names must have either one or two associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
            if i < len(cat_cols):
                # Categorical
                C1[col] = jdata.loc[:, subcol_1].to_numpy().astype(int, copy=False)
                C2[col] = jdata.loc[:, subcol_2].to_numpy().astype(int, copy=False)
            else:
                # Continuous
                C1[col] = jdata.loc[:, subcol_1].to_numpy()
                C2[col] = jdata.loc[:, subcol_2].to_numpy()

        ## Joint firm indicator ##
        nkG2 = nk * G2

        ## Compute log-likelihood ##
        # Log pdfs
        lp_adj_first = np.zeros(shape=(nk, nij, nl))
        lp_adj_second = np.zeros(shape=(nk, nij, nl))
        log_pk1 = np.log(pk1)
        if any_controls:
            ## Account for control variables ##
            A1_sum, A2_sum, S1_sum_sq, S2_sum_sq = self._sum_by_non_nl(ni=nij, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts)

            for l in range(nl):
                # Update A1_sum/A2_sum/S1_sum_sq/S2_sum_sq to account for worker-interaction terms
                A1_sum_l, A2_sum_l, S1_sum_sq_l, S2_sum_sq_l = self._sum_by_nl_l(ni=nij, l=l, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts)
                ## Current firm classes ##
                lp1_curr = lognormpdf(Y1, A1[l, G1] + A1_sum + A1_sum_l, var=(S1[l, :] ** 2)[G1] + S1_sum_sq + S1_sum_sq_l, gpu=self.gpu)
                lp2_curr = lognormpdf(Y2, A2[l, G2] + A2_sum + A2_sum_l, var=(S2[l, :] ** 2)[G2] + S2_sum_sq + S2_sum_sq_l, gpu=self.gpu)
                for k in range(nk):
                    ## New firm classes ##
                    lp1_adj = lognormpdf(Y1, A1[l, k] + A1_sum + A1_sum_l, var=(S1[l, k] ** 2) + S1_sum_sq + S1_sum_sq_l, gpu=self.gpu)
                    lp2_adj = lognormpdf(Y2, A2[l, k] + A2_sum + A2_sum_l, var=(S2[l, k] ** 2) + S2_sum_sq + S2_sum_sq_l, gpu=self.gpu)
                    ## Log probability ##
                    lp_adj_first[k, :, l] = log_pk1[k + nkG2, l] + W1 * lp1_adj + W2 * lp2_curr
                    lp_adj_second[k, :, l] = log_pk1[G1 + nk * k, l] + W1 * lp1_curr + W2 * lp2_adj
        else:
            for l in range(nl):
                ## Current firm classes ##
                lp1_curr = fast_lognormpdf(Y1, A1[l, :], S1[l, :], G1, gpu=self.gpu)
                lp2_curr = fast_lognormpdf(Y2, A2[l, :], S2[l, :], G2, gpu=self.gpu)
                for k in range(nk):
                    ## New firm classes ##
                    lp1_adj = fast_lognormpdf(Y1, A1[l, :], S1[l, :], k, gpu=self.gpu)
                    lp2_adj = fast_lognormpdf(Y2, A2[l, :], S2[l, :], k, gpu=self.gpu)
                    ## Log probability ##
                    lp_adj_first[k, :, l] = log_pk1[k + nkG2, l] + W1 * lp1_adj + W2 * lp2_curr
                    lp_adj_second[k, :, l] = log_pk1[G1 + nk * k, l] + W1 * lp1_curr + W2 * lp2_adj
        del log_pk1, lp1_curr, lp2_curr, lp1_adj, lp2_adj

        ## Convert to log-sum-exp form ##
        lse_lp_adj_first = np.apply_along_axis(lambda a: logsumexp(a.reshape(nij, nl), axis=1, gpu=self.gpu), axis=1, arr=lp_adj_first.reshape(nk, nij * nl))
        lse_lp_adj_second = np.apply_along_axis(lambda a: logsumexp(a.reshape(nij, nl), axis=1, gpu=self.gpu), axis=1, arr=lp_adj_second.reshape(nk, nij * nl))

        ## Firm-level probabilities ##
        firm_level_lp_adj_first = np.apply_along_axis(lambda a: np.bincount(J1, a, minlength=nf), axis=1, arr=lse_lp_adj_first).T
        firm_level_lp_adj_second = np.apply_along_axis(lambda a: np.bincount(J2, a, minlength=nf), axis=1, arr=lse_lp_adj_second).T

        ### Stayers ###
        # Store wage outcomes and groups
        Y1 = sdata['y1'].to_numpy()
        # Y2 = sdata['y2'].to_numpy()
        J1 = sdata.loc[:, 'j1'].to_numpy()
        # J2 = sdata.loc[:, 'j2'].to_numpy()
        G1 = sdata['g1'].to_numpy().astype(int, copy=False)
        # G2 = sdata['g2'].to_numpy().astype(int, copy=False)

        # Weights
        if params['weighted'] and sdata._col_included('w'):
            W1 = sdata.loc[:, 'w1'].to_numpy()
            # W2 = sdata.loc[:, 'w2'].to_numpy()
        else:
            W1 = 1
            # W2 = 1

        ## Control variables ##
        C1 = {}
        C2 = {}
        for i, col in enumerate(cat_cols + cts_cols):
            # Get subcolumns associated with col
            subcols = to_list(sdata.col_reference_dict[col])
            n_subcols = len(subcols)
            if n_subcols == 1:
                # If column is constant over time
                subcol_1 = subcols[0]
                subcol_2 = subcols[0]
            elif n_subcols == 2:
                # If column can change over time
                subcol_1 = subcols[0]
                subcol_2 = subcols[1]
            else:
                raise NotImplementedError(f'Column names must have either one or two associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
            if i < len(cat_cols):
                # Categorical
                C1[col] = sdata.loc[:, subcol_1].to_numpy().astype(int, copy=False)
                C2[col] = sdata.loc[:, subcol_2].to_numpy().astype(int, copy=False)
            else:
                # Continuous
                C1[col] = sdata.loc[:, subcol_1].to_numpy()
                C2[col] = sdata.loc[:, subcol_2].to_numpy()

        ## Compute log-likelihood ##
        # Log pdfs
        lp_adj = np.zeros(shape=(nk, nis, nl))
        log_pk0 = np.log(pk0)
        if any_controls:
            ## Account for control variables ##
            A1_sum, A2_sum, S1_sum_sq, S2_sum_sq = self._sum_by_non_nl(ni=nis, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts)

            for l in range(nl):
                # Update A1_sum/A2_sum/S1_sum_sq/S2_sum_sq to account for worker-interaction terms
                A1_sum_l, A2_sum_l, S1_sum_sq_l, S2_sum_sq_l = self._sum_by_nl_l(ni=nis, l=l, C1=C1, C2=C2, A1_cat=A1_cat, A2_cat=A2_cat, S1_cat=S1_cat, S2_cat=S2_cat, A1_cts=A1_cts, A2_cts=A2_cts, S1_cts=S1_cts, S2_cts=S2_cts)
                for k in range(nk):
                    ## New firm classes ##
                    lp1_adj = lognormpdf(Y1, A1[l, k] + A1_sum + A1_sum_l, var=(S1[l, k] ** 2) + S1_sum_sq + S1_sum_sq_l, gpu=self.gpu)
                    # lp2_adj = lognormpdf(Y2, A2[l, k] + A2_sum + A2_sum_l, var=(S2[l, k] ** 2) + S2_sum_sq + S2_sum_sq_l, gpu=self.gpu)
                    ## Log probability ##
                    lp_adj[k, :, l] = log_pk0[k, l] + W1 * lp1_adj # + W2 * lp2_adj
        else:
            for l in range(nl):
                for k in range(nk):
                    ## New firm classes ##
                    lp1_adj = fast_lognormpdf(Y1, A1[l, :], S1[l, :], k, gpu=self.gpu)
                    # lp2_adj = fast_lognormpdf(Y2, A2[l, :], S2[l, :], k, gpu=self.gpu)
                    ## Log probability ##
                    lp_adj[k, :, l] = log_pk0[k, l] + W1 * lp1_adj # + W2 * lp2_adj
        del lp1_adj # , lp2_adj

        ## Convert to log-sum-exp form ##
        lse_lp_adj = np.apply_along_axis(lambda a: logsumexp(a.reshape(nis, nl), axis=1, gpu=self.gpu), axis=1, arr=lp_adj.reshape(nk, nis * nl))

        ## Firm-level probabilities ##
        firm_level_lp_adj_both = np.apply_along_axis(lambda a: np.bincount(J1, a, minlength=nf), axis=1, arr=lse_lp_adj).T

        ### Take firm-level argmax ###
        return np.argmax(firm_level_lp_adj_first + firm_level_lp_adj_second + firm_level_lp_adj_both, axis=1)

    def compute_connectedness_measure(self, all=False):
        '''
        Computes graph connectedness measure among the movers within each type and updates self.connectedness to be the smallest value.

        Arguments:
            all (bool): if True, set self.connectedness to be the vector of connectedness for all worker types instead of the minimum
        '''
        nl, nk = self.nl, self.nk
        EV = np.zeros(shape=nl)
        pk1 = np.reshape(self.pk1, (nk, nk, nl))
        pr = (self.NNm.T * pk1.T).T

        for l in range(nl):
            # Compute adjacency matrix
            A = pr[:, :, l]
            A /= A.sum()
            A = (A + A.T) / 2
            D = np.diag(np.sum(A, axis=1) ** (-0.5))
            L = np.eye(nk) - D @ A @ D
            try:
                evals, evecs = np.linalg.eig(L)
            except np.linalg.LinAlgError as e:
                warnings.warn("Linear algebra error encountered when computing connectedness measure. This can likely be corrected by increasing the value of 'd_prior_movers' in tw.blm_params().")
                raise np.linalg.LinAlgError(e)
            EV[l] = sorted(evals)[1]

        if all:
            self.connectedness = EV
        self.connectedness = np.abs(EV).min()

    def plot_log_earnings(self, period='first', xlabel='firm class k', ylabel='log-earnings', grid=True, dpi=None):
        '''
        Plot log-earnings by worker-firm type pairs.

        Arguments:
            period (str): 'first' plots log-earnings in the first period; 'second' plots log-earnings in the second period; 'all' plots the average over log-earnings in the first and second periods
            xlabel (str): label for x-axis
            ylabel (str): label for y-axis
            grid (bool): if True, plot grid
            dpi (float or None): dpi for plot
        '''
        nl, nk = self.nl, self.nk
        A1, A2 = self._sort_parameters(self.A1, self.A2, sort_firm_types=True)

        # Compute average log-earnings
        if period == 'first':
            A_all = A1
        elif period == 'second':
            A_all = A2
        elif period == 'all':
            # FIXME should the mean account for the log?
            A_all = (A1 + A2) / 2 # np.log((np.exp(self.A1) + np.exp(self.A2)) / 2)
        else:
            raise ValueError(f"`period` must be one of 'first', 'second' or 'all', but input specifies {period!r}.")

        # Plot
        if dpi is not None:
            plt.figure(dpi=dpi)
        x_axis = np.arange(1, nk + 1)
        for l in range(nl):
            plt.plot(x_axis, A_all[l, :])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(x_axis)
        if grid:
            plt.grid()
        plt.show()

    def plot_type_proportions(self, period='first', subset='all', xlabel='firm class k', ylabel='type proportions', title='Proportions of worker types', dpi=None):
        '''
        Plot proportions of worker types at each firm class.

        Arguments:
            period (str): 'first' plots type proportions in the first period; 'second' plots type proportions in the second period; 'all' plots the average over type proportions in the first and second periods
            subset (str): 'all' plots a weighted average over movers and stayers; 'movers' plots movers; 'stayers' plots stayers
            xlabel (str): label for x-axis
            ylabel (str): label for y-axis
            title (str): plot title
            dpi (float or None): dpi for plot
        '''
        nl, nk = self.nl, self.nk
        if subset == 'movers':
            if self.NNm is None:
                raise ValueError('The BLM estimation must be run on movers (and NNm must be computed) before plotting type proportions for movers.')
            A1, A2, pk1, NNm = self._sort_parameters(self.A1, self.A2, pk1=self.pk1, NNm=self.NNm, sort_firm_types=True)
        elif subset == 'stayers':
            if self.NNs is None:
                raise ValueError('The BLM estimation must be run on stayers (and NNs must be computed) before plotting type proportions for stayers.')
            A1, A2, pk0, NNs = self._sort_parameters(self.A1, self.A2, pk0=self.pk0, NNs=self.NNs, sort_firm_types=True)
        elif subset == 'all':
            if (self.NNm is None) or (self.NNs is None):
                raise ValueError('The BLM estimation must be run on both movers and stayers (and both NNm and NNs must be computed) before plotting type proportions for all.')
            A1, A2, pk1, pk0, NNm, NNs = self._sort_parameters(self.A1, self.A2, pk1=self.pk1, pk0=self.pk0, NNm=self.NNm, NNs=self.NNs, sort_firm_types=True)

        ## Extract subset(s) ##
        if subset == 'movers':
            NNm_1 = np.sum(NNm, axis=1)
            NNm_2 = np.sum(NNm, axis=0)
            reshaped_pk1 = np.reshape(pk1, (nk, nk, nl))
            pk_period1 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=1).T / NNm_1).T
            pk_period2 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=0).T / NNm_2).T
        elif subset == 'stayers':
            pk_period1 = pk0
            pk_period2 = pk0
        elif subset == 'all':
            NNm_1 = np.sum(NNm, axis=1)
            NNm_2 = np.sum(NNm, axis=0)
            # First, pk1 #
            reshaped_pk1 = np.reshape(pk1, (nk, nk, nl))
            pk1_period1 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=1).T / NNm_1).T
            pk1_period2 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=0).T / NNm_2).T
            # Second, take weighted average over pk1 and pk0 #
            pk_period1 = ((NNm_1 * pk1_period1.T + NNs * pk0.T) / (NNm_1 + NNs)).T
            pk_period2 = ((NNm_2 * pk1_period2.T + NNs * pk0.T) / (NNm_2 + NNs)).T
        else:
            raise ValueError(f"`subset` must be one of 'all', 'movers' or 'stayers', but input specifies {subset!r}.")

        ## Consider correct period(s) ##
        if period == 'first':
            pk_mean = pk_period1
        elif period == 'second':
            pk_mean = pk_period2
        elif period == 'all':
            pk_mean = (pk_period1 + pk_period2) / 2
        else:
            raise ValueError(f"`period` must be one of 'first', 'second' or 'all', but input specifies {period!r}.")

        ## Compute cumulative sum ##
        pk_cumsum = np.cumsum(pk_mean, axis=1)

        ## Plot ##
        fig, ax = plt.subplots(dpi=dpi)
        x_axis = np.arange(1, nk + 1).astype(str)
        ax.bar(x_axis, pk_mean[:, 0])
        for l in range(1, nl):
            ax.bar(x_axis, pk_mean[:, l], bottom=pk_cumsum[:, l - 1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.show()

    def plot_type_flows(self, method='stacked', title='Worker flows', axis_label='firm class k', subplot_title='worker type', n_cols=3, circle_scale=1, dpi=None, opacity=0.4, font_size=15):
        '''
        Plot flows of worker types between each firm class.

        Arguments:
            method (str): 'stacked' for stacked plot; 'sankey' for Sankey plot
            title (str): plot title
            axis_label (str): label for axes (for stacked)
            subplot_title (str): label for subplots (for stacked)
            n_cols (int): number of subplot columns (for stacked)
            circle_scale (float): size scale for circles (for stacked)
            dpi (float or None): dpi for plot (for stacked)
            opacity (float): opacity of flows (for Sankey)
            font_size (float): font size for plot (for Sankey)
        '''
        if self.NNm is None:
            raise ValueError('The BLM estimation must be run on movers (and NNm must be computed) before plotting type flows.')

        if method not in ['stacked', 'sankey']:
            raise ValueError(f"`method` must be one of 'stacked' or 'sankey', but input specifies {method!r}.")

        ## Extract parameters ##
        nl, nk = self.nl, self.nk
        _, _, pk1, NNm = self._sort_parameters(self.A1, self.A2, pk1=self.pk1, NNm=self.NNm, sort_firm_types=True)

        ## Compute worker flows ##
        reshaped_pk1 = np.reshape(pk1, (nk, nk, nl))
        mover_flows = (NNm.T * reshaped_pk1.T).T

        if method == 'stacked':
            ## Compute number of subplot rows ##
            n_rows = nl // n_cols
            if n_rows * n_cols < nl:
                # If the bottom column won't be filled
                n_rows += 1

            ## Create subplots ##
            fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True, dpi=dpi)

            ## Create axes ##
            x_vals, y_vals = np.meshgrid(np.arange(nk) + 1, np.arange(nk) + 1, indexing='ij')

            ## Generate plots ##
            l = 0
            for row in axs:
                for ax in row:
                    if l < nl:
                        ax.scatter(x_vals, y_vals, s=(circle_scale * mover_flows[:, :, l]))
                        ax.set_title(f'{subplot_title} {l + 1}')
                        ax.grid()
                        l += 1
                    else:
                        fig.delaxes(ax)

            plt.setp(axs, xticks=np.arange(nk) + 1, yticks=np.arange(nk) + 1)
            fig.supxlabel(f'{axis_label}, period 1')
            fig.supylabel(f'{axis_label}, period 2')
            fig.suptitle(f'{title}')
            plt.tight_layout()
            plt.show()
        elif method == 'sankey':
            colors = np.array(
                [
                    [31, 119, 180],
                    [255, 127, 14],
                    [44, 160, 44],
                    [214, 39, 40],
                    [148, 103, 189],
                    [140, 86, 75],
                    [227, 119, 194],
                    [127, 127, 127],
                    [188, 189, 34],
                    [23, 190, 207],
                    [255, 0, 255]
                ]
            )

            ## Sankey with legend ##
            # Source: https://stackoverflow.com/a/76223740/17333120
            sankey = go.Sankey(
                # Define nodes
                node=dict(
                    pad=15,
                    thickness=1,
                    line=dict(color='white', width=0),
                    label=[f'k={k + 1}' for k in range(nk)] + [f'k={k + 1}' for k in range(nk)],
                    color='white'
                ),
                link=dict(
                    # Source firm
                    source=np.repeat(np.arange(nk), nk * nl),
                    # Destination firm
                    target=np.tile(np.repeat(np.arange(nk), nl), nk) + nk,
                    # Worker type
                    label=[f'l={l + 1}' for _ in range(nk) for _ in range(nk) for l in range(nl)],
                    # Worker flows
                    value=mover_flows.flatten(),
                    # Color (specify mean for each l, and for each k go from -80 below the mean to +80 above the mean)
                    color=[f'rgba({str(list(np.minimum(255, np.maximum(0, colors[l, :] - 80) + 160 * k / (nk - 1))))[1: -1]}, {opacity})' for k in range(nk) for _ in range(nk) for l in range(nl)]
                )
            )

            legend = []
            for l in range(nl):
                legend.append(
                    go.Scatter(
                        mode='markers',
                        x=[None],
                        y=[None],
                        marker=dict(color=f'rgba({str(list(colors[l, :]))[1: -1]}, {opacity})', symbol='square'),
                        name=f'$l={l + 1}$',
                    )
                )

            traces = [sankey] + legend
            layout = go.Layout(
                showlegend=True,
                plot_bgcolor='rgba(0, 0, 0, 0)',
            )

            fig = go.Figure(data=traces, layout=layout)
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)
            fig.update_layout(title_text=title, font_size=font_size)
            fig.show()

class BLMEstimator:
    '''
    Class for estimating BLM using multiple sets of starting values.

    Arguments:
        params (ParamsDict): dictionary of parameters for BLM estimation. Run tw.blm_params().describe_all() for descriptions of all valid parameters.
    '''

    def __init__(self, params):
        self.params = params
        # No initial model
        self.model = None
        # No likelihoods yet
        self.liks_high = None
        self.liks_low = None
        # No paths of likelihoods yet
        self.liks_all = None
        # No connectedness yet
        self.connectedness_high = None
        self.connectedness_low = None

    def _fit_model(self, jdata, iter, rng=None):
        '''
        Generate model and run fit_movers_cstr_uncstr() given parameters.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            iter (int): iteration
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        model = BLMModel(self.params, rng)
        if iter % 2 == 0:
            # Include linear additivity
            model.fit_movers_cstr_uncstr(jdata, linear_additivity=True)
        else:
            # Don't include linear additivity
            model.fit_movers_cstr_uncstr(jdata, linear_additivity=False)
        return model

    def fit(self, jdata, sdata, n_init=20, n_best=5, ncore=1, rng=None):
        '''
        Estimate BLM using multiple sets of starting values.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            n_init (int): number of starting values
            n_best (int): take the n_best estimates with the highest likelihoods, and then take the estimate with the highest connectedness
            ncore (int): number of cores for multiprocessing
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        ## Estimate model ##
        # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
        seeds = rng.bit_generator._seed_seq.spawn(n_init)
        if ncore > 1:
            # Multiprocessing
            with Pool(processes=ncore) as pool:
                sim_model_lst = list(tqdm(pool.imap(tw.util.f_star, [(self._fit_model, (jdata, i, np.random.default_rng(seed))) for i, seed in enumerate(seeds)]), total=n_init))
                # sim_model_lst = pool.starmap(self._fit_model, tqdm([(jdata, np.random.default_rng(seed)) for seed in seeds], total=n_init))
        else:
            # No multiprocessing
            sim_model_lst = list(tqdm(map(tw.util.f_star, [(self._fit_model, (jdata, i, np.random.default_rng(seed))) for i, seed in enumerate(seeds)]), total=n_init))
            # sim_model_lst = itertools.starmap(self._fit_model, tqdm([(jdata, np.random.default_rng(seed)) for seed in seeds], total=n_init))

        # Sort by likelihoods FIXME better handling if connectedness is None
        sorted_zipped_models = sorted([(model.lik1, model) for model in sim_model_lst if ((model.connectedness is not None) and (not pd.isna(model.pk1).any()))], reverse=True, key=lambda a: a[0])
        sorted_lik_models = [model for _, model in sorted_zipped_models]

        # Make sure at least one model converged
        if len(sorted_lik_models) == 0:
            raise ValueError('All starting values converged to NaN. Please try a different set of starting values.')

        ## Save likelihood vs. connectedness for all models ##
        # Save likelihoods for n_best
        liks_high = np.zeros(shape=n_best)
        # Save connectedness for n_best
        connectedness_high = np.zeros(shape=n_best)
        # Save likelihoods for not n_best
        liks_low = np.zeros(shape=len(sorted_lik_models) - n_best)
        # Save connectedness for not n_best
        connectedness_low = np.zeros(shape=len(sorted_lik_models) - n_best)
        # Save paths of likelihoods
        liks_all = []
        # Save paths of connectedness
        connectedness_all = []
        for i, model in enumerate(sorted_lik_models):
            liks_all.append(model.liks1)
            connectedness_all.append(model.connectedness)
            if i < n_best:
                liks_high[i] = model.lik1
                connectedness_high[i] = model.connectedness
            else:
                liks_low[i - n_best] = model.lik1
                connectedness_low[i - n_best] = model.connectedness
        self.liks_high = liks_high
        self.connectedness_high = connectedness_high
        self.liks_low = liks_low
        self.connectedness_low = connectedness_low
        self.liks_all = liks_all
        self.connectedness_all = connectedness_all

        # Take the n_best best estimates and find the lowest connectedness
        best_lik_models = sorted_lik_models[: n_best]
        sorted_zipped_models = sorted([(model.connectedness, model) for model in best_lik_models], reverse=True, key=lambda a: a[0])
        best_model = sorted_zipped_models[0][1]

        if self.params['verbose'] in [1, 2, 3]:
            print('liks_max:', best_model.lik1)
        self.model = best_model
        # Using best estimated parameters from fit_movers(), run fit_stayers()
        if self.params['verbose'] in [1, 2, 3]:
            print('Fitting stayers')
        # ## Set pk0 based on pk1 ##
        # nl, nk, pk1, NNm = self.model.nl, self.model.nk, self.model.pk1, self.model.NNm
        # NNm_1 = np.sum(NNm, axis=1)
        # NNm_2 = np.sum(NNm, axis=0)
        # reshaped_pk1 = np.reshape(pk1, (nk, nk, nl))
        # pk_period1 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=1).T / NNm_1).T
        # pk_period2 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=0).T / NNm_2).T
        # self.model.pk0 = (pk_period1 + pk_period2) / 2
        self.model.fit_stayers(sdata)

    def plot_log_earnings(self, period='first', xlabel='firm class k', ylabel='log-earnings', grid=True, dpi=None):
        '''
        Plot log-earnings by worker-firm type pairs.

        Arguments:
            period (str): 'first' plots log-earnings in the first period; 'second' plots log-earnings in the second period; 'all' plots the average over log-earnings in the first and second periods
            xlabel (str): label for x-axis
            ylabel (str): label for y-axis
            grid (bool): if True, plot grid
            dpi (float or None): dpi for plot
        '''
        if self.model is not None:
            self.model.plot_log_earnings(period=period, xlabel=xlabel, ylabel=ylabel, grid=grid, dpi=dpi)
        else:
            warnings.warn('Estimation has not yet been run.')

    def plot_type_proportions(self, period='first', subset='all', xlabel='firm class k', ylabel='type proportions', title='Proportions of worker types', dpi=None):
        '''
        Plot proportions of worker types at each firm class.

        Arguments:
            period (str): 'first' plots type proportions in the first period; 'second' plots type proportions in the second period; 'all' plots the average over type proportions in the first and second periods
            subset (str): 'all' plots a weighted average over movers and stayers; 'movers' plots movers; 'stayers' plots stayers
            xlabel (str): label for x-axis
            ylabel (str): label for y-axis
            title (str): plot title
            dpi (float or None): dpi for plot
        '''
        if self.model is not None:
            self.model.plot_type_proportions(period=period, subset=subset, xlabel=xlabel, ylabel=ylabel, title=title, dpi=dpi)
        else:
            warnings.warn('Estimation has not yet been run.')

    def plot_type_flows(self, method='stacked', title='Worker flows', axis_label='firm class k', subplot_title='worker type', n_cols=3, circle_scale=1, dpi=None, opacity=0.4, font_size=15):
        '''
        Plot flows of worker types between each firm class.

        Arguments:
            method (str): 'stacked' for stacked plot; 'sankey' for Sankey plot
            title (str): plot title
            axis_label (str): label for axes (for stacked)
            subplot_title (str): label for subplots (for stacked)
            n_cols (int): number of subplot columns (for stacked)
            circle_scale (float): size scale for circles (for stacked)
            dpi (float or None): dpi for plot (for stacked)
            opacity (float): opacity of flows (for Sankey)
            font_size (float): font size for plot (for Sankey)
        '''
        if self.model is not None:
            self.model.plot_type_flows(method=method, title=title, axis_label=axis_label, subplot_title=subplot_title, n_cols=n_cols, circle_scale=circle_scale, dpi=dpi, opacity=opacity, font_size=font_size)
        else:
            warnings.warn('Estimation has not yet been run.')

    def plot_liks_connectedness(self, only_n_best=False, jitter=False, dpi=None):
        '''
        Plot likelihoods vs connectedness for the estimations run.

        Arguments:
            only_n_best (bool): if True, only plot the n_best estimates
            jitter (bool): if True, jitter points to prevent overlap
            dpi (float or None): dpi for plot
        '''
        if (self.model is not None) and (self.liks_high is not None) and (self.connectedness_high is not None) and (self.liks_low is not None) and (self.connectedness_low is not None):
            if dpi is not None:
                plt.figure(dpi=dpi)
            # So best estimation only graphed once, drop index from liks_high and connectedness_high
            liks_high_lst = list(self.liks_high)
            connectedness_high_lst = list(self.connectedness_high)
            drop_index = list(zip(liks_high_lst, connectedness_high_lst)).index((self.model.lik1, self.model.connectedness))
            del liks_high_lst[drop_index]
            del connectedness_high_lst[drop_index]
            # Now graph
            if jitter:
                plot = jitter_scatter
            else:
                plot = plt.scatter
            if not only_n_best:
                plot(self.liks_low, self.connectedness_low, marker='o', facecolors='None', edgecolors='C0')
            plot(liks_high_lst, connectedness_high_lst, marker='^', facecolors='None', edgecolors='C1')
            plt.scatter(self.model.lik1, self.model.connectedness, marker=(6, 2, 45), facecolors='C2')
            plt.xlabel('Likelihood')
            plt.ylabel('Connectedness')
            plt.show()
        else:
            warnings.warn('Estimation has not yet been run.')

class BLMBootstrap:
    '''
    Class for estimating BLM using bootstrapping.

    Arguments:
        params (ParamsDict): dictionary of parameters for BLM estimation. Run tw.blm_params().describe_all() for descriptions of all valid parameters.
        model (BLMModel or None): estimated BLM model. For use with parametric bootstrap. None if running standard bootstrap.
    '''

    def __init__(self, params, model=None):
        self.params = params
        self.model = model
        # No initial models
        self.models = None

    def fit(self, jdata, sdata, n_samples=5, n_init_estimator=20, n_best=5, frac_movers=0.1, frac_stayers=0.1, method='parametric', cluster_params=None, reallocate=False, reallocate_jointly=True, reallocate_period='first', ncore=1, verbose=True, rng=None):
        '''
        Estimate bootstrap.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            n_samples (int): number of bootstrap samples to estimate
            n_init_estimator (int): number of starting guesses to estimate for each bootstrap sample
            n_best (int): take the n_best estimates with the highest likelihoods, and then take the estimate with the highest connectedness, for each bootstrap sample
            frac_movers (float): fraction of movers to draw (with replacement) for each bootstrap sample. For use with standard bootstrap.
            frac_stayers (float): fraction of stayers to draw (with replacement) for each bootstrap sample. For use with standard bootstrap.
            method (str): if 'parametric', estimate BLM model on full data, simulate worker types and wages using estimated parameters, estimate BLM model on each set of simulated data, and construct bootstrapped errors; if 'standard', estimate standard bootstrap by sampling from original data, estimating BLM model on each sample, and constructing bootstrapped errors
            cluster_params (ParamsDict or None): dictionary of parameters for clustering firms. Run bpd.cluster_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.cluster_params().
            reallocate (bool): if True and `method` is 'parametric', draw worker type proportions independently of firm type; if False, uses worker type proportions that are conditional on firm type
            reallocate_jointly (bool): if True, worker type proportions take the average over movers and stayers (i.e. all workers use the same type proportions); if False, consider movers and stayers separately
            reallocate_period (str): if 'first', compute type proportions based on first period parameters; if 'second', compute type proportions based on second period parameters; if 'all', compute type proportions based on average over first and second period parameters
            ncore (int): number of cores for multiprocessing
            verbose (bool): if True, print progress during data cleaning for each sample
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)
        if cluster_params is None:
            grouping = bpd.grouping.KMeans(n_clusters=jdata.n_clusters())
            cluster_params = bpd.cluster_params({'grouping': grouping})

        # Parameter dictionary
        params = self.params

        # Update clustering parameters
        cluster_params = cluster_params.copy()
        cluster_params['is_sorted'] = True
        cluster_params['copy'] = False

        if method == 'parametric':
            ### Parametric bootstrap ###
            ## Copy original wages and firm types ##
            yj = jdata.loc[:, ['y1', 'y2']].to_numpy().copy()
            ys = sdata.loc[:, ['y1', 'y2']].to_numpy().copy()
            gj = jdata.loc[:, ['g1', 'g2']].to_numpy().astype(int, copy=True)
            gs = sdata.loc[:, 'g1'].to_numpy().astype(int, copy=True)

            ## Unpack parameters ##
            model = self.model
            NNm, NNs = model.NNm, model.NNs
            pk1, pk0 = model.pk1, model.pk0

            ## Reallocate ##
            if reallocate:
                pk1, pk0 = tw.simblm._reallocate(pk1=pk1, pk0=pk0, NNm=NNm, NNs=NNs, reallocate_period=reallocate_period, reallocate_jointly=reallocate_jointly)

            models = []
            for _ in trange(n_samples):
                ## Simulate worker types and wages ##
                bdf = _simulate_types_wages(model, jdata, sdata, gj=gj, gs=gs, pk1=pk1, pk0=pk0, qi_j=None, qi_s=None, qi_cum_j=None, qi_cum_s=None, optimal_reallocation=False, reallocation_constraint_category=None, worker_types_as_ids=False, simulate_wages=True, return_long_df=True, store_worker_types=False, weighted=True, rng=rng)

                ## Cluster ##
                bdf = bdf.cluster(cluster_params, rng=rng)
                clusters_dict = bdf.loc[:, ['j', 'g']].groupby('j', sort=False)['g'].first().to_dict()
                del bdf
                with bpd.util.ChainedAssignment():
                    # Update clusters in jdata and sdata
                    jdata.loc[:, 'g1'] = jdata.loc[:, 'j1'].map(clusters_dict)
                    jdata.loc[:, 'g2'] = jdata.loc[:, 'j2'].map(clusters_dict)
                    sdata.loc[:, 'g1'] = sdata.loc[:, 'j1'].map(clusters_dict)
                    sdata.loc[:, 'g2'] = sdata.loc[:, 'g1']
                ## Run BLM estimator ##
                blm_fit_i = BLMEstimator(params)
                blm_fit_i.fit(jdata=jdata, sdata=sdata, n_init=n_init_estimator, n_best=n_best, ncore=ncore, rng=rng)
                models.append(blm_fit_i.model)
                del blm_fit_i

            with bpd.util.ChainedAssignment():
                ## Re-assign original wages and firm types ##
                jdata.loc[:, ['y1', 'y2']] = yj
                sdata.loc[:, ['y1', 'y2']] = ys
                jdata.loc[:, ['g1', 'g2']] = gj
                sdata.loc[:, 'g1'], sdata.loc[:, 'g2'] = (gs, gs)
        elif method == 'standard':
            ### Standard bootstrap ###
            wj = None
            if params['weighted'] and jdata._col_included('w'):
                wj = jdata['w1'].to_numpy() + jdata['w2'].to_numpy()
            ws = None
            if params['weighted'] and sdata._col_included('w'):
                ws = sdata['w1'].to_numpy() + sdata['w2'].to_numpy()

            models = []
            for _ in trange(n_samples):
                jdata_i = jdata.sample(frac=frac_movers, replace=True, weights=wj, random_state=rng)
                sdata_i = sdata.sample(frac=frac_stayers, replace=True, weights=ws, random_state=rng)
                # Cluster
                bdf = bpd.BipartiteDataFrame(pd.concat([jdata_i, sdata_i], axis=0, copy=True))
                # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
                bdf._set_attributes(jdata)
                # Clean and cluster
                bdf = bdf.clean(bpd.clean_params({'is_sorted': True, 'copy': False, 'verbose': verbose})).to_long(is_sorted=True, copy=False).cluster(cluster_params, rng=rng)
                clusters_dict = bdf.loc[:, ['j', 'g']].groupby('j', sort=False)['g'].first().to_dict()
                del bdf
                # Update clusters in jdata_i and sdata_i
                jdata_i.loc[:, 'g1'] = jdata_i.loc[:, 'j1'].map(clusters_dict)
                jdata_i.loc[:, 'g2'] = jdata_i.loc[:, 'j2'].map(clusters_dict)
                sdata_i.loc[:, 'g1'] = sdata_i.loc[:, 'j1'].map(clusters_dict)
                sdata_i.loc[:, 'g2'] = sdata_i.loc[:, 'g1']
                # Run BLM estimator
                blm_fit_i = BLMEstimator(params)
                blm_fit_i.fit(jdata=jdata_i, sdata=sdata_i, n_init=n_init_estimator, n_best=n_best, ncore=ncore, rng=rng)
                models.append(blm_fit_i.model)
                del jdata_i, sdata_i, blm_fit_i
        else:
            raise ValueError(f"`method` must be one of 'parametric' or 'standard', but input specifies {method!r}.")

        self.models = models

    def plot_log_earnings(self, period='first', xlabel='firm class k', ylabel='log-earnings', grid=True, dpi=None):
        '''
        Plot log-earnings by worker-firm type pairs.

        Arguments:
            period (str): 'first' plots log-earnings in the first period; 'second' plots log-earnings in the second period; 'all' plots the average over log-earnings in the first and second periods
            xlabel (str): label for x-axis
            ylabel (str): label for y-axis
            grid (bool): if True, plot grid
            dpi (float or None): dpi for plot
        '''
        if self.models is None:
            warnings.warn('Estimation has not yet been run.')
        else:
            nl, nk = self.params.get_multiple(('nl', 'nk'))

            # Compute average log-earnings
            A_all = np.zeros((len(self.models), nl, nk))
            for i, model in enumerate(self.models):
                # Sort by firm effects
                A1, A2 = model._sort_parameters(model.A1, model.A2, sort_firm_types=True)
                # Extract average log-earnings for each model
                if period == 'first':
                    A_all[i, :, :] = A1
                elif period == 'second':
                    A_all[i, :, :] = A2
                elif period == 'all':
                    # FIXME should the mean account for the log?
                    A_all[i, :, :] = (A1 + A2) / 2 # np.log((np.exp(A1) + np.exp(A2)) / 2)
                else:
                    raise ValueError(f"`period` must be one of 'first', 'second' or 'all', but input specifies {period!r}.")

            # Take mean over all models, and compute 2.5 and 97.5 percentiles
            A_mean = np.mean(A_all, axis=0)
            A_lb = np.percentile(A_all, 2.5, axis=0)
            A_ub = np.percentile(A_all, 97.5, axis=0)
            A_ci = np.stack([A_mean - A_lb, A_ub - A_mean], axis=0)

            # Plot
            if dpi is not None:
                plt.figure(dpi=dpi)
            x_axis = np.arange(1, nk + 1)
            for l in range(nl):
                plt.errorbar(x_axis, A_mean[l, :], yerr=A_ci[:, l, :], capsize=3)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(x_axis)
            if grid:
                plt.grid()
            plt.show()

    def plot_type_proportions(self, period='first', subset='all', xlabel='firm class k', ylabel='type proportions', title='Proportions of worker types', dpi=None):
        '''
        Plot proportions of worker types at each firm class.

        Arguments:
            period (str): 'first' plots type proportions in the first period; 'second' plots type proportions in the second period; 'all' plots the average over type proportions in the first and second periods
            subset (str): 'all' plots a weighted average over movers and stayers; 'movers' plots movers; 'stayers' plots stayers
            xlabel (str): label for x-axis
            ylabel (str): label for y-axis
            title (str): plot title
            dpi (float or None): dpi for plot
        '''
        if self.models is None:
            warnings.warn('Estimation has not yet been run.')
        else:
            nl, nk = self.params.get_multiple(('nl', 'nk'))

            ## Compute average type proportions ##
            pk_mean = np.zeros((nk, nl))
            for model in self.models:
                # Sort by firm effects
                A1, A2, pk1, pk0, NNm, NNs = model._sort_parameters(model.A1, model.A2, pk1=model.pk1, pk0=model.pk0, NNm=model.NNm, NNs=model.NNs, sort_firm_types=True)

                ## Extract subset(s) ##
                if subset == 'movers':
                    NNm_1 = np.sum(NNm, axis=1)
                    NNm_2 = np.sum(NNm, axis=0)
                    reshaped_pk1 = np.reshape(pk1, (nk, nk, nl))
                    pk_period1 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=1).T / NNm_1).T
                    pk_period2 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=0).T / NNm_2).T
                elif subset == 'stayers':
                    pk_period1 = pk0
                    pk_period2 = pk0
                elif subset == 'all':
                    NNm_1 = np.sum(NNm, axis=1)
                    NNm_2 = np.sum(NNm, axis=0)
                    # First, pk1 #
                    reshaped_pk1 = np.reshape(pk1, (nk, nk, nl))
                    pk1_period1 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=1).T / NNm_1).T
                    pk1_period2 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=0).T / NNm_2).T
                    # Second, take weighted average over pk1 and pk0 #
                    pk_period1 = ((NNm_1 * pk1_period1.T + NNs * pk0.T) / (NNm_1 + NNs)).T
                    pk_period2 = ((NNm_2 * pk1_period2.T + NNs * pk0.T) / (NNm_2 + NNs)).T
                else:
                    raise ValueError(f"`subset` must be one of 'all', 'movers' or 'stayers', but input specifies {subset!r}.")

                ## Consider correct period(s) ##
                if period == 'first':
                    pk_mean += pk_period1
                elif period == 'second':
                    pk_mean += pk_period2
                elif period == 'all':
                    pk_mean += (pk_period1 + pk_period2) / 2
                else:
                    raise ValueError(f"`period` must be one of 'first', 'second' or 'all', but input specifies {period!r}.")

            ## Take mean over all models ##
            pk_mean /= len(self.models)

            ## Compute cumulative sum ##
            pk_cumsum = np.cumsum(pk_mean, axis=1)

            ## Plot ##
            fig, ax = plt.subplots(dpi=dpi)
            x_axis = np.arange(1, nk + 1).astype(str)
            ax.bar(x_axis, pk_mean[:, 0])
            for l in range(1, nl):
                ax.bar(x_axis, pk_mean[:, l], bottom=pk_cumsum[:, l - 1])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            plt.show()

    def plot_type_flows(self, method='stacked', title='Worker flows', axis_label='firm class k', subplot_title='worker type', n_cols=3, circle_scale=1, dpi=None, opacity=0.4, font_size=15):
        '''
        Plot flows of worker types between each firm class.

        Arguments:
            method (str): 'stacked' for stacked plot; 'sankey' for Sankey plot
            title (str): plot title
            axis_label (str): label for axes (for stacked)
            subplot_title (str): label for subplots (for stacked)
            n_cols (int): number of subplot columns (for stacked)
            circle_scale (float): size scale for circles (for stacked)
            dpi (float or None): dpi for plot (for stacked)
            opacity (float): opacity of flows (for Sankey)
            font_size (float): font size for plot (for Sankey)
        '''
        if self.models is None:
            warnings.warn('Estimation has not yet been run.')
        elif method not in ['stacked', 'sankey']:
            raise ValueError(f"`method` must be one of 'stacked' or 'sankey', but input specifies {method!r}.")
        else:
            ## Extract parameters ##
            nl, nk = self.nl, self.nk

            # Compute average type proportions
            pk1_mean = np.zeros((nk, nl))
            for model in self.models:
                # Sort by firm effects
                _, _, pk1, _, NNm, _ = model._sort_parameters(model.A1, model.A2, pk1=model.pk1, pk0=model.pk0, NNm=model.NNm, NNs=model.NNs, sort_firm_types=True)
                pk1_mean += pk1

            ## Take mean over all models ##
            pk1_mean /= len(self.models)

            ## Compute worker flows ##
            reshaped_pk1 = np.reshape(pk1_mean, (nk, nk, nl))
            mover_flows = (NNm.T * reshaped_pk1.T).T

            if method == 'stacked':
                ## Compute number of subplot rows ##
                n_rows = nl // n_cols
                if n_rows * n_cols < nl:
                    # If the bottom column won't be filled
                    n_rows += 1

                ## Create subplots ##
                fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, sharey=True, dpi=dpi)

                ## Create axes ##
                x_vals, y_vals = np.meshgrid(np.arange(nk) + 1, np.arange(nk) + 1, indexing='ij')

                ## Generate plots ##
                l = 0
                for row in axs:
                    for ax in row:
                        if l < nl:
                            ax.scatter(x_vals, y_vals, s=(circle_scale * mover_flows[:, :, l]))
                            ax.set_title(f'{subplot_title} {l + 1}')
                            ax.grid()
                            l += 1
                        else:
                            fig.delaxes(ax)

                plt.setp(axs, xticks=np.arange(nk) + 1, yticks=np.arange(nk) + 1)
                fig.supxlabel(f'{axis_label}, period 1')
                fig.supylabel(f'{axis_label}, period 2')
                fig.suptitle(f'{title}')
                plt.tight_layout()
                plt.show()
            elif method == 'sankey':
                colors = np.array(
                    [
                        [31, 119, 180],
                        [255, 127, 14],
                        [44, 160, 44],
                        [214, 39, 40],
                        [148, 103, 189],
                        [140, 86, 75],
                        [227, 119, 194],
                        [127, 127, 127],
                        [188, 189, 34],
                        [23, 190, 207],
                        [255, 0, 255]
                    ]
                )

                ## Sankey with legend ##
                # Source: https://stackoverflow.com/a/76223740/17333120
                sankey = go.Sankey(
                    # Define nodes
                    node=dict(
                        pad=15,
                        thickness=1,
                        line=dict(color='white', width=0),
                        label=[f'k={k + 1}' for k in range(nk)] + [f'k={k + 1}' for k in range(nk)],
                        color='white'
                    ),
                    link=dict(
                        # Source firm
                        source=np.repeat(np.arange(nk), nk * nl),
                        # Destination firm
                        target=np.tile(np.repeat(np.arange(nk), nl), nk) + nk,
                        # Worker type
                        label=[f'l={l + 1}' for _ in range(nk) for _ in range(nk) for l in range(nl)],
                        # Worker flows
                        value=mover_flows.flatten(),
                        # Color (specify mean for each l, and for each k go from -80 below the mean to +80 above the mean)
                        color=[f'rgba({str(list(np.minimum(255, np.maximum(0, colors[l, :] - 80) + 160 * k / (nk - 1))))[1: -1]}, {opacity})' for k in range(nk) for _ in range(nk) for l in range(nl)]
                    )
                )

                legend = []
                for l in range(nl):
                    legend.append(
                        go.Scatter(
                            mode='markers',
                            x=[None],
                            y=[None],
                            marker=dict(color=f'rgba({str(list(colors[l, :]))[1: -1]}, {opacity})', symbol='square'),
                            name=f'$l={l + 1}$',
                        )
                    )

                traces = [sankey] + legend
                layout = go.Layout(
                    showlegend=True,
                    plot_bgcolor='rgba(0, 0, 0, 0)',
                )

                fig = go.Figure(data=traces, layout=layout)
                fig.update_xaxes(visible=False)
                fig.update_yaxes(visible=False)
                fig.update_layout(title_text=title, font_size=font_size)
                fig.show()

class BLMVarianceDecomposition:
    '''
    Class for estimating BLM variance decomposition using bootstrapping. Results are stored in class attribute .res, which gives a dictionary where the key 'var_decomp' gives the results for the variance decomposition, and the key 'var_decomp_comp' optionally gives the results for the variance decomposition with complementarities.

    Arguments:
        params (ParamsDict): dictionary of parameters for BLM estimation. Run tw.blm_params().describe_all() for descriptions of all valid parameters.
        model (BLMModel): estimated BLM model
    '''

    def __init__(self, params, model):
        self.params = params
        self.model = model
        # No initial results
        self.res = None

    def fit(self, jdata, sdata, n_samples=5, reallocate=False, reallocate_jointly=True, reallocate_period='first', uncorrelated_errors=False, Q_var=None, Q_cov=None, complementarities=True, firm_clusters_as_ids=True, worker_types_as_ids=True, weighted=True, ncore=1, rng=None):
        '''
        Estimate variance decomposition.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            n_samples (int): number of bootstrap samples to estimate
            reallocate (bool): if True, draw worker type proportions independently of firm type; if False, uses worker type proportions that are conditional on firm type
            reallocate_jointly (bool): if True, worker type proportions take the average over movers and stayers (i.e. all workers use the same type proportions); if False, consider movers and stayers separately
            reallocate_period (str): if 'first', compute type proportions based on first period parameters; if 'second', compute type proportions based on second period parameters; if 'all', compute type proportions based on average over first and second period parameters
            uncorrelated_errors (bool): set to True if using weighted estimator and errors are assumed to be uncorrelated within job spells. Set to False if using weighted estimator and errors are assumed to be perfectly correlated within job spells.
            Q_var (list of Q variances): list of Q matrices to use when estimating variance term; None is equivalent to tw.Q.VarPsi() without controls, or tw.Q.VarCovariate('psi') with controls
            Q_cov (list of Q covariances): list of Q matrices to use when estimating covariance term; None is equivalent to tw.Q.CovPsiAlpha() without controls, or tw.Q.CovCovariate('psi', 'alpha') with controls
            complementarities (bool): if True, estimate R^2 of regression with complementarities (by adding in all worker-firm interactions). Only allowed when firm_clusters_as_ids=True and worker_types_as_ids=True.
            firm_clusters_as_ids (bool): if True, replace firm ids with firm clusters
            worker_types_as_ids (bool): if True, replace worker ids with simulated worker types
            weighted (bool): if True, use weighted estimators
            ncore (int): number of cores for multiprocessing
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if complementarities and ((not firm_clusters_as_ids) or (not worker_types_as_ids)):
            raise ValueError('If `complementarities=True`, then must also set `firm_clusters_as_ids=True` and `worker_types_as_ids=True`.')

        if rng is None:
            rng = np.random.default_rng(None)

        ## Unpack parameters ##
        params = self.params
        model = self.model
        nl = model.nl
        NNm, NNs = model.NNm, model.NNs
        pk1, pk0 = model.pk1, model.pk0

        # FE parameters
        no_cat_controls = (params['categorical_controls'] is None) or (len(params['categorical_controls']) == 0)
        no_cts_controls = (params['continuous_controls'] is None) or (len(params['continuous_controls']) == 0)
        no_controls = (no_cat_controls and no_cts_controls)
        if no_controls:
            # If no controls
            fe_params = tw.fe_params()
        else:
            # If controls
            fe_params = tw.fecontrol_params()
            if not no_cat_controls:
                fe_params['categorical_controls'] = params['categorical_controls'].keys()
            if not no_cts_controls:
                fe_params['continuous_controls'] = params['continuous_controls'].keys()
        fe_params['weighted'] = weighted
        fe_params['ho'] = False
        fe_params['uncorrelated_errors'] = uncorrelated_errors
        if Q_var is not None:
            fe_params['Q_var'] = Q_var
        if Q_cov is not None:
            fe_params['Q_cov'] = Q_cov
        if complementarities:
            fe_params_comp = fe_params.copy()
            fe_params_comp['Q_var'] = []
            fe_params_comp['Q_cov'] = []

        # Copy original wages, firm types, and optionally ids
        yj = jdata.loc[:, ['y1', 'y2']].to_numpy().copy()
        ys = sdata.loc[:, ['y1', 'y2']].to_numpy().copy()
        gj = jdata.loc[:, ['g1', 'g2']].to_numpy().astype(int, copy=True)
        gs = sdata.loc[:, 'g1'].to_numpy().astype(int, copy=True)
        if firm_clusters_as_ids:
            jj = jdata.loc[:, ['j1', 'j2']].to_numpy().copy()
            js = sdata.loc[:, 'j1'].to_numpy().copy()
            with bpd.util.ChainedAssignment():
                jdata.loc[:, ['j1', 'j2']] = gj
                sdata.loc[:, 'j1'], sdata.loc[:, 'j2'] = (gs, gs)
        if worker_types_as_ids:
            ij = jdata.loc[:, 'i'].to_numpy().copy()
            is_ = sdata.loc[:, 'i'].to_numpy().copy()
        tj = False
        ts = False
        if not jdata._col_included('t'):
            jdata = jdata.construct_artificial_time(is_sorted=True, copy=False)
            tj = True
        if not sdata._col_included('t'):
            sdata = sdata.construct_artificial_time(is_sorted=True, copy=False)
            ts = True

        ## Reallocate ##
        if reallocate:
            pk1, pk0 = tw.simblm._reallocate(pk1=pk1, pk0=pk0, NNm=NNm, NNs=NNs, reallocate_period=reallocate_period, reallocate_jointly=reallocate_jointly)

        ## Run bootstrap ##
        res_lst = []
        if complementarities:
            res_lst_comp = []
        for i in trange(n_samples):
            ## Simulate worker types and wages ##
            bdf = _simulate_types_wages(model, jdata, sdata, gj=gj, gs=gs, pk1=pk1, pk0=pk0, qi_j=None, qi_s=None, qi_cum_j=None, qi_cum_s=None, optimal_reallocation=False, reallocation_constraint_category=None, worker_types_as_ids=worker_types_as_ids, simulate_wages=True, return_long_df=True, store_worker_types=False, weighted=(weighted and uncorrelated_errors), rng=rng)
            ## Estimate OLS ##
            if no_controls:
                fe_estimator = tw.FEEstimator(bdf, fe_params)
            else:
                fe_estimator = tw.FEControlEstimator(bdf, fe_params)
            fe_estimator.fit()
            res_lst.append(fe_estimator.summary)
            if complementarities:
                ## Estimate OLS with complementarities ##
                bdf.loc[:, 'i'] = pd.factorize(bdf.loc[:, 'i'].to_numpy() + nl * bdf.loc[:, 'j'].to_numpy())[0]
                if no_controls:
                    fe_estimator = tw.FEEstimator(bdf, fe_params_comp)
                else:
                    fe_estimator = tw.FEControlEstimator(bdf, fe_params_comp)
                fe_estimator.fit()
                res_lst_comp.append(fe_estimator.summary)

        with bpd.util.ChainedAssignment():
            # Restore original wages and optionally ids
            jdata.loc[:, ['y1', 'y2']] = yj
            sdata.loc[:, ['y1', 'y2']] = ys
            if firm_clusters_as_ids:
                jdata.loc[:, ['j1', 'j2']] = jj
                sdata.loc[:, 'j1'], sdata.loc[:, 'j2'] = (js, js)
            if worker_types_as_ids:
                jdata.loc[:, 'i'] = ij
                sdata.loc[:, 'i'] = is_

        ## Unpack results ##
        res = {k: np.zeros(n_samples) for k in res_lst[0].keys()}
        for i in range(n_samples):
            for k, v in res_lst[i].items():
                res[k][i] = v
        if complementarities:
            res_comp = {k: np.zeros(n_samples) for k in res_lst_comp[0].keys()}
            for i in range(n_samples):
                for k, v in res_lst_comp[i].items():
                    res_comp[k][i] = v

        # Remove '_fe' from result names
        res = {k.replace('_fe', ''): v for k, v in res.items()}
        if complementarities:
            res_comp = {k.replace('_fe', ''): v for k, v in res_comp.items()}

        # Drop time column
        if tj:
            jdata = jdata.drop('t', axis=1, inplace=True, allow_optional=True)
        if ts:
            sdata = sdata.drop('t', axis=1, inplace=True, allow_optional=True)

        self.res = {'var_decomp': res, 'var_decomp_comp': None}
        if complementarities:
            self.res['var_decomp_comp'] = res_comp

class BLMReallocation:
    '''
    Class for estimating BLM reallocation exercise using bootstrapping. Results are stored in class attribute .res, which gives a dictionary with the following structure: baseline results are stored in key 'baseline'. Reallocation results are stored in key 'reallocation'. Within each sub-dictionary, primary outcome results are stored in the key 'outcome', means are stored in the key 'mean', categorical results are stored in the key 'cat', continuous results are stored in the key 'cts', type proportions for movers are stored in the key 'pk1', type proportions for stayers are stored in the key 'pk0', firm-level mover flow counts are stored in the key 'NNm', and firm-level stayer counts are stored in the key 'NNs'.

    Arguments:
        model (BLMModel): estimated BLM model
    '''

    def __init__(self, model):
        self.model = model
        # No initial results
        self.res = None

    def fit(self, jdata, sdata, quantiles=None, n_samples=5, reallocate_jointly=True, reallocate_period='first', categorical_sort_cols=None, continuous_sort_cols=None, unresidualize_col=None, optimal_reallocation=False, reallocation_constraint_category=None, reallocation_scaling_col=None, qi_j=None, qi_s=None, qi_cum_j=None, qi_cum_s=None, ncore=1, weighted=True, rng=None):
        '''
        Estimate reallocation exercise.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            quantiles (NumPy Array or None): income quantiles to compute; if None, computes percentiles from 1-100 (specifically, np.arange(101) / 100)
            n_samples (int): number of bootstrap samples to estimate
            reallocate_jointly (bool): if True, worker type proportions take the average over movers and stayers (i.e. all workers use the same type proportions); if False, consider movers and stayers separately
            reallocate_period (str): if 'first', compute type proportions based on first period parameters; if 'second', compute type proportions based on second period parameters; if 'all', compute type proportions based on average over first and second period parameters
            categorical_sort_cols (dict or None): in addition to standard quantiles results, return average income grouped by the alternative column(s) given (which are represented by the dictionary {column: number of quantiles to compute}). For categorical variables, use each group as a bin and take the average income within that bin. None is equivalent to {}.
            continuous_sort_cols (dict or None): in addition to standard quantiles results, return average income grouped by the alternative column(s) given (which are represented by the dictionary {column: list of quantiles to compute}). For continuous variables, create bins based on the list of quantiles given in the dictionary. The list of quantiles must start at 0 and end at 1. None is equivalent to {}.
            unresidualize_col (str or None): column with predicted values that are residualized out, which will be added back in before computing outcomes in order to unresidualize the values; None leaves outcomes unchanged
            optimal_reallocation (bool or str): if not False, reallocate workers to new firms to maximize ('max') or minimize ('min') total output
            reallocation_constraint_category (str or None): specify categorical column to constrain reallocation so that workers must reallocate within their own category; if None, no constraints on how workers can reallocate
            reallocation_scaling_col (str or None): specify column to use to scale outcomes when computing optimal reallocation (i.e. multiply outcomes by an observation-level factor); if None, don't scale outcomes
            qi_j (NumPy Array or None): (use with optimal_reallocation to assign workers to maximum probability worker type based on observation-level probabilities) probabilities for each mover observation to be each worker type; None if pk1 or qi_cum_j is not None
            qi_s (NumPy Array or None): (use with optimal_reallocation to assign workers to maximum probability worker type based on observation-level probabilities) probabilities for each stayer observation to be each worker type; None if pk0 or qi_cum_s is not None
            qi_cum_j (NumPy Array or None): (use with optimal_reallocation to assign workers to worker types probabilistically based on observation-level probabilities) cumulative probabilities for each mover observation to be each worker type; None if pk1 or qi_j is not None
            qi_cum_s (NumPy Array or None): (use with optimal_reallocation to assign workers to worker types probabilistically based on observation-level probabilities) cumulative probabilities for each stayer observation to be each worker type; None if pk0 or qi_s is not None
            ncore (int): number of cores for multiprocessing
            weighted (bool): if True, use weights
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if optimal_reallocation and ((qi_j is None) + (qi_s is None) + (qi_cum_j is None) + (qi_cum_s is None) != 2):
            raise ValueError('With `optimal_reallocation`, must specify one of `qi_j` or `qi_cum_j` for movers and `qi_s` or `qi_cum_s` for stayers.')

        if quantiles is None:
            quantiles = np.arange(101) / 100
        if categorical_sort_cols is None:
            categorical_sort_cols = {}
        if continuous_sort_cols is None:
            continuous_sort_cols = {}
        if rng is None:
            rng = np.random.default_rng(None)

        ## Unpack parameters ##
        model = self.model
        nl, nk = model.nl, model.nk
        NNm, NNs = model.NNm, model.NNs

        # Make sure continuous quantiles start at 0 and end at 1
        for col_cts, quantiles_cts in continuous_sort_cols.items():
            if quantiles_cts[0] != 0:
                raise ValueError(f'Lowest quantile associated with continuous column {col_cts} must be 0.')
            elif quantiles_cts[-1] != 1:
                raise ValueError(f'Highest quantile associated with continuous column {col_cts} must be 1.')

        # Copy original wages, firm ids, and firm types
        yj = jdata.loc[:, ['y1', 'y2']].to_numpy().copy()
        ys = sdata.loc[:, ['y1', 'y2']].to_numpy().copy()
        jj = jdata.loc[:, ['j1', 'j2']].to_numpy().copy()
        js = sdata.loc[:, 'j1'].to_numpy().copy()
        gj = jdata.loc[:, ['g1', 'g2']].to_numpy().astype(int, copy=True)
        gs = sdata.loc[:, 'g1'].to_numpy().astype(int, copy=True)
        tj = False
        ts = False
        if not jdata._col_included('t'):
            jdata = jdata.construct_artificial_time(is_sorted=True, copy=False)
            tj = True
        if not sdata._col_included('t'):
            sdata = sdata.construct_artificial_time(is_sorted=True, copy=False)
            ts = True

        ## Reallocate ##
        if optimal_reallocation:
            pk1, pk0 = None, None
        else:
            pk1, pk0 = tw.simblm._reallocate(pk1=model.pk1, pk0=model.pk0, NNm=NNm, NNs=NNs, reallocate_period=reallocate_period, reallocate_jointly=reallocate_jointly)

        ## Baseline ##
        res_cat_baseline = {}
        res_cts_baseline = {}
        if reallocation_scaling_col is not None:
            res_scaled_cat_baseline = {}
            res_scaled_cts_baseline = {}

        # Convert to BipartitePandas DataFrame
        bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
        # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
        bdf._set_attributes(jdata)
        bdf = bdf.to_long(is_sorted=True, copy=False)
        # Compute quantiles
        y = bdf.loc[:, 'y'].to_numpy()
        if unresidualize_col is not None:
            y += bdf.loc[:, unresidualize_col]
        if weighted and bdf._col_included('w'):
            w = bdf.loc[:, 'w'].to_numpy()
        else:
            w = None
        res_baseline = weighted_quantile(values=y, quantiles=quantiles, sample_weight=w)
        mean_baseline = weighted_mean(y, w)
        if reallocation_scaling_col is not None:
            scaling_col = to_list(bdf.col_reference_dict[reallocation_scaling_col])[0]
            scale = bdf.loc[:, scaling_col].to_numpy()
            y_scaled = scale * y
            res_scaled_baseline = weighted_quantile(values=y_scaled, quantiles=quantiles, sample_weight=w)
            mean_scaled_baseline = weighted_mean(y_scaled, w)
        if w is not None:
            # For sorting variables, use weighted y
            y = w * y
            if reallocation_scaling_col is not None:
                y_scaled = scale * y
        for col_cat in categorical_sort_cols.keys():
            ## Categorical sorting variables ##
            col = bdf.loc[:, col_cat].to_numpy()
            # Use categories as bins
            res_cat_baseline[col_cat] =\
                np.bincount(col, weights=y) / np.bincount(col, weights=w)
            if reallocation_scaling_col is not None:
                res_scaled_cat_baseline[col_cat] =\
                    np.bincount(col, weights=y_scaled) / np.bincount(col, weights=w)
        for col_cts, quantiles_cts in continuous_sort_cols.items():
            ## Continuous sorting variables ##
            col = bdf.loc[:, col_cts].to_numpy()
            # Create bins based on quantiles
            col_quantiles = weighted_quantile(values=col, quantiles=quantiles_cts, sample_weight=w)
            quantile_groups = pd.cut(col, col_quantiles, include_lowest=True).codes
            res_cts_baseline[col_cts] =\
                np.bincount(quantile_groups, weights=y) / np.bincount(quantile_groups, weights=w)
            if reallocation_scaling_col is not None:
                res_scaled_cts_baseline[col_cts] =\
                    np.bincount(quantile_groups, weights=y_scaled) / np.bincount(quantile_groups, weights=w)

        ## Run bootstrap ##
        res = np.zeros([n_samples, len(quantiles)])
        mean = np.zeros(n_samples)
        res_cat = {col: np.zeros([n_samples, n_quantiles]) for col, n_quantiles in categorical_sort_cols.items()}
        res_cts = {col: np.zeros([n_samples, len(quantiles) - 1]) for col, quantiles in continuous_sort_cols.items()}
        if reallocation_scaling_col is not None:
            res_scaled = np.zeros([n_samples, len(quantiles)])
            mean_scaled = np.zeros(n_samples)
            res_scaled_cat = {col: np.zeros([n_samples, n_quantiles]) for col, n_quantiles in categorical_sort_cols.items()}
            res_scaled_cts = {col: np.zeros([n_samples, len(quantiles) - 1]) for col, quantiles in continuous_sort_cols.items()}
        pk1_res = np.zeros([n_samples, nk * nk, nl])
        pk0_res = np.zeros([n_samples, nk, nl])
        NNm_res = np.zeros([n_samples, nk, nk], dtype=int)
        NNs_res = np.zeros([n_samples, nk], dtype=int)
        for i in trange(n_samples):
            ## Simulate worker types and wages ##
            bdf = _simulate_types_wages(model, jdata, sdata, gj=gj, gs=gs, pk1=pk1, pk0=pk0, qi_j=qi_j, qi_s=qi_s, qi_cum_j=qi_cum_j, qi_cum_s=qi_cum_s, optimal_reallocation=optimal_reallocation, reallocation_constraint_category=reallocation_constraint_category, reallocation_scaling_col=reallocation_scaling_col, worker_types_as_ids=False, simulate_wages=True, return_long_df=True, store_worker_types=True, weighted=weighted, rng=rng)

            ## Compute quantiles ##
            y = bdf.loc[:, 'y'].to_numpy()
            if unresidualize_col is not None:
                y += bdf.loc[:, unresidualize_col]
            if weighted and bdf._col_included('w'):
                w = bdf.loc[:, 'w'].to_numpy()
            else:
                w = None
            res[i, :] = weighted_quantile(values=y, quantiles=quantiles, sample_weight=w)
            mean[i] = weighted_mean(y, w)
            if reallocation_scaling_col is not None:
                scale = bdf.loc[:, scaling_col].to_numpy()
                y_scaled = scale * y
                res_scaled[i, :] = weighted_quantile(values=y_scaled, quantiles=quantiles, sample_weight=w)
                mean_scaled[i] = weighted_mean(y_scaled, w)
            if w is not None:
                # For sorting variables, use weighted y
                y = w * y
                if reallocation_scaling_col is not None:
                    y_scaled = scale * y
            for col_cat in categorical_sort_cols.keys():
                ## Categorical sorting variables ##
                col = bdf.loc[:, col_cat].to_numpy()
                # Use categories as bins
                res_cat[col_cat][i, :] =\
                    np.bincount(col, weights=y) / np.bincount(col, weights=w)
                if reallocation_scaling_col is not None:
                    res_scaled_cat[col_cat][i, :] =\
                        np.bincount(col, weights=y_scaled) / np.bincount(col, weights=w)
            for col_cts, quantiles_cts in continuous_sort_cols.items():
                ## Continuous sorting variables ##
                col = bdf.loc[:, col_cts].to_numpy()
                # Create bins based on quantiles
                col_quantiles = weighted_quantile(values=col, quantiles=quantiles_cts, sample_weight=w)
                quantile_groups = pd.cut(col, col_quantiles, include_lowest=True).codes
                res_cts[col_cts][i, :] =\
                    np.bincount(quantile_groups, weights=y) / np.bincount(quantile_groups, weights=w)
                if reallocation_scaling_col is not None:
                    res_scaled_cts[col_cts][i, :] =\
                        np.bincount(quantile_groups, weights=y_scaled) / np.bincount(quantile_groups, weights=w)

            ## Compute type proportions ##
            bdf = bdf.to_eventstudy(is_sorted=True, copy=False)
            # NOTE: unweighted
            m = (bdf.loc[:, 'm'] > 0)

            if not optimal_reallocation:
                # Compute pk1
                movers = bdf.loc[m, ['g1', 'g2', 'l']].to_numpy()
                pk1_i = np.bincount(movers[:, 2] + nl * movers[:, 1] + nl * nk * movers[:, 0], minlength=nk * nk * nl).reshape((nk * nk, nl))
                # Normalize rows to sum to 1
                pk1_i = DxM(1 / np.sum(pk1_i, axis=1), pk1_i)
                # Store
                pk1_res[i, :, :] = pk1_i
            # Compute pk0
            stayers = bdf.loc[~m, ['g1', 'l']].to_numpy()
            pk0_i = np.bincount(stayers[:, 1] + nl * stayers[:, 0], minlength=nk * nl).reshape((nk, nl))
            # Normalize rows to sum to 1
            pk0_i = DxM(1 / np.sum(pk0_i, axis=1), pk0_i)
            # Store
            pk0_res[i, :, :] = pk0_i

            ## Compute firm-level worker counts ##
            if not optimal_reallocation:
                NNm_res[i, :, :] = np.bincount(movers[:, 1] + nk * movers[:, 0], minlength=nk * nk).reshape((nk, nk))
                del movers
            NNs_res[i, :] = np.bincount(stayers[:, 0], minlength=nk)
            del stayers

        with bpd.util.ChainedAssignment():
            # Restore original wages, firm ids, and firm types
            jdata.loc[:, ['y1', 'y2']] = yj
            sdata.loc[:, ['y1', 'y2']] = ys
            jdata.loc[:, ['j1', 'j2']] = jj
            sdata.loc[:, 'j1'], sdata.loc[:, 'j2'] = (js, js)
            jdata.loc[:, ['g1', 'g2']] = gj
            sdata.loc[:, 'g1'], sdata.loc[:, 'g2'] = (gs, gs)

        # Drop time column
        if tj:
            jdata = jdata.drop('t', axis=1, inplace=True, allow_optional=True)
        if ts:
            sdata = sdata.drop('t', axis=1, inplace=True, allow_optional=True)

        # Store results
        self.res = {
            'baseline': {
                'outcome': res_baseline,
                'mean': mean_baseline,
                'cat': res_cat_baseline,
                'cts': res_cts_baseline
                },
            'reallocation': {
                'outcome': res,
                'mean': mean,
                'cat': res_cat,
                'cts': res_cts,
                'pk1': pk1_res,
                'pk0': pk0_res,
                'NNm': NNm_res,
                'NNs': NNs_res
                }
        }
        if reallocation_scaling_col is not None:
            self.res['baseline']['outcome_scaled'] = res_scaled_baseline
            self.res['baseline']['mean_scaled'] = mean_scaled_baseline
            self.res['baseline']['cat_scaled'] = res_scaled_cat_baseline
            self.res['baseline']['cts_scaled'] = res_scaled_cts_baseline
            self.res['reallocation']['outcome_scaled'] = res_scaled
            self.res['reallocation']['mean_scaled'] = mean_scaled
            self.res['reallocation']['cat_scaled'] = res_scaled_cat
            self.res['reallocation']['cts_scaled'] = res_scaled_cts

    def plot_type_proportions(self, period='first', subset='stayers', xlabel='firm class k', ylabel='type proportions', title='Proportions of worker types', dpi=None):
        '''
        Plot proportions of worker types at each firm class.

        Arguments:
            period (str): 'first' plots type proportions in the first period; 'second' plots type proportions in the second period; 'all' plots the average over type proportions in the first and second periods
            subset (str): 'all' plots a weighted average over movers and stayers; 'movers' plots movers; 'stayers' plots stayers (for optimal reallocation, all observations are stayers, so only 'stayers' will run)
            xlabel (str): label for x-axis
            ylabel (str): label for y-axis
            title (str): plot title
            dpi (float or None): dpi for plot
        '''
        if self.res is None:
            warnings.warn('Estimation has not yet been run.')
        else:
            model = self.model
            nl, nk = model.nl, model.nk
            pk1_res = self.res['reallocation']['pk1']
            pk0_res = self.res['reallocation']['pk0']
            NNm_res = self.res['reallocation']['NNm']
            NNs_res = self.res['reallocation']['NNs']

            ## Compute average type proportions ##
            pk_mean = np.zeros((nk, nl))
            for i in range(pk1_res.shape[0]):
                # Sort by firm effects
                A1, A2, pk1, pk0, NNm, NNs = model._sort_parameters(model.A1, model.A2, pk1=pk1_res[i, :, :], pk0=pk0_res[i, :, :], NNm=NNm_res[i, :, :], NNs=NNs_res[i, :], sort_firm_types=True)

                ## Extract subset(s) ##
                if subset == 'movers':
                    NNm_1 = np.sum(NNm, axis=1)
                    NNm_2 = np.sum(NNm, axis=0)
                    reshaped_pk1 = np.reshape(pk1, (nk, nk, nl))
                    pk_period1 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=1).T / NNm_1).T
                    pk_period2 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=0).T / NNm_2).T
                elif subset == 'stayers':
                    pk_period1 = pk0
                    pk_period2 = pk0
                elif subset == 'all':
                    NNm_1 = np.sum(NNm, axis=1)
                    NNm_2 = np.sum(NNm, axis=0)
                    # First, pk1 #
                    reshaped_pk1 = np.reshape(pk1, (nk, nk, nl))
                    pk1_period1 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=1).T / NNm_1).T
                    pk1_period2 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=0).T / NNm_2).T
                    # Second, take weighted average over pk1 and pk0 #
                    pk_period1 = ((NNm_1 * pk1_period1.T + NNs * pk0.T) / (NNm_1 + NNs)).T
                    pk_period2 = ((NNm_2 * pk1_period2.T + NNs * pk0.T) / (NNm_2 + NNs)).T
                else:
                    raise ValueError(f"`subset` must be one of 'all', 'movers' or 'stayers', but input specifies {subset!r}.")

                ## Consider correct period(s) ##
                if period == 'first':
                    pk_mean += pk_period1
                elif period == 'second':
                    pk_mean += pk_period2
                elif period == 'all':
                    pk_mean += (pk_period1 + pk_period2) / 2
                else:
                    raise ValueError(f"`period` must be one of 'first', 'second' or 'all', but input specifies {period!r}.")

            ## Take mean over all models ##
            pk_mean /= pk1_res.shape[0]

            ## Compute cumulative sum ##
            pk_cumsum = np.cumsum(pk_mean, axis=1)

            ## Plot ##
            fig, ax = plt.subplots(dpi=dpi)
            x_axis = np.arange(1, nk + 1).astype(str)
            ax.bar(x_axis, pk_mean[:, 0])
            for l in range(1, nl):
                ax.bar(x_axis, pk_mean[:, l], bottom=pk_cumsum[:, l - 1])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            plt.show()
