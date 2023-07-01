'''
Class for simulating bipartite static BLM networks.
'''
import numpy as np
from pandas import DataFrame
from paramsdict import ParamsDict, ParamsDictBase
from paramsdict.util import col_type
from bipartitepandas import BipartiteDataFrame
from bipartitepandas.util import to_list, _sort_cols

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
sim_blm_params = ParamsDict({
    'nl': (6, 'type_constrained', (int, _gteq1),
        '''
            (default=6) Number of worker types.
        ''', '>= 1'),
    'nk': (10, 'type_constrained', (int, _gteq1),
        '''
            (default=10) Number of firm types.
        ''', '>= 1'),
    'firm_size': (10, 'type_constrained', ((float, int), _gt0),
        '''
            (default=10) Average number of stayers per firm.
        ''', '> 0'),
    'categorical_controls': (None, 'dict_of_type_none', ParamsDictBase,
        '''
            (default=None) Dictionary linking column names to instances of tw.sim_categorical_control_params(). Each instance specifies a new categorical control variable. Run tw.sim_categorical_control_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
        ''', None),
    'continuous_controls': (None, 'dict_of_type_none', ParamsDictBase,
        '''
            (default=None) Dictionary linking column names to instances of tw.sim_continuous_control_params(). Each instance specifies a new continuous control variable. Run tw.sim_continuous_control_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
        ''', None),
    'NNm': (None, 'array_of_type_constrained_none', ('int', _min_gt0),
        '''
            (default=None) Matrix giving the number of movers who transition between each combination of firm types (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3); if None, set to 10 for each combination of firm types.
        ''', 'min >= 1'),
    'NNs': (None, 'array_of_type_constrained_none', ('int', _min_gt0),
        '''
            (default=None) Vector giving the number of stayers at each firm type (e.g. entry (1) gives the number of stayers at firm type 1); if None, set to 10 for each firm type.
        ''', 'min >= 1'),
    'mmult': (1, 'type_constrained', (int, _gteq1),
        '''
            (default=1) Factor by which to increase observations for movers (mmult * NNm).
        ''', '>= 1'),
    'smult': (1, 'type_constrained', (int, _gteq1),
        '''
            (default=1) Factor by which to increase observations for stayers (smult * NNs).
        ''', '>= 1'),
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
            (default=None) Dirichlet prior for pk1 (probability of being at each combination of firm types for movers). Must have length nl. None is equivalent to np.ones(nl).
        ''', 'min > 0'),
    'pk0_prior': (None, 'array_of_type_constrained_none', (('float', 'int'), _min_gt0),
        '''
            (default=None) Dirichlet prior for pk0 (probability of being at each firm type for stayers). Must have length nl. None is equivalent to np.ones(nl).
        ''', 'min > 0'),
    'strictly_monotone_A': (False, 'type', bool,
        '''
            (default=False) If True, set A1 and A2 to be strictly increasing by firm type for each worker type (otherwise, they are required to be increasing only by firm type over the average for all worker types).
        ''', None),
    'stationary_A': (False, 'type', bool,
        '''
            (default=False) If True, set A1 = A2.
        ''', None),
    'stationary_S': (False, 'type', bool,
        '''
            (default=False) If True, set S1 = S2.
        ''', None),
    'linear_additive': (False, 'type', bool,
        '''
            (default=False) If True, make A1 and A2 linearly additive.
        ''', None),
    'stationary_firm_type_variation': (False, 'type', bool,
        '''
            (default=False) If True, set constraints for A1 and A2 so that the firm type induced variation of worker-firm pair effects is the same in all periods. In particular, this sets A2 = np.mean(A2, axis=1) + A1 - np.mean(A1, axis=1).
        ''', None)
})

sim_categorical_control_params = ParamsDict({
    'n': (6, 'type_constrained', (int, _gteq2),
        '''
            (default=6) Number of types for the parameter.
        ''', '>= 2'),
    'a1_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A1_cat (mean of fixed effects in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A1_cat (mean of fixed effects in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A2_cat (mean of fixed effects in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2_cat (mean of fixed effects in second period).
        ''', '>= 0'),
    's1_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S1_cat (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's1_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S1_cat (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's2_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2_cat (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    's2_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S2_cat (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    'stationary_A': (False, 'type', bool,
        '''
            (default=False) If True, A1_cat and A2_cat must be equal.
        ''', None),
    'stationary_S': (False, 'type', bool,
        '''
            (default=False) If True, S1_cat and S2_cat must be equal.
        ''', None),
    'stationary_firm_type_variation': (False, 'type', bool,
        '''
            (default=False) If True, set constraints for A1_cat and A2_cat so that the firm type induced variation of worker-firm pair effects is the same in all periods. In particular, this sets A2_cat = np.mean(A2_cat, axis=1) + A1_cat - np.mean(A1_cat, axis=1).
        ''', None),
    'worker_type_interaction': (False, 'type', bool,
        '''
            (default=False) If True, effect can differ by worker type.
        ''', None)
})

sim_continuous_control_params = ParamsDict({
    'a1_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A1_cts (mean of coefficient in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A1_cts (mean of coefficient in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A2_cts (mean of coefficient in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2_cts (mean of coefficient in second period).
        ''', '>= 0'),
    's1_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S1_cts (standard deviation of coefficient in first period).
        ''', '>= 0'),
    's1_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S1_cts (standard deviation of coefficient in first period).
        ''', '>= 0'),
    's2_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2_cts (standard deviation of coefficient in second period).
        ''', '>= 0'),
    's2_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S2_cts (standard deviation of coefficient in second period).
        ''', '>= 0'),
    'stationary_A': (False, 'type', bool,
        '''
            (default=False) If True, A1_cts and A2_cts must be equal.
        ''', None),
    'stationary_S': (False, 'type', bool,
        '''
            (default=False) If True, S1_cts and S2_cts must be equal.
        ''', None),
    'stationary_firm_type_variation': (False, 'type', bool,
        '''
            (default=False) If True, set constraints for A1_cts and A2_cts so that the firm type induced variation of worker-firm pair effects is the same in all periods. In particular, this sets A2_cts = np.mean(A2_cts) + A1_cts - np.mean(A1_cts).
        ''', None),
    'worker_type_interaction': (False, 'type', bool,
        '''
            (default=False) If True, effect can differ by worker type.
        ''', None)
})

def _simulate_firm_classes_movers(nk, NNm):
    '''
    Simulate firm classes for movers.

    Arguments:
        nk (int): number of firm types
        NNm (NumPy Array): matrix giving the number of movers who transition between each combination of firm types (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3); if None, set to 10 for each combination of firm types

    Returns:
        (tuple of NumPy Arrays): firm classes for movers in the first and second period
    '''
    G1 = np.repeat(np.arange(nk), NNm.sum(axis=1))
    G2 = np.repeat(np.tile(np.arange(nk), nk), NNm.flatten())
    return (G1, G2)

def _simulate_firm_classes_stayers(nk, NNs):
    '''
    Simulate firm classes for stayers.

    Arguments:
        nk (int): number of firm types
        NNs (NumPy Array): vector giving the number of stayers at each firm type (e.g. entry (1) gives the number of stayers at firm type 1); if None, set to 10 for each firm type

    Returns:
        (NumPy Array): firm classes for stayers
    '''
    return np.repeat(np.arange(nk), NNs)

def _simulate_worker_types_movers(nl, nk, NNm=None, G1=None, G2=None, pk1=None, qi=None, qi_cum=None, simulating_data=False, rng=None):
    '''
    Simulate worker types for movers.

    Arguments:
        nl (int): number of worker types
        nk (int): number of firm types
        NNm (NumPy Array or None): matrix giving the number of movers who transition between each combination of firm types (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3); None if using real data
        G1 (NumPy Array or None): first period firm classes for movers; None if using simulated data
        G2 (NumPy Array or None): second period firm classes for movers; None if using simulated data
        pk1 (NumPy Array or None): (use to assign workers to worker types probabilistically based on dgp-level probabilities) probability of being at each combination of firm types for movers; None if qi or qi_cum is not None
        qi (NumPy Array or None): (use to assign workers to maximum probability worker type based on observation-level probabilities) probabilities for each stayer observation to be each worker type; None if pk1 or qi_cum is not None
        qi_cum (NumPy Array or None): (use to assign workers to worker types probabilistically based on observation-level probabilities) cumulative probabilities for each mover observation to be each worker type; None if pk1 or qi is not None
        simulating_data (bool): set to True if simulating data from SimBLM class
        rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

    Returns:
        (NumPy Array): worker types for movers
    '''
    if ((pk1 is None) + (qi is None) + (qi_cum is None)) != 2:
        raise ValueError('Only one of `pk1`, `qi`, and `qi_cum` can be used as input.')

    if qi is not None:
        ## Assign workers to max worker type ##
        return np.argmax(qi, axis=1)

    if rng is None:
        rng = np.random.default_rng(None)

    # Worker types
    if G1 is None:
        nmi = np.sum(NNm)
    else:
        nmi = len(G1)
    L = np.zeros(shape=nmi, dtype=int)

    if qi_cum is not None:
        # Source: https://stackoverflow.com/a/73082038/17333120
        worker_types_sim = rng.uniform(size=nmi)
        for l in range(1, nl):
            L[(qi_cum[:, l - 1] < worker_types_sim) & (worker_types_sim <= qi_cum[:, l])] = l
        return L

    worker_types = np.arange(nl)
    i = 0
    for k1 in range(nk):
        for k2 in range(nk):
            # Iterate over all firm type combinations a worker can transition between
            if simulating_data:
                ni = NNm[k1, k2]
                rows_kk = np.arange(i, i + ni)
                i += ni
            else:
                rows_kk = np.where((G1 == k1) & (G2 == k2))[0]
                ni = len(rows_kk)

            # Draw worker types
            L[rows_kk] = rng.choice(worker_types, size=ni, replace=True, p=pk1[k1 + nk * k2, :])

    return L

def _simulate_worker_types_stayers(nl, nk, NNs=None, G=None, pk0=None, qi=None, qi_cum=None, simulating_data=False, rng=None):
    '''
    Simulate worker types for stayers.

    Arguments:
        nl (int): number of worker types
        nk (int): number of firm types
        NNs (NumPy Array): vector giving the number of stayers at each firm type (e.g. entry (1) gives the number of stayers at firm type 1); None if using real data
        G (NumPy Array or None): firm classes for stayers; None if using simulated data
        pk0 (NumPy Array or None): (use to assign workers to worker types probabilistically based on dgp-level probabilities) probability of being at each firm type for stayers; None if qi or qi_cum is not None
        qi (NumPy Array or None): (use to assign workers to maximum probability worker type based on observation-level probabilities) probabilities for each stayer observation to be each worker type; None if pk0 or qi_cum is not None
        qi_cum (NumPy Array or None): (use to assign workers to worker types probabilistically based on observation-level probabilities) cumulative probabilities for each stayer observation to be each worker type; None if pk0 or qi is not None
        simulating_data (bool): set to True if simulating data from SimBLM class
        rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

    Returns:
        (NumPy Array): worker types for stayers
    '''
    if ((pk0 is None) + (qi is None) + (qi_cum is None)) != 2:
        raise ValueError('Only one of `pk0`, `qi`, and `qi_cum` can be used as input.')

    if qi is not None:
        ## Assign workers to max worker type ##
        return np.argmax(qi, axis=1)

    if rng is None:
        rng = np.random.default_rng(None)

    # Worker types
    if simulating_data:
        nsi = np.sum(NNs)
    else:
        nsi = len(G)
    L = np.zeros(shape=nsi, dtype=int)

    if qi_cum is not None:
        # Source: https://stackoverflow.com/a/73082038/17333120
        worker_types_sim = rng.uniform(size=nsi)
        for l in range(1, nl):
            L[(qi_cum[:, l - 1] < worker_types_sim) & (worker_types_sim <= qi_cum[:, l])] = l
        return L

    worker_types = np.arange(nl)
    i = 0
    for k in range(nk):
        # Iterate over firm types
        if simulating_data:
            ni = NNs[k]
            rows_k = np.arange(i, i + ni)
            i += ni
        else:
            rows_k = np.where(G == k)[0]
            ni = len(rows_k)

        # Draw worker types
        L[rows_k] = rng.choice(worker_types, size=ni, replace=True, p=pk0[k, :])

    return L

def _simulate_controls_movers(nmi, cat_dict=None, cts_cols=None, dynamic=False, rng=None):
    '''
    Simulate controls for movers.

    Arguments:
        nmi (int): number of mover observations
        cat_dict (dict or None): dictionary linking categorical controls to categorical parameter dictionaries; None is equivalent to {}
        cts_cols (list or None): list of continuous controls; None is equivalent to []
        dynamic (bool): if False, simulating static BLM; if True, simulating dynamic BLM
        rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

    Returns:
        (tuple of NumPy Arrays): (categorical for period 1; categorical for period 2; continuous for period 1; continuous for period 2)
    '''
    if rng is None:
        rng = np.random.default_rng(None)
    if cat_dict is None:
        cat_dict = {}
    if cts_cols is None:
        cts_cols = []

    ## Categorical ##
    # FIXME add custom probabilities?
    A1_cat_draws = {col: rng.choice(np.arange(col_dict['n']), size=nmi, replace=True) for col, col_dict in cat_dict.items()}
    A2_cat_draws = {col: rng.choice(np.arange(col_dict['n']), size=nmi, replace=True) for col, col_dict in cat_dict.items()}

    ## Continuous ##
    A1_cts_draws = {col: rng.normal(size=nmi) for col in cts_cols}
    A2_cts_draws = {col: rng.normal(size=nmi) for col in cts_cols}

    if dynamic:
        ## Update labels ##
        A1_cat_draws = {k + '1': v for k, v in A1_cat_draws.items()}
        A3_cat_draws = {k + '3': v for k, v in A2_cat_draws.items()}
        A2_cat_draws = {k[: -1] + '2': v for k, v in A1_cat_draws.items()}
        A4_cat_draws = {k[: -1] + '4': v for k, v in A3_cat_draws.items()}
        A1_cts_draws = {k + '1': v for k, v in A1_cts_draws.items()}
        A3_cts_draws = {k + '3': v for k, v in A2_cts_draws.items()}
        A2_cts_draws = {k[: -1] + '2': v for k, v in A1_cts_draws.items()}
        A4_cts_draws = {k[: -1] + '4': v for k, v in A3_cts_draws.items()}

        return (A1_cat_draws, A2_cat_draws, A3_cat_draws, A4_cat_draws, A1_cts_draws, A2_cts_draws, A3_cts_draws, A4_cts_draws)
    else:
        ## Update labels ##
        A1_cat_draws = {k + '1': v for k, v in A1_cat_draws.items()}
        A2_cat_draws = {k + '2': v for k, v in A2_cat_draws.items()}
        A1_cts_draws = {k + '1': v for k, v in A1_cts_draws.items()}
        A2_cts_draws = {k + '2': v for k, v in A2_cts_draws.items()}

        return (A1_cat_draws, A2_cat_draws, A1_cts_draws, A2_cts_draws)

def _simulate_controls_stayers(nsi, cat_dict=None, cts_cols=None, dynamic=False, rng=None):
    '''
    Simulate controls for stayers.

    Arguments:
        nsi (int): number of stayers observations
        cat_dict (dict or None): dictionary linking categorical controls to categorical parameter dictionaries; None is equivalent to {}
        cts_cols (list or None): list of continuous controls; None is equivalent to []
        dynamic (bool): if False, simulating static BLM; if True, simulating dynamic BLM
        rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

    Returns:
        (tuple of NumPy Arrays): (categorical for period 1; categorical for period 2; continuous for period 1; continuous for period 2)
    '''
    if rng is None:
        rng = np.random.default_rng(None)
    if cat_dict is None:
        cat_dict = {}
    if cts_cols is None:
        cts_cols = []

    ## Categorical ##
    # FIXME add custom probabilities?
    A1_cat_draws = {col: rng.choice(np.arange(col_dict['n']), size=nsi, replace=True) for col, col_dict in cat_dict.items()}

    ## Continuous ##
    A1_cts_draws = {col: rng.normal(size=nsi) for col in cts_cols}

    if dynamic:
        ## Draw period 2 ##
        A2_cat_draws = {col: rng.choice(np.arange(col_dict['n']), size=nsi, replace=True) for col, col_dict in cat_dict.items()}
        A2_cts_draws = {col: rng.normal(size=nsi) for col in cts_cols}
        ## Update labels ##
        A1_cat_draws = {k + '1': v for k, v in A1_cat_draws.items()}
        A3_cat_draws = {k + '3': v for k, v in A2_cat_draws.items()}
        A2_cat_draws = {k[: -1] + '2': v for k, v in A1_cat_draws.items()}
        A4_cat_draws = {k[: -1] + '4': v for k, v in A3_cat_draws.items()}
        A1_cts_draws = {k + '1': v for k, v in A1_cts_draws.items()}
        A3_cts_draws = {k + '3': v for k, v in A2_cts_draws.items()}
        A2_cts_draws = {k[: -1] + '2': v for k, v in A1_cts_draws.items()}
        A4_cts_draws = {k[: -1] + '4': v for k, v in A3_cts_draws.items()}

        return (A1_cat_draws, A2_cat_draws, A3_cat_draws, A4_cat_draws, A1_cts_draws, A2_cts_draws, A3_cts_draws, A4_cts_draws)
    else:
        ## Update labels ##
        A1_cat_draws = {k + '1': v for k, v in A1_cat_draws.items()}
        A2_cat_draws = {k[: -1] + '2': v.copy() for k, v in A1_cat_draws.items()}
        A1_cts_draws = {k + '1': v for k, v in A1_cts_draws.items()}
        A2_cts_draws = {k[: -1] + '2': v.copy() for k, v in A1_cts_draws.items()}

        return (A1_cat_draws, A2_cat_draws, A1_cts_draws, A2_cts_draws)

def _simulate_firms(jdata, sdata, firm_size, dynamic=False, rng=None):
    '''
    Simulate firms in-place.

    Arguments:
        jdata (Pandas DataFrame): data for movers
        sdata (Pandas DataFrame): data for stayers
        firm_size (float): average number of stayers per firm
        dynamic (bool): if False, simulating static BLM; if True, simulating dynamic BLM
        rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)
    '''
    if rng is None:
        rng = np.random.default_rng(None)

    if dynamic:
        j2 = 'j4'
        g2 = 'g4'
    else:
        j2 = 'j2'
        g2 = 'g2'

    ## Stayers ##
    # Draw firm ids for stayers (note each cluster must link to at least 2 firm ids so that movers are able to move firms within the cluster)
    sdata.loc[:, 'j1'] = np.hstack(sdata.groupby('g1').apply(lambda df: rng.integers(max(2, round(len(df) / firm_size)), size=len(df))))

    # Make firm ids contiguous
    sdata.loc[:, 'j1'] = sdata.groupby(['g1', 'j1']).ngroup()
    sdata.loc[:, 'j2'] = sdata.loc[:, 'j1'].copy()
    if dynamic:
        sdata.loc[:, 'j3'] = sdata.loc[:, 'j1'].copy()
        sdata.loc[:, 'j4'] = sdata.loc[:, 'j1'].copy()

    # Link firm ids to clusters
    j_per_g_dict = sdata.groupby('g1')['j1'].unique().to_dict()
    for g, linked_j in j_per_g_dict.items():
        # Make sure each cluster has at least 2 linked firm ids
        if len(linked_j) == 1:
            raise ValueError(f"Cluster {g} has only 1 linked firm. However, each cluster must link to at least 2 firm ids so that movers are able to move firms within the cluster. Please alter parameter values to ensure enough firms can be generated.")

    ## Movers ##
    # Draw firm ids for movers
    jdata.loc[:, 'j1'] = np.hstack(jdata.groupby('g1').apply(lambda df: rng.choice(j_per_g_dict[df.iloc[0]['g1']], size=len(df))))
    groupby_g2 = jdata.groupby(g2)
    jdata.loc[:, j2] = np.hstack(groupby_g2.apply(lambda df: rng.choice(j_per_g_dict[df.iloc[0][g2]], size=len(df))))

    # Make sure movers actually move
    # FIXME find a deterministic way to do this
    same_firm_mask = (jdata.loc[:, 'j1'].to_numpy() == jdata.loc[:, j2].to_numpy())
    while same_firm_mask.any():
        same_firm_rows = jdata.loc[same_firm_mask, :].index
        jdata.loc[same_firm_rows, j2] = np.hstack(groupby_g2.apply(lambda df: rng.choice(j_per_g_dict[df.iloc[0][g2]], size=len(df))))[same_firm_rows]
        same_firm_mask = (jdata.loc[:, 'j1'].to_numpy() == jdata.loc[:, j2].to_numpy())

    if dynamic:
        # Set 'j2' and 'j3'
        jdata.loc[:, 'j2'] = jdata.loc[:, 'j1'].copy()
        jdata.loc[:, 'j3'] = jdata.loc[:, 'j4'].copy()

def _simulate_wages_movers(jdata, L, blm_model=None, A1=None, A2=None, S1=None, S2=None, A1_cat=None, A2_cat=None, S1_cat=None, S2_cat=None, A1_cts=None, A2_cts=None, S1_cts=None, S2_cts=None, controls_dict=None, cat_cols=None, cts_cols=None, G1=None, G2=None, w1=None, w2=None, rng=None, **kwargs):
    '''
    Simulate wages for movers.

    Arguments:
        jdata (BipartitePandas DataFrame): movers data
        L (NumPy Array): worker types for movers
        blm_model (BLMModel or None): BLM model with estimated parameter values; None if other parameters are not None
        A1 (NumPy Array or None): mean of fixed effects in the first period; None if blm_model is not None
        A2 (NumPy Array or None): mean of fixed effects in the second period; None if blm_model is not None
        S1 (NumPy Array or None): standard deviation of fixed effects in the first period; None if blm_model is not None
        S2 (NumPy Array or None): standard deviation of fixed effects in the second period; None if blm_model is not None
        A1_cat (dict of NumPy Arrays or None): dictionary of column names linked to mean of fixed effects in the first period for categorical control variables; None if no categorical controls
        A2_cat (dict of NumPy Arrays or None): dictionary of column names linked to mean of fixed effects in the second period for categorical control variables; None if no categorical controls
        S1_cat (dict of NumPy Arrays or None): dictionary of column names linked to standard deviation of fixed effects in the first period for categorical control variables; None if no categorical controls
        S2_cat (dict of NumPy Arrays or None): dictionary of column names linked to standard deviation of fixed effects in the second period for categorical control variables; None if no categorical controls
        A1_cts (dict of NumPy Arrays or None): dictionary of column names linked to mean of coefficients in the first period for continuous control variables; None if no continuous controls
        A2_cts (dict of NumPy Arrays or None): dictionary of column names linked to mean of coefficients in the second period for continuous control variables; None if no continuous controls
        S1_cts (dict of NumPy Arrays or None): dictionary of column names linked to standard deviation of coefficients in the first period for continuous control variables; None if no continuous controls
        S2_cts (dict of NumPy Arrays or None): dictionary of column names linked to standard deviation of coefficients in the second period for continuous control variables; None if no continuous controls
        controls_dict (dict or None): dictionary of all control variables; None if no controls
        cat_cols (list or None): list of categorical controls; None if no categorical controls
        cts_cols (list or None): list of continuous controls; None if no continuous controls
        G1 (NumPy Array or None): firm classes for movers in the first period; if None, extract from jdata
        G2 (NumPy Array or None): firm classes for movers in the second period; if None, extract from jdata
        w1 (NumPy Array or None): mover weights for the first period; if None, don't weight
        w2 (NumPy Array or None): mover weights for the second period; if None, don't weight
        rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        kwargs (dict): keyword arguments

    Returns:
        (tuple of NumPy Arrays): wages for movers in the first and second periods
    '''
    if rng is None:
        rng = np.random.default_rng(None)
    if w1 is None:
        w1 = 1
    if w2 is None:
        w2 = 1

    # Unpack values
    nmi = len(L)
    if G1 is None:
        G1 = jdata.loc[:, 'g1'].to_numpy()
    if G2 is None:
        G2 = jdata.loc[:, 'g2'].to_numpy()

    if blm_model is not None:
        # Unpack BLMModel
        A1 = blm_model.A1
        A2 = blm_model.A2
        S1 = blm_model.S1
        S2 = blm_model.S2
        A1_cat = blm_model.A1_cat
        A2_cat = blm_model.A2_cat
        S1_cat = blm_model.S1_cat
        S2_cat = blm_model.S2_cat
        A1_cts = blm_model.A1_cts
        A2_cts = blm_model.A2_cts
        S1_cts = blm_model.S1_cts
        S2_cts = blm_model.S2_cts
        controls_dict = blm_model.controls_dict
        cat_cols = blm_model.cat_cols
        cts_cols = blm_model.cts_cols

    ## Draw wages ##
    if (controls_dict is not None) and (len(controls_dict) > 0):
        #### Simulate control variable wages ####
        A1_sum = A1[L, G1]
        A2_sum = A2[L, G2]
        S1_sum_sq = (S1 ** 2)[L, G1]
        S2_sum_sq = (S2 ** 2)[L, G2]
        if cat_cols is None:
            cat_cols = []
        if cts_cols is None:
            cts_cols = []
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
                ### Categorical ###
                if controls_dict[col]['worker_type_interaction']:
                    ## Worker-interaction ##
                    A1_sum += A1_cat[col][L, jdata.loc[:, subcol_1]]
                    A2_sum += A2_cat[col][L, jdata.loc[:, subcol_2]]
                    S1_sum_sq += (S1_cat[col] ** 2)[L, jdata.loc[:, subcol_1]]
                    S2_sum_sq += (S2_cat[col] ** 2)[L, jdata.loc[:, subcol_2]]
                else:
                    ## Non-worker-interaction ##
                    A1_sum += A1_cat[col][jdata.loc[:, subcol_1]]
                    A2_sum += A2_cat[col][jdata.loc[:, subcol_2]]
                    S1_sum_sq += (S1_cat[col] ** 2)[jdata.loc[:, subcol_1]]
                    S2_sum_sq += (S2_cat[col] ** 2)[jdata.loc[:, subcol_2]]
            else:
                ### Continuous ###
                if controls_dict[col]['worker_type_interaction']:
                    ## Worker-interaction ##
                    A1_sum += A1_cts[col][L] * jdata.loc[:, subcol_1]
                    A2_sum += A2_cts[col][L] * jdata.loc[:, subcol_2]
                    S1_sum_sq += (S1_cts[col] ** 2)[L]
                    S2_sum_sq += (S2_cts[col] ** 2)[L]
                else:
                    ## Non-worker-interaction ##
                    A1_sum += A1_cts[col] * jdata.loc[:, subcol_1]
                    A2_sum += A2_cts[col] * jdata.loc[:, subcol_2]
                    S1_sum_sq += S1_cts[col] ** 2
                    S2_sum_sq += S2_cts[col] ** 2

        Y1 = rng.normal(loc=A1_sum, scale=np.sqrt(S1_sum_sq / w1), size=nmi)
        Y2 = rng.normal(loc=A2_sum, scale=np.sqrt(S2_sum_sq / w2), size=nmi)
    else:
        #### No control variables ####
        Y1 = rng.normal(loc=A1[L, G1], scale=S1[L, G1] / np.sqrt(w1), size=nmi)
        Y2 = rng.normal(loc=A2[L, G2], scale=S2[L, G2] / np.sqrt(w2), size=nmi)

    return (Y1, Y2)

def _simulate_wages_stayers(sdata, L, blm_model=None, A1=None, S1=None, A1_cat=None, S1_cat=None, A1_cts=None, S1_cts=None, controls_dict=None, cat_cols=None, cts_cols=None, G=None, w=None, rng=None, **kwargs):
    '''
    Simulate wages for stayers.

    Arguments:
        sdata (BipartitePandas DataFrame): stayers data
        L (NumPy Array): worker types for stayers
        blm_model (BLMModel): BLM model with estimated parameter values
        A1 (NumPy Array or None): mean of fixed effects in the first period; None if blm_model is not None
        S1 (NumPy Array or None): standard deviation of fixed effects in the first period; None if blm_model is not None
        A1_cat (dict of NumPy Arrays or None): dictionary of column names linked to mean of fixed effects in the first period for categorical control variables; None if no categorical controls
        S1_cat (dict of NumPy Arrays or None): dictionary of column names linked to standard deviation of fixed effects in the first period for categorical control variables; None if no categorical controls
        A1_cts (dict of NumPy Arrays or None): dictionary of column names linked to mean of coefficients in the first period for continuous control variables; None if no continuous controls
        S1_cts (dict of NumPy Arrays or None): dictionary of column names linked to standard deviation of coefficients in the first period for continuous control variables; None if no continuous controls
        controls_dict (dict or None): dictionary of all control variables; None if no controls
        cat_cols (list or None): list of categorical controls; None if no categorical controls
        cts_cols (list or None): list of continuous controls; None if no continuous controls
        G (NumPy Array or None): firm classes for stayers; if None, extract from sdata
        w (NumPy Array or None): mover weights; if None, don't weight
        rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        kwargs (dict): keyword arguments

    Returns:
        (tuple of NumPy Arrays): wages for stayers in the first and second periods
    '''
    if rng is None:
        rng = np.random.default_rng(None)
    if w is None:
        w = 1

    # Unpack values
    nsi = len(L)
    if G is None:
        G = sdata.loc[:, 'g1'].to_numpy()

    if blm_model is not None:
        # Unpack BLMModel
        A1 = blm_model.A1
        S1 = blm_model.S1
        A1_cat = blm_model.A1_cat
        S1_cat = blm_model.S1_cat
        A1_cts = blm_model.A1_cts
        S1_cts = blm_model.S1_cts
        controls_dict = blm_model.controls_dict
        cat_cols = blm_model.cat_cols
        cts_cols = blm_model.cts_cols

    ## Draw wages ##
    A1_sum = A1[L, G]
    if (controls_dict is not None) and (len(controls_dict) > 0):
        #### Simulate control variable wages ####
        S1_sum_sq = (S1 ** 2)[L, G]
        if cat_cols is None:
            cat_cols = []
        if cts_cols is None:
            cts_cols = []
        for i, col in enumerate(cat_cols + cts_cols):
            # Get subcolumns associated with col
            subcol_1 = to_list(sdata.col_reference_dict[col])[0]
            if i < len(cat_cols):
                ### Categorical ###
                if controls_dict[col]['worker_type_interaction']:
                    ## Worker-interaction ##
                    A1_sum += A1_cat[col][L, sdata.loc[:, subcol_1]]
                    S1_sum_sq += (S1_cat[col] ** 2)[L, sdata.loc[:, subcol_1]]
                else:
                    ## Non-worker-interaction ##
                    A1_sum += A1_cat[col][sdata.loc[:, subcol_1]]
                    S1_sum_sq += (S1_cat[col] ** 2)[sdata.loc[:, subcol_1]]
            else:
                ### Continuous ###
                if controls_dict[col]['worker_type_interaction']:
                    ## Worker-interaction ##
                    A1_sum += A1_cts[col][L] * sdata.loc[:, subcol_1]
                    S1_sum_sq += (S1_cts[col] ** 2)[L]
                else:
                    ## Non-worker-interaction ##
                    A1_sum += A1_cts[col] * sdata.loc[:, subcol_1]
                    S1_sum_sq += S1_cts[col] ** 2

        Y1 = rng.normal(loc=A1_sum, scale=np.sqrt(S1_sum_sq / w), size=nsi)
        Y2 = rng.normal(loc=A1_sum, scale=np.sqrt(S1_sum_sq / w), size=nsi)
    else:
        #### No control variables ####
        S1_sum = S1[L, G]
        Y1 = rng.normal(loc=A1_sum, scale=S1_sum / np.sqrt(w), size=nsi)
        Y2 = rng.normal(loc=A1_sum, scale=S1_sum / np.sqrt(w), size=nsi)

    return (Y1, Y2)

def _min_firm_type(A1, A2, primary_period='first'):
        '''
        Find the lowest firm type.

        Arguments:
            A1 (NumPy Array): mean of fixed effects in the first period
            A2 (NumPy Array): mean of fixed effects in the second period
            primary_period (str): period to normalize and sort over. 'first' uses first period parameters; 'second' uses second period parameters; 'all' uses the average over first and second period parameters.

        Returns:
            (int): lowest firm type
        '''
        # Compute parameters from primary period
        if primary_period == 'first':
            A_mean = A1
        elif primary_period == 'second':
            A_mean = A2
        elif primary_period == 'all':
            A_mean = (A1 + A2) / 2

        # Return lowest firm type
        return np.mean(A_mean, axis=0).argsort()[0]

# def _normalize(nl, A1, A2, A1_cat, A2_cat, cat_dict, primary_period='first'):
#     '''
#     Normalize means given categorical controls.

#     Arguments:
#         nl (int): number of worker types
#         A1 (NumPy Array): mean of fixed effects in the first period
#         A2 (NumPy Array): mean of fixed effects in the second period
#         A1_cat (dict of NumPy Arrays): dictionary linking column names to the mean of fixed effects in the first period for categorical control variables
#         A2_cat (dict of NumPy Arrays): dictionary linking column names to the mean of fixed effects in the second period for categorical control variables
#         cat_dict (dict or None): dictionary linking categorical controls to categorical parameter dictionaries; None is equivalent to {}
#         primary_period (str): period to normalize and sort over. 'first' uses first period parameters; 'second' uses second period parameters; 'all' uses the average over first and second period parameters.

#     Returns:
#         (tuple): tuple of normalized parameters (A1, A2, A1_cat, A2_cat)
#     '''
#     # Unpack parameters
#     A1, A2, A1_cat, A2_cat = A1.copy(), A2.copy(), A1_cat.copy(), A2_cat.copy()

#     if len(cat_dict) > 0:
#         # Compute minimum firm type
#         min_firm_type = _min_firm_type(A1, A2, primary_period)
#         # Check if any columns interact with worker type and/or are stationary (tv stands for time-varying, which is equivalent to non-stationary; and wi stands for worker-interaction)
#         any_tv_nwi = False
#         any_tnv_nwi = False
#         any_tv_wi = False
#         any_tnv_wi = False
#         for col in cat_dict.keys():
#             # Check if column is stationary
#             is_stationary = cat_dict[col]['stationary_A']

#             if cat_dict[col]['worker_type_interaction']:
#                 # If the column interacts with worker types
#                 if is_stationary:
#                     any_tnv_wi = True
#                     tnv_wi_col = col
#                 else:
#                     any_tv_wi = True
#                     tv_wi_col = col
#                     break
#             else:
#                 if is_stationary:
#                     any_tnv_nwi = True
#                     tnv_nwi_col = col
#                 else:
#                     any_tv_nwi = True
#                     tv_nwi_col = col

#         ## Normalize parameters ##
#         if any_tv_wi:
#             for l in range(nl):
#                 # First period
#                 adj_val_1 = A1[l, min_firm_type]
#                 A1[l, :] -= adj_val_1
#                 A1_cat[tv_wi_col][l, :] += adj_val_1
#                 # Second period
#                 adj_val_2 = A2[l, min_firm_type]
#                 A2[l, :] -= adj_val_2
#                 A2_cat[tv_wi_col][l, :] += adj_val_2
#         else:
#             primary_period_dict = {
#                 'first': 0,
#                 'second': 1,
#                 'all': range(2)
#             }
#             secondary_period_dict = {
#                 'first': 1,
#                 'second': 0,
#                 'all': range(2)
#             }
#             params_dict = {
#                 0: [A1, A1_cat],
#                 1: [A2, A2_cat]
#             }
#             Ap = [params_dict[pp] for pp in to_list(primary_period_dict[primary_period])]
#             As = [params_dict[sp] for sp in to_list(secondary_period_dict[primary_period])]
#             if any_tnv_wi:
#                 ## Normalize primary period ##
#                 for l in range(nl):
#                     # Compute normalization
#                     adj_val_1 = Ap[0][0][l, min_firm_type]
#                     for Ap_sub in Ap[1:]:
#                         adj_val_1 += Ap_sub[0][l, min_firm_type]
#                     adj_val_1 /= len(Ap)
#                     # Normalize
#                     A1[l, :] -= adj_val_1
#                     A1_cat[tnv_wi_col][l, :] += adj_val_1
#                     A2[l, :] -= adj_val_1
#                     A2_cat[tnv_wi_col][l, :] += adj_val_1
#                 if any_tv_nwi:
#                     ## Normalize lowest type pair from secondary period ##
#                     for As_sub in As:
#                         adj_val_2 = As_sub[0][0, min_firm_type]
#                         As_sub[0] -= adj_val_2
#                         As_sub[1][tv_nwi_col] += adj_val_2
#             else:
#                 if any_tv_nwi:
#                     ## Normalize lowest type pair in both periods ##
#                     # First period
#                     adj_val_1 = A1[0, min_firm_type]
#                     A1 -= adj_val_1
#                     A1_cat[tv_nwi_col] += adj_val_1
#                     # Second period
#                     adj_val_2 = A2[0, min_firm_type]
#                     A2 -= adj_val_2
#                     A2_cat[tv_nwi_col] += adj_val_2
#                 elif any_tnv_nwi:
#                     ## Normalize lowest type pair in primary period ##
#                     # Compute normalization
#                     adj_val_1 = Ap[0][0][0, min_firm_type]
#                     for Ap_sub in Ap[1:]:
#                         adj_val_1 += Ap_sub[0][0, min_firm_type]
#                     adj_val_1 /= len(Ap)
#                     # Normalize
#                     A1 -= adj_val_1
#                     A1_cat[tnv_nwi_col] += adj_val_1
#                     A2 -= adj_val_1
#                     A2_cat[tnv_nwi_col] += adj_val_1

#     return (A1, A2, A1_cat, A2_cat)

def _reallocate(pk1, pk0, NNm, NNs, reallocate_period='first', reallocate_jointly=True):
    '''
    Draw worker type proportions independently of firm type.

    Arguments:
        pk1 (NumPy Array): probability of being at each combination of firm types for movers
        pk0 (NumPy Array): probability of being at each firm type for stayers
        NNm (NumPy Array): matrix giving the number of movers who transition between each combination of firm types (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3); if None, set to 10 for each combination of firm types
        NNs (NumPy Array): vector giving the number of stayers at each firm type (e.g. entry (1) gives the number of stayers at firm type 1); if None, set to 10 for each firm type
        reallocate_period (str): if 'first', compute type proportions based on first period parameters; if 'second', compute type proportions based on second period parameters; if 'all', compute type proportions based on average over first and second period parameters
        reallocate_jointly (bool): if True, worker type proportions take the average over movers and stayers (i.e. all workers use the same type proportions); if False, consider movers and stayers separately

    Returns:
        (tuple of NumPy Arrays): reallocated (pk1, pk0)
    '''
    nk, nl = pk0.shape
    NNm_1 = np.sum(NNm, axis=1)
    NNm_2 = np.sum(NNm, axis=0)
    nm = np.sum(NNm)
    ns = np.sum(NNs)
    ## Extract subset(s) ##
    # First, pk1 #
    reshaped_pk1 = np.reshape(pk1, (nk, nk, nl))
    pk1_period1 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=1).T / NNm_1).T
    pk1_period2 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=0).T / NNm_2).T
    if reallocate_jointly:
        # Second, take weighted average over pk1 and pk0 #
        pk1_period1 = ((NNm_1 * pk1_period1.T + NNs * pk0.T) / (NNm_1 + NNs)).T
        pk1_period2 = ((NNm_2 * pk1_period2.T + NNs * pk0.T) / (NNm_2 + NNs)).T
        pk0_period1 = pk1_period1
        pk0_period2 = pk1_period2
    else:
        pk0_period1 = pk0
        pk0_period1 = pk0

    ## Consider correct period(s) ##
    if reallocate_period == 'first':
        pk1 = pk1_period1
        pk0 = pk0_period1
    elif reallocate_period == 'second':
        pk1 = pk1_period2
        pk0 = pk0_period2
    elif reallocate_period == 'all':
        pk1 = (pk1_period1 + pk1_period2) / 2
        pk0 = (pk0_period1 + pk0_period2) / 2
    else:
        raise ValueError(f"`reallocate_period` must be one of 'first', 'second', 'all', or None, but input specifies {reallocate_period!r}.")

    ## Compute unconditional pk1 and pk0 ##
    if reallocate_jointly:
        if reallocate_period == 'first':
            pk1 = np.sum(((NNm_1 + NNs) * pk1.T).T, axis=0) / (nm + ns)
        elif reallocate_period == 'second':
            pk1 = np.sum(((NNm_2 + NNs) * pk1.T).T, axis=0) / (nm + ns)
        elif reallocate_period == 'all':
            pk1 = np.sum((((NNm_1 + NNm_2) / 2 + NNs) * pk1.T).T, axis=0) / (nm + ns)
        pk0 = pk1
    else:
        if reallocate_period == 'first':
            pk1 = np.sum((NNm_1 * pk1.T).T, axis=0) / nm
        elif reallocate_period == 'second':
            pk1 = np.sum((NNm_2 * pk1.T).T, axis=0) / nm
        elif reallocate_period == 'all':
            pk1 = np.sum((((NNm_1 + NNm_2) / 2) * pk1.T).T, axis=0) / nm
        pk0 = np.sum((NNs * pk0.T).T, axis=0) / ns

    # Repeat unconditional mean across all firm types
    pk1 = np.tile(pk1, (nk * nk, 1))
    pk0 = np.tile(pk0, (nk, 1))

    return (pk1, pk0)

class SimBLM:
    '''
    Class for simulating bipartite BLM networks of firms and workers.

    Arguments:
        sim_params (ParamsDict): dictionary of parameters for simulating data. Run tw.sim_blm_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.sim_blm_params().
    '''

    def __init__(self, sim_params=None):
        if sim_params is None:
            sim_params = sim_blm_params()

        # Store parameters
        self.params = sim_params
        nl, nk, NNm, NNs = self.params.get_multiple(('nl', 'nk', 'NNm', 'NNs'))

        if NNm is None:
            self.NNm = 10 * np.ones(shape=(nk, nk), dtype=int)
        else:
            self.NNm = NNm
        if NNs is None:
            self.NNs = 10 * np.ones(shape=nk, dtype=int)
        else:
            self.NNs = NNs

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
        all_cols = cat_cols + cts_cols
        if len(all_cols) > len(set(all_cols)):
            for col in all_cols:
                if all_cols.count(col) > 1:
                    raise ValueError(f'Control variable names must be unique, but {col!r} appears multiple times.')

        self.dims = (nl, nk)

    def _sort_A(self, A1, A2):
        '''
        Sort A1 and A2 by cluster means.

        Arguments:
            A1 (NumPy Array): mean of fixed effects in first period
            A2 (NumPy Array): mean of fixed effects in second period

        Returns:
            (tuple): sorted arrays
        '''
        # Extract parameters
        nl, strictly_monotone_A = self.params.get_multiple(('nl', 'strictly_monotone_A'))

        if strictly_monotone_A:
            ## Make A1 and A2 monotone by worker type ##
            for l in range(nl):
                A1[l, :] = np.sort(A1[l, :], axis=0)
                A2[l, :] = np.sort(A2[l, :], axis=0)

        # A_sum = A1 + A2

        ## Sort worker effects ##
        worker_effect_order = np.mean(A1, axis=1).argsort()
        A1 = A1[worker_effect_order, :]
        A2 = A2[worker_effect_order, :]

        if not strictly_monotone_A:
            ## Sort firm effects ##
            firm_effect_order = np.mean(A1, axis=0).argsort()
            A1 = A1[:, firm_effect_order]
            A2 = A2[:, firm_effect_order]

        return A1, A2

    def _gen_params(self, rng=None):
        '''
        Generate parameter values to use for simulating bipartite BLM data.

        Arguments:
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (dict): keys are 'A1', 'A2', 'S1', 'S2', 'pk1', 'pk0', 'A1_cat', 'A2_cat', 'S1_cat', 'S2_cat', 'A1_cts', 'A2_cts', 'S1_cts', and 'S2_cts'. 'A1' gives the mean of fixed effects in the first period; 'A2' gives the mean of fixed effects in the second period; 'S1' gives the standard deviation of fixed effects in the first period; 'S2' gives the standard deviation of fixed effects in the second period; 'pk1' gives the probability of being at each combination of firm types for movers; and 'pk0' gives the probability of being at each firm type for stayers. 'cat' indicates a categorical control variable and 'cts' indicates a continuous control variable.
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        nl, nk = self.params.get_multiple(('nl', 'nk'))
        a1_mu, a1_sig, a2_mu, a2_sig, s1_low, s1_high, s2_low, s2_high, pk1_prior, pk0_prior = self.params.get_multiple(('a1_mu', 'a1_sig', 'a2_mu', 'a2_sig', 's1_low', 's1_high', 's2_low', 's2_high', 'pk1_prior', 'pk0_prior'))
        controls_dict, cat_cols, cts_cols = self.controls_dict, self.cat_cols, self.cts_cols
        stationary_firm_type_variation, stationary_A, stationary_S = self.params.get_multiple(('stationary_firm_type_variation', 'stationary_A', 'stationary_S'))
        dims = self.dims

        #### Draw parameters ####
        # Model for Y1 | Y2, l, k for movers and stayers
        if self.params['linear_additive']:
            alpha_1 = rng.normal(loc=a1_mu / 2, scale=a1_sig / np.sqrt(2), size=nl)
            psi_1 = rng.normal(loc=a1_mu / 2, scale=a1_sig / np.sqrt(2), size=nk)
            A1 = alpha_1[:, None] + psi_1
        else:
            A1 = rng.normal(loc=a1_mu, scale=a1_sig, size=dims)
        S1 = rng.uniform(low=s1_low, high=s1_high, size=dims)
        # Model for Y4 | Y3, l, k for movers and stayers
        if self.params['linear_additive']:
            alpha_2 = rng.normal(loc=a2_mu / 2, scale=a2_sig / np.sqrt(2), size=nl)
            psi_2 = rng.normal(loc=a2_mu / 2, scale=a2_sig / np.sqrt(2), size=nk)
            A2 = alpha_2[:, None] + psi_2
        else:
            A2 = rng.normal(loc=a2_mu, scale=a2_sig, size=dims)
        S2 = rng.uniform(low=s2_low, high=s2_high, size=dims)
        # Model for p(K | l, l') for movers
        if pk1_prior is None:
            pk1_prior = np.ones(nl)
        pk1 = rng.dirichlet(alpha=pk1_prior, size=nk ** 2)
        # Model for p(K | l, l') for stayers
        if pk0_prior is None:
            pk0_prior = np.ones(nl)
        pk0 = rng.dirichlet(alpha=pk0_prior, size=nk)

        ### Control variables ###
        ## Categorical ##
        A1_cat = {col:
                rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=(nl, controls_dict[col]['n']))
                    if controls_dict[col]['worker_type_interaction'] else
                rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=controls_dict[col]['n'])
            for col in cat_cols}
        A2_cat = {col:
                rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=(nl, controls_dict[col]['n']))
                    if controls_dict[col]['worker_type_interaction'] else
                rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=controls_dict[col]['n'])
            for col in cat_cols}
        S1_cat = {col:
                rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=(nl, controls_dict[col]['n']))
                    if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=controls_dict[col]['n'])
            for col in cat_cols}
        S2_cat = {col:
                rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=(nl, controls_dict[col]['n']))
                    if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=controls_dict[col]['n'])
            for col in cat_cols}
        # Stationary #
        for col in cat_cols:
            if controls_dict[col]['stationary_A']:
                A2_cat[col] = A1_cat[col]
            if controls_dict[col]['stationary_S']:
                S2_cat[col] = S1_cat[col]
        # Stationary firm type variation #
        for col in cat_cols:
            if controls_dict[col]['stationary_firm_type_variation']:
                if controls_dict[col]['worker_type_interaction']:
                    A2_cat[col] = np.mean(A2_cat[col], axis=1) + A1_cat[col] - np.mean(A1_cat[col], axis=1)
                else:
                    A2_cat[col] = np.mean(A2_cat[col]) + A1_cat[col] - np.mean(A1_cat[col])
        ## Continuous ##
        A1_cts = {col:
                rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=nl)
                    if controls_dict[col]['worker_type_interaction'] else
                rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=1)
            for col in cts_cols}
        A2_cts = {col:
                rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=nl)
                    if controls_dict[col]['worker_type_interaction'] else
                rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=1)
            for col in cts_cols}
        S1_cts = {col:
                rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=nl)
                    if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=1)
            for col in cts_cols}
        S2_cts = {col:
                rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=nl)
                    if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=1)
            for col in cts_cols}
        # Stationary #
        for col in cts_cols:
            if controls_dict[col]['stationary_A']:
                A2_cts[col] = A1_cts[col]
            if controls_dict[col]['stationary_S']:
                S2_cts[col] = S1_cts[col]
        # Stationary firm type variation #
        for col in cts_cols:
            if controls_dict[col]['stationary_firm_type_variation']:
                A2_cts[col] = np.mean(A2_cts[col]) + A1_cts[col] - np.mean(A1_cts[col])

        ## Sort parameters ##
        A1, A2 = self._sort_A(A1, A2)

        # ## Normalize ##
        # A1, A2, A1_cat, A2_cat = _normalize(nl, A1, A2, A1_cat, A2_cat, self.cat_dict, primary_period='first')

        ## Apply constraints ##
        if stationary_A:
            A2 = A1

        if stationary_S:
            S2 = S1

        if stationary_firm_type_variation:
            A2 = np.mean(A2, axis=1) + A1 - np.mean(A1, axis=1)

        return {'A1': A1, 'A2': A2, 'S1': S1, 'S2': S2, 'pk1': pk1, 'pk0': pk0, 'A1_cat': A1_cat, 'A2_cat': A2_cat, 'S1_cat': S1_cat, 'S2_cat': S2_cat, 'A1_cts': A1_cts, 'A2_cts': A2_cts, 'S1_cts': S1_cts, 'S2_cts': S2_cts}

    def _simulate_movers(self, pk1, rng=None):
        '''
        Simulate data for movers (simulates firm types, not firms).

        Arguments:
            pk1 (NumPy Array): probability of being at each combination of firm types for movers
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (Pandas DataFrame): data for movers (y1/y2: wage; g1/g2: firm type; l: worker type; m=1: mover indicator)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        nl, nk, mmult = self.params.get_multiple(('nl', 'nk', 'mmult'))
        cat_dict, cts_cols = self.cat_dict, self.cts_cols

        # Number of movers who transition between each combination of firm types
        NNm = mmult * self.NNm
        nmi = np.sum(NNm)

        ## Firm classes ##
        G1, G2 = _simulate_firm_classes_movers(nk=nk, NNm=NNm)

        ## Worker types ##
        L = _simulate_worker_types_movers(
            nl=nl, nk=nk, NNm=NNm, G1=G1, G2=G2, pk1=pk1,
            qi=None, qi_cum=None, simulating_data=True, rng=rng
        )

        ## Control variables ##
        A1_cat_draws, A2_cat_draws, A1_cts_draws, A2_cts_draws = \
            _simulate_controls_movers(nmi=nmi, cat_dict=cat_dict, cts_cols=cts_cols, dynamic=False, rng=rng)

        ## Convert to DataFrame ##
        return DataFrame(data={'y1': np.zeros(nmi), 'y2': np.zeros(nmi), 'g1': G1, 'g2': G2, 'l': L, 'm': np.ones(nmi, dtype=int), **A1_cat_draws, **A2_cat_draws, **A1_cts_draws, **A2_cts_draws})

    def _simulate_stayers(self, pk0, rng=None):
        '''
        Simulate data for stayers (simulates firm types, not firms).

        Arguments:
            pk0 (NumPy Array): probability of being at each firm type for stayers
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (Pandas DataFrame): data for stayers (y1/y2: wage; g1/g2: firm type; l: worker type; m=0: stayer indicator)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        nl, nk, smult = self.params.get_multiple(('nl', 'nk', 'smult'))
        cat_dict, cts_cols = self.cat_dict, self.cts_cols

        # Number of stayers at each firm type
        NNs = smult * self.NNs
        nsi = np.sum(NNs)

        ## Firm classes ##
        G = _simulate_firm_classes_stayers(nk=nk, NNs=NNs)

        ## Worker types ##
        L = _simulate_worker_types_stayers(
            nl=nl, nk=nk, NNs=NNs, G=G, pk0=pk0,
            qi=None, qi_cum=None, simulating_data=True, rng=rng
        )

        ## Control variables ##
        A1_cat_draws, A2_cat_draws, A1_cts_draws, A2_cts_draws = \
            _simulate_controls_stayers(nsi=nsi, cat_dict=cat_dict, cts_cols=cts_cols, dynamic=False, rng=rng)

        ## Convert to DataFrame ##
        return DataFrame(data={'y1': np.zeros(nsi), 'y2': np.zeros(nsi), 'g1': G, 'g2': G.copy(), 'l': L, 'm': np.zeros(nsi, dtype=int), **A1_cat_draws, **A2_cat_draws, **A1_cts_draws, **A2_cts_draws})

    def simulate(self, return_parameters=False, rng=None):
        '''
        Simulate data (movers and stayers). All firms have the same expected size. Columns are as follows: y1/y2=wage; j1/j2=firm id; g1/g2=firm type; l=worker type.

        Arguments:
            return_parameters (bool): if True, return tuple of (simulated data, simulated parameters); otherwise, return only simulated data
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (dict or tuple of dicts): sim_data gives {'jdata': movers BipartiteDataFrame, 'sdata': stayers BipartiteDataFrame}, while sim_params gives {'A1': A1, 'A2': A2, 'S1': S1, 'S2': S2, 'pk1': pk1, 'pk0': pk0, 'A1_cat': A1_cat, 'A2_cat': A2_cat, 'S1_cat': S1_cat, 'S2_cat': S2_cat, 'A1_cts': A1_cts, 'A2_cts': A2_cts, 'S1_cts': S1_cts, 'S2_cts': S2_cts}; if return_parameters=True, returns (sim_data, sim_params); if return_parameters=False, returns sim_data
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        ## Simulate parameters ##
        sim_params = self._gen_params(rng=rng)

        ## Simulate movers and stayers data ##
        jdata = self._simulate_movers(pk1=sim_params['pk1'], rng=rng)
        sdata = self._simulate_stayers(pk0=sim_params['pk0'], rng=rng)

        ## Simulate firms ##
        _simulate_firms(jdata, sdata, self.params['firm_size'], dynamic=False, rng=rng)

        ## Add i column ##
        nm = len(jdata)
        ns = len(sdata)
        jdata.loc[:, 'i'] = np.arange(nm, dtype=int)
        sdata.loc[:, 'i'] = nm + np.arange(ns, dtype=int)

        ## Sort columns ##
        sorted_cols = _sort_cols(jdata.columns)
        jdata = jdata.reindex(sorted_cols, axis=1, copy=False)
        sdata = sdata.reindex(sorted_cols, axis=1, copy=False)

        ## Convert into BipartiteDataFrame ##
        jdata = BipartiteDataFrame(
            jdata,
            custom_dtype_dict={col: 'categorical' for col in self.cat_cols + ['l']},
            custom_how_collapse_dict={col: 'first' for col in self.cat_cols + ['l']},
            custom_long_es_split_dict={'l': False}
        )
        sdata = BipartiteDataFrame(
            sdata,
            custom_dtype_dict={col: 'categorical' for col in self.cat_cols + ['l']},
            custom_how_collapse_dict={col: 'first' for col in self.cat_cols + ['l']},
            custom_long_es_split_dict={'l': False}
        )

        ## Simulate wages ##
        Y1m, Y2m = _simulate_wages_movers(jdata, L=jdata.loc[:, 'l'], controls_dict=self.controls_dict, cat_cols=self.cat_cols, cts_cols=self.cts_cols, rng=rng, **sim_params)
        Y1s, Y2s = _simulate_wages_stayers(sdata, L=sdata.loc[:, 'l'], controls_dict=self.controls_dict, cat_cols=self.cat_cols, cts_cols=self.cts_cols, rng=rng, **sim_params)
        jdata.loc[:, 'y1'], jdata.loc[:, 'y2'] = (Y1m, Y2m)
        sdata.loc[:, 'y1'], sdata.loc[:, 'y2'] = (Y1s, Y2s)

        ## Combine into dictionary ##
        sim_data = {'jdata': jdata, 'sdata': sdata}

        if return_parameters:
            return sim_data, sim_params
        return sim_data
