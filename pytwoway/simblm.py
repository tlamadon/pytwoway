'''
Class for simulating bipartite BLM networks.
'''
import numpy as np
from pandas import DataFrame
from bipartitepandas.util import ParamsDict, _is_subtype

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq2(a):
    return a >= 2
def _gteq1(a):
    return a >= 1
def _gteq0(a):
    return a >= 0
def _gt0(a):
    return a > 0
def _min_gt0(a):
    return np.min(a) > 0

# Define default parameter dictionaries
_sim_params_default = ParamsDict({
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
        ''', '>= 1'),
    'categorical_time_varying_worker_interaction_controls_dict': (None, 'dict_of_type_none', ParamsDict,
        '''
            (default=None) Dictionary linking column names to instances of tw.sim_categorical_time_varying_worker_interaction_params(). Each instance specifies a new control variable where the effect interacts with worker types and can vary between the first and second periods. Run tw.sim_categorical_time_varying_worker_interaction_params().describe_all() for descriptions of all valid parameters. None is equivalent to {}.
        ''', None),
    'categorical_time_nonvarying_worker_interaction_controls_dict': (None, 'dict_of_type_none', ParamsDict,
        '''
            (default=None) Dictionary linking column names to instances of tw.sim_categorical_time_nonvarying_worker_interaction_params(). Each instance specifies a new control variable where the effect interacts with worker types and is the same in the first and second periods. Run tw.sim_categorical_time_nonvarying_worker_interaction_params().describe_all() for descriptions of all valid parameters. None is equivalent to {}.
        ''', None),
    'categorical_time_varying_controls_dict': (None, 'dict_of_type_none', ParamsDict,
        '''
            (default=None) Dictionary linking column names to instances of tw.sim_categorical_time_varying_params(). Each instance specifies a new control variable where the effect can vary between the first and second periods. Run tw.sim_categorical_time_varying_params().describe_all() for descriptions of all valid parameters. None is equivalent to {}.
        ''', None),
    'categorical_time_nonvarying_controls_dict': (None, 'dict_of_type_none', ParamsDict,
        '''
            (default=None) Dictionary linking column names to instances of tw.sim_categorical_time_nonvarying_params(). Each instance specifies a new control variable where the effect is the same in the first and second periods. Run tw.sim_categorical_time_nonvarying_params().describe_all() for descriptions of all valid parameters. None is equivalent to {}.
        ''', None),
    'continuous_time_varying_worker_interaction_controls_dict': (None, 'dict_of_type_none', ParamsDict,
        '''
            (default=None) Dictionary linking column names to instances of tw.sim_continuous_time_varying_worker_interaction_params(). Each instance specifies a new control variable where the effect interacts with worker types and can vary between the first and second periods. Run tw.sim_continuous_time_varying_worker_interaction_params().describe_all() for descriptions of all valid parameters. None is equivalent to {}.
        ''', None),
    'continuous_time_nonvarying_worker_interaction_controls_dict': (None, 'dict_of_type_none', ParamsDict,
        '''
            (default=None) Dictionary linking column names to instances of tw.sim_continuous_time_nonvarying_worker_interaction_params(). Each instance specifies a new control variable where the effect interacts with worker types and is the same in the first and second periods. Run tw.sim_continuous_time_nonvarying_worker_interaction_params().describe_all() for descriptions of all valid parameters. None is equivalent to {}.
        ''', None),
    'continuous_time_varying_controls_dict': (None, 'dict_of_type_none', ParamsDict,
        '''
            (default=None) Dictionary linking column names to instances of tw.sim_continuous_time_varying_params(). Each instance specifies a new control variable where the effect can vary between the first and second periods. Run tw.sim_continuous_time_varying_params().describe_all() for descriptions of all valid parameters. None is equivalent to {}.
        ''', None),
    'continuous_time_nonvarying_controls_dict': (None, 'dict_of_type_none', ParamsDict,
        '''
            (default=None) Dictionary linking column names to instances of tw.sim_continuous_time_nonvarying_params(). Each instance specifies a new control variable where the effect is the same in the first and second periods. Run tw.sim_continuous_time_nonvarying_params().describe_all() for descriptions of all valid parameters. None is equivalent to {}.
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
    'a1_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A1 (mean of fixed effects in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A1 (mean of fixed effects in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A2 (mean of fixed effects in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2 (mean of fixed effects in second period).
        ''', '>= 0'),
    's1_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S1 (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's1_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S1 (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's2_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2 (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    's2_high': (0.5, 'type_constrained', ((float, int), _gteq0),
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
    'strictly_monotone_a': (False, 'type', bool,
        '''
            (default=False) If True, set A1 and A2 to be strictly increasing by firm type for each worker type (otherwise, they are required to be increasing only by firm type over the average for all worker types).
        ''', None),
    'fixb': (False, 'type', bool,
        '''
            (default=False) If True, set A2 = np.mean(A2, axis=1) + A1 - np.mean(A1, axis=1).
        ''', None),
    'stationary': (False, 'type', bool,
        '''
            (default=False) If True, set A1 = A2.
        ''', None)
})

def sim_params(update_dict=None):
    '''
    Dictionary of default sim_params. Run tw.sim_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of sim_params
    '''
    new_dict = _sim_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

_sim_categorical_time_varying_worker_interaction_params_default = ParamsDict({
    'n': (6, 'type_constrained', (int, _gteq2),
        '''
            (default=6) Number of types for the parameter.
        ''', '>= 2'),
    'a1_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A1_cat_wi (mean of fixed effects in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A1_cat_wi (mean of fixed effects in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A2_cat_wi (mean of fixed effects in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2_cat_wi (mean of fixed effects in second period).
        ''', '>= 0'),
    's1_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S1_cat_wi (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's1_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S1_cat_wi (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's2_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2_cat_wi (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    's2_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S2_cat_wi (standard deviation of fixed effects in second period).
        ''', '>= 0')
})

def sim_categorical_time_varying_worker_interaction_params(update_dict=None):
    '''
    Dictionary of default sim_categorical_time_varying_worker_interaction_params. Run tw.sim_categorical_time_varying_worker_interaction_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of sim_categorical_time_varying_worker_interaction_params
    '''
    new_dict = _sim_categorical_time_varying_worker_interaction_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

_sim_categorical_time_nonvarying_worker_interaction_params_default = ParamsDict({
    'n': (6, 'type_constrained', (int, _gteq2),
        '''
            (default=6) Number of types for the parameter.
        ''', '>= 2'),
    'a_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A_cat_wi (mean of fixed effects in first and second periods).
        ''', None),
    'a_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A_cat_wi (mean of fixed effects in first and second periods).
        ''', '>= 0'),
    's_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S_cat_wi (standard deviation of fixed effects in first and second periods).
        ''', '>= 0'),
    's_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S_cat_wi (standard deviation of fixed effects in first and second periods).
        ''', '>= 0')
})

def sim_categorical_time_nonvarying_worker_interaction_params(update_dict=None):
    '''
    Dictionary of default sim_categorical_time_nonvarying_worker_interaction_params. Run tw.sim_categorical_time_nonvarying_worker_interaction_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of sim_categorical_time_nonvarying_worker_interaction_params
    '''
    new_dict = _sim_categorical_time_nonvarying_worker_interaction_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

_sim_categorical_time_varying_params_default = ParamsDict({
    'n': (6, 'type_constrained', (int, _gteq2),
        '''
            (default=6) Number of types for the parameter.
        ''', '>= 2'),
    'a1_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A1_cat (mean of fixed effects in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A1_cat (mean of fixed effects in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A2_cat (mean of fixed effects in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2_cat (mean of fixed effects in second period).
        ''', '>= 0'),
    's1_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S1_cat (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's1_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S1_cat (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's2_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2_cat (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    's2_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S2_cat (standard deviation of fixed effects in second period).
        ''', '>= 0')
})

def sim_categorical_time_varying_params(update_dict=None):
    '''
    Dictionary of default sim_categorical_time_varying_params. Run tw.sim_categorical_time_varying_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of sim_categorical_time_varying_params
    '''
    new_dict = _sim_categorical_time_varying_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

_sim_categorical_time_nonvarying_params_default = ParamsDict({
    'n': (6, 'type_constrained', (int, _gteq2),
        '''
            (default=6) Number of types for the parameter.
        ''', '>= 2'),
    'a_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A_cat (mean of fixed effects in first and second periods).
        ''', None),
    'a_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A_cat (mean of fixed effects in first and second periods).
        ''', '>= 0'),
    's_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S_cat (standard deviation of fixed effects in first and second periods).
        ''', '>= 0'),
    's_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S_cat (standard deviation of fixed effects in first and second periods).
        ''', '>= 0')
})

def sim_categorical_time_nonvarying_params(update_dict=None):
    '''
    Dictionary of default sim_categorical_time_nonvarying_params. Run tw.sim_categorical_time_nonvarying_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of sim_categorical_time_nonvarying_params
    '''
    new_dict = _sim_categorical_time_nonvarying_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

_sim_continuous_time_varying_worker_interaction_params_default = ParamsDict({
    'a1_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A1_cts_wi (mean of coefficient in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A1_cts_wi (mean of coefficient in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A2_cts_wi (mean of coefficient in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2_cts_wi (mean of coefficient in second period).
        ''', '>= 0'),
    's1_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S1_cts_wi (standard deviation of coefficient in first period).
        ''', '>= 0'),
    's1_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S1_cts_wi (standard deviation of coefficient in first period).
        ''', '>= 0'),
    's2_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2_cts_wi (standard deviation of coefficient in second period).
        ''', '>= 0'),
    's2_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S2_cts_wi (standard deviation of coefficient in second period).
        ''', '>= 0')
})

def sim_continuous_time_varying_worker_interaction_params(update_dict=None):
    '''
    Dictionary of default sim_continuous_time_varying_worker_interaction_params. Run tw.sim_continuous_time_varying_worker_interaction_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of sim_continuous_time_varying_worker_interaction_params
    '''
    new_dict = _sim_continuous_time_varying_worker_interaction_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

_sim_continuous_time_nonvarying_worker_interaction_params_default = ParamsDict({
    'a_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A_cts_wi (mean of coefficient in first and second periods).
        ''', None),
    'a_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A_cts_wi (mean of coefficient in first and second periods).
        ''', '>= 0'),
    's_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S_cts_wi (standard deviation of coefficient in first and second periods).
        ''', '>= 0'),
    's_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S_cts_wi (standard deviation of coefficient in first and second periods).
        ''', '>= 0')
})

def sim_continuous_time_nonvarying_worker_interaction_params(update_dict=None):
    '''
    Dictionary of default sim_continuous_time_nonvarying_worker_interaction_params. Run tw.sim_continuous_time_nonvarying_worker_interaction_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of sim_continuous_time_nonvarying_worker_interaction_params
    '''
    new_dict = _sim_continuous_time_nonvarying_worker_interaction_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

_sim_continuous_time_varying_params_default = ParamsDict({
    'a1_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A1_cts (mean of coefficient in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A1_cts (mean of coefficient in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A2_cts (mean of coefficient in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2_cts (mean of coefficient in second period).
        ''', '>= 0'),
    's1_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S1_cts (standard deviation of coefficient in first period).
        ''', '>= 0'),
    's1_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S1_cts (standard deviation of coefficient in first period).
        ''', '>= 0'),
    's2_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2_cts (standard deviation of coefficient in second period).
        ''', '>= 0'),
    's2_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S2_cts (standard deviation of coefficient in second period).
        ''', '>= 0')
})

def sim_continuous_time_varying_params(update_dict=None):
    '''
    Dictionary of default sim_continuous_time_varying_params. Run tw.sim_continuous_time_varying_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of sim_continuous_time_varying_params
    '''
    new_dict = _sim_continuous_time_varying_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

_sim_continuous_time_nonvarying_params_default = ParamsDict({
    'a_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A_cat (mean of coefficient in first and second periods).
        ''', None),
    'a_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A_cat (mean of coefficient in first and second periods).
        ''', '>= 0'),
    's_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S_cat (standard deviation of coefficient in first and second periods).
        ''', '>= 0'),
    's_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S_cat (standard deviation of coefficient in first and second periods).
        ''', '>= 0')
})

def sim_continuous_time_nonvarying_params(update_dict=None):
    '''
    Dictionary of default sim_continuous_time_nonvarying_params. Run tw.sim_continuous_time_nonvarying_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict): user parameter values

    Returns:
        (ParamsDict) dictionary of sim_continuous_time_nonvarying_params
    '''
    new_dict = _sim_continuous_time_nonvarying_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

class SimBLM:
    '''
    Class of SimBLM, where SimBLM simulates a bipartite BLM network of firms and workers.

    Arguments:
        sim_params (ParamsDict): dictionary of parameters for simulating data. Run tw.sim_params().describe_all() for descriptions of all valid parameters.
    '''

    def __init__(self, sim_params=sim_params()):
        # Store parameters
        self.params = sim_params
        nl, nk, NNm, NNs = self.params.get_multiple(('nl', 'nk', 'NNm', 'NNs'))

        if NNm is None:
            self.NNm = 10 * np.ones(shape=(nk, nk)).astype(int, copy=False)
        else:
            self.NNm = NNm
        if NNs is None:
            self.NNs = 10 * np.ones(shape=nk).astype(int, copy=False)
        else:
            self.NNs = NNs

        ## Unpack control variable parameters
        cat_tv_wi_dict = self.params['categorical_time_varying_worker_interaction_controls_dict']
        cat_tnv_wi_dict = self.params['categorical_time_nonvarying_worker_interaction_controls_dict']
        cat_tv_dict = self.params['categorical_time_varying_controls_dict']
        cat_tnv_dict = self.params['categorical_time_nonvarying_controls_dict']
        cts_tv_wi_dict = self.params['continuous_time_varying_worker_interaction_controls_dict']
        cts_tnv_wi_dict = self.params['continuous_time_nonvarying_worker_interaction_controls_dict']
        cts_tv_dict = self.params['continuous_time_varying_controls_dict']
        cts_tnv_dict = self.params['continuous_time_nonvarying_controls_dict']
        ## Check if control variable parameters are None
        if cat_tv_wi_dict is None:
            cat_tv_wi_dict = {}
        if cat_tnv_wi_dict is None:
            cat_tnv_wi_dict = {}
        if cat_tv_dict is None:
            cat_tv_dict = {}
        if cat_tnv_dict is None:
            cat_tnv_dict = {}
        if cts_tv_wi_dict is None:
            cts_tv_wi_dict = {}
        if cts_tnv_wi_dict is None:
            cts_tnv_wi_dict = {}
        if cts_tv_dict is None:
            cts_tv_dict = {}
        if cts_tnv_dict is None:
            cts_tnv_dict = {}
        ## Create dictionary of all control variables
        controls_dict = cat_tv_wi_dict.copy()
        controls_dict.update(cat_tnv_wi_dict)
        controls_dict.update(cat_tv_dict)
        controls_dict.update(cat_tnv_dict)
        controls_dict.update(cts_tv_wi_dict)
        controls_dict.update(cts_tnv_wi_dict)
        controls_dict.update(cts_tv_dict)
        controls_dict.update(cts_tnv_dict)
        ## Control variable ordering
        cat_tv_wi_cols = sorted(cat_tv_wi_dict.keys())
        cat_tnv_wi_cols = sorted(cat_tnv_wi_dict.keys())
        cat_tv_cols = sorted(cat_tv_dict.keys())
        cat_tnv_cols = sorted(cat_tnv_dict.keys())
        cts_tv_wi_cols = sorted(cts_tv_wi_dict.keys())
        cts_tnv_wi_cols = sorted(cts_tnv_wi_dict.keys())
        cts_tv_cols = sorted(cts_tv_dict.keys())
        cts_tnv_cols = sorted(cts_tnv_dict.keys())
        ## Store control variable attributes
        # Dictionaries
        self.controls_dict = controls_dict
        self.cat_tv_wi_dict = cat_tv_wi_dict
        self.cat_tnv_wi_dict = cat_tnv_wi_dict
        self.cat_tv_dict = cat_tv_dict
        self.cat_tnv_dict = cat_tnv_dict
        self.cts_tv_wi_dict = cts_tv_wi_dict
        self.cts_tnv_wi_dict = cts_tnv_wi_dict
        self.cts_tv_dict = cts_tv_dict
        self.cts_tnv_dict = cts_tnv_dict
        # Lists
        self.cat_tv_wi_cols = cat_tv_wi_cols
        self.cat_tnv_wi_cols = cat_tnv_wi_cols
        self.cat_tv_cols = cat_tv_cols
        self.cat_tnv_cols = cat_tnv_cols
        self.cts_tv_wi_cols = cts_tv_wi_cols
        self.cts_tnv_wi_cols = cts_tnv_wi_cols
        self.cts_tv_cols = cts_tv_cols
        self.cts_tnv_cols = cts_tnv_cols

        # Check that no control variables appear in multiple groups
        all_cols = cat_tv_wi_cols + cat_tnv_wi_cols + cat_tv_cols + cat_tnv_cols + cts_tv_wi_cols + cts_tnv_wi_cols + cts_tv_cols + cts_tnv_cols
        if len(all_cols) > len(set(all_cols)):
            for col in all_cols:
                if all_cols.count(col) > 1:
                    raise ValueError(f'Control variable names must be unique, but {col!r} appears in multiple groups.')

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
        nl, strictly_monotone_a = self.params.get_multiple(('nl', 'strictly_monotone_a'))
        A_mean = (A1 + A2) / 2

        if strictly_monotone_a:
            ## Make A1 and A2 monotone by worker type ##
            for l in range(nl):
                A1[l] = np.sort(A1[l], axis=0)
                A2[l] = np.sort(A2[l], axis=0)

        ## Sort worker effects ##
        worker_effect_order = np.mean(A1, axis=1).argsort()
        A1 = A1[worker_effect_order, :]
        A2 = A2[worker_effect_order, :]

        if not strictly_monotone_a:
            ## Sort firm effects ##
            firm_effect_order = np.mean(A_mean, axis=0).argsort()
            A1 = A1[:, firm_effect_order]
            A2 = A2[:, firm_effect_order]

        return A1, A2

    def _gen_params(self, rng=None):
        '''
        Generate parameter values to use for simulating bipartite BLM data.

        Arguments:
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (dict): keys are 'A1', 'A2', 'S1', 'S2', 'pk1', 'pk0', 'A1_cat_wi', 'A2_cat_wi', 'A_cat_wi', 'S1_cat_wi', 'S2_cat_wi', 'S_cat_wi', 'A1_cat', 'A2_cat', 'A_cat', 'S1_cat', 'S2_cat', 'S_cat', 'A1_cts_wi', 'A2_cts_wi', 'A_cts_wi', 'S1_cts_wi', 'S2_cts_wi', 'S_cts_wi', 'A1_cts', 'A2_cts', 'A_cts', 'S1_cts', 'S2_cts', and 'S_cts'. 'A1' gives the mean of fixed effects in the first period; 'A2' gives the mean of fixed effects in the second period; 'S1' gives the standard deviation of fixed effects in the first period; 'S2' gives the standard deviation of fixed effects in the second period; 'pk1' gives the probability of being at each combination of firm types for movers; and 'pk0' gives the probability of being at each firm type for stayers. 'cat' indicates a categorical variable, 'cts' indicates a continuous variable, and 'wi' indicates worker-interaction (the effect interacts with the unobserved worker types).
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        nl, nk = self.params.get_multiple(('nl', 'nk'))
        a1_mu, a1_sig, a2_mu, a2_sig, s1_low, s1_high, s2_low, s2_high, pk1_prior, pk0_prior = self.params.get_multiple(('a1_mu', 'a1_sig', 'a2_mu', 'a2_sig', 's1_low', 's1_high', 's2_low', 's2_high', 'pk1_prior', 'pk0_prior'))
        controls_dict, cat_tv_wi_cols, cat_tnv_wi_cols, cat_tv_cols, cat_tnv_cols, cts_tv_wi_cols, cts_tnv_wi_cols, cts_tv_cols, cts_tnv_cols = self.controls_dict, self.cat_tv_wi_cols, self.cat_tnv_wi_cols, self.cat_tv_cols, self.cat_tnv_cols, self.cts_tv_wi_cols, self.cts_tnv_wi_cols, self.cts_tv_cols, self.cts_tnv_cols
        fixb, stationary = self.params.get_multiple(('fixb', 'stationary'))
        dims = self.dims

        ##### Draw parameters #####
        # Model for Y1 | Y2, l, k for movers and stayers
        A1 = rng.normal(loc=a1_mu, scale=a1_sig, size=dims)
        S1 = rng.uniform(low=s1_low, high=s1_high, size=dims)
        # Model for Y4 | Y3, l, k for movers and stayers
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
        #### Control variables ####
        ### Categorical ###
        ## Worker-interaction ##
        # Time-variable
        A1_cat_wi = {col: rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=(nl, controls_dict[col]['n'])) for col in cat_tv_wi_cols}
        S1_cat_wi = {col: rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=(nl, controls_dict[col]['n'])) for col in cat_tv_wi_cols}
        A2_cat_wi = {col: rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=(nl, controls_dict[col]['n'])) for col in cat_tv_wi_cols}
        S2_cat_wi = {col: rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=(nl, controls_dict[col]['n'])) for col in cat_tv_wi_cols}
        # Time non-variable
        A_cat_wi = {col: rng.normal(loc=controls_dict[col]['a_mu'], scale=controls_dict[col]['a_sig'], size=(nl, controls_dict[col]['n'])) for col in cat_tnv_wi_cols}
        S_cat_wi = {col: rng.uniform(low=controls_dict[col]['s_low'], high=controls_dict[col]['s_high'], size=(nl, controls_dict[col]['n'])) for col in cat_tnv_wi_cols}
        ## Non-worker-interaction ##
        # Time-variable
        A1_cat = {col: rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=controls_dict[col]['n']) for col in cat_tv_cols}
        S1_cat = {col: rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=controls_dict[col]['n']) for col in cat_tv_cols}
        A2_cat = {col: rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=controls_dict[col]['n']) for col in cat_tv_cols}
        S2_cat = {col: rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=controls_dict[col]['n']) for col in cat_tv_cols}
        # Time non-variable
        A_cat = {col: rng.normal(loc=controls_dict[col]['a_mu'], scale=controls_dict[col]['a_sig'], size=controls_dict[col]['n']) for col in cat_tnv_cols}
        S_cat = {col: rng.uniform(low=controls_dict[col]['s_low'], high=controls_dict[col]['s_high'], size=controls_dict[col]['n']) for col in cat_tnv_cols}
        ### Continuous ###
        ## Worker-interaction ##
        # Time-variable
        A1_cts_wi = {col: rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=nl) for col in cts_tv_wi_cols}
        S1_cts_wi = {col: rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=nl) for col in cat_tv_wi_cols}
        A2_cts_wi = {col: rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=nl) for col in cts_tv_wi_cols}
        S2_cts_wi = {col: rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=nl) for col in cts_tv_wi_cols}
        # Time non-variable
        A_cts_wi = {col: rng.normal(loc=controls_dict[col]['a_mu'], scale=controls_dict[col]['a_sig'], size=nl) for col in cts_tnv_wi_cols}
        S_cts_wi = {col: rng.uniform(low=controls_dict[col]['s_low'], high=controls_dict[col]['s_high'], size=nl) for col in cts_tnv_wi_cols}
        ## Non-worker-interaction ##
        # Time-variable
        A1_cts = {col: rng.normal(loc=controls_dict[col]['a1_mu'], scale=controls_dict[col]['a1_sig'], size=1) for col in cts_tv_cols}
        S1_cts = {col: rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=1) for col in cts_tv_cols}
        A2_cts = {col: rng.normal(loc=controls_dict[col]['a2_mu'], scale=controls_dict[col]['a2_sig'], size=1) for col in cts_tv_cols}
        S2_cts = {col: rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=1) for col in cts_tv_cols}
        # Time non-variable
        A_cts = {col: rng.normal(loc=controls_dict[col]['a_mu'], scale=controls_dict[col]['a_sig'], size=1) for col in cts_tnv_cols}
        S_cts = {col: rng.uniform(low=controls_dict[col]['s_low'], high=controls_dict[col]['s_high'], size=1) for col in cts_tnv_cols}

        ## Sort parameters ##
        A1, A2 = self._sort_A(A1, A2)

        if fixb:
            A2 = np.mean(A2, axis=1) + A1 - np.mean(A1, axis=1)

        if stationary:
            A2 = A1

        return {'A1': A1, 'A2': A2, 'S1': S1, 'S2': S2, 'pk1': pk1, 'pk0': pk0, 'A1_cat_wi': A1_cat_wi, 'A2_cat_wi': A2_cat_wi, 'A_cat_wi': A_cat_wi, 'S1_cat_wi': S1_cat_wi, 'S2_cat_wi': S2_cat_wi, 'S_cat_wi': S_cat_wi, 'A1_cat': A1_cat, 'A2_cat': A2_cat, 'A_cat': A_cat, 'S1_cat': S1_cat, 'S2_cat': S2_cat, 'S_cat': S_cat, 'A1_cts_wi': A1_cts_wi, 'A2_cts_wi': A2_cts_wi, 'A_cts_wi': A_cts_wi, 'S1_cts_wi': S1_cts_wi, 'S2_cts_wi': S2_cts_wi, 'S_cts_wi': S_cts_wi, 'A1_cts': A1_cts, 'A2_cts': A2_cts, 'A_cts': A_cts, 'S1_cts': S1_cts, 'S2_cts': S2_cts, 'S_cts': S_cts}

    def _simulate_movers(self, A1, A2, S1, S2, pk1, pk0, A1_cat_wi, A2_cat_wi, A_cat_wi,  S1_cat_wi, S2_cat_wi, S_cat_wi, A1_cat, A2_cat, A_cat, S1_cat, S2_cat, S_cat, A1_cts_wi, A2_cts_wi, A_cts_wi,  S1_cts_wi, S2_cts_wi, S_cts_wi, A1_cts, A2_cts, A_cts, S1_cts, S2_cts, S_cts, rng=None):
        '''
        Simulate data for movers (simulates firm types, not firms).

        Arguments:
            A1 (NumPy Array): mean of fixed effects in the first period
            A2 (NumPy Array): mean of fixed effects in the second period
            S1 (NumPy Array): standard deviation of fixed effects in the first period
            S2 (NumPy Array): standard deviation of fixed effects in the second period
            pk1 (NumPy Array): probability of being at each combination of firm types for movers
            pk0 (NumPy Array): probability of being at each firm type for stayers (used only for _simulate_stayers)
            A1_cat_wi (dict of NumPy Arrays): dictionary of column names linked to mean of fixed effects in the first period for categorical control variables that interact with worker types
            A2_cat_wi (dict of NumPy Arrays): dictionary of column names linked to mean of fixed effects in the second period for categorical control variables that interact with worker types
            A_cat_wi (dict of NumPy Arrays): dictionary of column names linked to mean of fixed effects in both periods for categorical control variables that interact with worker types
            S1_cat_wi (dict of NumPy Arrays): dictionary of column names linked to standard deviation of fixed effects in the first period for categorical control variables that interact with worker types
            S2_cat_wi (dict of NumPy Arrays): dictionary of column names linked to standard deviation of fixed effects in the second period for categorical control variables that interact with worker types
            S_cat_wi (dict of NumPy Arrays): dictionary of column names linked to standard deviation of fixed effects in both periods for categorical control variables that interact with worker types
            A1_cat (dict of NumPy Arrays): dictionary of column names linked to mean of fixed effects in the first period for categorical control variables
            A2_cat (dict of NumPy Arrays): dictionary of column names linked to mean of fixed effects in the second period for categorical control variables
            A_cat (dict of NumPy Arrays): dictionary of column names linked to mean of fixed effects in both periods for categorical control variables
            S1_cat (dict of NumPy Arrays): dictionary of column names linked to standard deviation of fixed effects in the first period for categorical control variables
            S2_cat (dict of NumPy Arrays): dictionary of column names linked to standard deviation of fixed effects in the second period for categorical control variables
            S_cat (dict of NumPy Arrays): dictionary of column names linked to standard deviation of fixed effects in both periods for categorical control variables
            A1_cts_wi (dict of NumPy Arrays): dictionary of column names linked to mean of coefficients in the first period for continuous control variables that interact with worker types
            A2_cts_wi (dict of NumPy Arrays): dictionary of column names linked to mean of coefficients in the second period for continuous control variables that interact with worker types
            A_cts_wi (dict of NumPy Arrays): dictionary of column names linked to mean of coefficients in both periods for continuous control variables that interact with worker types
            S1_cts_wi (dict of NumPy Arrays): dictionary of column names linked to standard deviation of coefficients in the first period for continuous control variables that interact with worker types
            S2_cts_wi (dict of NumPy Arrays): dictionary of column names linked to standard deviation of coefficients in the second period for continuous control variables that interact with worker types
            S_cts_wi (dict of NumPy Arrays): dictionary of column names linked to standard deviation of coefficients in both periods for continuous control variables that interact with worker types
            A1_cts (dict of NumPy Arrays): dictionary of column names linked to mean of coefficients in the first period for continuous control variables
            A2_cts (dict of NumPy Arrays): dictionary of column names linked to mean of coefficients in the second period for continuous control variables
            A_cts (dict of NumPy Arrays): dictionary of column names linked to mean of coefficients in both periods for continuous control variables
            S1_cts (dict of NumPy Arrays): dictionary of column names linked to standard deviation of coefficients in the first period for continuous control variables
            S2_cts (dict of NumPy Arrays): dictionary of column names linked to standard deviation of coefficients in the second period for continuous control variables
            S_cts (dict of NumPy Arrays): dictionary of column names linked to standard deviation of coefficients in both periods for continuous control variables
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (Pandas DataFrame): data for movers (y1/y2: wage; g1/g2: firm type; l: worker type)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        nl, nk, mmult = self.params.get_multiple(('nl', 'nk', 'mmult'))

        # Number of movers who transition between each combination of firm types
        NNm = mmult * self.NNm
        nmi = np.sum(NNm)

        Y1 = np.zeros(shape=nmi)
        Y2 = np.zeros(shape=nmi)
        G1 = np.zeros(shape=nmi).astype(int, copy=False)
        G2 = np.zeros(shape=nmi).astype(int, copy=False)
        L = np.zeros(shape=nmi).astype(int, copy=False)

        # Worker types
        worker_types = np.arange(nl)

        i = 0
        for k1 in range(nk):
            for k2 in range(nk):
                # Iterate over all firm type combinations a worker can transition between
                ni = NNm[k1, k2]
                I = np.arange(i, i + ni)
                jj = k1 + nk * k2
                G1[I] = k1
                G2[I] = k2

                # Draw worker types
                Li = rng.choice(worker_types, size=ni, replace=True, p=pk1[jj, :])
                L[I] = Li

                # Draw wages
                Y1[I] = rng.normal(loc=A1[Li, k1], scale=S1[Li, k1], size=ni)
                Y2[I] = rng.normal(loc=A2[Li, k2], scale=S2[Li, k2], size=ni)

                i += ni

        if len(self.controls_dict) > 0:
            #### Draw custom columns #### FIXME add custom probabilities?
            ### Categorical ###
            ## Worker-interaction ##
            A1_cat_wi_draws = {cat_col: rng.choice(np.arange(cat_dict['n']), size=nmi, replace=True) for cat_col, cat_dict in self.cat_tv_wi_dict.items()}
            A2_cat_wi_draws = {cat_col: rng.choice(np.arange(cat_dict['n']), size=nmi, replace=True) for cat_col, cat_dict in self.cat_tv_wi_dict.items()}
            A_cat_wi_draws = {cat_col: rng.choice(np.arange(cat_dict['n']), size=nmi, replace=True) for cat_col, cat_dict in self.cat_tnv_wi_dict.items()}
            ## Non-worker-interaction ##
            A1_cat_draws = {cat_col: rng.choice(np.arange(cat_dict['n']), size=nmi, replace=True) for cat_col, cat_dict in self.cat_tv_dict.items()}
            A2_cat_draws = {cat_col: rng.choice(np.arange(cat_dict['n']), size=nmi, replace=True) for cat_col, cat_dict in self.cat_tv_dict.items()}
            A_cat_draws = {cat_col: rng.choice(np.arange(cat_dict['n']), size=nmi, replace=True) for cat_col, cat_dict in self.cat_tnv_dict.items()}
            ## Variances ##
            S1_cat_wi_draws = A1_cat_wi_draws
            S2_cat_wi_draws = A2_cat_wi_draws
            S_cat_wi_draws = A_cat_wi_draws
            S1_cat_draws = A1_cat_draws
            S2_cat_draws = A2_cat_draws
            S_cat_draws = A_cat_draws
            ### Continuous ###
            ## Worker-interaction ##
            A1_cts_wi_draws = {cts_col: rng.normal(size=nmi) for cts_col in self.cts_tv_wi_dict.keys()}
            A2_cts_wi_draws = {cts_col: rng.normal(size=nmi) for cts_col in self.cts_tv_wi_dict.keys()}
            A_cts_wi_draws = {cts_col: rng.normal(size=nmi) for cts_col in self.cts_tnv_wi_dict.keys()}
            ## Non-worker-interaction ##
            A1_cts_draws = {cts_col: rng.normal(size=nmi) for cts_col in self.cts_tv_dict.keys()}
            A2_cts_draws = {cts_col: rng.normal(size=nmi) for cts_col in self.cts_tv_dict.keys()}
            A_cts_draws = {cts_col: rng.normal(size=nmi) for cts_col in self.cts_tnv_dict.keys()}

            # Simulate control variable wages
            A1_sum = np.zeros(shape=nmi)
            A2_sum = np.zeros(shape=nmi)
            S1_sum_sq = np.zeros(shape=nmi)
            S2_sum_sq = np.zeros(shape=nmi)
            ### Categorical ###
            ## Worker-interaction ##
            for cat_col in self.cat_tv_wi_cols:
                A1_sum += A1_cat_wi[cat_col][A1_cat_wi_draws[cat_col], L]
                A2_sum += A2_cat_wi[cat_col][A2_cat_wi_draws[cat_col], L]
                S1_sum_sq += S1_cat_wi[cat_col][S1_cat_wi_draws[cat_col], L] ** 2
                S2_sum_sq += S2_cat_wi[cat_col][S2_cat_wi_draws[cat_col], L] ** 2
            for cat_col in self.cat_tnv_wi_cols:
                A1_sum += A_cat_wi[cat_col][A_cat_wi_draws[cat_col], L]
                A2_sum += A_cat_wi[cat_col][A_cat_wi_draws[cat_col], L]
                S1_sum_sq += S_cat_wi[cat_col][S_cat_wi_draws[cat_col], L] ** 2
                S2_sum_sq += S_cat_wi[cat_col][S_cat_wi_draws[cat_col], L] ** 2
            ## Non-worker-interaction ##
            for cat_col in self.cat_tv_cols:
                A1_sum += A1_cat[cat_col][A1_cat_draws[cat_col]]
                A2_sum += A2_cat[cat_col][A2_cat_draws[cat_col]]
                S1_sum_sq += S1_cat[cat_col][S1_cat_draws[cat_col]] ** 2
                S2_sum_sq += S2_cat[cat_col][S2_cat_draws[cat_col]] ** 2
            for cat_col in self.cat_tnv_cols:
                A1_sum += A_cat[cat_col][A_cat_draws[cat_col]]
                A2_sum += A_cat[cat_col][A_cat_draws[cat_col]]
                S1_sum_sq += S_cat[cat_col][S_cat_draws[cat_col]] ** 2
                S2_sum_sq += S_cat[cat_col][S_cat_draws[cat_col]] ** 2
            ### Continuous ###
            ## Worker-interaction ##
            for cts_col in self.cts_tv_wi_cols:
                A1_sum += A1_cts_wi[cts_col][L] * A1_cts_wi_draws[cts_col]
                A2_sum += A2_cts_wi[cts_col][L] * A2_cts_wi_draws[cts_col]
                S1_sum_sq += S1_cts_wi[cts_col][L] ** 2
                S2_sum_sq += S2_cts_wi[cts_col][L] ** 2
            for cts_col in self.cts_tnv_wi_cols:
                A1_sum += A_cts_wi[cts_col][L] * A_cts_wi_draws[cts_col]
                A2_sum += A_cts_wi[cts_col][L] * A_cts_wi_draws[cts_col]
                S1_sum_sq += S_cts_wi[cts_col][L] ** 2
                S2_sum_sq += S_cts_wi[cts_col][L] ** 2
            ## Non-worker-interaction ##
            for cts_col in self.cts_tv_cols:
                A1_sum += A1_cts[cts_col] * A1_cts_draws[cts_col]
                A2_sum += A2_cts[cts_col] * A2_cts_draws[cts_col]
                S1_sum_sq += S1_cts[cts_col] ** 2
                S2_sum_sq += S2_cts[cts_col] ** 2
            for cts_col in self.cts_tnv_cols:
                A1_sum += A_cts[cts_col] * A_cts_draws[cts_col]
                A2_sum += A_cts[cts_col] * A_cts_draws[cts_col]
                S1_sum_sq += S_cts[cts_col] ** 2
                S2_sum_sq += S_cts[cts_col] ** 2

            Y1 += rng.normal(loc=A1_sum, scale=np.sqrt(S1_sum_sq), size=nmi)
            Y2 += rng.normal(loc=A2_sum, scale=np.sqrt(S2_sum_sq), size=nmi)

            A1_cat_wi_draws = {k + '1': v for k, v in A1_cat_wi_draws.items()}
            A2_cat_wi_draws = {k + '2': v for k, v in A2_cat_wi_draws.items()}
            A1_cat_draws = {k + '1': v for k, v in A1_cat_draws.items()}
            A2_cat_draws = {k + '2': v for k, v in A2_cat_draws.items()}

            return DataFrame(data={'y1': Y1, 'y2': Y2, 'g1': G1, 'g2': G2, 'l': L, **A1_cat_wi_draws, **A2_cat_wi_draws, **A_cat_wi_draws, **A1_cat_draws, **A2_cat_draws, **A_cat_draws, **A1_cts_wi_draws, **A2_cts_wi_draws, **A_cts_wi_draws, **A1_cts_draws, **A2_cts_draws, **A_cts_draws})

        return DataFrame(data={'y1': Y1, 'y2': Y2, 'g1': G1, 'g2': G2, 'l': L})

    def _simulate_stayers(self, A1, A2, S1, S2, pk1, pk0, A1_cat_wi, A2_cat_wi, A_cat_wi,  S1_cat_wi, S2_cat_wi, S_cat_wi, A1_cat, A2_cat, A_cat, S1_cat, S2_cat, S_cat, A1_cts_wi, A2_cts_wi, A_cts_wi,  S1_cts_wi, S2_cts_wi, S_cts_wi, A1_cts, A2_cts, A_cts, S1_cts, S2_cts, S_cts, rng=None):
        '''
        Simulate data for stayers (simulates firm types, not firms).

        Arguments:
            A1 (NumPy Array): mean of fixed effects in the first period
            A2 (NumPy Array): mean of fixed effects in the second period
            S1 (NumPy Array): standard deviation of fixed effects in the first period
            S2 (NumPy Array): standard deviation of fixed effects in the second period
            pk1 (NumPy Array): probability of being at each combination of firm types for movers (used only for _simulate_movers)
            pk0 (NumPy Array): probability of being at each firm type for stayers
            A1_cat_wi (dict of NumPy Arrays): dictionary of column names linked to mean of fixed effects in the first period for categorical control variables that interact with worker types
            A2_cat_wi (dict of NumPy Arrays): dictionary of column names linked to mean of fixed effects in the second period for categorical control variables that interact with worker types
            A_cat_wi (dict of NumPy Arrays): dictionary of column names linked to mean of fixed effects in both periods for categorical control variables that interact with worker types
            S1_cat_wi (dict of NumPy Arrays): dictionary of column names linked to standard deviation of fixed effects in the first period for categorical control variables that interact with worker types
            S2_cat_wi (dict of NumPy Arrays): dictionary of column names linked to standard deviation of fixed effects in the second period for categorical control variables that interact with worker types
            S_cat_wi (dict of NumPy Arrays): dictionary of column names linked to standard deviation of fixed effects in both periods for categorical control variables that interact with worker types
            A1_cat (dict of NumPy Arrays): dictionary of column names linked to mean of fixed effects in the first period for categorical control variables
            A2_cat (dict of NumPy Arrays): dictionary of column names linked to mean of fixed effects in the second period for categorical control variables
            A_cat (dict of NumPy Arrays): dictionary of column names linked to mean of fixed effects in both periods for categorical control variables
            S1_cat (dict of NumPy Arrays): dictionary of column names linked to standard deviation of fixed effects in the first period for categorical control variables
            S2_cat (dict of NumPy Arrays): dictionary of column names linked to standard deviation of fixed effects in the second period for categorical control variables
            S_cat (dict of NumPy Arrays): dictionary of column names linked to standard deviation of fixed effects in both periods for categorical control variables
            A1_cts_wi (dict of NumPy Arrays): dictionary of column names linked to mean of coefficients in the first period for continuous control variables that interact with worker types
            A2_cts_wi (dict of NumPy Arrays): dictionary of column names linked to mean of coefficients in the second period for continuous control variables that interact with worker types
            A_cts_wi (dict of NumPy Arrays): dictionary of column names linked to mean of coefficients in both periods for continuous control variables that interact with worker types
            S1_cts_wi (dict of NumPy Arrays): dictionary of column names linked to standard deviation of coefficients in the first period for continuous control variables that interact with worker types
            S2_cts_wi (dict of NumPy Arrays): dictionary of column names linked to standard deviation of coefficients in the second period for continuous control variables that interact with worker types
            S_cts_wi (dict of NumPy Arrays): dictionary of column names linked to standard deviation of coefficients in both periods for continuous control variables that interact with worker types
            A1_cts (dict of NumPy Arrays): dictionary of column names linked to mean of coefficients in the first period for continuous control variables
            A2_cts (dict of NumPy Arrays): dictionary of column names linked to mean of coefficients in the second period for continuous control variables
            A_cts (dict of NumPy Arrays): dictionary of column names linked to mean of coefficients in both periods for continuous control variables
            S1_cts (dict of NumPy Arrays): dictionary of column names linked to standard deviation of coefficients in the first period for continuous control variables
            S2_cts (dict of NumPy Arrays): dictionary of column names linked to standard deviation of coefficients in the second period for continuous control variables
            S_cts (dict of NumPy Arrays): dictionary of column names linked to standard deviation of coefficients in both periods for continuous control variables
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (Pandas DataFrame): data for stayers (y1/y2: wage; g1/g2: firm type; l: worker type)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        nl, nk, smult = self.params.get_multiple(('nl', 'nk', 'smult'))

        # Number of stayers at each firm type
        NNs = smult * self.NNs
        nsi = np.sum(NNs)

        Y1 = np.zeros(shape=nsi)
        Y2 = np.zeros(shape=nsi)
        G = np.zeros(shape=nsi).astype(int, copy=False)
        L = np.zeros(shape=nsi).astype(int, copy=False)

        # Worker types
        worker_types = np.arange(nl)

        i = 0
        for k in range(nk):
            # Iterate over firm types
            ni = NNs[k]
            I = np.arange(i, i + ni)
            G[I] = k

            # Draw worker types
            Li = rng.choice(worker_types, size=ni, replace=True, p=pk0[k, :])
            L[I] = Li

            # Draw wages
            Y1[I] = rng.normal(loc=A1[Li, k], scale=S1[Li, k], size=ni)
            Y2[I] = rng.normal(loc=A2[Li, k], scale=S2[Li, k], size=ni)

            i += ni

        if len(self.controls_dict) > 0:
            #### Draw custom columns #### FIXME add custom probabilities?
            ### Categorical ###
            ## Worker-interaction ##
            A1_cat_wi_draws = {cat_col: rng.choice(np.arange(cat_dict['n']), size=nsi, replace=True) for cat_col, cat_dict in self.cat_tv_wi_dict.items()}
            A2_cat_wi_draws = {cat_col: rng.choice(np.arange(cat_dict['n']), size=nsi, replace=True) for cat_col, cat_dict in self.cat_tv_wi_dict.items()}
            A_cat_wi_draws = {cat_col: rng.choice(np.arange(cat_dict['n']), size=nsi, replace=True) for cat_col, cat_dict in self.cat_tnv_wi_dict.items()}
            ## Non-worker-interaction ##
            A1_cat_draws = {cat_col: rng.choice(np.arange(cat_dict['n']), size=nsi, replace=True) for cat_col, cat_dict in self.cat_tv_dict.items()}
            A2_cat_draws = {cat_col: rng.choice(np.arange(cat_dict['n']), size=nsi, replace=True) for cat_col, cat_dict in self.cat_tv_dict.items()}
            A_cat_draws = {cat_col: rng.choice(np.arange(cat_dict['n']), size=nsi, replace=True) for cat_col, cat_dict in self.cat_tnv_dict.items()}
            ## Variances ##
            S1_cat_wi_draws = A1_cat_wi_draws
            S2_cat_wi_draws = A2_cat_wi_draws
            S_cat_wi_draws = A_cat_wi_draws
            S1_cat_draws = A1_cat_draws
            S2_cat_draws = A2_cat_draws
            S_cat_draws = A_cat_draws
            ### Continuous ###
            ## Worker-interaction ##
            A1_cts_wi_draws = {cts_col: rng.normal(size=nsi) for cts_col in self.cts_tv_wi_dict.keys()}
            A2_cts_wi_draws = {cts_col: rng.normal(size=nsi) for cts_col in self.cts_tv_wi_dict.keys()}
            A_cts_wi_draws = {cts_col: rng.normal(size=nsi) for cts_col in self.cts_tnv_wi_dict.keys()}
            ## Non-worker-interaction ##
            A1_cts_draws = {cts_col: rng.normal(size=nsi) for cts_col in self.cts_tv_dict.keys()}
            A2_cts_draws = {cts_col: rng.normal(size=nsi) for cts_col in self.cts_tv_dict.keys()}
            A_cts_draws = {cts_col: rng.normal(size=nsi) for cts_col in self.cts_tnv_dict.keys()}

            # Simulate control variable wages
            A1_sum = np.zeros(shape=nsi)
            A2_sum = np.zeros(shape=nsi)
            S1_sum_sq = np.zeros(shape=nsi)
            S2_sum_sq = np.zeros(shape=nsi)
            ### Categorical ###
            ## Worker-interaction ##
            for cat_col in self.cat_tv_wi_cols:
                A1_sum += A1_cat_wi[cat_col][A1_cat_wi_draws[cat_col], L]
                A2_sum += A2_cat_wi[cat_col][A2_cat_wi_draws[cat_col], L]
                S1_sum_sq += S1_cat_wi[cat_col][S1_cat_wi_draws[cat_col], L] ** 2
                S2_sum_sq += S2_cat_wi[cat_col][S2_cat_wi_draws[cat_col], L] ** 2
            for cat_col in self.cat_tnv_wi_cols:
                A1_sum += A_cat_wi[cat_col][A_cat_wi_draws[cat_col], L]
                A2_sum += A_cat_wi[cat_col][A_cat_wi_draws[cat_col], L]
                S1_sum_sq += S_cat_wi[cat_col][S_cat_wi_draws[cat_col], L] ** 2
                S2_sum_sq += S_cat_wi[cat_col][S_cat_wi_draws[cat_col], L] ** 2
            ## Non-worker-interaction ##
            for cat_col in self.cat_tv_cols:
                A1_sum += A1_cat[cat_col][A1_cat_draws[cat_col]]
                A2_sum += A2_cat[cat_col][A2_cat_draws[cat_col]]
                S1_sum_sq += S1_cat[cat_col][S1_cat_draws[cat_col]] ** 2
                S2_sum_sq += S2_cat[cat_col][S2_cat_draws[cat_col]] ** 2
            for cat_col in self.cat_tnv_cols:
                A1_sum += A_cat[cat_col][A_cat_draws[cat_col]]
                A2_sum += A_cat[cat_col][A_cat_draws[cat_col]]
                S1_sum_sq += S_cat[cat_col][S_cat_draws[cat_col]] ** 2
                S2_sum_sq += S_cat[cat_col][S_cat_draws[cat_col]] ** 2
            ### Continuous ###
            ## Worker-interaction ##
            for cts_col in self.cts_tv_wi_cols:
                A1_sum += A1_cts_wi[cts_col][L] * A1_cts_wi_draws[cts_col]
                A2_sum += A2_cts_wi[cts_col][L] * A2_cts_wi_draws[cts_col]
                S1_sum_sq += S1_cts_wi[cts_col][L] ** 2
                S2_sum_sq += S2_cts_wi[cts_col][L] ** 2
            for cts_col in self.cts_tnv_wi_cols:
                A1_sum += A_cts_wi[cts_col][L] * A_cts_wi_draws[cts_col]
                A2_sum += A_cts_wi[cts_col][L] * A_cts_wi_draws[cts_col]
                S1_sum_sq += S_cts_wi[cts_col][L] ** 2
                S2_sum_sq += S_cts_wi[cts_col][L] ** 2
            ## Non-worker-interaction ##
            for cts_col in self.cts_tv_cols:
                A1_sum += A1_cts[cts_col] * A1_cts_draws[cts_col]
                A2_sum += A2_cts[cts_col] * A2_cts_draws[cts_col]
                S1_sum_sq += S1_cts[cts_col] ** 2
                S2_sum_sq += S2_cts[cts_col] ** 2
            for cts_col in self.cts_tnv_cols:
                A1_sum += A_cts[cts_col] * A_cts_draws[cts_col]
                A2_sum += A_cts[cts_col] * A_cts_draws[cts_col]
                S1_sum_sq += S_cts[cts_col] ** 2
                S2_sum_sq += S_cts[cts_col] ** 2

            Y1 += rng.normal(loc=A1_sum, scale=np.sqrt(S1_sum_sq), size=nsi)
            Y2 += rng.normal(loc=A2_sum, scale=np.sqrt(S2_sum_sq), size=nsi)

            A1_cat_wi_draws = {k + '1': v for k, v in A1_cat_wi_draws.items()}
            A2_cat_wi_draws = {k + '2': v for k, v in A2_cat_wi_draws.items()}
            A1_cat_draws = {k + '1': v for k, v in A1_cat_draws.items()}
            A2_cat_draws = {k + '2': v for k, v in A2_cat_draws.items()}

            return DataFrame(data={'y1': Y1, 'y2': Y2, 'g1': G, 'g2': G, 'l': L, **A1_cat_wi_draws, **A2_cat_wi_draws, **A_cat_wi_draws, **A1_cat_draws, **A2_cat_draws, **A_cat_draws, **A1_cts_wi_draws, **A2_cts_wi_draws, **A_cts_wi_draws, **A1_cts_draws, **A2_cts_draws, **A_cts_draws})

        return DataFrame(data={'y1': Y1, 'y2': Y2, 'g1': G, 'g2': G, 'l': L})

    def simulate(self, return_parameters=False, rng=None):
        '''
        Simulates data (movers and stayers) and attached firms ids. All firms have the same expected size. Columns are as follows: y1/y2=wage; j1/j2=firm id; g1/g2=firm type; l=worker type

        Arguments:
            return_parameters (bool): if True, return tuple of (simulated data, simulated parameters); otherwise, return only simulated data
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (dict or tuple of dicts): sim_data gives {'jdata': movers data, 'sdata': stayers data}, while sim_params gives {'A1': A1, 'A2': A2, 'S1': S1, 'S2': S2, 'pk1': pk1, 'pk0': pk0, 'A1_cat_wi': A1_cat_wi, 'A2_cat_wi': A2_cat_wi, 'A_cat_wi': A_cat_wi, 'S1_cat_wi': S1_cat_wi, 'S2_cat_wi': S2_cat_wi, 'S_cat_wi': S_cat_wi, 'A1_cat': A1_cat, 'A2_cat': A2_cat, 'A_cat': A_cat, 'S1_cat': S1_cat, 'S2_cat': S2_cat, 'S_cat': S_cat}; if return_parameters=True, returns (sim_data, sim_params); if return_parameters=False, returns sim_data
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        sim_params = self._gen_params(rng=rng)
        jdata = self._simulate_movers(**sim_params, rng=rng)
        sdata = self._simulate_stayers(**sim_params, rng=rng)

        ## Stayers ##
        # Draw firm ids for stayers (note each cluster must link to at least 2 firm ids so that movers are able to move firms within the cluster)
        sdata.loc[:, 'j1'] = np.hstack(sdata.groupby('g1').apply(lambda df: rng.integers(max(2, round(len(df) / self.params['firm_size'])), size=len(df))))

        # Make firm ids contiguous
        sdata.loc[:, 'j1'] = sdata.groupby(['g1', 'j1']).ngroup()
        sdata.loc[:, 'j2'] = sdata.loc[:, 'j1']

        # Link firm ids to clusters
        j_per_g_dict = sdata.groupby('g1')['j1'].unique().to_dict()
        for g, linked_j in j_per_g_dict.items():
            # Make sure each cluster has at least 2 linked firm ids
            if len(linked_j) == 1:
                raise ValueError(f"Cluster {g} has only 1 linked firm. However, each cluster must link to at least 2 firm ids so that movers are able to move firms within the cluster. Please alter parameter values to ensure enough firms can be generated.")

        ## Movers ##
        # Draw firm ids for movers
        jdata.loc[:, 'j1'] = np.hstack(jdata.groupby('g1').apply(lambda df: rng.choice(j_per_g_dict[df.iloc[0]['g1']], size=len(df))))
        groupby_g2 = jdata.groupby('g2')
        jdata.loc[:, 'j2'] = np.hstack(groupby_g2.apply(lambda df: rng.choice(j_per_g_dict[df.iloc[0]['g2']], size=len(df))))

        # Make sure movers actually move
        # FIXME find a deterministic way to do this
        same_firm_mask = (jdata.loc[:, 'j1'].to_numpy() == jdata.loc[:, 'j2'].to_numpy())
        while same_firm_mask.any():
            same_firm_rows = jdata.loc[same_firm_mask, :].index
            jdata.loc[same_firm_rows, 'j2'] = np.hstack(groupby_g2.apply(lambda df: rng.choice(j_per_g_dict[df.iloc[0]['g2']], size=len(df))))[same_firm_rows]
            same_firm_mask = (jdata.loc[:, 'j1'].to_numpy() == jdata.loc[:, 'j2'].to_numpy())

        sim_data = {'jdata': jdata, 'sdata': sdata} # {'jdata': jdata[['y1', 'y2', 'j1', 'j2', 'g1', 'g2', 'l']], 'sdata': sdata[['y1', 'y2', 'j1', 'j2', 'g1', 'g2', 'l']]}

        if return_parameters:
            return sim_data, sim_params
        return sim_data
