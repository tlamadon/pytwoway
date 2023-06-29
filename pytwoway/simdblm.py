'''
Class for simulating bipartite dynamic BLM networks.
'''
import numpy as np
from pandas import DataFrame
from paramsdict import ParamsDict, ParamsDictBase
from paramsdict.util import col_type
from bipartitepandas import BipartiteDataFrame
from bipartitepandas.util import to_list, _sort_cols
from pytwoway import simblm

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
sim_dynamic_blm_params = ParamsDict({
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
    'endogeneity': (True, 'type', bool,
        '''
            (default=True) If True, simulate model with endogeneity (i.e. the firm type after a move can affect earnings before the move).
        ''', None),
    'state_dependence': (True, 'type', bool,
        '''
            (default=True) If True, simulate model with state dependence (i.e. the firm type before a move can affect earnings after the move).
        ''', None),
    'categorical_controls': (None, 'dict_of_type_none', ParamsDictBase,
        '''
            (default=None) Dictionary linking column names to instances of tw.sim_dynamic_categorical_control_params(). Each instance specifies a new categorical control variable. Run tw.sim_dynamic_categorical_control_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
        ''', None),
    'continuous_controls': (None, 'dict_of_type_none', ParamsDictBase,
        '''
            (default=None) Dictionary linking column names to instances of tw.sim_dynamic_continuous_control_params(). Each instance specifies a new continuous control variable. Run tw.sim_dynamic_continuous_control_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
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
    'R12': (0.6, 'type', (float, int),
        '''
            (default=0.6) Persistence parameter between periods 1 and 2.
        ''', None),
    'R43': (0.6, 'type', (float, int),
        '''
            (default=0.6) Persistence parameter between periods 3 and 4.
        ''', None),
    'R32m': (0.6, 'type', (float, int),
        '''
            (default=0.6) Persistence parameter between periods 2 and 3 for movers.
        ''', None),
    'R32s': (0.6, 'type', (float, int),
        '''
            (default=0.6) Persistence parameter between periods 2 and 3 for stayers.
        ''', None),
    ## A ##
    'a12_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A12 (mean of fixed effects).
        ''', None),
    'a12_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A12 (mean of fixed effects).
        ''', '>= 0'),
    'a43_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A43 (mean of fixed effects).
        ''', None),
    'a43_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A43 (mean of fixed effects).
        ''', '>= 0'),
    'a2ma_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A2a for movers (mean of fixed effects).
        ''', None),
    'a2ma_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2a for movers (mean of fixed effects).
        ''', '>= 0'),
    'a2mb_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A2b for movers (mean of fixed effects).
        ''', None),
    'a2mb_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2b for movers (mean of fixed effects).
        ''', '>= 0'),
    'a3ma_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A3a for movers (mean of fixed effects).
        ''', None),
    'a3ma_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A3a for movers (mean of fixed effects).
        ''', '>= 0'),
    'a3mb_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A3b for movers (mean of fixed effects).
        ''', None),
    'a3mb_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A3b for movers (mean of fixed effects).
        ''', '>= 0'),
    'a2s_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A2 for stayers (mean of fixed effects).
        ''', None),
    'a2s_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2 for stayers (mean of fixed effects).
        ''', '>= 0'),
    'a3s_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of simulated A3 for stayers (mean of fixed effects).
        ''', None),
    'a3s_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of simulated A3 for stayers (mean of fixed effects).
        ''', '>= 0'),
    ## S ##
    's12_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S12 (standard deviation of fixed effects).
        ''', '>= 0'),
    's12_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S12 (standard deviation of fixed effects).
        ''', '>= 0'),
    's43_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S43 (standard deviation of fixed effects).
        ''', '>= 0'),
    's43_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S43 (standard deviation of fixed effects).
        ''', '>= 0'),
    's2ma_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2a for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2ma_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S2a for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2mb_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2b for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2mb_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S2b for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3ma_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S3a for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3ma_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S3a for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3mb_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S3b for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3mb_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S3b for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2s_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2 for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2s_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S2 for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3s_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S3 for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3s_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S3 for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    ## Other ##
    'pk1_prior': (None, 'array_of_type_constrained_none', (('float', 'int'), _min_gt0),
        '''
            (default=None) Dirichlet prior for pk1 (probability of being at each combination of firm types for movers). Must have length nl. None is equivalent to np.ones(nl).
        ''', 'min > 0'),
    'pk0_prior': (None, 'array_of_type_constrained_none', (('float', 'int'), _min_gt0),
        '''
            (default=None) Dirichlet prior for pk0 (probability of being at each firm type for stayers). Must have length nl. None is equivalent to np.ones(nl).
        ''', 'min > 0'),
    'strictly_monotone_A': (True, 'type', bool,
        '''
            (default=True) If True, set A to be strictly increasing by firm type for each worker type in each period (otherwise, each period is required to be increasing only by firm type over the average for all worker types).
        ''', None),
    'stationary_A': (False, 'type', bool,
        '''
            (default=False) If True, set A equal across each period.
        ''', None),
    'stationary_S': (False, 'type', bool,
        '''
            (default=False) If True, set S each across each period.
        ''', None),
    'linear_additive': (False, 'type', bool,
        '''
            (default=False) If True, make A linearly additive in each period.
        ''', None),
    'stationary_firm_type_variation': (True, 'type', bool,
        '''
            (default=True) If True, set constraints for A in each period so that the firm type induced variation of worker-firm pair effects is the same in all periods. In particular, this sets A2 = np.mean(A2, axis=1) + A1 - np.mean(A1, axis=1), for each set of periods 1 and 2.
        ''', None)
})

sim_dynamic_categorical_control_params = ParamsDict({
    'n': (6, 'type_constrained', (int, _gteq2),
        '''
            (default=6) Number of types for the parameter.
        ''', '>= 2'),
    ## A ##
    'a12_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A12_cat (mean of fixed effects).
        ''', None),
    'a12_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A12_cat (mean of fixed effects).
        ''', '>= 0'),
    'a43_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A43_cat (mean of fixed effects).
        ''', None),
    'a43_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A43_cat (mean of fixed effects).
        ''', '>= 0'),
    'a2ma_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A2a_cat for movers (mean of fixed effects).
        ''', None),
    'a2ma_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2a_cat for movers (mean of fixed effects).
        ''', '>= 0'),
    'a2mb_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A2b_cat for movers (mean of fixed effects).
        ''', None),
    'a2mb_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2b_cat for movers (mean of fixed effects).
        ''', '>= 0'),
    'a2s_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A2_cat for stayers (mean of fixed effects).
        ''', None),
    'a2s_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2_cat for stayers (mean of fixed effects).
        ''', '>= 0'),
    'a3ma_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A3a_cat for movers (mean of fixed effects).
        ''', None),
    'a3ma_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A3a_cat for movers (mean of fixed effects).
        ''', '>= 0'),
    'a3mb_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A3b_cat for movers (mean of fixed effects).
        ''', None),
    'a3mb_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A3b_cat for movers (mean of fixed effects).
        ''', '>= 0'),
    'a3s_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A3_cat for stayers (mean of fixed effects).
        ''', None),
    'a3s_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A3_cat for stayers (mean of fixed effects).
        ''', '>= 0'),
    ## S ##
    's12_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S12_cat (standard deviation of fixed effects).
        ''', '>= 0'),
    's12_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S12_cat (standard deviation of fixed effects).
        ''', '>= 0'),
    's43_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S43_cat (standard deviation of fixed effects).
        ''', '>= 0'),
    's43_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S43_cat (standard deviation of fixed effects).
        ''', '>= 0'),
    's2ma_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2a_cat for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2ma_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2a_cat for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2mb_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2b_cat for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2mb_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2b_cat for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2s_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2_cat for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2s_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2_cat for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3ma_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S3a_cat for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3ma_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S3a_cat for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3mb_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S3b_cat for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3mb_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S3b_cat for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3s_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S3_cat for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3s_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S3_cat for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    ## Other ##
    'stationary_A': (False, 'type', bool,
        '''
            (default=False) If True, set A_cat equal across each period.
        ''', None),
    'stationary_S': (False, 'type', bool,
        '''
            (default=False) If True, set S_cat equal across each period.
        ''', None),
    'stationary_firm_type_variation': (False, 'type', bool,
        '''
            (default=False) If True, set constraints for A1_cat and A2_cat so that the firm type induced variation of worker-firm pair effects is the same in all periods. In particular, this sets A2_cat = np.mean(A2_cat, axis=1) + A1_cat - np.mean(A1_cat, axis=1), for each set of periods 1 and 2.
        ''', None),
    'worker_type_interaction': (False, 'type', bool,
        '''
            (default=False) If True, effect can differ by worker type.
        ''', None)
})

sim_dynamic_continuous_control_params = ParamsDict({
    ## A ##
    'a12_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A12_cts (mean of fixed effects).
        ''', None),
    'a12_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A12_cts (mean of fixed effects).
        ''', '>= 0'),
    'a43_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A43_cts (mean of fixed effects).
        ''', None),
    'a43_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A43_cts (mean of fixed effects).
        ''', '>= 0'),
    'a2ma_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A2a_cts for movers (mean of fixed effects).
        ''', None),
    'a2ma_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2a_cts for movers (mean of fixed effects).
        ''', '>= 0'),
    'a2mb_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A2b_cts for movers (mean of fixed effects).
        ''', None),
    'a2mb_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2b_cts for movers (mean of fixed effects).
        ''', '>= 0'),
    'a2s_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A2_cts for stayers (mean of fixed effects).
        ''', None),
    'a2s_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2_cts for stayers (mean of fixed effects).
        ''', '>= 0'),
    'a3ma_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A3a_cts for movers (mean of fixed effects).
        ''', None),
    'a3ma_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A3a_cts for movers (mean of fixed effects).
        ''', '>= 0'),
    'a3mb_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A3b_cts for movers (mean of fixed effects).
        ''', None),
    'a3mb_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A3b_cts for movers (mean of fixed effects).
        ''', '>= 0'),
    'a3s_mu': (1, 'type', (float, int, list, col_type),
        '''
            (default=1) Mean of starting values for A3_cts for stayers (mean of fixed effects).
        ''', None),
    'a3s_sig': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A3_cts for stayers (mean of fixed effects).
        ''', '>= 0'),
    ## S ##
    's12_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S12_cts (standard deviation of fixed effects).
        ''', '>= 0'),
    's12_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S12_cts (standard deviation of fixed effects).
        ''', '>= 0'),
    's43_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S43_cts (standard deviation of fixed effects).
        ''', '>= 0'),
    's43_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S43_cts (standard deviation of fixed effects).
        ''', '>= 0'),
    's2ma_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2a_cts for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2ma_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2a_cts for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2mb_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2b_cts for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2mb_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2b_cts for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2s_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2_cts for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2s_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2_cts for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3m_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S3_cts for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3m_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S3_cts for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3s_low': (0.3, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S3_cts for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3s_high': (0.5, 'type_constrained', ((float, int, list, col_type), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S3_cts for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    ## Other ##
    'stationary_A': (False, 'type', bool,
        '''
            (default=False) If True, set A_cts equal across each period.
        ''', None),
    'stationary_S': (False, 'type', bool,
        '''
            (default=False) If True, set S_cts equal across each period.
        ''', None),
    'stationary_firm_type_variation': (False, 'type', bool,
        '''
            (default=False) If True, set constraints for A1_cts and A2_cts so that the firm type induced variation of worker-firm pair effects is the same in all periods. In particular, this sets A2_cts = np.mean(A2_cts) + A1_cts - np.mean(A1_cts), for each set of periods 1 and 2.
        ''', None),
    'worker_type_interaction': (False, 'type', bool,
        '''
            (default=False) If True, effect can differ by worker type.
        ''', None)
})

def _simulate_wages_movers(jdata, L, dynamic_blm_model=None, A=None, S=None, R12=None, R43=None, R32m=None, A_cat=None, S_cat=None, A_cts=None, S_cts=None, controls_dict=None, cat_cols=None, cts_cols=None, rng=None, **kwargs):
    '''
    Simulate wages for movers.

    Arguments:
        jdata (BipartitePandas DataFrame): movers data
        L (NumPy Array): worker types for movers
        dynamic_blm_model (DynamicBLMModel or None): Dynamic BLM model with estimated parameter values; None if other parameters are not None
        A (dict of NumPy Arrays or None): dictionary linking periods to the mean of fixed effects in that period; None if dynamic_blm_model is not None
        S (dict of NumPy Arrays or None): dictionary linking periods to the standard deviation of fixed effects in that period; None if dynamic_blm_model is not None
        R12 (float or None): persistence parameter between periods 1 and 2; None if dynamic_blm_model is not None
        R43 (float or None): persistence parameter between periods 3 and 4; None if dynamic_blm_model is not None
        R32m (float or None): persistence parameter between periods 2 and 3 for movers; None if dynamic_blm_model is not None
        A_cat (dict of dicts of NumPy Arrays or None): dictionary linking each categorical control column name to a dictionary linking periods to the mean of fixed effects in that period; None if dynamic_blm_model is not None
        S_cat (dict of dicts of NumPy Arrays or None): dictionary linking each categorical control column name to a dictionary linking periods to the standard deviation of fixed effects in that period; None if dynamic_blm_model is not None
        A_cts (dict of dicts of NumPy Arrays or None): dictionary linking each continuous control column name to a dictionary linking periods to the mean of fixed effects in that period; None if dynamic_blm_model is not None
        S_cts (dict of dicts of NumPy Arrays or None): dictionary linking each continuous control column name to a dictionary linking periods to the standard deviation of fixed effects in that period; None if dynamic_blm_model is not None
        controls_dict (dict or None): dictionary of all control variables; None if no controls
        cat_cols (list or None): list of categorical controls; None if no categorical controls
        cts_cols (list or None): list of continuous controls; None if no continuous controls
        rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        kwargs (dict): keyword arguments

    Returns:
        (tuple of NumPy Arrays): wages for movers in all four periods
    '''
    if rng is None:
        rng = np.random.default_rng(None)

    # Define variables
    periods_movers = ['12', '43', '2ma', '3ma', '2mb', '3mb']
    first_periods = ['12', '2ma', '3mb', '2s']
    periods_movers_dict = {period: 0 if period in first_periods else 1 for period in periods_movers}

    # Unpack values
    nmi = len(jdata)
    G1 = jdata.loc[:, 'g1'].to_numpy()
    G2 = jdata.loc[:, 'g4'].to_numpy()
    G = [G1, G2]

    if dynamic_blm_model is not None:
        # Unpack DynamicBLMModel
        A = dynamic_blm_model.A
        S = dynamic_blm_model.S
        R12 = dynamic_blm_model.R12
        R43 = dynamic_blm_model.R43
        R32m = dynamic_blm_model.R32m
        A_cat = dynamic_blm_model.A_cat
        S_cat = dynamic_blm_model.S_cat
        A_cts = dynamic_blm_model.A_cts
        S_cts = dynamic_blm_model.S_cts
        controls_dict = dynamic_blm_model.controls_dict
        cat_cols = dynamic_blm_model.cat_cols
        cts_cols = dynamic_blm_model.cts_cols

    ## Draw wages ##
    A_sum = {period:
                A[period][L, G[periods_movers_dict[period]]]
                    if period[-1] != 'b' else
                A[period][G[periods_movers_dict[period]]]
            for period in periods_movers}
    if (controls_dict is not None) and (len(controls_dict) > 0):
        #### Simulate control variable wages ####
        S_sum_sq = {period:
                        (S[period] ** 2)[L, G[periods_movers_dict[period]]]
                            if period[-1] != 'b' else
                        (S[period] ** 2)[G[periods_movers_dict[period]]]
                    for period in periods_movers}
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
                subcols = [subcols[0], subcols[0]]
            elif n_subcols == 4:
                # If column can change over time
                subcols = [subcols[0], subcols[3]]
            else:
                raise NotImplementedError(f'Column names must have either one or four associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
            if i < len(cat_cols):
                ### Categorical ###
                if controls_dict[col]['worker_type_interaction']:
                    ## Worker-interaction ##
                    for period in periods_movers:
                        subcol = periods_movers_dict[period]
                        if period[-1] != 'b':
                            A_sum[period] += A_cat[col][period][L, jdata.loc[:, subcol]]
                            S_sum_sq[period] += (S_cat[col][period] ** 2)[L, jdata.loc[:, subcol]]
                        else:
                            A_sum[period] += A_cat[col][period][jdata.loc[:, subcol]]
                            S_sum_sq[period] += (S_cat[col][period] ** 2)[jdata.loc[:, subcol]]
                else:
                    ## Non-worker-interaction ##
                    for period in periods_movers:
                        subcol = periods_movers_dict[period]
                        A_sum[period] += A_cat[col][period][jdata.loc[:, subcol]]
                        S_sum_sq[period] += (S_cat[col][period] ** 2)[jdata.loc[:, subcol]]
            else:
                ### Continuous ###
                if controls_dict[col]['worker_type_interaction']:
                    ## Worker-interaction ##
                    for period in periods_movers:
                        subcol = periods_movers_dict[period]
                        if period[-1] != 'b':
                            A_sum[period] += A_cts[col][period][L] * jdata.loc[:, subcol]
                            S_sum_sq[period] += (S_cts[col][period] ** 2)[L]
                        else:
                            A_sum[period] += A_cts[col][period] * jdata.loc[:, subcol]
                            S_sum_sq[period] += S_cts[col][period] ** 2
                else:
                    ## Non-worker-interaction ##
                    for period in periods_movers:
                        subcol = periods_movers_dict[period]
                        A_sum[period] += A_cts[col][period] * jdata.loc[:, subcol]
                        S_sum_sq[period] += S_cts[col][period] ** 2

        Y2 = rng.normal( \
            loc=A_sum['2ma'] + A_sum['2mb'], \
            scale=np.sqrt(S_sum_sq['2ma']), \
            size=nmi)
        Y1 = rng.normal( \
            loc=A_sum['12'] + R12 * (Y2 - A_sum['2ma']), \
            scale=np.sqrt(S_sum_sq['12']), \
            size=nmi)
        Y3 = rng.normal( \
            loc=A_sum['3ma'] + A_sum['3mb'] + R32m * (Y2 - A_sum['2ma'] - A_sum['2mb']), \
            scale=np.sqrt(S_sum_sq['3ma']), \
            size=nmi)
        Y4 = rng.normal( \
            loc=A_sum['43'] + R43 * (Y3 - A_sum['3ma']), \
            scale=np.sqrt(S_sum_sq['43']), \
            size=nmi)
    else:
        #### No control variables ####
        S_sum = {period:
                    S[period][L, G[periods_movers_dict[period]]]
                        if period[-1] != 'b' else
                    S[period][G[periods_movers_dict[period]]]
                for period in periods_movers}

        Y2 = rng.normal( \
            loc=A_sum['2ma'] + A_sum['2mb'], \
            scale=S_sum['2ma'], \
            size=nmi)
        Y1 = rng.normal( \
            loc=A_sum['12'] + R12 * (Y2 - A_sum['2ma']), \
            scale=S_sum['12'], \
            size=nmi)
        Y3 = rng.normal( \
            loc=A_sum['3ma'] + A_sum['3mb'] + R32m * (Y2 - A_sum['2ma'] - A_sum['2mb']), \
            scale=S_sum['3ma'], \
            size=nmi)
        Y4 = rng.normal( \
            loc=A_sum['43'] + R43 * (Y3 - A_sum['3ma']), \
            scale=S_sum['43'], \
            size=nmi)

    return (Y1, Y2, Y3, Y4)

def _simulate_wages_stayers(sdata, L, dynamic_blm_model=None, A=None, S=None, R12=None, R43=None, R32s=None, A_cat=None, S_cat=None, A_cts=None, S_cts=None, controls_dict=None, cat_cols=None, cts_cols=None, rng=None, **kwargs):
    '''
    Simulate wages for stayers.

    Arguments:
        sdata (BipartitePandas DataFrame): stayers data
        L (NumPy Array): worker types for stayers
        dynamic_blm_model (DynamicBLMModel or None): Dynamic BLM model with estimated parameter values; None if other parameters are not None
        A (dict of NumPy Arrays or None): dictionary linking periods to the mean of fixed effects in that period; None if dynamic_blm_model is not None
        S (dict of NumPy Arrays or None): dictionary linking periods to the standard deviation of fixed effects in that period; None if dynamic_blm_model is not None
        R12 (float or None): persistence parameter between periods 1 and 2; None if dynamic_blm_model is not None
        R43 (float or None): persistence parameter between periods 3 and 4; None if dynamic_blm_model is not None
        R32s (float or None): persistence parameter between periods 2 and 3 for stayers; None if dynamic_blm_model is not None
        A_cat (dict of dicts of NumPy Arrays or None): dictionary linking each categorical control column name to a dictionary linking periods to the mean of fixed effects in that period; None if dynamic_blm_model is not None
        S_cat (dict of dicts of NumPy Arrays or None): dictionary linking each categorical control column name to a dictionary linking periods to the standard deviation of fixed effects in that period; None if dynamic_blm_model is not None
        A_cts (dict of dicts of NumPy Arrays or None): dictionary linking each continuous control column name to a dictionary linking periods to the mean of fixed effects in that period; None if dynamic_blm_model is not None
        S_cts (dict of dicts of NumPy Arrays or None): dictionary linking each continuous control column name to a dictionary linking periods to the standard deviation of fixed effects in that period; None if dynamic_blm_model is not None
        controls_dict (dict or None): dictionary of all control variables; None if no controls
        cat_cols (list or None): list of categorical controls; None if no categorical controls
        cts_cols (list or None): list of continuous controls; None if no continuous controls
        rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        kwargs (dict): keyword arguments

    Returns:
        (tuple of NumPy Arrays): wages for stayers in all four periods
    '''
    if rng is None:
        rng = np.random.default_rng(None)

    # Define variables
    periods_stayers = ['12', '43', '2ma', '3ma', '2s', '3s']
    first_periods = ['12', '2ma', '3mb', '2s']
    periods_stayers_dict = {period: 0 if period in first_periods else 1 for period in periods_stayers}

    # Unpack values
    nsi = len(sdata)
    G = sdata.loc[:, 'g1'].to_numpy()

    if dynamic_blm_model is not None:
        # Unpack DynamicBLMModel
        A = dynamic_blm_model.A
        S = dynamic_blm_model.S
        R12 = dynamic_blm_model.R12
        R43 = dynamic_blm_model.R43
        R32s = dynamic_blm_model.R32s
        A_cat = dynamic_blm_model.A_cat
        S_cat = dynamic_blm_model.S_cat
        A_cts = dynamic_blm_model.A_cts
        S_cts = dynamic_blm_model.S_cts
        controls_dict = dynamic_blm_model.controls_dict
        cat_cols = dynamic_blm_model.cat_cols
        cts_cols = dynamic_blm_model.cts_cols

    ## Draw wages ##
    A_sum = {period:
                A[period][L, G]
                    if period[-1] != 'b' else
                A[period][G]
            for period in periods_stayers}
    if (controls_dict is not None) and (len(controls_dict) > 0):
        #### Simulate control variable wages ####
        S_sum_sq = {period:
                        (S[period] ** 2)[L, G]
                            if period[-1] != 'b' else
                        (S[period] ** 2)[G]
                    for period in periods_stayers}
        if cat_cols is None:
            cat_cols = []
        if cts_cols is None:
            cts_cols = []
        for i, col in enumerate(cat_cols + cts_cols):
            # Get subcolumns associated with col
            subcols = to_list(sdata.col_reference_dict[col])
            n_subcols = len(subcols)
            if n_subcols == 1:
                # If column is constant over time
                subcols = [subcols[0], subcols[0]]
            elif n_subcols == 4:
                # If column can change over time
                subcols = [subcols[0], subcols[3]]
            else:
                raise NotImplementedError(f'Column names must have either one or four associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
            if i < len(cat_cols):
                ### Categorical ###
                if controls_dict[col]['worker_type_interaction']:
                    ## Worker-interaction ##
                    for period in periods_stayers:
                        subcol = periods_stayers_dict[period]
                        A_sum[period] += A_cat[col][period][L, sdata.loc[:, subcol]]
                        S_sum_sq[period] += (S_cat[col][period] ** 2)[L, sdata.loc[:, subcol]]
                else:
                    ## Non-worker-interaction ##
                    for period in periods_stayers:
                        subcol = periods_stayers_dict[period]
                        A_sum[period] += A_cat[col][period][sdata.loc[:, subcol]]
                        S_sum_sq[period] += (S_cat[col][period] ** 2)[sdata.loc[:, subcol]]
            else:
                ### Continuous ###
                if controls_dict[col]['worker_type_interaction']:
                    ## Worker-interaction ##
                    for period in periods_stayers:
                        subcol = periods_stayers_dict[period]
                        A_sum[period] += A_cts[col][period][L] * sdata.loc[:, subcol]
                        S_sum_sq[period] += (S_cts[col][period] ** 2)[L]
                else:
                    ## Non-worker-interaction ##
                    for period in periods_stayers:
                        subcol = periods_stayers_dict[period]
                        A_sum[period] += A_cts[col][period] * sdata.loc[:, subcol]
                        S_sum_sq[period] += S_cts[col][period] ** 2

        Y2 = rng.normal( \
            loc=A_sum['2s'], \
            scale=np.sqrt(S_sum_sq['2s']), \
            size=nsi)
        Y1 = rng.normal( \
            loc=A_sum['12'] + R12 * (Y2 - A_sum['2ma']), \
            scale=np.sqrt(S_sum_sq['12']), \
            size=nsi)
        Y3 = rng.normal( \
            loc=A_sum['3s'] + R32s * (Y2 - A_sum['2s']), \
            scale=np.sqrt(S_sum_sq['3s']), \
            size=nsi)
        Y4 = rng.normal( \
            loc=A_sum['43'] + R43 * (Y3 - A_sum['3ma']), \
            scale=np.sqrt(S_sum_sq['43']), \
            size=nsi)
    else:
        #### No control variables ####
        S_sum = {period:
                    S[period][L, G]
                        if period[-1] != 'b' else
                    S[period][G]
                for period in periods_stayers}

        Y2 = rng.normal( \
            loc=A_sum['2s'], \
            scale=S_sum['2s'], \
            size=nsi)
        Y1 = rng.normal( \
            loc=A_sum['12'] + R12 * (Y2 - A_sum['2ma']), \
            scale=S_sum['12'], \
            size=nsi)
        Y3 = rng.normal( \
            loc=A_sum['3s'] + R32s * (Y2 - A_sum['2s']), \
            scale=S_sum['3s'], \
            size=nsi)
        Y4 = rng.normal( \
            loc=A_sum['43'] + R43 * (Y3 - A_sum['3ma']), \
            scale=S_sum['43'], \
            size=nsi)

    return (Y1, Y2, Y3, Y4)

class SimDynamicBLM:
    '''
    Class for simulating bipartite dynamic BLM networks of firms and workers.

    Arguments:
        sim_params (ParamsDict): dictionary of parameters for simulating data. Run tw.sim_dynamic_blm_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.sim_dynamic_blm_params().
    '''

    def __init__(self, sim_params=None):
        if sim_params is None:
            sim_params = sim_dynamic_blm_params()
        # Store parameters
        self.params = sim_params
        nl, nk, NNm, NNs = sim_params.get_multiple(('nl', 'nk', 'NNm', 'NNs'))

        if NNm is None:
            self.NNm = 10 * np.ones(shape=(nk, nk), dtype=int)
        else:
            self.NNm = NNm
        if NNs is None:
            self.NNs = 10 * np.ones(shape=nk, dtype=int)
        else:
            self.NNs = NNs

        # rho
        self.R12, self.R43, self.R32m, self.R32s = sim_params.get_multiple(('R12', 'R43', 'R32m', 'R32s'))

        # Periods
        self.all_periods = ['12', '43', '2ma', '3ma', '2mb', '3mb', '2s', '3s']
        self.periods_movers = ['12', '43', '2ma', '3ma', '2mb', '3mb']
        self.periods_stayers = ['12', '43', '2ma', '3ma', '2s', '3s']
        first_periods = ['12', '2ma', '3mb', '2s']
        second_periods = ['43', '2mb', '3ma', '3s']
        self.periods_dict = {period: 0 if period in first_periods else 1 for period in self.all_periods}

        ## Unpack control variable parameters ##
        cat_dict = sim_params['categorical_controls']
        cts_dict = sim_params['continuous_controls']
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

    def _sort_A(self, A):
        '''
        Sort A for each period by that period's cluster means.

        Arguments:
            A (dict of NumPy Arrays): dictionary linking periods to the mean of fixed effects in that period

        Returns:
            (dict of NumPy Arrays): sorted arrays
        '''
        # Extract parameters
        nl, strictly_monotone_A = self.params.get_multiple(('nl', 'strictly_monotone_A'))

        if strictly_monotone_A:
            ## Make A1 and A2 monotone by worker type ##
            for l in range(nl):
                for period in self.all_periods:
                    if period[-1] != 'b':
                        A[period][l, :] = np.sort(A[period][l, :], axis=0)

        # A_sum = A1 + A2

        ## Sort worker effects ##
        worker_effect_order = np.mean(A['12'], axis=1).argsort()
        for period in self.all_periods:
            if period[-1] != 'b':
                A[period] = A[period][worker_effect_order, :]

        if not strictly_monotone_A:
            ## Sort firm effects ##
            firm_effect_order = np.mean(A['12'], axis=0).argsort()
            for period in self.all_periods:
                if period[-1] != 'b':
                    A[period] = A[period][:, firm_effect_order]
                else:
                    A[period] = A[period][firm_effect_order]

        return A

    def _gen_params(self, rng=None):
        '''
        Generate parameter values to use for simulating bipartite BLM data.

        Arguments:
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (dict of dicts of NumPy Arrays): keys are 'A', 'S', 'pk1', 'pk0', 'A_cat', 'S_cat', 'A_cts', and 'S_cts'. 'A' includes means and 'S' includes standard deviations. 'cat' indicates a categorical control variable and 'cts' indicates a continuous control variable. Each of A/A_cat/A_cts/S/S_cat/S_cts has each period as keys, and each period links to the parameter values for that period. 'pk1' gives the probability of being at each combination of firm types for movers and 'pk0' gives the probability of being at each firm type for stayers.
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        params = self.params
        nl, nk = params.get_multiple(('nl', 'nk'))
        all_periods = self.all_periods
        pk1_prior, pk0_prior = params.get_multiple(('pk1_prior', 'pk0_prior'))
        controls_dict, cat_cols, cts_cols = self.controls_dict, self.cat_cols, self.cts_cols
        stationary_firm_type_variation, stationary_A, stationary_S = params.get_multiple(('stationary_firm_type_variation', 'stationary_A', 'stationary_S'))
        endogeneity, state_dependence = params.get_multiple(('endogeneity', 'state_dependence'))
        dims = self.dims

        #### Draw parameters ####
        A = {}
        S = {}
        for period in all_periods:
            if params['linear_additive']:
                if period[-1] != 'b':
                    alpha_t = rng.normal( \
                        loc=params[f'a{period}_mu'] / 2, \
                        scale=params[f'a{period}_sig'] / np.sqrt(2), \
                        size=nl)
                    psi_t = rng.normal( \
                        loc=params[f'a{period}_mu'] / 2, \
                        scale=params[f'a{period}_sig'] / np.sqrt(2), \
                        size=nk)
                    A[period] = alpha_t[:, None] + psi_t
                else:
                    alpha_t = rng.normal( \
                        loc=params[f'a{period}_mu'] / 2, \
                        scale=params[f'a{period}_sig'] / np.sqrt(2), \
                        size=1)
                    psi_t = rng.normal( \
                        loc=params[f'a{period}_mu'] / 2, \
                        scale=params[f'a{period}_sig'] / np.sqrt(2), \
                        size=nk)
                    A[period] = alpha_t + psi_t
            else:
                if period[-1] != 'b':
                    A[period] = rng.normal( \
                        loc=params[f'a{period}_mu'], \
                        scale=params[f'a{period}_sig'], \
                        size=dims)
                else:
                    A[period] = rng.normal( \
                        loc=params[f'a{period}_mu'], \
                        scale=params[f'a{period}_sig'], \
                        size=nk)
            if period[-1] != 'b':
                S[period] = rng.uniform( \
                    low=params[f's{period}_low'], \
                    high=params[f's{period}_high'], \
                    size=dims)
            else:
                S[period] = rng.uniform( \
                    low=params[f's{period}_low'], \
                    high=params[f's{period}_high'], \
                    size=nk)

        # Model for p(K | l, l') for movers
        if pk1_prior is None:
            pk1_prior = np.ones(nl)
        pk1 = rng.dirichlet(alpha=pk1_prior, size=nk ** 2)
        # Model for p(K | l, l') for stayers
        if pk0_prior is None:
            pk0_prior = np.ones(nl)
        pk0 = rng.dirichlet(alpha=pk0_prior, size=nk)

        ## Sort parameters ##
        A = self._sort_A(A)

        ## Impose constraints ##
        if stationary_A:
            for period in all_periods[1:]:
                A[period] = A['12']

        if stationary_S:
            for period in all_periods[1:]:
                S[period] = S['12']

        if stationary_firm_type_variation:
            for period in all_periods[1:]:
                if period[-1] != 'b':
                    A[period] = (np.mean(A[period], axis=1) + A['12'].T - np.mean(A['12'], axis=1)).T

        # Normalize 2mb and 3mb #
        min_firm_type = np.mean(A['12'], axis=0).argsort()[0]
        A['2mb'] -= A['2mb'][min_firm_type]
        A['3mb'] -= A['3mb'][min_firm_type]

        if not endogeneity:
            A['2mb'][:] = 0
            S['2mb'][:] = 0

        if not state_dependence:
            A['3mb'][:] = 0
            S['3mb'][:] = 0

        ### Control variables ###
        ## Categorical ##
        A_cat = {
            col: {
                period:
                    rng.normal(loc=controls_dict[col][f'a{period}_mu'], scale=controls_dict[col][f'a{period}_sig'], size=(nl, controls_dict[col]['n']))
                        if (controls_dict[col]['worker_type_interaction'] and (period[-1] != 'b')) else
                    rng.normal(loc=controls_dict[col][f'a{period}_mu'], scale=controls_dict[col][f'a{period}_sig'], size=controls_dict[col]['n'])
                for period in all_periods
            }
            for col in cat_cols
        }
        S_cat = {
            col: {
                period:
                    rng.uniform(low=controls_dict[col][f's{period}_low'], high=controls_dict[col][f's{period}_high'], size=(nl, controls_dict[col]['n']))
                        if (controls_dict[col]['worker_type_interaction'] and (period[-1] != 'b')) else
                    rng.uniform(low=controls_dict[col][f's{period}_low'], high=controls_dict[col][f's{period}_high'], size=controls_dict[col]['n'])
                for period in all_periods
            }
            for col in cat_cols
        }
        ## Impose constraints ##
        # Stationary #
        for col in cat_cols:
            for period in all_periods[1:]:
                if controls_dict[col]['stationary_A']:
                    A_cat[col][period] = A_cat[col]['12']
                if controls_dict[col]['stationary_S']:
                    S_cat[col][period] = S_cat[col]['12']
        # Stationary firm type variation #
        for col in cat_cols:
            for period in all_periods[1:]:
                if controls_dict[col]['stationary_firm_type_variation'] and (period[-1] != 'b'):
                    if controls_dict[col]['worker_type_interaction']:
                        A_cat[col][period] = (np.mean(A_cat[col][period], axis=1) + A_cat[col]['12'].T - np.mean(A_cat[col]['12'], axis=1)).T
                    else:
                        A_cat[col][period] = np.mean(A_cat[col][period]) + A_cat[col]['12'] - np.mean(A_cat[col]['12'])
        # Normalize 2mb and 3mb #
        for col in cat_cols:
            A_cat[col]['2mb'] -= A_cat[col]['2mb'][min_firm_type]
            A_cat[col]['3mb'] -= A_cat[col]['3mb'][min_firm_type]
        # Endogeneity and state dependence
        for col in cat_cols:
            if not endogeneity:
                A_cat[col]['2mb'][:] = 0
                S_cat[col]['2mb'][:] = 0
            if not state_dependence:
                A_cat[col]['3mb'][:] = 0
                S_cat[col]['3mb'][:] = 0
        ## Continuous ##
        A_cts = {
            col: {
                period:
                    rng.normal(loc=controls_dict[col][f'a{period}_mu'], scale=controls_dict[col][f'a{period}_sig'], size=nl)
                        if (controls_dict[col]['worker_type_interaction'] and (period[-1] != 'b')) else
                    rng.normal(loc=controls_dict[col][f'a{period}_mu'], scale=controls_dict[col][f'a{period}_sig'], size=1)
                for period in all_periods
            }
            for col in cts_cols
        }
        S_cts = {
            col: {
                period:
                    rng.uniform(low=controls_dict[col][f's{period}_low'], high=controls_dict[col][f's{period}_high'], size=nl)
                        if (controls_dict[col]['worker_type_interaction'] and (period[-1] != 'b')) else
                    rng.uniform(low=controls_dict[col][f's{period}_low'], high=controls_dict[col][f's{period}_high'], size=1)
                for period in all_periods
            }
            for col in cts_cols
        }
        ## Impose constraints ##
        # Stationary #
        for col in cts_cols:
            for period in all_periods[1:]:
                if controls_dict[col]['stationary_A']:
                    A_cts[col][period] = A_cts[col]['12']
                if controls_dict[col]['stationary_S']:
                    S_cts[col][period] = S_cts[col]['12']
        # Stationary firm type variation #
        for col in cts_cols:
            for period in all_periods[1:]:
                if controls_dict[col]['stationary_firm_type_variation'] and (period[-1] != 'b'):
                    A_cts[col][period] = np.mean(A_cts[col][period]) + A_cts[col]['12'] - np.mean(A_cts[col]['12'])
        # Endogeneity and state dependence
        for col in cts_cols:
            if not endogeneity:
                A_cts[col]['2mb'] = 0
                S_cts[col]['2mb'] = 0
            if not state_dependence:
                A_cts[col]['3mb'] = 0
                S_cts[col]['3mb'] = 0

        return {'A': A, 'S': S, 'pk1': pk1, 'pk0': pk0, 'A_cat': A_cat, 'S_cat': S_cat, 'A_cts': A_cts, 'S_cts': S_cts}

    def _simulate_movers(self, pk1, rng=None):
        '''
        Simulate data for movers (simulates firm types, not firms).

        Arguments:
            pk1 (NumPy Array): probability of being at each combination of firm types for movers
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (Pandas DataFrame): data for movers (y1/y2/y3/y4: wage; g1/g2/g3/g4: firm type; l: worker type; m=1: mover indicator)
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
        G1, G2 = simblm._simulate_firm_classes_movers(nk=nk, NNm=NNm)

        ## Worker types ##
        L = simblm._simulate_worker_types_movers(
            nl=nl, nk=nk, NNm=NNm, G1=G1, G2=G2, pk1=pk1,
            qi_cum=None, simulating_data=True, rng=rng
        )

        ## Control variables ##
        A1_cat_draws, A2_cat_draws, A3_cat_draws, A4_cat_draws, \
            A1_cts_draws, A2_cts_draws, A3_cts_draws, A4_cts_draws \
            = simblm._simulate_controls_movers(nmi=nmi, cat_dict=cat_dict, cts_cols=cts_cols, dynamic=True, rng=rng)

        ## Convert to DataFrame ##
        return DataFrame(data={'y1': np.zeros(nmi), 'y2': np.zeros(nmi), 'y3': np.zeros(nmi), 'y4': np.zeros(nmi), 'g1': G1, 'g2': G1.copy(), 'g3': G2, 'g4': G2.copy(), 'l': L, 'm': np.ones(nmi, dtype=int), **A1_cat_draws, **A2_cat_draws, **A3_cat_draws, **A4_cat_draws, **A1_cts_draws, **A2_cts_draws, **A3_cts_draws, **A4_cts_draws})

    def _simulate_stayers(self, pk0, rng=None):
        '''
        Simulate data for stayers (simulates firm types, not firms).

        Arguments:
            pk0 (NumPy Array): probability of being at each firm type for stayers
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (Pandas DataFrame): data for stayers (y1/y2/y3/y4: wage; g1/g2/g3/g4: firm type; l: worker type; m=0: stayer indicator)
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
        G = simblm._simulate_firm_classes_stayers(nk=nk, NNs=NNs)

        ## Worker types ##
        L = simblm._simulate_worker_types_stayers(
            nl=nl, nk=nk, NNs=NNs, G=G, pk0=pk0,
            qi_cum=None, simulating_data=True, rng=rng
        )

        ## Control variables ##
        A1_cat_draws, A2_cat_draws, A3_cat_draws, A4_cat_draws, \
            A1_cts_draws, A2_cts_draws, A3_cts_draws, A4_cts_draws \
            = simblm._simulate_controls_stayers(nsi=nsi, cat_dict=cat_dict, cts_cols=cts_cols, dynamic=True, rng=rng)

        ## Convert to DataFrame ##
        return DataFrame(data={'y1': np.zeros(nsi), 'y2': np.zeros(nsi), 'y3': np.zeros(nsi), 'y4': np.zeros(nsi), 'g1': G, 'g2': G.copy(), 'g3': G.copy(), 'g4': G.copy(), 'l': L, 'm': np.zeros(nsi, dtype=int), **A1_cat_draws, **A2_cat_draws, **A3_cat_draws, **A4_cat_draws, **A1_cts_draws, **A2_cts_draws, **A3_cts_draws, **A4_cts_draws})

    def simulate(self, return_parameters=False, rng=None):
        '''
        Simulate data (movers and stayers). All firms have the same expected size. Columns are as follows: y1/y2/y3/y4=wage; j1/j2/j3/j4=firm id; g1/g2/g3/g4=firm type; l=worker type.

        Arguments:
            return_parameters (bool): if True, return tuple of (simulated data, simulated parameters); otherwise, return only simulated data
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (dict or tuple of dicts): sim_data gives {'jdata': movers BipartiteDataFrame, 'sdata': stayers BipartiteDataFrame}, while sim_params gives keys 'A', 'S', 'pk1', 'pk0', 'R12', 'R43', 'R32m', 'R32s', 'A_cat', 'S_cat', 'A_cts', and 'S_cts' ('A' gives mean; 'S' gives standard deviation; 'cat' gives categorical controls; 'cts' gives continuous controls; 'pk1' gives the probability of being at each combination of firm types for movers; 'pk0' gives the probability of being at each firm type for stayers; and the 'R' terms are the persistance parameters). If return_parameters=True, returns (sim_data, sim_params); if return_parameters=False, returns sim_data.
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        ## Simulate parameters ##
        sim_params = self._gen_params(rng=rng)

        ## Simulate movers and stayers data ##
        jdata = self._simulate_movers(pk1=sim_params['pk1'], rng=rng)
        sdata = self._simulate_stayers(pk0=sim_params['pk0'], rng=rng)

        ## Simulate firms ##
        simblm._simulate_firms(jdata, sdata, self.params['firm_size'], dynamic=True, rng=rng)

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
        Y1m, Y2m, Y3m, Y4m = _simulate_wages_movers(jdata, L=jdata.loc[:, 'l'], R12=self.R12, R43=self.R43, R32m=self.R32m, controls_dict=self.controls_dict, cat_cols=self.cat_cols, cts_cols=self.cts_cols, rng=rng, **sim_params)
        Y1s, Y2s, Y3s, Y4s = _simulate_wages_stayers(sdata, L=sdata.loc[:, 'l'], R12=self.R12, R43=self.R43, R32s=self.R32s, controls_dict=self.controls_dict, cat_cols=self.cat_cols, cts_cols=self.cts_cols, rng=rng, **sim_params)
        jdata.loc[:, 'y1'], jdata.loc[:, 'y2'], jdata.loc[:, 'y3'], jdata.loc[:, 'y4'] = (Y1m, Y2m, Y3m, Y4m)
        sdata.loc[:, 'y1'], sdata.loc[:, 'y2'], sdata.loc[:, 'y3'], sdata.loc[:, 'y4'] = (Y1s, Y2s, Y3s, Y4s)

        ## Combine into dictionary ##
        sim_data = {'jdata': jdata, 'sdata': sdata}

        if return_parameters:
            sim_params['R12'] = self.R12
            sim_params['R43'] = self.R43
            sim_params['R32m'] = self.R32m
            sim_params['R32s'] = self.R32s
            return sim_data, sim_params
        return sim_data
