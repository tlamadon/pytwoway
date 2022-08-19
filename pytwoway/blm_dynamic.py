'''
Implement the dynamic, 4-period non-linear estimator from Bonhomme, Lamadon, & Manresa.
'''
from tqdm.auto import tqdm, trange
import copy
import warnings
import itertools
try:
    from multiprocess import Pool
except ImportError:
    from multiprocessing import Pool
import numpy as np
import pandas as pd
# from scipy.special import logsumexp
from scipy.sparse import csc_matrix
from scipy.optimize import minimize as opt
from matplotlib import pyplot as plt
import bipartitepandas as bpd
from bipartitepandas.util import ParamsDict, to_list, HiddenPrints # , _is_subtype
import pytwoway as tw
from pytwoway import constraints as cons
from pytwoway.util import DxSP, DxM, diag_of_sp_prod, jitter_scatter, logsumexp, lognormpdf, fast_lognormpdf

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
_blm_dynamic_params_default = ParamsDict({
    ## Class parameters ##
    'nl': (6, 'type_constrained', (int, _gteq1),
        '''
            (default=6) Number of worker types.
        ''', '>= 1'),
    'nk': (None, 'type_constrained_none', (int, _gteq1),
        '''
            (default=None) Number of firm types. None will raise an error when running the estimator.
        ''', '>= 1'),
    'categorical_controls': (None, 'dict_of_type_none', ParamsDict,
        '''
            (default=None) Dictionary linking column names to instances of tw.categorical_control_dynamic_params(). Each instance specifies a new categorical control variable and how its starting values should be generated. Run tw.categorical_control_dynamic_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
        ''', None),
    'continuous_controls': (None, 'dict_of_type_none', ParamsDict,
        '''
            (default=None) Dictionary linking column names to instances of tw.continuous_control_dynamic_params(). Each instance specifies a new continuous control variable and how its starting values should be generated. Run tw.continuous_control_dynamic_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
        ''', None),
    'primary_period': ('first', 'set', ['first', 'second', 'all'],
        '''
            (default='first') Period to normalize and sort over. 'first' uses first period parameters; 'second' uses second period parameters; 'all' uses the average over first and second period parameters.
        ''', None),
    'verbose': (1, 'set', [0, 1, 2, 3],
        '''
            (default=1) If 0, print no output; if 1, print each major step in estimation; if 2, print warnings during estimation; if 3, print likelihoods at each iteration.
        ''', None),
    ## Starting values ##
    'a12_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A12 (mean of fixed effects).
        ''', None),
    'a12_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A12 (mean of fixed effects).
        ''', '>= 0'),
    'a43_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A43 (mean of fixed effects).
        ''', None),
    'a43_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A43 (mean of fixed effects).
        ''', '>= 0'),
    'a2ma_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A2a for movers (mean of fixed effects).
        ''', None),
    'a2ma_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2a for movers (mean of fixed effects).
        ''', '>= 0'),
    'a2mb_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A2b for movers (mean of fixed effects).
        ''', None),
    'a2mb_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2b for movers (mean of fixed effects).
        ''', '>= 0'),
    'a2s_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A2 for stayers (mean of fixed effects).
        ''', None),
    'a2s_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2 for stayers (mean of fixed effects).
        ''', '>= 0'),
    'a3ma_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A3a for movers (mean of fixed effects).
        ''', None),
    'a3ma_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A3a for movers (mean of fixed effects).
        ''', '>= 0'),
    'a3mb_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A3b for movers (mean of fixed effects).
        ''', None),
    'a3mb_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A3b for movers (mean of fixed effects).
        ''', '>= 0'),
    'a3s_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A3 for stayers (mean of fixed effects).
        ''', None),
    'a3s_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A3 for stayers (mean of fixed effects).
        ''', '>= 0'),
    's12_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S12 (standard deviation of fixed effects).
        ''', '>= 0'),
    's12_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S12 (standard deviation of fixed effects).
        ''', '>= 0'),
    's43_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S43 (standard deviation of fixed effects).
        ''', '>= 0'),
    's43_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S43 (standard deviation of fixed effects).
        ''', '>= 0'),
    's2m_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2 for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2m_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S2 for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2s_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2 for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2s_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S2 for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3m_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S3 for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3m_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S3 for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3s_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S3 for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3s_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S3 for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    'pk1_prior': (None, 'array_of_type_constrained_none', (('float', 'int'), _min_gt0),
        '''
            (default=None) Dirichlet prior for pk1 (probability of being at each combination of firm types for movers). Must have length nl. None is equivalent to np.ones(nl).
        ''', 'min > 0'),
    # 'pk0_prior': (None, 'array_of_type_constrained_none', (('float', 'int'), _min_gt0),
    #     '''
    #         (default=None) Dirichlet prior for pk0 (probability of being at each firm type for stayers). Must have length nl. None is equivalent to np.ones(nl).
    #     ''', 'min > 0'),
    ## fit_movers() and fit_stayers() parameters ##
    'normalize': (True, 'type', bool,
        '''
            (default=True) If True, normalize estimator during estimation if there are categorical controls with constraints. With particular constraints, the estimator may be identified without normalization, in which case this should be set to False.
        ''', None),
    'return_qi': (False, 'type', bool,
        '''
            (default=False) If True, return qi matrix after first loop.
        ''', None),
    # fit_movers() parameters ##
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
            (default=True) If False, do not update A12/A43/A2ma/A2mb/A2s/A3ma/A3mb/A3s.
        ''', None),
    'update_s': (True, 'type', bool,
        '''
            (default=True) If False, do not update S12/S43/S2m/S2s/S3m/S3s.
        ''', None),
    'update_pk1': (True, 'type', bool,
        '''
            (default=True) If False, do not update pk1.
        ''', None),
    'cons_a': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on A12/A43/A2ma/A2mb/A2s/A3ma/A3mb/A3s. None is equivalent to [].
        ''', None),
    'cons_s': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on S12/S43/S2m/S2s/S3m/S3s. None is equivalent to [].
        ''', None),
    'cons_a_all': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on A12/A43/A2ma/A2mb/A2s/A3ma/A3mb/A3s plus their categorical and continuous control variants. None is equivalent to [].
        ''', None),
    'cons_s_all': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on S12/S43/S2m/S2s/S3m/S3s plus their categorical and continuous control variants. None is equivalent to [].
        ''', None),
    's_lower_bound': (1e-7, 'type_constrained', ((float, int), _gt0),
        '''
            (default=1e-7) Lower bound on estimated S12/S43/S2m/S2s/S3m/S3s plus their categorical and continuous control variants.
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
    # fit_stayers() parameters ##
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

def blm_dynamic_params(update_dict=None):
    '''
    Dictionary of default blm_dynamic_params. Run tw.blm_dynamic_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict or None): user parameter values; None is equivalent to {}

    Returns:
        (ParamsDict) dictionary of blm_params
    '''
    new_dict = _blm_dynamic_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

_categorical_control_dynamic_params_default = ParamsDict({
    'n': (None, 'type_constrained_none', (int, _gteq2),
        '''
            (default=6) Number of types for the parameter. None will raise an error when running the estimator.
        ''', '>= 2'),
    'a12_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A12_cat (mean of fixed effects).
        ''', None),
    'a12_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A12_cat (mean of fixed effects).
        ''', '>= 0'),
    'a43_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A43_cat (mean of fixed effects).
        ''', None),
    'a43_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A43_cat (mean of fixed effects).
        ''', '>= 0'),
    'a2ma_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A2a_cat for movers (mean of fixed effects).
        ''', None),
    'a2ma_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2a_cat for movers (mean of fixed effects).
        ''', '>= 0'),
    'a2mb_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A2b_cat for movers (mean of fixed effects).
        ''', None),
    'a2mb_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2b_cat for movers (mean of fixed effects).
        ''', '>= 0'),
    'a2s_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A2_cat for stayers (mean of fixed effects).
        ''', None),
    'a2s_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2_cat for stayers (mean of fixed effects).
        ''', '>= 0'),
    'a3ma_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A3a_cat for movers (mean of fixed effects).
        ''', None),
    'a3ma_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A3a_cat for movers (mean of fixed effects).
        ''', '>= 0'),
    'a3mb_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A3b_cat for movers (mean of fixed effects).
        ''', None),
    'a3mb_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A3b_cat for movers (mean of fixed effects).
        ''', '>= 0'),
    'a3s_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A3_cat for stayers (mean of fixed effects).
        ''', None),
    'a3s_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A3_cat for stayers (mean of fixed effects).
        ''', '>= 0'),
    's12_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S12_cat (standard deviation of fixed effects).
        ''', '>= 0'),
    's12_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S12_cat (standard deviation of fixed effects).
        ''', '>= 0'),
    's43_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S43_cat (standard deviation of fixed effects).
        ''', '>= 0'),
    's43_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S43_cat (standard deviation of fixed effects).
        ''', '>= 0'),
    's2m_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2_cat for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2m_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2_cat for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2s_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2_cat for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2s_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2_cat for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3m_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S3_cat for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3m_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S3_cat for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3s_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S3_cat for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3s_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S3_cat for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    'worker_type_interaction': (False, 'type', bool,
        '''
            (default=False) If True, effect can differ by worker type.
        ''', None),
    'cons_a': (None, 'list_of_type_none', (cons.Linear, cons.LinearAdditive, cons.Monotonic, cons.MonotonicMean, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on A12_cat/A43_cat/A2ma_cat/A2mb_cat/A2s_cat/A3ma_cat/A3mb_cat/A3s_cat. None is equivalent to [].
        ''', None),
    'cons_s': (None, 'list_of_type_none', (cons.Linear, cons.LinearAdditive, cons.Monotonic, cons.MonotonicMean, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on S12_cat/S43_cat/S2m_cat/S2s_cat/S3m_cat/S3s_cat. None is equivalent to [].
        ''', None)
})

def categorical_control_dynamic_params(update_dict=None):
    '''
    Dictionary of default categorical_control_dynamic_params. Run tw.categorical_control_dynamic_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict or None): user parameter values; None is equivalent to {}

    Returns:
        (ParamsDict) dictionary of categorical_control_dynamic_params
    '''
    new_dict = _categorical_control_dynamic_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

_continuous_control_dynamic_params_default = ParamsDict({
    'a12_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A12_cts (mean of fixed effects).
        ''', None),
    'a12_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A12_cts (mean of fixed effects).
        ''', '>= 0'),
    'a43_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A43_cts (mean of fixed effects).
        ''', None),
    'a43_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A43_cts (mean of fixed effects).
        ''', '>= 0'),
    'a2ma_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A2a_cts for movers (mean of fixed effects).
        ''', None),
    'a2ma_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2a_cts for movers (mean of fixed effects).
        ''', '>= 0'),
    'a2mb_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A2b_cts for movers (mean of fixed effects).
        ''', None),
    'a2mb_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2b_cts for movers (mean of fixed effects).
        ''', '>= 0'),
    'a2s_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A2_cts for stayers (mean of fixed effects).
        ''', None),
    'a2s_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2_cts for stayers (mean of fixed effects).
        ''', '>= 0'),
    'a3ma_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A3a_cts for movers (mean of fixed effects).
        ''', None),
    'a3ma_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A3a_cts for movers (mean of fixed effects).
        ''', '>= 0'),
    'a3mb_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A3b_cts for movers (mean of fixed effects).
        ''', None),
    'a3mb_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A3b_cts for movers (mean of fixed effects).
        ''', '>= 0'),
    'a3s_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A3_cts for stayers (mean of fixed effects).
        ''', None),
    'a3s_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A3_cts for stayers (mean of fixed effects).
        ''', '>= 0'),
    's12_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S12_cts (standard deviation of fixed effects).
        ''', '>= 0'),
    's12_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S12_cts (standard deviation of fixed effects).
        ''', '>= 0'),
    's43_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S43_cts (standard deviation of fixed effects).
        ''', '>= 0'),
    's43_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S43_cts (standard deviation of fixed effects).
        ''', '>= 0'),
    's2m_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2_cts for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2m_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2_cts for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2s_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2_cts for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's2s_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2_cts for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3m_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S3_cts for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3m_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S3_cts for movers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3s_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S3_cts for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    's3s_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S3_cts for stayers (standard deviation of fixed effects).
        ''', '>= 0'),
    'worker_type_interaction': (False, 'type', bool,
        '''
            (default=False) If True, effect can differ by worker type.
        ''', None),
    'cons_a': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on A12_cts/A43_cts/A2ma_cts/A2mb_cts/A2s_cts/A3ma_cts/A3mb_cts/A3s_cts. None is equivalent to [].
        ''', None),
    'cons_s': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects that define constraints on S12_cts/S43_cts/S2m_cts/S2s_cts/S3m_cts/S3s_cts. None is equivalent to [].
        ''', None)
})

def continuous_control_dynamic_params(update_dict=None):
    '''
    Dictionary of default continuous_control_dynamic_params. Run tw.continuous_control_dynamic_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict or None): user parameter values; None is equivalent to {}

    Returns:
        (ParamsDict) dictionary of continuous_control_dynamic_params
    '''
    new_dict = _continuous_control_dynamic_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

def _simulate_types_wages(jdata, sdata, gj, gs, blm_model, reallocate=False, reallocate_jointly=True, reallocate_period='first', wj=None, ws=None, rng=None):
    '''
    Using data and estimated BLM parameters, simulate worker types and wages.

    Arguments:
        jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
        sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
        gj (NumPy Array): mover firm types for both periods
        gs (NumPy Array): stayer firm types for the first period
        blm_model (BLMModel): BLM model with estimated parameters
        reallocate (bool): if True, draw worker type proportions independently of firm type; if False, uses worker type proportions that are conditional on firm type
        reallocate_jointly (bool): if True, worker type proportions take the average over movers and stayers (i.e. all workers use the same type proportions); if False, consider movers and stayers separately
        reallocate_period (str): if 'first', compute type proportions based on first period parameters; if 'second', compute type proportions based on second period parameters; if 'all', compute type proportions based on average over first and second period parameters
        wj (NumPy Array or None): mover weights for both periods; if None, don't weight
        ws (NumPy Array or None): stayer weights for the first period; if None, don't weight
        rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

    Returns:
        (tuple of NumPy Arrays): (yj --> tuple of wages for movers, where first element in first period wages and second element is second period wages; ys --> wages for stayers; Lm --> vector of mover types; Ls --> vector of stayer types)
    '''
    if rng is None:
        rng = np.random.default_rng(None)

    nl, nk, nmi, nsi = blm_model.nl, blm_model.nk, len(jdata), len(sdata)
    A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, pk0 = blm_model.A1, blm_model.A2, blm_model.S1, blm_model.S2, blm_model.A1_cat, blm_model.A2_cat, blm_model.S1_cat, blm_model.S2_cat, blm_model.A1_cts, blm_model.A2_cts, blm_model.S1_cts, blm_model.S2_cts, blm_model.pk1, blm_model.pk0
    controls_dict, cat_cols, cts_cols = blm_model.controls_dict, blm_model.cat_cols, blm_model.cts_cols

    # Correct datatype for gj and gs
    gj = gj.astype(int, copy=False)
    gs = gs.astype(int, copy=False)

    # Unpack weights
    if wj is None:
        wj = np.array([[1, 1]])
    if ws is None:
        ws = 1

    # Worker types
    worker_types = np.arange(nl)

    if reallocate:
        ### Re-allocate ###
        NNm = blm_model.NNm
        NNs = blm_model.NNs
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

    ## Movers ##
    Lm = np.zeros(shape=len(jdata), dtype=int)
    for k1 in range(nk):
        for k2 in range(nk):
            ## Iterate over all firm type combinations a worker can transition between ##
            # Find movers who work at this combination of firm types
            rows_kk = np.where((gj[:, 0] == k1) & (gj[:, 1] == k2))[0]
            ni = len(rows_kk)
            jj = k1 + nk * k2

            # Draw worker types
            Lm[rows_kk] = rng.choice(worker_types, size=ni, replace=True, p=pk1[jj, :])

    A1_sum = A1[Lm, gj[:, 0]]
    A2_sum = A2[Lm, gj[:, 1]]
    S1_sum = S1[Lm, gj[:, 0]]
    S2_sum = S2[Lm, gj[:, 1]]

    if len(controls_dict) > 0:
        #### Simulate control variable wages ####
        S1_sum_sq = S1_sum ** 2
        S2_sum_sq = S2_sum ** 2
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
                    A1_sum += A1_cat[col][Lm, jdata.loc[:, subcol_1]]
                    A2_sum += A2_cat[col][Lm, jdata.loc[:, subcol_2]]
                    S1_sum_sq += S1_cat[col][Lm, jdata.loc[:, subcol_1]] ** 2
                    S2_sum_sq += S2_cat[col][Lm, jdata.loc[:, subcol_2]] ** 2
                else:
                    ## Non-worker-interaction ##
                    A1_sum += A1_cat[col][jdata.loc[:, subcol_1]]
                    A2_sum += A2_cat[col][jdata.loc[:, subcol_2]]
                    S1_sum_sq += S1_cat[col][jdata.loc[:, subcol_1]] ** 2
                    S2_sum_sq += S2_cat[col][jdata.loc[:, subcol_2]] ** 2
            else:
                ### Continuous ###
                if controls_dict[col]['worker_type_interaction']:
                    ## Worker-interaction ##
                    A1_sum += A1_cts[col][Lm] * jdata.loc[:, subcol_1]
                    A2_sum += A2_cts[col][Lm] * jdata.loc[:, subcol_2]
                    S1_sum_sq += S1_cts[col][Lm] ** 2
                    S2_sum_sq += S2_cts[col][Lm] ** 2
                else:
                    ## Non-worker-interaction ##
                    A1_sum += A1_cts[col] * jdata.loc[:, subcol_1]
                    A2_sum += A2_cts[col] * jdata.loc[:, subcol_2]
                    S1_sum_sq += S1_cts[col] ** 2
                    S2_sum_sq += S2_cts[col] ** 2

        Y1 = rng.normal(loc=A1_sum, scale=np.sqrt(S1_sum_sq) / wj[:, 0], size=nmi)
        Y2 = rng.normal(loc=A2_sum, scale=np.sqrt(S2_sum_sq) / wj[:, 1], size=nmi)
        del S1_sum_sq, S2_sum_sq
    else:
        #### No control variables ####
        Y1 = rng.normal(loc=A1_sum, scale=S1_sum / wj[:, 0], size=nmi)
        Y2 = rng.normal(loc=A2_sum, scale=S2_sum / wj[:, 1], size=nmi)
    yj = (Y1, Y2)
    del A1_sum, A2_sum, S1_sum, S2_sum, Y1, Y2

    ## Stayers ##
    Ls = np.zeros(shape=len(sdata), dtype=int)
    for k in range(nk):
        ## Iterate over all firm types a worker can work at ##
        # Find movers who work at this firm type
        rows_k = np.where(gs == k)[0]
        ni = len(rows_k)

        # Draw worker types
        Ls[rows_k] = rng.choice(worker_types, size=ni, replace=True, p=pk0[k, :])

    A1_sum = A1[Ls, gs]
    S1_sum = S1[Ls, gs]

    if len(controls_dict) > 0:
        #### Simulate control variable wages ####
        S1_sum_sq = S1_sum ** 2
        for i, col in enumerate(cat_cols + cts_cols):
            # Get subcolumns associated with col
            subcol_1 = to_list(sdata.col_reference_dict[col])[0]
            if i < len(cat_cols):
                ### Categorical ###
                if controls_dict[col]['worker_type_interaction']:
                    ## Worker-interaction ##
                    A1_sum += A1_cat[col][Ls, sdata.loc[:, subcol_1]]
                    S1_sum_sq += S1_cat[col][Ls, sdata.loc[:, subcol_1]] ** 2
                else:
                    ## Non-worker-interaction ##
                    A1_sum += A1_cat[col][sdata.loc[:, subcol_1]]
                    S1_sum_sq += S1_cat[col][sdata.loc[:, subcol_1]] ** 2
            else:
                ### Continuous ###
                if controls_dict[col]['worker_type_interaction']:
                    ## Worker-interaction ##
                    A1_sum += A1_cts[col][Ls] * sdata.loc[:, subcol_1]
                    S1_sum_sq += S1_cts[col][Ls] ** 2
                else:
                    ## Non-worker-interaction ##
                    A1_sum += A1_cts[col] * sdata.loc[:, subcol_1]
                    S1_sum_sq += S1_cts[col] ** 2

        Y1 = rng.normal(loc=A1_sum, scale=np.sqrt(S1_sum_sq) / ws, size=nsi)
    else:
        #### No control variables ####
        Y1 = rng.normal(loc=A1_sum, scale=S1_sum / ws, size=nsi)
    ys = Y1

    return (yj, ys, Lm, Ls)

class BLMModel:
    '''
    Class for estimating BLM using a single set of starting values.

    Arguments:
        params (ParamsDict): dictionary of parameters for BLM estimation. Run tw.blm_params().describe_all() for descriptions of all valid parameters.
        rhos (tuple of floats): rho values estimated using stayers
        rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
    '''
    def __init__(self, params, rhos, rng=None):
        if rng is None:
            rng = np.random.default_rng(None)

        # Store parameters
        self.params = params.copy()
        params = self.params
        self.rng = rng
        nl, nk = params.get_multiple(('nl', 'nk'))
        # Make sure that nk is specified
        if nk is None:
            raise ValueError(f"tw.blm_params() key 'nk' must be changed from the default value of None.")
        self.nl, self.nk = nl, nk

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
        cat_dict = params['categorical_controls']
        cts_dict = params['continuous_controls']
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
        self.any_controls = len(control_cols) > 0
        # Check if any control variables interact with worker type
        self.any_worker_type_interactions = any([col_dict['worker_type_interaction'] for col_dict in controls_dict.values()])
        # Check if any control variables don't interact with worker type
        self.any_non_worker_type_interactions = any([not col_dict['worker_type_interaction'] for col_dict in controls_dict.values()])

        ## Generate starting values ##
        s_lb = params['s_lower_bound']
        # rho is already computed
        self.R12 = rhos['rho_1']
        self.R43 = rhos['rho_4']
        self.R32m = 0.6
        # We simulate starting values for everything else
        self.A = {
            period:
                rng.normal(loc=params[f'a{period}_mu'], scale=params[f'a{period}_sig'], size=dims)
                    if (period[-1] != 'b') else
                rng.normal(loc=params[f'a{period}_mu'], scale=params[f'a{period}_sig'], size=nk)
            for period in ['12', '43', '2ma', '2mb', '3ma', '3mb', '2s', '3s']
        }
        self.S = {
            period:
                rng.uniform(low=max(params[f's{period}_low'], s_lb), high=params[f's{period}_high'], size=dims)
                    if (period[-1] != 'b') else
                rng.uniform(low=max(params[f's{period}_low'], s_lb), high=params[f's{period}_high'], size=nk)
            for period in ['12', '43', '2ma', '2mb', '3ma', '3mb', '2s', '3s']
        }
        # Model for p(K | l, l') for movers
        if params['pk1_prior'] is None:
            pk1_prior = np.ones(nl)
        self.pk1 = rng.dirichlet(alpha=params['pk1_prior'], size=nk ** 2)
        # Model for p(K | l, l') for stayers
        # if params['pk0_prior'] is None:
        #     pk0_prior = np.ones(nl)
        # self.pk0 = rng.dirichlet(alpha=params['pk0_prior'], size=nk)
        self.pk0 = np.ones((nk, nl)) / nl

        ### Control variables ###
        ## Categorical ##
        self.A_cat = {
            col: {
                period:
                    rng.normal(loc=controls_dict[col][f'a{period}_mu'], scale=controls_dict[col][f'a{period}_sig'], size=(nl, controls_dict[col]['n']))
                        if (controls_dict[col]['worker_type_interaction'] and (period[-1] != 'b')) else
                    rng.normal(loc=controls_dict[col][f'a{period}_mu'], scale=controls_dict[col][f'a{period}_sig'], size=controls_dict[col]['n'])
                for period in ['12', '43', '2ma', '2mb', '3ma', '3mb', '2s', '3s']
            }
            for col in cat_cols
        }
        self.S_cat = {
            col: {
                period:
                    rng.uniform(low=max(controls_dict[col][f's{period}_low'], s_lb), high=controls_dict[col][f's{period}_high'], size=(nl, controls_dict[col]['n']))
                        if (controls_dict[col]['worker_type_interaction'] and (period[-1] != 'b')) else
                    rng.uniform(low=max(controls_dict[col][f's{period}_low'], s_lb), high=controls_dict[col][f's{period}_high'], size=controls_dict[col]['n'])
                for period in ['12', '43', '2ma', '2mb', '3ma', '3mb', '2s', '3s']
            }
            for col in cat_cols
        }
        ## Continuous ##
        self.A_cts = {
            col: {
                period:
                    rng.normal(loc=controls_dict[col][f'a{period}_mu'], scale=controls_dict[col][f'a{period}_sig'], size=nl)
                        if (controls_dict[col]['worker_type_interaction'] and (period[-1] != 'b')) else
                    rng.normal(loc=controls_dict[col][f'a{period}_mu'], scale=controls_dict[col][f'a{period}_sig'], size=1)
                for period in ['12', '43', '2ma', '2mb', '3ma', '3mb', '2s', '3s']
            }
            for col in cts_cols
        }
        self.S_cts = {
            col: {
                period:
                    rng.uniform(low=max(controls_dict[col][f's{period}_low'], s_lb), high=controls_dict[col][f's{period}_high'], size=nl)
                        if (controls_dict[col]['worker_type_interaction'] and (period[-1] != 'b')) else
                    rng.uniform(low=max(controls_dict[col][f's{period}_low'], s_lb), high=controls_dict[col][f's{period}_high'], size=1)
                for period in ['12', '43', '2ma', '2mb', '3ma', '3mb', '2s', '3s']
            }
            for col in cts_cols
        }

    def _min_firm_type(self, A1, A2):
        '''
        Find the lowest firm type.

        Arguments:
            A1 (NumPy Array): mean of fixed effects in the first period
            A2 (NumPy Array): mean of fixed effects in the second period

        Returns:
            (int): lowest firm type
        '''
        params = self.params

        # Compute parameters from primary period
        if params['primary_period'] == 'first':
            A_mean = A1
        elif params['primary_period'] == 'second':
            A_mean = A2
        elif params['primary_period'] == 'all':
            A_mean = (A1 + A2) / 2

        # Return lowest firm type
        return np.mean(A_mean, axis=0).argsort()[0]

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
            # Check that there is at least one constraint on a categorical column
            any_cat_constraints = (params['cons_a_all'] is not None)
            # Check if any columns interact with worker type and/or are stationary (tv stands for time-varying, which is equivalent to non-stationary; and wi stands for worker-interaction)
            any_tv_nwi = False
            any_tnv_nwi = False
            any_tv_wi = False
            any_tnv_wi = False
            for col in cat_cols:
                # Check if column is stationary
                is_stationary = False
                if controls_dict[col]['cons_a'] is not None:
                    any_cat_constraints = True
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
                    any_cat_constraints = True
                    if is_stationary:
                        any_tnv_nwi = True
                    else:
                        any_tv_nwi = True

            if any_cat_constraints:
                # Add constraints only if at least one categorical control uses constraints
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
                        cons_a.add_constraints(cons.NormalizeAll(min_firm_type=min_firm_type, nnt=range(2)))
                    else:
                        if any_tnv_wi:
                            # Normalize primary period
                            cons_a.add_constraints(cons.NormalizeAll(min_firm_type=min_firm_type, cross_period_normalize=True, nnt=pp))
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
                    # Reorder part 1: e.g. nk=2, and type 0 > type 1, then 0, 1, 2, 3 would reorder to 1, 0, 3, 2 (i.e. reorder within groups)
                    pk1_order_1 = np.tile(firm_type_order, nk) + nk * np.repeat(range(nk), nk)
                    pk1 = pk1[pk1_order_1, :]
                    # Reorder part 2: e.g. nk=2, and type 0 > type 1, then 0, 1, 2, 3 would reorder to 2, 3, 0, 1 (i.e. reorder between groups)
                    pk1_order_2 = nk * np.repeat(firm_type_order, nk) + np.tile(range(nk), nk)
                    pk1 = pk1[pk1_order_2, :]
                    # adj_pk1 = np.reshape(pk1, (nk, nk, nl))
                    # adj_pk1 = adj_pk1[firm_type_order, :, :]
                    # adj_pk1 = adj_pk1[:, firm_type_order, :]
                    # pk1 = np.reshape(adj_pk1, (nk * nk, nl))
                if NNm is not None:
                    NNm = NNm[firm_type_order, :]
                    NNm = NNm[:, firm_type_order]
                if NNs is not None:
                    NNs = NNs[firm_type_order]

        return (a for a in (A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, pk0, NNm, NNs) if a is not None)

    def _normalize(self, A1, A2, A1_cat, A2_cat):
        '''
        Normalize means given categorical controls.

        Arguments:
            A1 (NumPy Array): mean of fixed effects in the first period
            A2 (NumPy Array): mean of fixed effects in the second period
            A1_cat (dict of NumPy Arrays): dictionary linking column names to the mean of fixed effects in the first period for categorical control variables
            A2_cat (dict of NumPy Arrays): dictionary linking column names to the mean of fixed effects in the second period for categorical control variables

        Returns:
            (tuple): tuple of normalized parameters (A1, A2, A1_cat, A2_cat)
        '''
        # Unpack parameters
        params = self.params
        nl, nk = self.nl, self.nk
        cat_cols, cat_dict = self.cat_cols, self.cat_dict
        A1, A2, A1_cat, A2_cat = A1.copy(), A2.copy(), A1_cat.copy(), A2_cat.copy()

        if len(cat_cols) > 0:
            # Compute minimum firm type
            min_firm_type = self._min_firm_type(A1, A2)
            # Check if any columns interact with worker type and/or are stationary (tv stands for time-varying, which is equivalent to non-stationary; and wi stands for worker-interaction)
            any_tv_nwi = False
            any_tnv_nwi = False
            any_tv_wi = False
            any_tnv_wi = False
            for col in cat_cols:
                # Check if column is stationary
                is_stationary = False
                if cat_dict[col]['cons_a'] is not None:
                    for subcons_a in to_list(cat_dict[col]['cons_a']):
                        if isinstance(subcons_a, cons.Stationary):
                            is_stationary = True
                            break

                if cat_dict[col]['worker_type_interaction']:
                    # If the column interacts with worker types
                    if is_stationary:
                        any_tnv_wi = True
                        tnv_wi_col = col
                    else:
                        any_tv_wi = True
                        tv_wi_col = col
                        break
                else:
                    if is_stationary:
                        any_tnv_nwi = True
                        tnv_nwi_col = col
                    else:
                        any_tv_nwi = True
                        tv_nwi_col = col

            ## Normalize parameters ##
            if any_tv_wi:
                for l in range(nl):
                    # First period
                    adj_val_1 = A1[l, min_firm_type]
                    A1[l, :] -= adj_val_1
                    A1_cat[tv_wi_col][l, :] += adj_val_1
                    # Second period
                    adj_val_2 = A2[l, min_firm_type]
                    A2[l, :] -= adj_val_2
                    A2_cat[tv_wi_col][l, :] += adj_val_2
            else:
                primary_period_dict = {
                    'first': 0,
                    'second': 1,
                    'all': range(2)
                }
                secondary_period_dict = {
                    'first': 1,
                    'second': 0,
                    'all': range(2)
                }
                params_dict = {
                    0: [A1, A1_cat],
                    1: [A2, A2_cat]
                }
                Ap = [params_dict[pp] for pp in to_list(primary_period_dict[params['primary_period']])]
                As = [params_dict[sp] for sp in to_list(secondary_period_dict[params['primary_period']])]
                if any_tnv_wi:
                    ## Normalize primary period ##
                    for l in range(nl):
                        # Compute normalization
                        adj_val_1 = Ap[0][0][l, min_firm_type]
                        for Ap_sub in Ap[1:]:
                            adj_val_1 += Ap_sub[0][l, min_firm_type]
                        adj_val_1 /= len(Ap)
                        # Normalize
                        A1[l, :] -= adj_val_1
                        A1_cat[tnv_wi_col][l, :] += adj_val_1
                        A2[l, :] -= adj_val_1
                        A2_cat[tnv_wi_col][l, :] += adj_val_1
                    if any_tv_nwi:
                        ## Normalize lowest type pair from secondary period ##
                        for As_sub in As:
                            adj_val_2 = As_sub[0][0, min_firm_type]
                            As_sub[0] -= adj_val_2
                            As_sub[1][tv_nwi_col] += adj_val_2
                else:
                    if any_tv_nwi:
                        ## Normalize lowest type pair in both periods ##
                        # First period
                        adj_val_1 = A1[0, min_firm_type]
                        A1 -= adj_val_1
                        A1_cat[tv_nwi_col] += adj_val_1
                        # Second period
                        adj_val_2 = A2[0, min_firm_type]
                        A2 -= adj_val_2
                        A2_cat[tv_nwi_col] += adj_val_2
                    elif any_tnv_nwi:
                        ## Normalize lowest type pair in primary period ##
                        # Compute normalization
                        adj_val_1 = Ap[0][0][0, min_firm_type]
                        for Ap_sub in Ap[1:]:
                            adj_val_1 += Ap_sub[0][0, min_firm_type]
                        adj_val_1 /= len(Ap)
                        # Normalize
                        A1 -= adj_val_1
                        A1_cat[tnv_nwi_col] += adj_val_1
                        A2 -= adj_val_1
                        A2_cat[tnv_nwi_col] += adj_val_1

        return (A1, A2, A1_cat, A2_cat)

    def _sum_by_non_nl(self, ni, C_dict, A_cat, S_cat, A_cts, S_cts, compute_A=True, compute_S=True):
        '''
        Compute A_sum/S_sum_sq for non-worker-interaction terms.

        Arguments:
            ni (int): number of observations
            C_dict (dict of dicts): link each period to a dictionary linking column names to control variable data for the corresponding period
            A_cat (dict of dict of NumPy Arrays): dictionary linking column names to periods, where each period links to a dictionary of the mean of fixed effects for that categorical control variable in that period
            S_cat (dict of dict of NumPy Arrays): dictionary linking column names to periods, where each period links to a dictionary of the standard deviation of fixed effects for that categorical control variable in that period
            A_cts (dict of dict of NumPy Arrays): dictionary linking column names to periods, where each period links to a dictionary of the mean of fixed effects for that continuous control variable in that period
            S_cts (dict of dict of NumPy Arrays): dictionary linking column names to periods, where each period links to a dictionary of the standard deviation of fixed effects for that continuous control variable in that period
            compute_A (bool): if True, compute and return A terms
            compute_S (bool): if True, compute and return S terms

        Returns:
            (tuple of dicts of NumPy Arrays): (A_sum, S_sum_sq), where each dictionary links to periods, and each period links to the sum of estimated effects for control variables that do not interact with worker type in that period (A terms are dropped if compute_A=False, and S terms are dropped if compute_S=False)
        '''
        if (not compute_A) and (not compute_S):
            raise ValueError('`compute_A`=False and `compute_S`=False. Must specify at least one to be True.')

        # if not self.any_non_worker_type_interactions:
        #     # If all control variables interact with worker type
        #     if compute_A and compute_S:
        #         return [0] * 10
        #     return [0] * 6

        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        controls_dict = self.controls_dict

        if compute_A:
            A_sum = {
                period: np.zeros(ni) for period in ['12', '43', '2ma', '2mb', '3ma', '3mb']
            }
        if compute_S:
            S_sum_sq = {
                period: np.zeros(ni) for period in ['12', '43', '2ma', '2mb', '3ma', '3mb']
            }

        ## Categorical ##
        for col in cat_cols:
            A_cat_col = A_cat[col]
            S_cat_col = S_cat[col]
            for period in ['12', '43', '2ma', '2mb', '3ma', '3mb']:
                A_cat_t = A_cat_col[period]
                S_cat_t = S_cat_col[period]
                if (not controls_dict[col]['worker_type_interaction']) or (period[-1] == 'b'):
                    C_t = C_dict[period][col]
                    if compute_A:
                        A_sum[period] += A_cat_t[C_t]
                    if compute_S:
                        S_sum_sq[period] += S_cat_t[C_t] ** 2

        ## Continuous ##
        for col in cts_cols:
            A_cts_col = A_cts[col]
            S_cts_col = S_cts[col]
            for period in ['12', '43', '2ma', '2mb', '3ma', '3mb']:
                A_cts_t = A_cts_col[period]
                S_cts_t = S_cts_col[period]
                if (not controls_dict[col]['worker_type_interaction']) or (period[-1] == 'b'):
                    if compute_A:
                        C_t = C_dict[period][col]
                        A_sum[period] += A_cts_t[col] * C_t
                    if compute_S:
                        S_sum_sq[period] += S_cts_t[col] ** 2

        if compute_A and compute_S:
            return (A_sum, S_sum_sq)
        if compute_A:
            return A_sum
        if compute_S:
            return S_sum_sq

    def _sum_by_nl_l(self, ni, l, C_dict, A_cat, S_cat, A_cts, S_cts, compute_A=True, compute_S=True):
        '''
        Compute A_sum/S_sum_sq to account for worker-interaction terms for a particular worker type.

        Arguments:
            ni (int): number of observations
            l (int): worker type (must be in range(0, nl))
            C_dict (dict of dicts): link each period to a dictionary linking column names to control variable data for the corresponding period
            A_cat (dict of dict of NumPy Arrays): dictionary linking column names to periods, where each period links to a dictionary of the mean of fixed effects for that categorical control variable in that period
            S_cat (dict of dict of NumPy Arrays): dictionary linking column names to periods, where each period links to a dictionary of the standard deviation of fixed effects for that categorical control variable in that period
            A_cts (dict of dict of NumPy Arrays): dictionary linking column names to periods, where each period links to a dictionary of the mean of fixed effects for that continuous control variable in that period
            S_cts (dict of dict of NumPy Arrays): dictionary linking column names to periods, where each period links to a dictionary of the standard deviation of fixed effects for that continuous control variable in that period
            compute_A (bool): if True, compute and return A terms
            compute_S (bool): if True, compute and return S terms

        Returns:
            (tuple of dicts of NumPy Arrays): (A_sum_l, S_sum_sq_l), where each dictionary links to periods, and each period links to the sum of estimated effects for control variables that interact with worker type, specifically for worker type l (A terms are dropped if compute_A=False, and S terms are dropped if compute_S=False)
        '''
        if (not compute_A) and (not compute_S):
            raise ValueError('compute_A=False and compute_S=False. Must specify at least one to be True.')

        # if not self.any_worker_type_interactions:
        #     # If no control variables interact with worker type
        #     if compute_A and compute_S:
        #         return [0] * 10
        #     return [0] * 6

        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        controls_dict = self.controls_dict

        if compute_A:
            A_sum_l = {
                period: np.zeros(ni) for period in ['12', '43', '2ma', '2mb', '3ma', '3mb']
            }
        if compute_S:
            S_sum_sq_l = {
                period: np.zeros(ni) for period in ['12', '43', '2ma', '2mb', '3ma', '3mb']
            }

        ## Categorical ##
        for col in cat_cols:
            A_cat_col = A_cat[col]
            S_cat_col = S_cat[col]
            for period in ['12', '43', '2ma', '2mb', '3ma', '3mb']:
                A_cat_t = A_cat_col[period]
                S_cat_t = S_cat_col[period]
                if controls_dict[col]['worker_type_interaction'] and (period[-1] != 'b'):
                    C_t = C_dict[period][col]
                    if compute_A:
                        A_sum_l[period] += A_cat_t[l, C_t]
                    if compute_S:
                        S_sum_sq_l[period] += S_cat_t[l, C_t] ** 2

        ## Continuous ##
        for col in cts_cols:
            A_cts_col = A_cts[col]
            S_cts_col = S_cts[col]
            for period in ['12', '43', '2ma', '2mb', '3ma', '3mb']:
                A_cts_t = A_cts_col[period]
                S_cts_t = S_cts_col[period]
                if controls_dict[col]['worker_type_interaction'] and (period[-1] != 'b'):
                    if compute_A:
                        C_t = C_dict[period][col]
                        A_sum_l[period] += A_cts_t[col][l] * C_t
                    if compute_S:
                        S_sum_sq_l[period] += S_cts_t[col][l] ** 2

        if compute_A and compute_S:
            return (A_sum_l, S_sum_sq_l)
        if compute_A:
            return A_sum_l
        if compute_S:
            return S_sum_sq_l

    def _var_stayers(self, sdata, rho_1, rho_4, rho_t, weights=None, diff=False):
        '''
        Compute var(alpha | g1, g2) and var(epsilon | g1) using stayers.

        Arguments:
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            rho_1 (float): rho in period 1
            rho_4 (float): rho in period 4
            rho_t (float): rho in period t
            weights (tuple or None): weights for rho; if None, all elements of rho have equal weight
            diff (bool): if True, estimate rho in differences rather than levels

        Returns:
            (dict): dictionary with results of variance estimation on stayers. If rho values cannot be resolved, returns {'eps_sq': np.inf}.
        '''
        ## Extract parameters ##
        nk = sdata.n_clusters()

        ## Compute var(epsilon | g1) ##
        groupby_g1 = sdata.groupby('g1')
        W = groupby_g1['i'].size().to_numpy()

        ## Y ##
        var_dict = {t: groupby_g1[f'y{t + 1}'].var(ddof=0).to_numpy() for t in range(4)}
        cov_dict = {(t1, t2): groupby_g1.apply(lambda df: df[f'y{t1 + 1}'].cov(df[f'y{t2 + 1}'], ddof=0)).to_numpy() for t1 in range(4) for t2 in range(4) if t2 > t1}

        # Combine var_dict and cov_dict into YY_lst
        YY_lst = []
        for t in range(4):
            YY_lst.append(var_dict[t])
        for t1 in range(4):
            for t2 in range(4):
                if t2 > t1:
                    YY_lst.append(cov_dict[(t1, t2)])

        ## X ##
        XX_lst = [np.zeros((nk, 5 * nk)) for i in range(10)]

        for g1 in range(nk):
            ## Compute var(alpha | g1, g2) ##
            Gs = 0

            for XX in XX_lst:
                XX[g1, g1] = 1

            Gs += nk
            
            ## Compute var(nu_1 | g1) ##
            XX_lst[0][g1, Gs + g1] = 1

            Gs += nk
            
            ## Compute var(nu_2 | g1) ##
            XX_lst[0][g1, Gs + g1] = rho_1 ** 2
            XX_lst[1][g1, Gs + g1] = 1
            XX_lst[2][g1, Gs + g1] = rho_t ** 2
            XX_lst[3][g1, Gs + g1] = rho_4 ** 2 + rho_t ** 2 # FIXME here should r4!!!!!
            XX_lst[4][g1, Gs + g1] = rho_1
            XX_lst[5][g1, Gs + g1] = rho_1 * rho_t
            XX_lst[6][g1, Gs + g1] = rho_1 * rho_t * rho_4
            XX_lst[7][g1, Gs + g1] = rho_t
            XX_lst[8][g1, Gs + g1] = rho_t * rho_4
            XX_lst[0][g1, Gs + g1] = rho_t ** 2 * rho_4

            Gs += nk
            
            ## Compute var(nu_3 | g1) ##
            XX_lst[2][g1, Gs + g1] = 1
            XX_lst[3][g1, Gs + g1] = rho_4 ** 2
            XX_lst[9][g1, Gs + g1] = rho_4

            Gs += nk

            ## Compute var(nu_4 | g1) ##
            XX_lst[3][g1, Gs + g1] = 1

        if diff:
            ### If computing rhos in differences ###
            D = np.array(
                [
                    [-1, 1, 0, 0],
                    [0, -1, 1, 0],
                    [0, 0, -1, 1]
                ]
            )
            DD = np.kron(D, D)
            # FIXME I changed this to 8 since it wasn't working with 10
            DD = np.kron(DD, np.eye(8))

            ## Combine matrices ##
            rho_order = np.array([
                    0, 4, 5, 6,
                    4, 1, 7, 8,
                    5, 7, 2, 9,
                    6, 8, 9, 3
            ])
            XX = np.vstack([XX_lst[t] for t in rho_order])
            YY = np.hstack([YY_lst[t] for t in rho_order])
            
            # Update XX and YY
            XX = (DD @ XX)[:, 10:]
            YY = DD @ YY

            # Update weights
            weights = np.repeat(W, 9)

        else:
            ### If computing rhos in levels ###
            ## Combine matrices ##
            XX = np.vstack([XX for XX in XX_lst])
            YY = np.hstack([YY for YY in YY_lst])

            if weights is None:
                weights = np.repeat(W, 10)

        ## Fit constrained linear model ##
        cons_lm = cons.QPConstrained(1, XX.shape[1])
        cons_lm.add_constraints(cons.BoundedBelow(lb=0, nt=1))
        DpXX = np.diag(weights) @ XX
        cons_lm.solve(DpXX.T @ XX, -(DpXX.T @ YY))

        ## Results ##
        beta_hat = cons_lm.res
        if beta_hat is None:
            return {'eps_sq': np.inf}
        eps = (YY - XX @ beta_hat)
        eps_sq = eps.T @ np.diag(weights) @ eps

        if diff:
            beta_hat_full = np.concatenate([np.zeros(10), beta_hat])
        else:
            beta_hat_full = beta_hat

        BE_sd = np.sqrt(np.maximum(beta_hat_full[: nk], 0))
        nu1_sd = np.sqrt(np.maximum(beta_hat_full[nk: 2 * nk], 0))
        nu2_sd = np.sqrt(np.maximum(beta_hat_full[2 * nk: 3 * nk], 0))
        nu3_sd = np.sqrt(np.maximum(beta_hat_full[3 * nk: 4 * nk], 0))
        nu4_sd = np.sqrt(np.maximum(beta_hat_full[4 * nk: 5 * nk], 0))

        return {
            'rho_1': rho_1,
            'rho_4': rho_4,
            'rho_t': rho_t,
            'BE_sd': BE_sd,
            'nu1_sd': nu1_sd,
            'nu2_sd': nu2_sd,
            'nu3_sd': nu3_sd,
            'nu4_sd': nu4_sd,
            'eps': eps,
            'eps_sq': eps_sq,
            'weights': weights,
            'XX_beta': XX @ beta_hat,
            'YY': YY
        }

    def _rho_init(self, sdata, rho_0=(0.5, 0.5, 0.5), weights=None, diff=False):
        '''
        Generate starting value for rho using stayers.

        Arguments:
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            rho_0 (tuple): initial guess for rho
            weights (tuple or None): weights for rho; if None, all elements of rho have equal weight
            diff (bool): if True, estimate rho in differences rather than levels

        Returns:
            (dict): dictionary with results of variance estimation on stayers
        '''
        # Define function to optimize rho
        rho_optim_fn = lambda rhos: self._var_stayers(sdata, *rhos, weights=weights, diff=diff)['eps_sq']
        # Initialize and fit optimizer
        optim_fn = opt(fun=rho_optim_fn, x0=rho_0, method='BFGS')
        # Extract optimal rho
        rho_optim = optim_fn.x
        # Run _var_stayers with optimal rho
        return self._var_stayers(sdata, *rho_optim, weights=weights, diff=diff)

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
        ## Unpack parameters ##
        params = self.params
        nl, nk, ni = self.nl, self.nk, jdata.shape[0]
        R12, R43, R32m = self.R12, self.R43, self.R32m
        A, A_cat, A_cts, S, S_cat, S_cts = self.A, self.A_cat, self.A_cts, self.S, self.S_cat, self.S_cts
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        cat_dict, cts_dict = self.cat_dict, self.cts_dict
        controls_dict = self.controls_dict
        any_controls, any_non_worker_type_interactions = self.any_controls, self.any_non_worker_type_interactions

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        Y3 = jdata.loc[:, 'y3'].to_numpy()
        Y4 = jdata.loc[:, 'y4'].to_numpy()
        G1 = jdata.loc[:, 'g1'].to_numpy().astype(int, copy=False)
        G2 = jdata.loc[:, 'g4'].to_numpy().astype(int, copy=False)

        # Control variables
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
                subcol_3 = subcols[0]
                subcol_4 = subcols[0]
            elif n_subcols == 4:
                # If column can change over time
                subcol_1 = subcols[0]
                subcol_2 = subcols[3]
            else:
                raise NotImplementedError(f'Column names must have either one or four associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
            if i < len(cat_cols):
                # Categorical
                C1[col] = jdata.loc[:, subcol_1].to_numpy().astype(int, copy=False)
                C2[col] = jdata.loc[:, subcol_2].to_numpy().astype(int, copy=False)
            else:
                # Continuous
                C1[col] = jdata.loc[:, subcol_1].to_numpy()
                C2[col] = jdata.loc[:, subcol_2].to_numpy()
        C1_periods = ['12', '2ma', '3mb']
        C2_periods = ['43', '2mb', '3ma']
        C_dict = {period: C1 if period in C1_periods else C2 for period in ['12', '43', '2ma', '2mb', '3ma', '3mb']}
        ## Sparse matrix representations ##
        GG1 = csc_matrix((np.ones(ni), (range(ni), G1)), shape=(ni, nk))
        GG2 = csc_matrix((np.ones(ni), (range(ni), G2)), shape=(ni, nk))
        CC1 = {col: csc_matrix((np.ones(ni), (range(ni), C1[col])), shape=(ni, controls_dict[col]['n'])) for col in cat_cols}
        CC2 = {col: csc_matrix((np.ones(ni), (range(ni), C2[col])), shape=(ni, controls_dict[col]['n'])) for col in cat_cols}
        CC3 = {col: csc_matrix((np.ones(ni), (range(ni), C3[col])), shape=(ni, controls_dict[col]['n'])) for col in cat_cols}
        CC4 = {col: csc_matrix((np.ones(ni), (range(ni), C4[col])), shape=(ni, controls_dict[col]['n'])) for col in cat_cols}

        # Joint firm indicator
        KK = G1 + nk * G2

        # Transition probability matrix
        GG12 = csc_matrix((np.ones(ni), (range(ni), KK)), shape=(ni, nk ** 2))

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
        # # Track minimum firm type to check whether estimator stuck in a loop
        # min_firm_types = []
        # Whether results should be stored
        store_res = True

        # ## Sort ##
        # A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, self.pk0 = self._sort_parameters(A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, self.pk0)

        ## Constraints ##
        # if params['force_min_firm_type']:
        #     # If forcing minimum firm type
        #     prev_min_firm_type = min_firm_type
        #     min_firm_type = min_firm_type
        # else:
        #     # If not forcing minimum firm type
        #     prev_min_firm_type = self._min_firm_type(A1, A2)
        cons_a, cons_s, cons_a_dict, cons_s_dict = self._gen_constraints() # prev_min_firm_type)

        for iter in range(params['n_iters_movers']):
            # ---------- E-Step ----------
            # We compute the posterior probabilities for each row
            # We iterate over the worker types, should not be be too costly since the vector is quite large within each iteration
            log_pk1 = np.log(pk1)
            if any_controls > 0:
                ## Account for control variables ##
                if iter == 0:
                    A_sum, S_sum_sq = self._sum_by_non_nl(ni=ni, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts)
                else:
                    S_sum_sq = self._sum_by_non_nl(ni=ni, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_A=False)

                for l in range(nl):
                    # Update A_sum/S_sum_sq to account for worker-interaction terms
                    A_sum_l, S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts)

                    lp1 = lognormpdf(
                        Y1 - R12 * (Y2 - A['2ma'][l, G1] - A_sum['2ma'] - A_sum_l['2ma']),
                        A['12'][l, G1] + A_sum['12'] + A_sum_l['12'],
                        var=S['12'][l, G1] ** 2 + S_sum_sq['12'] + S_sum_sq_l['12']
                    )
                    lp2 = lognormpdf(
                        Y2,
                        (A['2ma'][l, G1] + A['2mb'][G2]) + (A_sum['2ma'] + A_sum['2mb']) + (A_sum_l['2ma'] + A_sum_l['2mb']),
                        var=S['2m'][l, G1] ** 2 + S_sum_sq['2m'] + S_sum_sq_l['2m']
                    )
                    lp3 = lognormpdf(
                        Y3 - R32m * (Y2 - (A['2ma'][l, G1] + A['2mb'][G2]) - (A_sum['2ma'] + A_sum['2mb']) - (A_sum_l['2ma'] + A_sum_l['2mb'])),
                        (A['3ma'][l, G2] + A['3mb'][G1]) + (A_sum['3ma'] + A_sum['3mb']) + (A_sum_l['3ma'] + A_sum_l['3mb']),
                        var=S['3m'][l, G2] ** 2 + S_sum_sq['3m'] + S_sum_sq_l['3m']
                    )
                    lp4 = lognormpdf(
                        Y4 - R43 * (Y3 - A['3ma'][l, G2] - A_sum['3ma'] - A_sum_l['3ma']),
                        A['43'][l, G2] + A_sum['43'] + A_sum_l['43'],
                        var=S['43'][l, G2] ** 2 + S_sum_sq['43'] + S_sum_sq_l['43']
                    )

                    lp[:, l] = log_pk1[KK, l] + lp1 + lp2 + lp3 + lp4
            else:
                # Loop over firm classes so means/variances are single values rather than vectors (computing log/square is much faster this way)
                for g1 in range(nk):
                    for g2 in range(nk):
                        I = (G1 == g1) and (G2 == g2)
                        for l in range(nl):
                            lp1 = lognormpdf(
                                Y1[I] - R12 * (Y2[I] - A['2ma'][l, g1]),
                                A['12'][l, g1],
                                sd=S['12'][l, g1]
                            )
                            lp2 = lognormpdf(
                                Y2[I],
                                A['2ma'][l, g1] + A['2mb'][g2],
                                sd=S['2m'][l, g1]
                            )
                            lp3 = lognormpdf(
                                Y3[I] - R32m * (Y2[I] - A['2ma'][l, g1] - A['2mb'][g2]),
                                A['3ma'][l, g2] + A['3mb'][g1],
                                sd=S['3m'][l, g2]
                            )
                            lp4 = lognormpdf(
                                Y4[I] - R43 * (Y3[I] - A['3ma'][l, g2]),
                                A['43'][l, g2],
                                sd=S['43'][l, g2]
                            )

                            lp[I, l] = log_pk1[KK[I], l] + lp1 + lp2 + lp3 + lp4
            del log_pk1, lp1, lp2, lp3, lp4

            # We compute log sum exp to get likelihoods and probabilities
            lse_lp = logsumexp(lp, axis=1)
            qi = np.exp(lp.T - lse_lp).T
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
                min_firm_type = self._min_firm_type(A1, A2)

            if ((abs(lik1 - prev_lik) < params['threshold_movers']) and (min_firm_type == prev_min_firm_type)):
                # Break loop
                break
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
            GG1_weighted = []
            GG2_weighted = []
            for l in range(nl):
                # (We might be better off trying this within numba or something)
                l_index, r_index = l * nk, (l + 1) * nk
                # Shared weighted terms
                GG1_weighted.append(DxSP(W1 * qi[:, l] / S1[l, G1], GG1).T)
                GG2_weighted.append(DxSP(W2 * qi[:, l] / S2[l, G2], GG2).T)
                ## Compute XwX terms ##
                XwX[l_index: r_index] = diag_of_sp_prod(GG1_weighted[l], GG1)
                XwX[ts + l_index: ts + r_index] = np.asarray(GG2_weighted[l].multiply(GG2.T).sum(axis=1))[:, 0]
                if params['update_a']:
                    # Update A_sum to account for worker-interaction terms
                    A12_sum_l, A43_sum_l, A2ma_sum_l, A2mb_sum_l, A3ma_sum_l, A3mb_sum_l = self._sum_by_nl_l(
                        ni=ni, l=l, C1=C1, C2=C2, C3=C3, C4=C4,
                        A12_cat=A12_cat, A43_cat=A43_cat, S12_cat=S12_cat, S43_cat=S43_cat,
                        A2ma_cat=A2ma_cat, A2mb_cat=A2mb_cat, S2m_cat=S2m_cat,
                        A3ma_cat=A3ma_cat, A3mb_cat=A3mb_cat, S3m_cat=S3m_cat,
                        A12_cts=A12_cts, A43_cts=A43_cts, S12_cts=S12_cts, S43_cts=S43_cts,
                        A2ma_cts=A2ma_cts, A2mb_cts=A2mb_cts, S2m_cts=S2m_cts,
                        A3ma_cts=A3ma_cts, A3mb_cts=A3mb_cts, S3m_cts=S3m_cts, compute_S=False)
                    ## Compute XwY terms ##
                    XwY[l_index: r_index] = GG1_weighted[l] @ (Y1_adj - A1_sum_l)
                    XwY[ts + l_index: ts + r_index] = GG2_weighted[l] @ (Y2_adj - A2_sum_l)
                    del A12_sum_l, A43_sum_l, A2ma_sum_l, A2mb_sum_l, A3ma_sum_l, A3mb_sum_l
            if not params['update_s']:
                del GG1_weighted, GG2_weighted

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
                    if cons_a.res is None:
                        # If constraints inconsistent, keep A1 and A2 the same
                        if params['verbose'] in [2, 3]:
                            print(f'Passing A1/A2: {e}')
                    else:
                        res_a1, res_a2 = cons_a.res[: len(cons_a.res) // 2], cons_a.res[len(cons_a.res) // 2:]
                        # if pd.isna(res_a1).any() or pd.isna(res_a2).any():
                        #     raise ValueError('Estimated A1/A2 has NaN values')
                        A1 = np.reshape(res_a1, self.dims)
                        A2 = np.reshape(res_a2, self.dims)

                except ValueError as e:
                    # If constraints inconsistent, keep A1 and A2 the same
                    if params['verbose'] in [2, 3]:
                        print(f'Passing A1/A2: {e}')

            ## Categorical ##
            CC1_cat_weighted = {}
            CC2_cat_weighted = {}
            for col in cat_cols:
                CC1_cat_weighted[col] = []
                CC2_cat_weighted[col] = []
                col_n = cat_dict[col]['n']
                if not cat_dict[col]['worker_type_interaction']:
                    Y1_adj += A1_cat[col][C1[col]]
                    Y2_adj += A2_cat[col][C2[col]]
                for l in range(nl):
                    l_index, r_index = l * col_n, (l + 1) * col_n
                    ## Compute shared terms ##
                    if cat_dict[col]['worker_type_interaction']:
                        S1_cat_l = S1_cat[col][l, C1[col]]
                        S2_cat_l = S2_cat[col][l, C2[col]]
                    else:
                        S1_cat_l = S1_cat[col][C1[col]]
                        S2_cat_l = S2_cat[col][C2[col]]
                    CC1_cat_weighted[col].append(DxSP(W1 * qi[:, l] / S1_cat_l, CC1[col]).T)
                    CC2_cat_weighted[col].append(DxSP(W2 * qi[:, l] / S2_cat_l, CC2[col]).T)
                    del S1_cat_l, S2_cat_l
                    ## Compute XwX_cat terms ##
                    XwX_cat[col][l_index: r_index] = diag_of_sp_prod(CC1_cat_weighted[col][l], CC1[col])
                    XwX_cat[col][ts_cat[col] + l_index: ts_cat[col] + r_index] = np.asarray(CC2_cat_weighted[col][l].multiply(CC2[col].T).sum(axis=1))[:, 0]
                    if params['update_a']:
                        # Update A_sum to account for worker-interaction terms
                        A12_sum_l, A43_sum_l, A2ma_sum_l, A2mb_sum_l, A3ma_sum_l, A3mb_sum_l = self._sum_by_nl_l(
                            ni=ni, l=l, C1=C1, C2=C2, C3=C3, C4=C4,
                            A12_cat=A12_cat, A43_cat=A43_cat, S12_cat=S12_cat, S43_cat=S43_cat,
                            A2ma_cat=A2ma_cat, A2mb_cat=A2mb_cat, S2m_cat=S2m_cat,
                            A3ma_cat=A3ma_cat, A3mb_cat=A3mb_cat, S3m_cat=S3m_cat,
                            A12_cts=A12_cts, A43_cts=A43_cts, S12_cts=S12_cts, S43_cts=S43_cts,
                            A2ma_cts=A2ma_cts, A2mb_cts=A2mb_cts, S2m_cts=S2m_cts,
                            A3ma_cts=A3ma_cts, A3mb_cts=A3mb_cts, S3m_cts=S3m_cts, compute_S=False)
                        if cat_dict[col]['worker_type_interaction']:
                            A1_sum_l -= A1_cat[col][l, C1[col]]
                            A2_sum_l -= A2_cat[col][l, C2[col]]
                        ## Compute XwY_cat terms ##
                        XwY_cat[col][l_index: r_index] = CC1_cat_weighted[col][l] @ (Y1_adj - A1_sum_l - A1[l, G1])
                        XwY_cat[col][ts_cat[col] + l_index: ts_cat[col] + r_index] = CC2_cat_weighted[col][l] @ (Y2_adj - A2_sum_l - A2[l, G2])
                        del A12_sum_l, A43_sum_l, A2ma_sum_l, A2mb_sum_l, A3ma_sum_l, A3mb_sum_l
                if not params['update_s']:
                    del CC1_cat_weighted[col], CC2_cat_weighted[col]

                # We solve the system to get all the parameters (use dense solver)
                XwX_cat[col] = np.diag(XwX_cat[col])
                if params['update_a']:
                    try:
                        a_solver = cons_a_dict[col]
                        a_solver.solve(XwX_cat[col], -XwY_cat[col], solver='quadprog')
                        if a_solver.res is None:
                            # If constraints inconsistent, keep A1_cat and A2_cat the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing A1_cat/A2_cat for column {col!r}: {e}')
                        else:
                            res_a1, res_a2 = a_solver.res[: len(a_solver.res) // 2], a_solver.res[len(a_solver.res) // 2:]
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
            CC1_cts_weighted = {}
            CC2_cts_weighted = {}
            for col in cts_cols:
                CC1_cts_weighted[col] = []
                CC2_cts_weighted[col] = []
                if not cts_dict[col]['worker_type_interaction']:
                    Y1_adj += A1_cts[col] * C1[col]
                    Y2_adj += A2_cts[col] * C2[col]
                for l in range(nl):
                    ## Compute shared terms ##
                    if cts_dict[col]['worker_type_interaction']:
                        S1_cts_l = S1_cts[col][l]
                        S2_cts_l = S2_cts[col][l]
                    else:
                        S1_cts_l = S1_cts[col]
                        S2_cts_l = S2_cts[col]
                    CC1_cts_weighted[col].append(C1[col].T * (W1 * qi[:, l] / S1_cts_l))
                    CC2_cts_weighted[col].append(C2[col].T * (W2 * qi[:, l] / S2_cts_l))
                    del S1_cts_l, S2_cts_l
                    ## Compute XwX_cts terms ##
                    XwX_cts[col][l] = (CC1_cts_weighted[col][l] @ C1[col])
                    XwX_cts[col][nl + l] = (CC2_cts_weighted[col][l] @ C2[col])
                    if params['update_a']:
                        # Update A_sum to account for worker-interaction terms
                        A12_sum_l, A43_sum_l, A2ma_sum_l, A2mb_sum_l, A3ma_sum_l, A3mb_sum_l = self._sum_by_nl_l(
                            ni=ni, l=l, C1=C1, C2=C2, C3=C3, C4=C4,
                            A12_cat=A12_cat, A43_cat=A43_cat, S12_cat=S12_cat, S43_cat=S43_cat,
                            A2ma_cat=A2ma_cat, A2mb_cat=A2mb_cat, S2m_cat=S2m_cat,
                            A3ma_cat=A3ma_cat, A3mb_cat=A3mb_cat, S3m_cat=S3m_cat,
                            A12_cts=A12_cts, A43_cts=A43_cts, S12_cts=S12_cts, S43_cts=S43_cts,
                            A2ma_cts=A2ma_cts, A2mb_cts=A2mb_cts, S2m_cts=S2m_cts,
                            A3ma_cts=A3ma_cts, A3mb_cts=A3mb_cts, S3m_cts=S3m_cts, compute_S=False)
                        if cts_dict[col]['worker_type_interaction']:
                            A1_sum_l -= A1_cts[col][l] * C1[col]
                            A2_sum_l -= A2_cts[col][l] * C2[col]
                        ## Compute XwY_cts terms ##
                        XwY_cts[col][l] = CC1_cts_weighted[col][l] @ (Y1_adj - A1_sum_l - A1[l, G1])
                        XwY_cts[col][nl + l] = CC2_cts_weighted[col][l] @ (Y2_adj - A2_sum_l - A2[l, G2])
                        del A12_sum_l, A43_sum_l, A2ma_sum_l, A2mb_sum_l, A3ma_sum_l, A3mb_sum_l
                if not params['update_s']:
                    del CC1_cts_weighted[col], CC2_cts_weighted[col]

                # We solve the system to get all the parameters (use dense solver)
                XwX_cts[col] = np.diag(XwX_cts[col])
                if params['update_a']:
                    try:
                        a_solver = cons_a_dict[col]
                        a_solver.solve(XwX_cts[col], -XwY_cts[col], solver='quadprog')
                        if a_solver.res is None:
                            # If constraints inconsistent, keep A1_cts and A2_cts the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing A1_cts/A2_cts for column {col!r}: {e}')
                        else:
                            res_a1, res_a2 = a_solver.res[: len(a_solver.res) // 2], a_solver.res[len(a_solver.res) // 2:]
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
                # Next we extract the variances
                if iter == 0:
                    XwS = np.zeros(shape=2 * ts)

                    ## Categorical ##
                    if len(cat_cols) > 0:
                        XwS_cat = {col: np.zeros(shape=2 * col_ts) for col, col_ts in ts_cat.items()}
                    ## Continuous ##
                    if len(cts_cols) > 0:
                        XwS_cts = {col: np.zeros(shape=2 * nl) for col in cts_cols}

                ## Update S ##
                for l in range(nl):
                    # Update A_sum to account for worker-interaction terms
                    A12_sum_l, A43_sum_l, A2ma_sum_l, A2mb_sum_l, A3ma_sum_l, A3mb_sum_l = self._sum_by_nl_l(
                        ni=ni, l=l, C1=C1, C2=C2, C3=C3, C4=C4,
                        A12_cat=A12_cat, A43_cat=A43_cat, S12_cat=S12_cat, S43_cat=S43_cat,
                        A2ma_cat=A2ma_cat, A2mb_cat=A2mb_cat, S2m_cat=S2m_cat,
                        A3ma_cat=A3ma_cat, A3mb_cat=A3mb_cat, S3m_cat=S3m_cat,
                        A12_cts=A12_cts, A43_cts=A43_cts, S12_cts=S12_cts, S43_cts=S43_cts,
                        A2ma_cts=A2ma_cts, A2mb_cts=A2mb_cts, S2m_cts=S2m_cts,
                        A3ma_cts=A3ma_cts, A3mb_cts=A3mb_cts, S3m_cts=S3m_cts, compute_S=False)
                    eps1_l_sq = (Y1_adj - A1_sum_l - A1[l, G1]) ** 2
                    eps2_l_sq = (Y2_adj - A2_sum_l - A2[l, G2]) ** 2
                    del A12_sum_l, A43_sum_l, A2ma_sum_l, A2mb_sum_l, A3ma_sum_l, A3mb_sum_l
                    ## XwS terms ##
                    l_index, r_index = l * nk, (l + 1) * nk
                    XwS[l_index: r_index] = GG1_weighted[l] @ eps1_l_sq
                    XwS[ts + l_index: ts + r_index] = GG2_weighted[l] @ eps2_l_sq
                    ## Categorical ##
                    for col in cat_cols:
                        col_n = cat_dict[col]['n']
                        l_index, r_index = l * col_n, (l + 1) * col_n
                        ## XwS_cat terms ##
                        XwS_cat[col][l_index: r_index] = CC1_cat_weighted[col][l] @ eps1_l_sq
                        XwS_cat[col][ts_cat[col] + l_index: ts_cat[col] + r_index] = CC2_cat_weighted[col][l] @ eps2_l_sq
                    ## Continuous ##
                    for col in cts_cols:
                        ## XwS_cts terms ##
                        # NOTE: take absolute value
                        XwS_cts[col][l] = np.abs(CC1_cts_weighted[col][l] @ eps1_l_sq)
                        XwS_cts[col][nl + l] = np.abs(CC2_cts_weighted[col][l] @ eps2_l_sq)
                    del eps1_l_sq, eps2_l_sq
                del GG1_weighted, GG2_weighted, CC1_cat_weighted, CC2_cat_weighted

                try:
                    cons_s.solve(XwX, -XwS, solver='quadprog')
                    if cons_s.res is None:
                        # If constraints inconsistent, keep S1 and S2 the same
                        if params['verbose'] in [2, 3]:
                            print(f'Passing S1/S2: {e}')
                    else:
                        res_s1, res_s2 = cons_s.res[: len(cons_s.res) // 2], cons_s.res[len(cons_s.res) // 2:]
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
                    try:
                        col_n = cat_dict[col]['n']
                        s_solver = cons_s_dict[col]
                        s_solver.solve(XwX_cat[col], -XwS_cat[col], solver='quadprog')
                        if s_solver.res is None:
                            # If constraints inconsistent, keep S1_cat and S2_cat the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing S1_cat/S2_cat for column {col!r}: {e}')
                        else:
                            res_s1, res_s2 = s_solver.res[: len(s_solver.res) // 2], s_solver.res[len(s_solver.res) // 2:]
                            # if pd.isna(res_s1).any() or pd.isna(res_s2).any():
                            #     raise ValueError(f'Estimated S1_cat/S2_cat has NaN values for column {col!r}')
                            if cat_dict[col]['worker_type_interaction']:
                                S1_cat[col] = np.sqrt(np.reshape(res_s1, (nl, col_n)))
                                S2_cat[col] = np.sqrt(np.reshape(res_s2, (nl, col_n)))
                            else:
                                S1_cat[col] = np.sqrt(res_s1[: col_n])
                                S2_cat[col] = np.sqrt(res_s2[: col_n])

                    except ValueError as e:
                        # If constraints inconsistent, keep S1_cat and S2_cat the same
                        if params['verbose'] in [2, 3]:
                            print(f'Passing S1_cat/S2_cat for column {col!r}: {e}')

                ## Continuous ##
                for col in cts_cols:
                    try:
                        s_solver = cons_s_dict[col]
                        s_solver.solve(XwX_cts[col], -XwS_cts[col], solver='quadprog')
                        if s_solver.res is None:
                            # If constraints inconsistent, keep S1_cts and S2_cts the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing S1_cts/S2_cts for column {col!r}: {e}')
                        else:
                            res_s1, res_s2 = s_solver.res[: len(s_solver.res) // 2], s_solver.res[len(s_solver.res) // 2:]
                            # if pd.isna(res_s1).any() or pd.isna(res_s2).any():
                            #     raise ValueError(f'Estimated S1_cts/S2_cts has NaN values for column {col!r}')
                            if cts_dict[col]['worker_type_interaction']:
                                S1_cts[col] = np.sqrt(res_s1)
                                S2_cts[col] = np.sqrt(res_s2)
                            else:
                                S1_cts[col] = np.sqrt(res_s1[0])
                                S2_cts[col] = np.sqrt(res_s2[0])

                    except ValueError as e:
                        # If constraints inconsistent, keep S1_cts and S2_cts the same
                        if params['verbose'] in [2, 3]:
                            print(f'Passing S1_cts/S2_cts for column {col!r}: {e}')

            if params['update_pk1']:
                # NOTE: add dirichlet prior
                pk1 = GG12.T @ (qi + d_prior - 1)
                # Normalize rows to sum to 1
                pk1 = DxM(1 / np.sum(pk1, axis=1), pk1)

                if pd.isna(pk1).any():
                    raise ValueError('Estimated pk1 has NaN values. Please try a different set of starting values.')

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
            if len(cat_cols) > 0:
                ## Normalize ##
                # NOTE: normalize here because constraints don't normalize unless categorical controls are using constraints, and even when used, constraints don't always normalize to exactly 0
                A1, A2, A1_cat, A2_cat = self._normalize(A1, A2, A1_cat, A2_cat)

            ## Sort parameters ##
            A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, self.pk0 = self._sort_parameters(A1, A2, S1, S2, A1_cat, A2_cat, S1_cat, S2_cat, A1_cts, A2_cts, S1_cts, S2_cts, pk1, self.pk0)

            if len(cat_cols) > 0:
                ## Normalize again ##
                A1, A2, A1_cat, A2_cat = self._normalize(A1, A2, A1_cat, A2_cat)

            # Store parameters
            self.A1, self.A2, self.S1, self.S2 = A1, A2, S1, S2
            self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat = A1_cat, A2_cat, S1_cat, S2_cat
            self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts = A1_cts, A2_cts, S1_cts, S2_cts
            self.pk1, self.lik1 = pk1, lik1
            self.liks1 = liks1 # np.concatenate([self.liks1, liks1])

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
        A12, A43, S12, S43 = self.A12, self.A43, self.S12, self.S43
        A2ma, A2mb, S2m = self.A2ma, self.A2mb, self.S2m
        A3ma, A3mb, S3m = self.A3ma, self.A3ma, self.S3m
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        any_controls = self.any_controls
        # Fix error from bad initial guesses causing probabilities to be too low
        d_prior = params['d_prior_stayers']

        # Store wage outcomes and groups
        Y1 = sdata['y1'].to_numpy()
        Y2 = sdata['y2'].to_numpy()
        G1 = sdata['g1'].to_numpy().astype(int, copy=False)
        # G2 = sdata['g2'].to_numpy().astype(int, copy=False)
        GG1 = csc_matrix((np.ones(ni), (range(ni), G1)), shape=(ni, nk))

        if any_controls:
            ## Control variables ##
            C1 = {}
            C2 = {}
            C3 = {}
            C4 = {}
            for i, col in enumerate(cat_cols + cts_cols):
                # Get subcolumns associated with col
                subcols = to_list(sdata.col_reference_dict[col])
                n_subcols = len(subcols)
                if n_subcols == 1:
                    # If column is constant over time
                    subcol_1 = subcols[0]
                    subcol_2 = subcols[0]
                    subcol_3 = subcols[0]
                    subcol_4 = subcols[0]
                elif n_subcols == 4:
                    # If column can change over time
                    subcol_1 = subcols[0]
                    subcol_2 = subcols[1]
                    subcol_3 = subcols[2]
                    subcol_4 = subcols[3]
                else:
                    raise NotImplementedError(f'Column names must have either one or four associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
                if i < len(cat_cols):
                    # Categorical
                    C1[col] = sdata.loc[:, subcol_1].to_numpy().astype(int, copy=False)
                    C2[col] = sdata.loc[:, subcol_2].to_numpy().astype(int, copy=False)
                    C3[col] = sdata.loc[:, subcol_3].to_numpy().astype(int, copy=False)
                    C4[col] = sdata.loc[:, subcol_4].to_numpy().astype(int, copy=False)
                else:
                    # Continuous
                    C1[col] = sdata.loc[:, subcol_1].to_numpy()
                    C2[col] = sdata.loc[:, subcol_2].to_numpy()
                    C3[col] = sdata.loc[:, subcol_3].to_numpy()
                    C4[col] = sdata.loc[:, subcol_4].to_numpy()

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

        if any_controls:
            ## Account for control variables ##
            A12_sum, A43_sum, A2ma_sum, A2mb_sum, A3ma_sum, A3mb_sum, S12_sum_sq, S43_sum_sq, S2m_sum_sq, S3m_sum_sq = self._sum_by_non_nl(
                ni=ni, C1=C1, C2=C2, C3=C3, C4=C4,
                A12_cat=self.A12_cat, A43_cat=self.A43_cat, S12_cat=self.S12_cat, S43_cat=self.S43_cat,
                A2ma_cat=self.A2ma_cat, A2mb_cat=self.A2mb_cat, S2m_cat=self.S2m_cat,
                A3ma_cat=self.A3ma_cat, A3mb_cat=self.A3mb_cat, S3m_cat=self.S3m_cat,
                A12_cts=self.A12_cts, A43_cts=self.A43_cts, S12_cts=self.S12_cts, S43_cts=self.S43_cts,
                A2ma_cts=self.A2ma_cts, A2mb_cts=self.A2mb_cts, S2m_cts=self.S2m_cts,
                A3ma_cts=self.A3ma_cts, A3mb_cts=self.A3mb_cts, S3m_cts=self.S3m_cts)

            for l in range(nl):
                # Update A_sum/S_sum_sq to account for worker-interaction terms
                A12_sum_l, A43_sum_l, A2ma_sum_l, A2mb_sum_l, A3ma_sum_l, A3mb_sum_l, S12_sum_sq_l, S43_sum_sq_l, S2m_sum_sq_l, S3m_sum_sq_l = self._sum_by_nl_l(
                    ni=ni, l=l, C1=C1, C2=C2, C3=C3, C4=C4,
                    A12_cat=self.A12_cat, A43_cat=self.A43_cat, S12_cat=self.S12_cat, S43_cat=self.S43_cat,
                    A2ma_cat=self.A2ma_cat, A2mb_cat=self.A2mb_cat, S2m_cat=self.S2m_cat,
                    A3ma_cat=self.A3ma_cat, A3mb_cat=self.A3mb_cat, S3m_cat=self.S3m_cat,
                    A12_cts=self.A12_cts, A43_cts=self.A43_cts, S12_cts=self.S12_cts, S43_cts=self.S43_cts,
                    A2ma_cts=self.A2ma_cts, A2mb_cts=self.A2mb_cts, S2m_cts=self.S2m_cts,
                    A3ma_cts=self.A3ma_cts, A3mb_cts=self.A3mb_cts, S3m_cts=self.S3m_cts)
                lp1 = lognormpdf(Y1, A1_sum + A1_sum_l + A1[l, G1], np.sqrt(S1_sum_sq + S1_sum_sq_l + S1[l, G1] ** 2))
                # lp2 = lognormpdf(Y2, A2_sum + A2_sum_l + A2[l, G2], np.sqrt(S2_sum_sq + S2_sum_sq_l + S2[l, G2] ** 2))
                lp_stable[:, l] = W1 * lp1 # + W2 * lp2
        else:
            for l in range(nl):
                lp1 = lognormpdf(Y1, A1[l, G1], S1[l, G1])
                # lp2 = lognormpdf(Y2, A2[l, G2], S2[l, G2])
                lp_stable[:, l] = W1 * lp1 # + W2 * lp2
        del lp1 #, lp2

        for iter in range(params['n_iters_stayers']):

            # ---------- E-Step ----------
            # We compute the posterior probabilities for each row
            # We iterate over the worker types, should not be be
            # too costly since the vector is quite large within each iteration
            for l in range(nl):
                lp[:, l] = lp_stable[:, l] + np.log(pk0[G1, l])

            # We compute log sum exp to get likelihoods and probabilities
            lse_lp = logsumexp(lp, axis=1)
            qi = np.exp(lp.T - lse_lp).T
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
                break
            prev_lik = lik0

            # ---------- M-step ----------
            # NOTE: add dirichlet prior
            pk0 = GG1.T @ (W * (qi.T + d_prior - 1)).T
            # Normalize rows to sum to 1
            pk0 = DxM(1 / np.sum(pk0, axis=1), pk0)

        self.pk0, self.lik0 = pk0, lik0
        self.liks0 = liks0 # np.concatenate([self.liks0, liks0])

        # Update NNs
        if compute_NNs:
            NNs = sdata['g1'].value_counts(sort=False)
            NNs.sort_index(inplace=True)
            self.NNs = NNs.to_numpy()

    def fit_movers_cstr_uncstr(self, jdata, compute_NNm=True):
        '''
        Run fit_movers(), first constrained, then using results as starting values, run unconstrained.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
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
        if self.params['verbose'] in [1, 2, 3]:
            print('Fitting movers with A fixed')
        self.fit_movers(jdata, compute_NNm=False)
        ##### Loop 2 #####
        # Now update A with Linear Additive constraint
        self.params['update_a'] = True
        if self.nl > 1:
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
        # Restore user constraints
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

        for kk in range(nl):
            # Compute adjacency matrix
            A = pr[:, :, kk]
            A /= A.sum()
            A = (A + A.T) / 2
            D = np.diag(np.sum(A, axis=1) ** (-0.5))
            L = np.eye(nk) - D @ A @ D
            try:
                evals, evecs = np.linalg.eig(L)
            except np.linalg.LinAlgError as e:
                warnings.warn("Linear algebra error encountered when computing connectedness measure. This can likely be corrected by increasing the value of 'd_prior_movers' in tw.blm_params().")
                raise np.linalg.LinAlgError(e)
            EV[kk] = sorted(evals)[1]

        if all:
            self.connectedness = EV
        self.connectedness = np.abs(EV).min()

    def plot_log_earnings(self, period='first', grid=True, dpi=None):
        '''
        Plot log-earnings by worker-firm type pairs.

        Arguments:
            period (str): 'first' plots log-earnings in the first period; 'second' plots log-earnings in the second period; 'all' plots the average over log-earnings in the first and second periods
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
        plt.xlabel('firm class k')
        plt.ylabel('log-earnings')
        plt.xticks(x_axis)
        if grid:
            plt.grid()
        plt.show()

    def plot_type_proportions(self, period='first', subset='all', dpi=None):
        '''
        Plot proportions of worker types at each firm class.

        Arguments:
            period (str): 'first' plots type proportions in the first period; 'second' plots type proportions in the second period; 'all' plots the average over type proportions in the first and second periods
            subset (str): 'all' plots a weighted average over movers and stayers; 'movers' plots movers; 'stayers' plots stayers
            dpi (float or None): dpi for plot
        '''
        nl, nk = self.nl, self.nk
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
        ax.set_xlabel('firm class k')
        ax.set_ylabel('type proportions')
        ax.set_title('Proportions of worker types')
        plt.show()

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

    def _fit_model(self, jdata, rng=None):
        '''
        Generate model and run fit_movers_cstr_uncstr() given parameters.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        model = BLMModel(self.params, rng)
        model.fit_movers_cstr_uncstr(jdata)
        return model

    def fit(self, jdata, sdata, n_init=20, n_best=5, ncore=1, rng=None):
        '''
        EM model for movers and stayers.

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
                sim_model_lst = pool.starmap(self._fit_model, tqdm([(jdata, np.random.default_rng(seed)) for seed in seeds], total=n_init))
        else:
            # No multiprocessing
            sim_model_lst = itertools.starmap(self._fit_model, tqdm([(jdata, np.random.default_rng(seed)) for seed in seeds], total=n_init))

        # Sort by likelihoods FIXME better handling if connectedness is None
        sorted_zipped_models = sorted([(model.lik1, model) for model in sim_model_lst if model.connectedness is not None], reverse=True, key=lambda a: a[0])
        sorted_lik_models = [model for _, model in sorted_zipped_models]

        # Save likelihood vs. connectedness for all models
        liks_high = np.zeros(shape=n_best) # Save likelihoods for n_best
        connectedness_high = np.zeros(shape=n_best) # Save connectedness for n_best
        liks_low = np.zeros(shape=n_init - n_best) # Save likelihoods for not n_best
        connectedness_low = np.zeros(shape=n_init - n_best) # Save connectedness for not n_best
        liks_all = [] # Save paths of likelihoods
        for i, model in enumerate(sorted_lik_models):
            liks_all.append(model.liks1)
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
        self.model.fit_stayers(sdata)

    def plot_log_earnings(self, period='first', grid=True, dpi=None):
        '''
        Plot log-earnings by worker-firm type pairs.

        Arguments:
            period (str): 'first' plots log-earnings in the first period; 'second' plots log-earnings in the second period; 'all' plots the average over log-earnings in the first and second periods
            grid (bool): if True, plot grid
            dpi (float or None): dpi for plot
        '''
        if self.model is not None:
            self.model.plot_log_earnings(period=period, grid=grid, dpi=dpi)
        else:
            warnings.warn('Estimation has not yet been run.')

    def plot_type_proportions(self, period='first', subset='all', dpi=None):
        '''
        Plot proportions of worker types at each firm class.

        Arguments:
            period (str): 'first' plots type proportions in the first period; 'second' plots type proportions in the second period; 'all' plots the average over type proportions in the first and second periods
            subset (str): 'all' plots a weighted average over movers and stayers; 'movers' plots movers; 'stayers' plots stayers
            dpi (float or None): dpi for plot
        '''
        if self.model is not None:
            self.model.plot_type_proportions(period=period, subset=subset, dpi=dpi)
        else:
            warnings.warn('Estimation has not yet been run.')

    def plot_liks_connectedness(self, jitter=False, dpi=None):
        '''
        Plot likelihoods vs connectedness for the estimations run.

        Arguments:
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
    '''

    def __init__(self, params):
        self.params = params
        # No initial models
        self.models = None

    def fit(self, jdata, sdata, blm_model=None, n_samples=5, n_init_estimator=20, n_best=5, frac_movers=0.1, frac_stayers=0.1, method='parametric', cluster_params=None, reallocate=False, reallocate_jointly=True, reallocate_period='first', ncore=1, verbose=True, rng=None):
        '''
        Estimate bootstrap.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            blm_model (BLMModel or None): BLM model estimated using true data; if None, estimate model inside the method. For use with parametric bootstrap.
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
            cluster_params = bpd.cluster_params()

        # Update clustering parameters
        cluster_params = cluster_params.copy()
        cluster_params['is_sorted'] = True
        cluster_params['copy'] = False

        if method == 'parametric':
            # Copy original wages and firm types
            yj = jdata.loc[:, ['y1', 'y2']].to_numpy().copy()
            ys = sdata.loc[:, ['y1', 'y2']].to_numpy().copy()
            gj = jdata.loc[:, ['g1', 'g2']].to_numpy().copy()
            gs = sdata.loc[:, 'g1'].to_numpy().copy()
            ## Weights ##
            wj = None
            ws = None
            if jdata._col_included('w'):
                wj = jdata.loc[:, ['w1', 'w2']].to_numpy()
            if sdata._col_included('w'):
                ws = sdata.loc[:, 'w1'].to_numpy()

            if blm_model is None:
                # Run initial BLM estimator
                blm_fit_init = BLMEstimator(self.params)
                blm_fit_init.fit(jdata=jdata, sdata=sdata, n_init=n_init_estimator, n_best=n_best, ncore=ncore, rng=rng)
                blm_model = blm_fit_init.model

            # Run parametric bootstrap
            models = []
            for _ in trange(n_samples):
                # Simulate worker types then draw wages
                yj_i, ys_i = _simulate_types_wages(jdata=jdata, sdata=sdata, gj=gj, gs=gs, blm_model=blm_model, reallocate=reallocate, reallocate_jointly=reallocate_jointly, reallocate_period=reallocate_period, wj=wj, ws=ws, rng=rng)[:2]
                with bpd.util.ChainedAssignment():
                    jdata.loc[:, 'y1'], jdata.loc[:, 'y2'] = (yj_i[0], yj_i[1])
                    sdata.loc[:, 'y1'], sdata.loc[:, 'y2'] = (ys_i, ys_i)
                # Cluster
                bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False)).to_long(is_sorted=True, copy=False).cluster(cluster_params, rng=rng)
                clusters_dict = bdf.loc[:, ['j', 'g']].groupby('j', sort=False)['g'].first().to_dict()
                del bdf
                with bpd.util.ChainedAssignment():
                    # Update clusters in jdata and sdata
                    jdata.loc[:, 'g1'] = jdata.loc[:, 'j1'].map(clusters_dict)
                    jdata.loc[:, 'g2'] = jdata.loc[:, 'j2'].map(clusters_dict)
                    sdata.loc[:, 'g1'] = sdata.loc[:, 'j1'].map(clusters_dict)
                    sdata.loc[:, 'g2'] = sdata.loc[:, 'g1']
                # Run BLM estimator
                blm_fit_i = BLMEstimator(self.params)
                blm_fit_i.fit(jdata=jdata, sdata=sdata, n_init=n_init_estimator, n_best=n_best, ncore=ncore, rng=rng)
                models.append(blm_fit_i.model)
                del blm_fit_i

            with bpd.util.ChainedAssignment():
                # Re-assign original wages and firm types
                jdata.loc[:, ['y1', 'y2']] = yj
                sdata.loc[:, ['y1', 'y2']] = ys
                jdata.loc[:, ['g1', 'g2']] = gj
                sdata.loc[:, 'g1'], sdata.loc[:, 'g2'] = (gs, gs)
        elif method == 'standard':
            wj = None
            if self.params['weighted'] and jdata._col_included('w'):
                wj = jdata['w1'].to_numpy() + jdata['w2'].to_numpy()
            ws = None
            if self.params['weighted'] and sdata._col_included('w'):
                ws = sdata['w1'].to_numpy() + sdata['w2'].to_numpy()

            models = []
            for _ in trange(n_samples):
                jdata_i = jdata.sample(frac=frac_movers, replace=True, weights=wj, random_state=rng)
                sdata_i = sdata.sample(frac=frac_stayers, replace=True, weights=ws, random_state=rng)
                # Cluster
                bdf = bpd.BipartiteDataFrame(pd.concat([jdata_i, sdata_i], axis=0, copy=True)).clean(bpd.clean_params({'is_sorted': True, 'copy': False, 'verbose': verbose})).to_long(is_sorted=True, copy=False).cluster(cluster_params, rng=rng)
                clusters_dict = bdf.loc[:, ['j', 'g']].groupby('j', sort=False)['g'].first().to_dict()
                del bdf
                # Update clusters in jdata_i and sdata_i
                jdata_i.loc[:, 'g1'] = jdata_i.loc[:, 'j1'].map(clusters_dict)
                jdata_i.loc[:, 'g2'] = jdata_i.loc[:, 'j2'].map(clusters_dict)
                sdata_i.loc[:, 'g1'] = sdata_i.loc[:, 'j1'].map(clusters_dict)
                sdata_i.loc[:, 'g2'] = sdata_i.loc[:, 'g1']
                # Run BLM estimator
                blm_fit_i = BLMEstimator(self.params)
                blm_fit_i.fit(jdata=jdata_i, sdata=sdata_i, n_init=n_init_estimator, n_best=n_best, ncore=ncore, rng=rng)
                models.append(blm_fit_i.model)
                del jdata_i, sdata_i, blm_fit_i
        else:
            raise ValueError(f"`method` must be one of 'parametric' or 'standard', but input specifies {method!r}.")

        self.models = models

    def plot_log_earnings(self, period='first', grid=True, dpi=None):
        '''
        Plot log-earnings by worker-firm type pairs.

        Arguments:
            period (str): 'first' plots log-earnings in the first period; 'second' plots log-earnings in the second period; 'all' plots the average over log-earnings in the first and second periods
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
            plt.xlabel('firm class k')
            plt.ylabel('log-earnings')
            plt.xticks(x_axis)
            if grid:
                plt.grid()
            plt.show()

    def plot_type_proportions(self, period='first', subset='all', dpi=None):
        '''
        Plot proportions of worker types at each firm class.

        Arguments:
            period (str): 'first' plots type proportions in the first period; 'second' plots type proportions in the second period; 'all' plots the average over type proportions in the first and second periods
            subset (str): 'all' plots a weighted average over movers and stayers; 'movers' plots movers; 'stayers' plots stayers
            dpi (float or None): dpi for plot
        '''
        if self.models is None:
            warnings.warn('Estimation has not yet been run.')
        else:
            nl, nk = self.params.get_multiple(('nl', 'nk'))

            # Compute average type proportions
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
            ax.set_xlabel('firm class k')
            ax.set_ylabel('type proportions')
            ax.set_title('Proportions of worker types')
            plt.show()

class BLMVarianceDecomposition:
    '''
    Class for estimating BLM variance decomposition using bootstrapping.

    Arguments:
        params (ParamsDict): dictionary of parameters for BLM estimation. Run tw.blm_params().describe_all() for descriptions of all valid parameters.
    '''

    def __init__(self, params):
        self.params = params
        # No initial results
        self.res = None

    def fit(self, jdata, sdata, blm_model=None, n_samples=5, n_init_estimator=20, n_best=5, reallocate=False, reallocate_jointly=True, reallocate_period='first', Q_var=None, Q_cov=None, firm_clusters_as_ids=True, worker_types_as_ids=True, ncore=1, rng=None):
        '''
        Estimate variance decomposition.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            blm_model (BLMModel or None): BLM model estimated using true data; if None, estimate model inside the method
            n_samples (int): number of bootstrap samples to estimate
            n_init_estimator (int): number of starting guesses to estimate for each bootstrap sample
            n_best (int): take the n_best estimates with the highest likelihoods, and then take the estimate with the highest connectedness, for each bootstrap sample
            reallocate (bool): if True, draw worker type proportions independently of firm type; if False, uses worker type proportions that are conditional on firm type
            reallocate_jointly (bool): if True, worker type proportions take the average over movers and stayers (i.e. all workers use the same type proportions); if False, consider movers and stayers separately
            reallocate_period (str): if 'first', compute type proportions based on first period parameters; if 'second', compute type proportions based on second period parameters; if 'all', compute type proportions based on average over first and second period parameters
            Q_var (list of Q variances): list of Q matrices to use when estimating variance term; None is equivalent to tw.Q.VarPsi() without controls, or tw.Q.VarCovariate('psi') with controls
            Q_cov (list of Q covariances): list of Q matrices to use when estimating covariance term; None is equivalent to tw.Q.CovPsiAlpha() without controls, or tw.Q.CovCovariate('psi', 'alpha') with controls
            firm_clusters_as_ids (bool): if True, regress on firm clusters; if False, regress on firm ids
            worker_types_as_ids (bool): if True, regress on true, simulated worker types; if False, regress on worker ids
            ncore (int): number of cores for multiprocessing
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # FE parameters
        no_cat_controls = (self.params['categorical_controls'] is None) or (len(self.params['categorical_controls']) == 0)
        no_cts_controls = (self.params['continuous_controls'] is None) or (len(self.params['continuous_controls']) == 0)
        no_controls = (no_cat_controls and no_cts_controls)
        if no_controls:
            # If no controls
            fe_params = tw.fe_params()
        else:
            # If controls
            fe_params = tw.fecontrol_params()
            if not no_cat_controls:
                fe_params['categorical_controls'] = self.params['categorical_controls'].keys()
            if not no_cts_controls:
                fe_params['continuous_controls'] = self.params['continuous_controls'].keys()
        fe_params['ho'] = False
        if Q_var is not None:
            fe_params['Q_var'] = Q_var
        if Q_cov is not None:
            fe_params['Q_cov'] = Q_cov

        # Copy original wages, firm types, and optionally ids
        yj = jdata.loc[:, ['y1', 'y2']].to_numpy().copy()
        ys = sdata.loc[:, ['y1', 'y2']].to_numpy().copy()
        gj = jdata.loc[:, ['g1', 'g2']].to_numpy()
        gs = sdata.loc[:, 'g1'].to_numpy()
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

        ## Weights ##
        wj = None
        ws = None
        if jdata._col_included('w'):
            wj = jdata.loc[:, ['w1', 'w2']].to_numpy()
        if sdata._col_included('w'):
            ws = sdata.loc[:, 'w1'].to_numpy()

        if blm_model is None:
            # Run initial BLM estimator
            blm_fit_init = BLMEstimator(self.params)
            blm_fit_init.fit(jdata=jdata, sdata=sdata, n_init=n_init_estimator, n_best=n_best, ncore=ncore, rng=rng)
            blm_model = blm_fit_init.model

        # Run bootstrap
        res_lst = []
        for i in trange(n_samples):
            # Simulate worker types then draw wages
            yj_i, ys_i, Lm_i, Ls_i = _simulate_types_wages(jdata=jdata, sdata=sdata, gj=gj, gs=gs, blm_model=blm_model, reallocate=reallocate, reallocate_jointly=reallocate_jointly, reallocate_period=reallocate_period, wj=wj, ws=ws, rng=rng)
            with bpd.util.ChainedAssignment():
                if worker_types_as_ids:
                    jdata.loc[:, 'i'] = Lm_i
                    sdata.loc[:, 'i'] = Ls_i
                jdata.loc[:, 'y1'], jdata.loc[:, 'y2'] = (yj_i[0], yj_i[1])
                sdata.loc[:, 'y1'], sdata.loc[:, 'y2'] = (ys_i, ys_i)
            # Convert to BipartitePandas DataFrame
            bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
            # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
            bdf._set_attributes(jdata)
            bdf = bdf.to_long(is_sorted=True, copy=False)
            # Estimate OLS
            if no_controls:
                fe_estimator = tw.FEEstimator(bdf, fe_params)
            else:
                fe_estimator = tw.FEControlEstimator(bdf, fe_params)
            fe_estimator.fit()
            res_lst.append(fe_estimator.summary)

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

        # Remove '_fe' from result names
        res = {k.replace('_fe', ''): v for k, v in res.items()}

        # Drop time column
        if tj:
            jdata = jdata.drop('t', axis=1, inplace=True, allow_optional=True)
        if ts:
            sdata = sdata.drop('t', axis=1, inplace=True, allow_optional=True)

        self.res = res
