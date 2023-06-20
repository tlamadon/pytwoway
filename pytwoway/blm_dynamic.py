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
from scipy.sparse import csc_matrix, lil_matrix, hstack, vstack
from scipy.optimize import minimize as opt
from matplotlib import pyplot as plt
from paramsdict import ParamsDict, ParamsDictBase
from paramsdict.util import col_type
import bipartitepandas as bpd
from bipartitepandas.util import to_list, HiddenPrints # , _is_subtype
import pytwoway as tw
from pytwoway import constraints as cons
from pytwoway.util import weighted_quantile, DxSP, DxM, diag_of_sp_prod, jitter_scatter, logsumexp, lognormpdf, fast_lognormpdf

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
dynamic_blm_params = ParamsDict({
    ### Class parameters ###
    'nl': (6, 'type_constrained', (int, _gteq1),
        '''
            (default=6) Number of worker types.
        ''', '>= 1'),
    'nk': (None, 'type_constrained_none', (int, _gteq1),
        '''
            (default=None) Number of firm types. None will raise an error when running the estimator.
        ''', '>= 1'),
    'endogeneity': (True, 'type', bool,
        '''
            (default=True) If True, estimate model with endogeneity (i.e. the firm type after a move can affect earnings before the move).
        ''', None),
    'state_dependence': (True, 'type', bool,
        '''
            (default=True) If True, estimate model with state dependence (i.e. the firm type before a move can affect earnings after the move).
        ''', None),
    'categorical_controls': (None, 'dict_of_type_none', ParamsDictBase,
        '''
            (default=None) Dictionary linking column names to instances of tw.dynamic_categorical_control_params(). Each instance specifies a new categorical control variable and how its starting values should be generated. Run tw.dynamic_categorical_control_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
        ''', None),
    'continuous_controls': (None, 'dict_of_type_none', ParamsDictBase,
        '''
            (default=None) Dictionary linking column names to instances of tw.dynamic_continuous_control_params(). Each instance specifies a new continuous control variable and how its starting values should be generated. Run tw.dynamic_continuous_control_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
        ''', None),
    'primary_period': ('first', 'set', ['first', 'second', 'all'],
        '''
            (default='first') Period to normalize and sort over. 'first' uses first period parameters; 'second' uses second period parameters; 'all' uses the average over first and second period parameters.
        ''', None),
    'verbose': (1, 'set', [0, 1, 2, 3],
        '''
            (default=1) If 0, print no output; if 1, print each major step in estimation; if 2, print warnings during estimation; if 3, print likelihoods at each iteration.
        ''', None),
    ### Starting values ###
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
    ### fit_movers() parameters ###
    'n_iters_movers': (1000, 'type_constrained', (int, _gteq1),
        '''
            (default=1000) Maximum number of EM iterations for movers.
        ''', '>= 1'),
    'threshold_movers': (1e-7, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1e-7) Threshold to break EM loop for movers.
        ''', '>= 0'),
    'update_a_movers': (True, 'type', bool,
        '''
            (default=True) If False, do not update A12/A43/A2ma/A2mb/A3ma/A3mb.
        ''', None),
    'update_s_movers': (True, 'type', bool,
        '''
            (default=True) If False, do not update S12/S43/S2ma/S2mb/S3ma/S3mb.
        ''', None),
    'update_pk1': (True, 'type', bool,
        '''
            (default=True) If False, do not update pk1.
        ''', None),
    'update_rho12': (True, 'type', bool,
        '''
            (default=True) If False, do not update rho12.
        ''', None),
    'update_rho43': (True, 'type', bool,
        '''
            (default=True) If False, do not update rho43.
        ''', None),
    'update_rho32m': (True, 'type', bool,
        '''
            (default=True) If False, do not update rho32 for movers.
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
    'd_X_diag_movers_A': (1 + 1e-10, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1 + 1e-10) Account for numerical rounding causing X'X to not be positive definite when computing A by adding (d_X_diag_movers_A - 1) to the diagonal of X'X.
        ''', '>= 1'),
    'd_X_diag_movers_S': (1 + 1e-5, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1 + 1e-5) Account for numerical rounding causing X'X to not be positive definite when computing S by adding (d_X_diag_movers_S - 1) to the diagonal of X'X.
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
    ### fit_stayers() parameters ###
    'n_iters_stayers': (1000, 'type_constrained', (int, _gteq1),
        '''
            (default=1000) Maximum number of EM iterations for stayers.
        ''', '>= 1'),
    'threshold_stayers': (1e-7, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=1e-7) Threshold to break EM loop for stayers.
        ''', '>= 0'),
    'update_a_stayers': (True, 'type', bool,
        '''
            (default=True) If False, do not update A2s/A3s.
        ''', None),
    'update_s_stayers': (True, 'type', bool,
        '''
            (default=True) If False, do not update S2s/A3s.
        ''', None),
    'update_pk0': (True, 'type', bool,
        '''
            (default=True) If False, do not update pk0.
        ''', None),
    'update_rho32s': (True, 'type', bool,
        '''
            (default=True) If False, do not update rho32 for stayers.
        ''', None),
    'd_prior_stayers': (1 + 1e-7, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1 + 1e-7) Account for probabilities being too small by adding (d_prior - 1) to pk0.
        ''', '>= 1'),
    'd_X_diag_stayers': (1 + 1e-10, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1 + 1e-10) Account for numerical rounding causing X'X to not be positive definite by adding (d_X_diag_stayers - 1) to the diagonal of X'X.
        ''', '>= 1')
})

dynamic_categorical_control_params = ParamsDict({
    'n': (None, 'type_constrained_none', (int, _gteq2),
        '''
            (default=6) Number of types for the parameter. None will raise an error when running the estimator.
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

dynamic_continuous_control_params = ParamsDict({
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

def double_bincount(group1, group2, weights=None):
    '''
    Perform groupby-sum on 2 groups with given weights and return the corresponding matrix representation.

    Arguments:
        group1 (NumPy Array): first group
        group2 (NumPy Array): second group
        weights (NumPy Array or None): weights; None is equivalent to all weights being equal

    Returns:
        (NumPy Array): groupby-sum matrix representation
    '''
    if weights is None:
        df = pd.DataFrame({'g1': group1, 'g2': group2})
        return df.groupby('g1')['g2'].value_counts().unstack(fill_value=0).to_numpy()
    df = pd.DataFrame({'g1': group1, 'g2': group2, 'w': weights})
    return df.groupby(['g1', 'g2'])['w'].sum().unstack(fill_value=0).to_numpy()

def _var_stayers(sdata, rho_1, rho_4, rho_t, weights=None, diff=False):
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
    cons_lm.solve(DpXX.T @ XX, -(DpXX.T @ YY), solver='quadprog')

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

def rho_init(sdata, rho_0=(0.6, 0.6, 0.6), weights=None, diff=False):
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
    rho_optim_fn = lambda rhos: _var_stayers(sdata, *rhos, weights=weights, diff=diff)['eps_sq']
    # Initialize and fit optimizer
    optim_fn = opt(fun=rho_optim_fn, x0=rho_0, method='BFGS')
    # Extract optimal rho
    rho_optim = optim_fn.x
    # Run _var_stayers with optimal rho
    return _var_stayers(sdata, *rho_optim, weights=weights, diff=diff)

def _simulate_types_wages(jdata, sdata, gj, gs, blm_model, reallocate=False, reallocate_jointly=True, reallocate_period='first', rng=None):
    '''
    Using data and estimated dynamic BLM parameters, simulate worker types and wages.

    Arguments:
        jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
        sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
        gj (NumPy Array): mover firm types for both periods
        gs (NumPy Array): stayer firm types for the first period
        blm_model (DynamicBLMModel): dynamic BLM model with estimated parameters
        reallocate (bool): if True, draw worker type proportions independently of firm type; if False, uses worker type proportions that are conditional on firm type
        reallocate_jointly (bool): if True, worker type proportions take the average over movers and stayers (i.e. all workers use the same type proportions); if False, consider movers and stayers separately
        reallocate_period (str): if 'first', compute type proportions based on first period parameters; if 'second', compute type proportions based on second period parameters; if 'all', compute type proportions based on average over first and second period parameters
        rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

    Returns:
        (tuple of NumPy Arrays): (yj --> tuple of wages for movers, where element gives that period's wages; ys --> tuple of wages for movers, where element gives that period's wages; Lm --> vector of mover types; Ls --> vector of stayer types)
    '''
    if rng is None:
        rng = np.random.default_rng(None)

    nl, nk, nmi, nsi = blm_model.nl, blm_model.nk, len(jdata), len(sdata)
    A, A_cat, A_cts = blm_model.A, blm_model.A_cat, blm_model.A_cts
    S, S_cat, S_cts = blm_model.S, blm_model.S_cat, blm_model.S_cts
    R12, R43, R32m, R32s = blm_model.R12, blm_model.R43, blm_model.R32m, blm_model.R32s
    pk1, pk0 = blm_model.pk1, blm_model.pk0
    controls_dict, cat_cols, cts_cols = blm_model.controls_dict, blm_model.cat_cols, blm_model.cts_cols
    periods_movers = ['12', '43', '2ma', '3ma', '2mb', '3mb']
    periods_stayers = ['12', '43', '2ma', '3ma', '2s', '3s']
    first_periods = ['12', '2ma', '3mb', '2s']
    second_periods = ['43', '2mb', '3ma', '3s']
    periods_movers_dict = {period: 0 if period in first_periods else 1 for period in periods_movers}
    periods_stayers_dict = {period: 0 if period in first_periods else 1 for period in periods_stayers}

    # Correct datatype for gj and gs
    gj = gj.astype(int, copy=False)
    gs = gs.astype(int, copy=False)

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

    A_sum = {period:
                A[period][Lm, gj[:, periods_movers_dict[period]]]
                    if period[-1] != 'b' else
                A[period][gj[:, periods_movers_dict[period]]]
            for period in periods_movers}
    S_sum_sq = {period:
                    S[period][Lm, gj[:, periods_movers_dict[period]]] ** 2
                        if period[-1] != 'b' else
                    S[period][gj[:, periods_movers_dict[period]]] ** 2
                for period in periods_movers}

    #### Simulate control variable wages ####
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
                        A_sum[period] += A_cat[col][period][Lm, jdata.loc[:, subcol]]
                        S_sum_sq[period] += S_cat[col][period][Lm, jdata.loc[:, subcol]] ** 2
                    else:
                        A_sum[period] += A_cat[col][period][jdata.loc[:, subcol]]
                        S_sum_sq[period] += S_cat[col][period][jdata.loc[:, subcol]] ** 2
            else:
                ## Non-worker-interaction ##
                for period in periods_movers:
                    subcol = periods_movers_dict[period]
                    A_sum[period] += A_cat[col][period][jdata.loc[:, subcol]]
                    S_sum_sq[period] += S_cat[col][period][jdata.loc[:, subcol]] ** 2
        else:
            ### Continuous ###
            if controls_dict[col]['worker_type_interaction']:
                ## Worker-interaction ##
                for period in periods_movers:
                    subcol = periods_movers_dict[period]
                    if period[-1] != 'b':
                        A_sum[period] += A_cts[col][period][Lm] * jdata.loc[:, subcol]
                        S_sum_sq[period] += S_cts[col][period][Lm] ** 2
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
        size=nmi) # scale=np.sqrt(S_sum_sq['2ma'] + S_sum_sq['2mb']), \
    Y1 = rng.normal( \
        loc=A_sum['12'] + R12 * (Y2 - A_sum['2ma']), \
        scale=np.sqrt(S_sum_sq['12']), \
        size=nmi) # scale=np.sqrt(S_sum_sq['12'] + (R12 ** 2) * S_sum_sq['2ma']), \
    Y3 = rng.normal( \
        loc=A_sum['3ma'] + A_sum['3mb'] + R32m * (Y2 - A_sum['2ma'] - A_sum['2mb']), \
        scale=np.sqrt(S_sum_sq['3ma']), \
        size=nmi) # scale=np.sqrt(S_sum_sq['3ma'] + S_sum_sq['3mb'] + (R32m ** 2) * (S_sum_sq['2ma'] + S_sum_sq['2mb'])), \
    Y4 = rng.normal( \
        loc=A_sum['43'] + R43 * (Y3 - A_sum['3ma']), \
        scale=np.sqrt(S_sum_sq['43']), \
        size=nmi) # scale=np.sqrt(S_sum_sq['43'] + (R43 ** 2) * S_sum_sq['3ma']), \
    yj = (Y1, Y2, Y3, Y4)
    del A_sum, S_sum_sq, Y1, Y2, Y3, Y4

    ## Stayers ##
    Ls = np.zeros(shape=len(sdata), dtype=int)
    for k in range(nk):
        ## Iterate over all firm types a worker can work at ##
        # Find movers who work at this firm type
        rows_k = np.where(gs == k)[0]
        ni = len(rows_k)

        # Draw worker types
        Ls[rows_k] = rng.choice(worker_types, size=ni, replace=True, p=pk0[k, :])

    A_sum = {period:
                A[period][Ls, gs]
                    if period[-1] != 'b' else
                A[period][gs]
            for period in periods_stayers}
    S_sum_sq = {period:
                    S[period][Ls, gs] ** 2
                        if period[-1] != 'b' else
                    S[period][gs] ** 2
                for period in periods_stayers}

    if len(controls_dict) > 0:
        #### Simulate control variable wages ####
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
                    for period in periods_stayers:
                        subcol = periods_stayers_dict[period]
                        A_sum[period] += A_cat[col][period][Ls, sdata.loc[:, subcol]]
                        S_sum_sq[period] += S_cat[col][period][Ls, sdata.loc[:, subcol]] ** 2
                else:
                    ## Non-worker-interaction ##
                    for period in periods_stayers:
                        subcol = periods_stayers_dict[period]
                        A_sum[period] += A_cat[col][period][sdata.loc[:, subcol]]
                        S_sum_sq[period] += S_cat[col][period][sdata.loc[:, subcol]] ** 2
            else:
                ### Continuous ###
                if controls_dict[col]['worker_type_interaction']:
                    ## Worker-interaction ##
                    for period in periods_stayers:
                        subcol = periods_stayers_dict[period]
                        A_sum[period] += A_cts[col][period][Ls] * sdata.loc[:, subcol]
                        S_sum_sq[period] += S_cts[col][period][Ls] ** 2
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
        size=nsi) # scale=np.sqrt(S_sum_sq['12'] + (R12 ** 2) * S_sum_sq['2ma']), \
    Y3 = rng.normal( \
        loc=A_sum['3s'] + R32s * (Y2 - A_sum['2s']), \
        scale=np.sqrt(S_sum_sq['3s']), \
        size=nsi) # scale=np.sqrt(S_sum_sq['3s'] + (R32s ** 2) * S_sum_sq['2s']), \
    Y4 = rng.normal( \
        loc=A_sum['43'] + R43 * (Y3 - A_sum['3ma']), \
        scale=np.sqrt(S_sum_sq['43']), \
        size=nsi) # scale=np.sqrt(S_sum_sq['43'] + (R43 ** 2) * S_sum_sq['3ma']), \
    ys = (Y1, Y2, Y3, Y4)

    return (yj, ys, Lm, Ls)

class DynamicBLMModel:
    '''
    Class for estimating dynamic BLM using a single set of starting values.

    Arguments:
        params (ParamsDict): dictionary of parameters for dynamic BLM estimation. Run tw.dynamic_blm_params().describe_all() for descriptions of all valid parameters.
        rhos (dict of floats): rho values (persistance parameters) estimated using stayers; must contain keys 'rho_1' and 'rho_4'
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
            raise ValueError(f"tw.dynamic_blm_params() key 'nk' must be changed from the default value of None.")
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
        # Periods to estimate
        all_periods = ['12', '43', '2ma', '3ma', '2mb', '3mb', '2s', '3s']
        periods = ['12', '43', '2ma', '3ma', '2mb', '3mb', '2s', '3s']
        if not params['endogeneity']:
            periods.remove('2mb')
        if not params['state_dependence']:
            periods.remove('3mb')
        self.all_periods = all_periods
        self.periods = periods
        self.periods_movers = [period for period in ['12', '43', '2ma', '3ma', '2mb', '3mb'] if period in periods]
        self.periods_variance = ['12', '43', '2ma', '3ma']
        self.periods_stayers = ['12', '43', '2ma', '3ma', '2s', '3s']
        self.periods_update_stayers = ['2s', '3s']
        self.first_periods = ['12', '2ma', '3mb', '2s']
        self.second_periods = ['43', '2mb', '3ma', '3s']

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
        self.any_controls = (len(control_cols) > 0)
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
        self.R32s = 0.6
        # We simulate starting values for everything else
        self.A = {
            period:
                rng.normal(loc=params[f'a{period}_mu'], scale=params[f'a{period}_sig'], size=dims)
                    if (period[-1] != 'b') else
                rng.normal(loc=params[f'a{period}_mu'], scale=params[f'a{period}_sig'], size=nk)
            for period in all_periods
        }
        self.S = {
            period:
                rng.uniform(low=max(params[f's{period}_low'], s_lb), high=params[f's{period}_high'], size=dims)
                    if (period[-1] != 'b') else
                rng.uniform(low=max(params[f's{period}_low'], s_lb), high=params[f's{period}_high'], size=nk)
            for period in all_periods
        }
        # Model for p(K | l, l') for movers
        if params['pk1_prior'] is None:
            pk1_prior = np.ones(nl)
        else:
            pk1_prior = params['pk1_prior']
        self.pk1 = rng.dirichlet(alpha=pk1_prior, size=nk ** 2)
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
                for period in all_periods
            }
            for col in cat_cols
        }
        self.S_cat = {
            col: {
                period:
                    rng.uniform(low=max(controls_dict[col][f's{period}_low'], s_lb), high=controls_dict[col][f's{period}_high'], size=(nl, controls_dict[col]['n']))
                        if (controls_dict[col]['worker_type_interaction'] and (period[-1] != 'b')) else
                    rng.uniform(low=max(controls_dict[col][f's{period}_low'], s_lb), high=controls_dict[col][f's{period}_high'], size=controls_dict[col]['n'])
                for period in all_periods
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
                for period in all_periods
            }
            for col in cts_cols
        }
        self.S_cts = {
            col: {
                period:
                    rng.uniform(low=max(controls_dict[col][f's{period}_low'], s_lb), high=controls_dict[col][f's{period}_high'], size=nl)
                        if (controls_dict[col]['worker_type_interaction'] and (period[-1] != 'b')) else
                    rng.uniform(low=max(controls_dict[col][f's{period}_low'], s_lb), high=controls_dict[col][f's{period}_high'], size=1)
                for period in all_periods
            }
            for col in cts_cols
        }

        ## Set parameters to 0 for endogeneity and state dependence ##
        if not params['endogeneity']:
            self.A['2mb'][:] = 0
            self.S['2mb'][:] = 0
            for col in cat_cols:
                self.A_cat['2mb'][:] = 0
                self.S_cat['2mb'][:] = 0
            for col in cts_cols:
                self.A_cts['2mb'] = 0
                self.S_cts['2mb'] = 0
        if not params['state_dependence']:
            self.A['3mb'][:] = 0
            self.S['3mb'][:] = 0
            for col in cat_cols:
                self.A_cat['3mb'][:] = 0
                self.S_cat['3mb'][:] = 0
            for col in cts_cols:
                self.A_cts['3mb'] = 0
                self.S_cts['3mb'] = 0

        ## NNm and NNs ##
        self.NNm = None
        self.NNs = None

    def _min_firm_type(self, A):
        '''
        Find the lowest firm type.

        Arguments:
            A (dict of NumPy Arrays): dictionary linking periods to the mean of fixed effects in that period

        Returns:
            (int): lowest firm type
        '''
        params = self.params

        # Compute parameters from primary period
        if params['primary_period'] == 'first':
            A_mean = A['12']
        elif params['primary_period'] == 'second':
            A_mean = A['43']
        elif params['primary_period'] == 'all':
            A_mean = (A['12'] + A['43']) / 2

        # Return lowest firm type
        return np.mean(A_mean, axis=0).argsort()[0]

    def _gen_constraints(self, min_firm_type, for_movers):
        '''
        Generate constraints for estimating A and S in fit_movers() and fit_stayers().

        Arguments:
            min_firm_type (int): lowest firm type
            for_movers (bool): if True, generate constraints for movers; if False, generate constraints for stayers

        Returns:
            (tuple of constraints): (cons_a --> constraints for base A1 and A2, cons_s --> constraints for base S1 and S2, cons_a_dict --> constraints for A1 and A2 for control variables, cons_s_dict --> controls for S1 and S2 for control variables)
        '''
        # Unpack parameters
        params = self.params
        nl, nk = self.nl, self.nk
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        controls_dict = self.controls_dict

        ## Number of periods for constraints ##
        # If endogeneity or state-dependence, estimate 'b' parameters that do not interact with worker type
        constrain_b = False
        if for_movers:
            nt = len(self.periods_movers)
            if params['endogeneity'] or params['state_dependence']:
                # If endogeneity or state-dependence, estimate 'b' parameters that do not interact with worker type
                nnt_b = []
                for i, period in enumerate(self.periods_movers):
                    if period[-1] == 'b':
                        nnt_b.append(i)
                constrain_b = (len(nnt_b) > 0)
            nt_S = len(self.periods_variance)
        else:
            nt = len(self.periods_update_stayers)
            nt_S = nt

        ## General ##
        cons_a = cons.QPConstrained(nl, nk)
        cons_s = cons.QPConstrained(nl, nk)
        cons_s.add_constraints(cons.BoundedBelow(lb=params['s_lower_bound'], nt=nt_S))
        if constrain_b:
            cons_a.add_constraints(cons.NoWorkerTypeInteraction(nnt=nnt_b, nt=nt, dynamic=True))
            # cons_s.add_constraints(cons.NoWorkerTypeInteraction(nnt=nnt_b, nt=4, dynamic=True))
            # Normalize 2mb and 3mb so the lowest effect is 0 (otherwise these are free parameters)
            cons_a.add_constraints(cons.NormalizeLowest(min_firm_type=min_firm_type, nnt=nnt_b, nt=nt, dynamic=True))

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
            cons_s_dict[col].add_constraints(cons.BoundedBelow(lb=params['s_lower_bound'], nt=nt_S))

            if not controls_dict[col]['worker_type_interaction']:
                cons_a_dict[col].add_constraints(cons.NoWorkerTypeInteraction(nt=nt, dynamic=True))
                cons_s_dict[col].add_constraints(cons.NoWorkerTypeInteraction(nt=nt_S, dynamic=True))
            elif constrain_b:
                cons_a_dict[col].add_constraints(cons.NoWorkerTypeInteraction(nnt=nnt_b, nt=nt, dynamic=True))
                if col in cat_cols:
                    # Normalize 2mb and 3mb so the lowest effect is 0 (otherwise these are free parameters)
                    cons_a_dict[col].add_constraints(cons.NormalizeLowest(min_firm_type=min_firm_type, nnt=nnt_b, nt=nt, dynamic=True))

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
                cons_a.add_constraints(cons.MonotonicMean(md=params['d_mean_worker_effect'], cross_period_mean=True, nnt=pp, nt=nt, dynamic=True))
                if params['normalize']:
                    ## Lowest firm type ##
                    if params['force_min_firm_type'] and params['force_min_firm_type_constraint']:
                        cons_a.add_constraints(cons.MinFirmType(min_firm_type=min_firm_type, md=params['d_mean_firm_effect'], is_min=True, cross_period_mean=True, nnt=pp, nt=nt, dynamic=True))
                    ## Normalize ##
                    if any_tv_wi:
                        # Normalize everything
                        cons_a.add_constraints(cons.NormalizeAll(min_firm_type=min_firm_type, nnt=range(nt), nt=nt, dynamic=True))
                    else:
                        if any_tnv_wi:
                            # Normalize primary period
                            cons_a.add_constraints(cons.NormalizeAll(min_firm_type=min_firm_type, cross_period_normalize=True, nnt=pp, nt=nt, dynamic=True))
                            if any_tv_nwi:
                                # Normalize lowest type pair in each period
                                cons_a.add_constraints(cons.NormalizeLowest(min_firm_type=min_firm_type, nnt=range(nt), nt=nt, dynamic=True))
                        else:
                            if any_tv_nwi:
                                # Normalize lowest type pair in each period
                                cons_a.add_constraints(cons.NormalizeLowest(min_firm_type=min_firm_type, nnt=range(nt), nt=nt, dynamic=True))
                            elif any_tnv_nwi:
                                # Normalize lowest type pair in primary period
                                cons_a.add_constraints(cons.NormalizeLowest(min_firm_type=min_firm_type, cross_period_normalize=True, nnt=pp, nt=nt, dynamic=True))

        return (cons_a, cons_s, cons_a_dict, cons_s_dict)

    def _sort_parameters(self, A, S=None, A_cat=None, S_cat=None, A_cts=None, S_cts=None, pk1=None, pk0=None, NNm=None, NNs=None, sort_firm_types=False, reverse=False):
        '''
        Sort parameters by worker type order (and optionally firm type order).

        Arguments:
            A (dict of NumPy Arrays): dictionary linking periods to the mean of fixed effects in that period
            S (dict of NumPy Arrays or None): dictionary linking periods to the standard deviation of fixed effects in that period; if None, S is not sorted or returned
            A_cat (dict of dicts of NumPy Arrays or None): dictionary linking each categorical control column name to a dictionary linking periods to the mean of fixed effects in that period; if None, A_cat is not sorted or returned
            S_cat (dict of dicts of NumPy Arrays or None): dictionary linking each categorical control column name to a dictionary linking periods to the standard deviation of fixed effects in that period; if None, S_cat is not sorted or returned
            A_cts (dict of dicts of NumPy Arrays or None): dictionary linking each continuous control column name to a dictionary linking periods to the mean of fixed effects in that period; if None, A_cts is not sorted or returned
            S_cts (dict of dicts of NumPy Arrays or None): dictionary linking each continuous control column name to a dictionary linking periods to the standard deviation of fixed effects in that period; if None, S_cts is not sorted or returned
            pk1 (NumPy Array or None): probability of being at each combination of firm types for movers; if None, pk1 is not sorted or returned
            pk0 (NumPy Array or None): probability of being at each firm type for stayers; if None, pk0 is not sorted or returned
            NNm (NumPy Array or None): the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3); if None, NNm is not sorted or returned
            NNs (NumPy Array or None): the number of stayers at each firm type (e.g. entry (1) gives the number of stayers at firm type 1); if None, NNs is not sorted or returned
            sort_firm_types (bool): if True, also sort by firm type order
            reverse (bool): if True, sort in reverse order

        Returns (tuple of NumPy Arrays and dicts of NumPy Arrays): sorted parameters that are not None from (A, S, A_cat, S_cat, A_cts, S_cts, pk1, pk0, NNm, NNs)
        '''
        # Copy parameters
        A, S, A_cat, S_cat, A_cts, S_cts, pk1, pk0, NNm, NNs = copy.deepcopy((A, S, A_cat, S_cat, A_cts, S_cts, pk1, pk0, NNm, NNs))
        # Unpack attributes
        params = self.params
        nl, nk = self.nl, self.nk
        controls_dict = self.controls_dict
        ## Primary period ##
        if params['primary_period'] == 'first':
            A_mean = A['12']
        elif params['primary_period'] == 'second':
            A_mean = A['43']
        elif params['primary_period'] == 'all':
            A_mean = (A['12'] + A['43']) / 2

        ## Sort worker types ##
        worker_type_order = np.mean(A_mean, axis=1).argsort()
        if reverse:
            worker_type_order = list(reversed(worker_type_order))
        if np.any(worker_type_order != np.arange(nl)):
            # Sort if out of order
            A = {k: v[worker_type_order, :] if (k[-1] != 'b') else v for k, v in A.items()}
            if S is not None:
                S = {k: v[worker_type_order, :] if (k[-1] != 'b') else v for k, v in S.items()}
            if pk1 is not None:
                pk1 = pk1[:, worker_type_order]
            if pk0 is not None:
                pk0 = pk0[:, worker_type_order]
            # Sort control variables #
            if A_cat is not None:
                for col in A_cat.keys():
                    A_cat[col] = {k: v[worker_type_order, :] if (controls_dict[col]['worker_type_interaction'] and (k[-1] != 'b')) else v for k, v in A_cat[col].items()}
            if S_cat is not None:
                for col in S_cat.keys():
                    S_cat[col] = {k: v[worker_type_order, :] if (controls_dict[col]['worker_type_interaction'] and (k[-1] != 'b')) else v for k, v in S_cat[col].items()}
            if A_cts is not None:
                for col in A_cts.keys():
                    A_cts[col] = {k: v[worker_type_order] if (controls_dict[col]['worker_type_interaction'] and (k[-1] != 'b')) else v for k, v in A_cts[col].items()}
            if S_cts is not None:
                for col in S_cts.keys():
                    S_cts[col] = {k: v[worker_type_order] if (controls_dict[col]['worker_type_interaction'] and (k[-1] != 'b')) else v for k, v in S_cts[col].items()}

        if sort_firm_types:
            ## Sort firm types ##
            firm_type_order = np.mean(A_mean, axis=0).argsort()
            if reverse:
                firm_type_order = list(reversed(firm_type_order))
            if np.any(firm_type_order != np.arange(nk)):
                # Sort if out of order
                A = {k: v[:, firm_type_order] if (k[-1] != 'b') else v[firm_type_order] for k, v in A.items()}
                if S is not None:
                    S = {k: v[:, firm_type_order] if (k[-1] != 'b') else v[firm_type_order] for k, v in S.items()}
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

        res = [a for a in (A, S, A_cat, S_cat, A_cts, S_cts, pk1, pk0, NNm, NNs) if a is not None]
        if len(res) == 1:
            res = res[0]

        return res

    def _normalize(self, A, A_cat):
        '''
        Normalize means given categorical controls.

        Arguments:
            A (dict of NumPy Arrays): dictionary linking periods to the mean of fixed effects in that period
            A_cat (dict of dicts of NumPy Arrays or None): dictionary linking each categorical control column name to a dictionary linking periods to the mean of fixed effects in that period

        Returns:
            (tuple): tuple of normalized parameters (A, A_cat)
        '''
        # Unpack parameters
        params = self.params
        nl, nk = self.nl, self.nk
        cat_cols, cat_dict = self.cat_cols, self.cat_dict
        A, A_cat = A.copy(), A_cat.copy()

        if len(cat_cols) > 0:
            # Compute minimum firm type
            min_firm_type = self._min_firm_type(A)
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
                for period in self.all_periods:
                    ## Iterate over periods ##
                    if period[-1] != 'b':
                        for l in range(nl):
                            ## Iterate over worker types ##
                            # Normalize each worker type-period pair
                            adj_val_tl = A[period][l, min_firm_type]
                            A[period][l, :] -= adj_val_tl
                            A_cat[tv_wi_col][period][l, :] += adj_val_tl
                    else:
                        # Normalize each worker type-period pair
                        adj_val_tl = A[period][min_firm_type]
                        A[period] -= adj_val_tl
                        A_cat[tv_wi_col][period] += adj_val_tl
            else:
                primary_period_dict = {
                    'first': '12',
                    'second': '43',
                    'all': ['12', '43']
                }
                secondary_period_dict = {
                    'first': '43',
                    'second': '12',
                    'all': ['12', '43']
                }
                A_list = [A, A_cat]
                Ap = to_list(primary_period_dict[params['primary_period']])
                As = to_list(secondary_period_dict[params['primary_period']])
                if any_tnv_wi:
                    ## Normalize primary period ##
                    for l in range(nl):
                        ## Iterate over worker types ##
                        # Compute normalization
                        adj_val_1 = 0
                        for Ap_sub in Ap:
                            adj_val_1 += A_list[0][Ap_sub][l, min_firm_type]
                        adj_val_1 /= len(Ap)

                        for period in self.all_periods:
                            ## Iterate over periods ##
                            # Normalize each worker type-period pair
                            if period[-1] != 'b':
                                A[period][l, :] -= adj_val_1
                                A_cat[tnv_wi_col][period][l, :] += adj_val_1
                            elif l == 0:
                                A[period] -= adj_val_1
                                A_cat[tnv_wi_col][period] += adj_val_1

                    if any_tv_nwi:
                        ## Normalize lowest type pair from each period ##
                        for period in self.all_periods:
                            ## Iterate over periods ##
                            # Normalize each period
                            if period[-1] != 'b':
                                adj_val_t = A[period][0, min_firm_type]
                            else:
                                adj_val_t = A[period][min_firm_type]
                            A[period] -= adj_val_t
                            A_cat[tv_nwi_col][period] += adj_val_t
                else:
                    if any_tv_nwi:
                        ## Normalize lowest type pair from each period ##
                        for period in self.all_periods:
                            ## Iterate over periods ##
                            # Normalize each period
                            if period[-1] != 'b':
                                adj_val_t = A[period][0, min_firm_type]
                            else:
                                adj_val_t = A[period][min_firm_type]
                            A[period] -= adj_val_t
                            A_cat[tv_nwi_col][period] += adj_val_t
                    elif any_tnv_nwi:
                        ## Normalize lowest type pair in primary period ##
                        # Compute normalization
                        adj_val_1 = 0
                        for Ap_sub in Ap:
                            adj_val_1 += A_list[0][Ap_sub][0, min_firm_type]
                        adj_val_1 /= len(Ap)

                        for period in self.all_periods:
                            ## Iterate over periods ##
                            # Normalize each period
                            A[period] -= adj_val_1
                            A_cat[tnv_nwi_col][period] += adj_val_1

        return (A, A_cat)

    def _sum_by_non_nl(self, ni, C_dict, A_cat, S_cat, A_cts, S_cts, compute_A=True, compute_S=True, periods=None):
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
            periods (list or None): list of periods to sum over; if None, sum over all periods

        Returns:
            (tuple of dicts of NumPy Arrays): (A_sum, S_sum_sq), where each dictionary links to periods, and each period links to the sum of estimated effects for control variables that do not interact with worker type in that period (A terms are dropped if compute_A=False, and S terms are dropped if compute_S=False)
        '''
        if (not compute_A) and (not compute_S):
            raise ValueError('`compute_A`=False and `compute_S`=False. Must specify at least one to be True.')

        if periods is None:
            periods = self.periods

        # if not self.any_non_worker_type_interactions:
        #     # If all control variables interact with worker type
        #     if compute_A and compute_S:
        #         return [0] * 10
        #     return [0] * 6

        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        controls_dict = self.controls_dict

        if compute_A:
            A_sum = {
                period: np.zeros(ni) if period in periods else 0 for period in self.all_periods
            }
        if compute_S:
            S_sum_sq = {
                period: np.zeros(ni) if period in periods else 0 for period in self.all_periods
            }

        ## Categorical ##
        for col in cat_cols:
            A_cat_col = A_cat[col]
            S_cat_col = S_cat[col]
            for period in periods:
                A_cat_t = A_cat_col[period]
                S_cat_t = S_cat_col[period]
                if (not controls_dict[col]['worker_type_interaction']) or (period[-1] == 'b'):
                    C_t = C_dict[period][col]
                    if compute_A:
                        A_sum[period] += A_cat_t[C_t]
                    if compute_S:
                        S_sum_sq[period] += (S_cat_t ** 2)[C_t]

        ## Continuous ##
        for col in cts_cols:
            A_cts_col = A_cts[col]
            S_cts_col = S_cts[col]
            for period in periods:
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

    def _sum_by_nl_l(self, ni, l, C_dict, A_cat, S_cat, A_cts, S_cts, compute_A=True, compute_S=True, periods=None):
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
            periods (list or None): list of periods to sum over; if None, sum over all periods

        Returns:
            (tuple of dicts of NumPy Arrays): (A_sum_l, S_sum_sq_l), where each dictionary links to periods, and each period links to the sum of estimated effects for control variables that interact with worker type, specifically for worker type l (A terms are dropped if compute_A=False, and S terms are dropped if compute_S=False)
        '''
        if (not compute_A) and (not compute_S):
            raise ValueError('compute_A=False and compute_S=False. Must specify at least one to be True.')

        if periods is None:
            periods = self.periods

        # if not self.any_worker_type_interactions:
        #     # If no control variables interact with worker type
        #     if compute_A and compute_S:
        #         return [0] * 10
        #     return [0] * 6

        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        controls_dict = self.controls_dict

        if compute_A:
            A_sum_l = {
                period: np.zeros(ni) if period in periods else 0 for period in self.all_periods
            }
        if compute_S:
            S_sum_sq_l = {
                period: np.zeros(ni) if period in periods else 0 for period in self.all_periods
            }

        ## Categorical ##
        for col in cat_cols:
            A_cat_col = A_cat[col]
            S_cat_col = S_cat[col]
            for period in periods:
                A_cat_t = A_cat_col[period]
                S_cat_t = S_cat_col[period]
                if controls_dict[col]['worker_type_interaction'] and (period[-1] != 'b'):
                    C_t = C_dict[period][col]
                    if compute_A:
                        A_sum_l[period] += A_cat_t[l, C_t]
                    if compute_S:
                        S_sum_sq_l[period] += (S_cat_t[l, :] ** 2)[C_t]

        ## Continuous ##
        for col in cts_cols:
            A_cts_col = A_cts[col]
            S_cts_col = S_cts[col]
            for period in periods:
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
                A, A_cat, A_cts = copy.deepcopy((self.A, self.A_cat, self.A_cts))
                S, S_cat, S_cts = copy.deepcopy((self.S, self.S_cat, self.S_cts))
                pk1 = copy.deepcopy(self.pk1)

                ## Estimate with min_firm_type == k ##
                blm_k = DynamicBLMModel(params)
                # Set initial guesses
                blm_k.A, blm_k.A_cat, blm_k.A_cts = A, A_cat, A_cts
                blm_k.S, blm_k.S_cat, blm_k.S_cts = S, S_cat, S_cts
                blm_k.pk1 = pk1
                # Fit estimator
                blm_k._fit_movers(jdata=jdata, compute_NNm=False, min_firm_type=k)

                ## Store best estimator ##
                if (best_model is None) or (blm_k.lik1 > best_model.lik1):
                    best_model = blm_k
            ## Update parameters with best model ##
            self.A, self.A_cat, self.A_cts = best_model.A, best_model.A_cat, best_model.A_cts
            self.S, self.S_cat, self.S_cts = best_model.S, best_model.S_cat, best_model.S_cts
            self.pk1 = best_model.pk1
            self.liks1, self.lik1 = best_model.liks1, best_model.lik1

            if compute_NNm:
                # Update NNm
                self.NNm = jdata.groupby('g1')['g4'].value_counts().unstack(fill_value=0).to_numpy()

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
        periods, periods_var = self.periods_movers, self.periods_variance
        R12, R43, R32m = self.R12, self.R43, self.R32m
        A, A_cat, A_cts = self.A, self.A_cat, self.A_cts
        S, S_cat, S_cts = self.S, self.S_cat, self.S_cts
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        cat_dict, cts_dict = self.cat_dict, self.cts_dict
        controls_dict = self.controls_dict
        any_controls = self.any_controls
        endogeneity, state_dependence = params.get_multiple(('endogeneity', 'state_dependence'))
        update_rho = (params['update_rho12'] or params['update_rho43'] or params['update_rho32m'])

        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        Y3 = jdata.loc[:, 'y3'].to_numpy()
        Y4 = jdata.loc[:, 'y4'].to_numpy()
        G1 = jdata.loc[:, 'g1'].to_numpy().astype(int, copy=False)
        G2 = jdata.loc[:, 'g4'].to_numpy().astype(int, copy=False)

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

        ## Sparse matrix representations ##
        GG1 = csc_matrix((np.ones(ni), (range(ni), G1)), shape=(ni, nk))
        GG2 = csc_matrix((np.ones(ni), (range(ni), G2)), shape=(ni, nk))
        CC1 = {col: csc_matrix((np.ones(ni), (range(ni), C1[col])), shape=(ni, controls_dict[col]['n'])) for col in cat_cols}
        CC2 = {col: csc_matrix((np.ones(ni), (range(ni), C2[col])), shape=(ni, controls_dict[col]['n'])) for col in cat_cols}
        ## Dictionaries linking periods to vectors/matrices ##
        # G_dict = {period: G1 if period in self.first_periods else G2 for period in periods}
        # GG_dict = {period: GG1 if period in self.first_periods else GG2 for period in periods}
        C_dict = {period: C1 if period in self.first_periods else C2 for period in periods}
        # CC_dict = {period: CC1 if period in self.first_periods else CC2 for period in periods}

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
        # Track minimum firm type to check whether estimator stuck in a loop
        min_firm_types = []
        # Whether results should be stored
        store_res = True

        ## Sort ##
        A, S, A_cat, S_cat, A_cts, S_cts, pk1, self.pk0 = self._sort_parameters(A, S, A_cat, S_cat, A_cts, S_cts, pk1, self.pk0)

        ## Constraints ##
        if params['force_min_firm_type']:
            # If forcing minimum firm type
            prev_min_firm_type = min_firm_type
            min_firm_type = min_firm_type
        else:
            # If not forcing minimum firm type
            prev_min_firm_type = self._min_firm_type(A)
        cons_a, cons_s, cons_a_dict, cons_s_dict = self._gen_constraints(min_firm_type=prev_min_firm_type, for_movers=True)

        for iter in range(params['n_iters_movers']):
            # ---------- E-Step ----------
            # We compute the posterior probabilities for each row
            # We iterate over the worker types, should not be be too costly since the vector is quite large within each iteration
            log_pk1 = np.log(pk1)
            if any_controls:
                ## Account for control variables ##
                if iter == 0:
                    A_sum, S_sum_sq = self._sum_by_non_nl(ni=ni, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)
                else:
                    S_sum_sq = self._sum_by_non_nl(ni=ni, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_A=False, periods=periods)

                for l in range(nl):
                    # Update A_sum/S_sum_sq to account for worker-interaction terms
                    A_sum_l, S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)

                    lp1 = lognormpdf(
                        Y1 - R12 * (Y2 - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma'])),
                        A['12'][l, G1] + A_sum['12'] + A_sum_l['12'],
                        var=\
                            (S['12'][l, :] ** 2)[G1] + S_sum_sq['12'] + S_sum_sq_l['12'] # \
                            # + (R12 ** 2) \
                            #     * ((S['2ma'][l, :] ** 2)[G1] \
                            #         + S_sum_sq['2ma'] \
                            #         + S_sum_sq_l['2ma'])
                    )
                    lp2 = lognormpdf(
                        Y2 - (A['2mb'][G2] + A_sum['2mb'] + A_sum_l['2mb']),
                        (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']),
                        var=\
                            ((S['2ma'][l, :] ** 2)[G1] + S_sum_sq['2ma'] + S_sum_sq_l['2ma']) # \
                            # + ((S['2mb'] ** 2)[G2] + S_sum_sq['2mb'] + S_sum_sq_l['2mb'])
                    )
                    lp3 = lognormpdf(
                        Y3 - (A['3mb'][G1] + A_sum['3mb'] + A_sum_l['3mb']) \
                            - R32m * (Y2 \
                                - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']) \
                                - (A['2mb'][G2] + A_sum['2mb'] + A_sum_l['2mb'])),
                        A['3ma'][l, G2] + A_sum['3ma'] + A_sum_l['3ma'],
                        var=\
                            (S['3ma'][l, :] ** 2)[G2] + S_sum_sq['3ma'] + S_sum_sq_l['3ma'] # \
                            # + ((S['3mb'] ** 2)[G1] + S_sum_sq['3mb'] + S_sum_sq_l['3mb']) \
                            # + (R32m ** 2) \
                            #     * ((S['2ma'][l, :] ** 2)[G1] + (S['2mb'] ** 2)[G2] \
                            #         + (S_sum_sq['2ma'] + S_sum_sq['2mb']) \
                            #         + (S_sum_sq_l['2ma'] + S_sum_sq_l['2mb']))
                    )
                    lp4 = lognormpdf(
                        Y4 - R43 * (Y3 - (A['3ma'][l, G2] + A_sum['3ma'] + A_sum_l['3ma'])),
                        A['43'][l, G2] + A_sum['43'] + A_sum_l['43'],
                        var=\
                            (S['43'][l, :] ** 2)[G2] + S_sum_sq['43'] + S_sum_sq_l['43'] # \
                            # + (R43 ** 2) \
                            #     * ((S['3ma'][l, :] ** 2)[G2] \
                            #         + S_sum_sq['3ma'] \
                            #         + S_sum_sq_l['3ma'])
                    )

                    lp[:, l] = log_pk1[KK, l] + lp1 + lp2 + lp3 + lp4
            else:
                A_sum = {period: 0 for period in self.all_periods}
                S_sum_sq = {period: 0 for period in self.all_periods}
                # Loop over firm classes so means/variances are single values rather than vectors (computing log/square is faster this way)
                for g1 in range(nk):
                    for g2 in range(nk):
                        I = (G1 == g1) & (G2 == g2)
                        for l in range(nl):
                            lp1 = lognormpdf(
                                Y1[I] \
                                    - R12 * (Y2[I] - A['2ma'][l, g1]),
                                A['12'][l, g1],
                                var=S['12'][l, g1] ** 2 # + (R12 * S['2ma'][l, g1]) ** 2
                            )
                            lp2 = lognormpdf(
                                Y2[I] \
                                    - A['2mb'][g2],
                                A['2ma'][l, g1],
                                var=S['2ma'][l, g1] ** 2 # + S['2mb'][g2] ** 2
                            )
                            lp3 = lognormpdf(
                                Y3[I] \
                                    - A['3mb'][g1] \
                                    - R32m * (Y2[I] - (A['2ma'][l, g1] + A['2mb'][g2])),
                                A['3ma'][l, g2],
                                var=S['3ma'][l, g2] ** 2 # + S['3mb'][g1] ** 2 + (R32m ** 2) * (S['2ma'][l, g1] ** 2 + S['2mb'][g2] ** 2)
                            )
                            lp4 = lognormpdf(
                                Y4[I] \
                                    - R43 * (Y3[I] - A['3ma'][l, g2]),
                                A['43'][l, g2],
                                var=S['43'][l, g2] ** 2 # + (R43 * S['3ma'][l, g2]) ** 2
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
                min_firm_type = self._min_firm_type(A)

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

            # ---------- Update pk1 ----------
            if params['update_pk1']:
                # NOTE: add dirichlet prior
                pk1 = GG12.T @ (qi + d_prior - 1)
                # Normalize rows to sum to 1
                pk1 = DxM(1 / np.sum(pk1, axis=1), pk1)

                if pd.isna(pk1).any():
                    warnings.warn('Estimated pk1 has NaN values. Please try a different set of starting values.')
                    break
                    # raise ValueError('Estimated pk1 has NaN values. Please try a different set of starting values.')

            # ---------- M-step ----------
            # Alternate between updating A/S and updating rho
            if update_rho and ((iter % 2) == 1):
                ## Update rho ##
                if params['update_rho12']:
                    XX12 = np.zeros(nl * ni)
                    YY12 = np.zeros(nl * ni)
                    WW12 = np.zeros(nl * ni)
                if params['update_rho43']:
                    XX43 = np.zeros(nl * ni)
                    YY43 = np.zeros(nl * ni)
                    WW43 = np.zeros(nl * ni)
                if params['update_rho32m']:
                    XX32m = np.zeros(nl * ni)
                    YY32m = np.zeros(nl * ni)
                    WW32m = np.zeros(nl * ni)
                for l in range(nl):
                    A_sum_l, S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)
                    if params['update_rho12']:
                        XX12[l * ni: (l + 1) * ni] = Y2 - A['2ma'][l, G1] - A_sum['2ma'] - A_sum_l['2ma']
                        YY12[l * ni: (l + 1) * ni] = Y1 - A['12'][l, G1] - A_sum['12'] - A_sum_l['12']
                        SS12 = ( \
                            (S['12'][l, :] ** 2)[G1] \
                            + S_sum_sq['12'] + S_sum_sq_l['12']) # \
                            # + (R12 ** 2) \
                            #     * ((S['2ma'][l, :] ** 2)[G1] \
                            #         + S_sum_sq['2ma'] \
                            #         + S_sum_sq_l['2ma']))
                        WW12[l * ni: (l + 1) * ni] = qi[:, l] / np.sqrt(SS12)
                    if params['update_rho43']:
                        XX43[l * ni: (l + 1) * ni] = Y3 - A['3ma'][l, G2] - A_sum['3ma'] - A_sum_l['3ma']
                        YY43[l * ni: (l + 1) * ni] = Y4 - A['43'][l, G2] - A_sum['43'] - A_sum_l['43']
                        SS43 = ( \
                            (S['43'][l, :] ** 2)[G2] \
                            + S_sum_sq['43'] + S_sum_sq_l['43']) # \
                            # + (R43 ** 2) * (\
                            #     (S['3ma'][l, :] ** 2)[G2] \
                            #     + S_sum_sq['3ma'] \
                            #     + S_sum_sq_l['3ma']))
                        WW43[l * ni: (l + 1) * ni] = qi[:, l] / np.sqrt(SS43)
                    if params['update_rho32m']:
                        XX32m[l * ni: (l + 1) * ni] = Y2 - (A['2ma'][l, G1] + A['2mb'][G2]) - (A_sum['2ma'] + A_sum['2mb']) - (A_sum_l['2ma'] + A_sum_l['2mb'])
                        YY32m[l * ni: (l + 1) * ni] = Y3 - (A['3ma'][l, G2] + A['3mb'][G1]) - (A_sum['3ma'] + A_sum['3mb']) - (A_sum_l['3ma'] + A_sum_l['3mb'])
                        SS32m = ( \
                            (S['3ma'][l, :] ** 2)[G2] + S_sum_sq['3ma'] + S_sum_sq_l['3ma']) # \
                            # + (S['3mb'] ** 2)[G1] + S_sum_sq['3mb'] + S_sum_sq_l['3mb'] \
                            # + (R32m ** 2) \
                            #     * (((S['2ma'][l, :] ** 2)[G1] + (S['2mb'] ** 2)[G2]) \
                            #         + (S_sum_sq['2ma'] + S_sum_sq['2mb']) \
                            #         + (S_sum_sq_l['2ma'] + S_sum_sq_l['2mb'])))
                        WW32m[l * ni: (l + 1) * ni] = qi[:, l] / np.sqrt(SS32m)

                ## OLS ##
                if params['update_rho12']:
                    Xw = XX12 * WW12
                    XwX = np.sum(Xw * XX12)
                    XwY = np.sum(Xw * YY12)
                    R12 = XwY / XwX
                if params['update_rho43']:
                    Xw = XX43 * WW43
                    XwX = np.sum(Xw * XX43)
                    XwY = np.sum(Xw * YY43)
                    R43 = XwY / XwX
                if params['update_rho32m']:
                    Xw = XX32m * WW32m
                    XwX = np.sum(Xw * XX32m)
                    XwY = np.sum(Xw * YY32m)
                    R32m = XwY / XwX
                del Xw, XwX, XwY
            elif params['update_a_movers'] or params['update_s_movers']:
                # Constrained OLS (source: https://scaron.info/blog/quadratic-programming-in-python.html)
                # The regression has 6 * nl * nk parameters and 4 * nl * ni rows
                # To avoid duplicating the data 4 * nl times, we construct X'X and X'Y by looping over nl
                # We also note that X'X is block diagonal with nl matrices of dimension (6 * nk, 6 * nk)

                #### Initialize XX terms ####
                # XX =
                # +---------+---------+---------------+--------------+---------------+----------+
                # | A['12'] | A['43'] |    A['2ma']   |   A['3ma']   |    A['2mb']   | A['3mb'] |
                # +=========+=========+===============+==============+===============+==========+
                # |   GG1   |    0    |  -(R12 * GG1) |       0      |       0       |     0    |
                # +---------+---------+---------------+--------------+---------------+----------+
                # |    0    |    0    |      GG1      |       0      |      GG2      |     0    |
                # +---------+---------+---------------+--------------+---------------+----------+
                # |    0    |    0    | -(R32m * GG1) |      GG2     | -(R32m * GG2) |    GG1   |
                # +---------+---------+---------------+--------------+---------------+----------+
                # |    0    |   GG2   |       0       | -(R43 * GG2) |       0       |     0    |
                # +---------+---------+---------------+--------------+---------------+----------+
                ### General ###
                # Shift between periods
                ts = nl * nk
                # XX'XX (weighted)
                XXwXX = np.zeros((len(periods) * ts, len(periods) * ts))
                # 0 matrix
                XX0 = np.zeros((nk, nk))
                if params['update_a_movers']:
                    XXwY = np.zeros(shape=len(periods) * ts)
                if params['update_s_movers']:
                    XSwXS = np.zeros(len(periods_var) * ts)
                    XSwE = np.zeros(shape=len(periods_var) * ts)

                ### Categorical ###
                if len(cat_cols) > 0:
                    # Shift between periods
                    ts_cat = {col: nl * col_dict['n'] for col, col_dict in cat_dict.items()}
                    # XX_cat'XX_cat (weighted)
                    XXwXX_cat = {col: np.zeros((len(periods) * col_ts, len(periods) * col_ts)) for col, col_ts in ts_cat.items()}
                    # 0 matrix
                    XX0_cat = {col: np.zeros((col_dict['n'], col_dict['n'])) for col, col_dict in cat_dict.items()}
                    if params['update_a_movers']:
                        XXwY_cat = {col: np.zeros(shape=len(periods) * col_ts) for col, col_ts in ts_cat.items()}
                    if params['update_s_movers']:
                        XSwXS_cat = {col: np.zeros(shape=len(periods_var) * col_ts) for col, col_ts in ts_cat.items()}
                        XSwE_cat = {col: np.zeros(shape=len(periods_var) * col_ts) for col, col_ts in ts_cat.items()}

                ### Continuous ###
                if len(cts_cols) > 0:
                    # XX_cts'XX_cts (weighted)
                    XXwXX_cts = {col: np.zeros((len(periods) * nl, len(periods) * nl)) for col in cts_cols}
                    if params['update_a_movers']:
                        XXwY_cts = {col: np.zeros(shape=len(periods) * nl) for col in cts_cols}
                    if params['update_s_movers']:
                        XSwXS_cts = {col: np.zeros(shape=len(periods_var) * nl) for col in cts_cols}
                        XSwE_cts = {col: np.zeros(shape=len(periods_var) * nl) for col in cts_cols}

                ## Update A ##
                if params['update_s_movers']:
                    # Store weights computed for A for use when computing S
                    weights = []
                for l in range(nl):
                    l_index, r_index = l * nk * len(periods), (l + 1) * nk * len(periods)

                    ## Compute weights_l ##
                    weights_l = [
                            qi[:, l] / S['12'][l, G1],
                            qi[:, l] / S['2ma'][l, G1],
                            qi[:, l] / S['3ma'][l, G2],
                            qi[:, l] / S['43'][l, G2]
                    ]

                    ## Compute XXwXX_l ##
                    G1W1G1 = np.diag(np.bincount(G1, weights_l[0]))
                    G1W2G1 = np.diag(np.bincount(G1, weights_l[1]))
                    G1W3G1 = np.diag(np.bincount(G1, weights_l[2]))
                    if endogeneity:
                        G2W2G2 = np.diag(np.bincount(G2, weights_l[1]))
                    G2W3G2 = np.diag(np.bincount(G2, weights_l[2]))
                    G2W4G2 = np.diag(np.bincount(G2, weights_l[3]))
                    if endogeneity:
                        G1W2G2 = double_bincount(G1, G2, weights_l[1])
                    G1W3G2 = double_bincount(G1, G2, weights_l[2])

                    if params['update_s_movers']:
                        weights.append(weights_l)
                        ## XSwXS ##
                        l_index_S = l * nk * len(periods_var)

                        XSwXS[l_index_S + 0 * nk: l_index_S + 1 * nk] = \
                            np.diag(G1W1G1)
                        XSwXS[l_index_S + 1 * nk: l_index_S + 2 * nk] = \
                            np.diag(G1W2G1)
                        XSwXS[l_index_S + 2 * nk: l_index_S + 3 * nk] = \
                            np.diag(G2W3G2)
                        XSwXS[l_index_S + 3 * nk: l_index_S + 4 * nk] = \
                            np.diag(G2W4G2)

                    XXwXX_l = np.vstack(
                        [
                            np.hstack(
                                [G1W1G1, XX0, - R12 * G1W1G1, XX0]
                            ),
                            np.hstack(
                                [XX0, G2W4G2, XX0, - R43 * G2W4G2]
                            ),
                            np.hstack(
                                [- R12 * G1W1G1, XX0, (R12 ** 2) * G1W1G1 + G1W2G1 + (R32m ** 2) * G1W3G1, - R32m * G1W3G2]
                            ),
                            np.hstack(
                                [XX0, - R43 * G2W4G2, - R32m * G1W3G2.T, G2W3G2 + (R43 ** 2) * G2W4G2]
                            )
                        ]
                    )
                    if endogeneity:
                        XXwXX_l = np.hstack(
                            [
                                XXwXX_l,
                                np.vstack(
                                    [
                                        XX0,
                                        XX0,
                                        G1W2G2 + (R32m ** 2) * G1W3G2,
                                        - R32m * G2W3G2
                                    ]
                                )
                            ]
                        )
                        XXwXX_l = np.vstack(
                            [
                                XXwXX_l,
                                np.hstack(
                                    [
                                        XX0, XX0, G1W2G2.T + (R32m ** 2) * G1W3G2.T, - R32m * G2W3G2, G2W2G2 + (R32m ** 2) * G2W3G2
                                    ]
                                )
                            ]
                        )
                        if state_dependence:
                            XXwXX_l = np.hstack(
                                [
                                    XXwXX_l,
                                    np.vstack(
                                        [
                                            XX0,
                                            XX0,
                                            - R32m * G1W3G1,
                                            G1W3G2.T,
                                            - R32m * G1W3G2.T
                                        ]
                                    )
                                ]
                            )
                            XXwXX_l = np.vstack(
                                [
                                    XXwXX_l,
                                    np.hstack(
                                        [
                                            XX0, XX0, - R32m * G1W3G1, G1W3G2, - R32m * G1W3G2, G1W3G1
                                        ]
                                    )
                                ]
                            )
                    elif state_dependence:
                        XXwXX_l = np.hstack(
                            [
                                XXwXX_l,
                                np.vstack(
                                    [
                                        XX0,
                                        XX0,
                                        - R32m * G1W3G1,
                                        G1W3G2.T
                                    ]
                                )
                            ]
                        )
                        XXwXX_l = np.vstack(
                            [
                                XXwXX_l,
                                np.hstack(
                                    [
                                        XX0, XX0, - R32m * G1W3G1, G1W3G2, G1W3G1
                                    ]
                                )
                            ]
                        )
                    del G1W1G1, G1W2G1, G1W3G1, G2W3G2, G2W4G2, G1W3G2
                    if endogeneity:
                        del G2W2G2, G1W2G2

                    XXwXX[l_index: r_index, l_index: r_index] = XXwXX_l
                    del XXwXX_l

                    if params['update_a_movers']:
                        # Update A_sum to account for worker-interaction terms
                        A_sum_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=False, periods=periods)

                        # Yl_1
                        Yl_1 = \
                            Y1 \
                                - (A_sum['12'] + A_sum_l['12']) \
                                - R12 * (Y2 - (A_sum['2ma'] + A_sum_l['2ma']))
                        # Yl_2
                        Yl_2 = \
                            Y2 \
                                - (A_sum['2ma'] + A_sum_l['2ma']) \
                                - (A_sum['2mb'] + A_sum_l['2mb'])
                        # Yl_3
                        Yl_3 = \
                            Y3 \
                                - (A_sum['3ma'] + A_sum_l['3ma']) \
                                - (A_sum['3mb'] + A_sum_l['3mb']) \
                                - R32m * (Y2 \
                                    - (A_sum['2ma'] + A_sum_l['2ma']) \
                                    - (A_sum['2mb'] + A_sum_l['2mb']))
                        # Yl_4
                        Yl_4 = \
                            Y4 \
                                - (A_sum['43'] + A_sum_l['43']) \
                                - R43 * (Y3 - (A_sum['3ma'] + A_sum_l['3ma']))

                        ## Compute XXwY_l ##
                        XXwY_l = np.concatenate(
                            [
                                np.bincount(G1, weights_l[0] * Yl_1),
                                np.bincount(G2, weights_l[3] * Yl_4),
                                np.bincount(G1, - R12 * weights_l[0] * Yl_1 + weights_l[1] * Yl_2 - R32m * Yl_3),
                                np.bincount(G2, weights_l[2] * Yl_3 - R43 * weights_l[3] * Yl_4)
                            ]
                        )
                        if endogeneity:
                            XXwY_l = np.concatenate(
                                [
                                    XXwY_l, np.bincount(G2, weights_l[1] * Yl_2 - R32m * weights_l[2] * Yl_3)
                                ]
                            )
                        if state_dependence:
                            XXwY_l = np.concatenate(
                                [
                                    XXwY_l, np.bincount(G1, weights_l[2] * Yl_3)
                                ]
                            )
                        XXwY[l_index: r_index] = XXwY_l
                        del Yl_1, Yl_2, Yl_3, Yl_4, XXwY_l, A_sum_l
                    del weights_l

                if params['d_X_diag_movers_A'] > 1:
                    XXwXX += (params['d_X_diag_movers_A'] - 1) * np.eye(XXwXX.shape[0])

                # print('A before:')
                # print(A)
                # print('S before:')
                # print(S)
                # print('A_cat before:')
                # print(A_cat)
                # print('S_cat before:')
                # print(S_cat)
                # print('A_cts before:')
                # print(A_cts)
                # print('S_cts before:')
                # print(S_cts)

                # We solve the system to get all the parameters (use dense solver)
                if params['update_a_movers']:
                    if iter > 0:
                        ## Constraints ##
                        cons_a, cons_s, cons_a_dict, cons_s_dict = self._gen_constraints(min_firm_type=min_firm_type, for_movers=True)
                    try:
                        cons_a.solve(XXwXX, -XXwY, solver='quadprog')
                        if cons_a.res is None:
                            # If estimation fails, keep A the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing A: estimates are None')
                        else:
                            res_a = cons_a.res.reshape((nl, len(periods), nk))
                            for i, period in enumerate(periods):
                                if period[-1] != 'b':
                                    A[period] = res_a[:, i, :]
                                else:
                                    A[period] = res_a[0, i, :]
                            del res_a

                    except ValueError as e:
                        # If constraints inconsistent, keep A the same
                        if params['verbose'] in [2, 3]:
                            print(f'Passing A: {e}')
                    del XXwXX, XXwY

                ## Categorical ##
                if params['update_s_movers']:
                    # Store weights computed for A_cat for use when computing S_cat
                    weights_cat = {col: [] for col in cat_cols}
                for col in cat_cols:
                    col_n = cat_dict[col]['n']

                    if not cat_dict[col]['worker_type_interaction']:
                        # Adjust A_sum
                        for period in periods:
                            A_sum[period] -= A_cat[col][period][C_dict[period][col]]

                    for l in range(nl):
                        l_index, r_index = l * col_n * len(periods), (l + 1) * col_n * len(periods)

                        ## Compute weights_l ##
                        if cat_dict[col]['worker_type_interaction']:
                            S_l_dict = {period: S_cat[col][period][l, C_dict[period][col]] for period in periods}
                        else:
                            S_l_dict = {period: S_cat[col][period][C_dict[period][col]] for period in periods}

                        weights_l = [
                            qi[:, l] / S_l_dict['12'],
                            qi[:, l] / S_l_dict['2ma'],
                            qi[:, l] / S_l_dict['3ma'],
                            qi[:, l] / S_l_dict['43']
                        ]
                        del S_l_dict

                        ## Compute XXwXX_cat_l ##
                        C1W1C1 = np.diag(np.bincount(C1[col], weights_l[0]))
                        C1W2C1 = np.diag(np.bincount(C1[col], weights_l[1]))
                        C1W3C1 = np.diag(np.bincount(C1[col], weights_l[2]))
                        if endogeneity:
                            C2W2C2 = np.diag(np.bincount(C2[col], weights_l[1]))
                        C2W3C2 = np.diag(np.bincount(C2[col], weights_l[2]))
                        C2W4C2 = np.diag(np.bincount(C2[col], weights_l[3]))
                        if endogeneity:
                            C1W2C2 = double_bincount(C1[col], C2[col], weights_l[1])
                        C1W3C2 = double_bincount(C1[col], C2[col], weights_l[2])

                        if params['update_s_movers']:
                            weights_cat[col].append(weights_l)

                            ### XSwXS_cat ###
                            l_index_S = l * col_n * len(periods_var)

                            XSwXS_cat[l_index_S + 0 * col_n: l_index_S + 1 * col_n] = \
                                np.diag(C1W1C1)
                            XSwXS_cat[l_index_S + 1 * col_n: l_index_S + 2 * col_n] = \
                                np.diag(C1W2C1)
                            XSwXS_cat[l_index_S + 2 * col_n: l_index_S + 3 * col_n] = \
                                np.diag(C2W3C2)
                            XSwXS_cat[l_index_S + 3 * col_n: l_index_S + 4 * col_n] = \
                                np.diag(C2W4C2)

                        XXwXX_cat_l = np.vstack(
                            [
                                np.hstack(
                                    [C1W1C1, XX0_cat[col], - R12 * C1W1C1, XX0_cat[col]]
                                ),
                                np.hstack(
                                    [XX0_cat[col], C2W4C2, XX0_cat[col], - R43 * C2W4C2]
                                ),
                                np.hstack(
                                    [- R12 * C1W1C1, XX0_cat[col], (R12 ** 2) * C1W1C1 + C1W2C1 + (R32m ** 2) * C1W3C1, - R32m * C1W3C2]
                                ),
                                np.hstack(
                                    [XX0_cat[col], - R43 * C2W4C2, - R32m * C1W3C2.T, C2W3C2 + (R43 ** 2) * C2W4C2]
                                )
                            ]
                        )
                        if endogeneity:
                            XXwXX_cat_l = np.hstack(
                                [
                                    XXwXX_cat_l,
                                    np.vstack(
                                        [
                                            XX0_cat[col],
                                            XX0_cat[col],
                                            C1W2C2 + (R32m ** 2) * C1W3C2,
                                            - R32m * C2W3C2
                                        ]
                                    )
                                ]
                            )
                            XXwXX_cat_l = np.vstack(
                                [
                                    XXwXX_cat_l,
                                    np.hstack(
                                        [
                                            XX0_cat[col], XX0_cat[col], C1W2C2.T + (R32m ** 2) * C1W3C2.T, - R32m * C2W3C2, C2W2C2 + (R32m ** 2) * C2W3C2
                                        ]
                                    )
                                ]
                            )
                            if state_dependence:
                                XXwXX_cat_l = np.hstack(
                                    [
                                        XXwXX_cat_l,
                                        np.vstack(
                                            [
                                                XX0_cat[col],
                                                XX0_cat[col],
                                                - R32m * C1W3C1,
                                                C1W3C2.T,
                                                - R32m * C1W3C2.T
                                            ]
                                        )
                                    ]
                                )
                                XXwXX_cat_l = np.vstack(
                                    [
                                        XXwXX_cat_l,
                                        np.hstack(
                                            [
                                                XX0_cat[col], XX0_cat[col], - R32m * C1W3C1, C1W3C2, - R32m * C1W3C2, C1W3C1
                                            ]
                                        )
                                    ]
                                )
                        elif state_dependence:
                            XXwXX_cat_l = np.hstack(
                                [
                                    XXwXX_cat_l,
                                    np.vstack(
                                        [
                                            XX0_cat[col],
                                            XX0_cat[col],
                                            - R32m * C1W3C1,
                                            C1W3C2.T
                                        ]
                                    )
                                ]
                            )
                            XXwXX_cat_l = np.vstack(
                                [
                                    XXwXX_cat_l,
                                    np.hstack(
                                        [
                                            XX0_cat[col], XX0_cat[col], - R32m * C1W3C1, C1W3C2, C1W3C1
                                        ]
                                    )
                                ]
                            )
                        del C1W1C1, C1W2C1, C1W3C1, C2W3C2, C2W4C2, C1W3C2
                        if endogeneity:
                            del C2W2C2, C1W2C2
                        XXwXX_cat[col][l_index: r_index, l_index: r_index] = XXwXX_cat_l
                        del XXwXX_cat_l

                        if params['update_a_movers']:
                            # Update A_sum to account for worker-interaction terms
                            A_sum_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=False, periods=periods)
                            if cat_dict[col]['worker_type_interaction']:
                                # Adjust A_sum
                                for period in periods:
                                    A_sum_l[period] -= A_cat[col][period][l, C_dict[period][col]]

                            # Yl_cat_1
                            Yl_cat_1 = \
                                Y1 \
                                    - (A['12'][l, G1] + A_sum['12'] + A_sum_l['12']) \
                                    - R12 * (Y2 - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']))
                            # Yl_cat_2
                            Yl_cat_2 = \
                                Y2 \
                                    - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']) \
                                    - (A['2mb'][G2] + A_sum['2mb'] + A_sum_l['2mb'])
                            # Yl_cat_3
                            Yl_cat_3 = \
                                Y3 \
                                    - (A['3ma'][l, G2] + A_sum['3ma'] + A_sum_l['3ma']) \
                                    - (A['3mb'][G1] + A_sum['3mb'] + A_sum_l['3mb']) \
                                    - R32m * (Y2 \
                                        - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']) \
                                        - (A['2mb'][G2] + A_sum['2mb'] + A_sum_l['2mb']))
                            # Yl_cat_4
                            Yl_cat_4 = \
                                Y4 \
                                    - (A['43'][l, G2] + A_sum['43'] + A_sum_l['43']) \
                                    - R43 * (Y3 - (A['3ma'][l, G2] + A_sum['3ma'] + A_sum_l['3ma']))

                            ## Compute XXwY_cat_l ##
                            XXwY_cat_l = np.concatenate(
                                [
                                    np.bincount(C1[col], weights_l[0] * Yl_cat_1),
                                    np.bincount(C2[col], weights_l[3] * Yl_cat_4),
                                    np.bincount(C1[col], - R12 * weights_l[0] * Yl_cat_1 + weights_l[1] * Yl_cat_2 - R32m * Yl_cat_3),
                                    np.bincount(C2[col], weights_l[2] * Yl_cat_3 - R43 * weights_l[3] * Yl_cat_4)
                                ]
                            )
                            if endogeneity:
                                XXwY_cat_l = np.concatenate(
                                    [
                                        XXwY_cat_l, np.bincount(C2[col], weights_l[1] * Yl_cat_2 - R32m * weights_l[2] * Yl_cat_3)
                                    ]
                                )
                            if state_dependence:
                                XXwY_cat_l = np.concatenate(
                                    [
                                        XXwY_cat_l, np.bincount(C1[col], weights_l[2] * Yl_cat_3)
                                    ]
                                )
                            XXwY_cat[col][l_index: r_index] = XXwY_cat_l
                            del Yl_cat_1, Yl_cat_2, Yl_cat_3, Yl_cat_4, XXwY_cat_l, A_sum_l
                        del weights_l

                    if params['d_X_diag_movers_A'] > 1:
                        XXwXX_cat[col] += (params['d_X_diag_movers_A'] - 1) * np.eye(XXwXX_cat[col].shape[0])

                    # We solve the system to get all the parameters (use dense solver)
                    if params['update_a_movers']:
                        try:
                            a_solver = cons_a_dict[col]
                            a_solver.solve(XXwXX_cat[col], -XXwY_cat[col], solver='quadprog')
                            if a_solver.res is None:
                                # If estimation fails, keep A_cat the same
                                if params['verbose'] in [2, 3]:
                                    print(f'Passing A_cat for column {col!r}: estimates are None')
                            else:
                                res_a = a_solver.res.reshape((nl, len(periods), col_n))
                                for i, period in enumerate(periods):
                                    if cat_dict[col]['worker_type_interaction'] and (period[-1] != 'b'):
                                        A_cat[col][period] = res_a[:, i, :]
                                    else:
                                        A_cat[col][period] = res_a[0, i, :]
                                del res_a

                        except ValueError as e:
                            # If constraints inconsistent, keep A_cat the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing A_cat for column {col!r}: {e}')
                        del XXwXX_cat[col], XXwY_cat[col]

                    if not cat_dict[col]['worker_type_interaction']:
                        # Restore A_sum with updated values
                        for period in periods:
                            A_sum[period] += A_cat[col][period][C_dict[period][col]]

                ## Continuous ##
                if params['update_s_movers']:
                    # Store weights computed for A_cts for use when computing S_cts
                    weights_cts = {col: [] for col in cts_cols}
                for col in cts_cols:
                    if not cts_dict[col]['worker_type_interaction']:
                        # Adjust A_sum
                        for period in periods:
                            A_sum[period] -= A_cts[col][period] * C_dict[period][col]

                    for l in range(nl):
                        l_index, r_index = l * len(periods), (l + 1) * len(periods)

                        ## Compute weights_l ##
                        if cts_dict[col]['worker_type_interaction']:
                            S_l_dict = {period: S_cts[col][period][l] for period in periods}
                        else:
                            S_l_dict = {period: S_cts[col][period] for period in periods}

                        weights_l = [
                            qi[:, l] / S_l_dict['12'],
                            qi[:, l] / S_l_dict['2ma'],
                            qi[:, l] / S_l_dict['3ma'],
                            qi[:, l] / S_l_dict['43']
                        ]
                        del S_l_dict

                        ## Compute XXwXX_cts_l ##
                        C1W1C1 = np.sum(weights_l[0] * C1[col])
                        C1W2C1 = np.sum(weights_l[1] * C1[col])
                        C1W3C1 = np.sum(weights_l[2] * C1[col])
                        if endogeneity:
                            C2W2C2 = np.sum(weights_l[1] * C2[col])
                        C2W3C2 = np.sum(weights_l[2] * C2[col])
                        C2W4C2 = np.sum(weights_l[3] * C2[col])
                        if endogeneity:
                            C1W2C2 = np.sum(weights_l[1] * C1[col] * C2[col])
                        C1W3C2 = np.sum(weights_l[2] * C1[col] * C2[col])

                        if params['update_s_movers']:
                            weights_cts[col].append(weights_l)

                            ### XSwXS_cts ###
                            l_index_S = l * len(periods_var)
                            XSwXS_cts[l_index_S + 0] = C1W1C1
                            XSwXS_cts[l_index_S + 1] = C1W2C1
                            XSwXS_cts[l_index_S + 2] = C2W3C2
                            XSwXS_cts[l_index_S + 3] = C2W4C2

                        XXwXX_cts_l = np.array(
                            [
                                [C1W1C1, 0, - R12 * C1W1C1, 0],
                                [0, C2W4C2, 0, - R43 * C2W4C2],
                                [- R12 * C1W1C1, 0, (R12 ** 2) * C1W1C1 + C1W2C1 + (R32m ** 2) * C1W3C1, - R32m * C1W3C2],
                                [0, - R43 * C2W4C2, - R32m * C1W3C2.T, C2W3C2 + (R43 ** 2) * C2W4C2]
                            ]
                        )
                        if endogeneity:
                            XXwXX_cts_l = np.hstack(
                                [
                                    XXwXX_cts_l,
                                    np.array(
                                        [
                                            0,
                                            0,
                                            C1W2C2 + (R32m ** 2) * C1W3C2,
                                            - R32m * C2W3C2
                                        ]
                                    )
                                ]
                            )
                            XXwXX_cts_l = np.vstack(
                                [
                                    XXwXX_cts_l,
                                    np.array(
                                        [
                                            0, 0, C1W2C2.T + (R32m ** 2) * C1W3C2.T, - R32m * C2W3C2, C2W2C2 + (R32m ** 2) * C2W3C2
                                        ]
                                    )
                                ]
                            )
                            if state_dependence:
                                XXwXX_cts_l = np.hstack(
                                    [
                                        XXwXX_cts_l,
                                        np.array(
                                            [
                                                0,
                                                0,
                                                - R32m * C1W3C1,
                                                C1W3C2.T,
                                                - R32m * C1W3C2.T
                                            ]
                                        )
                                    ]
                                )
                                XXwXX_cts_l = np.vstack(
                                    [
                                        XXwXX_cts_l,
                                        np.array(
                                            [
                                                0, 0, - R32m * C1W3C1, C1W3C2, - R32m * C1W3C2, C1W3C1
                                            ]
                                        )
                                    ]
                                )
                        elif state_dependence:
                            XXwXX_cts_l = np.hstack(
                                [
                                    XXwXX_cts_l,
                                    np.array(
                                        [
                                            0,
                                            0,
                                            - R32m * C1W3C1,
                                            C1W3C2.T
                                        ]
                                    )
                                ]
                            )
                            XXwXX_cts_l = np.vstack(
                                [
                                    XXwXX_cts_l,
                                    np.array(
                                        [
                                            0, 0, - R32m * C1W3C1, C1W3C2, C1W3C1
                                        ]
                                    )
                                ]
                            )
                        del C1W1C1, C1W2C1, C1W3C1, C2W3C2, C2W4C2, C1W3C2
                        if endogeneity:
                            del C2W2C2, C1W2C2
                        XXwXX_cts[col][l_index: r_index, l_index: r_index] = XXwXX_cts_l
                        del XXwXX_cts_l

                        if params['update_a_movers']:
                            # Update A_sum to account for worker-interaction terms
                            A_sum_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=False, periods=periods)
                            if cts_dict[col]['worker_type_interaction']:
                                # Adjust A_sum
                                for period in periods:
                                    A_sum_l[period] -= A_cts[col][period][l] * C_dict[period][col]

                            # Yl_cts_1
                            Yl_cts_1 = \
                                Y1 \
                                    - (A['12'][l, G1] + A_sum['12'] + A_sum_l['12']) \
                                    - R12 * (Y2 - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']))
                            # Yl_cts_2
                            Yl_cts_2 = \
                                Y2 \
                                    - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']) \
                                    - (A['2mb'][G2] + A_sum['2mb'] + A_sum_l['2mb'])
                            # Yl_cts_3
                            Yl_cts_3 = \
                                Y3 \
                                    - (A['3ma'][l, G2] + A_sum['3ma'] + A_sum_l['3ma']) \
                                    - (A['3mb'][G1] + A_sum['3mb'] + A_sum_l['3mb']) \
                                    - R32m * (Y2 \
                                        - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']) \
                                        - (A['2mb'][G2] + A_sum['2mb'] + A_sum_l['2mb']))
                            # Yl_cts_4
                            Yl_cts_4 = \
                                Y4 \
                                    - (A['43'][l, G2] + A_sum['43'] + A_sum_l['43']) \
                                    - R43 * (Y3 - (A['3ma'][l, G2] + A_sum['3ma'] + A_sum_l['3ma']))

                            ## Compute XwY_cts_l ##
                            XwY_cts_l = np.array(
                                [
                                    np.sum(weights_l[0] * Yl_cts_1 * C1[col]),
                                    np.sum(weights_l[3] * Yl_cts_4 * C2[col]),
                                    np.sum((- R12 * weights_l[0] * Yl_cts_1 + weights_l[1] * Yl_cts_2 - R32m * Yl_cts_3) * C1[col]),
                                    np.sum((weights_l[2] * Yl_cts_3 - R43 * weights_l[3] * Yl_cts_4) * C2[col])
                                ]
                            )
                            if endogeneity:
                                XwY_cts_l = np.append(
                                    XwY_cts_l, np.sum((weights_l[1] * Yl_cts_2 - R32m * weights_l[2] * Yl_cts_3) * C2[col])
                                )
                            if state_dependence:
                                XwY_cts_l = np.append(
                                    XwY_cts_l, np.sum(weights_l[2] * Yl_cts_3 * C1[col])
                                )
                            XXwY_cts[col][l_index: r_index] = XwY_cts_l
                            del Yl_cts_1, Yl_cts_2, Yl_cts_3, Yl_cts_4, XwY_cts_l, A_sum_l
                        del weights_l

                    if params['d_X_diag_movers_A'] > 1:
                        XXwXX_cts[col] += (params['d_X_diag_movers_A'] - 1) * np.eye(XXwXX_cts[col].shape[0])

                    # We solve the system to get all the parameters (use dense solver)
                    if params['update_a_movers']:
                        try:
                            a_solver = cons_a_dict[col]
                            a_solver.solve(XXwXX_cts[col], -XXwY_cts[col], solver='quadprog')
                            if a_solver.res is None:
                                # If estimation fails, keep A_cts the same
                                if params['verbose'] in [2, 3]:
                                    print(f'Passing A_cts for column {col!r}: estimates are None')
                            else:
                                res_a = a_solver.res.reshape((nl, len(periods)))
                                for i, period in enumerate(periods):
                                    if cts_dict[col]['worker_type_interaction'] and (period[-1] != 'b'):
                                        A_cts[col][period] = res_a[:, i]
                                    else:
                                        A_cts[col][period] = res_a[0, i]
                                del res_a

                        except ValueError as e:
                            # If constraints inconsistent, keep A_cts the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing A_cts for column {col!r}: {e}')
                        del XXwXX_cts[col], XXwY_cts[col]

                    if not cts_dict[col]['worker_type_interaction']:
                        # Restore A_sum with updated values
                        for period in periods:
                            A_sum[period] += A_cts[col][period] * C_dict[period][col]

                if params['update_s_movers']:
                    ## Update the variances ##
                    # Residuals
                    eps_sq = []

                    ## Update S ##
                    for l in range(nl):
                        # Update A_sum/S_sum_sq to account for worker-interaction terms
                        if any_controls:
                            # If controls, calculate S
                            A_sum_l, S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=True, periods=periods)
                        else:
                            # If no controls, don't calculate S
                            A_sum_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=False, periods=periods)

                        ## Residuals ##
                        eps_l_sq = []
                        # eps_l_sq_1
                        eps_l_sq.append(
                            (Y1 \
                                - (A['12'][l, G1] + A_sum['12'] + A_sum_l['12']) \
                                - R12 * (Y2 - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma'])) \
                                ) ** 2
                        )
                        # eps_l_sq_2
                        eps_l_sq.append(
                            (Y2 \
                                - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']) \
                                - (A['2mb'][G2] + A_sum['2mb'] + A_sum_l['2mb'])
                                ) ** 2
                        )
                        # eps_l_sq_3
                        eps_l_sq.append(
                            (Y3 \
                                - (A['3ma'][l, G2] + A_sum['3ma'] + A_sum_l['3ma']) \
                                - (A['3mb'][G1] + A_sum['3mb'] + A_sum_l['3mb'])
                                - R32m * (Y2 \
                                    - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']) \
                                    - (A['2mb'][G2] + A_sum['2mb'] + A_sum_l['2mb'])) \
                                ) ** 2
                        )
                        # eps_l_sq_4
                        eps_l_sq.append(
                            (Y4 \
                                - (A['43'][l, G2] + A_sum['43'] + A_sum_l['43']) \
                                - R43 * (Y3 \
                                    - (A['3ma'][l, G2] + A_sum['3ma'] + A_sum_l['3ma'])) \
                                ) ** 2
                        )
                        eps_sq.append(eps_l_sq)
                        del A_sum_l, eps_l_sq

                        ## XSwE ##
                        l_index, r_index = l * nk * len(periods_var), (l + 1) * nk * len(periods_var)

                        for t in range(4):
                            weights[l][t] *= eps_sq[l][t]

                        if any_controls:
                            ## Account for other variables' contribution to variance ##
                            var_l_numerator = [
                                (S['12'][l, :] ** 2)[G1],
                                (S['2ma'][l, :] ** 2)[G1],
                                (S['3ma'][l, :] ** 2)[G2],
                                (S['43'][l, :] ** 2)[G2]
                            ]
                            var_l_denominator = [
                                (S['12'][l, :] ** 2)[G1] \
                                    + S_sum_sq['12'] + S_sum_sq_l['12'],
                                ((S['2ma'][l, :] ** 2)[G1] \
                                    + S_sum_sq['2ma'] + S_sum_sq_l['2ma']),
                                ((S['3ma'][l, :] ** 2)[G2] \
                                    + S_sum_sq['3ma'] + S_sum_sq_l['3ma']),
                                (S['43'][l, :] ** 2)[G2] \
                                    + S_sum_sq['43'] + S_sum_sq_l['43'],
                            ]
                            del S_sum_sq_l

                            for t in range(4):
                                weights[l][t] *= (var_l_numerator[t] / var_l_denominator[t])

                        XSwE[l_index + 0 * nk: l_index + 1 * nk] = \
                            np.bincount(G1, weights=weights[l][0])
                        XSwE[l_index + 1 * nk: l_index + 2 * nk] = \
                            np.bincount(G1, weights=weights[l][1])
                        XSwE[l_index + 2 * nk: l_index + 3 * nk] = \
                            np.bincount(G2, weights=weights[l][2])
                        XSwE[l_index + 3 * nk: l_index + 4 * nk] = \
                            np.bincount(G2, weights=weights[l][3])

                        weights[l] = 0
                    del weights

                    if params['d_X_diag_movers_S'] > 1:
                        XSwXS += (params['d_X_diag_movers_S'] - 1)

                    try:
                        cons_s.solve(np.diag(XSwXS), -XSwE, solver='quadprog')
                        if cons_s.res is None:
                            # If estimation fails, keep S the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing S: estimates are None')
                        else:
                            res_s = cons_s.res.reshape((nl, len(periods_var), nk))
                            for i, period in enumerate(periods_var):
                                # if period[-1] != 'b':
                                S[period] = np.sqrt(res_s[:, i, :])
                                # else:
                                #     S[period] = np.sqrt(split_res[i][: nk])
                            del res_s

                    except ValueError as e:
                        # If constraints inconsistent, keep S the same
                        if params['verbose'] in [2, 3]:
                            print(f'Passing S: {e}')
                    del XSwXS, XSwE

                    ## Categorical ##
                    for col in cat_cols:
                        col_n = cat_dict[col]['n']

                        for l in range(nl):
                            # Update S_sum_sq to account for worker-interaction terms
                            S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_A=False, compute_S=True, periods=periods)

                            ### XSwE_cat ###
                            l_index, r_index = l * col_n * len(periods_var), (l + 1) * col_n * len(periods_var)

                            ## Compute var_l_cat ##
                            if cat_dict[col]['worker_type_interaction']:
                                S_l_dict = {period: (S_cat[col][period][l, :] ** 2)[C_dict[period][col]] for period in periods}
                            else:
                                S_l_dict = {period: (S_cat[col][period] ** 2)[C_dict[period][col]] for period in periods}

                            ## Account for other variables' contribution to variance ##
                            var_l_numerator = [
                                S_l_dict['12'],
                                S_l_dict['2ma'],
                                S_l_dict['3ma'],
                                S_l_dict['43']
                            ]
                            var_l_denominator = [
                                (S['12'][l, :] ** 2)[G1] \
                                    + S_sum_sq['12'] + S_sum_sq_l['12'],
                                ((S['2ma'][l, :] ** 2)[G1] \
                                    + S_sum_sq['2ma'] + S_sum_sq_l['2ma']),
                                ((S['3ma'][l, :] ** 2)[G2] \
                                    + S_sum_sq['3ma'] + S_sum_sq_l['3ma']),
                                (S['43'][l, :] ** 2)[G2] \
                                    + S_sum_sq['43'] + S_sum_sq_l['43'],
                            ]
                            del S_sum_sq_l

                            for t in range(4):
                                weights_cat[col][l][t] *= ((var_l_numerator[t] / var_l_denominator[t]) * eps_sq[l][t])

                            XSwE_cat[l_index + 0 * col_n: l_index + 1 * col_n] = \
                                np.bincount(CC1[col], weights=weights_cat[col][l][0])
                            XSwE_cat[l_index + 1 * col_n: l_index + 2 * col_n] = \
                                np.bincount(CC1[col], weights=weights_cat[col][l][1])
                            XSwE_cat[l_index + 2 * col_n: l_index + 3 * col_n] = \
                                np.bincount(CC2[col], weights=weights_cat[col][l][2])
                            XSwE_cat[l_index + 3 * col_n: l_index + 4 * col_n] = \
                                np.bincount(CC2[col], weights=weights_cat[col][l][3])

                            weights_cat[col][l] = 0
                        del weights_cat[col]

                        if params['d_X_diag_movers_S'] > 1:
                            XSwXS_cat[col] += (params['d_X_diag_movers_S'] - 1)

                        try:
                            s_solver = cons_s_dict[col]
                            s_solver.solve(np.diag(XSwXS_cat[col]), -XSwE_cat[col], solver='quadprog')
                            if s_solver.res is None:
                                # If estimation fails, keep S_cat the same
                                if params['verbose'] in [2, 3]:
                                    print(f'Passing S_cat for column {col!r}: estimates are None')
                            else:
                                res_s = s_solver.res.reshape((nl, len(periods_var), col_n))

                                if not cat_dict[col]['worker_type_interaction']:
                                    for period in periods_var:
                                        S_sum_sq[period] -= (S_cat[col][period] ** 2)[C_dict[period][col]]

                                for i, period in enumerate(periods_var):
                                    if cat_dict[col]['worker_type_interaction']: # and (period[-1] != 'b'):
                                        S_cat[col][period] = np.sqrt(res_s[:, i, :])
                                    else:
                                        S_cat[col][period] = np.sqrt(res_s[0, i, :])
                                del res_s

                                if not cat_dict[col]['worker_type_interaction']:
                                    for period in periods_var:
                                        S_sum_sq[period] += (S_cat[col][period] ** 2)[C_dict[period][col]]

                        except ValueError as e:
                            # If constraints inconsistent, keep S_cat the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing S_cat for column {col!r}: {e}')
                        del XSwXS_cat[col], XSwE_cat[col]

                    ## Continuous ##
                    for col in cts_cols:
                        for l in range(nl):
                            # Update S_sum_sq to account for worker-interaction terms
                            S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_A=False, compute_S=True, periods=periods)

                            ### XSwE_cts ###
                            l_index, r_index = l * len(periods_var), (l + 1) * len(periods_var)

                            ## Compute var_l_cts ##
                            if cts_dict[col]['worker_type_interaction']:
                                S_l_dict = {period: S_cts[col][period][l] ** 2 for period in periods}
                            else:
                                S_l_dict = {period: S_cts[col][period] ** 2 for period in periods}

                            ## Account for other variables' contribution to variance ##
                            var_l_numerator = [
                                S_l_dict['12'],
                                S_l_dict['2ma'],
                                S_l_dict['3ma'],
                                S_l_dict['43']
                            ]
                            var_l_denominator = [
                                (S['12'][l, :] ** 2)[G1] \
                                    + S_sum_sq['12'] + S_sum_sq_l['12'],
                                ((S['2ma'][l, :] ** 2)[G1] \
                                    + S_sum_sq['2ma'] + S_sum_sq_l['2ma']),
                                ((S['3ma'][l, :] ** 2)[G2] \
                                    + S_sum_sq['3ma'] + S_sum_sq_l['3ma']),
                                (S['43'][l, :] ** 2)[G2] \
                                    + S_sum_sq['43'] + S_sum_sq_l['43'],
                            ]
                            del S_sum_sq_l
                            for t in range(4):
                                weights_cts[col][l][t] *= ((var_l_numerator[t] / var_l_denominator[t]) * eps_sq[l][t])

                            # NOTE: take absolute value
                            XSwE_cts[l_index + 0] = \
                                np.abs(np.sum(C1[col] * weights_cts[col][l][0]))
                            XSwE_cts[l_index + 1] = \
                                np.abs(np.sum(C1[col] * weights_cts[col][l][1]))
                            XSwE_cts[l_index + 2] = \
                                np.abs(np.sum(C2[col] * weights_cts[col][l][2]))
                            XSwE_cts[l_index + 3] = \
                                np.abs(np.sum(C2[col] * weights_cts[col][l][3]))

                            weights_cts[col][l] = 0
                        del weights_cts[col]

                        if params['d_X_diag_movers_S'] > 1:
                            XSwXS_cts[col] += (params['d_X_diag_movers_S'] - 1)

                        try:
                            s_solver = cons_s_dict[col]
                            s_solver.solve(np.diag(XSwXS_cts[col]), -XSwE_cts[col], solver='quadprog')
                            if s_solver.res is None:
                                # If estimation fails, keep S_cts the same
                                if params['verbose'] in [2, 3]:
                                    print(f'Passing S_cts for column {col!r}: estimates are None')
                            else:
                                res_s = s_solver.res.reshape((nl, len(periods_var)))

                                if not cts_dict[col]['worker_type_interaction']:
                                    for period in periods_var:
                                        S_sum_sq[period] -= S_cts[col][period] ** 2

                                for i, period in enumerate(periods_var):
                                    if cat_dict[col]['worker_type_interaction']: # and (period[-1] != 'b'):
                                        S_cts[col][period] = np.sqrt(res_s[:, i])
                                    else:
                                        S_cts[col][period] = np.sqrt(res_s[0, i])
                                del res_s

                                if not cts_dict[col]['worker_type_interaction']:
                                    for period in periods_var:
                                        S_sum_sq[period] -= S_cts[col][period] ** 2

                        except ValueError as e:
                            # If constraints inconsistent, keep S_cts the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing S_cts for column {col!r}: {e}')
                        del XSwXS_cts[col], XSwE_cts[col]

                    del eps_sq

                # print('A after:')
                # print(A)
                # print('S after:')
                # print(S)
                # print('A_cat after:')
                # print(A_cat)
                # print('S_cat after:')
                # print(S_cat)
                # print('A_cts after:')
                # print(A_cts)
                # print('S_cts after:')
                # print(S_cts)

        if store_res:
            if len(cat_cols) > 0:
                ## Normalize ##
                # NOTE: normalize here because constraints don't normalize unless categorical controls are using constraints, and even when used, constraints don't always normalize to exactly 0
                A, A_cat = self._normalize(A, A_cat)

            ## Sort parameters ##
            A, S, A_cat, S_cat, A_cts, S_cts, pk1, self.pk0 = self._sort_parameters(A, S, A_cat, S_cat, A_cts, S_cts, pk1, self.pk0)

            if len(cat_cols) > 0:
                ## Normalize again ##
                A, A_cat = self._normalize(A, A_cat)

            # Store parameters
            self.A, self.A_cat, self.A_cts = A, A_cat, A_cts
            self.S, self.S_cat, self.S_cts = S, S_cat, S_cts
            self.R12, self.R43, self.R32m = R12, R43, R32m
            self.pk1 = pk1
            self.lik1, self.liks1 = lik1, liks1 # np.concatenate([self.liks1, liks1])

            # Update NNm
            if compute_NNm:
                self.NNm = jdata.groupby('g1')['g4'].value_counts().unstack(fill_value=0).to_numpy()

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
        periods, periods_update = self.periods_stayers, self.periods_update_stayers
        R12, R43, R32s = self.R12, self.R43, self.R32s
        A, A_cat, A_cts = self.A, self.A_cat, self.A_cts
        S, S_cat, S_cts = self.S, self.S_cat, self.S_cts
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        cat_dict, cts_dict = self.cat_dict, self.cts_dict
        controls_dict = self.controls_dict
        any_controls = self.any_controls

        # Store wage outcomes and groups
        Y1 = sdata['y1'].to_numpy()
        Y2 = sdata['y2'].to_numpy()
        Y3 = sdata['y3'].to_numpy()
        Y4 = sdata['y4'].to_numpy()
        G1 = sdata['g1'].to_numpy().astype(int, copy=False)
        # G2 = sdata['g4'].to_numpy().astype(int, copy=False)

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
            elif n_subcols == 4:
                # If column can change over time
                subcol_1 = subcols[0]
                subcol_2 = subcols[3]
            else:
                raise NotImplementedError(f'Column names must have either one or four associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
            if i < len(cat_cols):
                # Categorical
                C1[col] = sdata.loc[:, subcol_1].to_numpy().astype(int, copy=False)
                C2[col] = sdata.loc[:, subcol_2].to_numpy().astype(int, copy=False)
            else:
                # Continuous
                C1[col] = sdata.loc[:, subcol_1].to_numpy()
                C2[col] = sdata.loc[:, subcol_2].to_numpy()

        ## Sparse matrix representations ##
        GG1 = csc_matrix((np.ones(ni), (range(ni), G1)), shape=(ni, nk))
        CC1 = {col: csc_matrix((np.ones(ni), (range(ni), C1[col])), shape=(ni, controls_dict[col]['n'])) for col in cat_cols}
        # CC2 = {col: csc_matrix((np.ones(ni), (range(ni), C2[col])), shape=(ni, controls_dict[col]['n'])) for col in cat_cols}
        ## Dictionaries linking periods to vectors/matrices ##
        C_dict = {period: C1 if period in self.first_periods else C2 for period in periods}
        # CC_dict = {period: CC1 if period in self.first_periods else CC2 for period in periods}

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
            A_sum, S_sum_sq = self._sum_by_non_nl(ni=ni, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)

            for l in range(nl):
                # Update A_sum/S_sum_sq to account for worker-interaction terms
                A_sum_l, S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)

                lp1 = lognormpdf(
                    Y1 - R12 * (Y2 - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma'])),
                    A['12'][l, G1] + A_sum['12'] + A_sum_l['12'],
                    var=\
                        (S['12'][l, :] ** 2)[G1] + S_sum_sq['12'] + S_sum_sq_l['12'] # \
                        # + (R12 ** 2) \
                        #     * ((S['2ma'][l, :] ** 2)[G1] \
                        #         + S_sum_sq['2ma'] \
                        #         + S_sum_sq_l['2ma'])
                )
                lp4 = lognormpdf(
                    Y4 - R43 * (Y3 - (A['3ma'][l, G1] + A_sum['3ma'] + A_sum_l['3ma'])),
                    A['43'][l, G1] + A_sum['43'] + A_sum_l['43'],
                    var=\
                        (S['43'][l, :] ** 2)[G1] + S_sum_sq['43'] + S_sum_sq_l['43'] # \
                        # + (R43 ** 2) \
                        #     * ((S['3ma'][l, :] ** 2)[G1] \
                        #         + S_sum_sq['3ma'] \
                        #         + S_sum_sq_l['3ma'])
                )

                lp_stable[:, l] = lp1 + lp4
        else:
            A_sum = {period: 0 for period in self.all_periods}
            S_sum_sq = {period: 0 for period in self.all_periods}
            # Loop over firm classes so means/variances are single values rather than vectors (computing log/square is faster this way)
            for g1 in range(nk):
                I = (G1 == g1)
                for l in range(nl):
                    lp1 = lognormpdf(
                        Y1[I] - R12 * (Y2[I] - A['2ma'][l, g1]),
                        A['12'][l, g1],
                        var=S['12'][l, g1] ** 2 # + (R12 * S['2ma'][l, g1]) ** 2
                    )
                    lp4 = lognormpdf(
                        Y4[I] - R43 * (Y3[I] - A['3ma'][l, g1]),
                        A['43'][l, g1],
                        var=S['43'][l, g1] ** 2 # + (R43 * S['3ma'][l, g1]) ** 2
                    )

                    lp_stable[I, l] = lp1 + lp4
        del lp1, lp4

        ## Constraints ##
        min_firm_type = self._min_firm_type(A)
        cons_a, cons_s, cons_a_dict, cons_s_dict = self._gen_constraints(min_firm_type=min_firm_type, for_movers=False)

        for iter in range(params['n_iters_stayers']):
            # ---------- E-Step ----------
            log_pk0 = np.log(pk0)
            if any_controls:
                ## Account for control variables ##
                if iter == 0:
                    A_sum, S_sum_sq = self._sum_by_non_nl(ni=ni, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)
                else:
                    S_sum_sq = self._sum_by_non_nl(ni=ni, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_A=False, periods=periods)

                for l in range(nl):
                    # Update A_sum/S_sum_sq to account for worker-interaction terms
                    A_sum_l, S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)

                    lp2 = lognormpdf(
                        Y2,
                        A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s'],
                        var=\
                            (S['2s'][l, :] ** 2)[G1] + S_sum_sq['2s'] + S_sum_sq_l['2s']
                    )
                    lp3 = lognormpdf(
                        Y3 - R32s * (Y2 - (A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s'])),
                        A['3s'][l, G1] + A_sum['3s'] + A_sum_l['3s'],
                        var=\
                            (S['3s'][l, :] ** 2)[G1] + S_sum_sq['3s'] + S_sum_sq_l['3s'] # \
                            # + (R32s ** 2) \
                            #     * ((S['2s'][l, :] ** 2)[G1] \
                            #         + S_sum_sq['2s'] + S_sum_sq_l['2s'])
                    )

                    lp[:, l] = log_pk0[G1, l] + lp2 + lp3
            else:
                A_sum = {period: 0 for period in self.all_periods}
                S_sum_sq = {period: 0 for period in self.all_periods}
                # Loop over firm classes so means/variances are single values rather than vectors (computing log/square is faster this way)
                for g1 in range(nk):
                    I = (G1 == g1)
                    for l in range(nl):
                        lp2 = lognormpdf(
                            Y2[I],
                            A['2s'][l, g1],
                            sd=S['2s'][l, g1]
                        )
                        lp3 = lognormpdf(
                            Y3[I] - R32s * (Y2[I] - A['2s'][l, g1]),
                            A['3s'][l, g1],
                            var=S['3s'][l, g1] ** 2 # + (R32s * S['2s'][l, g1]) ** 2
                        )

                        lp[I, l] = log_pk0[G1[I], l] + lp2 + lp3
            del log_pk0, lp2, lp3

            # We compute log sum exp to get likelihoods and probabilities
            lp += lp_stable
            lse_lp = logsumexp(lp, axis=1)
            qi = np.exp(lp.T - lse_lp).T
            if params['return_qi']:
                return qi
            lik0 = lse_lp.mean()
            del lse_lp
            if (iter > 0) and params['update_pk0']:
                # Account for Dirichlet prior
                lik_prior = (d_prior - 1) * np.sum(np.log(pk0))
                lik0 += lik_prior
            liks0.append(lik0)
            if params['verbose'] == 3:
                print('loop {}, liks {}'.format(iter, lik0))

            if (abs(lik0 - prev_lik) < params['threshold_movers']):
                # Break loop
                break
            prev_lik = lik0

            # ---------- Update pk0 ----------
            if params['update_pk0']:
                # NOTE: add dirichlet prior
                pk0 = GG1.T @ (qi + d_prior - 1)
                # Normalize rows to sum to 1
                pk0 = DxM(1 / np.sum(pk0, axis=1), pk0)

                if pd.isna(pk0).any():
                    warnings.warn('Estimated pk0 has NaN values. Please try a different set of starting values.')
                    break
                    # raise ValueError('Estimated pk0 has NaN values. Please try a different set of starting values.')

            # ---------- M-step ----------
            # Alternate between updating A/S and updating rho
            if params['update_rho32s'] and ((iter % 2) == 1):
                ## Update rho ##
                XX32s = np.zeros(nl * ni)
                YY32s = np.zeros(nl * ni)
                WW32s = np.zeros(nl * ni)
                for l in range(nl):
                    A_sum_l, S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)

                    XX32s[l * ni: (l + 1) * ni] = Y2 - A['2s'][l, G1] - A_sum['2s'] - A_sum_l['2s']
                    YY32s[l * ni: (l + 1) * ni] = Y3 - A['3s'][l, G1] - A_sum['3s'] - A_sum_l['3s']
                    SS32s = ( \
                        (S['3s'][l, :] ** 2)[G1] + S_sum_sq['3s'] + S_sum_sq_l['3s']) # \
                        # + (R32s ** 2) \
                        #     * ((S['2s'][l, :] ** 2)[G1] \
                        #         + S_sum_sq['2s'] + S_sum_sq_l['2s']))
                    WW32s[l * ni: (l + 1) * ni] = qi[:, l] / np.sqrt(SS32s)

                ## OLS ##
                Xw = XX32s * WW32s
                XwX = np.sum(Xw * XX32s)
                XwY = np.sum(Xw * YY32s)
                R32s = XwY / XwX
                del Xw, XwX, XwY
            elif params['update_a_stayers'] or params['update_s_stayers']:
                # Constrained OLS (source: https://scaron.info/blog/quadratic-programming-in-python.html)

                # The regression has 2 * nk parameters and 2 * ni rows
                # To avoid duplicating the data 2 * nl times, we construct X'X and X'Y by looping over nl
                # We also note that X'X is block diagonal with nl matrices of dimension (2 * nk, 2 * nk)

                ## General ##
                # Shift between periods
                ts = nl * nk
                # XX
                XX = lil_matrix((2 * ni, len(periods_update) * nk))
                # X'X (weighted)
                XXwXX = np.zeros((len(periods_update) * ts, len(periods_update) * ts))
                if params['update_a_stayers']:
                    XXwY = np.zeros(shape=len(periods_update) * ts)

                ## Compute X terms ##
                # X =
                # +---------+---------+---------------+---------+--------------+--------------+
                # | A['12'] | A['43'] | A['2s']       | A['3s'] |   A['2ma']   |   A['3ma']   |
                # +=========+=========+===============+=========+==============+==============+
                # |   GG1   |    0    | 0             | 0       | -(R12 * GG1) |       0      |
                # +---------+---------+---------------+---------+--------------+--------------+
                # |    0    |    0    | GG1           | 0       |       0      |       0      |
                # +---------+---------+---------------+---------+--------------+--------------+
                # |    0    |    0    | -(R32s * GG1) | GG1     |       0      |       0      |
                # +---------+---------+---------------+---------+--------------+--------------+
                # |    0    |   GG1   | 0             | 0       |       0      | -(R43 * GG1) |
                # +---------+---------+---------------+---------+--------------+--------------+

                # Y2 = A['2s']
                XX[0 * ni: 1 * ni, 0 * nk: 1 * nk] = GG1
                # Y3 = A['3s'] + R32s * (Y2 - A['2s'])
                XX[1 * ni: 2 * ni, 0 * nk: 1 * nk] = -(R32s * GG1)
                XX[1 * ni: 2 * ni, 1 * nk: 2 * nk] = GG1
                XX = XX.tocsc()

                ## Categorical ##
                if len(cat_cols) > 0:
                    ts_cat = {col: nl * col_dict['n'] for col, col_dict in cat_dict.items()}
                    # XX_cat
                    XX_cat = {col: lil_matrix((2 * ni, len(periods_update) * cat_dict[col]['n'])) for col in cat_cols}
                    # XX_cat'XX_cat (weighted)
                    XXwXX_cat = {col: np.zeros((len(periods_update) * col_ts, len(periods_update) * col_ts)) for col, col_ts in ts_cat.items()}
                    if params['update_a_stayers']:
                        XXwY_cat = {col: np.zeros(shape=len(periods_update) * col_ts) for col, col_ts in ts_cat.items()}

                    for col in cat_cols:
                        ## Compute XX_cat terms ##
                        col_n = cat_dict[col]['n']
                        # Y2 = A['2s']
                        XX_cat[col][0 * ni: 1 * ni, 0 * col_n: 1 * col_n] = CC1[col]
                        # Y3 = A['3s'] + R32s * (Y2 - A['2s'])
                        XX_cat[col][1 * ni: 2 * ni, 0 * col_n: 1 * col_n] = -(R32s * CC1[col])
                        XX_cat[col][1 * ni: 2 * ni, 1 * col_n: 2 * col_n] = CC1[col]
                        XX_cat[col] = XX_cat[col].tocsc()

                ### Continuous ###
                if len(cts_cols) > 0:
                    # XX_cts
                    XX_cts = {col: lil_matrix((2 * ni, len(periods_update))) for col in cts_cols}
                    # XX_cts'XX_cts (weighted)
                    XXwXX_cts = {col: np.zeros((len(periods_update) * nl, len(periods_update) * nl)) for col in cts_cols}
                    if params['update_a_stayers']:
                        XXwY_cts = {col: np.zeros(shape=len(periods_update) * nl) for col in cts_cols}

                    for col in cts_cols:
                        ## Compute XX_cts terms ##
                        # Y2 = A['2s']
                        XX_cts[col][0 * ni: 1 * ni, 0] = C1[col]
                        # Y3 = A['3s'] + R32s * (Y2 - A['2s'])
                        XX_cts[col][1 * ni: 2 * ni, 0] = -(R32s * C1[col])
                        XX_cts[col][1 * ni: 2 * ni, 1] = C1[col]
                        XX_cts[col] = XX_cts[col].tocsc()

                ## Update A ##
                if params['update_s_stayers']:
                    # Store weights computed for A for use when computing S
                    weights = []
                for l in range(nl):
                    l_index, r_index = l * nk * len(periods_update), (l + 1) * nk * len(periods_update)

                    ## Compute weights_l ##
                    var_l = np.concatenate(
                        [
                            (S['2s'][l, :] ** 2)[G1],
                            (S['3s'][l, :] ** 2)[G1] # + (R32s * S['2s'][l, :]) ** 2)[G1]
                        ]
                    )
                    weights_l = np.tile(qi[:, l], 2) / np.sqrt(var_l)
                    del var_l
                    if params['update_s_stayers']:
                        weights.append(weights_l)

                    ## Compute XXw_l ##
                    XXw_l = DxSP(weights_l, XX).T
                    del weights_l

                    ## Compute XwX_l ##
                    XXwXX[l_index: r_index, l_index: r_index] = (XXw_l @ XX).todense()

                    if params['update_a_stayers']:
                        Y_l = np.zeros(2 * ni)

                        # Update A_sum to account for worker-interaction terms
                        A_sum_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=False, periods=periods)

                        # Y2_l
                        Y_l[0 * ni: 1 * ni] = \
                            Y2 \
                                - (A_sum['2s'] + A_sum_l['2s'])
                        # Y3_l
                        Y_l[1 * ni: 2 * ni] = \
                            Y3 \
                                - (A_sum['3s'] + A_sum_l['3s']) \
                                - R32s * (Y2 - (A_sum['2s'] + A_sum_l['2s']))

                        ## Compute XXwY_l ##
                        XXwY[l_index: r_index] = XXw_l @ Y_l
                        del Y_l, A_sum_l
                    del XXw_l

                if params['d_X_diag_stayers'] > 1:
                    XXwXX += (params['d_X_diag_stayers'] - 1) * np.eye(XXwXX.shape[0])

                # print('A before:')
                # print(A)
                # print('S before:')
                # print(S)
                # print('A_cat before:')
                # print(A_cat)
                # print('S_cat before:')
                # print(S_cat)
                # print('A_cts before:')
                # print(A_cts)
                # print('S_cts before:')
                # print(S_cts)

                # We solve the system to get all the parameters (use dense solver)
                if params['update_a_stayers']:
                    if iter > 0:
                        ## Constraints ##
                        cons_a, cons_s, cons_a_dict, cons_s_dict = self._gen_constraints(min_firm_type=min_firm_type, for_movers=False)
                    try:
                        cons_a.solve(XXwXX, -XXwY, solver='quadprog')
                        if cons_a.res is None:
                            # If estimation fails, keep A the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing A: estimates are None')
                        else:
                            res_a = cons_a.res.reshape((nl, len(periods_update), nk))
                            A['2s'] = res_a[:, 0, :]
                            A['3s'] = res_a[:, 1, :]
                            del res_a

                    except ValueError as e:
                        # If constraints inconsistent, keep A the same
                        if params['verbose'] in [2, 3]:
                            print(f'Passing A: {e}')
                    del XXwXX, XXwY

                ## Categorical ##
                if params['update_s_stayers']:
                    # Store weights computed for A_cat for use when computing S_cat
                    weights_cat = {col: [] for col in cat_cols}
                for col in cat_cols:
                    col_n = cat_dict[col]['n']

                    if not cat_dict[col]['worker_type_interaction']:
                        # Adjust A_sum
                        for period in periods_update:
                            A_sum[period] -= A_cat[col][period][C_dict[period][col]]

                    for l in range(nl):
                        l_index, r_index = l * col_n * len(periods_update), (l + 1) * col_n * len(periods_update)

                        ## Compute weights_l ##
                        if cat_dict[col]['worker_type_interaction']:
                            S_l_dict = {period: (S_cat[col][period][l, :] ** 2)[C_dict[period][col]] for period in periods_update}
                        else:
                            S_l_dict = {period: (S_cat[col][period] ** 2)[C_dict[period][col]] for period in periods_update}

                        var_l = np.concatenate(
                            [
                                S_l_dict['2s'],
                                S_l_dict['3s'] # + (R32s ** 2) * S_l_dict['2s']
                            ]
                        )
                        del S_l_dict
                        weights_l = np.tile(qi[:, l], 2) / np.sqrt(var_l)
                        del var_l
                        if params['update_s_stayers']:
                            weights_cat[col].append(weights_l)

                        ## Compute XXw_cat_l ##
                        XXw_cat_l = DxSP(weights_l, XX_cat[col]).T
                        del weights_l

                        ## Compute XXwXX_cat_l ##
                        XXwXX_cat[col][l_index: r_index, l_index: r_index] = (XXw_cat_l @ XX_cat[col]).todense()

                        if params['update_a_stayers']:
                            Y_cat_l = np.zeros(2 * ni)

                            # Update A_sum to account for worker-interaction terms
                            A_sum_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=False, periods=periods_update)
                            if cat_dict[col]['worker_type_interaction']:
                                # Adjust A_sum
                                for period in periods_update:
                                    A_sum_l[period] -= A_cat[col][period][l, C_dict[period][col]]

                            # Y2_cat_l
                            Y_cat_l[0 * ni: 1 * ni] = \
                                Y2 \
                                    - (A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s'])
                            # Y3_cat_l
                            Y_cat_l[1 * ni: 2 * ni] = \
                                Y3 \
                                    - (A['3s'][l, G1] + A_sum['3s'] + A_sum_l['3s']) \
                                    - R32s * (Y2 - (A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s']))

                            ## Compute XwY_cat_l ##
                            XXwY_cat[col][l_index: r_index] = XXw_cat_l @ Y_cat_l
                            del Y_cat_l, A_sum_l
                        del XXw_cat_l

                    if params['d_X_diag_stayers'] > 1:
                        XXwXX_cat[col] += (params['d_X_diag_stayers'] - 1) * np.eye(XXwXX_cat[col].shape[0])

                    # We solve the system to get all the parameters (use dense solver)
                    if params['update_a_stayers']:
                        try:
                            a_solver = cons_a_dict[col]
                            a_solver.solve(XXwXX_cat[col], -XXwY_cat[col], solver='quadprog')
                            if a_solver.res is None:
                                # If estimation fails, keep A_cat the same
                                if params['verbose'] in [2, 3]:
                                    print(f'Passing A_cat for column {col!r}: estimates are None')
                            else:
                                res_a = a_solver.res.reshape((nl, len(periods_update), col_n))
                                if cat_dict[col]['worker_type_interaction']:
                                    A_cat[col]['2s'] = res_a[:, 0, :]
                                    A_cat[col]['3s'] = res_a[:, 1, :]
                                else:
                                    A_cat[col]['2s'] = res_a[0, 0, :]
                                    A_cat[col]['3s'] = res_a[0, 1, :]
                                del res_a

                        except ValueError as e:
                            # If constraints inconsistent, keep A_cat the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing A_cat for column {col!r}: {e}')
                        del XXwXX_cat[col], XXwY_cat[col]

                    if not cat_dict[col]['worker_type_interaction']:
                        # Restore A_sum with updated values
                        for period in periods_update:
                            A_sum[period] += A_cat[col][period][C_dict[period][col]]

                ## Continuous ##
                if params['update_s_stayers']:
                    # Store weights computed for A_cts for use when computing S_cts
                    weights_cts = {col: [] for col in cts_cols}
                for col in cts_cols:
                    if not cts_dict[col]['worker_type_interaction']:
                        # Adjust A_sum
                        for period in periods_update:
                            A_sum[period] -= A_cts[col][period] * C_dict[period][col]

                    for l in range(nl):
                        l_index, r_index = l * len(periods_update), (l + 1) * len(periods_update)

                        ## Compute weights_l ##
                        if cts_dict[col]['worker_type_interaction']:
                            S_l_dict = {period: S_cts[col][period][l] ** 2 for period in periods_update}
                        else:
                            S_l_dict = {period: S_cts[col][period] ** 2 for period in periods_update}

                        var_l = np.concatenate(
                            [
                                S_l_dict['2s'],
                                S_l_dict['3s'] # + (R32s ** 2) * S_l_dict['2s']
                            ]
                        )
                        del S_l_dict
                        weights_l = np.tile(qi[:, l], 2) / np.repeat(np.sqrt(var_l), ni)
                        del var_l
                        if params['update_s_stayers']:
                            weights_cts[col].append(weights_l)

                        ## Compute XXw_cts_l ##
                        XXw_cts_l = DxSP(weights_l, XX_cts[col]).T
                        del weights_l

                        ## Compute XXwXX_cts_l ##
                        XXwXX_cts[col][l_index: r_index, l_index: r_index] = (XXw_cts_l @ XX_cts[col]).todense()

                        if params['update_a_stayers']:
                            Y_cts_l = np.zeros(2 * ni)

                            # Update A_sum to account for worker-interaction terms
                            A_sum_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=False, periods=periods_update)
                            if cts_dict[col]['worker_type_interaction']:
                                # Adjust A_sum
                                for period in periods_update:
                                    A_sum_l[period] -= A_cts[col][period][l] * C_dict[period][col]

                            # Y2_cts_l
                            Y_cts_l[0 * ni: 1 * ni] = \
                                Y2 \
                                    - (A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s'])
                            # Y3_cts_l
                            Y_cts_l[1 * ni: 2 * ni] = \
                                Y3 \
                                    - (A['3s'][l, G1] + A_sum['3s'] + A_sum_l['3s']) \
                                    - R32s * (Y2 - (A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s']))

                            ## Compute XwY_cts_l ##
                            XXwY_cts[col][l_index: r_index] = XXw_cts_l @ Y_cts_l
                            del Y_cts_l, A_sum_l
                        del XXw_cts_l

                    if params['d_X_diag_stayers'] > 1:
                        XXwXX_cts[col] += (params['d_X_diag_stayers'] - 1) * np.eye(XXwXX_cts[col].shape[0])

                    # We solve the system to get all the parameters (use dense solver)
                    if params['update_a_stayers']:
                        try:
                            a_solver = cons_a_dict[col]
                            a_solver.solve(XXwXX_cts[col], -XXwY_cts[col], solver='quadprog')
                            if a_solver.res is None:
                                # If estimation fails, keep A_cts the same
                                if params['verbose'] in [2, 3]:
                                    print(f'Passing A_cts for column {col!r}: estimates are None')
                            else:
                                res_a = a_solver.res.reshape((nl, len(periods_update)))
                                if cts_dict[col]['worker_type_interaction']:
                                    A_cts[col]['2s'] = res_a[:, 0]
                                    A_cts[col]['3s'] = res_a[:, 1]
                                else:
                                    A_cts[col]['2s'] = res_a[0, 0]
                                    A_cts[col]['3s'] = res_a[0, 1]
                                del res_a

                        except ValueError as e:
                            # If constraints inconsistent, keep A_cts the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing A_cts for column {col!r}: {e}')
                        del XXwXX_cts[col], XXwY_cts[col]

                    if not cts_dict[col]['worker_type_interaction']:
                        # Restore A_sum with updated values
                        for period in periods_update:
                            A_sum[period] += A_cts[col][period] * C_dict[period][col]

                if params['update_s_stayers']:
                    ## Update the variances ##
                    if iter == 0:
                        XSwXS = np.zeros(len(periods_update) * ts)
                        XSwE = np.zeros(shape=len(periods_update) * ts)

                        ## Categorical ##
                        if len(cat_cols) > 0:
                            XSwXS_cat = {col: np.zeros(shape=len(periods_update) * col_ts) for col, col_ts in ts_cat.items()}
                            XSwE_cat = {col: np.zeros(shape=len(periods_update) * col_ts) for col, col_ts in ts_cat.items()}

                        ## Continuous ##
                        if len(cts_cols) > 0:
                            XSwXS_cts = {col: np.zeros(shape=len(periods_update) * nl) for col in cts_cols}
                            XSwE_cts = {col: np.zeros(shape=len(periods_update) * nl) for col in cts_cols}

                    ## Residuals ##
                    eps_sq = []

                    ## Update S ##
                    for l in range(nl):
                        # Update A_sum/S_sum_sq to account for worker-interaction terms
                        if any_controls:
                            # If controls, calculate S
                            A_sum_l, S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=True, periods=periods)
                        else:
                            # If no controls, don't calculate S
                            A_sum_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=False, periods=periods)

                        ## Residuals ##
                        eps_l_sq = np.zeros(2 * ni)
                        # eps_2_l_sq
                        eps_l_sq[0 * ni: 1 * ni] = \
                            (Y2 \
                                - (A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s'])
                                ) ** 2
                        # eps_3_l_sq
                        eps_l_sq[1 * ni: 2 * ni] = \
                            (Y3 \
                                - (A['3s'][l, G1] + A_sum['3s'] + A_sum_l['3s']) \
                                - R32s * (Y2 - (A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s']))
                                ) ** 2
                        eps_sq.append(eps_l_sq)
                        del A_sum_l, eps_l_sq

                        ## XSwXS and XSwE terms ##
                        l_index, r_index = l * nk * len(periods_update), (l + 1) * nk * len(periods_update)

                        ## First, XSwXS ##
                        XSwXS[l_index + 0 * nk: l_index + 1 * nk] = \
                            np.bincount(G1, weights=weights[l][0 * ni: 1 * ni])
                        XSwXS[l_index + 1 * nk: l_index + 2 * nk] = \
                            np.bincount(G1, weights=weights[l][1 * ni: 2 * ni])

                        ## Second, XSwE ##
                        weights[l] *= eps_sq[l]

                        if any_controls:
                            ## Account for other variables' contribution to variance ##
                            var_l_numerator = np.concatenate(
                                [
                                    (S['2s'][l, :] ** 2)[G1],
                                    (S['3s'][l, :] ** 2)[G1] # + (R32s * S['2s'][l, :]) ** 2)[G1]
                                ]
                            )
                            var_l_denominator = np.concatenate(
                                [
                                    (S['2s'][l, :] ** 2)[G1] \
                                        + S_sum_sq['2s'] + S_sum_sq_l['2s'],
                                    (S['3s'][l, :] ** 2)[G1] \
                                        + S_sum_sq['3s'] + S_sum_sq_l['3s'] # \
                                        # + (R32s ** 2) \
                                        #     * ((S['2s'][l, :] ** 2)[G1] \
                                        #         + S_sum_sq['2s'] + S_sum_sq_l['2s'])
                                ]
                            )
                            del S_sum_sq_l
                            weights[l] *= (var_l_numerator / var_l_denominator)

                        XSwE[l_index + 0 * nk: l_index + 1 * nk] = \
                            np.bincount(G1, weights=weights[l][0 * ni: 1 * ni])
                        XSwE[l_index + 1 * nk: l_index + 2 * nk] = \
                            np.bincount(G1, weights=weights[l][1 * ni: 2 * ni])

                        weights[l] = 0
                    del weights

                    try:
                        cons_s.solve(np.diag(XSwXS), -XSwE, solver='quadprog')
                        if cons_s.res is None:
                            # If estimation fails, keep S the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing S: estimates are None')
                        else:
                            res_s = cons_s.res.reshape((nl, len(periods_update), nk))
                            S['2s'] = np.sqrt(res_s[:, 0, :])
                            S['3s'] = np.sqrt(res_s[:, 1, :])
                            del res_s

                    except ValueError as e:
                        # If constraints inconsistent, keep S the same
                        if params['verbose'] in [2, 3]:
                            print(f'Passing S: {e}')

                    ## Categorical ##
                    for col in cat_cols:
                        col_n = cat_dict[col]['n']

                        for l in range(nl):
                            # Update S_sum_sq to account for worker-interaction terms
                            S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_A=False, compute_S=True, periods=periods)

                            ### XSwXS_cat and XSwE_cat terms ###
                            l_index, r_index = l * col_n * len(periods_update), (l + 1) * col_n * len(periods_update)

                            ### First, XSwXS_cat ###
                            XSwXS_cat[l_index + 0 * col_n: l_index + 1 * col_n] = \
                                np.bincount(CC1[col], weights=weights_cat[col][l][0 * ni: 1 * ni])
                            XSwXS_cat[l_index + 1 * col_n: l_index + 2 * col_n] = \
                                np.bincount(CC1[col], weights=weights_cat[col][l][1 * ni: 2 * ni])

                            ### Second, XSwE_cat ###
                            ## Compute var_l_cat ##
                            if cat_dict[col]['worker_type_interaction']:
                                S_l_dict = {period: (S_cat[col][period][l, :] ** 2)[C_dict[period][col]] for period in periods_update}
                            else:
                                S_l_dict = {period: (S_cat[col][period] ** 2)[C_dict[period][col]] for period in periods_update}

                            ## Account for other variables' contribution to variance ##
                            var_l_numerator = np.concatenate(
                                [
                                    S_l_dict['2s'],
                                    S_l_dict['3s'] # + (R32s ** 2) * S_l_dict['2s']
                                ]
                            )
                            var_l_denominator = np.concatenate(
                                [
                                    (S['2s'][l, :] ** 2)[G1] \
                                        + S_sum_sq['2s'] + S_sum_sq_l['2s'],
                                    (S['3s'][l, :] ** 2)[G1] \
                                        + S_sum_sq['3s'] + S_sum_sq_l['3s'] # \
                                        # + (R32s ** 2) \
                                        #     * ((S['2s'][l, :] ** 2)[G1] \
                                        #         + S_sum_sq['2s'] + S_sum_sq_l['2s'])
                                ]
                            )
                            del S_sum_sq_l
                            weights_cat[col][l] *= ((var_l_numerator / var_l_denominator) * eps_sq[l])

                            XSwE_cat[l_index + 0 * col_n: l_index + 1 * col_n] = \
                                np.bincount(CC1[col], weights=weights_cat[col][l][0 * ni: 1 * ni])
                            XSwE_cat[l_index + 1 * col_n: l_index + 2 * col_n] = \
                                np.bincount(CC1[col], weights=weights_cat[col][l][1 * ni: 2 * ni])

                            weights_cat[col][l] = 0
                        del weights_cat[col]

                        try:
                            s_solver = cons_s_dict[col]
                            s_solver.solve(np.diag(XSwXS_cat[col]), -XSwE_cat[col], solver='quadprog')
                            if s_solver.res is None:
                                # If estimation fails, keep S_cat the same
                                if params['verbose'] in [2, 3]:
                                    print(f'Passing S_cat for column {col!r}: estimates are None')
                            else:
                                res_s = s_solver.res.reshape((nl, len(periods_update), col_n))

                                if not cat_dict[col]['worker_type_interaction']:
                                    for period in periods_update:
                                        S_sum_sq[period] -= (S_cat[col][period] ** 2)[C_dict[period][col]]

                                if cat_dict[col]['worker_type_interaction']:
                                    S_cat[col]['2s'] = np.sqrt(res_s[:, 0, :])
                                    S_cat[col]['3s'] = np.sqrt(res_s[:, 1, :])
                                else:
                                    S_cat[col]['2s'] = np.sqrt(res_s[0, 0, :])
                                    S_cat[col]['3s'] = np.sqrt(res_s[0, 1, :])
                                del res_s

                                if not cat_dict[col]['worker_type_interaction']:
                                    for period in periods_update:
                                        S_sum_sq[period] += (S_cat[col][period] ** 2)[C_dict[period][col]]

                        except ValueError as e:
                            # If constraints inconsistent, keep S_cat the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing S_cat for column {col!r}: {e}')

                    ## Continuous ##
                    for col in cts_cols:
                        for l in range(nl):
                            # Update S_sum_sq to account for worker-interaction terms
                            S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_A=False, compute_S=True, periods=periods)

                            ### XSwXS_cts and XSwE_cts terms ###
                            l_index, r_index = l * len(periods_update), (l + 1) * len(periods_update)

                            ### First, XSwXS_cts ###
                            XSwXS_cts[l_index + 0] = \
                                np.sum(C1[col] * weights_cts[col][l][0 * ni: 1 * ni])
                            XSwXS_cts[l_index + 1] = \
                                np.sum(C1[col] * weights_cts[col][l][1 * ni: 2 * ni])

                            ### Second, XSwE_cts ###
                            ## Compute var_l_cts ##
                            if cts_dict[col]['worker_type_interaction']:
                                S_l_dict = {period: S_cts[col][period][l] ** 2 for period in periods_update}
                            else:
                                S_l_dict = {period: S_cts[col][period] ** 2 for period in periods_update}

                            ## Account for other variables' contribution to variance ##
                            var_l_numerator = np.concatenate(
                                [
                                    S_l_dict['2s'],
                                    S_l_dict['3s'] # + (R32s ** 2) * S_l_dict['2s']
                                ]
                            )
                            var_l_denominator = np.concatenate(
                                [
                                    (S['2s'][l, :] ** 2)[G1] \
                                        + S_sum_sq['2s'] + S_sum_sq_l['2s'],
                                    (S['3s'][l, :] ** 2)[G1] \
                                        + S_sum_sq['3s'] + S_sum_sq_l['3s'] # \
                                        # + (R32s ** 2) \
                                        #     * ((S['2s'][l, :] ** 2)[G1] \
                                        #         + S_sum_sq['2s'] + S_sum_sq_l['2s'])
                                ]
                            )
                            del S_sum_sq_l
                            weights_cts[col][l] *= ((var_l_numerator / var_l_denominator) * eps_sq[l])

                            # NOTE: take absolute value
                            XSwE_cts[l_index + 0] = \
                                np.abs(np.sum(C1[col] * weights_cts[col][l][0 * ni: 1 * ni]))
                            XSwE_cts[l_index + 1] = \
                                np.abs(np.sum(C1[col] * weights_cts[col][l][1 * ni: 2 * ni]))

                            weights_cts[col][l] = 0
                        del weights_cts[col]

                        try:
                            s_solver = cons_s_dict[col]
                            s_solver.solve(XSwXS_cts[col], -XSwE_cts[col], solver='quadprog')
                            if s_solver.res is None:
                                # If estimation fails, keep S_cts the same
                                if params['verbose'] in [2, 3]:
                                    print(f'Passing S_cts for column {col!r}: estimates are None')
                            else:
                                res_s = s_solver.res.reshape((nl, len(periods_update)))

                                if not cts_dict[col]['worker_type_interaction']:
                                    for period in periods_update:
                                        S_sum_sq[period] -= S_cts[col][period] ** 2

                                if cts_dict[col]['worker_type_interaction']:
                                    S_cts[col]['2s'] = np.sqrt(res_s[:, 0])
                                    S_cts[col]['3s'] = np.sqrt(res_s[:, 1])
                                else:
                                    S_cts[col]['2s'] = np.sqrt(res_s[0, 0])
                                    S_cts[col]['3s'] = np.sqrt(res_s[0, 1])
                                del res_s

                                if not cts_dict[col]['worker_type_interaction']:
                                    for period in periods_update:
                                        S_sum_sq[period] -= S_cts[col][period] ** 2

                        except ValueError as e:
                            # If constraints inconsistent, keep S_cts the same
                            if params['verbose'] in [2, 3]:
                                print(f'Passing S_cts for column {col!r}: {e}')
                        del XSwXS_cts[col], XSwE_cts[col]

                # print('A after:')
                # print(A)
                # print('S after:')
                # print(S)
                # print('A_cat after:')
                # print(A_cat)
                # print('S_cat after:')
                # print(S_cat)
                # print('A_cts after:')
                # print(A_cts)
                # print('S_cts after:')
                # print(S_cts)

        if len(cat_cols) > 0:
            ## Normalize ##
            # NOTE: normalize here because constraints don't normalize unless categorical controls are using constraints, and even when used, constraints don't always normalize to exactly 0
            A, A_cat = self._normalize(A, A_cat)

        # Store parameters
        self.A, self.A_cat, self.A_cts = A, A_cat, A_cts
        self.S, self.S_cat, self.S_cts = S, S_cat, S_cts
        self.R32s = R32s
        self.pk0 = pk0
        self.lik0, self.liks0 = lik0, liks0 # np.concatenate([self.liks0, liks0])

        # Update NNs
        if compute_NNs:
            NNs = sdata['g1'].value_counts(sort=False)
            NNs.sort_index(inplace=True)
            self.NNs = NNs.to_numpy()

    def fit_movers_cstr_uncstr(self, jdata, compute_NNm=True, blm_model=None, initialize_all=False):
        '''
        Run fit_movers(), first constrained, then using results as starting values, run unconstrained.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
            blm_model (BLMModel or None): already-estimated non-dynamic BLM model. Estimates from this model will be used as a baseline in half the starting values (the other half will be random). If None, all starting values will be random.
            initialize_all (bool): if True, initialize all parameters from blm_model (if blm_model is not None)
        '''
        ## First, simulate parameters but keep A fixed ##
        ## Second, use estimated parameters as starting point to run with A constrained to be linear ##
        ## Finally use estimated parameters as starting point to run without constraints ##
        # Save original parameters
        user_params = self.params.copy()
        ## Update parameters with blm_model ##
        if blm_model is not None:
            self.pk1 = copy.deepcopy(blm_model.pk1)
            self.pk0 = copy.deepcopy(blm_model.pk0)
            self.A['12'] = copy.deepcopy(blm_model.A1)
            self.A['43'] = copy.deepcopy(blm_model.A2)
            if initialize_all:
                self.A['2mb'] = copy.deepcopy(blm_model.A1)
                self.A['3mb'] = copy.deepcopy(blm_model.A2)
                self.A['2s'] = copy.deepcopy(blm_model.A1)
                self.A['3s'] = copy.deepcopy(blm_model.A2)
                self.A['2ma'][:] = 0
                self.A['3ma'][:] = 0
                self.R12 = 0
                self.R43 = 0
                self.R32m = 0
                self.R32s = 0
            ##### Loop 1 #####
            # First fix pk and A but update S
            self.params['update_a_movers'] = False
            self.params['update_s_movers'] = True
            self.params['update_pk1'] = False
            if self.params['verbose'] in [1, 2, 3]:
                print('Fitting movers with pk1 and A fixed')
            self.fit_movers(jdata, compute_NNm=False)
            self.params['update_pk1'] = True
        else:
            ##### Loop 1 #####
            # First fix A but update S and pk
            self.params['update_a_movers'] = False
            self.params['update_s_movers'] = True
            self.params['update_pk1'] = True
            if self.params['verbose'] in [1, 2, 3]:
                print('Fitting movers with A fixed')
            self.fit_movers(jdata, compute_NNm=False)
        ##### Loop 2 #####
        # Now update A with Linear Additive constraint
        self.params['update_a_movers'] = True
        if self.nl > 1:
            # Set constraints
            if user_params['cons_a_all'] is None:
                self.params['cons_a_all'] = cons.LinearAdditive(nt=len(self.periods_movers), dynamic=True)
            else:
                self.params['cons_a_all'] = to_list(user_params['cons_a_all']) + [cons.LinearAdditive(nt=len(self.periods_movers), dynamic=True)]
            if self.params['verbose'] in [1, 2, 3]:
                print('Fitting movers with Linear Additive constraint on A')
            self.fit_movers(jdata, compute_NNm=False)
        ##### Loop 3 #####
        # Now update A with Stationary Firm Type Variation constraint
        if self.nl > 1:
            # Set constraints
            if user_params['cons_a_all'] is None:
                self.params['cons_a_all'] = cons.StationaryFirmTypeVariation(nnt=range(1, 4), nt=len(self.periods_movers), dynamic=True)
            else:
                self.params['cons_a_all'] = to_list(user_params['cons_a_all']) + [cons.StationaryFirmTypeVariation(nnt=range(1, 4), nt=len(self.periods_movers), dynamic=True)]
            if self.params['verbose'] in [1, 2, 3]:
                print('Fitting movers with Stationary Firm Type Variation constraint on A')
            self.fit_movers(jdata, compute_NNm=False)
        ##### Loop 4 #####
        # Restore user constraints
        self.params['cons_a_all'] = user_params['cons_a_all']
        # Update d_X_diag_movers_A to be closer to 1
        self.params['d_X_diag_movers_A'] = 1 + (self.params['d_X_diag_movers_A'] - 1) / 2
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
        self.params['update_a_movers'] = True
        self.params['update_s_movers'] = False
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
        self.params['update_a_movers'] = False
        self.params['update_s_movers'] = True
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
        self.params['update_a_movers'] = False
        self.params['update_s_movers'] = False
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
                warnings.warn("Linear algebra error encountered when computing connectedness measure. This can likely be corrected by increasing the value of 'd_prior_movers' in tw.dynamic_blm_params().")
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
        A = self._sort_parameters(self.A, sort_firm_types=True)

        # Compute average log-earnings
        if period == 'first':
            A_all = A['12']
        elif period == 'second':
            A_all = A['43']
        elif period == 'all':
            # FIXME should the mean account for the log?
            A_all = (A['12'] + A['43']) / 2 # np.log((np.exp(self.A['12']) + np.exp(self.A['43'])) / 2)
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
        if subset == 'movers':
            if self.NNm is None:
                raise ValueError('The dynamic BLM estimation must be run on movers (and NNm must be computed) before plotting type proportions for movers.')
            A, pk1, NNm = self._sort_parameters(self.A, pk1=self.pk1, NNm=self.NNm, sort_firm_types=True)
        elif subset == 'stayers':
            if self.NNm is None:
                raise ValueError('The dynamic BLM estimation must be run on stayers (and NNs must be computed) before plotting type proportions for stayers.')
            A, pk0, NNs = self._sort_parameters(self.A, pk0=self.pk0, NNs=self.NNs, sort_firm_types=True)
        elif subset == 'all':
            if (self.NNm is None) or (self.NNs is None):
                raise ValueError('The dynamic BLM estimation must be run on both movers and stayers (and both NNm and NNs must be computed) before plotting type proportions for all.')
            A, pk1, pk0, NNm, NNs = self._sort_parameters(self.A, pk1=self.pk1, pk0=self.pk0, NNm=self.NNm, NNs=self.NNs, sort_firm_types=True)

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

class DynamicBLMEstimator:
    '''
    Class for estimating dynamic BLM using multiple sets of starting values.

    Arguments:
        params (ParamsDict): dictionary of parameters for dynamic BLM estimation. Run tw.dynamic_blm_params().describe_all() for descriptions of all valid parameters.
    '''

    def __init__(self, params):
        self.params = params
        # No initial model
        self.model = None
        # No likelihoods yet
        self.liks_high = None
        self.liks_low = None
        # No connectedness yet
        self.connectedness_high = None
        self.connectedness_low = None

    def _fit_model(self, jdata, iter, blm_model=None, rng=None):
        '''
        Generate model and run fit_movers_cstr_uncstr() given parameters.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            iter (int): iteration
            blm_model (BLMModel or None): already-estimated non-dynamic BLM model. Estimates from this model will be used as a baseline in half the starting values (the other half will be random). If None, all starting values will be random.
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        model = DynamicBLMModel(self.params, self.rho_0, rng)
        if iter % 2 == 0:
            model.fit_movers_cstr_uncstr(jdata, blm_model=blm_model, initialize_all=(iter == 0))
        else:
            model.fit_movers_cstr_uncstr(jdata, blm_model=None, initialize_all=False)
        return model

    def fit(self, jdata, sdata, n_init=20, n_best=5, blm_model=None, rho_0=(0.6, 0.6, 0.6), weights=None, diff=False, ncore=1, rng=None):
        '''
        Estimate dynamic BLM using multiple sets of starting values.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            n_init (int): number of starting values
            n_best (int): take the n_best estimates with the highest likelihoods, and then take the estimate with the highest connectedness
            blm_model (BLMModel or None): already-estimated non-dynamic BLM model. Estimates from this model will be used as a baseline in half the starting values (the other half will be random). If None, all starting values will be random.
            rho_0 (tuple): initial guess for rho
            weights (tuple or None): weights for rho; if None, all elements of rho have equal weight
            diff (bool): if True, estimate rho in differences rather than levels
            ncore (int): number of cores for multiprocessing
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        ## Estimate model ##
        # First, get starting values for rho
        self.rho_0 = rho_init(sdata, rho_0=rho_0, weights=weights, diff=diff)

        # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
        seeds = rng.bit_generator._seed_seq.spawn(n_init)
        if ncore > 1:
            # Multiprocessing
            with Pool(processes=ncore) as pool:
                sim_model_lst = list(tqdm(pool.imap(tw.util.f_star, [(self._fit_model, (jdata, i, blm_model, np.random.default_rng(seed))) for i, seed in enumerate(seeds)]), total=n_init))
                # sim_model_lst = pool.starmap(self._fit_model, tqdm([(jdata, i, blm_model, np.random.default_rng(seed)) for i, seed in enumerate(seeds)], total=n_init))
        else:
            # No multiprocessing
            sim_model_lst = list(tqdm(map(tw.util.f_star, [(self._fit_model, (jdata, i, blm_model, np.random.default_rng(seed))) for i, seed in enumerate(seeds)]), total=n_init))
            # sim_model_lst = itertools.starmap(self._fit_model, tqdm([(jdata, i, blm_model, np.random.default_rng(seed)) for i, seed in enumerate(seeds)], total=n_init))

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
        # Save paths of connectedness
        connectedness_all = []
        for i, model in enumerate(sorted_lik_models):
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

class DynamicBLMBootstrap:
    '''
    Class for estimating dynamic BLM using bootstrapping.

    Arguments:
        params (ParamsDict): dictionary of parameters for dynamic BLM estimation. Run tw.dynamic_blm_params().describe_all() for descriptions of all valid parameters.
    '''

    def __init__(self, params):
        self.params = params
        # No initial models
        self.models = None

    def fit(self, jdata, sdata, static_blm_model=None, dynamic_blm_model=None, n_samples=5, n_init_estimator=20, n_best=5, frac_movers=0.1, frac_stayers=0.1, method='parametric', cluster_params=None, reallocate=False, reallocate_jointly=True, reallocate_period='first', ncore=1, verbose=True, rng=None):
        '''
        Estimate bootstrap.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            static_blm_model (BLMModel or None): already-estimated non-dynamic BLM model. Estimates from this model will be used as a baseline in half the starting values (the other half will be random). If None, all starting values will be random.
            dynamic_blm_model (DynamicBLMModel or None): dynamic BLM model estimated using true data; if None, estimate model inside the method. For use with parametric bootstrap.
            n_samples (int): number of bootstrap samples to estimate
            n_init_estimator (int): number of starting guesses to estimate for each bootstrap sample
            n_best (int): take the n_best estimates with the highest likelihoods, and then take the estimate with the highest connectedness, for each bootstrap sample
            frac_movers (float): fraction of movers to draw (with replacement) for each bootstrap sample. For use with standard bootstrap.
            frac_stayers (float): fraction of stayers to draw (with replacement) for each bootstrap sample. For use with standard bootstrap.
            method (str): if 'parametric', estimate dynamic BLM model on full data, simulate worker types and wages using estimated parameters, estimate dynamic BLM model on each set of simulated data, and construct bootstrapped errors; if 'standard', estimate standard bootstrap by sampling from original data, estimating dynamic BLM model on each sample, and constructing bootstrapped errors
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

        # Update clustering parameters
        cluster_params = cluster_params.copy()
        cluster_params['is_sorted'] = True
        cluster_params['copy'] = False

        if method == 'parametric':
            # Copy original wages and firm types
            yj = jdata.loc[:, ['y1', 'y2', 'y3', 'y4']].to_numpy().copy()
            ys = sdata.loc[:, ['y1', 'y2', 'y3', 'y4']].to_numpy().copy()
            gj = jdata.loc[:, ['g1', 'g4']].to_numpy().copy()
            gs = sdata.loc[:, 'g1'].to_numpy().copy()

            if dynamic_blm_model is None:
                # Run initial BLM estimator
                blm_fit_init = DynamicBLMEstimator(self.params)
                blm_fit_init.fit(jdata=jdata, sdata=sdata, n_init=n_init_estimator, n_best=n_best, blm_model=static_blm_model, ncore=ncore, rng=rng)
                dynamic_blm_model = blm_fit_init.model

            # Run parametric bootstrap
            models = []
            for _ in trange(n_samples):
                # Simulate worker types then draw wages
                yj_i, ys_i = _simulate_types_wages(jdata=jdata, sdata=sdata, gj=gj, gs=gs, blm_model=dynamic_blm_model, reallocate=reallocate, reallocate_jointly=reallocate_jointly, reallocate_period=reallocate_period, rng=rng)[:2]
                with bpd.util.ChainedAssignment():
                    jdata.loc[:, 'y1'], jdata.loc[:, 'y2'], jdata.loc[:, 'y3'], jdata.loc[:, 'y4'] = (yj_i[0], yj_i[1], yj_i[2], yj_i[3])
                    sdata.loc[:, 'y1'], sdata.loc[:, 'y2'], sdata.loc[:, 'y3'], sdata.loc[:, 'y4'] = (ys_i[0], ys_i[1], ys_i[2], ys_i[3])
                # Cluster
                bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
                # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
                bdf._set_attributes(jdata)
                bdf = bdf.to_long(is_sorted=True, copy=False)
                # Cluster
                bdf = bdf.cluster(cluster_params, rng=rng)
                clusters_dict = bdf.loc[:, ['j', 'g']].groupby('j', sort=False)['g'].first().to_dict()
                del bdf
                with bpd.util.ChainedAssignment():
                    # Update clusters in jdata and sdata
                    jdata.loc[:, 'g1'] = jdata.loc[:, 'j1'].map(clusters_dict)
                    jdata.loc[:, 'g2'] = jdata.loc[:, 'g1']
                    jdata.loc[:, 'g4'] = jdata.loc[:, 'j4'].map(clusters_dict)
                    jdata.loc[:, 'g3'] = jdata.loc[:, 'g4']
                    sdata.loc[:, 'g1'] = sdata.loc[:, 'j1'].map(clusters_dict)
                    sdata.loc[:, 'g2'] = sdata.loc[:, 'g1']
                    sdata.loc[:, 'g3'] = sdata.loc[:, 'g1']
                    sdata.loc[:, 'g4'] = sdata.loc[:, 'g1']
                # Run dynamic BLM estimator
                blm_fit_i = DynamicBLMEstimator(self.params)
                blm_fit_i.fit(jdata=jdata, sdata=sdata, n_init=n_init_estimator, n_best=n_best, blm_model=static_blm_model, ncore=ncore, rng=rng)
                models.append(blm_fit_i.model)
                del blm_fit_i

            with bpd.util.ChainedAssignment():
                # Re-assign original wages and firm types
                jdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = yj
                sdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = ys
                jdata.loc[:, ['g1', 'g4']], jdata.loc[:, ['g2', 'g3']] = (gj, gj)
                sdata.loc[:, 'g1'], sdata.loc[:, 'g2'], sdata.loc[:, 'g3'], sdata.loc[:, 'g4'] = (gs, gs, gs, gs)
        elif method == 'standard':
            models = []
            for _ in trange(n_samples):
                jdata_i = jdata.sample(frac=frac_movers, replace=True, random_state=rng)
                sdata_i = sdata.sample(frac=frac_stayers, replace=True, random_state=rng)
                # Cluster
                bdf = bpd.BipartiteDataFrame(pd.concat([jdata_i, sdata_i], axis=0, copy=True))
                # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
                bdf._set_attributes(jdata)
                # Clean and cluster
                bdf = bdf.to_long(is_sorted=True, copy=False).clean(bpd.clean_params({'is_sorted': True, 'copy': False, 'verbose': verbose})).cluster(cluster_params, rng=rng)
                clusters_dict = bdf.loc[:, ['j', 'g']].groupby('j', sort=False)['g'].first().to_dict()
                del bdf
                # Update clusters in jdata_i and sdata_i
                jdata_i.loc[:, 'g1'] = jdata_i.loc[:, 'j1'].map(clusters_dict)
                jdata_i.loc[:, 'g2'] = jdata_i.loc[:, 'g1']
                jdata_i.loc[:, 'g4'] = jdata_i.loc[:, 'j4'].map(clusters_dict)
                jdata_i.loc[:, 'g3'] = jdata_i.loc[:, 'g4']
                sdata_i.loc[:, 'g1'] = sdata_i.loc[:, 'j1'].map(clusters_dict)
                sdata_i.loc[:, 'g2'] = sdata_i.loc[:, 'g1']
                sdata_i.loc[:, 'g3'] = sdata_i.loc[:, 'g1']
                sdata_i.loc[:, 'g4'] = sdata_i.loc[:, 'g1']
                # Run dynamic BLM estimator
                blm_fit_i = DynamicBLMEstimator(self.params)
                blm_fit_i.fit(jdata=jdata_i, sdata=sdata_i, n_init=n_init_estimator, n_best=n_best, blm_model=static_blm_model, ncore=ncore, rng=rng)
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
                A = model._sort_parameters(model.A, sort_firm_types=True)
                # Extract average log-earnings for each model
                if period == 'first':
                    A_all[i, :, :] = A['12']
                elif period == 'second':
                    A_all[i, :, :] = A['43']
                elif period == 'all':
                    # FIXME should the mean account for the log?
                    A_all[i, :, :] = (A['12'] + A['43']) / 2 # np.log((np.exp(A['12']) + np.exp(A['43'])) / 2)
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
                A, pk1, pk0, NNm, NNs = model._sort_parameters(model.A, pk1=model.pk1, pk0=model.pk0, NNm=model.NNm, NNs=model.NNs, sort_firm_types=True)

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

class DynamicBLMVarianceDecomposition:
    '''
    Class for estimating dynamic BLM variance decomposition using bootstrapping. Results are stored in class attribute .res, which gives a dictionary where the key 'var_decomp' gives the results for the variance decomposition, and the key 'var_decomp_comp' optionally gives the results for the variance decomposition with complementarities.

    Arguments:
        params (ParamsDict): dictionary of parameters for dynamic BLM estimation. Run tw.dynamic_blm_params().describe_all() for descriptions of all valid parameters.
    '''

    def __init__(self, params):
        self.params = params
        # No initial results
        self.res = None

    def fit(self, jdata, sdata, blm_model=None, n_samples=5, n_init_estimator=20, n_best=5, reallocate=False, reallocate_jointly=True, reallocate_period='first', Q_var=None, Q_cov=None, complementarities=True, firm_clusters_as_ids=True, worker_types_as_ids=True, ncore=1, rng=None):
        '''
        Estimate variance decomposition.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            blm_model (DynamicBLMModel or None): already-estimated dynamic BLM model; if None, estimate model inside the method
            n_samples (int): number of bootstrap samples to estimate
            n_init_estimator (int): number of starting guesses to estimate for each bootstrap sample
            n_best (int): take the n_best estimates with the highest likelihoods, and then take the estimate with the highest connectedness, for each bootstrap sample
            reallocate (bool): if True, draw worker type proportions independently of firm type; if False, uses worker type proportions that are conditional on firm type
            reallocate_jointly (bool): if True, worker type proportions take the average over movers and stayers (i.e. all workers use the same type proportions); if False, consider movers and stayers separately
            reallocate_period (str): if 'first', compute type proportions based on first period parameters; if 'second', compute type proportions based on second period parameters; if 'all', compute type proportions based on average over first and second period parameters
            Q_var (list of Q variances): list of Q matrices to use when estimating variance term; None is equivalent to tw.Q.VarPsi() without controls, or tw.Q.VarCovariate('psi') with controls
            Q_cov (list of Q covariances): list of Q matrices to use when estimating covariance term; None is equivalent to tw.Q.CovPsiAlpha() without controls, or tw.Q.CovCovariate('psi', 'alpha') with controls
            complementarities (bool): if True, estimate R^2 of regression with complementarities (by adding in all worker-firm interactions). Only allowed when firm_clusters_as_ids=True and worker_types_as_ids=True.
            firm_clusters_as_ids (bool): if True, regress on firm clusters; if False, regress on firm ids
            worker_types_as_ids (bool): if True, regress on true, simulated worker types; if False, regress on worker ids
            ncore (int): number of cores for multiprocessing
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if complementarities and ((not firm_clusters_as_ids) or (not worker_types_as_ids)):
            raise ValueError('If `complementarities=True`, then must also set `firm_clusters_as_ids=True` and `worker_types_as_ids=True`.')

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
        fe_params['weighted'] = False
        fe_params['ho'] = False
        if Q_var is not None:
            fe_params['Q_var'] = Q_var
        if Q_cov is not None:
            fe_params['Q_cov'] = Q_cov
        if complementarities:
            fe_params_comp = fe_params.copy()
            fe_params_comp['Q_var'] = []
            fe_params_comp['Q_cov'] = []

        # Copy original wages, firm types, and optionally ids
        yj = jdata.loc[:, ['y1', 'y2', 'y3', 'y4']].to_numpy().copy()
        ys = sdata.loc[:, ['y1', 'y2', 'y3', 'y4']].to_numpy().copy()
        gj = jdata.loc[:, ['g1', 'g4']].to_numpy()
        gs = sdata.loc[:, 'g1'].to_numpy()
        if firm_clusters_as_ids:
            jj = jdata.loc[:, ['j1', 'j4']].to_numpy().copy()
            js = sdata.loc[:, 'j1'].to_numpy().copy()
            with bpd.util.ChainedAssignment():
                jdata.loc[:, ['j1', 'j4']] = gj
                sdata.loc[:, 'j1'], sdata.loc[:, 'j4'] = (gs, gs)
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

        if blm_model is None:
            # Run initial BLM estimator
            blm_fit_init = DynamicBLMEstimator(self.params)
            blm_fit_init.fit(jdata=jdata, sdata=sdata, n_init=n_init_estimator, n_best=n_best, ncore=ncore, rng=rng)
            blm_model = blm_fit_init.model

        # Run bootstrap
        res_lst = []
        if complementarities:
            res_lst_comp = []
            nl = blm_model.nl
        for i in trange(n_samples):
            # Simulate worker types then draw wages
            yj_i, ys_i, Lm_i, Ls_i = _simulate_types_wages(jdata=jdata, sdata=sdata, gj=gj, gs=gs, blm_model=blm_model, reallocate=reallocate, reallocate_jointly=reallocate_jointly, reallocate_period=reallocate_period, rng=rng)
            with bpd.util.ChainedAssignment():
                jdata.loc[:, 'y1'], jdata.loc[:, 'y2'], jdata.loc[:, 'y3'], jdata.loc[:, 'y4'] = (yj_i[0], yj_i[1], yj_i[2], yj_i[3])
                sdata.loc[:, 'y1'], sdata.loc[:, 'y2'], sdata.loc[:, 'y3'], sdata.loc[:, 'y4'] = (ys_i[0], ys_i[1], ys_i[2], ys_i[3])
                if worker_types_as_ids:
                    jdata.loc[:, 'i'] = Lm_i
                    sdata.loc[:, 'i'] = Ls_i
            # Convert to BipartitePandas DataFrame
            bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
            # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
            bdf._set_attributes(jdata)
            # If simulating worker types, data is not sorted
            bdf = bdf.to_long(is_sorted=(not worker_types_as_ids), copy=False)
            # Estimate OLS
            if no_controls:
                fe_estimator = tw.FEEstimator(bdf, fe_params)
            else:
                fe_estimator = tw.FEControlEstimator(bdf, fe_params)
            fe_estimator.fit()
            res_lst.append(fe_estimator.summary)
            if complementarities:
                # Estimate OLS with complementarities
                bdf.loc[:, 'i'] = pd.factorize(bdf.loc[:, 'i'].to_numpy() + nl * bdf.loc[:, 'j'].to_numpy())[0]
                if no_controls:
                    fe_estimator = tw.FEEstimator(bdf, fe_params_comp)
                else:
                    fe_estimator = tw.FEControlEstimator(bdf, fe_params_comp)
                fe_estimator.fit()
                res_lst_comp.append(fe_estimator.summary)

        with bpd.util.ChainedAssignment():
            # Restore original wages and optionally ids
            jdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = yj
            sdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = ys
            if firm_clusters_as_ids:
                jdata.loc[:, ['j1', 'j4']] = jj
                sdata.loc[:, 'j4'], sdata.loc[:, 'j4'] = (js, js)
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

class DynamicBLMReallocation:
    '''
    Class for estimating dynamic BLM reallocation exercise using bootstrapping. Results are stored in class attribute .res, which gives a dictionary with the following structure: baseline results are stored in key 'baseline'. Reallocation results are stored in key 'reallocation'. Within each sub-dictionary, primary outcome results are stored in the key 'outcome', categorical results are stored in the key 'cat', and continuous results are stored in the key 'cts'.

    Arguments:
        params (ParamsDict): dictionary of parameters for dynamic BLM estimation. Run tw.dynamic_blm_params().describe_all() for descriptions of all valid parameters.
    '''

    def __init__(self, params):
        self.params = params
        # No initial results
        self.res = None

    def fit(self, jdata, sdata, quantiles=None, blm_model=None, n_samples=5, n_init_estimator=20, n_best=5, reallocate_jointly=True, reallocate_period='first', categorical_sort_cols=None, continuous_sort_cols=None, ncore=1, rng=None):
        '''
        Estimate variance decomposition.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            quantiles (NumPy Array or None): income quantiles to compute; if None, computes percentiles from 1-100 (specifically, np.arange(101) / 100)
            blm_model (DynamicBLMModel or None): already-estimated dynamic BLM model; if None, estimate model inside the method
            n_samples (int): number of bootstrap samples to estimate
            n_init_estimator (int): number of starting guesses to estimate for each bootstrap sample
            n_best (int): take the n_best estimates with the highest likelihoods, and then take the estimate with the highest connectedness, for each bootstrap sample
            reallocate_jointly (bool): if True, worker type proportions take the average over movers and stayers (i.e. all workers use the same type proportions); if False, consider movers and stayers separately
            reallocate_period (str): if 'first', compute type proportions based on first period parameters; if 'second', compute type proportions based on second period parameters; if 'all', compute type proportions based on average over first and second period parameters
            categorical_sort_cols (dict or None): in addition to standard quantiles results, return average income grouped by the alternative column(s) given (which are represented by the dictionary {column: number of quantiles to compute}). For categorical variables, use each group as a bin and take the average income within that bin. None is equivalent to {}.
            continuous_sort_cols (dict or None): in addition to standard quantiles results, return average income grouped by the alternative column(s) given (which are represented by the dictionary {column: list of quantiles to compute}). For continuous variables, create bins based on the list of quantiles given in the dictionary. The list of quantiles must start at 0 and end at 1. None is equivalent to {}.
            ncore (int): number of cores for multiprocessing
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if quantiles is None:
            quantiles = np.arange(101) / 100
        if categorical_sort_cols is None:
            categorical_sort_cols = {}
        if continuous_sort_cols is None:
            continuous_sort_cols = {}
        if rng is None:
            rng = np.random.default_rng(None)

        # Make sure continuous quantiles start at 0 and end at 1
        for col_cts, quantiles_cts in continuous_sort_cols.items():
            if quantiles_cts[0] != 0:
                raise ValueError(f'Lowest quantile associated with continuous column {col_cts} must be 0.')
            elif quantiles_cts[-1] != 1:
                raise ValueError(f'Highest quantile associated with continuous column {col_cts} must be 1.')

        # Copy original wages and firm types
        yj = jdata.loc[:, ['y1', 'y2', 'y3', 'y4']].to_numpy().copy()
        ys = sdata.loc[:, ['y1', 'y2', 'y3', 'y4']].to_numpy().copy()
        gj = jdata.loc[:, ['g1', 'g4']].to_numpy()
        gs = sdata.loc[:, 'g1'].to_numpy()
        tj = False
        ts = False
        if not jdata._col_included('t'):
            jdata = jdata.construct_artificial_time(is_sorted=True, copy=False)
            tj = True
        if not sdata._col_included('t'):
            sdata = sdata.construct_artificial_time(is_sorted=True, copy=False)
            ts = True

        if blm_model is None:
            # Run initial BLM estimator
            blm_fit_init = DynamicBLMEstimator(self.params)
            blm_fit_init.fit(jdata=jdata, sdata=sdata, n_init=n_init_estimator, n_best=n_best, ncore=ncore, rng=rng)
            blm_model = blm_fit_init.model

        ## Baseline ##
        res_cat_baseline = {}
        res_cts_baseline = {}

        # Convert to BipartitePandas DataFrame
        bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
        # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
        bdf._set_attributes(jdata)
        bdf = bdf.to_long(is_sorted=True, copy=False)
        # Compute quantiles (no weights for dynamic BLM)
        y = bdf.loc[:, 'y'].to_numpy()
        res_baseline = weighted_quantile(values=y, quantiles=quantiles, sample_weight=None)
        for col_cat in categorical_sort_cols.keys():
            ## Categorical sorting variables ##
            col = bdf.loc[:, col_cat].to_numpy()
            # Use categories as bins
            res_cat_baseline[col_cat] =\
                np.bincount(col, weights=y) / np.bincount(col, weights=None)
        for col_cts, quantiles_cts in continuous_sort_cols.items():
            ## Continuous sorting variables ##
            col = bdf.loc[:, col_cts].to_numpy()
            # Create bins based on quantiles
            col_quantiles = weighted_quantile(values=col, quantiles=quantiles_cts, sample_weight=None)
            quantile_groups = pd.cut(col, col_quantiles, include_lowest=True).codes
            res_cts_baseline[col_cts] =\
                np.bincount(quantile_groups, weights=y) / np.bincount(quantile_groups, weights=None)

        # Run bootstrap
        res = np.zeros([n_samples, len(quantiles)])
        res_cat = {col: np.zeros([n_samples, n_quantiles]) for col, n_quantiles in categorical_sort_cols.items()}
        res_cts = {col: np.zeros([n_samples, len(quantiles) - 1]) for col, quantiles in continuous_sort_cols.items()}
        for i in trange(n_samples):
            # Simulate worker types then draw wages
            yj_i, ys_i = _simulate_types_wages(jdata=jdata, sdata=sdata, gj=gj, gs=gs, blm_model=blm_model, reallocate=True, reallocate_jointly=reallocate_jointly, reallocate_period=reallocate_period, rng=rng)[: 2]
            with bpd.util.ChainedAssignment():
                jdata.loc[:, 'y1'], jdata.loc[:, 'y2'], jdata.loc[:, 'y3'], jdata.loc[:, 'y4'] = (yj_i[0], yj_i[1], yj_i[2], yj_i[3])
                sdata.loc[:, 'y1'], sdata.loc[:, 'y2'], sdata.loc[:, 'y3'], sdata.loc[:, 'y4'] = (ys_i[0], ys_i[1], ys_i[2], ys_i[3])
            # Convert to BipartitePandas DataFrame
            bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
            # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
            bdf._set_attributes(jdata)
            bdf = bdf.to_long(is_sorted=True, copy=False)
            # Compute quantiles (no weights for dynamic BLM)
            y = bdf.loc[:, 'y'].to_numpy()
            res[i, :] = weighted_quantile(values=y, quantiles=quantiles, sample_weight=None)
            for col_cat in categorical_sort_cols.keys():
                ## Categorical sorting variables ##
                col = bdf.loc[:, col_cat].to_numpy()
                # Use categories as bins
                res_cat[col_cat][i, :] =\
                    np.bincount(col, weights=y) / np.bincount(col, weights=None)
            for col_cts, quantiles_cts in continuous_sort_cols.items():
                ## Continuous sorting variables ##
                col = bdf.loc[:, col_cts].to_numpy()
                # Create bins based on quantiles
                col_quantiles = weighted_quantile(values=col, quantiles=quantiles_cts, sample_weight=None)
                quantile_groups = pd.cut(col, col_quantiles, include_lowest=True).codes
                res_cts[col_cts][i, :] =\
                    np.bincount(quantile_groups, weights=y) / np.bincount(quantile_groups, weights=None)

        with bpd.util.ChainedAssignment():
            # Restore original wages and optionally ids
            jdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = yj
            sdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = ys

        # Drop time column
        if tj:
            jdata = jdata.drop('t', axis=1, inplace=True, allow_optional=True)
        if ts:
            sdata = sdata.drop('t', axis=1, inplace=True, allow_optional=True)

        # Store results
        self.res = {
            'baseline': {
                'outcome': res_baseline,
                'cat': res_cat_baseline,
                'cts': res_cts_baseline
                },
            'reallocation': {
                'outcome': res,
                'cat': res_cat,
                'cts': res_cts
                }
        }

class DynamicBLMTransitions:
    '''
    Class for estimating dynamic BLM transition probability exercise using bootstrapping. Results are stored in class attribute .res, which gives a 4-D NumPy Array where the first dimension gives each particular simulation; the second dimension gives the subset of data considered (index 0 gives the full data; index 1 gives the first conditional decile of earnings; and index 2 gives the tenth conditional decile of earnings); the third dimension gives the starting group of clusters being considered; and the fourth dimension gives the destination group of clusters being considered (where the first index of the fourth dimension considers all destinations).

    Arguments:
        params (ParamsDict): dictionary of parameters for dynamic BLM estimation. Run tw.dynamic_blm_params().describe_all() for descriptions of all valid parameters.
    '''

    def __init__(self, params):
        self.params = params
        # No initial results
        self.res = None

    def fit(self, jdata, sdata, cluster_groups=None, blm_model=None, n_samples=5, n_init_estimator=20, n_best=5, cluster_params=None, ncore=1, rng=None):
        '''
        Estimate transition probabilities.

        Arguments:
            jdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for movers
            sdata (BipartitePandas DataFrame): event study or collapsed event study format labor data for stayers
            cluster_groups (list of lists or None): how to group firm clusters, where each element of the primary list gives a list of firm clusters in the corresponding group; if None, tries to divide firms in 3 evenly sized groups
            blm_model (DynamicBLMModel or None): already-estimated dynamic BLM model; if None, estimate model inside the method
            n_samples (int): number of bootstrap samples to estimate
            n_init_estimator (int): number of starting guesses to estimate for each bootstrap sample
            n_best (int): take the n_best estimates with the highest likelihoods, and then take the estimate with the highest connectedness, for each bootstrap sample
            cluster_params (ParamsDict or None): dictionary of parameters for clustering firms. Run bpd.cluster_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.cluster_params().
            ncore (int): number of cores for multiprocessing
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        nk = jdata.n_clusters()
        if cluster_groups is None:
            # Evenly divide into 3 groups
            nk3 = nk // 3
            g1 = np.arange(nk3)
            g2 = np.arange(nk3, nk - nk3)
            g3 = np.arange(nk - nk3, nk)
            cluster_groups = [g1, g2, g3]
        if cluster_params is None:
            grouping = bpd.grouping.KMeans(n_clusters=nk)
            cluster_params = bpd.cluster_params({'grouping': grouping})
        if rng is None:
            rng = np.random.default_rng(None)

        # Update clustering parameters
        cluster_params = cluster_params.copy()
        cluster_params['is_sorted'] = True
        cluster_params['copy'] = False

        # Copy original wages, firm types, and ids
        yj = jdata.loc[:, ['y1', 'y2', 'y3', 'y4']].to_numpy().copy()
        ys = sdata.loc[:, ['y1', 'y2', 'y3', 'y4']].to_numpy().copy()
        gj = jdata.loc[:, ['g1', 'g4']].to_numpy()
        gs = sdata.loc[:, 'g1'].to_numpy()
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

        if blm_model is None:
            # Run initial BLM estimator
            blm_fit_init = DynamicBLMEstimator(self.params)
            blm_fit_init.fit(jdata=jdata, sdata=sdata, n_init=n_init_estimator, n_best=n_best, ncore=ncore, rng=rng)
            blm_model = blm_fit_init.model

        # Run bootstrap
        res = np.zeros([n_samples, 3, len(cluster_groups), len(cluster_groups) + 1])
        for i in trange(n_samples):
            # Simulate worker types then draw wages
            yj_i, ys_i, Lm_i, Ls_i = _simulate_types_wages(jdata=jdata, sdata=sdata, gj=gj, gs=gs, blm_model=blm_model, reallocate=False, rng=rng)
            with bpd.util.ChainedAssignment():
                jdata.loc[:, 'y1'], jdata.loc[:, 'y2'], jdata.loc[:, 'y3'], jdata.loc[:, 'y4'] = (yj_i[0], yj_i[1], yj_i[2], yj_i[3])
                sdata.loc[:, 'y1'], sdata.loc[:, 'y2'], sdata.loc[:, 'y3'], sdata.loc[:, 'y4'] = (ys_i[0], ys_i[1], ys_i[2], ys_i[3])
                jdata.loc[:, 'i'] = Lm_i
                sdata.loc[:, 'i'] = Ls_i

            # Convert to BipartitePandas DataFrame
            bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
            # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
            bdf._set_attributes(jdata)

            # Replace i to ensure unstacking works properly
            i_orig = bdf.loc[:, 'i'].to_numpy().copy()
            with bpd.util.ChainedAssignment():
                bdf.loc[:, 'i'] = np.arange(len(bdf))
            # Cluster
            bdf = bdf.cluster(cluster_params, rng=rng)
            with bpd.util.ChainedAssignment():
                # Restore i
                bdf.loc[:, 'i'] = i_orig

            # Compute conditional earnings deciles
            first_decile = np.zeros(len(bdf), dtype=int)
            tenth_decile = np.zeros(len(bdf), dtype=int)
            for l in range(blm_model.nl):
                for k in range(blm_model.nk):
                    # Earnings deciles conditional on worker type and firm destination class
                    bdf_lk = bdf.loc[(bdf.loc[:, 'i'].to_numpy() == l) & (bdf.loc[:, 'g2'].to_numpy() == k), 'y2']
                    if len(bdf_lk) > 0:
                        # If any observations meet the criteria
                        deciles = np.percentile(bdf_lk.to_numpy(), [10, 90])
                        first_decile[bdf_lk.index] = (bdf_lk.to_numpy() <= deciles[0])
                        tenth_decile[bdf_lk.index] = (bdf_lk.to_numpy() >= deciles[1])
            first_decile = first_decile.astype(bool)
            tenth_decile = tenth_decile.astype(bool)

            ## Full data ##
            for j, cg2 in enumerate(cluster_groups):
                bdf_cg2 = bdf.loc[bdf.loc[:, 'g2'].isin(cg2), :]
                # All transitions
                res[i, 0, j, 0] = np.sum(bdf_cg2.loc[:, 'm'].to_numpy() > 0) / len(bdf_cg2)
                for k, cg3 in enumerate(cluster_groups):
                    # Transitions to specific cluster groups
                    bdf_cg3 = bdf_cg2.loc[bdf_cg2.loc[:, 'g3'].isin(cg3), :]
                    res[i, 0, j, k + 1] = np.sum(bdf_cg3.loc[:, 'm'].to_numpy() > 0) / len(bdf_cg2)

            ## First conditional decile of earnings ##
            bdf_1 = bdf.loc[first_decile, :]
            for j, cg2 in enumerate(cluster_groups):
                bdf_cg2 = bdf_1.loc[bdf_1.loc[:, 'g2'].isin(cg2), :]
                # All transitions
                res[i, 1, j, 0] = np.sum(bdf_cg2.loc[:, 'm'].to_numpy() > 0) / len(bdf_cg2)
                for k, cg3 in enumerate(cluster_groups):
                    # Transitions to specific cluster groups
                    bdf_cg3 = bdf_cg2.loc[bdf_cg2.loc[:, 'g3'].isin(cg3), :]
                    res[i, 1, j, k + 1] = np.sum(bdf_cg3.loc[:, 'm'].to_numpy() > 0) / len(bdf_cg2)

            ## Tenth conditional decile of earnings ##
            bdf_10 = bdf.loc[first_decile, :]
            for j, cg2 in enumerate(cluster_groups):
                bdf_cg2 = bdf_10.loc[bdf_10.loc[:, 'g2'].isin(cg2), :]
                # All transitions
                res[i, 2, j, 0] = np.sum(bdf_cg2.loc[:, 'm'].to_numpy() > 0) / len(bdf_cg2)
                for k, cg3 in enumerate(cluster_groups):
                    # Transitions to specific cluster groups
                    bdf_cg3 = bdf_cg2.loc[bdf_cg2.loc[:, 'g3'].isin(cg3), :]
                    res[i, 2, j, k + 1] = np.sum(bdf_cg3.loc[:, 'm'].to_numpy() > 0) / len(bdf_cg2)

        with bpd.util.ChainedAssignment():
            # Restore original wages and ids
            jdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = yj
            sdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = ys
            jdata.loc[:, 'i'] = ij
            sdata.loc[:, 'i'] = is_

        # Drop time column
        if tj:
            jdata = jdata.drop('t', axis=1, inplace=True, allow_optional=True)
        if ts:
            sdata = sdata.drop('t', axis=1, inplace=True, allow_optional=True)

        # Store results
        self.res = res
