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
import plotly.graph_objects as go
from paramsdict import ParamsDict, ParamsDictBase
from paramsdict.util import col_type
import bipartitepandas as bpd
from bipartitepandas.util import to_list, HiddenPrints # , _is_subtype
import pytwoway as tw
from pytwoway import constraints as cons
from pytwoway.util import weighted_mean, weighted_quantile, DxSP, DxM, diag_of_sp_prod, jitter_scatter, logsumexp, lognormpdf, fast_lognormpdf

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
            (default=None) Prior for pk1 (probability of being at each combination of firm types for movers). In particular, pk1 now generates as a convex combination of the prior plus a Dirichlet random variable. Must have shape (nk * nk, nl). None puts all weight on the Dirichlet variable.
        ''', 'min > 0'),
    'pk0_prior': (None, 'array_of_type_constrained_none', (('float', 'int'), _min_gt0),
        '''
            (default=None) Prior for pk0 (probability of being at each firm type for stayers). Must have shape (nk, nl). None is equivalent to np.ones((nk, nl)) / nl.
        ''', 'min > 0'),
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
    'update_rho_period_movers': (1, 'type_constrained', (int, _gteq1),
        '''
            (default=1) Number of iterations between updating rho for movers. Higher values may lead to faster convergence of the EM algorithm.
        ''', '>= 1'),
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
            (default=1 + 1e-10) Account for numerical rounding causing X'X (for means) to not be positive definite when computing A by adding (d_X_diag_movers_A - 1) to the diagonal of X'X.
        ''', '>= 1'),
    'd_X_diag_movers_S': (1 + 1e-10, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1 + 1e-10) Account for numerical rounding causing X'X (for variances) to not be positive definite when computing S by adding (d_X_diag_movers_S - 1) to the diagonal of X'X.
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
    'update_rho_period_stayers': (1, 'type_constrained', (int, _gteq1),
        '''
            (default=1) Number of iterations between updating rho for stayers. Higher values may lead to faster convergence of the EM algorithm.
        ''', '>= 1'),
    'd_prior_stayers': (1 + 1e-7, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1 + 1e-7) Account for probabilities being too small by adding (d_prior - 1) to pk0.
        ''', '>= 1'),
    'd_X_diag_stayers_A': (1 + 1e-10, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1 + 1e-10) Account for numerical rounding causing X'X (for means) to not be positive definite by adding (d_X_diag_stayers_A - 1) to the diagonal of X'X.
        ''', '>= 1'),
    'd_X_diag_stayers_S': (1 + 1e-10, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1 + 1e-10) Account for numerical rounding causing X'X (for variances) to not be positive definite by adding (d_X_diag_stayers_S - 1) to the diagonal of X'X.
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

def double_bincount(groups, n_groups, weights=None):
    '''
    Perform groupby-sum on 2 groups with given weights and return the corresponding matrix representation.

    Arguments:
        groups (NumPy Array): group1 + n_groups * group2
        n_groups (int): number of groups
        weights (NumPy Array or None): weights; None is unweighted

    Returns:
        (NumPy Array): groupby-sum matrix representation
    '''
    return np.bincount(groups, weights).reshape((n_groups, n_groups)).T

def _var_stayers(sdata, rho_1, rho_4, rho_t, weights=None, diff=False):
    '''
    Compute var(alpha | g1, g2) and var(epsilon | g1) using stayers.

    Arguments:
        sdata (BipartitePandas DataFrame): extended event study format labor data for stayers
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
        sdata (BipartitePandas DataFrame): extended event study format labor data for stayers
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

def _optimal_reallocation(model, jdata, sdata, gj, gs, Lm, Ls, method='max', reallocation_scaling_col=None, rng=None):
    '''
    Reallocate workers to firms in order to maximize total expected output.

    Arguments:
        model (DynamicBLMModel): dynamic BLM model with estimated parameters
        jdata (BipartitePandas DataFrame): extended event study format labor data for movers
        sdata (BipartitePandas DataFrame): extended event study format labor data for stayers
        gj (NumPy Array or None): firm classes for movers in the first and last periods
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
    if reallocation_scaling_col is None:
        Y = (
            model.A['12'] + model.R12 * (model.A['2s'] - model.A['2ma']) \
            + model.A['2s'] \
            + model.A['3s'] \
            + model.A['43'] + model.R43 * (model.A['3s'] - model.A['3ma'])
        )
    else:
        # Multiply Y by scaling factor
        scaling_cols = to_list(sdata.col_reference_dict[reallocation_scaling_col])
        scaling_col_1 = scaling_cols[0]
        if len(scaling_cols) == 1:
            scaling_col_2 = scaling_col_1
            scaling_col_3 = scaling_col_1
            scaling_col_4 = scaling_col_1
        elif len(scaling_cols) == 4:
            scaling_col_2 = scaling_cols[1]
            scaling_col_3 = scaling_cols[2]
            scaling_col_4 = scaling_cols[3]
        scaling_col_s_1 = sdata.loc[:, scaling_col_1].to_numpy()
        scaling_col_s_2 = sdata.loc[:, scaling_col_2].to_numpy()
        scaling_col_s_3 = sdata.loc[:, scaling_col_3].to_numpy()
        scaling_col_s_4 = sdata.loc[:, scaling_col_4].to_numpy()
        scaling_col_j_1 = jdata.loc[:, scaling_col_1].to_numpy()
        scaling_col_j_2 = jdata.loc[:, scaling_col_2].to_numpy()
        scaling_col_j_3 = jdata.loc[:, scaling_col_3].to_numpy()
        scaling_col_j_4 = jdata.loc[:, scaling_col_4].to_numpy()

        Ys = scaling_col_s_1[:, None] * (model.A['12'] + model.R12 * (model.A['2s'] - model.A['2ma']))[Ls, :] \
            + scaling_col_s_2[:, None] * model.A['2s'][Ls, :] \
            + scaling_col_s_3[:, None] * model.A['3s'][Ls, :] \
            + scaling_col_s_4[:, None] * (model.A['43'] + model.R43 * (model.A['3s'] - model.A['3ma']))[Ls, :]
        Ym = scaling_col_j_1[:, None] * (model.A['12'] + model.R12 * (model.A['2s'] - model.A['2ma']))[Lm, :] \
            + scaling_col_j_2[:, None] * model.A['2s'][Lm, :] \
            + scaling_col_j_3[:, None] * model.A['3s'][Lm, :] \
            + scaling_col_j_4[:, None] * (model.A['43'] + model.R43 * (model.A['3s'] - model.A['3ma']))[Lm, :]
        Y = np.append(Ym, Ys, axis=0)
    if method == 'min':
        Y *= -1
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

def _simulate_types_wages(model, jdata, sdata, gj=None, gs=None, pk1=None, pk0=None, qi_j=None, qi_s=None, qi_cum_j=None, qi_cum_s=None, optimal_reallocation=False, reallocation_constraint_category=None, reallocation_scaling_col=None, worker_types_as_ids=True, simulate_wages=True, return_long_df=True, store_worker_types=True, rng=None):
    '''
    Using data and estimated BLM parameters, simulate worker types (and optionally wages).

    Arguments:
        model (DynamicBLMModel): dynamic BLM model with estimated parameters
        jdata (BipartitePandas DataFrame): extended event study format labor data for movers
        sdata (BipartitePandas DataFrame): extended event study format labor data for stayers
        gj (NumPy Array or None): firm classes for movers in the first and last periods; if None, extract from jdata
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
        rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

    Returns:
        (BipartitePandas DataFrame or tuple of NumPy Arrays): if return_long_df is True, return BipartitePandas DataFrame with simulated data; if False, return tuple of (Lm --> vector of mover types; Ls --> vector of stayer types; yj --> tuple of wages for movers for all four periods; ys --> tuple of wages for stayers for all four periods) if simulating wages; otherwise, (Lm, Ls)
    '''
    if optimal_reallocation and ((pk1 is not None) or (pk0 is not None)):
        raise ValueError('Cannot specify `optimal_reallocation` with `pk1` and `pk0`.')
    if optimal_reallocation and (optimal_reallocation not in ['max', 'min']):
        raise ValueError(f"`optimal_reallocation` must be one of False, 'max', or 'min', but input specifies {optimal_reallocation!r}.")

    if rng is None:
        rng = np.random.default_rng(None)

    ## Unpack parameters ##
    nl, nk = model.nl, model.nk

    ## Firm classes ##
    if gj is None:
        gj = jdata.loc[:, ['g1', 'g4']].to_numpy().astype(int, copy=True)
    if gs is None:
        gs = sdata.loc[:, 'g1'].to_numpy().astype(int, copy=True)

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
                    adata.loc[cat_i_a, 'g1'], adata.loc[cat_i_a, 'g2'], adata.loc[cat_i_a, 'g3'], adata.loc[cat_i_a, 'g4'] = (gs_i, gs_i, gs_i, gs_i)
                    adata.loc[cat_i_a, 'j1'], adata.loc[cat_i_a, 'j2'], adata.loc[cat_i_a, 'j3'], adata.loc[cat_i_a, 'j4'] = (gs_i, gs_i, gs_i, gs_i)
            del cat_cons_col, cat_cons_col_a, cat_cons_col_s, cat_cons_col_j, cat_i_a, cat_i_j, cat_i_s, gs_i
        else:
            gs = _optimal_reallocation(model, jdata, sdata, gj, gs, Lm, Ls, method=optimal_reallocation, reallocation_scaling_col=reallocation_scaling_col, rng=rng)
            # Set G1/G2 and J1/J2
            adata.loc[:, 'g1'], adata.loc[:, 'g2'], adata.loc[:, 'g3'], adata.loc[:, 'g4'] = (gs, gs, gs, gs)
            adata.loc[:, 'j1'], adata.loc[:, 'j2'], adata.loc[:, 'j3'], adata.loc[:, 'j4'] = (gs, gs, gs, gs)
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
            yj = tw.simdblm._simulate_wages_movers(jdata, Lm, dynamic_blm_model=model, G1=gj[:, 0], G2=gj[:, 1], rng=rng)
        else:
            yj = (np.array([]), np.array([]), np.array([]), np.array([]))
        if len(sdata) > 0:
            ys = tw.simdblm._simulate_wages_stayers(sdata, Ls, dynamic_blm_model=model, G=gs, rng=rng)
        else:
            ys = (np.array([]), np.array([]), np.array([]), np.array([]))

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
            bdf.loc[:, 'y3'] = np.append(yj[2], ys[2])
            bdf.loc[:, 'y4'] = np.append(yj[3], ys[3])
            del yj, ys
        del Lm, Ls

    # If simulating worker types, data is not sorted
    bdf = bdf.to_long(is_sorted=(not worker_types_as_ids), copy=False)

    return bdf

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
        self.pk1 = rng.dirichlet(alpha=np.ones(nl), size=nk * nk)
        if params['pk1_prior'] is not None:
            self.pk1 = (self.pk1 + params['pk1_prior']) / 2
        # Model for p(K | l, l') for stayers
        if params['pk0_prior'] is None:
            self.pk0 = np.ones((nk, nl)) / nl
        else:
            self.pk0 = params['pk0_prior'].copy()

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
            # Normalize 2mb and 3mb so that for the lowest firm type, each worker type is 0 (otherwise these are free parameters)
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
                        cons_a.add_constraints(cons.NormalizeAllWorkerTypes(min_firm_type=min_firm_type, nnt=range(nt), nt=nt, dynamic=True))
                    else:
                        if any_tnv_wi:
                            # Normalize primary period
                            cons_a.add_constraints(cons.NormalizeAllWorkerTypes(min_firm_type=min_firm_type, cross_period_normalize=True, nnt=pp, nt=nt, dynamic=True))
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

    def sorted_firm_classes(self):
        '''
        Return list of sorted firm classes based on estimated parameters.

        Returns:
            (NumPy Array): new firm class order
        '''
        ## Unpack attributes ##
        params = self.params
        A = self.A

        ## Primary period ##
        if params['primary_period'] == 'first':
            A_mean = A['12']
        elif params['primary_period'] == 'second':
            A_mean = A['43']
        elif params['primary_period'] == 'all':
            A_mean = (A['12'] + A['43']) / 2

        return np.mean(A_mean, axis=0).argsort()

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
            jdata (BipartitePandas DataFrame): extended event study format labor data for movers
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
            jdata (BipartitePandas DataFrame): extended event study format labor data for movers
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

        # Dictionary linking periods to vectors
        C_dict = {period: C1 if period in self.first_periods else C2 for period in periods}

        # Joint firm indicator
        KK = G1 + nk * G2
        KK2 = np.tile(KK, (nl, 1)).T
        KK3 = KK2 + nk * nk * np.arange(nl)
        KK2 = KK3.flatten()
        del KK3
        KK_dict = {col: C1[col] + col_dict['n'] * C2[col] for col, col_dict in cat_dict.items()}

        # # Transition probability matrix
        # GG12 = csc_matrix((np.ones(ni), (range(ni), KK)), shape=(ni, nk * nk))

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
                            (S['12'][l, :] ** 2)[G1] + S_sum_sq['12'] + S_sum_sq_l['12']
                    )
                    lp2 = lognormpdf(
                        Y2 - (A['2mb'][G2] + A_sum['2mb'] + A_sum_l['2mb']),
                        (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']),
                        var=\
                            ((S['2ma'][l, :] ** 2)[G1] + S_sum_sq['2ma'] + S_sum_sq_l['2ma'])
                    )
                    lp3 = lognormpdf(
                        Y3 - (A['3mb'][G1] + A_sum['3mb'] + A_sum_l['3mb']) \
                            - R32m * (Y2 \
                                - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']) \
                                - (A['2mb'][G2] + A_sum['2mb'] + A_sum_l['2mb'])),
                        A['3ma'][l, G2] + A_sum['3ma'] + A_sum_l['3ma'],
                        var=\
                            (S['3ma'][l, :] ** 2)[G2] + S_sum_sq['3ma'] + S_sum_sq_l['3ma']
                    )
                    lp4 = lognormpdf(
                        Y4 - R43 * (Y3 - (A['3ma'][l, G2] + A_sum['3ma'] + A_sum_l['3ma'])),
                        A['43'][l, G2] + A_sum['43'] + A_sum_l['43'],
                        var=\
                            (S['43'][l, :] ** 2)[G2] + S_sum_sq['43'] + S_sum_sq_l['43']
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
                                var=S['12'][l, g1] ** 2
                            )
                            lp2 = lognormpdf(
                                Y2[I] \
                                    - A['2mb'][g2],
                                A['2ma'][l, g1],
                                var=S['2ma'][l, g1] ** 2
                            )
                            lp3 = lognormpdf(
                                Y3[I] \
                                    - A['3mb'][g1] \
                                    - R32m * (Y2[I] - (A['2ma'][l, g1] + A['2mb'][g2])),
                                A['3ma'][l, g2],
                                var=S['3ma'][l, g2] ** 2
                            )
                            lp4 = lognormpdf(
                                Y4[I] \
                                    - R43 * (Y3[I] - A['3ma'][l, g2]),
                                A['43'][l, g2],
                                var=S['43'][l, g2] ** 2
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
                pk1 = np.bincount(KK2, weights=(qi + d_prior - 1).flatten()).reshape(nl, nk * nk).T
                # Normalize rows to sum to 1
                pk1 = DxM(1 / np.sum(pk1, axis=1), pk1)

                if pd.isna(pk1).any():
                    warnings.warn('Estimated pk1 has NaN values. Please try a different set of starting values.')
                    break
                    # raise ValueError('Estimated pk1 has NaN values. Please try a different set of starting values.')

            # ---------- M-step ----------
            # Alternate between updating A/S and updating rho
            if update_rho and ((iter % (params['update_rho_period_movers'] + 1)) == 1):
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
                            + S_sum_sq['12'] + S_sum_sq_l['12'])
                        WW12[l * ni: (l + 1) * ni] = qi[:, l] / np.sqrt(SS12)
                    if params['update_rho43']:
                        XX43[l * ni: (l + 1) * ni] = Y3 - A['3ma'][l, G2] - A_sum['3ma'] - A_sum_l['3ma']
                        YY43[l * ni: (l + 1) * ni] = Y4 - A['43'][l, G2] - A_sum['43'] - A_sum_l['43']
                        SS43 = ( \
                            (S['43'][l, :] ** 2)[G2] \
                            + S_sum_sq['43'] + S_sum_sq_l['43'])
                        WW43[l * ni: (l + 1) * ni] = qi[:, l] / np.sqrt(SS43)
                    if params['update_rho32m']:
                        XX32m[l * ni: (l + 1) * ni] = Y2 - (A['2ma'][l, G1] + A['2mb'][G2]) - (A_sum['2ma'] + A_sum['2mb']) - (A_sum_l['2ma'] + A_sum_l['2mb'])
                        YY32m[l * ni: (l + 1) * ni] = Y3 - (A['3ma'][l, G2] + A['3mb'][G1]) - (A_sum['3ma'] + A_sum['3mb']) - (A_sum_l['3ma'] + A_sum_l['3mb'])
                        SS32m = ( \
                            (S['3ma'][l, :] ** 2)[G2] + S_sum_sq['3ma'] + S_sum_sq_l['3ma'])
                        WW32m[l * ni: (l + 1) * ni] = qi[:, l] / np.sqrt(SS32m)

                ## OLS ##
                if params['update_rho12']:
                    Xw = XX12 * WW12
                    XwX = np.sum(Xw * XX12)
                    XwY = np.sum(Xw * YY12)
                    R12 = XwY / XwX
                    del XX12, YY12, SS12, WW12
                if params['update_rho43']:
                    Xw = XX43 * WW43
                    XwX = np.sum(Xw * XX43)
                    XwY = np.sum(Xw * YY43)
                    R43 = XwY / XwX
                    del XX43, YY43, SS43, WW43
                if params['update_rho32m']:
                    Xw = XX32m * WW32m
                    XwX = np.sum(Xw * XX32m)
                    XwY = np.sum(Xw * YY32m)
                    R32m = XwY / XwX
                    del XX32m, YY32m, SS32m, WW32m
                del Xw, XwX, XwY
            elif params['update_a_movers'] or params['update_s_movers']:
                # Constrained OLS (source: https://scaron.info/blog/quadratic-programming-in-python.html)
                # The regression has 6 * nl * nk parameters and 4 * nl * ni rows
                # To avoid duplicating the data 4 * nl times, we construct X'X and X'Y by looping over nl
                # We also note that X'X is block diagonal with nl matrices of dimension (6 * nk, 6 * nk)

                #### Initialize X terms ####
                # X =
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
                # 0 matrix
                XX0 = np.zeros((nk, nk))
                if params['update_a_movers']:
                    XXwXX = np.zeros((len(periods) * ts, len(periods) * ts))
                    XXwY = np.zeros(shape=len(periods) * ts)
                if params['update_s_movers']:
                    XSwXS = np.zeros(len(periods_var) * ts)
                    XSwE = np.zeros(shape=len(periods_var) * ts)

                ### Categorical ###
                if len(cat_cols) > 0:
                    # Shift between periods
                    ts_cat = {col: nl * col_dict['n'] for col, col_dict in cat_dict.items()}
                    # 0 matrix
                    XX0_cat = {col: np.zeros((col_dict['n'], col_dict['n'])) for col, col_dict in cat_dict.items()}
                    if params['update_a_movers']:
                        XXwXX_cat = {col: np.zeros((len(periods) * col_ts, len(periods) * col_ts)) for col, col_ts in ts_cat.items()}
                        XXwY_cat = {col: np.zeros(shape=len(periods) * col_ts) for col, col_ts in ts_cat.items()}
                    if params['update_s_movers']:
                        XSwXS_cat = {col: np.zeros(shape=len(periods_var) * col_ts) for col, col_ts in ts_cat.items()}
                        XSwE_cat = {col: np.zeros(shape=len(periods_var) * col_ts) for col, col_ts in ts_cat.items()}

                ### Continuous ###
                if len(cts_cols) > 0:
                    if params['update_a_movers']:
                        XXwXX_cts = {col: np.zeros((len(periods) * nl, len(periods) * nl)) for col in cts_cols}
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

                    ## Compute GwG terms ##
                    G1W1G1 = np.diag(np.bincount(G1, weights_l[0]))
                    G1W2G1 = np.diag(np.bincount(G1, weights_l[1]))
                    G2W3G2 = np.diag(np.bincount(G2, weights_l[2]))
                    G2W4G2 = np.diag(np.bincount(G2, weights_l[3]))
                    if params['update_a_movers']:
                        G1W3G1 = np.diag(np.bincount(G1, weights_l[2]))
                        if endogeneity:
                            G2W2G2 = np.diag(np.bincount(G2, weights_l[1]))
                        if endogeneity:
                            G1W2G2 = double_bincount(KK, nk, weights_l[1])
                        G1W3G2 = double_bincount(KK, nk, weights_l[2])

                    if params['update_s_movers']:
                        ## Compute XSwXS_l ##
                        weights.append(weights_l)
                        l_index_S = l * nk * len(periods_var)

                        XSwXS[l_index_S + 0 * nk: l_index_S + 1 * nk] = \
                            np.diag(G1W1G1)
                        XSwXS[l_index_S + 1 * nk: l_index_S + 2 * nk] = \
                            np.diag(G2W4G2)
                        XSwXS[l_index_S + 2 * nk: l_index_S + 3 * nk] = \
                            np.diag(G1W2G1)
                        XSwXS[l_index_S + 3 * nk: l_index_S + 4 * nk] = \
                            np.diag(G2W3G2)

                    if params['update_a_movers']:
                        ## Compute XXwXX_l ##
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

                        XXwXX[l_index: r_index, l_index: r_index] = XXwXX_l
                        del XXwXX_l, G1W3G1, G1W3G2
                        if endogeneity:
                            del G2W2G2, G1W2G2
                    del G1W1G1, G1W2G1, G2W3G2, G2W4G2

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
                                np.bincount(G1, - R12 * weights_l[0] * Yl_1 + weights_l[1] * Yl_2 - R32m * weights_l[2] * Yl_3),
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

                if params['update_a_movers'] and (params['d_X_diag_movers_A'] > 1):
                    XXwXX += (params['d_X_diag_movers_A'] - 1) * np.eye(XXwXX.shape[0])
                if params['update_s_movers'] and (params['d_X_diag_movers_S'] > 1):
                    XSwXS += (params['d_X_diag_movers_S'] - 1)

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
                            S_l_dict = {period: S_cat[col][period][l, C_dict[period][col]] for period in periods_var}
                        else:
                            S_l_dict = {period: S_cat[col][period][C_dict[period][col]] for period in periods_var}

                        weights_l = [
                            qi[:, l] / S_l_dict['12'],
                            qi[:, l] / S_l_dict['2ma'],
                            qi[:, l] / S_l_dict['3ma'],
                            qi[:, l] / S_l_dict['43']
                        ]
                        del S_l_dict

                        ## Compute CwC terms ##
                        C1W1C1 = np.diag(np.bincount(C1[col], weights_l[0]))
                        C1W2C1 = np.diag(np.bincount(C1[col], weights_l[1]))
                        C2W3C2 = np.diag(np.bincount(C2[col], weights_l[2]))
                        C2W4C2 = np.diag(np.bincount(C2[col], weights_l[3]))
                        if params['update_a_movers']:
                            C1W3C1 = np.diag(np.bincount(C1[col], weights_l[2]))
                            if endogeneity:
                                C2W2C2 = np.diag(np.bincount(C2[col], weights_l[1]))
                            if endogeneity:
                                C1W2C2 = double_bincount(KK_dict[col], col_n, weights_l[1])
                            C1W3C2 = double_bincount(KK_dict[col], col_n, weights_l[2])

                        if params['update_s_movers']:
                            ### Compute XSwXS_cat_l ###
                            weights_cat[col].append(weights_l)
                            l_index_S = l * col_n * len(periods_var)

                            XSwXS_cat[col][l_index_S + 0 * col_n: l_index_S + 1 * col_n] = \
                                np.diag(C1W1C1)
                            XSwXS_cat[col][l_index_S + 1 * col_n: l_index_S + 2 * col_n] = \
                                np.diag(C2W4C2)
                            XSwXS_cat[col][l_index_S + 2 * col_n: l_index_S + 3 * col_n] = \
                                np.diag(C1W2C1)
                            XSwXS_cat[col][l_index_S + 3 * col_n: l_index_S + 4 * col_n] = \
                                np.diag(C2W3C2)

                        if params['update_a_movers']:
                            ## Compute XXwXX_cat_l ##
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

                            XXwXX_cat[col][l_index: r_index, l_index: r_index] = XXwXX_cat_l
                            del XXwXX_cat_l, C1W3C1, C1W3C2
                            if endogeneity:
                                del C2W2C2, C1W2C2
                        del C1W1C1, C1W2C1, C2W3C2, C2W4C2

                        if params['update_a_movers']:
                            # Update A_sum to account for worker-interaction terms
                            A_sum_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=False, periods=periods)
                            if cat_dict[col]['worker_type_interaction']:
                                # Adjust A_sum
                                for period in periods:
                                    if period[-1] != 'b':
                                        A_sum_l[period] -= A_cat[col][period][l, C_dict[period][col]]
                                    else:
                                        A_sum_l[period] -= A_cat[col][period][C_dict[period][col]]

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
                                    np.bincount(C1[col], - R12 * weights_l[0] * Yl_cat_1 + weights_l[1] * Yl_cat_2 - R32m * weights_l[2] * Yl_cat_3),
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

                    if params['update_a_movers'] and (params['d_X_diag_movers_A'] > 1):
                        XXwXX_cat[col] += (params['d_X_diag_movers_A'] - 1) * np.eye(XXwXX_cat[col].shape[0])
                    if params['update_s_movers'] and (params['d_X_diag_movers_S'] > 1):
                        XSwXS_cat[col] += (params['d_X_diag_movers_S'] - 1)

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

                        ## Compute CwC terms ##
                        C1W1C1 = np.sum(weights_l[0] * C1[col])
                        C1W2C1 = np.sum(weights_l[1] * C1[col])
                        C2W3C2 = np.sum(weights_l[2] * C2[col])
                        C2W4C2 = np.sum(weights_l[3] * C2[col])
                        if params['update_a_movers']:
                            C1W3C1 = np.sum(weights_l[2] * C1[col])
                            if endogeneity:
                                C2W2C2 = np.sum(weights_l[1] * C2[col])
                            if endogeneity:
                                C1W2C2 = np.sum(weights_l[1] * C1[col] * C2[col])
                            C1W3C2 = np.sum(weights_l[2] * C1[col] * C2[col])

                        if params['update_s_movers']:
                            ### Compute XSwXS_cts_l ###
                            weights_cts[col].append(weights_l)
                            l_index_S = l * len(periods_var)

                            XSwXS_cts[col][l_index_S + 0] = C1W1C1
                            XSwXS_cts[col][l_index_S + 1] = C2W4C2
                            XSwXS_cts[col][l_index_S + 2] = C1W2C1
                            XSwXS_cts[col][l_index_S + 3] = C2W3C2

                        if params['update_a_movers']:
                        ## Compute XXwXX_cts_l ##
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

                            XXwXX_cts[col][l_index: r_index, l_index: r_index] = XXwXX_cts_l
                            del XXwXX_cts_l, C1W3C1, C1W3C2
                            if endogeneity:
                                del C2W2C2, C1W2C2
                        del C1W1C1, C1W2C1, C2W3C2, C2W4C2

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

                            ## Compute XXwY_cts_l ##
                            XXwY_cts_l = np.array(
                                [
                                    np.sum(weights_l[0] * Yl_cts_1 * C1[col]),
                                    np.sum(weights_l[3] * Yl_cts_4 * C2[col]),
                                    np.sum((- R12 * weights_l[0] * Yl_cts_1 + weights_l[1] * Yl_cts_2 - R32m * weights_l[2] * Yl_cts_3) * C1[col]),
                                    np.sum((weights_l[2] * Yl_cts_3 - R43 * weights_l[3] * Yl_cts_4) * C2[col])
                                ]
                            )
                            if endogeneity:
                                XXwY_cts_l = np.append(
                                    XXwY_cts_l, np.sum((weights_l[1] * Yl_cts_2 - R32m * weights_l[2] * Yl_cts_3) * C2[col])
                                )
                            if state_dependence:
                                XXwY_cts_l = np.append(
                                    XXwY_cts_l, np.sum(weights_l[2] * Yl_cts_3 * C1[col])
                                )
                            XXwY_cts[col][l_index: r_index] = XXwY_cts_l
                            del Yl_cts_1, Yl_cts_2, Yl_cts_3, Yl_cts_4, XXwY_cts_l, A_sum_l
                        del weights_l

                    if params['update_a_movers'] and (params['d_X_diag_movers_A'] > 1):
                        XXwXX_cts[col] += (params['d_X_diag_movers_A'] - 1) * np.eye(XXwXX_cts[col].shape[0])
                    if params['update_s_movers'] and (params['d_X_diag_movers_S'] > 1):
                        XSwXS_cts[col] += (params['d_X_diag_movers_S'] - 1)

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
                            np.bincount(G1, weights[l][0])
                        XSwE[l_index + 1 * nk: l_index + 2 * nk] = \
                            np.bincount(G2, weights[l][3])
                        XSwE[l_index + 2 * nk: l_index + 3 * nk] = \
                            np.bincount(G1, weights[l][1])
                        XSwE[l_index + 3 * nk: l_index + 4 * nk] = \
                            np.bincount(G2, weights[l][2])

                        weights[l] = 0
                    del weights

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
                                S_l_dict = {period: (S_cat[col][period][l, :] ** 2)[C_dict[period][col]] for period in periods_var}
                            else:
                                S_l_dict = {period: (S_cat[col][period] ** 2)[C_dict[period][col]] for period in periods_var}

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

                            XSwE_cat[col][l_index + 0 * col_n: l_index + 1 * col_n] = \
                                np.bincount(C1[col], weights_cat[col][l][0])
                            XSwE_cat[col][l_index + 1 * col_n: l_index + 2 * col_n] = \
                                np.bincount(C2[col], weights_cat[col][l][3])
                            XSwE_cat[col][l_index + 2 * col_n: l_index + 3 * col_n] = \
                                np.bincount(C1[col], weights_cat[col][l][1])
                            XSwE_cat[col][l_index + 3 * col_n: l_index + 4 * col_n] = \
                                np.bincount(C2[col], weights_cat[col][l][2])

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
                            XSwE_cts[col][l_index + 0] = \
                                np.abs(np.sum(C1[col] * weights_cts[col][l][0]))
                            XSwE_cts[col][l_index + 1] = \
                                np.abs(np.sum(C2[col] * weights_cts[col][l][3]))
                            XSwE_cts[col][l_index + 2] = \
                                np.abs(np.sum(C1[col] * weights_cts[col][l][1]))
                            XSwE_cts[col][l_index + 3] = \
                                np.abs(np.sum(C2[col] * weights_cts[col][l][2]))

                            weights_cts[col][l] = 0
                        del weights_cts[col]

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
            sdata (BipartitePandas DataFrame): extended event study format labor data for stayers
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
        any_controls = self.any_controls

        # Store wage outcomes and groups
        Y1 = sdata['y1'].to_numpy()
        Y2 = sdata['y2'].to_numpy()
        Y3 = sdata['y3'].to_numpy()
        Y4 = sdata['y4'].to_numpy()
        G1 = sdata['g1'].to_numpy().astype(int, copy=False)

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

        # Dictionary linking periods to vectors
        C_dict = {period: C1 if period in self.first_periods else C2 for period in periods}

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
            A_sum, S_sum_sq = self._sum_by_non_nl(ni=ni, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)

            for l in range(nl):
                # Update A_sum/S_sum_sq to account for worker-interaction terms
                A_sum_l, S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)

                lp1 = lognormpdf(
                    Y1 - R12 * (Y2 - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma'])),
                    A['12'][l, G1] + A_sum['12'] + A_sum_l['12'],
                    var=\
                        (S['12'][l, :] ** 2)[G1] + S_sum_sq['12'] + S_sum_sq_l['12']
                )
                lp4 = lognormpdf(
                    Y4 - R43 * (Y3 - (A['3ma'][l, G1] + A_sum['3ma'] + A_sum_l['3ma'])),
                    A['43'][l, G1] + A_sum['43'] + A_sum_l['43'],
                    var=\
                        (S['43'][l, :] ** 2)[G1] + S_sum_sq['43'] + S_sum_sq_l['43']
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
                        var=S['12'][l, g1] ** 2
                    )
                    lp4 = lognormpdf(
                        Y4[I] - R43 * (Y3[I] - A['3ma'][l, g1]),
                        A['43'][l, g1],
                        var=S['43'][l, g1] ** 2
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
                            (S['3s'][l, :] ** 2)[G1] + S_sum_sq['3s'] + S_sum_sq_l['3s']
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
                            var=S['3s'][l, g1] ** 2
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
            if iter == (params['n_iters_stayers'] - 1):
                print(f"Maximum iterations reached for stayers. It is recommended to increase `n_iters_stayers` from its current value of {params['n_iters_stayers']}.")
            prev_lik = lik0

            # ---------- Update pk0 ----------
            if params['update_pk0']:
                # NOTE: add dirichlet prior
                # NOTE: this is equivalent to pk0 = GG1.T @ (qi + d_prior - 1)
                pk0 = np.bincount(KK, (qi + d_prior - 1).flatten()).reshape(nl, nk).T
                # Normalize rows to sum to 1
                pk0 = DxM(1 / np.sum(pk0, axis=1), pk0)

                if pd.isna(pk0).any():
                    warnings.warn('Estimated pk0 has NaN values. Please try a different set of starting values.')
                    break
                    # raise ValueError('Estimated pk0 has NaN values. Please try a different set of starting values.')

            # ---------- M-step ----------
            # Alternate between updating A/S and updating rho
            if params['update_rho32s'] and ((iter % (params['update_rho_period_stayers'] + 1)) == 1):
                ## Update rho ##
                XX32s = np.zeros(nl * ni)
                YY32s = np.zeros(nl * ni)
                WW32s = np.zeros(nl * ni)
                for l in range(nl):
                    A_sum_l, S_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)

                    XX32s[l * ni: (l + 1) * ni] = Y2 - A['2s'][l, G1] - A_sum['2s'] - A_sum_l['2s']
                    YY32s[l * ni: (l + 1) * ni] = Y3 - A['3s'][l, G1] - A_sum['3s'] - A_sum_l['3s']
                    SS32s = ( \
                        (S['3s'][l, :] ** 2)[G1] + S_sum_sq['3s'] + S_sum_sq_l['3s'])
                    WW32s[l * ni: (l + 1) * ni] = qi[:, l] / np.sqrt(SS32s)

                ## OLS ##
                Xw = XX32s * WW32s
                XwX = np.sum(Xw * XX32s)
                XwY = np.sum(Xw * YY32s)
                R32s = XwY / XwX
                del XX32s, YY32s, SS32s, WW32s, Xw, XwX, XwY
            elif params['update_a_stayers'] or params['update_s_stayers']:
                # Constrained OLS (source: https://scaron.info/blog/quadratic-programming-in-python.html)
                # The regression has 2 * nk parameters and 2 * ni rows
                # To avoid duplicating the data 2 * nl times, we construct X'X and X'Y by looping over nl
                # We also note that X'X is block diagonal with nl matrices of dimension (2 * nk, 2 * nk)

                ### Initialize X terms ###
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

                ### General ###
                # Shift between periods
                ts = nl * nk
                if params['update_a_stayers']:
                    XXwXX = np.zeros((len(periods_update) * ts, len(periods_update) * ts))
                    XXwY = np.zeros(shape=len(periods_update) * ts)
                if params['update_s_stayers']:
                    XSwXS = np.zeros(len(periods_update) * ts)
                    XSwE = np.zeros(shape=len(periods_update) * ts)

                ### Categorical ###
                if len(cat_cols) > 0:
                    # Shift between periods
                    ts_cat = {col: nl * col_dict['n'] for col, col_dict in cat_dict.items()}
                    if params['update_a_stayers']:
                        XXwXX_cat = {col: np.zeros((len(periods_update) * col_ts, len(periods_update) * col_ts)) for col, col_ts in ts_cat.items()}
                        XXwY_cat = {col: np.zeros(shape=len(periods_update) * col_ts) for col, col_ts in ts_cat.items()}
                    if params['update_s_stayers']:
                        XSwXS_cat = {col: np.zeros(shape=len(periods_update) * col_ts) for col, col_ts in ts_cat.items()}
                        XSwE_cat = {col: np.zeros(shape=len(periods_update) * col_ts) for col, col_ts in ts_cat.items()}

                ### Continuous ###
                if len(cts_cols) > 0:
                    if params['update_a_stayers']:
                        XXwXX_cts = {col: np.zeros((len(periods_update) * nl, len(periods_update) * nl)) for col in cts_cols}
                        XXwY_cts = {col: np.zeros(shape=len(periods_update) * nl) for col in cts_cols}
                    if params['update_s_stayers']:
                        XSwXS_cts = {col: np.zeros(shape=len(periods_update) * nl) for col in cts_cols}
                        XSwE_cts = {col: np.zeros(shape=len(periods_update) * nl) for col in cts_cols}

                ## Update A ##
                if params['update_s_stayers']:
                    # Store weights computed for A for use when computing S
                    weights = []
                for l in range(nl):
                    l_index, r_index = l * nk * len(periods_update), (l + 1) * nk * len(periods_update)

                    ## Compute weights_l ##
                    weights_l = [
                        qi[:, l] / S['2s'][l, G1],
                        qi[:, l] / S['3s'][l, G1]
                    ]

                    ## Compute GwG terms ##
                    G1W1G1 = np.diag(np.bincount(G1, weights_l[0]))
                    G1W2G1 = np.diag(np.bincount(G1, weights_l[1]))

                    if params['update_s_stayers']:
                        ## Compute XSwXS_l ##
                        weights.append(weights_l)
                        l_index_S = l * nk * len(periods_update)

                        XSwXS[l_index_S + 0 * nk: l_index_S + 1 * nk] = \
                            np.diag(G1W1G1)
                        XSwXS[l_index_S + 1 * nk: l_index_S + 2 * nk] = \
                            np.diag(G1W2G1)

                    if params['update_a_stayers']:
                        ## Compute XXwXX_l ##
                        XXwXX[l_index: r_index, l_index: r_index] = np.vstack(
                            [
                                np.hstack([G1W1G1 + (R32s ** 2) * G1W2G1, - R32s * G1W2G1]),
                                np.hstack([- R32s * G1W2G1, G1W2G1])
                            ]
                        )
                    del G1W1G1, G1W2G1

                    if params['update_a_stayers']:
                        # Update A_sum to account for worker-interaction terms
                        A_sum_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=False, periods=periods)

                        # Yl_2
                        Yl_2 = \
                            Y2 \
                                - (A_sum['2s'] + A_sum_l['2s'])
                        # Yl_3
                        Yl_3 = \
                            Y3 \
                                - (A_sum['3s'] + A_sum_l['3s']) \
                                - R32s * (Y2 - (A_sum['2s'] + A_sum_l['2s']))

                        ## Compute XXwY_l ##
                        XXwY[l_index: r_index] = np.concatenate(
                            [
                                np.bincount(G1, weights_l[0] * Yl_2 - R32s * weights_l[1] * Yl_3),
                                np.bincount(G1, weights_l[1] * Yl_3)
                            ]
                        )
                        del Yl_2, Yl_3, A_sum_l
                    del weights_l

                if params['update_a_stayers'] and (params['d_X_diag_stayers_A'] > 1):
                    XXwXX += (params['d_X_diag_stayers_A'] - 1) * np.eye(XXwXX.shape[0])
                if params['update_s_stayers'] and (params['d_X_diag_stayers_S'] > 1):
                    XSwXS += (params['d_X_diag_stayers_S'] - 1)

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
                            S_l_dict = {period: S_cat[col][period][l, C_dict[period][col]] for period in periods_update}
                        else:
                            S_l_dict = {period: S_cat[col][period][C_dict[period][col]] for period in periods_update}

                        weights_l = [
                            qi[:, l] / S_l_dict['2s'],
                            qi[:, l] / S_l_dict['3s']
                        ]

                        ## Compute CwC terms ##
                        C1W1C1 = np.diag(np.bincount(C1[col], weights_l[0]))
                        C1W2C1 = np.diag(np.bincount(C1[col], weights_l[1]))

                        if params['update_s_stayers']:
                            ## Compute XSwXS_cat_l ##
                            weights_cat[col].append(weights_l)
                            l_index_S = l * col_n * len(periods_update)

                            XSwXS_cat[col][l_index + 0 * col_n: l_index + 1 * col_n] = \
                                np.diag(C1W1C1)
                            XSwXS_cat[col][l_index + 1 * col_n: l_index + 2 * col_n] = \
                                np.diag(C1W2C1)

                        if params['update_a_stayers']:
                            ## Compute XXwXX_cat_l ##
                            XXwXX_cat[col][l_index: r_index, l_index: r_index] = np.vstack(
                                [
                                    np.hstack([C1W1C1 + (R32s ** 2) * C1W2C1, - R32s * C1W2C1]),
                                    np.hstack([- R32s * C1W2C1, C1W2C1])
                                ]
                            )
                        del C1W1C1, C1W2C1

                        if params['update_a_stayers']:
                            # Update A_sum to account for worker-interaction terms
                            A_sum_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=False, periods=periods_update)
                            if cat_dict[col]['worker_type_interaction']:
                                # Adjust A_sum
                                for period in periods_update:
                                    A_sum_l[period] -= A_cat[col][period][l, C_dict[period][col]]

                            # Yl_cat_2
                            Yl_cat_2 = \
                                Y2 \
                                    - (A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s'])
                            # Yl_cat_3
                            Yl_cat_3 = \
                                Y3 \
                                    - (A['3s'][l, G1] + A_sum['3s'] + A_sum_l['3s']) \
                                    - R32s * (Y2 - (A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s']))

                            ## Compute XXwY_cat_l ##
                            XXwY_cat[col][l_index: r_index] = np.concatenate(
                                [
                                    np.bincount(C1[col], weights_l[0] * Yl_cat_2 - R32s * weights_l[1] * Yl_cat_3),
                                    np.bincount(C1[col], weights_l[1] * Yl_cat_3)
                                ]
                            )
                            del Yl_cat_2, Yl_cat_3, A_sum_l
                        del weights_l

                    if params['update_a_stayers'] and (params['d_X_diag_stayers_A'] > 1):
                        XXwXX_cat[col] += (params['d_X_diag_stayers_A'] - 1) * np.eye(XXwXX_cat[col].shape[0])
                    if params['update_a_stayers'] and (params['d_X_diag_stayers_S'] > 1):
                        XSwXS_cat[col] += (params['d_X_diag_stayers_S'] - 1)

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
                            S_l_dict = {period: S_cts[col][period][l] for period in periods_update}
                        else:
                            S_l_dict = {period: S_cts[col][period] for period in periods_update}

                        weights_l = [
                            qi[:, l] / S_l_dict['2s'],
                            qi[:, l] / S_l_dict['3s']
                        ]

                        ## Compute CwC terms ##
                        C1W1C1 = np.sum(C1[col] * weights_l[0])
                        C1W2C1 = np.sum(C1[col] * weights_l[1])

                        if params['update_s_stayers']:
                            ## Compute XSwXS_cts_l ##
                            weights_cts[col].append(weights_l)
                            l_index_S = l * len(periods_update)

                            XSwXS_cts[col][l_index + 0] = C1W1C1
                            XSwXS_cts[col][l_index + 1] = C1W2C1

                        if params['update_a_stayers']:
                            ## Compute XXwXX_cts_l ##
                            XXwXX_cts[col][l_index: r_index, l_index: r_index] = np.array(
                                [
                                    [C1W1C1 + (R32s ** 2) * C1W2C1, - R32s * C1W2C1],
                                    [- R32s * C1W2C1, C1W2C1]
                                ]
                            )
                        del C1W1C1, C1W2C1

                        if params['update_a_stayers']:
                            # Update A_sum to account for worker-interaction terms
                            A_sum_l = self._sum_by_nl_l(ni=ni, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, compute_S=False, periods=periods_update)
                            if cts_dict[col]['worker_type_interaction']:
                                # Adjust A_sum
                                for period in periods_update:
                                    A_sum_l[period] -= A_cts[col][period][l] * C_dict[period][col]

                            # Yl_cts_2
                            Yl_cts_2 = \
                                Y2 \
                                    - (A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s'])
                            # Yl_cts_3
                            Yl_cts_3 = \
                                Y3 \
                                    - (A['3s'][l, G1] + A_sum['3s'] + A_sum_l['3s']) \
                                    - R32s * (Y2 - (A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s']))

                            ## Compute XXwY_cts_l ##
                            XXwY_cts[col][l_index: r_index] = np.array(
                                [
                                    np.sum(C1[col] * (weights_l[0] * Yl_cts_2 - R32s * weights_l[1] * Yl_cts_3)),
                                    np.sum(C1[col] * weights_l[1] * Yl_cts_3)
                                ]
                            )
                            del Yl_cts_2, Yl_cts_3, A_sum_l
                        del weights_l

                    if params['update_a_stayers'] and (params['d_X_diag_stayers_A'] > 1):
                        XXwXX_cts[col] += (params['d_X_diag_stayers_A'] - 1) * np.eye(XXwXX_cts[col].shape[0])
                    if params['update_s_stayers'] and (params['d_X_diag_stayers_S'] > 1):
                        XSwXS_cts[col] += (params['d_X_diag_stayers_S'] - 1)

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
                        # eps_l_sq_2
                        eps_l_sq.append(
                            (Y2 \
                                - (A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s'])
                                ) ** 2
                        )
                        # eps_l_sq_3
                        eps_l_sq.append(
                            (Y3 \
                                - (A['3s'][l, G1] + A_sum['3s'] + A_sum_l['3s']) \
                                - R32s * (Y2 - (A['2s'][l, G1] + A_sum['2s'] + A_sum_l['2s']))
                                ) ** 2
                        )
                        eps_sq.append(eps_l_sq)
                        del A_sum_l, eps_l_sq

                        ## XSwE ##
                        l_index, r_index = l * nk * len(periods_update), (l + 1) * nk * len(periods_update)

                        weights[l][0] *= eps_sq[l][0]
                        weights[l][1] *= eps_sq[l][1]

                        if any_controls:
                            ## Account for other variables' contribution to variance ##
                            var_l_numerator = [
                                (S['2s'][l, :] ** 2)[G1],
                                (S['3s'][l, :] ** 2)[G1]
                            ]
                            var_l_denominator = [
                                (S['2s'][l, :] ** 2)[G1] \
                                    + S_sum_sq['2s'] + S_sum_sq_l['2s'],
                                (S['3s'][l, :] ** 2)[G1] \
                                    + S_sum_sq['3s'] + S_sum_sq_l['3s']
                            ]
                            del S_sum_sq_l

                            weights[l][0] *= (var_l_numerator[0] / var_l_denominator[0])
                            weights[l][1] *= (var_l_numerator[1] / var_l_denominator[1])

                        XSwE[l_index + 0 * nk: l_index + 1 * nk] = \
                            np.bincount(G1, weights[l][0])
                        XSwE[l_index + 1 * nk: l_index + 2 * nk] = \
                            np.bincount(G1, weights[l][1])

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

                            ### XSwE_cat ###
                            l_index, r_index = l * col_n * len(periods_update), (l + 1) * col_n * len(periods_update)

                            ## Compute var_l_cat ##
                            if cat_dict[col]['worker_type_interaction']:
                                S_l_dict = {period: (S_cat[col][period][l, :] ** 2)[C_dict[period][col]] for period in periods_update}
                            else:
                                S_l_dict = {period: (S_cat[col][period] ** 2)[C_dict[period][col]] for period in periods_update}

                            ## Account for other variables' contribution to variance ##
                            var_l_numerator = [
                                S_l_dict['2s'],
                                S_l_dict['3s']
                            ]
                            var_l_denominator = np.concatenate(
                                [
                                    (S['2s'][l, :] ** 2)[G1] \
                                        + S_sum_sq['2s'] + S_sum_sq_l['2s'],
                                    (S['3s'][l, :] ** 2)[G1] \
                                        + S_sum_sq['3s'] + S_sum_sq_l['3s']
                                ]
                            )
                            del S_sum_sq_l

                            weights_cat[col][l][0] *= ((var_l_numerator[0] / var_l_denominator[0]) * eps_sq[l][0])
                            weights_cat[col][l][1] *= ((var_l_numerator[1] / var_l_denominator[1]) * eps_sq[l][1])

                            XSwE_cat[col][l_index + 0 * col_n: l_index + 1 * col_n] = \
                                np.bincount(C1[col], weights_cat[col][l][0])
                            XSwE_cat[col][l_index + 1 * col_n: l_index + 2 * col_n] = \
                                np.bincount(C1[col], weights_cat[col][l][1])

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

                            ### XSwE_cts ###
                            l_index, r_index = l * len(periods_update), (l + 1) * len(periods_update)

                            ## Compute var_l_cts ##
                            if cts_dict[col]['worker_type_interaction']:
                                S_l_dict = {period: S_cts[col][period][l] ** 2 for period in periods_update}
                            else:
                                S_l_dict = {period: S_cts[col][period] ** 2 for period in periods_update}

                            ## Account for other variables' contribution to variance ##
                            var_l_numerator = [
                                S_l_dict['2s'],
                                S_l_dict['3s']
                            ]
                            var_l_denominator = [
                                (S['2s'][l, :] ** 2)[G1] \
                                    + S_sum_sq['2s'] + S_sum_sq_l['2s'],
                                (S['3s'][l, :] ** 2)[G1] \
                                    + S_sum_sq['3s'] + S_sum_sq_l['3s']
                            ]
                            del S_sum_sq_l

                            weights_cts[col][l][0] *= ((var_l_numerator[0] / var_l_denominator[0]) * eps_sq[l][0])
                            weights_cts[col][l][1] *= ((var_l_numerator[1] / var_l_denominator[1]) * eps_sq[l][1])

                            # NOTE: take absolute value
                            XSwE_cts[col][l_index + 0] = \
                                np.abs(np.sum(C1[col] * weights_cts[col][l][0]))
                            XSwE_cts[col][l_index + 1] = \
                                np.abs(np.sum(C1[col] * weights_cts[col][l][1]))

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

    def fit_movers_cstr_uncstr(self, jdata, linear_additivity=True, compute_NNm=True, blm_model=None, initialize_all=False):
        '''
        Run fit_movers(), first constrained, then using results as starting values, run unconstrained.

        Arguments:
            jdata (BipartitePandas DataFrame): extended event study format labor data for movers
            linear_additivity (bool): if True, include the loop with the linear additivity constraint
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
            blm_model (BLMModel or None): estimated static BLM model. Estimates from this model will be used as a baseline in half the starting values (the other half will be random). If None, all starting values will be random.
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
                self.A['2ma'] = copy.deepcopy(blm_model.A1)
                self.A['3ma'] = copy.deepcopy(blm_model.A2)
                self.A['2s'] = copy.deepcopy(blm_model.A1)
                self.A['3s'] = copy.deepcopy(blm_model.A2)
                self.A['2mb'][:] = 0
                self.A['3mb'][:] = 0
                self.R12 = 0.6
                self.R43 = 0.6
                self.R32m = 0.6
                self.R32s = 0.6
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
        if linear_additivity and (self.nl > 1):
            # Set constraints
            if user_params['cons_a_all'] is None:
                # Linear addivity can't be applied to endogeneity or state dependence terms
                # If linear additivity is applied to 2ma and 3ma, because each is normalized to be constant for the lowest firm type, this constrains them to be equal across all worker types - this is too restrictive, so don't impose linear additivity for these terms
                self.params['cons_a_all'] = cons.LinearAdditive(nnt=[0, 1], nt=len(self.periods_movers), dynamic=True)
            else:
                # Linear addivity can't be applied to endogeneity or state dependence terms
                # If linear additivity is applied to 2ma and 3ma, because each is normalized to be constant for the lowest firm type, this constrains them to be equal across all worker types - this is too restrictive, so don't impose linear additivity for these terms
                self.params['cons_a_all'] = to_list(user_params['cons_a_all']) + [cons.LinearAdditive(nnt=[0, 1], nt=len(self.periods_movers), dynamic=True)]
            if self.params['verbose'] in [1, 2, 3]:
                print('Fitting movers with Linear Additive constraint on A')
            self.fit_movers(jdata, compute_NNm=False)
        # ##### Loop 3 #####
        # # Now update A with Stationary Firm Type Variation constraint
        # if self.nl > 1:
        #     # Set constraints
        #     if user_params['cons_a_all'] is None:
        #         self.params['cons_a_all'] = cons.StationaryFirmTypeVariation(nt=len(self.periods_movers), R_version=True, dynamic=True)
        #     else:
        #         self.params['cons_a_all'] = to_list(user_params['cons_a_all']) + [cons.StationaryFirmTypeVariation(nt=len(self.periods_movers), R_version=True, dynamic=True)]
        #     if self.params['verbose'] in [1, 2, 3]:
        #         print('Fitting movers with Stationary Firm Type Variation constraint on A')
        #     self.fit_movers(jdata, compute_NNm=False)
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

    def fit_stayers_cstr_uncstr(self, sdata, linear_additivity=True, compute_NNs=True):
        '''
        Run fit_stayers(), first constrained, then using results as starting values, run unconstrained.

        Arguments:
            sdata (BipartitePandas DataFrame): extended event study format labor data for stayers
            linear_additivity (bool): if True, include the loop with the linear additivity constraint
            compute_NNs (bool): if True, compute vector giving the number of stayers at each firm type (e.g. entry (1) gives the number of stayers at firm type 1)
        '''
        ## First, simulate parameters but keep A fixed ##
        ## Second, use estimated parameters as starting point to run with A constrained to be linear ##
        ## Finally use estimated parameters as starting point to run without constraints ##
        # Save original parameters
        user_params = self.params.copy()
        ##### Loop 1 #####
        # First fix A but update S and pk
        self.params['update_a_stayers'] = False
        self.params['update_s_stayers'] = True
        self.params['update_pk0'] = True
        if self.params['verbose'] in [1, 2, 3]:
            print('Fitting stayers with A fixed')
        self.fit_stayers(sdata, compute_NNs=False)
        ##### Loop 2 #####
        # Now update A with Linear Additive constraint
        self.params['update_a_stayers'] = True
        if linear_additivity and (self.nl > 1):
            # Set constraints
            if user_params['cons_a_all'] is None:
                self.params['cons_a_all'] = cons.LinearAdditive(nt=2, dynamic=True)
            else:
                self.params['cons_a_all'] = to_list(user_params['cons_a_all']) + [cons.LinearAdditive(nt=2, dynamic=True)]
            if self.params['verbose'] in [1, 2, 3]:
                print('Fitting stayers with Linear Additive constraint on A')
            self.fit_stayers(sdata, compute_NNs=False)
        ##### Loop 3 #####
        # Restore user constraints
        self.params['cons_a_all'] = user_params['cons_a_all']
        # Update d_X_diag_stayers_A to be closer to 1
        self.params['d_X_diag_stayers_A'] = 1 + (self.params['d_X_diag_stayers_A'] - 1) / 2
        if self.params['verbose'] in [1, 2, 3]:
            print('Fitting unconstrained stayers')
        self.fit_stayers(sdata, compute_NNs=compute_NNs)
        # Restore original parameters
        self.params = user_params

    def fit_A(self, jdata, compute_NNm=True):
        '''
        Run fit_movers() and update A while keeping S and pk1 fixed.

        Arguments:
            jdata (BipartitePandas DataFrame): extended event study format labor data for movers
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
            jdata (BipartitePandas DataFrame): extended event study format labor data for movers
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
            jdata (BipartitePandas DataFrame): extended event study format labor data for movers
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

    def reclassify_firms(self, jdata, sdata):
        '''
        Reclassify firms.

        Arguments:
            jdata (BipartitePandas DataFrame): extended event study format labor data for movers
            sdata (BipartitePandas DataFrame): extended event study format labor data for stayers

        Returns:
            (NumPy Array): new firm classes (index corresponds to firm id)
        '''
        # Unpack parameters
        params = self.params
        nl, nk, nij, nis = self.nl, self.nk, jdata.shape[0], sdata.shape[0]
        R12, R43, R32m, R32s = self.R12, self.R43, self.R32m, self.R32s
        A, A_cat, A_cts = self.A, self.A_cat, self.A_cts
        S, S_cat, S_cts = self.S, self.S_cat, self.S_cts
        pk1, pk0 = self.pk1, self.pk0
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        cat_dict, cts_dict = self.cat_dict, self.cts_dict
        any_controls = self.any_controls

        # Number of firms (movers and stayers don't necessarily have the same firms)
        nf = max(jdata.loc[:, 'j1'].to_numpy().max(), jdata.loc[:, 'j4'].to_numpy().max(), sdata.loc[:, 'j1'].to_numpy().max()) + 1

        ### Movers ###
        # Unpack parameters
        periods, periods_var = self.periods_movers, self.periods_variance
        # Store wage outcomes and groups
        Y1 = jdata.loc[:, 'y1'].to_numpy()
        Y2 = jdata.loc[:, 'y2'].to_numpy()
        Y3 = jdata.loc[:, 'y3'].to_numpy()
        Y4 = jdata.loc[:, 'y4'].to_numpy()
        J1 = jdata.loc[:, 'j1'].to_numpy()
        J2 = jdata.loc[:, 'j4'].to_numpy()
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

        # Dictionary linking periods to vectors
        C_dict = {period: C1 if period in self.first_periods else C2 for period in periods}

        ## Joint firm indicator ##
        nkG2 = nk * G2

        ## Compute log-likelihood ##
        # Log pdfs
        lp_adj_first = np.zeros(shape=(nk, nij, nl))
        lp_adj_second = np.zeros(shape=(nk, nij, nl))
        log_pk1 = np.log(pk1)
        if any_controls:
            ## Account for control variables ##
            A_sum, S_sum_sq = self._sum_by_non_nl(ni=nij, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)

            for l in range(nl):
                # Update A_sum/S_sum_sq to account for worker-interaction terms
                A_sum_l, S_sum_sq_l = self._sum_by_nl_l(ni=nij, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)
                ## Current firm classes ##
                lp1_curr = lognormpdf(
                    Y1 - R12 * (Y2 - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma'])),
                    A['12'][l, G1] + A_sum['12'] + A_sum_l['12'],
                    var=\
                        (S['12'][l, :] ** 2)[G1] + S_sum_sq['12'] + S_sum_sq_l['12']
                )
                lp4_curr = lognormpdf(
                    Y4 - R43 * (Y3 - (A['3ma'][l, G2] + A_sum['3ma'] + A_sum_l['3ma'])),
                    A['43'][l, G2] + A_sum['43'] + A_sum_l['43'],
                    var=\
                        (S['43'][l, :] ** 2)[G2] + S_sum_sq['43'] + S_sum_sq_l['43']
                )
                for k in range(nk):
                    ## New firm classes ##
                    lp1_adj = lognormpdf(
                        Y1 - R12 * (Y2 - (A['2ma'][l, k] + A_sum['2ma'] + A_sum_l['2ma'])),
                        A['12'][l, k] + A_sum['12'] + A_sum_l['12'],
                        var=\
                            (S['12'][l, :] ** 2)[k] + S_sum_sq['12'] + S_sum_sq_l['12']
                    )
                    lp2_adj_1 = lognormpdf(
                        Y2 - (A['2mb'][G2] + A_sum['2mb'] + A_sum_l['2mb']),
                        (A['2ma'][l, k] + A_sum['2ma'] + A_sum_l['2ma']),
                        var=\
                            ((S['2ma'][l, :] ** 2)[k] + S_sum_sq['2ma'] + S_sum_sq_l['2ma'])
                    )
                    lp2_adj_2 = lognormpdf(
                        Y2 - (A['2mb'][k] + A_sum['2mb'] + A_sum_l['2mb']),
                        (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']),
                        var=\
                            ((S['2ma'][l, :] ** 2)[G1] + S_sum_sq['2ma'] + S_sum_sq_l['2ma'])
                    )
                    lp3_adj_1 = lognormpdf(
                        Y3 - (A['3mb'][k] + A_sum['3mb'] + A_sum_l['3mb']) \
                            - R32m * (Y2 \
                                - (A['2ma'][l, k] + A_sum['2ma'] + A_sum_l['2ma']) \
                                - (A['2mb'][G2] + A_sum['2mb'] + A_sum_l['2mb'])),
                        A['3ma'][l, G2] + A_sum['3ma'] + A_sum_l['3ma'],
                        var=\
                            (S['3ma'][l, :] ** 2)[G2] + S_sum_sq['3ma'] + S_sum_sq_l['3ma']
                    )
                    lp3_adj_2 = lognormpdf(
                        Y3 - (A['3mb'][G1] + A_sum['3mb'] + A_sum_l['3mb']) \
                            - R32m * (Y2 \
                                - (A['2ma'][l, G1] + A_sum['2ma'] + A_sum_l['2ma']) \
                                - (A['2mb'][k] + A_sum['2mb'] + A_sum_l['2mb'])),
                        A['3ma'][l, k] + A_sum['3ma'] + A_sum_l['3ma'],
                        var=\
                            (S['3ma'][l, :] ** 2)[k] + S_sum_sq['3ma'] + S_sum_sq_l['3ma']
                    )
                    lp4_adj = lognormpdf(
                        Y4 - R43 * (Y3 - (A['3ma'][l, k] + A_sum['3ma'] + A_sum_l['3ma'])),
                        A['43'][l, k] + A_sum['43'] + A_sum_l['43'],
                        var=\
                            (S['43'][l, :] ** 2)[k] + S_sum_sq['43'] + S_sum_sq_l['43']
                    )
                    ## Log probability ##
                    lp_adj_first[k, :, l] = log_pk1[k + nkG2, l] + lp1_adj + lp2_adj_1 + lp3_adj_1 + lp4_curr
                    lp_adj_second[k, :, l] = log_pk1[G1 + nk * k, l] + lp1_curr + lp2_adj_2 + lp3_adj_2 + lp4_adj
        else:
            # Loop over firm classes so means/variances are single values rather than vectors (computing log/square is faster this way)
            for g1 in range(nk):
                for g2 in range(nk):
                    I = (G1 == g1) & (G2 == g2)
                    for l in range(nl):
                        ## Current firm classes ##
                        lp1_curr = lognormpdf(
                            Y1[I] \
                                - R12 * (Y2[I] - A['2ma'][l, g1]),
                            A['12'][l, g1],
                            var=S['12'][l, g1] ** 2
                        )
                        lp4_curr = lognormpdf(
                            Y4[I] \
                                - R43 * (Y3[I] - A['3ma'][l, g2]),
                            A['43'][l, g2],
                            var=S['43'][l, g2] ** 2
                        )
                        for k in range(nk):
                            ## New firm classes ##
                            lp1_adj = lognormpdf(
                                Y1[I] \
                                    - R12 * (Y2[I] - A['2ma'][l, k]),
                                A['12'][l, k],
                                var=S['12'][l, k] ** 2
                            )
                            lp2_adj_1 = lognormpdf(
                                Y2[I] \
                                    - A['2mb'][g2],
                                A['2ma'][l, k],
                                var=S['2ma'][l, k] ** 2
                            )
                            lp2_adj_2 = lognormpdf(
                                Y2[I] \
                                    - A['2mb'][k],
                                A['2ma'][l, g1],
                                var=S['2ma'][l, g1] ** 2
                            )
                            lp3_adj_1 = lognormpdf(
                                Y3[I] \
                                    - A['3mb'][k] \
                                    - R32m * (Y2[I] - (A['2ma'][l, k] + A['2mb'][g2])),
                                A['3ma'][l, g2],
                                var=S['3ma'][l, g2] ** 2
                            )
                            lp3_adj_2 = lognormpdf(
                                Y3[I] \
                                    - A['3mb'][g1] \
                                    - R32m * (Y2[I] - (A['2ma'][l, g1] + A['2mb'][k])),
                                A['3ma'][l, k],
                                var=S['3ma'][l, k] ** 2
                            )
                            lp4_adj = lognormpdf(
                                Y4[I] \
                                    - R43 * (Y3[I] - A['3ma'][l, k]),
                                A['43'][l, k],
                                var=S['43'][l, k] ** 2
                            )
                            ## Log probability ##
                            lp_adj_first[k, I, l] = log_pk1[k + nkG2[I], l] + lp1_adj + lp2_adj_1 + lp3_adj_1 + lp4_curr
                            lp_adj_second[k, I, l] = log_pk1[G1[I] + nk * k, l] + lp1_curr + lp2_adj_2 + lp3_adj_2 + lp4_adj
        del log_pk1, lp1_curr, lp4_curr, lp1_adj, lp4_adj, lp2_adj_1, lp2_adj_2, lp3_adj_1, lp3_adj_2

        ## Convert to log-sum-exp form ##
        lse_lp_adj_first = np.apply_along_axis(lambda a: logsumexp(a.reshape(nij, nl), axis=1), axis=1, arr=lp_adj_first.reshape(nk, nij * nl))
        lse_lp_adj_second = np.apply_along_axis(lambda a: logsumexp(a.reshape(nij, nl), axis=1), axis=1, arr=lp_adj_second.reshape(nk, nij * nl))

        ## Firm-level probabilities ##
        firm_level_lp_adj_first = np.apply_along_axis(lambda a: np.bincount(J1, a, minlength=nf), axis=1, arr=lse_lp_adj_first).T
        firm_level_lp_adj_second = np.apply_along_axis(lambda a: np.bincount(J2, a, minlength=nf), axis=1, arr=lse_lp_adj_second).T

        ### Stayers ###
        # Unpack parameters
        periods, periods_update = self.periods_stayers, self.periods_update_stayers

        # Store wage outcomes and groups
        Y1 = sdata['y1'].to_numpy()
        Y2 = sdata['y2'].to_numpy()
        Y3 = sdata['y3'].to_numpy()
        Y4 = sdata['y4'].to_numpy()
        J1 = sdata.loc[:, 'j1'].to_numpy()
        # J2 = sdata.loc[:, 'j4'].to_numpy()
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

        # Dictionary linking periods to vectors
        C_dict = {period: C1 if period in self.first_periods else C2 for period in periods}

        ## Compute log-likelihood ##
        # Log pdfs
        lp_adj = np.zeros(shape=(nk, nis, nl))
        log_pk0 = np.log(pk0)
        if any_controls:
            ## Account for control variables ##
            A_sum, S_sum_sq = self._sum_by_non_nl(ni=nis, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)

            for l in range(nl):
                # Update A_sum/S_sum_sq to account for worker-interaction terms
                A_sum_l, S_sum_sq_l = self._sum_by_nl_l(ni=nis, l=l, C_dict=C_dict, A_cat=A_cat, S_cat=S_cat, A_cts=A_cts, S_cts=S_cts, periods=periods)
                for k in range(nk):
                    ## New firm classes ##
                    lp1_adj = lognormpdf(
                        Y1 - R12 * (Y2 - (A['2ma'][l, k] + A_sum['2ma'] + A_sum_l['2ma'])),
                        A['12'][l, k] + A_sum['12'] + A_sum_l['12'],
                        var=\
                            (S['12'][l, :] ** 2)[k] + S_sum_sq['12'] + S_sum_sq_l['12']
                    )
                    lp2_adj = lognormpdf(
                        Y2,
                        A['2s'][l, k] + A_sum['2s'] + A_sum_l['2s'],
                        var=\
                            (S['2s'][l, :] ** 2)[k] + S_sum_sq['2s'] + S_sum_sq_l['2s']
                    )
                    lp3_adj = lognormpdf(
                        Y3 - R32s * (Y2 - (A['2s'][l, k] + A_sum['2s'] + A_sum_l['2s'])),
                        A['3s'][l, k] + A_sum['3s'] + A_sum_l['3s'],
                        var=\
                            (S['3s'][l, :] ** 2)[k] + S_sum_sq['3s'] + S_sum_sq_l['3s']
                    )
                    lp4_adj = lognormpdf(
                        Y4 - R43 * (Y3 - (A['3ma'][l, k] + A_sum['3ma'] + A_sum_l['3ma'])),
                        A['43'][l, k] + A_sum['43'] + A_sum_l['43'],
                        var=\
                            (S['43'][l, :] ** 2)[k] + S_sum_sq['43'] + S_sum_sq_l['43']
                    )
                    ## Log probability ##
                    lp_adj[k, :, l] = log_pk0[k, l] + lp1_adj + lp2_adj + lp3_adj + lp4_adj
        else:
            # Loop over firm classes so means/variances are single values rather than vectors (computing log/square is faster this way)
            for l in range(nl):
                for k in range(nk):
                    ## New firm classes ##
                    lp1_adj = lognormpdf(
                        Y1 - R12 * (Y2 - A['2ma'][l, k]),
                        A['12'][l, k],
                        var=S['12'][l, k] ** 2
                    )
                    lp2_adj = lognormpdf(
                        Y2,
                        A['2s'][l, k],
                        sd=S['2s'][l, k]
                    )
                    lp3_adj = lognormpdf(
                        Y3 - R32s * (Y2 - A['2s'][l, k]),
                        A['3s'][l, k],
                        var=S['3s'][l, k] ** 2
                    )
                    lp4_adj = lognormpdf(
                        Y4 - R43 * (Y3 - A['3ma'][l, k]),
                        A['43'][l, k],
                        var=S['43'][l, k] ** 2
                    )
                    ## Log probability ##
                    lp_adj[k, :, l] = log_pk0[k, l] + lp1_adj + lp2_adj + lp3_adj + lp4_adj
        del log_pk0, lp1_adj, lp2_adj, lp3_adj, lp4_adj

        ## Convert to log-sum-exp form ##
        lse_lp_adj = np.apply_along_axis(lambda a: logsumexp(a.reshape(nis, nl), axis=1), axis=1, arr=lp_adj.reshape(nk, nis * nl))

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
                warnings.warn("Linear algebra error encountered when computing connectedness measure. This can likely be corrected by increasing the value of 'd_prior_movers' in tw.dynamic_blm_params().")
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
            raise ValueError('The dynamic BLM estimation must be run on movers (and NNm must be computed) before plotting type flows.')

        if method not in ['stacked', 'sankey']:
            raise ValueError(f"`method` must be one of 'stacked' or 'sankey', but input specifies {method!r}.")

        ## Extract parameters ##
        nl, nk = self.nl, self.nk
        _, pk1, NNm = self._sort_parameters(self.A, pk1=self.pk1, NNm=self.NNm, sort_firm_types=True)

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
            jdata (BipartitePandas DataFrame): extended event study format labor data for movers
            iter (int): iteration
            blm_model (BLMModel or None): estimated static BLM model. Estimates from this model will be used as a baseline in half the starting values (the other half will be random). If None, all starting values will be random.
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        model = DynamicBLMModel(self.params, self.rho_0, rng)
        if iter % 4 == 0:
            # Constrained-unconstrained with static BLM as baseline
            model.fit_movers_cstr_uncstr(jdata, linear_additivity=True, blm_model=blm_model, initialize_all=(iter == 0))
        elif iter % 4 == 1:
            # Constrained-unconstrained with linear additivity
            model.fit_movers_cstr_uncstr(jdata, linear_additivity=True, blm_model=None, initialize_all=False)
        else:
            # Constrained-unconstrained without linear additivity
            model.fit_movers_cstr_uncstr(jdata, linear_additivity=False, blm_model=None, initialize_all=False)
        return model

    def fit(self, jdata, sdata, n_init=20, n_best=5, blm_model=None, rho_0=(0.6, 0.6, 0.6), weights=None, diff=False, ncore=1, rng=None):
        '''
        Estimate dynamic BLM using multiple sets of starting values.

        Arguments:
            jdata (BipartitePandas DataFrame): extended event study format labor data for movers
            sdata (BipartitePandas DataFrame): extended event study format labor data for stayers
            n_init (int): number of starting values
            n_best (int): take the n_best estimates with the highest likelihoods, and then take the estimate with the highest connectedness
            blm_model (BLMModel or None): estimated static BLM model. Estimates from this model will be used as a baseline in half the starting values (the other half will be random). If None, all starting values will be random.
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
        self.model.A['2s'] = self.model.A['2ma'].copy()
        self.model.A['3s'] = self.model.A['3ma'].copy()
        self.model.S['2s'] = self.model.S['2ma'].copy()
        self.model.S['3s'] = self.model.S['3ma'].copy()
        # ## Set pk0 based on pk1 ##
        # nl, nk, pk1, NNm = self.model.nl, self.model.nk, self.model.pk1, self.model.NNm
        # NNm_1 = np.sum(NNm, axis=1)
        # NNm_2 = np.sum(NNm, axis=0)
        # reshaped_pk1 = np.reshape(pk1, (nk, nk, nl))
        # pk_period1 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=1).T / NNm_1).T
        # pk_period2 = (np.sum((NNm.T * reshaped_pk1.T).T, axis=0).T / NNm_2).T
        # self.model.pk0 = (pk_period1 + pk_period2) / 2
        self.model.fit_stayers_cstr_uncstr(sdata)

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

class DynamicBLMBootstrap:
    '''
    Class for estimating dynamic BLM using bootstrapping.

    Arguments:
        params (ParamsDict): dictionary of parameters for dynamic BLM estimation. Run tw.dynamic_blm_params().describe_all() for descriptions of all valid parameters.
        model (DynamicBLMModel or None): estimated dynamic BLM model. For use with parametric bootstrap. None if running standard bootstrap.
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
            jdata (BipartitePandas DataFrame): extended event study format labor data for movers
            sdata (BipartitePandas DataFrame): extended event study format labor data for stayers
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
            ### Parametric bootstrap ###
            # Copy original wages and firm types
            yj = jdata.loc[:, ['y1', 'y2', 'y3', 'y4']].to_numpy().copy()
            ys = sdata.loc[:, ['y1', 'y2', 'y3', 'y4']].to_numpy().copy()
            gj = jdata.loc[:, ['g1', 'g4']].to_numpy().astype(int, copy=True)
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
                bdf = _simulate_types_wages(model, jdata, sdata, gj=gj, gs=gs, pk1=pk1, pk0=pk0, qi_cum_j=None, qi_cum_s=None, optimal_reallocation=False, reallocation_constraint_category=None, worker_types_as_ids=False, simulate_wages=True, return_long_df=True, store_worker_types=False, rng=rng)

                ## Cluster ##
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
                blm_fit_i.fit(jdata=jdata, sdata=sdata, n_init=n_init_estimator, n_best=n_best, blm_model=None, ncore=ncore, rng=rng)
                models.append(blm_fit_i.model)
                del blm_fit_i

            with bpd.util.ChainedAssignment():
                # Re-assign original wages and firm types
                jdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = yj
                sdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = ys
                jdata.loc[:, ['g1', 'g4']], jdata.loc[:, ['g2', 'g3']] = (gj, gj)
                sdata.loc[:, 'g1'], sdata.loc[:, 'g2'], sdata.loc[:, 'g3'], sdata.loc[:, 'g4'] = (gs, gs, gs, gs)
        elif method == 'standard':
            ### Standard bootstrap ###
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
                blm_fit_i.fit(jdata=jdata_i, sdata=sdata_i, n_init=n_init_estimator, n_best=n_best, blm_model=None, ncore=ncore, rng=rng)
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
                _, pk1, _, NNm, _ = model._sort_parameters(model.A, pk1=model.pk1, pk0=model.pk0, NNm=model.NNm, NNs=model.NNs, sort_firm_types=True)
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

class DynamicBLMVarianceDecomposition:
    '''
    Class for estimating dynamic BLM variance decomposition using bootstrapping. Results are stored in class attribute .res, which gives a dictionary where the key 'var_decomp' gives the results for the variance decomposition, and the key 'var_decomp_comp' optionally gives the results for the variance decomposition with complementarities.

    Arguments:
        params (ParamsDict): dictionary of parameters for dynamic BLM estimation. Run tw.dynamic_blm_params().describe_all() for descriptions of all valid parameters.
        model (DynamicBLMModel or None): estimated dynamic BLM model
    '''

    def __init__(self, params, model):
        self.params = params
        self.model = model
        # No initial results
        self.res = None

    def fit(self, jdata, sdata, n_samples=5, reallocate=False, reallocate_jointly=True, reallocate_period='first', Q_var=None, Q_cov=None, complementarities=True, firm_clusters_as_ids=True, worker_types_as_ids=True, ncore=1, rng=None):
        '''
        Estimate variance decomposition.

        Arguments:
            jdata (BipartitePandas DataFrame): extended event study format labor data for movers
            sdata (BipartitePandas DataFrame): extended event study format labor data for stayers
            n_samples (int): number of bootstrap samples to estimate
            reallocate (bool): if True, draw worker type proportions independently of firm type; if False, uses worker type proportions that are conditional on firm type
            reallocate_jointly (bool): if True, worker type proportions take the average over movers and stayers (i.e. all workers use the same type proportions); if False, consider movers and stayers separately
            reallocate_period (str): if 'first', compute type proportions based on first period parameters; if 'second', compute type proportions based on second period parameters; if 'all', compute type proportions based on average over first and second period parameters
            Q_var (list of Q variances): list of Q matrices to use when estimating variance term; None is equivalent to tw.Q.VarPsi() without controls, or tw.Q.VarCovariate('psi') with controls
            Q_cov (list of Q covariances): list of Q matrices to use when estimating covariance term; None is equivalent to tw.Q.CovPsiAlpha() without controls, or tw.Q.CovCovariate('psi', 'alpha') with controls
            complementarities (bool): if True, also estimate R^2 of regression with complementarities (by adding in all worker-firm interactions). Only allowed when firm_clusters_as_ids=True and worker_types_as_ids=True.
            firm_clusters_as_ids (bool): if True, replace firm ids with firm clusters
            worker_types_as_ids (bool): if True, replace worker ids with simulated worker types
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
        gj = jdata.loc[:, ['g1', 'g4']].to_numpy().astype(int, copy=True)
        gs = sdata.loc[:, 'g1'].to_numpy().astype(int, copy=True)
        if firm_clusters_as_ids:
            jj = jdata.loc[:, ['j1', 'j4']].to_numpy().copy()
            js = sdata.loc[:, 'j1'].to_numpy().copy()
            with bpd.util.ChainedAssignment():
                jdata.loc[:, ['j1', 'j4']], jdata.loc[:, ['j2', 'j3']] = (gj, gj)
                sdata.loc[:, 'j1'], sdata.loc[:, 'j2'], sdata.loc[:, 'j3'], sdata.loc[:, 'j4'] = (gs, gs, gs, gs)
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
            bdf = _simulate_types_wages(model, jdata, sdata, gj=gj, gs=gs, pk1=pk1, pk0=pk0, qi_cum_j=None, qi_cum_s=None, optimal_reallocation=False, reallocation_constraint_category=None, worker_types_as_ids=worker_types_as_ids, simulate_wages=True, return_long_df=True, store_worker_types=False, rng=rng)

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
            jdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = yj
            sdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = ys
            if firm_clusters_as_ids:
                jdata.loc[:, ['j1', 'j4']], jdata.loc[:, ['j2', 'j3']] = (jj, jj)
                sdata.loc[:, 'j1'], sdata.loc[:, 'j2'], sdata.loc[:, 'j3'], sdata.loc[:, 'j4'] = (js, js, js, js)
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
    Class for estimating dynamic BLM reallocation exercise using bootstrapping. Results are stored in class attribute .res, which gives a dictionary with the following structure: baseline results are stored in key 'baseline'. Reallocation results are stored in key 'reallocation'. Within each sub-dictionary, primary outcome results are stored in the key 'outcome', means are stored in the key 'mean', categorical results are stored in the key 'cat', continuous results are stored in the key 'cts', type proportions for movers are stored in the key 'pk1', type proportions for stayers are stored in the key 'pk0', firm-level mover flow counts are stored in the key 'NNm', and firm-level stayer counts are stored in the key 'NNs'.

    Arguments:
        model (DynamicBLMModel or None): estimated dynamic BLM model
    '''

    def __init__(self, model):
        self.model = model
        # No initial results
        self.res = None

    def fit(self, jdata, sdata, quantiles=None, n_samples=5, reallocate_jointly=True, reallocate_period='first', categorical_sort_cols=None, continuous_sort_cols=None, unresidualize_col=None, optimal_reallocation=False, reallocation_constraint_category=None, reallocation_scaling_col=None, qi_j=None, qi_s=None, qi_cum_j=None, qi_cum_s=None, ncore=1, rng=None):
        '''
        Estimate variance decomposition.

        Arguments:
            jdata (BipartitePandas DataFrame): extended event study format labor data for movers
            sdata (BipartitePandas DataFrame): extended event study format labor data for stayers
            quantiles (NumPy Array or None): income quantiles to compute; if None, computes percentiles from 1-100 (specifically, np.arange(101) / 100)
            n_samples (int): number of bootstrap samples to estimate
            reallocate_jointly (bool): if True, worker type proportions take the average over movers and stayers (i.e. all workers use the same type proportions); if False, consider movers and stayers separately
            reallocate_period (str): if 'first', compute type proportions based on first period parameters; if 'second', compute type proportions based on second period parameters; if 'all', compute type proportions based on average over first and second period parameters
            categorical_sort_cols (dict or None): in addition to standard quantiles results, return average income grouped by the alternative column(s) given (which are represented by the dictionary {column: number of quantiles to compute}). For categorical variables, use each group as a bin and take the average income within that bin. None is equivalent to {}.
            continuous_sort_cols (dict or None): in addition to standard quantiles results, return average income grouped by the alternative column(s) given (which are represented by the dictionary {column: list of quantiles to compute}). For continuous variables, create bins based on the list of quantiles given in the dictionary. The list of quantiles must start at 0 and end at 1. None is equivalent to {}.
            unresidualize_col (str or None): column with predicted values that are residualized out, which will be added back in before computing outcomes in order to unresidualize the values; None leaves outcome unchanged
            optimal_reallocation (bool or str): if not False, reallocate workers to new firms to maximize ('max') or minimize ('min') total output
            reallocation_constraint_category (str or None): specify categorical column to constrain reallocation so that workers must reallocate within their own category; if None, no constraints on how workers can reallocate
            reallocation_scaling_col (str or None): specify column to use to scale outcomes when computing optimal reallocation (i.e. multiply outcomes by an observation-level factor); if None, don't scale outcomes
            qi_j (NumPy Array or None): (use with optimal_reallocation to assign workers to maximum probability worker type based on observation-level probabilities) probabilities for each mover observation to be each worker type; None if pk1 or qi_cum_j is not None
            qi_s (NumPy Array or None): (use with optimal_reallocation to assign workers to maximum probability worker type based on observation-level probabilities) probabilities for each stayer observation to be each worker type; None if pk0 or qi_cum_s is not None
            qi_cum_j (NumPy Array or None): (use with optimal_reallocation to assign workers to worker types probabilistically based on observation-level probabilities) cumulative probabilities for each mover observation to be each worker type; None if pk1 or qi_j is not None
            qi_cum_s (NumPy Array or None): (use with optimal_reallocation to assign workers to worker types probabilistically based on observation-level probabilities) cumulative probabilities for each stayer observation to be each worker type; None if pk0 or qi_s is not None
            ncore (int): number of cores for multiprocessing
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
        yj = jdata.loc[:, ['y1', 'y2', 'y3', 'y4']].to_numpy().copy()
        ys = sdata.loc[:, ['y1', 'y2', 'y3', 'y4']].to_numpy().copy()
        jj = jdata.loc[:, ['j1', 'j4']].to_numpy().copy()
        js = sdata.loc[:, 'j1'].to_numpy().copy()
        gj = jdata.loc[:, ['g1', 'g4']].to_numpy().astype(int, copy=True)
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
        # Compute quantiles (no weights for dynamic BLM)
        y = bdf.loc[:, 'y'].to_numpy()
        if unresidualize_col is not None:
            y += bdf.loc[:, unresidualize_col]
        res_baseline = weighted_quantile(values=y, quantiles=quantiles)
        mean_baseline = weighted_mean(y)
        if reallocation_scaling_col is not None:
            scaling_col = to_list(bdf.col_reference_dict[reallocation_scaling_col])[0]
            scale = bdf.loc[:, scaling_col].to_numpy()
            y_scaled = scale * y
            res_scaled_baseline = weighted_quantile(values=y_scaled, quantiles=quantiles)
            mean_scaled_baseline = weighted_mean(y_scaled)
        for col_cat in categorical_sort_cols.keys():
            ## Categorical sorting variables ##
            col = bdf.loc[:, col_cat].to_numpy()
            # Use categories as bins
            res_cat_baseline[col_cat] =\
                np.bincount(col, weights=y) / np.bincount(col)
            if reallocation_scaling_col is not None:
                res_scaled_cat_baseline[col_cat] =\
                    np.bincount(col, weights=y_scaled) / np.bincount(col)
        for col_cts, quantiles_cts in continuous_sort_cols.items():
            ## Continuous sorting variables ##
            col = bdf.loc[:, col_cts].to_numpy()
            # Create bins based on quantiles
            col_quantiles = weighted_quantile(values=col, quantiles=quantiles_cts)
            quantile_groups = pd.cut(col, col_quantiles, include_lowest=True).codes
            res_cts_baseline[col_cts] =\
                np.bincount(quantile_groups, weights=y) / np.bincount(quantile_groups)
            if reallocation_scaling_col is not None:
                res_scaled_cts_baseline[col_cts] =\
                    np.bincount(quantile_groups, weights=y_scaled) / np.bincount(quantile_groups)

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
            bdf = _simulate_types_wages(model, jdata, sdata, gj=gj, gs=gs, pk1=pk1, pk0=pk0, qi_j=qi_j, qi_s=qi_s, qi_cum_j=qi_cum_j, qi_cum_s=qi_cum_s, optimal_reallocation=optimal_reallocation, reallocation_constraint_category=reallocation_constraint_category, reallocation_scaling_col=reallocation_scaling_col, worker_types_as_ids=False, simulate_wages=True, return_long_df=True, store_worker_types=True, rng=rng)

            ## Compute quantiles (no weights for dynamic BLM) ##
            y = bdf.loc[:, 'y'].to_numpy()
            if unresidualize_col is not None:
                y += bdf.loc[:, unresidualize_col]
            res[i, :] = weighted_quantile(values=y, quantiles=quantiles)
            mean[i] = weighted_mean(y)
            if reallocation_scaling_col is not None:
                scale = bdf.loc[:, scaling_col].to_numpy()
                y_scaled = scale * y
                res_scaled[i, :] = weighted_quantile(values=y_scaled, quantiles=quantiles)
                mean_scaled[i] = weighted_mean(y_scaled)
            for col_cat in categorical_sort_cols.keys():
                ## Categorical sorting variables ##
                col = bdf.loc[:, col_cat].to_numpy()
                # Use categories as bins
                res_cat[col_cat][i, :] =\
                    np.bincount(col, weights=y) / np.bincount(col)
                if reallocation_scaling_col is not None:
                    res_scaled_cat[col_cat][i, :] =\
                        np.bincount(col, weights=y_scaled) / np.bincount(col)
            for col_cts, quantiles_cts in continuous_sort_cols.items():
                ## Continuous sorting variables ##
                col = bdf.loc[:, col_cts].to_numpy()
                # Create bins based on quantiles
                col_quantiles = weighted_quantile(values=col, quantiles=quantiles_cts)
                quantile_groups = pd.cut(col, col_quantiles, include_lowest=True).codes
                res_cts[col_cts][i, :] =\
                    np.bincount(quantile_groups, weights=y) / np.bincount(quantile_groups)
                if reallocation_scaling_col is not None:
                    res_scaled_cts[col_cts][i, :] =\
                        np.bincount(quantile_groups, weights=y_scaled) / np.bincount(quantile_groups)

            ## Compute type proportions ##
            bdf = bdf.to_extendedeventstudy(is_sorted=True, copy=False)
            # NOTE: unweighted
            m = (bdf.loc[:, 'm'] > 0)

            if not optimal_reallocation:
                # Compute pk1
                movers = bdf.loc[m, ['g1', 'g4', 'l']].to_numpy()
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
            jdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = yj
            sdata.loc[:, ['y1', 'y2', 'y3', 'y4']] = ys
            jdata.loc[:, ['j1', 'j3']], jdata.loc[:, ['j2', 'j4']] = (jj, jj)
            sdata.loc[:, 'j1'], sdata.loc[:, 'j2'], sdata.loc[:, 'j3'], sdata.loc[:, 'j4'] = (js, js, js, js)
            jdata.loc[:, ['g1', 'g3']], jdata.loc[:, ['g2', 'g4']] = (gj, gj)
            sdata.loc[:, 'g1'], sdata.loc[:, 'g2'], sdata.loc[:, 'g3'], sdata.loc[:, 'g4'] = (gs, gs, gs, gs)

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
                A, pk1, pk0, NNm, NNs = model._sort_parameters(model.A, pk1=pk1_res[i, :, :], pk0=pk0_res[i, :, :], NNm=NNm_res[i, :, :], NNs=NNs_res[i, :], sort_firm_types=True)

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

class DynamicBLMTransitions:
    '''
    Class for estimating dynamic BLM transition probability exercise using bootstrapping. Results are stored in class attribute .res, which gives a 4-D NumPy Array where the first dimension gives each particular simulation; the second dimension gives the subset of data considered (index 0 gives the full data; index 1 gives the first conditional decile of earnings; and index 2 gives the tenth conditional decile of earnings); the third dimension gives the starting group of clusters being considered; and the fourth dimension gives the destination group of clusters being considered (where the first index of the fourth dimension considers all destinations).

    Arguments:
        model (DynamicBLMModel or None): estimated dynamic BLM model
    '''

    def __init__(self, model):
        self.model = model
        # No initial results
        self.res = None

    def fit(self, jdata, sdata, cluster_groups=None, n_samples=5, cluster_params=None, ncore=1, rng=None):
        '''
        Estimate transition probabilities.

        Arguments:
            jdata (BipartitePandas DataFrame): extended event study format labor data for movers
            sdata (BipartitePandas DataFrame): extended event study format labor data for stayers
            cluster_groups (list of lists or None): how to group firm clusters, where each element of the primary list gives a list of firm clusters in the corresponding group; if None, tries to divide firms in 3 evenly sized groups
            n_samples (int): number of bootstrap samples to estimate
            cluster_params (ParamsDict or None): dictionary of parameters for clustering firms. Run bpd.cluster_params().describe_all() for descriptions of all valid parameters. None is equivalent to bpd.cluster_params().
            ncore (int): number of cores for multiprocessing
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        ## Unpack parameters ##
        model = self.model
        nl, nk = model.nl, model.nk
        pk1, pk0 = model.pk1, model.pk0

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
        gj = jdata.loc[:, ['g1', 'g4']].to_numpy().astype(int, copy=True)
        gs = sdata.loc[:, 'g1'].to_numpy().astype(int, copy=True)
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

        ## Run bootstrap ##
        res = np.zeros([n_samples, 3, len(cluster_groups), len(cluster_groups) + 1])
        for i in trange(n_samples):
            ## Simulate worker types and wages ##
            Lm_i, Ls_i, yj_i, ys_i = _simulate_types_wages(model, jdata, sdata, gj=gj, gs=gs, pk1=pk1, pk0=pk0, qi_cum_j=None, qi_cum_s=None, optimal_reallocation=False, reallocation_constraint_category=None, worker_types_as_ids=False, simulate_wages=True, return_long_df=False, store_worker_types=False, rng=rng)

            with bpd.util.ChainedAssignment():
                ## Update jdata and sdata ##
                jdata.loc[:, 'y1'], jdata.loc[:, 'y2'], jdata.loc[:, 'y3'], jdata.loc[:, 'y4'] = (yj_i[0], yj_i[1], yj_i[2], yj_i[3])
                sdata.loc[:, 'y1'], sdata.loc[:, 'y2'], sdata.loc[:, 'y3'], sdata.loc[:, 'y4'] = (ys_i[0], ys_i[1], ys_i[2], ys_i[3])
                jdata.loc[:, 'i'] = Lm_i
                sdata.loc[:, 'i'] = Ls_i

            ## Convert to BipartitePandas DataFrame ##
            bdf = bpd.BipartiteDataFrame(pd.concat([jdata, sdata], axis=0, copy=False))
            # Set attributes from jdata, so that conversion to long works (since pd.concat drops attributes)
            bdf._set_attributes(jdata)

            # Replace i to ensure unstacking works properly
            i_orig = bdf.loc[:, 'i'].to_numpy().copy()
            with bpd.util.ChainedAssignment():
                bdf.loc[:, 'i'] = np.arange(len(bdf))

            ## Cluster ##
            bdf = bdf.cluster(cluster_params, rng=rng)
            with bpd.util.ChainedAssignment():
                # Restore i
                bdf.loc[:, 'i'] = i_orig

            ## Compute conditional earnings deciles ##
            first_decile = np.zeros(len(bdf), dtype=int)
            tenth_decile = np.zeros(len(bdf), dtype=int)
            for l in range(nl):
                for k in range(nk):
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
