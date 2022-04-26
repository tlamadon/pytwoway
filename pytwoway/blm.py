'''
We implement the non-linear estimator from Bonhomme Lamadon & Manresa
'''

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.sparse import csc_matrix, diags
from scipy.stats import norm
from matplotlib import pyplot as plt
from multiprocessing import Pool
import itertools
import warnings
from pytwoway import jitter_scatter
from pytwoway import constraints as cons
from bipartitepandas.util import ParamsDict, to_list, _is_subtype
from tqdm import tqdm

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
_blm_params_default = ParamsDict({
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
            (default=None) Dictionary linking column names to instances of tw.categorical_control_params(). Each instance specifies a new categorical control variable and how its starting values should be generated. Run tw.categorical_control_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
        ''', None),
    'continuous_controls': (None, 'dict_of_type_none', ParamsDict,
        '''
            (default=None) Dictionary linking column names to instances of tw.continuous_control_params(). Each instance specifies a new continuous control variable and how its starting values should be generated. Run tw.continuous_control_params().describe_all() for descriptions of all valid parameters for simulating each control variable. None is equivalent to {}.
        ''', None),
    ## Starting values ##
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
    'verbose': (0, 'set', [0, 1, 2],
        '''
            (default=0) If 0, print no output; if 1, print additional output; if 2, print maximum output.
        ''', None),
    ## fit_movers() and fit_stayers() parameters ##
    'weighted': (True, 'type', bool,
        '''
            (default=True) If True, run estimator with weights. These comes from data columns 'w1' and 'w2'.
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
            (default=None) Constraint object or list of constraint objects with class method .get_constraints() that defines constraints on A1 and A2. None is equivalent to [].
        ''', None),
    'cons_s': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects with class method .get_constraints() that defines constraints on S1 and S2. None is equivalent to [].
        ''', None),
    'cons_a_all': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects with class method .get_constraints() that defines constraints on A1/A2/A1_cat/A2_cat/A1_cts/A2_cts. None is equivalent to [].
        ''', None),
    'cons_s_all': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects with class method .get_constraints() that defines constraints on S1/S2/S1_cat/S2_cat/S1_cts/S2_cts. None is equivalent to [].
        ''', None),
    's_lower_bound': (1e-10, 'type_constrained', ((float, int), _gt0),
        '''
            (default=1e-10) Lower bound on estimated S1/S2/S1_cat/S2_cat/S1_cts/S2_cts.
        ''', '> 0'),
    'd_prior_movers': (1 + 1e-7, 'type_constrained', ((float, int), _gteq1),
        '''
            (default=1 + 1e-7) Account for probabilities being too small by adding (d_prior - 1) to pk1.
        ''', '>= 1'),
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

def blm_params(update_dict=None):
    '''
    Dictionary of default blm_params. Run tw.blm_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict or None): user parameter values; None is equivalent to {}

    Returns:
        (ParamsDict) dictionary of blm_params
    '''
    new_dict = _blm_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

_categorical_control_params_default = ParamsDict({
    'n': (None, 'type_constrained_none', (int, _gteq2),
        '''
            (default=6) Number of types for the parameter. None will raise an error when running the estimator.
        ''', '>= 2'),
    'a1_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A1_cat (mean of fixed effects in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A1_cat (mean of fixed effects in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A2_cat (mean of fixed effects in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2_cat (mean of fixed effects in second period).
        ''', '>= 0'),
    's1_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S1_cat (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's1_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S1_cat (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's2_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2_cat (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    's2_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2_cat (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    'worker_type_interaction': (False, 'type', bool,
        '''
            (default=False) If True, effect can differ by worker type.
        ''', None),
    'cons_a': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects with class method .get_constraints() that defines constraints on A1 and A2. None is equivalent to [].
        ''', None),
    'cons_s': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects with class method .get_constraints() that defines constraints on S1 and S2. None is equivalent to [].
        ''', None)
})

def categorical_control_params(update_dict=None):
    '''
    Dictionary of default categorical_control_params. Run tw.categorical_control_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict or None): user parameter values; None is equivalent to {}

    Returns:
        (ParamsDict) dictionary of categorical_control_params
    '''
    new_dict = _categorical_control_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

_continuous_control_params_default = ParamsDict({
    'a1_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A1_cts (mean of coefficient in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A1_cts (mean of coefficient in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of starting values for A2_cts (mean of coefficient in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of starting values for A2_cts (mean of coefficient in second period).
        ''', '>= 0'),
    's1_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S1_cts (standard deviation of coefficient in first period).
        ''', '>= 0'),
    's1_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S1_cts (standard deviation of coefficient in first period).
        ''', '>= 0'),
    's2_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of starting values for S2_cts (standard deviation of coefficient in second period).
        ''', '>= 0'),
    's2_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of starting values for S2_cts (standard deviation of coefficient in second period).
        ''', '>= 0'),
    'worker_type_interaction': (False, 'type', bool,
        '''
            (default=False) If True, effect can differ by worker type.
        ''', None),
    'cons_a': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects with class method .get_constraints() that defines constraints on A1 and A2. None is equivalent to [].
        ''', None),
    'cons_s': (None, 'list_of_type_none', (cons.Linear, cons.Monotonic, cons.Stationary, cons.StationaryFirmTypeVariation, cons.BoundedBelow, cons.BoundedAbove),
        '''
            (default=None) Constraint object or list of constraint objects with class method .get_constraints() that defines constraints on S1 and S2. None is equivalent to [].
        ''', None)
})

def continuous_control_params(update_dict=None):
    '''
    Dictionary of default continuous_control_params. Run tw.continuous_control_params().describe_all() for descriptions of all valid parameters.

    Arguments:
        update_dict (dict or None): user parameter values; None is equivalent to {}

    Returns:
        (ParamsDict) dictionary of continuous_control_params
    '''
    new_dict = _continuous_control_params_default.copy()
    if update_dict is not None:
        new_dict.update(update_dict)
    return new_dict

def lognormpdf(x, mu, sd):
    return - 0.5 * np.log(2 * np.pi) - np.log(sd) - (x - mu) ** 2 / (2 * sd ** 2)

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
        self.params = params
        self.rng = rng
        nl, nk = self.params.get_multiple(('nl', 'nk'))
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
        self.any_controls = len(control_cols) > 0
        # Check if any control variables interact with worker type
        self.any_worker_type_interactions = any([col_dict['worker_type_interaction'] for col_dict in controls_dict.values()])
        # Check if any control variables don't interact with worker type
        self.any_non_worker_type_interactions = any([not col_dict['worker_type_interaction'] for col_dict in controls_dict.values()])

        ## Generate starting values ##
        a1_mu, a1_sig, a2_mu, a2_sig, s1_low, s1_high, s2_low, s2_high, pk1_prior, pk0_prior = self.params.get_multiple(('a1_mu', 'a1_sig', 'a2_mu', 'a2_sig', 's1_low', 's1_high', 's2_low', 's2_high', 'pk1_prior', 'pk0_prior'))
        # Model for Y1 | Y2, l, k for movers and stayers
        self.A1 = rng.normal(loc=a1_mu, scale=a1_sig, size=dims)
        self.S1 = rng.uniform(low=s1_low, high=s1_high, size=dims)
        # Model for Y4 | Y3, l, k for movers and stayers
        self.A2 = rng.normal(loc=a2_mu, scale=a2_sig, size=dims)
        self.S2 = rng.uniform(low=s2_low, high=s2_high, size=dims)
        # Model for p(K | l, l') for movers
        if pk1_prior is None:
            pk1_prior = np.ones(nl)
        self.pk1 = rng.dirichlet(alpha=pk1_prior, size=nk ** 2)
        # Model for p(K | l, l') for stayers
        if pk0_prior is None:
            pk0_prior = np.ones(nl)
        self.pk0 = rng.dirichlet(alpha=pk0_prior, size=nk)

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
                rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=(nl, controls_dict[col]['n']))
                if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=controls_dict[col]['n'])
            for col in cat_cols}
        self.S2_cat = {col:
                rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=(nl, controls_dict[col]['n']))
                if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=controls_dict[col]['n'])
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
                rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=nl)
                if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=controls_dict[col]['s1_low'], high=controls_dict[col]['s1_high'], size=1)
            for col in cts_cols}
        self.S2_cts = {col:
                rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=nl)
                if controls_dict[col]['worker_type_interaction'] else
                rng.uniform(low=controls_dict[col]['s2_low'], high=controls_dict[col]['s2_high'], size=1)
            for col in cts_cols}
        # # Stationary #
        # for col in cts_cols:
        #     if controls_dict[col]['stationary_A']:
        #         self.A2_cts[col] = self.A1_cts[col]
        #     if controls_dict[col]['stationary_S']:
        #         self.S2_cts[col] = self.S1_cts[col]

        for l in range(nl):
            self.A1[l] = np.sort(self.A1[l], axis=0)
            self.A2[l] = np.sort(self.A2[l], axis=0)

        # if self.fixb:
        #     self.A2 = np.mean(self.A2, axis=1) + self.A1 - np.mean(self.A1, axis=1)

        # if self.stationary:
        #     self.A2 = self.A1

    # def reset_params(self):
    #     nl = self.nl
    #     nk = self.nk
    #     # Model for Y1 | Y2, l, k for movers and stayers
    #     self.A1 = np.tile(sorted(rng.normal(size=nl)), (nk, 1))
    #     self.S1 = np.ones(shape=(nk, nl))
    #     # Model for Y4 | Y3, l, k for movers and stayers
    #     self.A2 = self.A1.copy()
    #     self.S2 = np.ones(shape=(nk, nl))
    #     # Model for p(K | l, l') for movers
    #     self.pk1 = np.ones(shape=(nk * nk, nl)) / nl
    #     # Model for p(K | l, l') for stayers
    #     self.pk0 = np.ones(shape=(nk, nl)) / nl

    def _sum_by_non_nl(self, ni, C1, C2, compute_A=True, compute_S=True):
        '''
        Compute A1_sum/A2_sum/S1_sum_sq/S2_sum_sq for non-worker-interaction terms.

        Arguments:
            ni (int): number of observations
            C1 (dict of NumPy Arrays): dictionary linking column names to control variable data for the first period
            C2 (dict of NumPy Arrays): dictionary linking column names to control variable data for the second period
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

        A1_cat, A2_cat, S1_cat, S2_cat = self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat
        A1_cts, A2_cts, S1_cts, S2_cts = self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts
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
                    S1_sum_sq += S1_cat[col][C1[col]] ** 2
                    S2_sum_sq += S2_cat[col][C2[col]] ** 2
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

    def _sum_by_nl_l(self, ni, l, C1, C2, compute_A=True, compute_S=True):
        '''
        Compute A1_sum/A2_sum/S1_sum_sq/S2_sum_sq to account for worker-interaction terms for a particular worker type.

        Arguments:
            ni (int): number of observations
            l (int): worker type (must be in range(0, nl))
            C1 (dict of NumPy Arrays): dictionary linking column names to control variable data for the first period
            C2 (dict of NumPy Arrays): dictionary linking column names to control variable data for the second period
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

        A1_cat, A2_cat, S1_cat, S2_cat = self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat
        A1_cts, A2_cts, S1_cts, S2_cts = self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts
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
                    S1_sum_sq_l += S1_cat[col][l, C1[col]] ** 2
                    S2_sum_sq_l += S2_cat[col][l, C2[col]] ** 2
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

    def fit_movers(self, jdata, normalize=True, compute_NNm=True):
        '''
        EM algorithm for movers.

        Arguments:
            jdata (BipartitePandas DataFrame): movers
            normalize (bool): if True and using categorical controls, normalize the lowest firm-worker pair to have effect 0
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
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
        # Control variables
        C1 = {}
        C2 = {}
        for col in cat_cols:
            # Get subcolumns associated with col
            subcols = to_list(jdata.col_reference_dict[col])
            n_subcols = len(subcols)
            if n_subcols == 1:
                # If column is constant over time
                C1[col] = jdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
                C2[col] = jdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
            elif n_subcols == 2:
                # If column can change over time
                C1[col] = jdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
                C2[col] = jdata.loc[:, subcols[1]].to_numpy().astype(int, copy=False)
            else:
                raise NotImplementedError(f'Column names must have either one or two associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
        for col in cts_cols:
            # Get subcolumns associated with col
            subcols = to_list(jdata.col_reference_dict[col])
            n_subcols = len(subcols)
            if n_subcols == 1:
                # If column is constant over time
                C1[col] = jdata.loc[:, subcols[0]].to_numpy()
                C2[col] = jdata.loc[:, subcols[0]].to_numpy()
            elif n_subcols == 2:
                # If column can change over time
                C1[col] = jdata.loc[:, subcols[0]].to_numpy()
                C2[col] = jdata.loc[:, subcols[1]].to_numpy()
            else:
                raise NotImplementedError(f'Column names must have either one or two associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
        ## Sparse matrix representations ##
        GG1 = csc_matrix((np.ones(ni), (range(ni), G1)), shape=(ni, nk))
        GG2 = csc_matrix((np.ones(ni), (range(ni), G2)), shape=(ni, nk))
        CC1 = {col: csc_matrix((np.ones(ni), (range(ni), C1[col])), shape=(ni, controls_dict[col]['n'])) for col in cat_cols}
        CC2 = {col: csc_matrix((np.ones(ni), (range(ni), C2[col])), shape=(ni, controls_dict[col]['n'])) for col in cat_cols}

        # Transition probability matrix
        GG12 = csc_matrix((np.ones(ni), (range(ni), G1 + nk * G2)), shape=(ni, nk ** 2))

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

        ### Constraints ###
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

        for iter in range(params['n_iters_movers']):

            # ---------- E-Step ----------
            # We compute the posterior probabilities for each row
            # We iterate over the worker types, should not be be
            # too costly since the vector is quite large within each iteration
            if any_controls > 0:
                ## Account for control variables ##
                if iter == 0:
                    A1_sum, A2_sum, S1_sum_sq, S2_sum_sq = self._sum_by_non_nl(ni=ni, C1=C1, C2=C2)
                else:
                    S1_sum_sq, S2_sum_sq = self._sum_by_non_nl(ni=ni, C1=C1, C2=C2, compute_A=False)

                KK = G1 + nk * G2
                for l in range(nl):
                    # Update A1_sum/A2_sum/S1_sum_sq/S2_sum_sq to account for worker-interaction terms
                    A1_sum_l, A2_sum_l, S1_sum_sq_l, S2_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2)
                    lp1 = lognormpdf(Y1, A1_sum + A1_sum_l + A1[l, G1], np.sqrt(S1_sum_sq + S1_sum_sq_l + S1[l, G1] ** 2))
                    lp2 = lognormpdf(Y2, A2_sum + A2_sum_l + A2[l, G2], np.sqrt(S2_sum_sq + S2_sum_sq_l + S2[l, G2] ** 2))
                    lp[:, l] = np.log(pk1[KK, l]) + lp1 + lp2
            else:
                KK = G1 + nk * G2
                for l in range(nl):
                    lp1 = lognormpdf(Y1, A1[l, G1], S1[l, G1])
                    lp2 = lognormpdf(Y2, A2[l, G2], S2[l, G2])
                    lp[:, l] = np.log(pk1[KK, l]) + lp1 + lp2
            lp = (lp.T + np.log(W1) + np.log(W2)).T
            del lp1, lp2

            # We compute log sum exp to get likelihoods and probabilities
            qi = np.exp(lp.T - logsumexp(lp, axis=1)).T
            # # Add dirichlet prior
            # qi += d_prior - 1
            # # Normalize rows to sum to 1
            # qi = (qi.T / np.sum(qi, axis=1).T).T
            if params['return_qi']:
                return qi
            lik1 = logsumexp(lp, axis=1).mean() # FIXME should this be returned?
            # lik_prior = (d_prior - 1) * np.sum(np.log(pk1))
            # lik1 += lik_prior
            liks1.append(lik1)
            if params['verbose'] == 2:
                print('loop {}, liks {}'.format(iter, lik1))

            if abs(lik1 - prev_lik) < params['threshold_movers']:
                break
            prev_lik = lik1

            # ---------- M-step ----------
            # For now we run a simple ols, however later we
            # want to add constraints!
            # see https://scaron.info/blog/quadratic-programming-in-python.html

            # The regression has 2 * nl * nk parameters and nl * ni rows
            # We do not necessarily want to construct the duplicated data by nl
            # Instead we will construct X'X and X'Y by looping over nl
            # We also note that X'X is block diagonal with 2*nl matrices of dimensions nk^2
            ## General ##
            # Shift for period 2
            ts = nl * nk
            # Only store the diagonal
            XwXd = np.zeros(shape=2 * ts)
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
            for l in range(nl):
                # (We might be better off trying this within numba or something)
                l_index, r_index = l * nk, (l + 1) * nk
                # Shared weighted terms
                GG1_weighted = GG1.T @ diags(W1 * qi[:, l] / S1[l, G1])
                GG2_weighted = GG2.T @ diags(W2 * qi[:, l] / S2[l, G2])
                ## Compute XwXd terms ##
                XwXd[l_index: r_index] = (GG1_weighted @ GG1).diagonal()
                XwXd[ts + l_index: ts + r_index] = (GG2_weighted @ GG2).diagonal()
                if params['update_a']:
                    # Update A1_sum and A2_sum to account for worker-interaction terms
                    A1_sum_l, A2_sum_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, compute_S=False)
                    ## Compute XwY terms ##
                    XwY[l_index: r_index] = GG1_weighted @ (Y1_adj - A1_sum_l)
                    XwY[ts + l_index: ts + r_index] = GG2_weighted @ (Y2_adj - A2_sum_l)
                    del A1_sum_l, A2_sum_l
            del GG1_weighted, GG2_weighted

            # print('A1 before:')
            # print(A1)
            # print('A2 before:')
            # print(A2)
            # print('S1 before:')
            # print(S1)
            # print('S2 before:')
            # print(S2)
            # print('A1_cat_wi before:')
            # print(A1_cat_wi)
            # print('A2_cat_wi before:')
            # print(A2_cat_wi)
            # print('S1_cat_wi before:')
            # print(S1_cat_wi)
            # print('S2_cat_wi before:')
            # print(S2_cat_wi)

            # We solve the system to get all the parameters (note: this won't work if XwX is sparse)
            XwX = np.diag(XwXd)
            if params['update_a']:
                try:
                    cons_a.solve(XwX, -XwY)
                    res_a1, res_a2 = cons_a.res[: len(cons_a.res) // 2], cons_a.res[len(cons_a.res) // 2:]
                    # if pd.isna(res_a1).any() or pd.isna(res_a2).any():
                    #     raise ValueError('Estimated A1/A2 has NaN values')
                    A1 = np.reshape(res_a1, self.dims)
                    A2 = np.reshape(res_a2, self.dims)

                except ValueError as e:
                    # If constraints inconsistent, keep A1 and A2 the same
                    if params['verbose'] in [1, 2]:
                        print(f'Passing A1/A2: {e}')
                    # stop
                    pass

            ## Categorical ##
            for col in cat_cols:
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
                    CC1_weighted = CC1[col].T @ diags(W1 * qi[:, l] / S1_cat_l)
                    CC2_weighted = CC2[col].T @ diags(W2 * qi[:, l] / S2_cat_l)
                    del S1_cat_l, S2_cat_l
                    ## Compute XwX_cat terms ##
                    XwX_cat[col][l_index: r_index] = (CC1_weighted @ CC1[col]).diagonal()
                    XwX_cat[col][ts_cat[col] + l_index: ts_cat[col] + r_index] = (CC2_weighted @ CC2[col]).diagonal()
                    if params['update_a']:
                        # Update A1_sum and A2_sum to account for worker-interaction terms
                        A1_sum_l, A2_sum_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, compute_S=False)
                        if cat_dict[col]['worker_type_interaction']:
                            A1_sum_l -= A1_cat[col][l, C1[col]]
                            A2_sum_l -= A2_cat[col][l, C2[col]]
                        ## Compute XwY_cat terms ##
                        XwY_cat[col][l_index: r_index] = CC1_weighted @ (Y1_adj - A1_sum_l - A1[l, G1])
                        XwY_cat[col][ts_cat[col] + l_index: ts_cat[col] + r_index] = CC2_weighted @ (Y2_adj - A2_sum_l - A2[l, G2])
                        del A1_sum_l, A2_sum_l
                del CC1_weighted, CC2_weighted

                # We solve the system to get all the parameters (note: this won't work if XwX_cat is sparse)
                XwX_cat[col] = np.diag(XwX_cat[col])
                if params['update_a']:
                    try:
                        a_solver = cons_a_dict[col]
                        a_solver.solve(XwX_cat[col], -XwY_cat[col])
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
                        if params['verbose'] in [1, 2]:
                            print(f'Passing A1_cat/A2_cat for column {col!r}: {e}')
                        # stop
                        pass
                if not cat_dict[col]['worker_type_interaction']:
                    Y1_adj -= A1_cat[col][C1[col]]
                    Y2_adj -= A2_cat[col][C2[col]]
            ## Continuous ##
            for col in cts_cols:
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
                    CC1_weighted = C1[col].T @ diags(W1 * qi[:, l] / S1_cts_l)
                    CC2_weighted = C2[col].T @ diags(W2 * qi[:, l] / S2_cts_l)
                    del S1_cts_l, S2_cts_l
                    ## Compute XwX_cts terms ##
                    XwX_cts[col][l] = (CC1_weighted @ C1[col])
                    XwX_cts[col][nl + l] = (CC2_weighted @ C2[col])
                    if params['update_a']:
                        # Update A1_sum and A2_sum to account for worker-interaction terms
                        A1_sum_l, A2_sum_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, compute_S=False)
                        if cts_dict[col]['worker_type_interaction']:
                            A1_sum_l -= A1_cts[col][l] * C1[col]
                            A2_sum_l -= A2_cts[col][l] * C2[col]
                        ## Compute XwY_cts terms ##
                        XwY_cts[col][l] = CC1_weighted @ (Y1_adj - A1_sum_l - A1[l, G1])
                        XwY_cts[col][nl + l] = CC2_weighted @ (Y2_adj - A2_sum_l - A2[l, G2])
                        del A1_sum_l, A2_sum_l
                del CC1_weighted, CC2_weighted

                # We solve the system to get all the parameters (note: this won't work if XwX_cts is sparse)
                XwX_cts[col] = np.diag(XwX_cts[col])
                if params['update_a']:
                    try:
                        a_solver = cons_a_dict[col]
                        a_solver.solve(XwX_cts[col], -XwY_cts[col])
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
                        if params['verbose'] in [1, 2]:
                            print(f'Passing A1_cts/A2_cts for column {col!r}: {e}')
                        # stop
                        pass
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
                    # Update A1_sum and A2_sum to account for worker-interaction terms
                    A1_sum_l, A2_sum_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2, compute_S=False)
                    eps1_l_sq = (Y1_adj - A1_sum_l - A1[l, G1]) ** 2
                    eps2_l_sq = (Y2_adj - A2_sum_l - A2[l, G2]) ** 2
                    del A1_sum_l, A2_sum_l
                    ## XwS terms ##
                    l_index, r_index = l * nk, (l + 1) * nk
                    XwS[l_index: r_index] = GG1.T @ diags(W1 * qi[:, l] / S1[l, G1]) @ eps1_l_sq
                    XwS[ts + l_index: ts + r_index] = GG2.T @ diags(W2 * qi[:, l] / S2[l, G2]) @ eps2_l_sq
                    ## Categorical ##
                    for col in cat_cols:
                        col_n = cat_dict[col]['n']
                        l_index, r_index = l * col_n, (l + 1) * col_n
                        if cat_dict[col]['worker_type_interaction']:
                            S1_cat_l = S1_cat[col][l, C1[col]]
                            S2_cat_l = S2_cat[col][l, C2[col]]
                        else:
                            S1_cat_l = S1_cat[col][C1[col]]
                            S2_cat_l = S2_cat[col][C2[col]]
                        ## XwS_cat terms ##
                        XwS_cat[col][l_index: r_index] = CC1[col].T @ diags(W1 * qi[:, l] / S1_cat_l) @ eps1_l_sq
                        XwS_cat[col][ts_cat[col] + l_index: ts_cat[col] + r_index] = CC2[col].T @ diags(W2 * qi[:, l] / S2_cat_l) @ eps2_l_sq
                        del S1_cat_l, S2_cat_l
                    ## Continuous ##
                    for col in cts_cols:
                        if cts_dict[col]['worker_type_interaction']:
                            S1_cts_l = S1_cts[col][l]
                            S2_cts_l = S2_cts[col][l]
                        else:
                            S1_cts_l = S1_cts[col]
                            S2_cts_l = S2_cts[col]
                        ## XwS_cts terms ##
                        # NOTE: take absolute value
                        XwS_cts[col][l] = np.abs(C1[col].T @ diags(W1 * qi[:, l] / S1_cts_l) @ eps1_l_sq)
                        XwS_cts[col][nl + l] = np.abs(C2[col].T @ diags(W2 * qi[:, l] / S2_cts_l) @ eps2_l_sq)
                        del S1_cts_l, S2_cts_l
                    del eps1_l_sq, eps2_l_sq

                try:
                    cons_s.solve(XwX, -XwS)
                    res_s1, res_s2 = cons_s.res[: len(cons_s.res) // 2], cons_s.res[len(cons_s.res) // 2:]
                    # if pd.isna(res_s1).any() or pd.isna(res_s2).any():
                    #     raise ValueError('Estimated S1/S2 has NaN values')
                    S1 = np.sqrt(np.reshape(res_s1, self.dims))
                    S2 = np.sqrt(np.reshape(res_s2, self.dims))

                except ValueError as e:
                    # If constraints inconsistent, keep S1 and S2 the same
                    if params['verbose'] in [1, 2]:
                        print(f'Passing S1/S2: {e}')
                    # stop
                    pass
                ## Categorical ##
                for col in cat_cols:
                    try:
                        col_n = cat_dict[col]['n']
                        s_solver = cons_s_dict[col]
                        s_solver.solve(XwX_cat[col], -XwS_cat[col])
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
                        if params['verbose'] in [1, 2]:
                            print(f'Passing S1_cat/S2_cat for column {col!r}: {e}')
                        # stop
                        pass
                ## Continuous ##
                for col in cts_cols:
                    try:
                        s_solver = cons_s_dict[col]
                        s_solver.solve(XwX_cts[col], -XwS_cts[col])
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
                        if params['verbose'] in [1, 2]:
                            print(f'Passing S1_cts/S2_cts for column {col!r}: {e}')
                        # stop
                        pass

            # print('A1 after:')
            # print(A1)
            # print('A2 after:')
            # print(A2)
            # print('S1 after:')
            # print(S1)
            # print('S2 after:')
            # print(S2)
            # print('A1_cat_wi after:')
            # print(A1_cat_wi)
            # print('A2_cat_wi after:')
            # print(A2_cat_wi)
            # print('S1_cat_wi after:')
            # print(S1_cat_wi)
            # print('S2_cat_wi after:')
            # print(S2_cat_wi)

            if params['update_pk1']:
                pk1 = GG12.T @ ((W1 + W2) * qi.T).T
                # Add dirichlet prior
                pk1 += d_prior - 1
                # Normalize rows to sum to 1
                pk1 = (pk1.T / np.sum(pk1, axis=1).T).T

        self.A1, self.A2, self.S1, self.S2 = A1, A2, S1, S2
        self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat = A1_cat, A2_cat, S1_cat, S2_cat
        self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts = A1_cts, A2_cts, S1_cts, S2_cts
        self.pk1, self.lik1 = pk1, lik1
        self.liks1 = liks1 # np.concatenate([self.liks1, liks1])

        # Sort parameters
        self._sort_matrices()

        if normalize and (len(cat_cols) > 0):
            # Normalize lowest firm-worker pair to have effect 0 if there are categorical controls
            min_firm_type = np.mean(self.A1 + self.A2, axis=0).argsort()[0]
            adj_val = (self.A1[0, min_firm_type] + self.A2[0, min_firm_type]) / 2
            self.A1 -= adj_val
            self.A2 -= adj_val
            self.A1_cat[cat_cols[0]] += adj_val
            self.A2_cat[cat_cols[0]] += adj_val

        # Update NNm
        if compute_NNm:
            self.NNm = jdata.groupby('g1')['g2'].value_counts().unstack(fill_value=0).to_numpy()

    def fit_stayers(self, sdata, compute_NNs=True):
        '''
        EM algorithm for stayers.

        Arguments:
            sdata (BipartitePandas DataFrame): stayers
            compute_NNs (bool): if True, compute vector giving the number of stayers at each firm type (e.g. entry (1) gives the number of stayers at firm type 1)
        '''
        # Unpack parameters
        params = self.params
        nl, nk, ni = self.nl, self.nk, sdata.shape[0]
        A1, A2, S1, S2 = self.A1, self.A2, self.S1, self.S2
        cat_cols, cts_cols = self.cat_cols, self.cts_cols
        any_controls = self.any_controls
        # Fix error from bad initial guesses causing probabilities to be too low
        d_prior = params['d_prior_stayers']

        # Store wage outcomes and groups
        Y1 = sdata['y1'].to_numpy()
        Y2 = sdata['y2'].to_numpy()
        G1 = sdata['g1'].to_numpy().astype(int, copy=False)
        G2 = sdata['g2'].to_numpy().astype(int, copy=False)
        GG1 = csc_matrix((np.ones(ni), (range(ni), G1)), shape=(ni, nk))
        # Weights
        if params['weighted'] and sdata._col_included('w'):
            W1 = sdata.loc[:, 'w1'].to_numpy()
            W2 = sdata.loc[:, 'w2'].to_numpy()
        else:
            W1 = 1
            W2 = 1
        if any_controls:
            ## Control variables ##
            C1 = {}
            C2 = {}
            for col in cat_cols:
                # Get subcolumns associated with col
                subcols = to_list(sdata.col_reference_dict[col])
                n_subcols = len(subcols)
                if n_subcols == 1:
                    # If column is constant over time
                    C1[col] = sdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
                    C2[col] = sdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
                elif n_subcols == 2:
                    # If column can change over time
                    C1[col] = sdata.loc[:, subcols[0]].to_numpy().astype(int, copy=False)
                    C2[col] = sdata.loc[:, subcols[1]].to_numpy().astype(int, copy=False)
                else:
                    raise NotImplementedError(f'Column names must have either one or two associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')
            for col in cts_cols:
                # Get subcolumns associated with col
                subcols = to_list(sdata.col_reference_dict[col])
                n_subcols = len(subcols)
                if n_subcols == 1:
                    # If column is constant over time
                    C1[col] = sdata.loc[:, subcols[0]].to_numpy()
                    C2[col] = sdata.loc[:, subcols[0]].to_numpy()
                elif n_subcols == 2:
                    # If column can change over time
                    C1[col] = sdata.loc[:, subcols[0]].to_numpy()
                    C2[col] = sdata.loc[:, subcols[1]].to_numpy()
                else:
                    raise NotImplementedError(f'Column names must have either one or two associated subcolumns, but {col!r} has {n_subcols!r} associated subcolumns.')

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
            A1_sum, A2_sum, S1_sum_sq, S2_sum_sq = self._sum_by_non_nl(ni=ni, C1=C1, C2=C2)

            for l in range(nl):
                # Update A1_sum/S1_sum_sq to account for worker-interaction terms
                A1_sum_l, A2_sum_l, S1_sum_sq_l, S2_sum_sq_l = self._sum_by_nl_l(ni=ni, l=l, C1=C1, C2=C2)
                lp1 = lognormpdf(Y1, A1_sum + A1_sum_l + A1[l, G1], np.sqrt(S1_sum_sq + S1_sum_sq_l + S1[l, G1] ** 2))
                lp2 = lognormpdf(Y2, A2_sum + A2_sum_l + A2[l, G2], np.sqrt(S2_sum_sq + S2_sum_sq_l + S2[l, G2] ** 2))
                lp_stable[:, l] = lp1 + lp2
        else:
            for l in range(nl):
                lp1 = lognormpdf(Y1, A1[l, G1], S1[l, G1])
                lp2 = lognormpdf(Y2, A2[l, G2], S2[l, G2])
                lp_stable[:, l] = lp1 + lp2
        lp_stable = (lp_stable.T + np.log(W1) + np.log(W2)).T
        del lp1, lp2

        for iter in range(params['n_iters_stayers']):

            # ---------- E-Step ----------
            # We compute the posterior probabilities for each row
            # We iterate over the worker types, should not be be
            # too costly since the vector is quite large within each iteration
            for l in range(nl):
                lp[:, l] = lp_stable[:, l] + np.log(pk0[G1, l])

            # We compute log sum exp to get likelihoods and probabilities
            qi = np.exp(lp.T - logsumexp(lp, axis=1)).T
            if params['return_qi']:
                return qi
            lik0 = logsumexp(lp, axis=1).mean() # FIXME should this be returned?
            liks0.append(lik0)
            if params['verbose'] == 2:
                print('loop {}, liks {}'.format(iter, lik0))

            if abs(lik0 - prev_lik) < params['threshold_stayers']:
                break
            prev_lik = lik0

            # ---------- M-step ----------
            pk0 = GG1.T @ ((W1 + W2) * qi.T).T
            # Add dirichlet prior
            pk0 += d_prior - 1
            # Normalize rows to sum to 1
            pk0 = (pk0.T / np.sum(pk0, axis=1).T).T

        self.pk0, self.lik0 = pk0, lik0
        self.liks0 = liks0 # np.concatenate([self.liks0, liks0])

        # Update NNs
        if compute_NNs:
            NNs = sdata['g1'].value_counts(sort=False)
            NNs.sort_index(inplace=True)
            self.NNs = NNs.to_numpy()

    def fit_movers_cstr_uncstr(self, jdata, normalize=True, compute_NNm=True):
        '''
        Run fit_movers(), first constrained, then using results as starting values, run unconstrained.

        Arguments:
            jdata (BipartitePandas DataFrame): movers
            normalize (bool): if True and using categorical controls, normalize the lowest firm-worker pair to have effect 0
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
        if self.params['verbose'] in [1, 2]:
            print('Fitting movers with A fixed')
        self.fit_movers(jdata, normalize=False, compute_NNm=False)
        ##### Loop 2 #####
        # Now update A
        self.params['update_a'] = True
        if self.nl > 2:
            # Set constraints
            self.params['cons_a_all'] = cons.Linear()
            if self.params['verbose'] in [1, 2]:
                print('Fitting movers with linear constraint on A')
            self.fit_movers(jdata, normalize=False, compute_NNm=False)
        ##### Loop 3 #####
        # Remove constraints
        self.params['cons_a_all'] = None
        if self.params['verbose'] in [1, 2]:
            print('Fitting unconstrained movers')
        self.fit_movers(jdata, normalize=normalize, compute_NNm=compute_NNm)
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
            jdata (BipartitePandas DataFrame): movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # Save original parameters
        user_params = self.params.copy()
        # Update parameters
        self.params['update_a'] = True
        self.params['update_s'] = False
        self.params['update_pk1'] = False
        # Estimate
        if self.params['verbose'] in [1, 2]:
            print('Running fit_A')
        self.fit_movers(jdata, compute_NNm=compute_NNm)
        # Restore original parameters
        self.params = user_params

    def fit_S(self, jdata, compute_NNm=True):
        '''
        Run fit_movers() and update S while keeping A and pk1 fixed.

        Arguments:
            jdata (BipartitePandas DataFrame): movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # Save original parameters
        user_params = self.params.copy()
        # Update parameters
        self.params['update_a'] = False
        self.params['update_s'] = True
        self.params['update_pk1'] = False
        # Estimate
        if self.params['verbose'] in [1, 2]:
            print('Running fit_S')
        self.fit_movers(jdata, compute_NNm=compute_NNm)
        # Restore original parameters
        self.params = user_params

    def fit_pk(self, jdata, compute_NNm=True):
        '''
        Run fit_movers() and update pk1 while keeping A and S fixed.

        Arguments:
            jdata (BipartitePandas DataFrame): movers
            compute_NNm (bool): if True, compute matrix giving the number of movers who transition from one firm type to another (e.g. entry (1, 3) gives the number of movers who transition from firm type 1 to firm type 3)
        '''
        # Save original parameters
        user_params = self.params.copy()
        # Update parameters
        self.params['update_a'] = False
        self.params['update_s'] = False
        self.params['update_pk1'] = True
        # Estimate
        if self.params['verbose'] in [1, 2]:
            print('Running fit_pk')
        self.fit_movers(jdata, compute_NNm=compute_NNm)
        # Restore original parameters
        self.params = user_params

    def _sort_matrices(self, firm_effects=False, reverse=False):
        '''
        Sort arrays by cluster means.

        Arguments:
            firm_effects (bool): if True, also sort by average firm effect
            reverse (bool): if True, sort in reverse order
        '''
        nk = self.nk
        controls_dict = self.controls_dict
        ## Compute sum of all effects ##
        A_sum = self.A1 + self.A2
        for control_dict in (self.A1_cat, self.A2_cat):
            for control_col, control_array in control_dict.items():
                if controls_dict[control_col]['worker_type_interaction']:
                    A_sum = (A_sum.T + np.mean(control_array, axis=1)).T
        ## Sort worker effects ##
        worker_effect_order = np.mean(self.A1 + self.A2, axis=1).argsort()
        if reverse:
            worker_effect_order = list(reversed(worker_effect_order))
        self.A1 = self.A1[worker_effect_order, :]
        self.A2 = self.A2[worker_effect_order, :]
        self.S1 = self.S1[worker_effect_order, :]
        self.S2 = self.S2[worker_effect_order, :]
        self.pk1 = self.pk1[:, worker_effect_order]
        self.pk0 = self.pk0[:, worker_effect_order]
        # Sort control variables #
        for control_dict in (self.A1_cat, self.A2_cat, self.S1_cat, self.S2_cat):
            for control_col, control_array in control_dict.items():
                if controls_dict[control_col]['worker_type_interaction']:
                    control_dict[control_col] = control_array[worker_effect_order, :]
        for control_dict in (self.A1_cts, self.A2_cts, self.S1_cts, self.S2_cts):
            for control_col, control_array in control_dict.items():
                if controls_dict[control_col]['worker_type_interaction']:
                    control_dict[control_col] = control_array[worker_effect_order]

        if firm_effects:
            ## Sort firm effects ##
            firm_effect_order = np.mean(self.A1 + self.A2, axis=0).argsort()
            if reverse:
                firm_effect_order = list(reversed(firm_effect_order))
            self.A1 = self.A1[:, firm_effect_order]
            self.A2 = self.A2[:, firm_effect_order]
            self.S1 = self.S1[:, firm_effect_order]
            self.S2 = self.S2[:, firm_effect_order]
            self.pk0 = self.pk0[firm_effect_order, :]
            # Reorder part 1: e.g. nk=2, and type 0 > type 1, then 0, 1, 2, 3 would reorder to 1, 0, 3, 2 (i.e. reorder within groups)
            pk1_order_1 = np.tile(firm_effect_order, nk) + nk * np.repeat(range(nk), nk)
            self.pk1 = self.pk1[pk1_order_1, :]
            # Reorder part 2: e.g. nk=2, and type 0 > type 1, then 0, 1, 2, 3 would reorder to 2, 3, 0, 1 (i.e. reorder between groups)
            pk1_order_2 = nk * np.repeat(firm_effect_order, nk) + np.tile(range(nk), nk)
            self.pk1 = self.pk1[pk1_order_2, :]

    def plot_A1(self, grid=True, dpi=None):
        '''
        Plot A1 (log-earnings in first period) by worker-firm type pairs.

        Arguments:
            grid (bool): if True, plot grid
            dpi (float or None): dpi for plot
        '''
        # Sort effects to be increasing in worker and firm type
        worker_effect_order = np.mean(self.A1 + self.A2, axis=1).argsort()
        firm_effect_order = np.mean(self.A1 + self.A2, axis=0).argsort()
        A1 = self.A1[worker_effect_order, :]
        A1 = A1[:, firm_effect_order]

        # Plot
        if dpi is not None:
            plt.figure(dpi=dpi)
        x_axis = np.arange(1, self.nk + 1)
        for l in range(self.nl):
            plt.plot(x_axis, A1[l, :])
        plt.legend()
        plt.xlabel('firm class k')
        plt.ylabel('log-earnings in first period')
        plt.xticks(x_axis)
        if grid:
            plt.grid()
        plt.show()

    def plot_A2(self, grid=True, dpi=None):
        '''
        Plot A2 (log-earnings in second period) by worker-firm type pairs.

        Arguments:
            grid (bool): if True, plot grid
            dpi (float or None): dpi for plot
        '''
        # Sort effects to be increasing in worker and firm type
        worker_effect_order = np.mean(self.A1 + self.A2, axis=1).argsort()
        firm_effect_order = np.mean(self.A1 + self.A2, axis=0).argsort()
        A2 = self.A2[worker_effect_order, :]
        A2 = A2[:, firm_effect_order]

        # Plot
        if dpi is not None:
            plt.figure(dpi=dpi)
        x_axis = np.arange(1, self.nk + 1)
        for l in range(self.nl):
            plt.plot(x_axis, A2[l, :])
        plt.legend()
        plt.xlabel('firm class k')
        plt.ylabel('log-earnings in second period')
        plt.xticks(x_axis)
        if grid:
            plt.grid()
        plt.show()

    def plot_log_earnings(self, grid=True, dpi=None):
        '''
        Plot log-earnings by worker-firm type pairs.

        Arguments:
            grid (bool): if True, plot grid
            dpi (float or None): dpi for plot
        '''
        nl, nk = self.nl, self.nk

        # Compute average log-earnings # FIXME should the mean account for the log?
        A_mean = (self.A1 + self.A2) / 2 # np.log((np.exp(self.A1) + np.exp(self.A2)) / 2)

        # Sort effects to be increasing in worker and firm type
        worker_effect_order = np.mean(self.A1 + self.A2, axis=1).argsort()
        firm_effect_order = np.mean(self.A1 + self.A2, axis=0).argsort()
        A_mean = A_mean[worker_effect_order, :]
        A_mean = A_mean[:, firm_effect_order]

        # Plot
        if dpi is not None:
            plt.figure(dpi=dpi)
        x_axis = np.arange(1, nk + 1)
        for l in range(nl):
            plt.plot(x_axis, A_mean[l, :])
        plt.xlabel('firm class k')
        plt.ylabel('log-earnings')
        plt.xticks(x_axis)
        if grid:
            plt.grid()
        plt.show()

    def plot_pk1_1(self, dpi=None):
        '''
        Plot pk1 (proportions of worker types at each firm class for movers) in the first period.

        Arguments:
            dpi (float or None): dpi for plot
        '''
        nl, nk = self.nl, self.nk

        # Generate type proportions
        reshaped_pk1 = np.reshape(self.pk1, (nk, nk, nl))
        pk1_mean = np.mean(reshaped_pk1, axis=1)

        # Sort effects to be increasing in worker and firm type
        worker_effect_order = np.mean(self.A1 + self.A2, axis=1).argsort()
        firm_effect_order = np.mean(self.A1 + self.A2, axis=0).argsort()
        sorted_pk1_mean = pk1_mean.T[worker_effect_order, :]
        sorted_pk1_mean = sorted_pk1_mean[:, firm_effect_order]
        pk1_cumsum = np.cumsum(sorted_pk1_mean, axis=0)

        # Plot
        fig, ax = plt.subplots(dpi=dpi)
        x_axis = np.arange(1, nk + 1).astype(str)
        ax.bar(x_axis, pk1_mean.T[0, :])
        for l in range(1, nl):
            ax.bar(x_axis, sorted_pk1_mean[l, :], bottom=pk1_cumsum[l - 1, :])
        ax.set_xlabel('firm class k')
        ax.set_ylabel('type proportions')
        ax.set_title('Proportions of worker types')
        plt.show()

    def plot_pk1_2(self, dpi=None):
        '''
        Plot pk1 (proportions of worker types at each firm class for movers) in the second period.

        Arguments:
            dpi (float or None): dpi for plot
        '''
        nl, nk = self.nl, self.nk

        # Generate type proportions
        reshaped_pk1 = np.reshape(self.pk1, (nk, nk, nl))
        pk1_mean = np.mean(reshaped_pk1, axis=0)

        # Sort effects to be increasing in worker and firm type
        worker_effect_order = np.mean(self.A1 + self.A2, axis=1).argsort()
        firm_effect_order = np.mean(self.A1 + self.A2, axis=0).argsort()
        sorted_pk1_mean = pk1_mean.T[worker_effect_order, :]
        sorted_pk1_mean = sorted_pk1_mean[:, firm_effect_order]
        pk1_cumsum = np.cumsum(sorted_pk1_mean, axis=0)

        # Plot
        fig, ax = plt.subplots(dpi=dpi)
        x_axis = np.arange(1, nk + 1).astype(str)
        ax.bar(x_axis, pk1_mean.T[0, :])
        for l in range(1, nl):
            ax.bar(x_axis, sorted_pk1_mean[l, :], bottom=pk1_cumsum[l - 1, :])
        ax.set_xlabel('firm class k')
        ax.set_ylabel('type proportions')
        ax.set_title('Proportions of worker types')
        plt.show()

    def plot_pk0(self, dpi=None):
        '''
        Plot pk0 (proportions of worker types at each firm class for stayers).

        Arguments:
            dpi (float or None): dpi for plot
        '''
        nl, nk = self.nl, self.nk

        # Sort effects to be increasing in worker and firm type
        worker_effect_order = np.mean(self.A1 + self.A2, axis=1).argsort()
        firm_effect_order = np.mean(self.A1 + self.A2, axis=0).argsort()
        sorted_pk0 = self.pk0.T[worker_effect_order, :]
        sorted_pk0 = sorted_pk0[:, firm_effect_order]

        # Generate type proportions
        pk0_cumsum = np.cumsum(sorted_pk0, axis=0)

        # Plot
        fig, ax = plt.subplots(dpi=dpi)
        x_axis = np.arange(1, nk + 1).astype(str)
        ax.bar(x_axis, sorted_pk0[0, :])
        for l in range(1, nl):
            ax.bar(x_axis, sorted_pk0[l, :], bottom=pk0_cumsum[l - 1, :])
        ax.set_xlabel('firm class k')
        ax.set_ylabel('type proportions')
        ax.set_title('Proportions of worker types')
        plt.show()

    def plot_type_proportions(self, dpi=None):
        '''
        Plot proportions of worker types at each firm class.

        Arguments:
            dpi (float or None): dpi for plot
        '''
        nl, nk = self.nl, self.nk

        ## Generate type proportions ##
        # First, pk1 #
        reshaped_pk1 = np.reshape(self.pk1, (nk, nk, nl))
        pk1_period1 = np.mean(reshaped_pk1, axis=1)
        pk1_period2 = np.mean(reshaped_pk1, axis=0)
        pk1_mean = (pk1_period1 + pk1_period2) / 2
        # Second, take mean over pk1 and pk0 #
        nm = np.sum(self.NNm)
        ns = np.sum(self.NNs)
        # Multiply nm by 2 because each mover has observations in the first and second periods
        pk_mean = (2 * nm * pk1_mean + ns * self.pk0) / (2 * nm + ns)

        # Sort effects to be increasing in worker and firm type
        worker_effect_order = np.mean(self.A1 + self.A2, axis=1).argsort()
        firm_effect_order = np.mean(self.A1 + self.A2, axis=0).argsort()
        sorted_pk_mean = pk_mean.T[worker_effect_order, :]
        sorted_pk_mean = sorted_pk_mean[:, firm_effect_order]
        pk_cumsum = np.cumsum(sorted_pk_mean, axis=0)

        ## Plot ##
        fig, ax = plt.subplots(dpi=dpi)
        x_axis = np.arange(1, nk + 1).astype(str)
        ax.bar(x_axis, sorted_pk_mean[0, :])
        for l in range(1, nl):
            ax.bar(x_axis, sorted_pk_mean[l, :], bottom=pk_cumsum[l - 1, :])
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

    def _sim_model(self, jdata, normalize=True, rng=None):
        '''
        Generate model and run fit_movers_cstr_uncstr() given parameters.

        Arguments:
            jdata (BipartitePandas DataFrame): movers
            normalize (bool): if True and using categorical controls, normalize the lowest firm-worker pair to have effect 0
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        model = BLMModel(self.params.copy(), rng)
        model.fit_movers_cstr_uncstr(jdata, normalize=normalize)
        return model

    def fit(self, jdata, sdata, n_init=20, n_best=5, normalize=True, ncore=1, rng=None):
        '''
        EM model for movers and stayers.

        Arguments:
            jdata (BipartitePandas DataFrame): movers
            sdata (BipartitePandas DataFrame): stayers
            n_init (int): number of starting values
            n_best (int): take the n_best estimates with the highest likelihoods, and then take the estimate with the highest connectedness
            normalize (bool): if True and using categorical controls, normalize the lowest firm-worker pair to have effect 0
            ncore (int): number of cores for multiprocessing
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Run sim_model()
        if ncore > 1:
            ## Multiprocessing
            # Multiprocessing rng source: https://albertcthomas.github.io/good-practices-random-number-generators/
            seeds = rng.bit_generator._seed_seq.spawn(n_init)
            with Pool(processes=ncore) as pool:
                sim_model_lst = pool.starmap(self._sim_model, tqdm([(jdata, normalize, np.random.default_rng(seed)) for seed in seeds], total=n_init))
        else:
            sim_model_lst = itertools.starmap(self._sim_model, tqdm([(jdata, normalize, rng) for _ in range(n_init)], total=n_init))

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

        if self.params['verbose'] in [1, 2]:
            print('liks_max:', best_model.lik1)
        self.model = best_model
        # Using best estimated parameters from fit_movers(), run fit_stayers()
        if self.params['verbose'] in [1, 2]:
            print('Running stayers')
        self.model.fit_stayers(sdata)

    def plot_A1(self, grid=True, dpi=None):
        '''
        Plot A1 (log-earnings in first period) by worker-firm type pairs.

        Arguments:
            grid (bool): if True, plot grid
            dpi (float or None): dpi for plot
        '''
        if self.model is not None:
            self.model.plot_A1(grid=grid, dpi=dpi)
        else:
            warnings.warn('Estimation has not yet been run.')

    def plot_A2(self, grid=True, dpi=None):
        '''
        Plot A2 (log-earnings in second period) by worker-firm type pairs.

        Arguments:
            grid (bool): if True, plot grid
            dpi (float or None): dpi for plot
        '''
        if self.model is not None:
            self.model.plot_A2(grid=grid, dpi=dpi)
        else:
            warnings.warn('Estimation has not yet been run.')

    def plot_log_earnings(self, grid=True, dpi=None):
        '''
        Plot log-earnings by worker-firm type pairs.

        Arguments:
            grid (bool): if True, plot grid
            dpi (float or None): dpi for plot
        '''
        if self.model is not None:
            self.model.plot_log_earnings(grid=grid, dpi=dpi)
        else:
            warnings.warn('Estimation has not yet been run.')

    def plot_pk1_1(self, dpi=None):
        '''
        Plot pk1 (proportions of worker types at each firm class for movers) in the first period.

        Arguments:
            dpi (float or None): dpi for plot
        '''
        if self.model is not None:
            self.model.plot_pk1_1(dpi=dpi)
        else:
            warnings.warn('Estimation has not yet been run.')

    def plot_pk1_2(self, dpi=None):
        '''
        Plot pk1 (proportions of worker types at each firm class for movers) in the second period.

        Arguments:
            dpi (float or None): dpi for plot
        '''
        if self.model is not None:
            self.model.plot_pk1_2(dpi=dpi)
        else:
            warnings.warn('Estimation has not yet been run.')

    def plot_pk0(self, dpi=None):
        '''
        Plot pk0 (proportions of worker types at each firm class for stayers).

        Arguments:
            dpi (float or None): dpi for plot
        '''
        if self.model is not None:
            self.model.plot_pk0(dpi=dpi)
        else:
            warnings.warn('Estimation has not yet been run.')

    def plot_type_proportions(self, dpi=None):
        '''
        Plot proportions of worker types at each firm class.

        Arguments:
            dpi (float or None): dpi for plot
        '''
        if self.model is not None:
            self.model.plot_type_proportions(dpi)
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

    def fit(self, jdata, sdata, n_samples=5, frac_movers=0.1, frac_stayers=0.1, n_init_estimator=20, n_best=5, normalize=True, ncore=1, rng=None):
        '''
        EM model for movers and stayers.

        Arguments:
            jdata (BipartitePandas DataFrame): movers
            sdata (BipartitePandas DataFrame): stayers
            n_samples (int): number of bootstrap samples to estimate
            frac_movers (float): fraction of movers to draw (with replacement) for each bootstrap sample
            frac_stayers (float): fraction of stayers to draw (with replacement) for each bootstrap sample
            n_init_estimator (int): number of starting guesses to estimate for each bootstrap sample
            n_best (int): take the n_best estimates with the highest likelihoods, and then take the estimate with the highest connectedness, for each bootstrap sample
            normalize (bool): if True and using categorical controls, normalize the lowest firm-worker pair to have effect 0
            ncore (int): number of cores for multiprocessing
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        wj = None
        if self.params['weighted'] and jdata._col_included('w'):
            wj = jdata['w1'].to_numpy() + jdata['w2'].to_numpy()
        ws = None
        if self.params['weighted'] and sdata._col_included('w'):
            ws = sdata['w1'].to_numpy() + sdata['w2'].to_numpy()

        models = []
        for _ in tqdm(range(n_samples)):
            jdata_i = jdata.sample(frac=frac_movers, replace=True, weights=wj, random_state=rng)
            sdata_i = sdata.sample(frac=frac_stayers, replace=True, weights=ws, random_state=rng)
            blm_fit_i = BLMEstimator(self.params)
            # Set normalize=False, because want to make sure the same firm type is always normalized - normalize manually later
            blm_fit_i.fit(jdata=jdata_i, sdata=sdata_i, n_init=n_init_estimator, n_best=n_best, normalize=False, ncore=ncore, rng=rng)
            models.append(blm_fit_i.model)
            del jdata_i, sdata_i, blm_fit_i

        cat_cols = models[0].cat_cols
        if normalize and (len(cat_cols) > 0):
            # Normalize lowest firm-worker pair to have effect 0 if there are categorical controls
            nl, nk = self.params.get_multiple(('nl', 'nk'))

            # Compute average log-earnings # FIXME should the mean account for the log?
            A_means = np.zeros((len(self.models), nl, nk))
            for i, model in enumerate(self.models):
                A_means[i, :, :] = (model.A1 + model.A2) / 2 # np.log((np.exp(model.A1) + np.exp(model.A2)) / 2)
            A_means_mean = np.mean(A_means, axis=0)

            min_firm_type = np.mean(A_means_mean, axis=0).argsort()[0]

            for model in models:
                adj_val = (model.A1[0, min_firm_type] + model.A2[0, min_firm_type]) / 2
                model.A1 -= adj_val
                model.A2 -= adj_val
                model.A1_cat[cat_cols[0]] += adj_val
                model.A2_cat[cat_cols[0]] += adj_val

        self.models = models

    def plot_log_earnings(self, grid=True, dpi=None):
        '''
        Plot log-earnings by worker-firm type pairs.

        Arguments:
            grid (bool): if True, plot grid
            dpi (float or None): dpi for plot
        '''
        if self.models is None:
            warnings.warn('Estimation has not yet been run.')
        else:
            nl, nk = self.params.get_multiple(('nl', 'nk'))

            # Compute average log-earnings # FIXME should the mean account for the log?
            A_means = np.zeros((len(self.models), nl, nk))
            for i, model in enumerate(self.models):
                A_means[i, :, :] = (model.A1 + model.A2) / 2 # np.log((np.exp(model.A1) + np.exp(model.A2)) / 2)
            A_means_mean = np.mean(A_means, axis=0)

            A_lb = np.percentile(A_means, 2.5, axis=0)
            A_ub = np.percentile(A_means, 97.5, axis=0)
            A_ci = np.stack([A_means_mean - A_lb, A_ub - A_means_mean], axis=0)

            # Sort by firm effects
            firm_effect_order = np.mean(A_means_mean, axis=0).argsort()
            A_means_mean = A_means_mean[:, firm_effect_order]
            A_ci = A_ci[:, :, firm_effect_order]

            # Plot
            if dpi is not None:
                plt.figure(dpi=dpi)
            x_axis = np.arange(1, nk + 1)
            for l in range(nl):
                plt.errorbar(x_axis, A_means_mean[l, :], yerr=A_ci[:, l, :], capsize=3)
            plt.xlabel('firm class k')
            plt.ylabel('log-earnings')
            plt.xticks(x_axis)
            if grid:
                plt.grid()
            plt.show()

    def plot_type_proportions(self, dpi=None):
        '''
        Plot proportions of worker types at each firm class.

        Arguments:
            dpi (float or None): dpi for plot
        '''
        if self.models is None:
            warnings.warn('Estimation has not yet been run.')
        else:
            nl, nk = self.params.get_multiple(('nl', 'nk'))

            # Compute average log-earnings - this is needed to compute the proper firm effect order # FIXME should the mean account for the log?
            A_means = np.zeros((len(self.models), nl, nk))
            for i, model in enumerate(self.models):
                A_means[i, :, :] = (model.A1 + model.A2) / 2 # np.log((np.exp(model.A1) + np.exp(model.A2)) / 2)
            A_means_mean = np.mean(A_means, axis=0)

            pk_mean = np.zeros((nk, nl))
            for model in self.models:
                ## Generate type proportions ##
                # First, pk1 #
                reshaped_pk1 = np.reshape(model.pk1, (nk, nk, nl))
                pk1_period1 = np.mean(reshaped_pk1, axis=1)
                pk1_period2 = np.mean(reshaped_pk1, axis=0)
                pk1_mean = (pk1_period1 + pk1_period2) / 2
                # Second, take mean over pk1 and pk0 #
                nm = np.sum(model.NNm)
                ns = np.sum(model.NNs)
                # Multiply nm by 2 because each mover has observations in the first and second periods
                pk_mean += (2 * nm * pk1_mean + ns * model.pk0) / (2 * nm + ns)
            pk_mean /= len(self.models)
            pk_cumsum = np.cumsum(pk_mean, axis=1)

            # Sort by firm effects
            firm_effect_order = np.mean(A_means_mean, axis=0).argsort()
            pk_mean = pk_mean[firm_effect_order, :]
            pk_cumsum = pk_cumsum[firm_effect_order, :]

            ## Plot ##
            fig, ax = plt.subplots(dpi=dpi)
            x_axis = np.arange(1, nk + 1).astype(str)
            ax.bar(x_axis, pk_mean.T[0, :])
            for l in range(1, nl):
                ax.bar(x_axis, pk_mean.T[l, :], bottom=pk_cumsum.T[l - 1, :])
            ax.set_xlabel('firm class k')
            ax.set_ylabel('type proportions')
            ax.set_title('Proportions of worker types')
            plt.show()
