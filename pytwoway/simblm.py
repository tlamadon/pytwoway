'''
Class for simulating bipartite BLM networks.
'''
import numpy as np
from pandas import DataFrame
from bipartitepandas.util import ParamsDict, _is_subtype

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq1(a):
    return a >= 1
def _gteq0(a):
    return a >= 0
def _gt0(a):
    return a > 0
def _min_gt0(a):
    return np.min(a) > 0
def _indep_split_dict(a):
    for v in a.values():
        if not((len(v) == 9) and _is_subtype(v['n'], int) and (v['n'] >= 2) and _is_subtype(v['a1_mu'], (float, int)) and _is_subtype(v['a2_mu'], (float, int)) and _is_subtype(v['a1_sig'], (float, int)) and (v['a1_sig'] >= 0) and _is_subtype(v['a2_sig'], (float, int)) and (v['a2_sig'] >= 0) and _is_subtype(v['s1_low'], (float, int)) and (v['s1_low'] >= 0) and _is_subtype(v['s2_low'], (float, int)) and (v['s2_low'] >= 0) and _is_subtype(v['s1_high'], (float, int)) and (v['s1_high'] >= 0) and _is_subtype(v['s2_high'], (float, int)) and (v['s2_high'] >= 0) and (v['s1_high'] >= v['s1_low']) and (v['s2_high'] >= v['s2_low'])):
            return False
    return True
def _indep_nosplit_dict(a):
    for v in a.values():
        if not ((len(v) == 5) and _is_subtype(v['n'], int) and (v['n'] >= 2) and _is_subtype(v['a_mu'], (float, int)) and _is_subtype(v['a_sig'], (float, int)) and (v['a_sig'] >= 0) and _is_subtype(v['s_low'], (float, int)) and (v['s_low'] >= 0) and _is_subtype(v['s_high'], (float, int)) and (v['s_high'] >= 0) and (v['s_high'] >= v['s_low'])):
            return False
    return True
def _dep_dict(a):
    return np.array([_is_subtype(v, int) and (v >= 2) for v in a.values()]).all()

# Define default parameter dictionary
_sim_params_default = ParamsDict({
    ## Class parameters
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
    'custom_dependent_dict': (None, 'type_constrained_none', (dict, _dep_dict),
        '''
            (default=None) Dictionary of custom general column names (to use as controls) linked to the number of types for that column, where the estimated parameters should be dependent on worker/firm type pairs. In other words, any column listed as a member of this parameter will have a different parameter estimated for each worker-firm type pair (the parameter value can still differ over time). None is equivalent to {}.
        ''', None),
    'custom_independent_split_dict': (None, 'type_constrained_none', (dict, _indep_split_dict),
        '''
            (default=None) Dictionary of custom general column names (to use as controls) linked to a dictionary with keys: 'n': the number of types for that column; 'a1_mu': mean of simulated mean for the first period; 'a1_sig': standard deviation of simulated mean for the first period; 's1_low': minimum value of simulated standard deviation for the first period; 's1_high': maximum value of simulated standard deviation for the first period; and 'a2_mu', 'a2_sig', 's2_low', and 's2_high' are equivalent for the second period; where the estimated parameters should be independent of worker/firm type pairs. In other words, any column listed as a member of this parameter will have the same parameter estimated for each worker-firm type pair (the parameter value can still differ over time). None is equivalent to {}.
        ''', None),
    'custom_independent_nosplit_dict': (None, 'type_constrained_none', (dict, _indep_nosplit_dict),
        '''
            (default=None) Dictionary of custom general column names (to use as controls) linked to a dictionary with keys: 'n': the number of types for that column; 'a_mu': mean of simulated mean for both periods; 'a_sig': standard deviation of simulated mean for both periods; 's_low': minimum value of simulated standard deviation for both periods; 's_high': maximum value of simulated standard deviation for both periods; where the estimated parameters should be independent of worker/firm type pairs. In other words, any column listed as a member of this parameter will have the same parameter estimated for each worker-firm type pair (the parameter value can still differ over time). None is equivalent to {}.
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
    's1_min': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S1 (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's1_max': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S1 (standard deviation of fixed effects in first period).
        ''', '>= 0'),
    's2_min': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S2 (standard deviation of fixed effects in second period).
        ''', '>= 0'),
    's2_max': (0.5, 'type_constrained', ((float, int), _gteq0),
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
            (default=False) If True, set A1 and A2 to be strictly increasing by firm type for each worker type (otherwise, they must simply be increasing by firm type over the average for all worker types).
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

        ## Unpack custom column parameters
        custom_dep_dict = self.params['custom_dependent_dict']
        custom_indep_split_dict = self.params['custom_independent_split_dict']
        custom_indep_nosplit_dict = self.params['custom_independent_nosplit_dict']
        ## Check if custom column parameters are None
        if custom_dep_dict is None:
            custom_dep_dict = {}
        if custom_indep_split_dict is None:
            custom_indep_split_dict = {}
        if custom_indep_nosplit_dict is None:
            custom_indep_nosplit_dict = {}
        ## Create dictionary of all custom columns
        custom_cols_dict = custom_dep_dict.copy()
        custom_cols_dict.update(custom_indep_split_dict.copy())
        custom_cols_dict.update(custom_indep_nosplit_dict.copy())
        ## Custom column order
        custom_dep_cols = sorted(custom_dep_dict.keys())
        custom_indep_split_cols = sorted(custom_indep_split_dict.keys())
        custom_indep_nosplit_cols = sorted(custom_indep_nosplit_dict.keys())
        custom_cols = custom_dep_cols + custom_indep_split_cols + custom_indep_nosplit_cols
        ## Store custom column attributes
        # Dictionaries
        self.custom_dep_dict = custom_dep_dict
        self.custom_indep_split_dict = custom_indep_split_dict
        self.custom_indep_nosplit_dict = custom_indep_nosplit_dict
        self.custom_cols_dict = custom_cols_dict
        # Lists
        self.custom_dep_cols = custom_dep_cols
        self.custom_indep_split_cols = custom_indep_split_cols
        self.custom_indep_nosplit_cols = custom_indep_nosplit_cols
        self.custom_cols = custom_cols

        for col in custom_dep_cols:
            # Make sure independent and dependent custom columns don't overlap
            if col in custom_indep_split_cols:
                raise NotImplementedError(f'Custom dependent columns and custom independent split columns cannot overlap, but input includes column {col!r} as a member of both.')
            if col in custom_indep_nosplit_cols:
                raise NotImplementedError(f'Custom dependent columns and custom independent no-split columns cannot overlap, but input includes column {col!r} as a member of both.')
        for col in custom_indep_split_cols:
            # Make sure independent split and independent no-split custom columns don't overlap
            if col in custom_indep_nosplit_cols:
                raise NotImplementedError(f'Custom independent split columns and custom independent no-split columns cannot overlap, but input includes column {col!r} as a member of both.')

        dims = [nl, nk]
        for col in custom_dep_cols:
            # dims must account for all dependent columns (and make sure the columns are in the correct order)
            dims.append(custom_dep_dict[col])
        self.dims = dims

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
        n_dims = len(self.dims)
        A_mean = (A1 + A2) / 2

        if strictly_monotone_a:
            ## Make A1 and A2 monotone by worker type ##
            for l in range(nl):
                A1[l] = np.sort(A1[l], axis=0)
                A2[l] = np.sort(A2[l], axis=0)

        ## Sort worker effects ##
        worker_effect_order = np.mean(A1, axis=tuple(range(n_dims)[1:])).argsort()
        A1 = A1[worker_effect_order, :]
        A2 = A2[worker_effect_order, :]

        if not strictly_monotone_a:
            ## Sort firm effects ##
            firm_effect_order = np.mean(A1, axis=(0, *range(n_dims)[2:])).argsort()
            A1 = A1[:, firm_effect_order]
            A2 = A2[:, firm_effect_order]

        return A1, A2

    def _gen_params(self, rng=None):
        '''
        Generate parameter values to use for simulating bipartite BLM data.

        Arguments:
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (dict): keys are 'A1', 'A2', 'S1', 'S2', 'pk1', 'pk0', 'A1_indep', 'A2_indep', 'A_indep', 'S1_indep', 'S2_indep', and 'S_indep'. 'A1' gives the mean of fixed effects in the first period; 'A2' gives the mean of fixed effects in the second period; 'S1' gives the standard deviation of fixed effects in the first period; 'S2' gives the standard deviation of fixed effects in the second period; 'pk1' gives the probability of being at each combination of firm types for movers; 'pk0' gives the probability of being at each firm type for stayers; 'A1_indep' gives the mean of fixed effects for custom split columns in the first period; 'A2_indep' gives the mean of fixed effects for custom split columns in the second period; 'A_indep' gives the mean of fixed effects for custom no-split columns; 'S1_indep' gives the standard deviation of fixed effects for custom split columns in the first period; 'S2_indep' gives the standard deviation of fixed effects for custom split columns in the second period; 'S_indep' gives the standard deviation of fixed effects for custom no-split columns.
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        nl, nk = self.params.get_multiple(('nl', 'nk'))
        a1_mu, a1_sig, a2_mu, a2_sig, s1_min, s1_max, s2_min, s2_max, pk1_prior, pk0_prior = self.params.get_multiple(('a1_mu', 'a1_sig', 'a2_mu', 'a2_sig', 's1_min', 's1_max', 's2_min', 's2_max', 'pk1_prior', 'pk0_prior'))
        custom_cols_dict, split_cols, nosplit_cols = self.custom_cols_dict, self.custom_indep_split_cols, self.custom_indep_nosplit_cols
        fixb, stationary = self.params.get_multiple(('fixb', 'stationary'))
        dims = self.dims

        ### Draw parameters ###
        # Model for Y1 | Y2, l, k for movers and stayers
        A1 = rng.normal(loc=a1_mu, scale=a1_sig, size=dims)
        S1 = rng.uniform(low=s1_min, high=s1_max, size=dims)
        # Model for Y4 | Y3, l, k for movers and stayers
        A2 = rng.normal(loc=a2_mu, scale=a2_sig, size=dims)
        S2 = rng.uniform(low=s2_min, high=s2_max, size=dims)
        # Model for p(K | l, l') for movers
        if pk1_prior is None:
            pk1_prior = np.ones(nl)
        pk1 = rng.dirichlet(alpha=pk1_prior, size=nk ** 2)
        # Model for p(K | l, l') for stayers
        if pk0_prior is None:
            pk0_prior = np.ones(nl)
        pk0 = rng.dirichlet(alpha=pk0_prior, size=nk)
        ## Control variables ##
        # Split
        A1_indep = {col: rng.normal(loc=custom_cols_dict[col]['a1_mu'], scale=custom_cols_dict[col]['a1_sig'], size=custom_cols_dict[col]['n']) for col in split_cols}
        S1_indep = {col: rng.uniform(low=custom_cols_dict[col]['s1_low'], high=custom_cols_dict[col]['s1_high'], size=custom_cols_dict[col]['n']) for col in split_cols}
        A2_indep = {col: rng.normal(loc=custom_cols_dict[col]['a2_mu'], scale=custom_cols_dict[col]['a2_sig'], size=custom_cols_dict[col]['n']) for col in split_cols}
        S2_indep = {col: rng.uniform(low=custom_cols_dict[col]['s2_low'], high=custom_cols_dict[col]['s2_high'], size=custom_cols_dict[col]['n']) for col in split_cols}
        # No-split
        A_indep = {col: rng.normal(loc=custom_cols_dict[col]['a_mu'], scale=custom_cols_dict[col]['a_sig'], size=custom_cols_dict[col]['n']) for col in nosplit_cols}
        S_indep = {col: rng.uniform(low=custom_cols_dict[col]['s_low'], high=custom_cols_dict[col]['s_high'], size=custom_cols_dict[col]['n']) for col in nosplit_cols}

        ## Sort parameters ##
        A1, A2 = self._sort_A(A1, A2)

        if fixb:
            A2 = np.mean(A2, axis=1) + A1 - np.mean(A1, axis=1)

        if stationary:
            A2 = A1

        return {'A1': A1, 'A2': A2, 'S1': S1, 'S2': S2, 'pk1': pk1, 'pk0': pk0, 'A1_indep': A1_indep, 'A2_indep': A2_indep, 'A_indep': A_indep, 'S1_indep': S1_indep, 'S2_indep': S2_indep, 'S_indep': S_indep}

    def _simulate_movers(self, A1, A2, S1, S2, pk1, pk0, A1_indep, A2_indep, A_indep,  S1_indep, S2_indep, S_indep, rng=None):
        '''
        Simulate data for movers (simulates firm types, not firms).

        Arguments:
            A1 (NumPy Array): mean of fixed effects in the first period
            A2 (NumPy Array): mean of fixed effects in the second period
            S1 (NumPy Array): standard deviation of fixed effects in the first period
            S2 (NumPy Array): standard deviation of fixed effects in the second period
            pk1 (NumPy Array): probability of being at each combination of firm types for movers
            pk0 (NumPy Array): probability of being at each firm type for stayers (used only for _simulate_stayers)
            A1_indep (NumPy Array): mean of fixed effects for custom split columns in the first period
            A2_indep (NumPy Array): mean of fixed effects for custom split columns in the second period
            A_indep (NumPy Array): mean of fixed effects for custom no-split columns
            S1_indep (NumPy Array): standard deviation of fixed effects for custom split columns in the first period
            S2_indep (NumPy Array): standard deviation of fixed effects for custom split columns in the second period
            S_indep (NumPy Array): standard deviation of fixed effects for custom no-split columns
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

        if len(self.custom_cols) > 0:
            # Draw custom columns FIXME add custom probabilities?
            A1_draws = {split_col: rng.choice(np.arange(split_dict['n']), size=nmi, replace=True) for split_col, split_dict in self.custom_indep_split_dict.items()}
            A2_draws = {split_col: rng.choice(np.arange(split_dict['n']), size=nmi, replace=True) for split_col, split_dict in self.custom_indep_split_dict.items()}
            A_draws = {nosplit_col: rng.choice(np.arange(nosplit_dict['n']), size=nmi, replace=True) for nosplit_col, nosplit_dict in self.custom_indep_nosplit_dict.items()}
            S1_draws = A1_draws
            S2_draws = A2_draws
            S_draws = A_draws

            # Simulate custom column wages
            A1_sum = np.zeros(shape=nmi)
            A2_sum = np.zeros(shape=nmi)
            S1_sum_sq = np.zeros(shape=nmi)
            S2_sum_sq = np.zeros(shape=nmi)
            for split_col in self.custom_indep_split_cols:
                A1_sum += A1_indep[split_col][A1_draws[split_col]]
                A2_sum += A2_indep[split_col][A2_draws[split_col]]
                S1_sum_sq += S1_indep[split_col][S1_draws[split_col]] ** 2
                S2_sum_sq += S2_indep[split_col][S2_draws[split_col]] ** 2
            for nosplit_col in self.custom_indep_nosplit_cols:
                A1_sum += A_indep[nosplit_col][A_draws[nosplit_col]]
                A2_sum += A_indep[nosplit_col][A_draws[nosplit_col]]
                S1_sum_sq += S_indep[nosplit_col][S_draws[nosplit_col]] ** 2
                S2_sum_sq += S_indep[nosplit_col][S_draws[nosplit_col]] ** 2

            Y1 += rng.normal(loc=A1_sum, scale=np.sqrt(S1_sum_sq), size=nmi)
            Y2 += rng.normal(loc=A2_sum, scale=np.sqrt(S2_sum_sq), size=nmi)

            A1_draws = {k + '1': v for k, v in A1_draws.items()}
            A2_draws = {k + '2': v for k, v in A2_draws.items()}

            return DataFrame(data={'y1': Y1, 'y2': Y2, 'g1': G1, 'g2': G2, 'l': L, **A1_draws, **A2_draws, **A_draws})

        return DataFrame(data={'y1': Y1, 'y2': Y2, 'g1': G1, 'g2': G2, 'l': L})

    def _simulate_stayers(self, A1, A2, S1, S2, pk1, pk0, A1_indep, A2_indep, A_indep,  S1_indep, S2_indep, S_indep, rng=None):
        '''
        Simulate data for stayers (simulates firm types, not firms).

        Arguments:
            A1 (NumPy Array): mean of fixed effects in the first period
            A2 (NumPy Array): mean of fixed effects in the second period
            S1 (NumPy Array): standard deviation of fixed effects in the first period
            S2 (NumPy Array): standard deviation of fixed effects in the second period
            pk1 (NumPy Array): probability of being at each combination of firm types for movers (used only for _simulate_movers)
            pk0 (NumPy Array): probability of being at each firm type for stayers
            A1_indep (NumPy Array): mean of fixed effects for custom split columns in the first period
            A2_indep (NumPy Array): mean of fixed effects for custom split columns in the second period
            A_indep (NumPy Array): mean of fixed effects for custom no-split columns
            S1_indep (NumPy Array): standard deviation of fixed effects for custom split columns in the first period
            S2_indep (NumPy Array): standard deviation of fixed effects for custom split columns in the second period
            S_indep (NumPy Array): standard deviation of fixed effects for custom no-split columns
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

        if len(self.custom_cols) > 0:
            # Draw custom columns FIXME add custom probabilities?
            A1_draws = {split_col: rng.choice(np.arange(split_dict['n']), size=nsi, replace=True) for split_col, split_dict in self.custom_indep_split_dict.items()}
            A2_draws = {split_col: rng.choice(np.arange(split_dict['n']), size=nsi, replace=True) for split_col, split_dict in self.custom_indep_split_dict.items()}
            A_draws = {nosplit_col: rng.choice(np.arange(nosplit_dict['n']), size=nsi, replace=True) for nosplit_col, nosplit_dict in self.custom_indep_nosplit_dict.items()}
            S1_draws = A1_draws
            S2_draws = A2_draws
            S_draws = A_draws

            # Simulate custom column wages
            A1_sum = np.zeros(shape=nsi)
            A2_sum = np.zeros(shape=nsi)
            S1_sum_sq = np.zeros(shape=nsi)
            S2_sum_sq = np.zeros(shape=nsi)
            for split_col in self.custom_indep_split_cols:
                A1_sum += A1_indep[split_col][A1_draws[split_col]]
                A2_sum += A2_indep[split_col][A2_draws[split_col]]
                S1_sum_sq += S1_indep[split_col][S1_draws[split_col]] ** 2
                S2_sum_sq += S2_indep[split_col][S2_draws[split_col]] ** 2
            for nosplit_col in self.custom_indep_nosplit_cols:
                A1_sum += A_indep[nosplit_col][A_draws[nosplit_col]]
                A2_sum += A_indep[nosplit_col][A_draws[nosplit_col]]
                S1_sum_sq += S_indep[nosplit_col][S_draws[nosplit_col]] ** 2
                S2_sum_sq += S_indep[nosplit_col][S_draws[nosplit_col]] ** 2

            Y1 += rng.normal(loc=A1_sum, scale=np.sqrt(S1_sum_sq), size=nsi)
            Y2 += rng.normal(loc=A2_sum, scale=np.sqrt(S2_sum_sq), size=nsi)

            A1_draws = {k + '1': v for k, v in A1_draws.items()}
            A2_draws = {k + '2': v for k, v in A2_draws.items()}

            return DataFrame(data={'y1': Y1, 'y2': Y2, 'g1': G, 'g2': G, 'l': L, **A1_draws, **A2_draws, **A_draws})

        return DataFrame(data={'y1': Y1, 'y2': Y2, 'g1': G, 'g2': G, 'l': L})

    def simulate(self, return_parameters=False, rng=None):
        '''
        Simulates data (movers and stayers) and attached firms ids. All firms have the same expected size. Columns are as follows: y1/y2=wage; j1/j2=firm id; g1/g2=firm type; l=worker type

        Arguments:
            return_parameters (bool): if True, return tuple of (simulated data, simulated parameters); otherwise, return only simulated data
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (dict or tuple of dicts): sim_data gives {'jdata': movers data, 'sdata': stayers data}, while sim_params gives {'A1': A1, 'A2': A2, 'S1': S1, 'S2': S2, 'pk1': pk1, 'pk0': pk0, 'A1_indep': A1_indep, 'A2_indep': A2_indep, 'A_indep': A_indep, 'S1_indep': S1_indep, 'S2_indep': S2_indep, 'S_indep': S_indep}; if return_parameters=True, returns (sim_data, sim_params); if return_parameters=False, returns sim_data
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
