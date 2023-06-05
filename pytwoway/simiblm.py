'''
Class for simulating bipartite interacted BLM networks.
'''
import numpy as np
from pandas import DataFrame
from paramsdict import ParamsDict
from bipartitepandas import BipartiteDataFrame
from bipartitepandas.util import _sort_cols

# NOTE: multiprocessing isn't compatible with lambda functions
def _gteq1(a):
    return a >= 1
def _gteq0(a):
    return a >= 0
def _gt0(a):
    return a > 0
def _0to1(a):
    return 0 <= a <= 1
def _in_minus_1_1(a):
    return -1 <= a <= 1
def _min_gt0(a):
    return np.min(a) > 0

# Define default parameter dictionaries
sim_interacted_blm_params = ParamsDict({
    'nk': (10, 'type_constrained', (int, _gteq1),
        '''
            (default=10) Number of firm types.
        ''', '>= 1'),
    'firm_size': (10, 'type_constrained', ((float, int), _gt0),
        '''
            (default=10) Average number of stayers per firm.
        ''', '> 0'),
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
    'eps_correlation': (0.25, 'type_constrained', ((float, int), _in_minus_1_1),
        '''
            (default=0.25) Correlation between epsilon in the first and second periods.
        ''', 'in [-1, 1]'),
    'a1_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A1 (firm-level intercept in first period).
        ''', None),
    'a1_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A1 (firm-level intercept in first period).
        ''', '>= 0'),
    'a2_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated A2 (firm-level intercept in second period).
        ''', None),
    'a2_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated A2 (firm-level intercept in second period).
        ''', '>= 0'),
    'b1_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated B1 (firm-level interaction in first period, before taking exponential).
        ''', None),
    'b1_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated B1 (firm-level interaction in first period, before taking exponential).
        ''', '>= 0'),
    'b2_mu': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated B2 (firm-level interaction in second period, before taking exponential).
        ''', None),
    'b2_sig': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated B2 (firm-level interaction in second period, before taking exponential).
        ''', '>= 0'),
    'alpha_mu_stayers': (1, 'type', (float, int),
        '''
            (default=1) Mean of simulated alpha for stayers (mean of worker-level interaction).
        ''', None),
    'alpha_sig_stayers': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated alpha for stayers (mean of worker-level interaction).
        ''', '>= 0'),
    'alpha_mu_movers_weight': (0.8, 'type', ((float, int), _0to1),
        '''
            (default=0.8) Mean of simulated alpha for movers, take a weighted average of `alpha_mu_stayers` over both periods, where this gives the weight given to the first period (mean of worker-level interaction).
        ''', 'in [0, 1]'),
    'alpha_sig_movers': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Standard error of simulated alpha for movers (mean of worker-level interaction).
        ''', '>= 0'),
    's_stayers_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S_stayers (standard deviation of alpha for stayers).
        ''', '>= 0'),
    's_stayers_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S_stayers (standard deviation of alpha for stayers).
        ''', '>= 0'),
    's_movers_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S_movers (standard deviation of alpha for movers).
        ''', '>= 0'),
    's_movers_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S_movers (standard deviation of alpha for movers).
        ''', '>= 0'),
    's_eps_low': (0.3, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.3) Minimum value of simulated S_eps (standard deviation of epsilon (noise)).
        ''', '>= 0'),
    's_eps_high': (0.5, 'type_constrained', ((float, int), _gteq0),
        '''
            (default=0.5) Maximum value of simulated S_eps (standard deviation of epsilon (noise)).
        ''', '>= 0'),
    'stationary_A': (False, 'type', bool,
        '''
            (default=False) If True, set A1 = A2.
        ''', None),
    'stationary_B': (False, 'type', bool,
        '''
            (default=False) If True, set B1 = B2.
        ''', None),
    'linear_additive': (False, 'type', bool,
        '''
            (default=False) If True, make the model linearly additive by setting B1=B2=1.
        ''', None),
})

class SimInteractedBLM:
    '''
    Class for simulating bipartite interacted BLM networks of firms and workers.

    Arguments:
        sim_params (ParamsDict): dictionary of parameters for simulating data. Run tw.sim_interacted_blm_params().describe_all() for descriptions of all valid parameters. None is equivalent to tw.sim_interacted_blm_params().
    '''

    def __init__(self, sim_params=None):
        if sim_params is None:
            sim_params = sim_interacted_blm_params()

        # Store parameters
        self.params = sim_params
        nk, NNm, NNs = self.params.get_multiple(('nk', 'NNm', 'NNs'))

        if NNm is None:
            self.NNm = 10 * np.ones(shape=(nk, nk), dtype=int)
        else:
            self.NNm = NNm
        if NNs is None:
            self.NNs = 10 * np.ones(shape=nk, dtype=int)
        else:
            self.NNs = NNs

    def _gen_params(self, rng=None):
        '''
        Generate parameter values to use for simulating bipartite interacted BLM data.

        Arguments:
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (dict): keys are 'A1', 'A2', 'B1', 'B2', 'alpha_stayers', 'alpha_movers', 'S_stayers', and 'S_movers'. 'A1' gives the firm-level intercepts in the first period; 'A2' gives the firm-level intercepts in the second period; 'B1' gives the firm-level interactions in the first period; 'B2' gives the firm-level interactions in the second period; 'alpha_stayers' gives the mean of worker-level interactions for stayers; 'alpha_movers' gives the mean of worker-level interactions for movers; 'S_stayers' gives the standard deviation of worker-level interactions for stayers; 'S_movers' gives the standard deviation of worker-level interactions for movers; and 'S_eps' gives the standard deviation of epsilon (noise).
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        ## Extract parameters ##
        nk = self.params['nk']
        a1_mu, a1_sig, a2_mu, a2_sig = \
            self.params.get_multiple(('a1_mu', 'a1_sig', 'a2_mu', 'a2_sig'))
        b1_mu, b1_sig, b2_mu, b2_sig = \
            self.params.get_multiple(('b1_mu', 'b1_sig', 'b2_mu', 'b2_sig'))
        alpha_mu_stayers, alpha_sig_stayers = \
            self.params.get_multiple(('alpha_mu_stayers', 'alpha_sig_stayers'))
        alpha_mu_movers_weight, alpha_sig_movers = \
            self.params.get_multiple(('alpha_mu_movers_weight', 'alpha_sig_movers'))
        s_stayers_low, s_stayers_high = \
            self.params.get_multiple(('s_stayers_low', 's_stayers_high'))
        s_movers_low, s_movers_high = \
            self.params.get_multiple(('s_movers_low', 's_movers_high'))
        s_eps_low, s_eps_high = \
            self.params.get_multiple(('s_eps_low', 's_eps_high'))

        ## Draw (most) parameters ##
        # Firm-level parameters
        A1 = rng.normal(loc=a1_mu, scale=a1_sig, size=nk)
        A2 = rng.normal(loc=a2_mu, scale=a2_sig, size=nk)
        if self.params['linear_additive']:
            B1 = np.ones(nk, dtype=int)
            B2 = np.ones(nk, dtype=int)
        else:
            B1 = np.exp(rng.normal(loc=b1_mu, scale=b1_sig, size=nk))
            B2 = np.exp(rng.normal(loc=b2_mu, scale=b2_sig, size=nk))
        if self.params['stationary_A']:
            A2 = A1
        if self.params['stationary_B']:
            B2 = B1

        # Stayer parameters
        alpha_stayers = rng.normal(loc=alpha_mu_stayers, scale=alpha_sig_stayers, size=nk)
        S_stayers = rng.uniform(low=s_stayers_low, high=s_stayers_high, size=nk)

        # Noise
        S_eps = rng.uniform(low=s_eps_low, high=s_eps_high, size=nk)

        ## Sort parameters ##
        alpha_stayers.sort()
        sort_order = np.argsort((A1 + B1 * alpha_stayers) + (A2 + B2 * alpha_stayers))
        A1 = A1[sort_order]
        A2 = A2[sort_order]
        B1 = B1[sort_order]
        B2 = B2[sort_order]
        alpha_stayers = alpha_stayers[sort_order]
        S_stayers = S_stayers[sort_order]

        ## Draw (remaining) parameters ##
        # Mover parameters
        alpha_movers = \
            alpha_mu_movers_weight * alpha_stayers[:, None] \
            + (1 - alpha_mu_movers_weight) * alpha_stayers[None, :] \
            + rng.normal(loc=0, scale=alpha_sig_movers, size=(nk, nk))
        S_movers = rng.uniform(low=s_movers_low, high=s_movers_high, size=(nk, nk))

        return {'A1': A1, 'A2': A2, 'B1': B1, 'B2': B2, 'alpha_stayers': alpha_stayers, 'alpha_movers': alpha_movers, 'S_stayers': S_stayers, 'S_movers': S_movers, 'S_eps': S_eps}

    def _simulate_movers(self, A1, A2, B1, B2, alpha_movers, alpha_stayers, S_movers, S_stayers, S_eps, rng=None):
        '''
        Simulate data for movers (simulates firm types, not firms).

        Arguments:
            A1 (NumPy Array): firm-level intercepts in the first period
            A2 (NumPy Array): firm-level intercepts in the second period
            B1 (NumPy Array): firm-level interactions in the first period
            B2 (NumPy Array): firm-level interactions in the second period
            alpha_movers (NumPy Array): mean of worker-level interactions for movers
            alpha_stayers (NumPy Array): mean of worker-level interactions for stayers (used only for _simulate_stayers)
            S_movers (NumPy Array): standard deviation of worker-level interactions for movers
            S_stayers (NumPy Array): standard deviation of worker-level interactions for stayers (used only for _simulate_stayers)
            S_eps (NumPy Array): standard deviation of epsilon (noise)
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (Pandas DataFrame): data for movers (y1/y2: wage; g1/g2: firm type; alpha: worker interaction; eps1/eps2: noise)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        nk, mmult, eps_corr = self.params.get_multiple(('nk', 'mmult', 'eps_correlation'))

        # Number of movers who transition between each combination of firm types
        NNm = mmult * self.NNm
        nmi = np.sum(NNm)

        Y1 = np.zeros(shape=nmi)
        Y2 = np.zeros(shape=nmi)
        G1 = np.zeros(shape=nmi, dtype=int)
        G2 = np.zeros(shape=nmi, dtype=int)
        alpha = np.zeros(shape=nmi, dtype=int)
        E1 = np.zeros(shape=nmi)
        E2 = np.zeros(shape=nmi)

        i = 0
        for k1 in range(nk):
            for k2 in range(nk):
                # Iterate over all firm type combinations a worker can transition between
                ni = NNm[k1, k2]
                I = np.arange(i, i + ni)
                G1[I] = k1
                G2[I] = k2

                # Draw alpha
                alpha[I] = alpha_movers[k1, k2] + rng.normal(loc=0, scale=S_movers[k1, k2], size=ni)

                # Draw epsilon
                eps1 = rng.normal(loc=0, scale=S_eps[k1], size=ni)
                eps2 = (eps_corr / S_eps[k1]) * eps_corr * eps1 \
                        + rng.normal(loc=0, scale=np.sqrt(1 - eps_corr ** 2) * S_eps[k2], size=ni)
                E1[I] = eps1
                E2[I] = eps2

                # Draw wages
                Y1[I] = A1[k1] + B1[k1] * alpha[I] + eps1
                Y2[I] = A2[k2] + B2[k2] * alpha[I] + eps2

                i += ni

        return DataFrame(data={'y1': Y1, 'y2': Y2, 'g1': G1, 'g2': G2, 'alpha': alpha, 'eps1': E1, 'eps2': E2})
    
    def _simulate_stayers(self, A1, A2, B1, B2, alpha_movers, alpha_stayers, S_movers, S_stayers, S_eps, rng=None):
        '''
        Simulate data for stayers (simulates firm types, not firms).

        Arguments:
            A1 (NumPy Array): firm-level intercepts in the first period
            A2 (NumPy Array): firm-level intercepts in the second period
            B1 (NumPy Array): firm-level interactions in the first period
            B2 (NumPy Array): firm-level interactions in the second period
            alpha_movers (NumPy Array): mean of worker-level interactions for movers (used only for _simulate_movers)
            alpha_stayers (NumPy Array): mean of worker-level interactions for stayers
            S_movers (NumPy Array): standard deviation of worker-level interactions for movers (used only for _simulate_movers)
            S_stayers (NumPy Array): standard deviation of worker-level interactions for stayers
            S_eps (NumPy Array): standard deviation of epsilon (noise)
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (Pandas DataFrame): data for stayers (y1/y2: wage; g1/g2: firm type; alpha: worker interaction; eps1/eps2: noise)
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Extract parameters
        nk, smult, eps_corr = self.params.get_multiple(('nk', 'smult', 'eps_correlation'))

        # Number of movers who transition between each combination of firm types
        NNs = smult * self.NNs
        nsi = np.sum(NNs)

        Y1 = np.zeros(shape=nsi)
        Y2 = np.zeros(shape=nsi)
        G = np.zeros(shape=nsi, dtype=int)
        alpha = np.zeros(shape=nsi, dtype=int)
        E1 = np.zeros(shape=nsi)
        E2 = np.zeros(shape=nsi)

        i = 0
        for k in range(nk):
            # Iterate over firm types
            ni = NNs[k]
            I = np.arange(i, i + ni)
            G[I] = k

            # Draw alpha
            alpha[I] = alpha_stayers[k] + rng.normal(loc=0, scale=S_stayers[k], size=ni)

            # Draw epsilon
            eps1 = rng.normal(loc=0, scale=S_eps[k], size=ni)
            eps2 = (eps_corr / S_eps[k]) * eps_corr * eps1 \
                    + rng.normal(loc=0, scale=np.sqrt(1 - eps_corr ** 2) * S_eps[k], size=ni)
            E1[I] = eps1
            E2[I] = eps2

            # Draw wages
            Y1[I] = A1[k] + B1[k] * alpha[I] + eps1
            Y2[I] = A2[k] + B2[k] * alpha[I] + eps2

            i += ni

        return DataFrame(data={'y1': Y1, 'y2': Y2, 'g1': G, 'g2': G, 'alpha': alpha, 'eps1': E1, 'eps2': E2})
    
    def simulate(self, return_parameters=False, rng=None):
        '''
        Simulate data (movers and stayers). All firms have the same expected size. Columns are as follows: y1/y2=wage; j1/j2=firm id; g1/g2=firm type; alpha=worker interaction; eps1/eps2=noise.

        Arguments:
            return_parameters (bool): if True, return tuple of (simulated data, simulated parameters); otherwise, return only simulated data
            rng (np.random.Generator): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (dict or tuple of dicts): sim_data gives {'jdata': movers BipartiteDataFrame, 'sdata': stayers BipartiteDataFrame}, while sim_params gives {'A1': A1, 'A2': A2, 'B1': B1, 'B2': B2, 'alpha_stayers': alpha_stayers, 'alpha_movers': alpha_movers, 'S_stayers': S_stayers, 'S_movers': S_movers, 'S_eps': S_eps}; if return_parameters=True, returns (sim_data, sim_params); if return_parameters=False, returns sim_data

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

        # Add m column
        jdata.loc[:, 'm'] = 1
        sdata.loc[:, 'm'] = 0

        # Add i column
        nm = len(jdata)
        ns = len(sdata)
        jdata.loc[:, 'i'] = np.arange(nm, dtype=int)
        sdata.loc[:, 'i'] = nm + np.arange(ns, dtype=int)

        # Sort columns
        sorted_cols = _sort_cols(jdata.columns)
        jdata = jdata.reindex(sorted_cols, axis=1, copy=False)
        sdata = sdata.reindex(sorted_cols, axis=1, copy=False)

        # Convert into BipartiteDataFrame
        jdata = BipartiteDataFrame(jdata, custom_long_es_split_dict={'alpha': False, 'eps': True})
        sdata = BipartiteDataFrame(sdata, custom_long_es_split_dict={'alpha': False, 'eps': True})

        # Combine into dictionary
        sim_data = {'jdata': jdata, 'sdata': sdata}

        if return_parameters:
            return sim_data, sim_params
        return sim_data
