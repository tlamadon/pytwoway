'''
Implement the interacted estimator from Bonhomme, Lamadon, & Manresa.
'''
from tqdm.auto import tqdm, trange
import numpy as np
from scipy.sparse.linalg import eigs
from bipartitepandas.util import ChainedAssignment

# rng = np.random.default_rng(None)
# a = bpd.BipartiteDataFrame(bpd.SimBipartite(bpd.sim_params({'n_workers': 1000})).simulate()).clean().collapse(level='match').construct_artificial_time()
# jdata = a[a.get_worker_m()].clean()

class InteractedBLMModel():
    '''
    Class for estimating interacted-BLM.
    '''
    def __init__(self):
        pass

    def fit_b(self, jdata, rng=None):
        '''
        Fit estimator.

        Arguments:
            jdata (BipartitePandas DataFrame): data for movers
            rng (np.random.Generator or None): NumPy random number generator; None is equivalent to np.random.default_rng(None)

        Returns:
            (NumPy Array): estimated b
        '''
        if rng is None:
            rng = np.random.default_rng(None)

        # Initial data construction
        nj = jdata.n_firms()
        j = jdata.loc[:, 'j'].to_numpy()
        y = jdata.loc[:, 'y'].to_numpy()
        # Construct graph
        G, _ = jdata._construct_graph(connectedness='leave_out_observation', is_sorted=True, copy=False)
        # Construct matrix to store results
        M0 = np.zeros(shape=(nj, nj))

        for j1 in trange(nj):
            ### Iterate over all firms ###
            # For each firm, find its neighboring firms
            j1_neighbors = G.neighborhood(j1, order=2, mindist=2)
            # Find workers who worked at firm j1
            obs_in_j1 = (j == j1)
            jdata.loc[:, 'obs_in_j1'] = obs_in_j1
            i_in_j1 = jdata.groupby('i', sort=False)['obs_in_j1'].transform('max').to_numpy()
            # Take subsets of data for workers who worked at firm j1
            jdata_j1 = jdata.loc[i_in_j1, :]
            j_j1 = j[i_in_j1]
            y_j1 = y[i_in_j1]
            for j2 in tqdm(j1_neighbors):
                ### Iterate over all neighbors ###
                # FIXME use symmetry and add condition j2 > j1?
                # Find workers who worked at both firms j1 and j2
                obs_in_j2 = (j_j1 == j2)
                with ChainedAssignment():
                    jdata_j1.loc[:, 'obs_in_j2'] = obs_in_j2
                i_in_j2 = jdata_j1.groupby('i', sort=False)['obs_in_j2'].transform('max').to_numpy()
                # Take subsets of data for workers who worked at both firms j1 and j2
                j_j12 = j_j1[i_in_j2]
                y_j12 = y_j1[i_in_j2]
                # Take subsets of data specifically for firms j1 and j2
                is_j12 = (j_j12 == j1) | (j_j12 == j2)
                j_j12 = j_j12[is_j12]
                y_j12 = y_j12[is_j12]
                if len(j_j12) >= 4:
                    ## If there are at least two workers with observations at both firms ##
                    # Split data for j1 and j2
                    y_j11 = y_j12[j_j12 == j1]
                    y_j22 = y_j12[j_j12 == j2]
                    # Split observations into entering/exiting groups
                    j_j12_first = j_j12[np.arange(len(j_j12)) % 2 == 0]
                    entering = (j_j12_first == j2)
                    exiting = (j_j12_first == j1)
                    if (np.sum(entering) > 0) and (np.sum(exiting) > 0):
                        # Need workers to both enter and exit
                        entering_y1 = np.mean(y_j11[entering])
                        entering_y2 = np.mean(y_j22[entering])
                        exiting_y1 = np.mean(y_j11[exiting])
                        exiting_y2 = np.mean(y_j22[exiting])
                        # Compute M0
                        M0[j1, j2] = (entering_y2 - exiting_y2) / (entering_y1 - exiting_y1)
        S0 = np.sum(M0, axis=1)

        _, evec = eigs((M0.T / S0).T, k=1, sigma=1)

        return evec
