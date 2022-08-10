'''
Tests Monte Carlo class.
'''
import pytest
import numpy as np
import pytwoway as tw

#######################
##### Monte Carlo #####
#######################

def test_monte_carlo():
    # Use Monte Carlo to test CRE, FE, FE-HO, and FE-HE estimators.
    twmc_net = tw.MonteCarlo()
    twmc_net.monte_carlo(N=50, ncore=4, rng=np.random.default_rng(1240))

    # Extract results
    var_psi_true = twmc_net.res['var(psi)_true']
    var_psi_cre = twmc_net.res['var(psi)_cre']
    var_psi_fe = twmc_net.res['var(psi)_fe']
    var_psi_ho = twmc_net.res['var(psi)_ho']
    var_psi_he = twmc_net.res['var(psi)_he']
    cov_psi_alpha_true = twmc_net.res['cov(psi, alpha)_true']
    cov_psi_alpha_cre = twmc_net.res['cov(psi, alpha)_cre']
    cov_psi_alpha_fe = twmc_net.res['cov(psi, alpha)_fe']
    cov_psi_alpha_ho = twmc_net.res['cov(psi, alpha)_ho']
    cov_psi_alpha_he = twmc_net.res['cov(psi, alpha)_he']

    # Compute mean percent differences from truth
    psi_diff_cre = abs(np.mean((var_psi_cre - var_psi_true) / var_psi_true))
    psi_diff_fe = abs(np.mean((var_psi_fe - var_psi_true) / var_psi_true))
    psi_diff_ho = abs(np.mean((var_psi_ho - var_psi_true) / var_psi_true))
    psi_diff_he = abs(np.mean((var_psi_he - var_psi_true) / var_psi_true))
    psi_alpha_diff_cre = abs(np.mean((cov_psi_alpha_cre - cov_psi_alpha_true) / cov_psi_alpha_true))
    psi_alpha_diff_fe = abs(np.mean((cov_psi_alpha_fe - cov_psi_alpha_true) / cov_psi_alpha_true))
    psi_alpha_diff_ho = abs(np.mean((cov_psi_alpha_ho - cov_psi_alpha_true) / cov_psi_alpha_true))
    psi_alpha_diff_he = abs(np.mean((cov_psi_alpha_he - cov_psi_alpha_true) / cov_psi_alpha_true))

    assert psi_diff_cre < 0.03
    assert psi_diff_fe < 0.02
    assert psi_diff_ho < 1e-2
    assert psi_diff_he < 1e-2
    assert psi_alpha_diff_cre < 1e-2
    assert psi_alpha_diff_fe < 0.015
    assert psi_alpha_diff_ho < 1e-3
    assert psi_alpha_diff_he < 1e-3

#####################
##### Attrition #####
#####################

# def test_attrition_increasing_1():
#     # Test attrition_increasing() and attrition_decreasing(), by checking that the fraction of movers remaining is approximately equal to what `subsets` specifies.

#     # First test increasing, then decreasing
#     subsets_lst = [np.linspace(0.1, 0.5, 5), np.linspace(0.5, 0.1, 5)]
#     attrition_fn = [tw.attrition.attrition_increasing, tw.attrition.attrition_decreasing]

#     for i in range(2):
#         # Non-collapsed
#         bdf = bpd.BipartiteLong(bpd.SimBipartite({'seed': 1234}).sim_network(), track_id_changes=True).clean_data().get_es()

#         orig_n_movers = len(bdf.loc[bdf['m'] == 1, 'i'].unique())
#         n_movers = []

#         for j in attrition_fn[i](bdf, subsets_lst[i], rng=np.random.default_rng(1234)):
#             n_movers.append(len(j.loc[j['m'] == 1, 'i'].unique()))

#         n_movers_vs_subsets = abs((np.array(n_movers) / orig_n_movers) - subsets_lst[i])

#         assert np.max(n_movers_vs_subsets) < 2e-4

#         # Collapsed
#         bdf = bpd.BipartiteLong(bpd.SimBipartite({'seed': 1234}).sim_network(), track_id_changes=True).clean_data().get_collapsed_long().get_es()

#         orig_n_movers = len(bdf.loc[bdf['m'] == 1, 'i'].unique())
#         n_movers = []

#         for j in attrition_fn[i](bdf, subsets_lst[i], rng=np.random.default_rng(1234)):
#             n_movers.append(len(j.loc[j['m'] == 1, 'i'].unique()))

#         n_movers_vs_subsets = abs((np.array(n_movers) / orig_n_movers) - subsets_lst[i])

#         assert np.max(n_movers_vs_subsets) < 2e-4
