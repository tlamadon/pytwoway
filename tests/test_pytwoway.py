'''
Tests for pytwoway
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pytwoway as tw

##############
##### FE #####
##############

def test_fe_ho_1():
    # Continuous time, 1 mover between firms 1 and 2, 1 between firms 2 and 4, and 1 stayer at firm 4, firm 4 gets reset to firm 3, and discontinuous time still counts as a move
    # psi1 = 5, psi2 = 3, psi4 = 4
    # alpha1 = 3, alpha2 = 2, alpha3 = 4
    worker_data = []
    worker_data.append({'firm': 0, 'time': 1, 'id': 0, 'comp': 8., 'index': 0})
    worker_data.append({'firm': 1, 'time': 2, 'id': 0, 'comp': 6., 'index': 1})
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 5., 'index': 2})
    worker_data.append({'firm': 3, 'time': 2, 'id': 1, 'comp': 6., 'index': 3})
    worker_data.append({'firm': 3, 'time': 1, 'id': 2, 'comp': 8., 'index': 4})
    worker_data.append({'firm': 3, 'time': 2, 'id': 2, 'comp': 8., 'index': 5})

    df = pd.concat([pd.DataFrame(worker, index=[worker['index']]) for worker in worker_data])

    col_dict = {'i': 'id', 'j': 'firm', 'y': 'comp', 't': 'time'}

    bdf = bpd.BipartiteLong(df, col_dict=col_dict)
    bdf = bdf.clean_data()
    bdf = bdf.get_collapsed_long()
    bdf = bdf.get_es()

    fe_params = {'ncore': 1, 'batch': 1, 'ndraw_pii': 50, 'levfile': '', 'ndraw_tr': 5, 'h2': False, 'out': 'res_fe.json',  'statsonly': False, 'Q': 'cov(alpha, psi)'}

    fe_solver = tw.FEEstimator(bdf.get_cs(), fe_params)
    fe_solver.fit_1()
    fe_solver.construct_Q()
    fe_solver.fit_2()

    psi_hat, alpha_hat = fe_solver.get_fe_estimates()

    assert abs(psi_hat[0] - 1) < 1e-5
    assert abs(psi_hat[1] + 1) < 1e-5
    assert abs(alpha_hat[0] - 7) < 1e-5
    assert abs(alpha_hat[1] - 6) < 1e-5
    assert abs(alpha_hat[2] - 8) < 1e-5

#######################
##### Monte Carlo #####
#######################

def test_fe_cre_1():
    # Use Monte Carlo to test FE, FE-HO, and CRE estimators.
    twmc_net = tw.TwoWayMonteCarlo()
    twmc_net.twfe_monte_carlo(N=50, ncore=1) # Can't do multiprocessing with Travis

    # Extract results
    true_psi_var = twmc_net.res['true_psi_var']
    true_psi_alpha_cov = twmc_net.res['true_psi_alpha_cov']
    fe_psi_var = twmc_net.res['fe_psi_var']
    fe_psi_alpha_cov = twmc_net.res['fe_psi_alpha_cov']
    fe_corr_psi_var = twmc_net.res['fe_corr_psi_var']
    fe_corr_psi_alpha_cov = twmc_net.res['fe_corr_psi_alpha_cov']
    cre_psi_var = twmc_net.res['cre_psi_var']
    cre_psi_alpha_cov = twmc_net.res['cre_psi_alpha_cov']

    # Compute mean percent differences from truth
    fe_psi_diff = np.mean(abs((fe_psi_var - true_psi_var) / true_psi_var))
    fe_psi_alpha_diff = np.mean(abs((fe_psi_alpha_cov - true_psi_alpha_cov) / true_psi_alpha_cov))
    fe_corr_psi_diff = np.mean(abs((fe_corr_psi_var - true_psi_var) / true_psi_var))
    fe_corr_psi_alpha_diff = np.mean(abs((fe_corr_psi_alpha_cov - true_psi_alpha_cov) / true_psi_alpha_cov))
    cre_psi_diff = np.mean(abs((cre_psi_var - true_psi_var) / true_psi_var))
    cre_psi_alpha_diff = np.mean(abs((cre_psi_alpha_cov - true_psi_alpha_cov) / true_psi_alpha_cov))

    assert fe_psi_diff < 0.03
    assert fe_psi_alpha_diff < 0.03
    assert fe_corr_psi_diff < 0.025
    assert fe_corr_psi_alpha_diff < 0.025
    assert cre_psi_diff < 0.025
    assert cre_psi_alpha_diff < 0.025

###############
##### BLM #####
###############

def test_blm_monotonic_1_1():
    # Test whether BLM likelihoods are monotonic, using default fit.
    nl = 6
    nk = 10
    mmult = 100
    smult = 100
    # Initiate BLMModel object
    blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True})
    # Make variance of worker types small
    blm_true.S1 /= 4
    blm_true.S2 /= 4
    jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
    sdata = blm_true._m2_mixt_simulate_stayers(blm_true.NNs * smult)
    blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 30})
    blm_fit.fit_movers(jdata)
    blm_fit.fit_stayers(sdata)
    liks1 = blm_fit.liks1[2:] - blm_fit.liks1[1: - 1] # Skip first
    liks0 = blm_fit.liks0[2:] - blm_fit.liks0[1: - 1] # Skip first

    assert liks1.min() > 0
    assert liks0.min() > 0

def test_blm_monotonic_1_2():
    # Test whether BLM likelihoods are monotonic, using constrained-unconstrained fit.
    nl = 6
    nk = 10
    mmult = 100
    smult = 100
    # Initiate BLMModel object
    blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True})
    # Make variance of worker types small
    blm_true.S1 /= 4
    blm_true.S2 /= 4
    jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
    sdata = blm_true._m2_mixt_simulate_stayers(blm_true.NNs * smult)
    blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 30})
    blm_fit.fit_movers_cstr_uncstr(jdata)
    blm_fit.fit_stayers(sdata)
    liks1 = blm_fit.liks1[2:] - blm_fit.liks1[1: - 1] # Skip first
    liks0 = blm_fit.liks0[2:] - blm_fit.liks0[1: - 1] # Skip first

    assert liks1.min() > 0
    assert liks0.min() > 0

def test_blm_qi_1():
    # Test whether BLM posterior probabilities are giving the most weight to the correct type.
    nl = 3
    nk = 4
    mmult = 1
    # Initiate BLMModel object
    blm = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True, 'return_qi': True})
    # Make variance of worker types small
    blm.S1 /= 10
    blm.S2 /= 10
    jdata = blm._m2_mixt_simulate_movers(blm.NNm * mmult)
    # Update BLM class attributes to equal model's
    # Estimate qi matrix
    qi_estimate = blm.fit_movers(jdata)
    max_qi_col = np.argmax(qi_estimate, axis=1)
    n_correct_qi = np.sum(max_qi_col == jdata['l'])

    assert (n_correct_qi / len(max_qi_col)) >= 0.95

# def test_blm_A_3():
#     # Test whether BLM estimates A properly, given true S and pk1.
#     nl = 6
#     nk = 10
#     mmult = 100
#     min_A1 = np.inf
#     min_A2 = np.inf
#     lik = - np.inf
#     for i in range(6):
#         # Initiate BLMModel object
#         blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True})
#         # Make variance of worker types small
#         blm_true.S1 /= 4
#         blm_true.S2 /= 4
#         jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
#         blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 400, 'update_s': False, 'update_pk1': False})
#         ## Start at truth for A1 and A2
#         blm_fit.A1 = blm_true.A1.copy()
#         blm_fit.A2 = blm_true.A2.copy()
#         ##
#         blm_fit.S1 = blm_true.S1
#         blm_fit.S2 = blm_true.S2
#         blm_fit.pk1 = blm_true.pk1
#         blm_fit.fit_movers(jdata)
#         # blm_fit._sort_matrices()

#         # Compute average percent difference from truth
#         val_1 = abs(np.mean(
#             (blm_true.A1.flatten() - blm_fit.A1.flatten()) / blm_true.A1.flatten()
#         ))
#         val_2 = abs(np.mean(
#             (blm_true.A2.flatten() - blm_fit.A2.flatten()) / blm_true.A2.flatten()
#         ))
#         if blm_fit.lik1 > lik:
#             lik = blm_fit.lik1
#             min_A1 = val_1
#             min_A2 = val_2

#     assert min_A1 < 0.2
#     assert min_A2 < 0.1

# def test_blm_S_4():
#     # Test whether BLM estimates S properly, given true A and pk1.
#     nl = 6
#     nk = 10
#     mmult = 100
#     min_S1 = np.inf
#     min_S2 = np.inf
#     lik = - np.inf
#     for i in range(6):
#         # Initiate BLMModel object
#         blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True})
#         # Make variance of worker types small
#         # blm_true.S1 /= 4
#         # blm_true.S2 /= 4
#         jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
#         blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 400, 'update_a': False, 'update_pk1': False})
#         blm_fit.A1 = blm_true.A1
#         blm_fit.A2 = blm_true.A2
#         ## Start at truth for S1 and S2
#         blm_fit.S1 = blm_true.S1.copy()
#         blm_fit.S2 = blm_true.S2.copy()
#         ##
#         blm_fit.pk1 = blm_true.pk1
#         blm_fit.fit_movers(jdata)
#         # blm_fit._sort_matrices()

#         # Compute average percent difference from truth
#         val_1 = abs(np.mean(
#             (blm_true.S1.flatten() - blm_fit.S1.flatten()) / blm_true.S1.flatten()
#         ))
#         val_2 = abs(np.mean(
#             (blm_true.S2.flatten() - blm_fit.S2.flatten()) / blm_true.S2.flatten()
#         ))
#         if blm_fit.lik1 > lik:
#             lik = blm_fit.lik1
#             min_S1 = val_1
#             min_S2 = val_2

#     assert min_S1 < 0.01
#     assert min_S2 < 0.01

# def test_blm_pk_5():
#     # Test whether BLM estimates pk1 and pk0 properly, given true A and S.
#     nl = 6
#     nk = 10
#     mmult = 100
#     smult = 100
#     min_pk1 = np.inf
#     min_pk0 = np.inf
#     lik1 = - np.inf
#     lik0 = - np.inf
#     for i in range(6):
#         # Initiate BLMModel object
#         blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True})
#         # Make variance of worker types small
#         blm_true.S1 /= 10
#         blm_true.S2 /= 10
#         jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
#         sdata = blm_true._m2_mixt_simulate_stayers(blm_true.NNs * smult)
#         blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 400, 'update_a': False, 'update_s': False})
#         blm_fit.A1 = blm_true.A1
#         blm_fit.A2 = blm_true.A2
#         blm_fit.S1 = blm_true.S1
#         blm_fit.S2 = blm_true.S2
#         ## Start at truth for pk1
#         blm_fit.pk1 = blm_true.pk1.copy()
#         ##
#         blm_fit.fit_movers(jdata)
#         # blm_fit._sort_matrices()
#         blm_fit.fit_stayers(sdata)
#         # blm_fit._sort_matrices()

#         # Compute average percent difference from truth
#         val_1 = abs(np.mean(
#             (blm_true.pk1.flatten() - blm_fit.pk1.flatten()) / blm_true.pk1.flatten()
#         ))
#         val_0 = abs(np.mean(
#             (blm_true.pk0.flatten() - blm_fit.pk0.flatten()) / blm_true.pk0.flatten()
#         ))
#         if blm_fit.lik1 > lik1:
#             lik1 = blm_fit.lik1
#             min_pk1 = val_1
#         if blm_fit.lik0 > lik0:
#             lik0 = blm_fit.lik0
#             min_pk0 = val_0

#     assert min_pk1 < 0.01
#     assert min_pk0 < 0.7 # This error has gone up to 0.684

# def test_blm_fit_6_1():
#     # Test whether BLM fit_movers() method works properly.
#     nl = 6
#     nk = 10
#     mmult = 100
#     smult = 100
#     min_A1 = np.inf
#     min_A2 = np.inf
#     min_S1 = np.inf
#     min_S2 = np.inf
#     min_pk1 = np.inf
#     min_pk0 = np.inf
#     lik1 = - np.inf
#     lik0 = - np.inf
#     for i in range(5):
#         # Initiate BLMModel object
#         blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True})
#         # Make variance of worker types small
#         blm_true.S1 /= 4
#         blm_true.S2 /= 4
#         jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
#         sdata = blm_true._m2_mixt_simulate_stayers(blm_true.NNs * smult)
#         blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 400})
#         ## Start at truth for all parameters
#         blm_fit.A1 = blm_true.A1.copy()
#         blm_fit.A2 = blm_true.A2.copy()
#         blm_fit.S1 = blm_true.S1.copy()
#         blm_fit.S2 = blm_true.S2.copy()
#         blm_fit.pk1 = blm_true.pk1.copy()
#         ##
#         blm_fit.fit_movers(jdata)
#         # blm_fit._sort_matrices()
#         blm_fit.fit_stayers(sdata)
#         # blm_fit._sort_matrices()

#         # Compute average percent difference from truth
#         val_A1 = abs(np.mean(
#             (blm_true.A1.flatten() - blm_fit.A1.flatten()) / blm_true.A1.flatten()
#         ))
#         val_A2 = abs(np.mean(
#             (blm_true.A2.flatten() - blm_fit.A2.flatten()) / blm_true.A2.flatten()
#         ))
#         val_S1 = abs(np.mean(
#             (blm_true.S1.flatten() - blm_fit.S1.flatten()) / blm_true.S1.flatten()
#         ))
#         val_S2 = abs(np.mean(
#             (blm_true.S2.flatten() - blm_fit.S2.flatten()) / blm_true.S2.flatten()
#         ))
#         val_pk1 = abs(np.mean(
#             (blm_true.pk1.flatten() - blm_fit.pk1.flatten()) / blm_true.pk1.flatten()
#         ))
#         val_pk0 = abs(np.mean(
#             (blm_true.pk0.flatten() - blm_fit.pk0.flatten()) / blm_true.pk0.flatten()
#         ))
#         if blm_fit.lik1 > lik1:
#             lik1 = blm_fit.lik1
#             min_A1 = val_A1
#             min_A2 = val_A2
#             min_S1 = val_S1
#             min_S2 = val_S2
#             min_pk1 = val_pk1
#         if blm_fit.lik0 > lik0:
#             lik0 = blm_fit.lik0
#             min_pk0 = val_pk0

#     # Compute average percent difference from truth
#     assert min_A1 < 0.01
#     assert min_A2 < 0.01
#     assert min_S1 < 0.3
#     assert min_S2 < 0.1
#     assert min_pk1 < 6
#     assert min_pk0 < 3

# def test_blm_fit_6_2():
#     # Test whether BLM fit_movers_cstr_uncstr() method works properly.
#     nl = 6
#     nk = 10
#     mmult = 100
#     smult = 100
#     min_A1 = np.inf
#     min_A2 = np.inf
#     min_S1 = np.inf
#     min_S2 = np.inf
#     min_pk1 = np.inf
#     min_pk0 = np.inf
#     lik1 = - np.inf
#     lik0 = - np.inf
#     for i in range(5):
#         # Initiate BLMModel object
#         blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True})
#         # Make variance of worker types small
#         blm_true.S1 /= 4
#         blm_true.S2 /= 4
#         jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
#         sdata = blm_true._m2_mixt_simulate_stayers(blm_true.NNs * smult)
#         blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 400})
#         ## Start at truth for all parameters
#         blm_fit.A1 = blm_true.A1.copy()
#         blm_fit.A2 = blm_true.A2.copy()
#         blm_fit.S1 = blm_true.S1.copy()
#         blm_fit.S2 = blm_true.S2.copy()
#         blm_fit.pk1 = blm_true.pk1.copy()
#         ##
#         blm_fit.fit_movers_cstr_uncstr(jdata)
#         # blm_fit._sort_matrices()
#         blm_fit.fit_stayers(sdata)
#         # blm_fit._sort_matrices()

#         # Compute average percent difference from truth
#         val_A1 = abs(np.mean(
#             (blm_true.A1.flatten() - blm_fit.A1.flatten()) / blm_true.A1.flatten()
#         ))
#         val_A2 = abs(np.mean(
#             (blm_true.A2.flatten() - blm_fit.A2.flatten()) / blm_true.A2.flatten()
#         ))
#         val_S1 = abs(np.mean(
#             (blm_true.S1.flatten() - blm_fit.S1.flatten()) / blm_true.S1.flatten()
#         ))
#         val_S2 = abs(np.mean(
#             (blm_true.S2.flatten() - blm_fit.S2.flatten()) / blm_true.S2.flatten()
#         ))
#         val_pk1 = abs(np.mean(
#             (blm_true.pk1.flatten() - blm_fit.pk1.flatten()) / blm_true.pk1.flatten()
#         ))
#         val_pk0 = abs(np.mean(
#             (blm_true.pk0.flatten() - blm_fit.pk0.flatten()) / blm_true.pk0.flatten()
#         ))
#         if blm_fit.lik1 > lik1:
#             lik1 = blm_fit.lik1
#             min_A1 = val_A1
#             min_A2 = val_A2
#             min_S1 = val_S1
#             min_S2 = val_S2
#             min_pk1 = val_pk1
#         if blm_fit.lik0 > lik0:
#             lik0 = blm_fit.lik0
#             min_pk0 = val_pk0

#     # Compute average percent difference from truth
#     assert min_A1 < 0.15
#     assert min_A2 < 0.05
#     assert min_S1 < 0.05
#     assert min_S2 < 0.15
#     assert min_pk1 < 0.75
#     assert min_pk0 < 15
