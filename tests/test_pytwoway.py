'''
Tests for pytwoway
'''
import pytest
import numpy as np
import pandas as pd
import bipartitepandas as bpd
import pytwoway as tw
from scipy.sparse import csc_matrix

##############
##### FE #####
##############

def get_fe_estimates(fe_solver):
    # Get (psi_hat_dict, alpha_hat_dict) from fe_solver
    j_vals = np.arange(fe_solver.nf)
    i_vals = np.arange(fe_solver.nw)
    psi_hat_dict = dict(zip(j_vals, np.concatenate([fe_solver.psi_hat, np.array([0])]))) # Add 0 for normalized firm
    alpha_hat_dict = dict(zip(i_vals, fe_solver.alpha_hat))

    return psi_hat_dict, alpha_hat_dict

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

    bdf = bpd.BipartiteLong(df, col_dict=col_dict).clean_data().get_collapsed_long()

    fe_params = {'ncore': 1, 'batch': 1, 'ndraw_pii': 50, 'levfile': '', 'ndraw_tr': 5, 'h2': False, 'out': 'res_fe.json',  'statsonly': False, 'Q': 'cov(alpha, psi)', 'seed': 1234}

    fe_solver = tw.FEEstimator(bdf, fe_params)
    fe_solver.fit_1()
    fe_solver.construct_Q()
    fe_solver.fit_2()

    psi_hat, alpha_hat = get_fe_estimates(fe_solver)

    assert abs(psi_hat[0] - 1) < 1e-10
    assert abs(psi_hat[1] + 1) < 1e-10
    assert abs(psi_hat[2]) < 1e-10
    assert abs(alpha_hat[0] - 7) < 1e-10
    assert abs(alpha_hat[1] - 6) < 1e-10
    assert abs(alpha_hat[2] - 8) < 1e-10

def test_fe_weights_2_a():
    # Test that FE weights are computed correctly.
    a = bpd.SimBipartite({'seed': 1234}).sim_network()
    # Non-collapsed data
    b = bpd.BipartiteLong(a).clean_data()
    # Collapsed data
    c = bpd.BipartiteLong(a).clean_data().get_collapsed_long()

    assert np.sum(c['w']) == len(b)

def test_fe_weights_2_b():
    # Test that FE weights are computed correctly.
    a = bpd.SimBipartite({'seed': 1234}).sim_network()
    # Non-collapsed data
    b = bpd.BipartiteLong(a).clean_data() # .get_es().get_cs()
    # Collapsed data
    c = bpd.BipartiteLong(a).clean_data().get_collapsed_long().get_es().get_long() # get_cs()

    assert np.sum(c['w']) == len(b)

def test_fe_weights_3():
    # Test that FE weights are computing sigma^2 plug-in sigma^2 bias-corrected, and vars/covs correctly with full data by trying with and withought weights.
    a = bpd.SimBipartite({'seed': 1234}).sim_network()
    # Simulate without weights
    b = bpd.BipartiteLong(a).clean_data().gen_m()
    fe_solver_b = tw.FEEstimator(b, {'seed': 1234})
    fe_solver_b.fit_1()
    fe_solver_b.construct_Q()
    fe_solver_b.fit_2()
    # Simulate with weight 1
    c = bpd.BipartiteLong(a).clean_data().gen_m()
    c['w'] = 1
    fe_solver_c = tw.FEEstimator(c, {'seed': 1234})
    fe_solver_c.fit_1()
    fe_solver_c.construct_Q()
    fe_solver_c.fit_2()

    # Collect sigma^2 plug-in
    sigma_pi_b = fe_solver_b.var_e_pi
    sigma_pi_c = fe_solver_c.var_e_pi

    assert sigma_pi_b == sigma_pi_c

    # Collect sigma^2 bias-corrected
    sigma_bc_b = fe_solver_b.var_e
    sigma_bc_c = fe_solver_c.var_e

    assert abs((sigma_bc_b - sigma_bc_c) / sigma_bc_b) < 5e-3

    # Collect vars/covs
    res_b = fe_solver_b.summary
    res_c = fe_solver_c.summary

    assert abs((res_b['var_fe'] - res_c['var_fe']) / res_b['var_fe']) < 1e-10
    assert abs((res_b['cov_fe'] - res_c['cov_fe']) / res_b['cov_fe']) < 1e-10
    assert abs((res_b['var_ho'] - res_c['var_ho']) / res_b['var_ho']) < 1e-2
    assert abs((res_b['cov_ho'] - res_c['cov_ho']) / res_b['cov_ho']) < 2e-2

def test_fe_weights_4():
    # Test that FE weights are computing all parameters correctly with simple data.
    worker_data = []
    worker_data.append({'firm': 0, 'time': 1, 'id': 0, 'comp': 8.})
    worker_data.append({'firm': 1, 'time': 2, 'id': 0, 'comp': 6.})
    worker_data.append({'firm': 1, 'time': 3, 'id': 0, 'comp': 6.})
    worker_data.append({'firm': 4, 'time': 4, 'id': 0, 'comp': 7.})
    worker_data.append({'firm': 5, 'time': 5, 'id': 0, 'comp': 9.})
    worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 5.})
    worker_data.append({'firm': 3, 'time': 2, 'id': 1, 'comp': 6.})
    worker_data.append({'firm': 3, 'time': 1, 'id': 2, 'comp': 8.})
    worker_data.append({'firm': 3, 'time': 2, 'id': 2, 'comp': 8.})

    a = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])#.sort_values(['id', 'time'])
    col_dict = {'i': 'id', 'j': 'firm', 'y': 'comp', 't': 'time'}
    # Simulate on non-collapsed data
    b = bpd.BipartiteLong(a, col_dict=col_dict).clean_data().gen_m()
    fe_solver_b = tw.FEEstimator(b, {'seed': 1234})
    fe_solver_b.fit_1()
    fe_solver_b.construct_Q()
    fe_solver_b.fit_2()
    # Simulate on collapsed data
    c = bpd.BipartiteLong(a, col_dict=col_dict).clean_data().get_collapsed_long()
    fe_solver_c = tw.FEEstimator(c, {'seed': 1234})
    fe_solver_c.fit_1()
    fe_solver_c.construct_Q()
    fe_solver_c.fit_2()
    # Collect parameter estimates
    b_psi, b_alpha = get_fe_estimates(fe_solver_b)
    c_psi, c_alpha = get_fe_estimates(fe_solver_c)
    # Convert to dataframes
    b_psi = pd.DataFrame(list(b_psi.items()), columns=['fid', 'psi'])
    b_alpha = pd.DataFrame(list(b_alpha.items()), columns=['wid', 'alpha'])
    c_psi = pd.DataFrame(list(c_psi.items()), columns=['fid', 'psi'])
    c_alpha = pd.DataFrame(list(c_alpha.items()), columns=['wid', 'alpha'])
    # Test correlation
    assert abs(np.corrcoef(b_psi['psi'], c_psi['psi'])[0, 1] - 1) < 1e-10
    assert abs(np.corrcoef(b_alpha['alpha'], c_alpha['alpha'])[0, 1] - 1) < 1e-10
    # Test coefficients
    for i in range(len(b_psi)):
        assert abs(b_psi.iloc[i]['psi'] - c_psi.iloc[i]['psi']) < 1e-10
    for i in range(len(b_alpha)):
        assert abs(b_alpha.iloc[i]['alpha'] - c_alpha.iloc[i]['alpha']) < 1e-10
    # Collect sigma^2 plug-in
    sigma_pi_b = fe_solver_b.var_e_pi
    sigma_pi_c = fe_solver_c.var_e_pi

    assert abs(sigma_pi_b - sigma_pi_c) < 1e-30

    # Collect sigma^2 bias-corrected
    sigma_bc_b = fe_solver_b.var_e
    sigma_bc_c = fe_solver_c.var_e

    if abs(sigma_bc_b - sigma_bc_c) != np.inf:
        assert abs(sigma_bc_b - sigma_bc_c) < 1e-10 # abs((sigma_bc_b - sigma_bc_c) / sigma_bc_b) < 1e-3
        # Test vars/covs
        res_b = fe_solver_b.summary
        res_c = fe_solver_c.summary

        assert abs((res_b['var_fe'] - res_c['var_fe']) / res_b['var_fe']) < 1e-10
        assert abs((res_b['cov_fe'] - res_c['cov_fe']) / res_b['cov_fe']) < 1e-10
        assert abs((res_b['var_ho'] - res_c['var_ho']) / res_b['var_ho']) < 1e-10
        assert abs((res_b['cov_ho'] - res_c['cov_ho']) / res_b['cov_ho']) < 1e-10

# def test_fe_weights_5(): # FIXME this test doesn't work because un-collapsing data gives different estimates
#     # Test that FE weights are computing all parameters correctly with full data by collapsing then un-collapsing data.
#     a = bpd.SimBipartite({'seed': 1234}).sim_network()
#     # Simulate on non-collapsed data
#     b = bpd.BipartiteLong(a).clean_data().gen_m()
#     fe_solver_b = tw.FEEstimator(b, {'seed': 1234})
#     fe_solver_b.fit_1()
#     fe_solver_b.construct_Q()
#     fe_solver_b.fit_2()
#     # Simulate on collapsed then un-collapsed data
#     c = bpd.BipartiteLong(a).clean_data().get_collapsed_long().uncollapse().drop('w')
#     fe_solver_c = tw.FEEstimator(c, {'seed': 1234})
#     fe_solver_c.fit_1()
#     fe_solver_c.construct_Q()
#     fe_solver_c.fit_2()
#     # Collect parameter estimates
#     b_psi, b_alpha = get_fe_estimates(fe_solver_b)
#     c_psi, c_alpha = get_fe_estimates(fe_solver_c)
#     # Convert to dataframes
#     b_psi = pd.DataFrame(list(b_psi.items()), columns=['fid', 'psi'])
#     b_alpha = pd.DataFrame(list(b_alpha.items()), columns=['wid', 'alpha'])
#     c_psi = pd.DataFrame(list(c_psi.items()), columns=['fid', 'psi'])
#     c_alpha = pd.DataFrame(list(c_alpha.items()), columns=['wid', 'alpha'])
#     # Test correlation
#     assert abs(np.corrcoef(b_psi['psi'], c_psi['psi'])[0, 1] - 1) < 1e-15
#     assert abs(np.corrcoef(b_alpha['alpha'], c_alpha['alpha'])[0, 1] - 1) < 1e-15
#     # Test coefficients
#     for i in range(len(b_psi)):
#         assert abs(b_psi.iloc[i]['psi'] - c_psi.iloc[i]['psi']) < 1e-10
#     for i in range(len(b_alpha)):
#         assert abs(b_alpha.iloc[i]['alpha'] - c_alpha.iloc[i]['alpha']) < 1e-10
#     # Collect sigma^2 bias-corrected
#     sigma_bc_b = fe_solver_b.var_e
#     sigma_bc_c = fe_solver_c.var_e

#     assert abs((sigma_bc_b - sigma_bc_c) / sigma_bc_b) < 1e-3
#     # Test vars/covs
#     res_b = fe_solver_b.summary
#     res_c = fe_solver_c.summary

#     assert abs((res_b['var_fe'] - res_c['var_fe']) / res_b['var_fe']) < 1e-10
#     assert abs((res_b['cov_fe'] - res_c['cov_fe']) / res_b['cov_fe']) < 1e-10
#     assert abs((res_b['var_ho'] - res_c['var_ho']) / res_b['var_ho']) < 1e-2
#     assert abs((res_b['cov_ho'] - res_c['cov_ho']) / res_b['cov_ho']) < 1e-2

def test_fe_weights_6():
    # Test that FE weights are computing alpha, psi, and sigma^2 correctly with full data.
    a = bpd.SimBipartite({'seed': 1234}).sim_network()
    # Simulate on non-collapsed data
    b = bpd.BipartiteLong(a).clean_data().gen_m()
    fe_solver_b = tw.FEEstimator(b, {'seed': 1234})
    fe_solver_b.fit_1()
    fe_solver_b.construct_Q()
    fe_solver_b.fit_2()
    # Simulate on collapsed data
    c = bpd.BipartiteLong(a).clean_data().get_collapsed_long()
    fe_solver_c = tw.FEEstimator(c, {'seed': 1234})
    fe_solver_c.fit_1()
    fe_solver_c.construct_Q()
    fe_solver_c.fit_2()
    # Collect parameter estimates
    b_psi, b_alpha = get_fe_estimates(fe_solver_b)
    c_psi, c_alpha = get_fe_estimates(fe_solver_c)
    # Convert to dataframes
    b_psi = pd.DataFrame(list(b_psi.items()), columns=['fid', 'psi'])
    b_alpha = pd.DataFrame(list(b_alpha.items()), columns=['wid', 'alpha'])
    c_psi = pd.DataFrame(list(c_psi.items()), columns=['fid', 'psi'])
    c_alpha = pd.DataFrame(list(c_alpha.items()), columns=['wid', 'alpha'])
    # Test correlation
    assert abs(np.corrcoef(b_psi['psi'], c_psi['psi'])[0, 1] - 1) < 1e-10
    assert abs(np.corrcoef(b_alpha['alpha'], c_alpha['alpha'])[0, 1] - 1) < 1e-10
    # Test coefficients
    for i in range(len(b_psi)):
        assert abs(b_psi.iloc[i]['psi'] - c_psi.iloc[i]['psi']) < 1e-8
    for i in range(len(b_alpha)):
        assert abs(b_alpha.iloc[i]['alpha'] - c_alpha.iloc[i]['alpha']) < 1e-8
    # # Collect sigma^2 plug-in # FIXME these won't be the same
    # sigma_pi_b = fe_solver_b.var_e_pi
    # sigma_pi_c = fe_solver_c.var_e_pi

    # assert sigma_pi_b == sigma_pi_c

    # Collect sigma^2 bias-corrected
    sigma_bc_b = fe_solver_b.var_e
    sigma_bc_c = fe_solver_c.var_e

    if abs(sigma_bc_b - sigma_bc_c) != np.inf:
        assert abs((sigma_bc_b - sigma_bc_c) / sigma_bc_b) < 2e-2
        # Test against true sigma^2
        a['E'] = a['y'] - a['alpha'] - a['psi']
        sigma_true = a['E'].var()
        assert abs(sigma_bc_b - sigma_true) / abs(sigma_true) < 6e-3
        assert abs(sigma_bc_c - sigma_true) / abs(sigma_true) < 2e-2
        # Test vars/covs
        res_b = fe_solver_b.summary
        res_c = fe_solver_c.summary

        assert abs((res_b['var_fe'] - res_c['var_fe']) / res_b['var_fe']) < 5e-10
        assert abs((res_b['cov_fe'] - res_c['cov_fe']) / res_b['cov_fe']) < 5e-10
        assert abs((res_b['var_ho'] - res_c['var_ho']) / res_b['var_ho']) < 1e-2
        assert abs((res_b['cov_ho'] - res_c['cov_ho']) / res_b['cov_ho']) < 2e-2

# The purpose of the following code is to measure the importance of correctly weighting observations, but it doesn't actually contain any tests
# def test_fe_weights_7():
#     # Test that FE weights are computing weighted parameters correctly with simple data that has high dependence on weighting.
#     # Firm 0: 1; 1: -1; 2: 15; 3: 0; 4: 0; 5: 2 
#     # Worker 0: 7; 1: 6; 2: 8; 3: 51; 4: -50
#     worker_data = []
#     worker_data.append({'firm': 0, 'time': 1, 'id': 0, 'comp': 8.})
#     worker_data.append({'firm': 1, 'time': 2, 'id': 0, 'comp': 6.})
#     worker_data.append({'firm': 1, 'time': 3, 'id': 0, 'comp': 6.})
#     worker_data.append({'firm': 4, 'time': 4, 'id': 0, 'comp': 7.})
#     worker_data.append({'firm': 5, 'time': 5, 'id': 0, 'comp': 9.})
#     worker_data.append({'firm': 1, 'time': 1, 'id': 1, 'comp': 5.})
#     worker_data.append({'firm': 3, 'time': 2, 'id': 1, 'comp': 6.})
#     worker_data.append({'firm': 3, 'time': 1, 'id': 2, 'comp': 8.})
#     worker_data.append({'firm': 3, 'time': 2, 'id': 2, 'comp': 8.})
#     worker_data.append({'firm': 5, 'time': 1, 'id': 3, 'comp': 53.})
#     worker_data.append({'firm': 5, 'time': 2, 'id': 3, 'comp': 53.})
#     worker_data.append({'firm': 5, 'time': 3, 'id': 3, 'comp': 53.})
#     worker_data.append({'firm': 5, 'time': 4, 'id': 3, 'comp': 53.})
#     worker_data.append({'firm': 5, 'time': 5, 'id': 3, 'comp': 53.})
#     worker_data.append({'firm': 3, 'time': 6, 'id': 3, 'comp': 51.})
#     worker_data.append({'firm': 3, 'time': 7, 'id': 3, 'comp': 51.})
#     worker_data.append({'firm': 3, 'time': 8, 'id': 3, 'comp': 51.})
#     worker_data.append({'firm': 3, 'time': 9, 'id': 3, 'comp': 51.})
#     worker_data.append({'firm': 3, 'time': 10, 'id': 3, 'comp': 51.})
#     worker_data.append({'firm': 4, 'time': 1, 'id': 4, 'comp': -50.})
#     worker_data.append({'firm': 4, 'time': 2, 'id': 4, 'comp': -50.})
#     worker_data.append({'firm': 4, 'time': 3, 'id': 4, 'comp': -50.})
#     worker_data.append({'firm': 4, 'time': 4, 'id': 4, 'comp': -50.})
#     worker_data.append({'firm': 4, 'time': 5, 'id': 4, 'comp': -50.})
#     worker_data.append({'firm': 2, 'time': 6, 'id': 4, 'comp': -35.})
#     worker_data.append({'firm': 2, 'time': 7, 'id': 4, 'comp': -35.})
#     worker_data.append({'firm': 2, 'time': 8, 'id': 4, 'comp': -35.})
#     worker_data.append({'firm': 2, 'time': 9, 'id': 4, 'comp': -35.})
#     worker_data.append({'firm': 2, 'time': 10, 'id': 4, 'comp': -35.})

#     a = pd.concat([pd.DataFrame(worker, index=[i]) for i, worker in enumerate(worker_data)])#.sort_values(['id', 'time'])
#     col_dict = {'i': 'id', 'j': 'firm', 'y': 'comp', 't': 'time'}
#     # Simulate on non-collapsed data
#     b = bpd.BipartiteLong(a, col_dict=col_dict).clean_data().gen_m()
#     fe_solver_b = tw.FEEstimator(b, {'seed': 1234})
#     fe_solver_b.fit_1()
#     fe_solver_b.construct_Q()
#     fe_solver_b.fit_2()
#     # Simulate on collapsed data with weights all reset to 1
#     c = bpd.BipartiteLong(a, col_dict=col_dict).clean_data().get_collapsed_long()
#     c['w'] = 1
#     fe_solver_c = tw.FEEstimator(c, {'seed': 1234})
#     fe_solver_c.fit_1()
#     fe_solver_c.construct_Q()
#     fe_solver_c.fit_2()
#     # Simulate on collapsed data with correct weights
#     d = bpd.BipartiteLong(a, col_dict=col_dict).clean_data().get_collapsed_long()
#     fe_solver_d = tw.FEEstimator(d, {'seed': 1234})
#     fe_solver_d.fit_1()
#     fe_solver_d.construct_Q()
#     fe_solver_d.fit_2()
#     # Collect parameter estimates
#     b_psi, b_alpha = get_fe_estimates(fe_solver_b)
#     c_psi, c_alpha = get_fe_estimates(fe_solver_c)
#     d_psi, d_alpha = get_fe_estimates(fe_solver_d)
#     # Convert to dataframes
#     b_psi = pd.DataFrame(list(b_psi.items()), columns=['fid', 'psi'])
#     b_alpha = pd.DataFrame(list(b_alpha.items()), columns=['wid', 'alpha'])
#     c_psi = pd.DataFrame(list(c_psi.items()), columns=['fid', 'psi'])
#     c_alpha = pd.DataFrame(list(c_alpha.items()), columns=['wid', 'alpha'])
#     d_psi = pd.DataFrame(list(d_psi.items()), columns=['fid', 'psi'])
#     d_alpha = pd.DataFrame(list(d_alpha.items()), columns=['wid', 'alpha'])
#     # # Test correlation
#     # assert abs(np.corrcoef(b_psi['psi'], c_psi['psi'])[0, 1] - 1) < 1e-10
#     # assert abs(np.corrcoef(b_alpha['alpha'], c_alpha['alpha'])[0, 1] - 1) < 1e-10
#     # # Test coefficients
#     # for i in range(len(b_psi)):
#     #     assert abs(b_psi.iloc[i]['psi'] - c_psi.iloc[i]['psi']) < 1e-10
#     # for i in range(len(b_alpha)):
#     #     assert abs(b_alpha.iloc[i]['alpha'] - c_alpha.iloc[i]['alpha']) < 1e-10
#     # Collect sigma^2 plug-in
#     sigma_pi_b = fe_solver_b.var_e_pi
#     sigma_pi_c = fe_solver_c.var_e_pi
#     sigma_pi_d = fe_solver_d.var_e_pi

#     # assert abs(sigma_pi_b - sigma_pi_c) < 1e-30

#     # Collect sigma^2 bias-corrected
#     sigma_bc_b = fe_solver_b.var_e
#     sigma_bc_c = fe_solver_c.var_e
#     sigma_bc_d = fe_solver_d.var_e

#     # assert abs((sigma_bc_b - sigma_bc_c) / sigma_bc_b) < 1e-3
#     # Test vars/covs
#     res_b = fe_solver_b.summary
#     res_c = fe_solver_c.summary
#     res_d = fe_solver_d.summary

#     # assert abs((res_b['var_fe'] - res_c['var_fe']) / res_b['var_fe']) < 1e-10
#     # assert abs((res_b['cov_fe'] - res_c['cov_fe']) / res_b['cov_fe']) < 1e-10
#     # assert abs((res_b['var_ho'] - res_c['var_ho']) / res_b['var_ho']) < 1e-10
#     # assert abs((res_b['cov_ho'] - res_c['cov_ho']) / res_b['cov_ho']) < 1e-10

def construct_Jq_Wq(fe_solver):
    '''
    USED IN TEST test_fe_he_8()
    Construct Jq and Wq matrices.

    Returns:
        Jq (Pandas DataFrame): left matrix for computing Q
        Wq (Pandas DataFrame): right matrix for computing Q
    '''
    # Construct Q matrix
    fe_solver.adata['Jq'] = 1
    fe_solver.adata['Wq'] = 1
    fe_solver.adata['Jq_row'] = fe_solver.adata['Jq'].cumsum() - 1
    fe_solver.adata['Wq_row'] = fe_solver.adata['Wq'].cumsum() - 1
    fe_solver.adata['Jq_col'] = fe_solver.adata['j']
    fe_solver.adata['Wq_col'] = fe_solver.adata['i']
    # Construct Jq, Wq matrices
    Jq = fe_solver.adata[fe_solver.adata['Jq'] == 1].reset_index(drop=True)
    nJ = len(Jq)
    nJ_row = Jq['Jq_row'].max() + 1 # FIXME len(Jq['Jq_row'].unique())
    nJ_col = Jq['Jq_col'].max() + 1 # FIXME len(Jq['Jq_col'].unique())
    Jq = csc_matrix((np.ones(nJ), (Jq['Jq_row'], Jq['Jq_col'])), shape=(nJ_row, nJ_col))
    if nJ_col == fe_solver.nf: # If looking at firms, normalize one to 0
        Jq = Jq[:, range(fe_solver.nf - 1)]

    Wq = fe_solver.adata[fe_solver.adata['Wq'] == 1].reset_index(drop=True)
    nW = len(Wq)
    nW_row = Wq['Wq_row'].max() + 1 # FIXME len(Wq['Wq_row'].unique())
    nW_col = Wq['Wq_col'].max() + 1 # FIXME len(Wq['Wq_col'].unique())
    Wq = csc_matrix((np.ones(nW), (Wq['Wq_row'], Wq['Wq_col'])), shape=(nW_row, nW_col)) # FIXME Should we use nJ because require Jq, Wq to have the same size?
    # if nW_col == self.nf: # If looking at firms, normalize one to 0
    #     Wq = Wq[:, range(self.nf - 1)]

    return Jq, Wq

def test_fe_he_8():
    # Test that HE sigma^i are comparable between non-collapsed and collapsed data
    a = bpd.SimBipartite({'num_ind': 1000, 'seed': 1234}).sim_network()
    # Simulate on non-collapsed data
    b = bpd.BipartiteLong(a).clean_data({'connectedness': 'biconnected'}).gen_m()
    fe_solver_b = tw.FEEstimator(b, {'he': True, 'seed': 1234})
    fe_solver_b.fit_1()
    fe_solver_b.construct_Q()
    fe_solver_b.fit_2()
    # Simulate on collapsed data
    c = bpd.BipartiteLong(a).clean_data({'connectedness': 'biconnected'}).get_collapsed_long()
    fe_solver_c = tw.FEEstimator(c, {'he': True, 'seed': 1234})
    fe_solver_c.fit_1()
    fe_solver_c.construct_Q()
    fe_solver_c.fit_2()

    # Add columns with estimated parameters
    b_params = get_fe_estimates(fe_solver_b)
    b['alpha_hat'] = b['i'].map(b_params[1])
    b['psi_hat'] = b['j'].map(b_params[0])
    c_params = get_fe_estimates(fe_solver_c)
    c['alpha_hat'] = c['i'].map(c_params[1])
    c['psi_hat'] = c['j'].map(c_params[0])

    ##### Create sigma_sq_hat #####
    # Non-collapsed
    # First, create X
    Jq, Wq = construct_Jq_Wq(fe_solver_b)
    Jq = Jq.toarray()
    Wq = Wq.toarray()
    X = np.concatenate([Wq.T, Jq.T]).T # alpha then psi
    # Now compute P_{ii} := X'_i @ (X'X)^{-1} @ X_i
    XX_inv = np.linalg.inv(X.T @ X)
    P_ii = np.diag(X @ XX_inv @ X.T)
    # Compute sigma_sq_hat
    sigma_sq_hat_b = np.expand_dims(b['y'], 1) @ np.expand_dims(b['y'] - b['alpha_hat'] - b['psi_hat'], 1).T / (1 - P_ii)

    # Collapsed
    # First, create X = (D_p)^{1/2}CA
    Jq_collapsed, Wq_collapsed = construct_Jq_Wq(fe_solver_c)
    Jq_collapsed = Jq_collapsed.toarray()
    Wq_collapsed = Wq_collapsed.toarray()
    X_collapsed = np.sqrt(fe_solver_c.Dp) @ np.concatenate([Wq_collapsed.T, Jq_collapsed.T]).T
    # Now compute P_{ii} := X'_i @ (X'X)^{-1} @ X_i
    XX_collapsed_inv = np.linalg.inv(X_collapsed.T @ X_collapsed)
    P_ii_collapsed = np.diag(X_collapsed @ XX_collapsed_inv @ X_collapsed.T)
    # Compute sigma_sq_hat
    sigma_sq_hat_c = c['y'] * (c['y'] - c['alpha_hat'] - c['psi_hat']) / (1 - P_ii_collapsed)

    ##### Create C matrix #####
    # Create spells
    data = b.copy() # b already sorted by i and t
    # Introduce lagged i and j
    data['i_l1'] = data['i'].shift(periods=1)
    data['j_l1'] = data['j'].shift(periods=1)

    # Generate spell ids
    # Source: https://stackoverflow.com/questions/59778744/pandas-grouping-and-aggregating-consecutive-rows-with-same-value-in-column
    new_spell = (data['j'] != data['j_l1']) | (data['i'] != data['i_l1']) # Allow for i != i_l1 to ensure that consecutive workers at the same firm get counted as different spells
    data['spell_id'] = new_spell.cumsum()
    data['spell_length'] = data.groupby('spell_id')['i'].transform('size')

    # Create C
    C = csc_matrix((1 / data['spell_length'], (data['spell_id'] - 1, data.index)), shape=(data['spell_id'].max(), len(data))).toarray()

    ##### Compare #####
    collapsed_b = np.diag(C @ sigma_sq_hat_b @ C.T) * (C @ (1 - P_ii))
    collapsed_c = sigma_sq_hat_c * (1 - P_ii_collapsed)

    assert np.max(abs(collapsed_b - collapsed_c)) < 1e-13

#######################
##### Monte Carlo #####
#######################

def test_fe_cre_1():
    # Use Monte Carlo to test CRE, FE, FE-HO, and FE-HE estimators.
    twmc_net = tw.TwoWayMonteCarlo()
    twmc_net.twfe_monte_carlo(N=50, ncore=1) # Can't do multiprocessing with Travis

    # Extract results
    true_psi_var = twmc_net.res['true_psi_var']
    true_psi_alpha_cov = twmc_net.res['true_psi_alpha_cov']
    cre_psi_var = twmc_net.res['cre_psi_var']
    cre_psi_alpha_cov = twmc_net.res['cre_psi_alpha_cov']
    fe_psi_var = twmc_net.res['fe_psi_var']
    fe_psi_alpha_cov = twmc_net.res['fe_psi_alpha_cov']
    fe_ho_psi_var = twmc_net.res['fe_ho_psi_var']
    fe_ho_psi_alpha_cov = twmc_net.res['fe_ho_psi_alpha_cov']
    fe_he_psi_var = twmc_net.res['fe_he_psi_var']
    fe_he_psi_alpha_cov = twmc_net.res['fe_he_psi_alpha_cov']

    # Compute mean percent differences from truth
    cre_psi_diff = np.mean(abs((cre_psi_var - true_psi_var) / true_psi_var))
    cre_psi_alpha_diff = np.mean(abs((cre_psi_alpha_cov - true_psi_alpha_cov) / true_psi_alpha_cov))
    fe_psi_diff = np.mean(abs((fe_psi_var - true_psi_var) / true_psi_var))
    fe_psi_alpha_diff = np.mean(abs((fe_psi_alpha_cov - true_psi_alpha_cov) / true_psi_alpha_cov))
    fe_ho_psi_diff = np.mean(abs((fe_ho_psi_var - true_psi_var) / true_psi_var))
    fe_ho_psi_alpha_diff = np.mean(abs((fe_ho_psi_alpha_cov - true_psi_alpha_cov) / true_psi_alpha_cov))
    fe_he_psi_diff = np.mean(abs((fe_he_psi_var - true_psi_var) / true_psi_var))
    fe_he_psi_alpha_diff = np.mean(abs((fe_he_psi_alpha_cov - true_psi_alpha_cov) / true_psi_alpha_cov))

    assert cre_psi_diff < 0.05
    assert cre_psi_alpha_diff < 0.05
    assert fe_psi_diff < 0.05
    assert fe_psi_alpha_diff < 0.05
    assert fe_ho_psi_diff < 0.05
    assert fe_ho_psi_alpha_diff < 0.05
    assert fe_he_psi_diff < 0.05
    assert fe_he_psi_alpha_diff < 0.05

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
#         bdf = bpd.BipartiteLong(bpd.SimBipartite({'seed': 1234}).sim_network(), include_id_reference_dict=True).clean_data().get_es()

#         orig_n_movers = len(bdf.loc[bdf['m'] == 1, 'i'].unique())
#         n_movers = []

#         for j in attrition_fn[i](bdf, subsets_lst[i], rng=np.random.default_rng(1234)):
#             n_movers.append(len(j.loc[j['m'] == 1, 'i'].unique()))

#         n_movers_vs_subsets = abs((np.array(n_movers) / orig_n_movers) - subsets_lst[i])

#         assert np.max(n_movers_vs_subsets) < 2e-4

#         # Collapsed
#         bdf = bpd.BipartiteLong(bpd.SimBipartite({'seed': 1234}).sim_network(), include_id_reference_dict=True).clean_data().get_collapsed_long().get_es()

#         orig_n_movers = len(bdf.loc[bdf['m'] == 1, 'i'].unique())
#         n_movers = []

#         for j in attrition_fn[i](bdf, subsets_lst[i], rng=np.random.default_rng(1234)):
#             n_movers.append(len(j.loc[j['m'] == 1, 'i'].unique()))

#         n_movers_vs_subsets = abs((np.array(n_movers) / orig_n_movers) - subsets_lst[i])

#         assert np.max(n_movers_vs_subsets) < 2e-4

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
    blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True}, seed=1234)
    # Make variance of worker types small
    blm_true.S1 /= 4
    blm_true.S2 /= 4
    jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
    sdata = blm_true._m2_mixt_simulate_stayers(blm_true.NNs * smult)
    blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 30}, seed=5678)
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
    blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True}, seed=1234)
    # Make variance of worker types small
    blm_true.S1 /= 4
    blm_true.S2 /= 4
    jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
    sdata = blm_true._m2_mixt_simulate_stayers(blm_true.NNs * smult)
    blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 30}, seed=5678)
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
    blm = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True, 'return_qi': True}, seed=1234)
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

def test_blm_A_3():
    # Test whether BLM estimates A properly, given true S and pk1.
    nl = 6
    nk = 10
    mmult = 100
    min_A1 = np.inf
    min_A2 = np.inf
    lik = - np.inf
    for i in range(4):
        # Initiate BLMModel object
        blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True}, seed=1234 + i)
        # Make variance of worker types small
        blm_true.S1 /= 4
        blm_true.S2 /= 4
        jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
        blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 1, 'update_s': False, 'update_pk1': False}, seed=5678 + i)
        ## Start at truth for A1 and A2
        blm_fit.A1 = blm_true.A1.copy()
        blm_fit.A2 = blm_true.A2.copy()
        ##
        blm_fit.S1 = blm_true.S1
        blm_fit.S2 = blm_true.S2
        blm_fit.pk1 = blm_true.pk1
        blm_fit.fit_movers(jdata)
        # blm_fit._sort_matrices()

        # Compute average percent difference from truth
        val_1 = abs(np.mean(
            (blm_true.A1.flatten() - blm_fit.A1.flatten()) / blm_true.A1.flatten()
        ))
        val_2 = abs(np.mean(
            (blm_true.A2.flatten() - blm_fit.A2.flatten()) / blm_true.A2.flatten()
        ))
        if blm_fit.lik1 > lik:
            lik = blm_fit.lik1
            min_A1 = val_1
            min_A2 = val_2

    assert 0 < min_A1 < 0.2
    assert 0 < min_A2 < 0.15

def test_blm_S_4():
    # Test whether BLM estimates S properly, given true A and pk1.
    nl = 6
    nk = 10
    mmult = 100
    min_S1 = np.inf
    min_S2 = np.inf
    lik = - np.inf
    for i in range(3):
        # Initiate BLMModel object
        blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True}, seed=1234 + i)
        # Make variance of worker types small
        # blm_true.S1 /= 4
        # blm_true.S2 /= 4
        jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
        blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 1, 'update_a': False, 'update_pk1': False}, seed=5678 + i)
        blm_fit.A1 = blm_true.A1
        blm_fit.A2 = blm_true.A2
        ## Start at truth for S1 and S2
        blm_fit.S1 = blm_true.S1.copy()
        blm_fit.S2 = blm_true.S2.copy()
        ##
        blm_fit.pk1 = blm_true.pk1
        blm_fit.fit_movers(jdata)
        # blm_fit._sort_matrices()

        # Compute average percent difference from truth
        val_1 = abs(np.mean(
            (blm_true.S1.flatten() - blm_fit.S1.flatten()) / blm_true.S1.flatten()
        ))
        val_2 = abs(np.mean(
            (blm_true.S2.flatten() - blm_fit.S2.flatten()) / blm_true.S2.flatten()
        ))
        if blm_fit.lik1 > lik:
            lik = blm_fit.lik1
            min_S1 = val_1
            min_S2 = val_2

    assert 0 < min_S1 < 0.015
    assert 0 < min_S2 < 0.01

def test_blm_pk_5():
    # Test whether BLM estimates pk1 and pk0 properly, given true A and S.
    nl = 6
    nk = 10
    mmult = 100
    smult = 100
    min_pk1 = np.inf
    min_pk0 = np.inf
    lik1 = - np.inf
    lik0 = - np.inf
    for i in range(3):
        # Initiate BLMModel object
        blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True}, seed=1234 + i)
        # Make variance of worker types small
        blm_true.S1 /= 10
        blm_true.S2 /= 10
        jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
        sdata = blm_true._m2_mixt_simulate_stayers(blm_true.NNs * smult)
        blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 1, 'update_a': False, 'update_s': False}, seed=5678 + i)
        blm_fit.A1 = blm_true.A1
        blm_fit.A2 = blm_true.A2
        blm_fit.S1 = blm_true.S1
        blm_fit.S2 = blm_true.S2
        ## Start at truth for pk1
        blm_fit.pk1 = blm_true.pk1.copy()
        ##
        blm_fit.fit_movers(jdata)
        # blm_fit._sort_matrices()
        blm_fit.fit_stayers(sdata)
        # blm_fit._sort_matrices()

        # Compute average percent difference from truth
        val_1 = abs(np.mean(
            (blm_true.pk1.flatten() - blm_fit.pk1.flatten()) / blm_true.pk1.flatten()
        ))
        val_0 = abs(np.mean(
            (blm_true.pk0.flatten() - blm_fit.pk0.flatten()) / blm_true.pk0.flatten()
        ))
        if blm_fit.lik1 > lik1:
            lik1 = blm_fit.lik1
            min_pk1 = val_1
        if blm_fit.lik0 > lik0:
            lik0 = blm_fit.lik0
            min_pk0 = val_0

    assert 0 < min_pk1 < 0.05
    assert 0 < min_pk0 < 0.15 # 0.7 # This error has gone up to 4.096 @FIXME FIX THIS

def test_blm_fit_6_1():
    # Test whether BLM fit_movers() method works properly.
    nl = 6
    nk = 10
    mmult = 100
    smult = 100
    min_A1 = np.inf
    min_A2 = np.inf
    min_S1 = np.inf
    min_S2 = np.inf
    min_pk1 = np.inf
    min_pk0 = np.inf
    lik1 = - np.inf
    lik0 = - np.inf
    for i in range(4):
        # Initiate BLMModel object
        blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True}, seed=1234 + i)
        # Make variance of worker types small
        blm_true.S1 /= 4
        blm_true.S2 /= 4
        jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
        sdata = blm_true._m2_mixt_simulate_stayers(blm_true.NNs * smult)
        blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 1}, seed=5678 + i)
        ## Start at truth for all parameters
        blm_fit.A1 = blm_true.A1.copy()
        blm_fit.A2 = blm_true.A2.copy()
        blm_fit.S1 = blm_true.S1.copy()
        blm_fit.S2 = blm_true.S2.copy()
        blm_fit.pk1 = blm_true.pk1.copy()
        ##
        blm_fit.fit_movers(jdata)
        # blm_fit._sort_matrices()
        blm_fit.fit_stayers(sdata)
        # blm_fit._sort_matrices()

        # Compute average percent difference from truth
        val_A1 = abs(np.mean(
            (blm_true.A1.flatten() - blm_fit.A1.flatten()) / blm_true.A1.flatten()
        ))
        val_A2 = abs(np.mean(
            (blm_true.A2.flatten() - blm_fit.A2.flatten()) / blm_true.A2.flatten()
        ))
        val_S1 = abs(np.mean(
            (blm_true.S1.flatten() - blm_fit.S1.flatten()) / blm_true.S1.flatten()
        ))
        val_S2 = abs(np.mean(
            (blm_true.S2.flatten() - blm_fit.S2.flatten()) / blm_true.S2.flatten()
        ))
        val_pk1 = abs(np.mean(
            (blm_true.pk1.flatten() - blm_fit.pk1.flatten()) / blm_true.pk1.flatten()
        ))
        val_pk0 = abs(np.mean(
            (blm_true.pk0.flatten() - blm_fit.pk0.flatten()) / blm_true.pk0.flatten()
        ))
        if blm_fit.lik1 > lik1:
            lik1 = blm_fit.lik1
            min_A1 = val_A1
            min_A2 = val_A2
            min_S1 = val_S1
            min_S2 = val_S2
            min_pk1 = val_pk1
        if blm_fit.lik0 > lik0:
            lik0 = blm_fit.lik0
            min_pk0 = val_pk0

    # Compute average percent difference from truth
    assert 0 < min_A1 < 0.04
    assert 0 < min_A2 < 0.15
    assert 0 < min_S1 < 0.4
    assert 0 < min_S2 < 0.35
    # assert 0 < min_pk1 < 5 # This error has gone up to 4.791 @FIXME FIX THIS
    # assert 0 < min_pk0 < 6 # This error has gone up to 5.093 @FIXME FIX THIS

def test_blm_fit_6_2():
    # Test whether BLM fit_movers_cstr_uncstr() method works properly.
    nl = 6
    nk = 10
    mmult = 100
    smult = 100
    min_A1 = np.inf
    min_A2 = np.inf
    min_S1 = np.inf
    min_S2 = np.inf
    min_pk1 = np.inf
    min_pk0 = np.inf
    lik1 = - np.inf
    lik0 = - np.inf
    for i in range(4):
        # Initiate BLMModel object
        blm_true = tw.BLMModel({'nl': nl, 'nk': nk, 'simulation': True}, seed=1234 + i)
        # Make variance of worker types small
        blm_true.S1 /= 4
        blm_true.S2 /= 4
        jdata = blm_true._m2_mixt_simulate_movers(blm_true.NNm * mmult)
        sdata = blm_true._m2_mixt_simulate_stayers(blm_true.NNs * smult)
        blm_fit = tw.BLMModel({'nl': nl, 'nk': nk, 'maxiters': 1}, seed=5678 + i)
        ## Start at truth for all parameters
        blm_fit.A1 = blm_true.A1.copy()
        blm_fit.A2 = blm_true.A2.copy()
        blm_fit.S1 = blm_true.S1.copy()
        blm_fit.S2 = blm_true.S2.copy()
        blm_fit.pk1 = blm_true.pk1.copy()
        ##
        blm_fit.fit_movers_cstr_uncstr(jdata)
        # blm_fit._sort_matrices()
        blm_fit.fit_stayers(sdata)
        # blm_fit._sort_matrices()

        # Compute average percent difference from truth
        val_A1 = abs(np.mean(
            (blm_true.A1.flatten() - blm_fit.A1.flatten()) / blm_true.A1.flatten()
        ))
        val_A2 = abs(np.mean(
            (blm_true.A2.flatten() - blm_fit.A2.flatten()) / blm_true.A2.flatten()
        ))
        val_S1 = abs(np.mean(
            (blm_true.S1.flatten() - blm_fit.S1.flatten()) / blm_true.S1.flatten()
        ))
        val_S2 = abs(np.mean(
            (blm_true.S2.flatten() - blm_fit.S2.flatten()) / blm_true.S2.flatten()
        ))
        val_pk1 = abs(np.mean(
            (blm_true.pk1.flatten() - blm_fit.pk1.flatten()) / blm_true.pk1.flatten()
        ))
        val_pk0 = abs(np.mean(
            (blm_true.pk0.flatten() - blm_fit.pk0.flatten()) / blm_true.pk0.flatten()
        ))
        if blm_fit.lik1 > lik1:
            lik1 = blm_fit.lik1
            min_A1 = val_A1
            min_A2 = val_A2
            min_S1 = val_S1
            min_S2 = val_S2
            min_pk1 = val_pk1
        if blm_fit.lik0 > lik0:
            lik0 = blm_fit.lik0
            min_pk0 = val_pk0

    # Compute average percent difference from truth
    assert min_A1 < 0.05 # 0.15
    assert min_A2 < 0.2 # 0.05
    assert min_S1 < 0.07 # 0.05
    assert min_S2 < 0.05 # 0.15
    # assert min_pk1 < 5 # This error has gone up to 4.216 @FIXME FIX THIS
    # assert min_pk0 < 5 # This error has gone up to 4.353 @FIXME FIX THIS
